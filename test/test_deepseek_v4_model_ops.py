import unittest

import torch
from torch.nn import functional as F

from llaisys.models.deepseek_v4_backends import CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_torch_model
from llaisys.models.deepseek_v4_reference import apply_rotary, rms_norm


class ModelStructureBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(912)
        contracts = {name: contract for name, contract in MODEL_CONTRACTS.items() if name not in CONTRACTS}
        registry = OperatorRegistry(contracts)
        register_torch_model(registry)
        self.ops = registry.bind({name: "torch" for name in contracts})
        self.devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def test_norm_and_strided_rotary_preserve_math_and_storage(self):
        for device in self.devices:
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(device=device, dtype=dtype):
                    x = torch.randn(2, 7, 3, 128, device=device).to(dtype)
                    weight = torch.randn(128, device=device)
                    torch.testing.assert_close(self.ops.function("rms_norm")(x, weight, 1e-6),
                                               rms_norm(x, weight, 1e-6), atol=0, rtol=0)
                    phase = torch.randn(7, 32, device=device)
                    freqs = torch.polar(torch.ones_like(phase), phase)
                    for inverse in (False, True):
                        original = x.clone()
                        rope_view = x[..., -64:]
                        expected = apply_rotary(rope_view, freqs, inverse=inverse)
                        returned = self.ops.function("rotary_inplace")(rope_view, freqs, inverse)
                        self.assertEqual(returned.data_ptr(), rope_view.data_ptr())
                        torch.testing.assert_close(x[..., -64:], expected, atol=0, rtol=0)
                        torch.testing.assert_close(x[..., :-64], original[..., :-64], atol=0, rtol=0)

    def test_indexer_scores_preserve_bf16_boundaries_and_empty_candidates(self):
        function = self.ops.function("indexer_scores")
        for device in self.devices:
            query = torch.randn(2, 7, 4, 128, device=device).bfloat16()
            weights = torch.randn(2, 7, 4, device=device).bfloat16()
            for candidates in (0, 3, 129):
                cache = torch.randn(2, candidates, 128, device=device).bfloat16()
                expected = torch.einsum("bshd,btd->bsht", query, cache)
                expected = (expected.relu_() * weights.unsqueeze(-1)).sum(2)
                actual = function(query, cache, weights)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual.dtype, torch.bfloat16)

    def test_router_bias_only_selects_hash_and_score_normalization(self):
        function = self.ops.function("router")
        for device in self.devices:
            logits = torch.randn(17, 7, device=device)
            bias = torch.tensor([9., 0, 0, 0, 0, 0, 8.], device=device)
            hash_ids = torch.tensor([[5, 1, 3]], dtype=torch.int32, device=device).expand(17, -1)
            for mode in ("softmax", "sigmoid", "sqrtsoftplus"):
                scores = (logits.softmax(-1) if mode == "softmax" else logits.sigmoid()
                          if mode == "sigmoid" else F.softplus(logits).sqrt())
                for hashed in (False, True):
                    with self.subTest(device=device, mode=mode, hashed=hashed):
                        expected_ids = hash_ids if hashed else (scores + bias).topk(3, dim=-1)[1]
                        expected_weights = scores.gather(1, expected_ids.long())
                        if mode != "softmax":
                            expected_weights /= expected_weights.sum(-1, keepdim=True)
                        actual_weights, actual_ids = function(logits, None if hashed else bias,
                                                              hash_ids if hashed else None, 3, mode, 1.5)
                        torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
                        torch.testing.assert_close(actual_weights, expected_weights * 1.5, atol=0, rtol=0)

    def test_compressor_pool_matches_fp32_expression_and_fp64_oracle(self):
        function = self.ops.function("compressor_pool")
        for device in self.devices:
            for ratio in (8, 128):
                values = torch.randn(2, 3, ratio, 64, device=device)
                scores = torch.randn_like(values)
                scores[:, 0, :ratio//2] = -torch.inf
                for decode in (False, True):
                    current_values = values[:, 0] if decode else values
                    current_scores = scores[:, 0] if decode else scores
                    expected = (current_values * current_scores.softmax(-2)).sum(-2, keepdim=decode)
                    oracle = (current_values.double() * current_scores.double().softmax(-2)).sum(-2, keepdim=decode)
                    actual = function(current_values, current_scores)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(actual.double(), oracle, atol=1e-6, rtol=1e-5)

    def test_expert_activation_applies_route_scale_before_cast(self):
        function = self.ops.function("expert_activation")
        for device in self.devices:
            gate = (torch.randn(17, 128, device=device) * 20).bfloat16()
            up = (torch.randn_like(gate.float()) * 20).bfloat16()
            weights = torch.rand(17, 1, device=device)
            for limit in (0., 10.):
                g = gate.float().clamp(max=limit) if limit > 0 else gate.float()
                u = up.float().clamp(-limit, limit) if limit > 0 else up.float()
                for routed in (False, True):
                    expected = F.silu(g) * u
                    if routed:
                        expected = weights * expected
                    actual = function(gate, up, limit, weights if routed else None)
                    torch.testing.assert_close(actual, expected.bfloat16(), atol=0, rtol=0)
                    if routed:
                        premature_cast = (weights * (F.silu(g) * u).bfloat16()).bfloat16()
                        self.assertFalse(torch.equal(actual, premature_cast))

    def test_dispatch_and_combine_match_original_expert_order(self):
        for device in self.devices:
            for tokens in (0, 1, 37):
                with self.subTest(device=device, tokens=tokens):
                    # Experts 0 and 6 stay unused; one expert is selected by
                    # every token. This exercises empty and imbalanced ranges.
                    ids = torch.tensor([[5, 1, 3]], device=device).expand(tokens, -1)
                    x = torch.randn(tokens, 128, device=device).bfloat16()
                    weights = torch.rand(tokens, 3, device=device)
                    x_before, weights_before = x.clone(), weights.clone()
                    packed = self.ops.function("moe_dispatch")(x, weights, ids, 7)
                    self.assertEqual(packed.hidden.shape, (tokens * 3, 128))
                    self.assertEqual(packed.expert_offsets.device, x.device)
                    self.assertEqual(packed.expert_offsets.dtype, torch.int64)
                    offsets = packed.expert_offsets.tolist()
                    self.assertEqual(offsets, [0, 0, tokens, tokens, tokens*2, tokens*2, tokens*3, tokens*3])
                    outputs = torch.empty_like(packed.hidden)
                    expected = torch.zeros_like(x, dtype=torch.float32)
                    for expert, (start, end) in enumerate(zip(offsets, offsets[1:])):
                        rows, choices = torch.where(ids == expert)
                        torch.testing.assert_close(packed.token_indices[start:end], rows, atol=0, rtol=0)
                        torch.testing.assert_close(packed.hidden[start:end], x[rows], atol=0, rtol=0)
                        torch.testing.assert_close(packed.route_weights[start:end], weights[rows, choices, None], atol=0, rtol=0)
                        contribution = ((x[rows].float() + expert) * weights[rows, choices, None]).bfloat16()
                        outputs[start:end] = contribution
                        expected[rows] += contribution
                    actual = self.ops.function("moe_combine")(outputs, packed)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(x, x_before, atol=0, rtol=0)
                    torch.testing.assert_close(weights, weights_before, atol=0, rtol=0)
                    self.assertEqual(actual.dtype, torch.float32)
                    if tokens:
                        packed.hidden.zero_()
                        torch.testing.assert_close(x, x_before, atol=0, rtol=0)

    def test_invalid_structural_contracts_fail_and_are_counted(self):
        cases = {
            "rms_norm": (torch.zeros(2, 4), torch.ones(4, dtype=torch.bfloat16), 1e-6),
            "rotary_inplace": (torch.zeros(1, 2, 3), torch.ones(2, 1, dtype=torch.complex64)),
            "indexer_scores": (torch.zeros(1, 2, 3, 4), torch.zeros(1, 5, 4), torch.zeros(1, 2, 3)),
            "router": (torch.zeros(2, 4), torch.zeros(4), None, 5, "sqrtsoftplus", 1.5),
            "compressor_pool": (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4)),
            "expert_activation": (torch.zeros(2, 4), torch.zeros(2, 4), 10),
            "moe_dispatch": (torch.zeros(2, 4), torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.int64), 4),
            "moe_combine": (torch.zeros(2, 4), None),
        }
        for name, arguments in cases.items():
            with self.subTest(operator=name), self.assertRaises(ValueError):
                self.ops.function(name)(*arguments)
            self.assertEqual(self.ops.report()[name]["failures"], 1)
            self.assertFalse(self.ops.report()[name]["fallback"])


if __name__ == "__main__":
    unittest.main()

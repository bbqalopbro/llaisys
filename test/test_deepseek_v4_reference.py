import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import types

import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).parents[1]
    / "python"
    / "llaisys"
    / "models"
    / "deepseek_v4_reference.py"
)
SPEC = importlib.util.spec_from_file_location("llaisys_deepseek_v4_reference", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MANIFEST_PATH = (
    Path(__file__).parents[1] / "python" / "llaisys" / "models" / "deepseek_v4.py"
)
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "llaisys_deepseek_v4_reference_manifest", MANIFEST_PATH
)
MANIFEST_MODULE = importlib.util.module_from_spec(MANIFEST_SPEC)
sys.modules[MANIFEST_SPEC.name] = MANIFEST_MODULE
assert MANIFEST_SPEC.loader is not None
MANIFEST_SPEC.loader.exec_module(MANIFEST_MODULE)


class DeepSeekV4AttentionReferenceTests(unittest.TestCase):
    def test_fp8_activation_quant_is_finite_and_bounded(self):
        x = torch.linspace(-500.0, 500.0, 256).reshape(2, 128)
        actual = MODULE.simulate_fp8_activation_quant(x)
        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLessEqual(float(actual.abs().max()), 512.0)

    def test_fp4_activation_quant_uses_e2m1_grid(self):
        x = torch.tensor([
            0.0, 0.4, 0.9, 1.4, 2.1, 2.9, 4.1, 5.9,
            -0.4, -0.9, -1.4, -2.1, -2.9, -4.1, -5.9, 0.0,
        ]).repeat(2).reshape(1, 32)
        actual = MODULE.simulate_fp4_activation_quant(x)
        table = torch.tensor(MODULE.FP4_E2M1_TABLE)
        for value in actual.flatten():
            self.assertTrue(torch.isclose(value, table).any())

    def test_activation_quant_rejects_unaligned_dimension(self):
        with self.assertRaises(ValueError):
            MODULE.simulate_fp8_activation_quant(torch.ones(1, 127))
        with self.assertRaises(ValueError):
            MODULE.simulate_fp4_activation_quant(torch.ones(1, 31))

    def test_fp4_unpack_uses_published_e2m1_table(self):
        packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8).view(torch.int8)
        scale = torch.tensor([[2.0]])
        actual = MODULE.dequantize_fp4_e2m1(packed, scale, group_size=4)
        torch.testing.assert_close(
            actual, torch.tensor([[1.0, 2.0, 0.0, -12.0]])
        )

    def test_rope_inverse_round_trip(self):
        torch.manual_seed(1)
        x = torch.randn(2, 5, 3, 8)
        freqs = MODULE.precompute_freqs_cis(8, 5)
        actual = MODULE.apply_rotary(MODULE.apply_rotary(x, freqs), freqs, inverse=True)
        torch.testing.assert_close(actual, x, atol=2e-6, rtol=2e-6)

    def test_sparse_attention_masks_and_sink(self):
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        kv = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]]])
        indices = torch.tensor([[[0, -1, 1]]])
        sink = torch.tensor([0.25, -0.5])
        actual = MODULE.sparse_latent_attention(q, kv, sink, indices, scale=1.0)

        expected = []
        for head in range(2):
            logits = torch.tensor([
                torch.dot(q[0, 0, head], kv[0, 0]),
                torch.dot(q[0, 0, head], kv[0, 1]),
                sink[head],
            ])
            probability = logits.softmax(0)
            expected.append(probability[0] * kv[0, 0] + probability[1] * kv[0, 1])
        torch.testing.assert_close(actual[0, 0], torch.stack(expected))

    def test_ratio4_overlap_compression(self):
        # Two groups make the cross-group first-half behavior observable.
        ratio, dimension = 4, 4
        kv = torch.arange(1, 1 + 8 * 2 * dimension, dtype=torch.float32).reshape(1, 8, 2 * dimension)
        score = torch.zeros_like(kv)
        ape = torch.zeros(ratio, 2 * dimension)
        norm = torch.ones(dimension)
        actual = MODULE.compress_projected_prefill(kv, score, ape, norm, ratio)

        raw = torch.stack([
            kv[0, :ratio, dimension:].mean(0),
            torch.cat(
                [kv[0, :ratio, :dimension], kv[0, ratio:, dimension:]], dim=0
            ).mean(0),
        ])
        expected = MODULE.rms_norm(raw.unsqueeze(0), norm)
        torch.testing.assert_close(actual, expected)

    def test_published_index_shapes(self):
        window = MODULE.window_indices(4, 2, 6, 0)
        compressed = MODULE.compressed_indices(4, 2, 8, 0, 8)
        self.assertEqual(window.shape, (2, 6, 4))
        self.assertEqual(compressed.shape, (2, 8, 2))
        self.assertTrue(torch.equal(window[0, 0], torch.tensor([0, -1, -1, -1])))
        self.assertTrue(torch.equal(compressed[0, 3], torch.tensor([8, -1])))


class DeepSeekV4MoEReferenceTests(unittest.TestCase):
    def test_selection_bias_does_not_change_routing_weight(self):
        x = torch.tensor([[1.0, 0.0]])
        gate = torch.tensor([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        bias = torch.tensor([-10.0, 0.0, 10.0])
        weights, indices = MODULE.route_experts(
            x, gate, 1, score_function="sqrtsoftplus", selection_bias=bias
        )
        self.assertEqual(indices.item(), 2)
        self.assertEqual(weights.item(), 1.0)

    def test_hash_router_uses_token_table(self):
        x = torch.randn(2, 3)
        gate = torch.randn(4, 3)
        table = torch.tensor([[1, 3], [2, 0], [3, 1]], dtype=torch.int64)
        weights, indices = MODULE.route_experts(
            x, gate, 2, input_ids=torch.tensor([2, 0]), token_to_expert=table
        )
        torch.testing.assert_close(indices, torch.tensor([[3, 1], [1, 3]]))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))

    def test_routed_and_shared_expert_combine(self):
        torch.manual_seed(7)
        tokens, dim, inter, experts, topk = 3, 4, 6, 3, 2
        x = torch.randn(tokens, dim)
        w1 = torch.randn(experts, inter, dim)
        w2 = torch.randn(experts, dim, inter)
        w3 = torch.randn(experts, inter, dim)
        shared = (torch.randn(inter, dim), torch.randn(dim, inter), torch.randn(inter, dim))
        indices = torch.tensor([[0, 2], [1, 0], [2, 1]])
        weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.1, 0.9]])
        actual = MODULE.moe_reference(x, weights, indices, w1, w2, w3, shared)

        expected = MODULE.swiglu_expert(x, *shared)
        for token in range(tokens):
            for slot in range(topk):
                expert = indices[token, slot]
                value = MODULE.swiglu_expert(
                    x[token : token + 1], w1[expert], w2[expert], w3[expert]
                )[0]
                expected[token] += weights[token, slot] * value
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


class DeepSeekV4CacheContractTests(unittest.TestCase):
    def test_v4_payload_is_not_classic_split_mla(self):
        spec = MODULE.DeepSeekV4CacheSpec(16, 512, 128, 2, (0, 4, 128))
        self.assertEqual(spec.layer_bytes("window_latent", 0), 16 * 512 * 2)
        self.assertEqual(spec.layer_bytes("compressed_latent", 1), 4 * 512 * 2)
        self.assertEqual(spec.layer_bytes("compressed_latent", 2), 1 * 512 * 2)
        self.assertEqual(spec.layer_bytes("index_latent", 1), 4 * 128 * 2)
        self.assertEqual(spec.layer_bytes("index_latent", 2), 0)


class PublishedReferenceParityTests(unittest.TestCase):
    MODEL_PATH = Path(
        "/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py"
    )

    @classmethod
    def setUpClass(cls):
        if not cls.MODEL_PATH.is_file():
            raise unittest.SkipTest("published DeepSeek reference source unavailable")
        kernel = types.ModuleType("kernel")
        kernel.act_quant = lambda x, *args, **kwargs: x
        kernel.fp4_act_quant = lambda x, *args, **kwargs: x
        kernel.fp8_gemm = lambda *args, **kwargs: None
        kernel.fp4_gemm = lambda *args, **kwargs: None
        kernel.sparse_attn = lambda *args, **kwargs: None
        kernel.hc_split_sinkhorn = lambda *args, **kwargs: None
        spec = importlib.util.spec_from_file_location("published_deepseek_v4_model", cls.MODEL_PATH)
        cls.published = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        with mock.patch.dict(sys.modules, {"kernel": kernel}):
            spec.loader.exec_module(cls.published)

    def test_rope_and_index_generation_match_published_source(self):
        torch.manual_seed(11)
        x = torch.randn(2, 5, 3, 8)
        freqs = MODULE.precompute_freqs_cis(8, 5)
        expected = self.published.apply_rotary_emb(x.clone(), freqs)
        torch.testing.assert_close(MODULE.apply_rotary(x, freqs), expected)
        torch.testing.assert_close(
            MODULE.window_indices(4, 2, 6, 0),
            self.published.get_window_topk_idxs(4, 2, 6, 0).long(),
        )
        torch.testing.assert_close(
            MODULE.compressed_indices(4, 2, 8, 0, 8),
            self.published.get_compress_topk_idxs(4, 2, 8, 0, 8).long(),
        )

    def test_ratio4_compressor_matches_published_source(self):
        torch.manual_seed(12)
        args = self.published.ModelArgs(
            max_batch_size=1, max_seq_len=16, dim=4, head_dim=4,
            rope_head_dim=2, norm_eps=1e-6,
        )
        compressor = self.published.Compressor(args, compress_ratio=4, head_dim=4)
        for parameter in compressor.parameters():
            torch.nn.init.uniform_(parameter, -0.2, 0.2)
        compressor.norm.weight.data.fill_(1.0)
        compressor.kv_cache = torch.zeros(1, 4, 4)
        compressor.freqs_cis = torch.ones(16, 1, dtype=torch.complex64)
        x = torch.randn(1, 8, 4)
        expected = compressor(x, 0)
        projected_kv = compressor.wkv(x.float())
        projected_score = compressor.wgate(x.float())
        actual = MODULE.compress_projected_prefill(
            projected_kv, projected_score, compressor.ape,
            compressor.norm.weight, 4, eps=args.norm_eps,
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_router_matches_published_source(self):
        torch.manual_seed(13)
        args = self.published.ModelArgs(
            dim=4, n_layers=2, n_hash_layers=0, n_routed_experts=5,
            n_activated_experts=2, score_func="sqrtsoftplus", route_scale=1.7,
        )
        gate = self.published.Gate(1, args)
        torch.nn.init.uniform_(gate.weight, -0.25, 0.25)
        if gate.bias is not None:
            torch.nn.init.uniform_(gate.bias, -0.05, 0.05)
        x = torch.randn(6, 4)
        expected_weights, expected_indices = gate(x)
        weights, indices = MODULE.route_experts(
            x, gate.weight, 2, score_function="sqrtsoftplus",
            route_scale=1.7, selection_bias=gate.bias,
        )
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, expected_weights)

    def test_complete_tiny_layer_matches_published_source(self):
        torch.manual_seed(14)
        args = self.published.ModelArgs(
            max_batch_size=1, max_seq_len=8, dtype="bf16", expert_dtype=None,
            vocab_size=32, dim=8, moe_inter_dim=6, n_layers=1,
            n_hash_layers=0, n_heads=2, n_routed_experts=4,
            n_shared_experts=1, n_activated_experts=2,
            q_lora_rank=4, head_dim=4, rope_head_dim=2,
            o_groups=1, o_lora_rank=4, window_size=4,
            compress_ratios=(0,), hc_mult=2, hc_sinkhorn_iters=4,
        )
        block = self.published.Block(0, args)
        for parameter in block.parameters():
            if parameter.is_floating_point():
                torch.nn.init.uniform_(parameter, -0.15, 0.15)
        for name, parameter in block.named_parameters():
            if name.endswith("norm.weight"):
                parameter.data.fill_(1.0)

        self.published.sparse_attn = MODULE.sparse_latent_attention
        self.published.hc_split_sinkhorn = (
            lambda mixes, scale, base, hc, iterations, eps:
            MODULE.hyperconnection_split(
                mixes, scale, base, hc, iterations=iterations, eps=eps
            )
        )
        x = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
        input_ids = torch.tensor([[1, 2, 3]])
        expected = block(x.clone(), 0, input_ids)

        residual = x
        value, post, combination = MODULE.hyperconnection_pre(
            x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base,
            norm_eps=args.norm_eps, sinkhorn_iterations=args.hc_sinkhorn_iters,
            hc_eps=args.hc_eps,
        )
        value = MODULE.rms_norm(value, block.attn_norm.weight, args.norm_eps)
        attn = block.attn
        query_rank = MODULE.rms_norm(
            F.linear(value, attn.wq_a.weight), attn.q_norm.weight, args.norm_eps
        )
        query = F.linear(query_rank, attn.wq_b.weight).unflatten(-1, (2, 4))
        query *= torch.rsqrt(query.square().mean(-1, keepdim=True) + args.norm_eps)
        freqs = attn.freqs_cis[: value.shape[1]]
        query = torch.cat(
            [query[..., :-2], MODULE.apply_rotary(query[..., -2:], freqs)], dim=-1
        )
        latent = MODULE.rms_norm(
            F.linear(value, attn.wkv.weight), attn.kv_norm.weight, args.norm_eps
        )
        latent = torch.cat(
            [latent[..., :-2], MODULE.apply_rotary(latent[..., -2:], freqs)], dim=-1
        )
        index = MODULE.window_indices(4, 1, value.shape[1], 0)
        attention_output = MODULE.sparse_latent_attention(
            query, latent, attn.attn_sink, index, attn.softmax_scale
        )
        attention_output = torch.cat(
            [attention_output[..., :-2], MODULE.apply_rotary(
                attention_output[..., -2:], freqs, inverse=True
            )], dim=-1,
        )
        attention_output = attention_output.reshape(1, 3, 1, -1)
        wo_a = attn.wo_a.weight.reshape(1, args.o_lora_rank, -1)
        attention_output = torch.einsum(
            "bsgd,grd->bsgr", attention_output, wo_a
        ).flatten(2)
        attention_output = F.linear(attention_output, attn.wo_b.weight)
        value = MODULE.hyperconnection_post(
            attention_output, residual, post, combination
        )

        residual = value
        value, post, combination = MODULE.hyperconnection_pre(
            value, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base,
            norm_eps=args.norm_eps, sinkhorn_iterations=args.hc_sinkhorn_iters,
            hc_eps=args.hc_eps,
        )
        value = MODULE.rms_norm(value, block.ffn_norm.weight, args.norm_eps)
        flat = value.reshape(-1, args.dim)
        route_weight, route_index = MODULE.route_experts(
            flat, block.ffn.gate.weight, args.n_activated_experts,
            score_function=args.score_func, route_scale=args.route_scale,
            selection_bias=block.ffn.gate.bias,
        )
        experts = [expert for expert in block.ffn.experts]
        w1 = torch.stack([expert.w1.weight for expert in experts])
        w2 = torch.stack([expert.w2.weight for expert in experts])
        w3 = torch.stack([expert.w3.weight for expert in experts])
        shared = block.ffn.shared_experts
        ffn_output = MODULE.moe_reference(
            value, route_weight, route_index, w1, w2, w3,
            (shared.w1.weight, shared.w2.weight, shared.w3.weight),
            swiglu_limit=args.swiglu_limit,
        )
        actual = MODULE.hyperconnection_post(
            ffn_output, residual, post, combination
        )
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


class PublishedQuantizedWeightSliceTests(unittest.TestCase):
    MODEL_DIR = Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731")

    @unittest.skipUnless(MODEL_DIR.is_dir(), "published checkpoint unavailable")
    def test_real_fp8_and_fp4_crops_form_finite_linears(self):
        manifest = MANIFEST_MODULE.DeepSeekV4WeightManifest.from_directory(self.MODEL_DIR)
        fp8 = manifest.load_tensor_slice(
            "layers.0.attn.wkv.weight", (slice(0, 128), slice(0, 128))
        )
        fp8_scale = manifest.load_tensor_slice(
            "layers.0.attn.wkv.scale", (slice(0, 1), slice(0, 1))
        )
        dense = MODULE.dequantize_fp8_blocks(fp8, fp8_scale)
        self.assertEqual(tuple(dense.shape), (128, 128))
        self.assertTrue(torch.isfinite(dense).all())

        fp4 = manifest.load_tensor_slice(
            "layers.0.ffn.experts.0.w1.weight", (slice(0, 4), slice(0, 64))
        )
        fp4_scale = manifest.load_tensor_slice(
            "layers.0.ffn.experts.0.w1.scale", (slice(0, 4), slice(0, 4))
        )
        expert = MODULE.dequantize_fp4_e2m1(fp4, fp4_scale)
        self.assertEqual(tuple(expert.shape), (4, 128))
        self.assertTrue(torch.isfinite(expert).all())
        x = torch.randn(2, 128)
        self.assertTrue(torch.isfinite(F.linear(x, dense)).all())
        self.assertTrue(torch.isfinite(F.linear(x, expert)).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
class DeepSeekV4CudaReferenceTests(unittest.TestCase):
    def test_attention_and_moe_match_cpu_on_allocated_gpu(self):
        torch.manual_seed(21)
        q = torch.randn(1, 3, 4, 8)
        kv = torch.randn(1, 6, 8)
        indices = torch.tensor([[[0, -1, -1], [0, 1, -1], [0, 1, 2]]])
        sink = torch.randn(4)
        cpu_attention = MODULE.sparse_latent_attention(q, kv, sink, indices)
        gpu_attention = MODULE.sparse_latent_attention(
            q.cuda(), kv.cuda(), sink.cuda(), indices.cuda()
        ).cpu()
        torch.testing.assert_close(gpu_attention, cpu_attention, atol=2e-5, rtol=2e-5)

        tokens, dim, inter, experts = 4, 8, 12, 4
        x = torch.randn(tokens, dim)
        gate = torch.randn(experts, dim)
        route_weight, route_index = MODULE.route_experts(x, gate, 2)
        w1 = torch.randn(experts, inter, dim)
        w2 = torch.randn(experts, dim, inter)
        w3 = torch.randn(experts, inter, dim)
        shared = (torch.randn(inter, dim), torch.randn(dim, inter), torch.randn(inter, dim))
        cpu_moe = MODULE.moe_reference(
            x, route_weight, route_index, w1, w2, w3, shared
        )
        gpu_moe = MODULE.moe_reference(
            x.cuda(), route_weight.cuda(), route_index.cuda(), w1.cuda(),
            w2.cuda(), w3.cuda(), tuple(value.cuda() for value in shared),
        ).cpu()
        torch.testing.assert_close(gpu_moe, cpu_moe, atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from llaisys.models.deepseek_v4_backends import CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_torch_model
from llaisys.models.deepseek_v4_model import DeepSeekV4Model, InferenceConfig, load_converted_weights
from llaisys.models.deepseek_v4_model.weights import read_header, validate_header
from llaisys.models import deepseek_v4_reference as reference


def tiny_config():
    return InferenceConfig(dim=128, vocab_size=32, n_layers=3, n_heads=2, head_dim=128,
                           q_lora_rank=128, o_groups=1, o_lora_rank=128, moe_inter_dim=128,
                           n_routed_experts=2, n_activated_experts=1, compress_ratios=(0, 4, 128),
                           n_hash_layers=1, index_n_heads=2, index_topk=4, window_size=8,
                           hc_mult=2, max_seq_len=160, weight_dtype="bf16", expert_dtype="bf16")


def hadamard(x, scale=1):
    value = x.float()
    width = 1
    while width < value.shape[-1]:
        groups = value.reshape(*value.shape[:-1], -1, 2 * width)
        left, right = groups[..., :width], groups[..., width:]
        value = torch.cat((left + right, left - right), dim=-1).flatten(-2)
        width *= 2
    return (value * scale).to(x.dtype)


def torch_ops(fixed_rows=0):
    def fp8(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if not inplace:
            raise NotImplementedError("this tiny BF16 fixture only uses inplace activation QDQ")
        x.copy_(reference.simulate_fp8_activation_quant(x, block_size, power_of_two_scale=scale_fmt is not None))
        return x

    def fp4(x, block_size=32, inplace=False):
        if not inplace:
            raise NotImplementedError("this tiny BF16 fixture only uses inplace activation QDQ")
        x.copy_(reference.simulate_fp4_activation_quant(x, block_size))
        return x

    def unused(*args, **kwargs):
        raise AssertionError("quantized GEMM must not execute in the BF16 fixture")

    functions = {"act_quant": fp8, "fp4_act_quant": fp4, "fp8_gemm": unused, "fp4_gemm": unused,
                 "sparse_attn": reference.sparse_latent_attention,
                 "hc_split_sinkhorn": lambda x, s, b, hc=4, n=20, eps=1e-6:
                 reference.hyperconnection_split(x, s, b, hc, iterations=n, eps=eps)}
    registry = OperatorRegistry(MODEL_CONTRACTS)
    for name, function in functions.items():
        registry.register("torch-test", name, function, version="1", contract=CONTRACTS[name])
    register_torch_model(registry, backend="torch-test", fixed_rows=fixed_rows)
    return registry.bind({name: "torch-test" for name in MODEL_CONTRACTS})


def initialized_model(*, fixed_rows=0, indexer_tie_policy="published", attention_metadata_policy="published", max_seq_len=None):
    cfg = replace(tiny_config(), indexer_tie_policy=indexer_tie_policy, attention_metadata_policy=attention_metadata_policy)
    if max_seq_len is not None:
        cfg = replace(cfg, max_seq_len=max_seq_len)
    model = DeepSeekV4Model(cfg, torch_ops(fixed_rows), hadamard, device="cpu")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("tid2eid"):
                param.random_(0, 2)
            elif name.endswith("norm.weight"):
                param.fill_(1)
            else:
                param.uniform_(-0.1, 0.1)
    return model


class IndependentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_config_and_meta_weights_do_not_allocate_cache(self):
        model = DeepSeekV4Model(tiny_config(), torch_ops(), hadamard)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertEqual(list(model.buffers()), [])
        with self.assertRaises(RuntimeError):
            model.new_request()
        with self.assertRaises(ValueError):
            replace(tiny_config(), compress_ratios=(4,))

    def test_request_interleaving_reset_and_release(self):
        model = initialized_model()
        first, second = model.new_request(), model.new_request()
        ids = torch.tensor([[1, 2, 3]])
        expected = model(ids, first).logits
        model(torch.tensor([[7, 9]]), second)
        first.reset()
        torch.testing.assert_close(model(ids, first).logits, expected, atol=0, rtol=0)
        self.assertEqual(second.position, 2)
        self.assertNotEqual(first.layers[0].kv_cache.data_ptr(), second.layers[0].kv_cache.data_ptr())
        first.close()
        self.assertEqual(first.layers, [])
        with self.assertRaises(RuntimeError):
            model(ids, first)

    def test_input_validation_does_not_mutate_cache(self):
        model = initialized_model()
        state = model.new_request()
        for ids in (torch.tensor([[32]]), torch.tensor([[-1]]), torch.tensor([[1.0]]),
                    torch.zeros(2, 3, dtype=torch.int64), torch.empty(1, 0, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                model(ids, state)
            self.assertEqual(state.position, 0)
            self.assertTrue(state.valid)
        model(torch.tensor([[1, 2]]), state)
        model(torch.tensor([[3, 4]]), state)
        self.assertEqual(state.position, 4)
        with self.assertRaisesRegex(ValueError, "capacity"):
            model(torch.ones(1, model.config.max_seq_len, dtype=torch.int64), state)
        self.assertEqual(state.position, 4)

    def test_non_aligned_chunks_across_window_and_compression_boundaries(self):
        torch.manual_seed(909)
        model = initialized_model()
        inputs = torch.randint(0, model.config.vocab_size, (1, 137))
        full_state = model.new_request()
        expected = model(inputs, full_state, capture_layers=True)
        for sizes in ((3, 2, 7, 115, 10), (1, 130, 6), (127, 2, 8), (8, 8, 121)):
            with self.subTest(chunks=sizes):
                state = model.new_request()
                position = 0
                for size in sizes:
                    output = model(inputs[:, position:position+size], state, capture_layers=True)
                    position += size
                self.assertEqual(state.position, 137)
                actual, baseline = output.logits.float(), expected.logits.float()
                self.assertLessEqual(((actual-baseline).norm()/baseline.norm()).item(), 0.01)
                self.assertGreaterEqual(torch.nn.functional.cosine_similarity(actual, baseline).item(), 0.999)
                self.assertTrue(torch.equal(actual.argmax(-1), baseline.argmax(-1)))
                # Decode after chunked prefill must consume retained caches.
                for _ in range(3):
                    token = baseline.argmax(-1).unsqueeze(1)
                    actual = model(token, state).logits.float()
                    baseline = model(token, full_state).logits.float()
                    self.assertLessEqual(((actual-baseline).norm()/baseline.norm()).item(), 0.01)
                full_state.reset()
                expected = model(inputs, full_state, capture_layers=True)

    def test_projected_compression_groups_survive_non_aligned_chunks(self):
        torch.manual_seed(909)
        model = initialized_model()
        values = torch.randn(1, 137, model.config.dim, dtype=torch.bfloat16)
        for layer in (1, 2):
            compressor = model.layers[layer].attn.compressor
            full = model.new_request().layers[layer]
            compressor(values, 0, full.compressor, full.kv_cache[:, model.config.window_size:], full.freqs)
            for sizes in ((127, 2, 8), (3, 2, 7, 115, 10), (65, 65, 7)):
                state = model.new_request().layers[layer]
                position = 0
                for size in sizes:
                    compressor(values[:, position:position+size], position, state.compressor,
                               state.kv_cache[:, model.config.window_size:], state.freqs)
                    position += size
                with self.subTest(ratio=compressor.ratio, chunks=sizes):
                    torch.testing.assert_close(state.kv_cache, full.kv_cache, atol=0, rtol=0)

    def test_explicit_fixed_profile_chunk_layers_caches_and_decode_are_exact(self):
        # This tests a separate, explicit numerical policy. It does not replace
        # the published-profile chunk regression or its original acceptance gate.
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            torch.manual_seed(909)
            model = initialized_model(fixed_rows=32, indexer_tie_policy="index_ascending",
                                      attention_metadata_policy="fixed").to(device)
            inputs = torch.randint(0, model.config.vocab_size, (1, 137)).to(device)
            for sizes in ((3, 2, 7, 115, 10), (1, 130, 6), (127, 2, 8), (8, 8, 121)):
                with self.subTest(device=device, chunks=sizes):
                    full, chunk = model.new_request(), model.new_request()
                    try:
                        expected = model(inputs, full, capture_layers=True)
                        position = 0
                        for size in sizes:
                            actual = model(inputs[:, position:position+size], chunk, capture_layers=True)
                            position += size
                        for step in range(4):
                            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                            for lhs, rhs in zip(actual.layer_hiddens, expected.layer_hiddens):
                                torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
                            for lhs, rhs in zip(chunk.layers, full.layers):
                                torch.testing.assert_close(lhs.kv_cache, rhs.kv_cache, atol=0, rtol=0)
                                if lhs.index_cache is not None:
                                    torch.testing.assert_close(lhs.index_cache, rhs.index_cache, atol=0, rtol=0)
                            self.assertEqual(chunk.position, full.position)
                            if step < 3:
                                token = expected.logits.argmax(-1).unsqueeze(1)
                                expected = model(token, full, capture_layers=True)
                                actual = model(token, chunk, capture_layers=True)
                    finally:
                        full.close()
                        chunk.close()

    def test_multi_token_chunk_is_a_single_layer_invocation_not_token_replay(self):
        model = initialized_model()
        state = model.new_request()
        model(torch.tensor([[1, 2, 3]]), state)
        with patch.object(model.layers[0], "forward", wraps=model.layers[0].forward) as execute:
            output = model(torch.tensor([[4, 5, 6, 7, 8]]), state)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.args[0].shape[1], 5)
            self.assertEqual(state.position, 8)
            self.assertTrue(torch.isfinite(output.logits).all())

    def test_model_ownership_and_failed_execution(self):
        model = initialized_model()
        state = model.new_request()
        other = initialized_model()
        with self.assertRaises(ValueError):
            other(torch.tensor([[1]]), state)
        with patch.object(model.layers[1], "forward", side_effect=RuntimeError("kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                model(torch.tensor([[1, 2, 3]]), state)
        self.assertFalse(state.valid)
        with self.assertRaises(RuntimeError):
            model(torch.tensor([[1]]), state)
        state.reset()
        self.assertTrue(torch.isfinite(model(torch.tensor([[1]]), state).logits).all())

    @unittest.skipUnless(Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py").is_file(),
                         "published model required")
    def test_full_model_and_layer_outputs_match_published_bf16_fixture(self):
        cfg, ops = tiny_config(), torch_ops()
        module_path = "/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py"
        spec = importlib.util.spec_from_file_location("independent_model_published_oracle", module_path)
        published = importlib.util.module_from_spec(spec)
        kernel = types.ModuleType("kernel")
        for name in CONTRACTS:
            setattr(kernel, name, ops.function(name))
        with patch.dict(sys.modules, {"kernel": kernel}):
            spec.loader.exec_module(published)
        published.rotate_activation = lambda x: hadamard(x, x.shape[-1] ** -0.5)
        fields = published.ModelArgs.__dataclass_fields__
        values = {key: value for key, value in vars(cfg).items() if key in fields}
        values.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0, max_batch_size=1,
                      dspark_block_size=0, temperature=0)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            source = published.Transformer(published.ModelArgs(**values))
            model = initialized_model()
            source.load_state_dict(model.state_dict(), strict=True)
            for length in (3, 131):
                state = model.new_request()
                for name, buffer in source.named_buffers():
                    if name.endswith("score_state"):
                        buffer.fill_(-torch.inf)
                    elif name.endswith(("kv_cache", "kv_state")):
                        buffer.zero_()
                inputs = torch.randint(0, cfg.vocab_size, (1, length))
                captured = []
                handles = [layer.register_forward_hook(lambda m, i, o: captured.append(o[:, -1].clone()))
                           for layer in source.layers]
                try:
                    for step in range(4):
                        captured.clear()
                        _, expected, _ = source(inputs, state.position)
                        actual = model(inputs, state, capture_layers=True)
                        for lhs, rhs in zip(actual.layer_hiddens, captured):
                            torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
                        torch.testing.assert_close(actual.logits, expected, atol=0, rtol=0)
                        inputs = expected.argmax(-1).unsqueeze(1)
                finally:
                    for handle in handles:
                        handle.remove()
        finally:
            torch.set_default_dtype(previous)


class StrictWeightLoaderTests(unittest.TestCase):
    def test_reload_invalidates_old_requests_and_failed_load_poison_is_explicit(self):
        source = initialized_model()
        with tempfile.TemporaryDirectory() as directory:
            valid_path, invalid_path = Path(directory) / "valid.safetensors", Path(directory) / "invalid.safetensors"
            save_file(source.state_dict(), str(valid_path))
            altered = dict(source.state_dict())
            key = "layers.0.ffn.gate.tid2eid"
            altered[key] = torch.full_like(altered[key], 99)
            save_file(altered, str(invalid_path))
            target = DeepSeekV4Model(tiny_config(), torch_ops(), hadamard)
            load_converted_weights(target, valid_path, device="cpu")
            old_state = target.new_request()
            with self.assertRaisesRegex(ValueError, "expert ID"):
                load_converted_weights(target, invalid_path, device="cpu")
            with self.assertRaises(RuntimeError):
                target.new_request()
            with self.assertRaises(RuntimeError):
                target(torch.tensor([[1]]), old_state)
            load_converted_weights(target, valid_path, device="cpu")
            with self.assertRaisesRegex(ValueError, "different model or weight generation"):
                target(torch.tensor([[1]]), old_state)
            self.assertTrue(torch.isfinite(target(torch.tensor([[1]]), target.new_request()).logits).all())

    def test_roundtrip_and_missing_extra_dtype_shape_rejection(self):
        source = initialized_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(source.state_dict(), str(path))
            header, _ = read_header(path)
            target = DeepSeekV4Model(tiny_config(), torch_ops(), hadamard)
            report = load_converted_weights(target, path, device="cpu")
            self.assertEqual(report["tensor_count"], len(source.state_dict()))
            for lhs, rhs in zip(source.parameters(), target.parameters()):
                torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
            for corrupt in ("missing", "extra", "dtype", "shape", "unknown_mtp"):
                altered = json.loads(json.dumps(header))
                if corrupt == "missing":
                    altered.pop("head.weight")
                elif corrupt == "extra":
                    altered["mystery.weight"] = altered["head.weight"]
                elif corrupt == "dtype":
                    altered["head.weight"]["dtype"] = "I64"
                elif corrupt == "shape":
                    altered["head.weight"]["shape"] = [1, 2]
                else:
                    altered["mtp.0.unrecognized"] = altered["head.weight"]
                with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                    validate_header(target, altered)

    def test_truncated_or_oversized_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", 2**50))
            with self.assertRaises(ValueError):
                read_header(path)
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                read_header(path)


if __name__ == "__main__":
    unittest.main()

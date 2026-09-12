"""Compression boundary/state tests against independent and published oracles."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NATIVE = load("compressor_native_binding", ROOT / "python/llaisys/models/deepseek_v4_native.py")
REFERENCE = load("compressor_torch_reference", ROOT / "python/llaisys/models/deepseek_v4_reference.py")
LIBRARY = ROOT / "python/llaisys/libllaisys/libllaisys.so"
PUBLISHED = Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py")


def projected_oracle(kv, score, ape, ratio):
    """Materialize complete groups independently, without mutable state."""
    batch, length, width = kv.shape
    groups = length // ratio
    dimension = width // (2 if ratio == 4 else 1)
    values = kv[:, :groups * ratio].reshape(batch, groups, ratio, width)
    scores = score[:, :groups * ratio].reshape(batch, groups, ratio, width) + ape
    if ratio == 4:
        previous_kv = torch.cat((torch.zeros_like(values[:, :1, :, :dimension]),
                                 values[:, :-1, :, :dimension]), dim=1)
        previous_score = torch.cat((torch.full_like(scores[:, :1, :, :dimension], -torch.inf),
                                    scores[:, :-1, :, :dimension]), dim=1)
        values = torch.cat((previous_kv, values[..., dimension:]), dim=2)
        scores = torch.cat((previous_score, scores[..., dimension:]), dim=2)
    return (values * scores.softmax(2)).sum(2)


@unittest.skipUnless(torch.cuda.is_available() and LIBRARY.is_file(), "built library and CUDA allocation required")
class DeepSeekV4CompressorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(704)
        self.ops = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY)

    def inputs(self, ratio, dimension, length, batch=2):
        width = dimension * (2 if ratio == 4 else 1)
        kv = torch.randn(batch, length, width, device="cuda", dtype=torch.float32)
        score = 4 * torch.randn_like(kv)
        ape = torch.randn(ratio, width, device="cuda", dtype=torch.float32)
        shape = (batch, (2 if ratio == 4 else 1) * ratio, width)
        # Deliberately dirty to verify reset/masking at position zero.
        return kv, score, ape, torch.full(shape, 71., device="cuda"), torch.full(shape, 93., device="cuda")

    def test_pooling_matches_independent_softmax_at_real_dimensions(self):
        for ratio, dimension in ((4, 128), (4, 512), (128, 512)):
            with self.subTest(ratio=ratio, dimension=dimension):
                kv, score, ape, ks, ss = self.inputs(ratio, dimension, 2 * ratio + 3)
                actual = self.ops.compress_projected(kv, score, ape, ks, ss, ratio, 0)
                expected = projected_oracle(kv, score, ape, ratio)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_full_replay_and_non_aligned_chunks_preserve_state(self):
        for ratio in (4, 128):
            kv, score, ape, ks, ss = self.inputs(ratio, 128, 2 * ratio + 3)
            full = self.ops.compress_projected(kv, score, ape, ks, ss, ratio, 0)
            expected_ks, expected_ss = ks.clone(), ss.clone()
            for sizes in ([1] * kv.shape[1], [ratio - 1, 2, ratio - 2, 4]):
                position = 0
                outputs = []
                for size in sizes:
                    outputs.append(self.ops.compress_projected(
                        kv[:, position:position+size].contiguous(),
                        score[:, position:position+size].contiguous(),
                        ape, ks, ss, ratio, position,
                    ))
                    position += size
                self.assertEqual(position, kv.shape[1])
                torch.testing.assert_close(torch.cat(outputs, 1), full, atol=0, rtol=0)
                torch.testing.assert_close(ks, expected_ks, atol=0, rtol=0)
                torch.testing.assert_close(ss, expected_ss, atol=0, rtol=0)

    def test_ratio128_pool_preserves_bf16_rounding(self):
        # Small FP32 errors can be invisible to an allclose check but land on
        # opposite sides of BF16 midpoints and change downstream MoE routing.
        for seed in range(10):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                kv, score, ape, ks, ss = self.inputs(128, 512, 259, batch=2)
                actual = self.ops.compress_projected(kv, score, ape, ks, ss, 128, 0)
                expected = projected_oracle(kv, score, ape, 128)
                torch.testing.assert_close(actual.bfloat16(), expected.bfloat16(), atol=0, rtol=0)

    def test_long_projected_inputs_preserve_full_replay_and_chunk_state(self):
        # Isolate cache/pooling state from projection shape differences found
        # in the 2105-token full-model diagnostic. All paths see identical rows.
        length = 2105
        for ratio, dimension in ((4, 128), (4, 512), (128, 512)):
            with self.subTest(ratio=ratio, dimension=dimension):
                kv, score, ape, ks, ss = self.inputs(ratio, dimension, length, batch=1)
                full = self.ops.compress_projected(kv, score, ape, ks, ss, ratio, 0)
                expected_ks, expected_ss = ks.clone(), ss.clone()
                torch.testing.assert_close(full, projected_oracle(kv, score, ape, ratio), atol=3e-6, rtol=3e-6)
                chunks = [127, 1, 3, 126, 2, 129, 257, 511, 949]
                self.assertEqual(sum(chunks), length)
                for sizes in ([1] * length, chunks):
                    outputs, position = [], 0
                    for size in sizes:
                        outputs.append(self.ops.compress_projected(
                            kv[:, position:position + size].contiguous(),
                            score[:, position:position + size].contiguous(),
                            ape, ks, ss, ratio, position,
                        ))
                        position += size
                    torch.testing.assert_close(torch.cat(outputs, 1), full, atol=0, rtol=0)
                    torch.testing.assert_close(ks, expected_ks, atol=0, rtol=0)
                    torch.testing.assert_close(ss, expected_ss, atol=0, rtol=0)

    def test_partial_group_is_not_emitted_and_reset_discards_history(self):
        for ratio in (4, 128):
            kv, score, ape, ks, ss = self.inputs(ratio, 128, ratio - 1)
            result = self.ops.compress_projected(kv, score, ape, ks, ss, ratio, 0)
            self.assertEqual(result.shape, (2, 0, 128))
            width = kv.shape[-1]
            offset = ratio if ratio == 4 else 0
            torch.testing.assert_close(ks[:, offset:offset+ratio-1], kv)
            torch.testing.assert_close(ss[:, offset:offset+ratio-1], score + ape[:-1])
            if ratio == 4:
                self.assertTrue(torch.isneginf(ss[:, :ratio]).all().item())
                self.assertEqual(torch.count_nonzero(ks[:, :ratio]).item(), 0)
            self.assertEqual(width, ape.shape[-1])

    def test_large_scores_remain_finite(self):
        kv, score, ape, ks, ss = self.inputs(128, 128, 128)
        score.add_(10000)
        actual = self.ops.compress_projected(kv, score, ape, ks, ss, 128, 0)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, projected_oracle(kv, score, ape, 128), atol=3e-6, rtol=3e-6)

    def test_uses_current_cuda_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            kv, score, ape, ks, ss = self.inputs(4, 128, 11)
            actual = self.ops.compress_projected(kv, score, ape, ks, ss, 4, 0)
            expected = projected_oracle(kv, score, ape, 4)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_invalid_layout_and_alias_are_rejected(self):
        kv, score, ape, ks, ss = self.inputs(4, 128, 8)
        for call in (
            lambda: self.ops.compress_projected(kv, score, ape, ks, ks, 4, 0),
            lambda: self.ops.compress_projected(kv, score, ape, kv, ss, 4, 0),
            lambda: self.ops.compress_projected(kv, score, ape, ks, ss, 3, 0),
            lambda: self.ops.compress_projected(kv, score, ape, ks, ss, 4, -1),
            lambda: self.ops.compress_projected(kv.transpose(1, 2), score, ape, ks, ss, 4, 0),
        ):
            with self.assertRaises(NATIVE.DeepSeekV4NativeError):
                call()

    @unittest.skipUnless(PUBLISHED.is_file(), "published model source required")
    @torch.inference_mode()
    def test_adapter_matches_published_prefill_and_decode(self):
        kernel = types.ModuleType("kernel")
        for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn"):
            setattr(kernel, name, lambda *args, **kw: None)
        with mock.patch.dict(sys.modules, {"kernel": kernel}):
            published = load("compressor_published_oracle", PUBLISHED)

        def fp8(x, block_size, scale_fmt, scale_dtype, inplace):
            x.copy_(REFERENCE.simulate_fp8_activation_quant(x, block_size, power_of_two_scale=scale_fmt is not None))
            return x

        def fp4(x, block_size, inplace):
            x.copy_(REFERENCE.simulate_fp4_activation_quant(x, block_size))
            return x

        published.act_quant, published.fp4_act_quant = fp8, fp4
        native_forward = NATIVE.make_native_compressor_forward(published, self.ops)
        for ratio, dimension, rotate in ((4, 512, False), (4, 128, True), (128, 512, False)):
            args = published.ModelArgs(max_batch_size=2, max_seq_len=3 * ratio + 16,
                                       dim=16, head_dim=dimension, rope_head_dim=64)
            with torch.device("cuda"):
                original = published.Compressor(args, ratio, dimension, rotate)
                for parameter in original.parameters():
                    parameter.uniform_(-0.1, 0.1)
                original.norm.weight.fill_(1)
                original.kv_cache = torch.zeros(2, args.max_seq_len // ratio, dimension, dtype=torch.bfloat16)
                original.freqs_cis = REFERENCE.precompute_freqs_cis(64, args.max_seq_len).cuda()
                replacement = copy.deepcopy(original)
                # This prefix leaves an unfinished group; decode closes it.
                prefix = ratio - 1
                x = torch.randn(2, 2 * ratio + 3, 16, dtype=torch.bfloat16)
                position = 0
                for size in [prefix] + [1] * (x.shape[1] - prefix):
                    expected = original(x[:, position:position+size], position)
                    actual = native_forward(replacement, x[:, position:position+size], position)
                    self.assertEqual(expected is None, actual is None)
                    if expected is not None:
                        torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.008)
                    torch.testing.assert_close(replacement.kv_state, original.kv_state, atol=0, rtol=0)
                    torch.testing.assert_close(replacement.score_state, original.score_state, atol=0, rtol=0)
                    position += size
                # Full prefill reuses dirty state, then decode follows again.
                for compressor in (original, replacement):
                    compressor.kv_state.zero_()
                    compressor.score_state.fill_(-torch.inf)
                expected = original(x, 0)
                actual = native_forward(replacement, x, 0)
                torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.008)
                with self.assertRaises(NATIVE.DeepSeekV4NativeError):
                    native_forward(replacement, x[:, :1], 1)


if __name__ == "__main__":
    unittest.main()

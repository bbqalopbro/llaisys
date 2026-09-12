"""Indexer score quantization and causal Top-K boundaries on actual H=64,D=128."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).parents[1]
LIBRARY = ROOT / "python/llaisys/libllaisys/libllaisys.so"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "python/llaisys/models" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NATIVE = load("indexer_native_binding", "deepseek_v4_native.py")
REFERENCE = load("indexer_quant_reference", "deepseek_v4_reference.py")


def score_oracle(query, latent, weights):
    dots = torch.einsum("bshd,btd->bsht", query, latent)
    return (dots.relu() * weights.unsqueeze(-1)).sum(2).float()


def topk_oracle(scores, topk, start, offset):
    batch, sequence, candidates = scores.shape
    valid = (torch.arange(sequence, device=scores.device) + start + 1) // 4
    columns = torch.arange(candidates, device=scores.device)
    masked = scores.masked_fill(columns[None, None, :] >= valid[None, :, None], -torch.inf)
    selected = torch.argsort(masked, dim=-1, descending=True, stable=True)[..., :min(topk, candidates)]
    return torch.where(selected < valid[None, :, None], selected + offset, -1).int()


@unittest.skipUnless(torch.cuda.is_available() and LIBRARY.is_file(), "built library and CUDA allocation required")
class DeepSeekV4IndexerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(907)
        self.ops = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY)

    def inputs(self, candidates, batch=2, sequence=5):
        query = torch.randn(batch, sequence, 64, 128, device="cuda", dtype=torch.bfloat16)
        latent = torch.randn(batch, candidates, 128, device="cuda", dtype=torch.bfloat16)
        query = REFERENCE.simulate_fp4_activation_quant(query, 32)
        if candidates:
            latent = REFERENCE.simulate_fp4_activation_quant(latent, 32)
        weights = torch.randn(batch, sequence, 64, device="cuda", dtype=torch.bfloat16)
        return query, latent, weights

    def test_scores_match_published_bf16_boundaries(self):
        for candidates in (1, 2, 3, 4, 32, 64, 65, 512, 513, 1024):
            with self.subTest(candidates=candidates):
                inputs = self.inputs(candidates)
                actual = self.ops.indexer_scores(*inputs)
                expected = score_oracle(*inputs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_zero_candidates_do_not_launch_gemm_or_sort(self):
        scores = self.ops.indexer_scores(*self.inputs(0))
        indices = self.ops.indexer_topk(scores, 512, 0, 128)
        self.assertEqual(indices.shape, (2, 5, 0))
        self.assertEqual(self.ops.operation_counts["indexer_scores_cublas"], 0)
        self.assertEqual(self.ops.operation_counts["indexer_topk_cub"], 0)

    def test_prefill_causal_mask_and_partial_compression_groups(self):
        scores = torch.randn(2, 15, 4, device="cuda")
        # A future high score must never win selection.
        scores[..., -1] = 10000
        actual = self.ops.indexer_topk(scores, 512, 0, 15)
        torch.testing.assert_close(actual, topk_oracle(scores, 512, 0, 15), atol=0, rtol=0)
        self.assertTrue((actual[:, :3] == -1).all().item())
        self.assertTrue((actual[:, 3] >= 0).sum(-1).eq(1).all().item())

    def test_top512_really_selects_from_larger_candidate_pools(self):
        for candidates in (513, 1024, 4096):
            scores = torch.randn(2, 1, candidates, device="cuda")
            actual = self.ops.indexer_topk(scores, 512, candidates * 4 - 1, 128)
            expected = topk_oracle(scores, 512, candidates * 4 - 1, 128)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(actual.shape, (2, 1, 512))
            for row in actual[:, 0]:
                self.assertEqual(row.unique().numel(), 512)

    def test_ties_and_negative_scores_use_stable_lower_index_order(self):
        for value in (0.0, -3.0):
            scores = torch.full((1, 1, 1024), value, device="cuda")
            actual = self.ops.indexer_topk(scores, 512, 4095, 128)
            torch.testing.assert_close(actual[0, 0], torch.arange(128, 640, device="cuda", dtype=torch.int32), atol=0, rtol=0)
        scores = torch.full((1, 3, 9), -1.0, device="cuda")
        actual = self.ops.indexer_topk(scores, 512, 3, 5)
        torch.testing.assert_close(actual, topk_oracle(scores, 512, 3, 5), atol=0, rtol=0)

    def test_nondefault_stream_and_repeated_workspace_shapes(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for candidates in (65, 513, 65):
                inputs = self.inputs(candidates, batch=1, sequence=1)
                scores = self.ops.indexer_scores(*inputs)
                actual = self.ops.indexer_topk(scores, 512, candidates * 4 - 1, 128)
                expected = topk_oracle(score_oracle(*inputs), 512, candidates * 4 - 1, 128)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        stream.synchronize()

    def test_invalid_shape_dtype_and_positions_are_rejected(self):
        query, latent, weights = self.inputs(8)
        scores = torch.zeros(2, 5, 8, device="cuda")
        for call in (
            lambda: self.ops.indexer_scores(query[:, :, :32], latent, weights),
            lambda: self.ops.indexer_scores(query.float(), latent, weights),
            lambda: self.ops.indexer_scores(query, latent, weights[:, :, :32]),
            lambda: self.ops.indexer_topk(scores, 0, 0, 0),
            lambda: self.ops.indexer_topk(scores, 512, -1, 0),
            lambda: self.ops.indexer_topk(scores, 512, 0, -1),
            lambda: self.ops.indexer_topk(scores, 512, 0, 0, ratio=128),
        ):
            with self.assertRaises(NATIVE.DeepSeekV4NativeError):
                call()


if __name__ == "__main__":
    unittest.main()

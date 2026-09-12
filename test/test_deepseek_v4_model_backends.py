import unittest

import torch

from llaisys.models.deepseek_v4_backends import (
    CONTRACTS, MODEL_CONTRACTS, BackendSelectionError, OperatorRegistry, register_torch_model,
)


class ModelProjectionBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def bind(self, rows=0):
        contracts = {key: value for key, value in MODEL_CONTRACTS.items() if key not in CONTRACTS}
        registry = OperatorRegistry(contracts)
        register_torch_model(registry, backend="selected", fixed_rows=rows)
        return registry.bind({key: "selected" for key in contracts})

    def test_default_adapter_is_exact_torch(self):
        backend = self.bind()
        for dtype in (torch.bfloat16, torch.float32):
            x, weight = torch.randn(2, 7, 128).to(dtype), torch.randn(33, 128).to(dtype)
            torch.testing.assert_close(backend.function("dense_linear")(x, weight),
                                       torch.nn.functional.linear(x, weight), atol=0, rtol=0)
            grouped_x, grouped_weight = torch.randn(2, 7, 4, 128).to(dtype), torch.randn(4, 33, 128).to(dtype)
            torch.testing.assert_close(backend.function("grouped_linear")(grouped_x, grouped_weight),
                                       torch.einsum("bsgd,grd->bsgr", grouped_x, grouped_weight), atol=0, rtol=0)

    def test_fixed_rows_linear_and_grouped_match_across_non_aligned_chunks(self):
        torch.manual_seed(910)
        backend = self.bind(32)
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for dtype in (torch.bfloat16, torch.float32):
                for grouped in (False, True):
                    with self.subTest(device=device, dtype=dtype, grouped=grouped):
                        x = torch.randn((2, 137, 4, 128) if grouped else (2, 137, 128)).to(device=device, dtype=dtype)
                        weight = torch.randn((4, 33, 128) if grouped else (33, 128)).to(device=device, dtype=dtype)
                        function = backend.function("grouped_linear" if grouped else "dense_linear")
                        full = function(x, weight)
                        pieces = [function(x[:, start:end], weight) for start, end in ((0, 65), (65, 130), (130, 137))]
                        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=0, rtol=0)
                        self.assertEqual(full.dtype, dtype)

    def test_hc_reduction_has_an_explicit_shape_stable_reference_policy(self):
        value = torch.randn(1, 134, 4, 128)
        torch.testing.assert_close(self.bind().function("hc_sum")(value, 2), value.sum(dim=2), atol=0, rtol=0)
        function = self.bind(32).function("hc_sum")
        pieces = [function(value[:, start:end], 2) for start, end in ((0, 65), (65, 130), (130, 134))]
        torch.testing.assert_close(torch.cat(pieces, dim=1), function(value, 2), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            function(value.bfloat16(), 2)

    def test_inverse_rms_preserves_dtype_rounding_and_is_partition_stable(self):
        torch.manual_seed(911)
        default = self.bind().function("row_inv_rms")
        fixed = self.bind(32).function("row_inv_rms")
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for dtype in (torch.bfloat16, torch.float32):
                for width in (128, 512, 4096, 16384):
                    with self.subTest(device=device, dtype=dtype, width=width):
                        value = torch.randn(2, 137, width).to(device=device, dtype=dtype)
                        expected = torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                        torch.testing.assert_close(default(value, 1e-6), expected, atol=0, rtol=0)
                        full = fixed(value, 1e-6)
                        pieces = [fixed(value[:, start:end], 1e-6)
                                  for start, end in ((0, 65), (65, 130), (130, 137))]
                        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=0, rtol=0)
                        self.assertEqual(full.dtype, dtype)
                        self.assertEqual(full.shape, (2, 137, 1))
                        self.assertTrue(torch.isfinite(full).all())
                        if dtype == torch.bfloat16:
                            # BF16 square/mean/rsqrt is intentional for query
                            # normalization; upcasting the expression changes it.
                            upcast = torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6).to(dtype)
                            self.assertFalse(torch.equal(expected, upcast))
        self.assertEqual(fixed(torch.empty(2, 0, 128), 1e-6).shape, (2, 0, 1))
        for invalid in (torch.empty(2, 0), torch.empty(128), torch.ones(2, 128, dtype=torch.int32)):
            with self.assertRaises(ValueError):
                fixed(invalid, 1e-6)

    def test_model_operators_can_select_different_backends(self):
        contracts = {key: value for key, value in MODEL_CONTRACTS.items() if key not in CONTRACTS}
        registry = OperatorRegistry(contracts)
        register_torch_model(registry, backend="published")
        register_torch_model(registry, backend="fixed", fixed_rows=32)
        selection = {key: "published" for key in contracts}
        selection["row_inv_rms"] = "fixed"
        selected = registry.bind(selection)
        value = torch.randn(2, 7, 128)
        selected.function("row_inv_rms")(value, 1e-6)
        selected.function("dense_linear")(value, torch.randn(32, 128))
        report = selected.report()
        self.assertEqual(report["row_inv_rms"]["backend"], "fixed")
        self.assertEqual(report["dense_linear"]["backend"], "published")
        self.assertEqual(report["row_inv_rms"]["calls"], 1)
        self.assertFalse(any(row["fallback"] for row in report.values()))

    def test_indexer_tie_policy_is_explicit_and_padding_stable(self):
        topk = self.bind().function("indexer_topk")
        scores = torch.tensor([[[3., 4., 4., 4., 1.]]])
        torch.testing.assert_close(topk(scores, 2, "published"), scores.topk(2, dim=-1)[1])
        expected = torch.tensor([[[1, 2]]])
        torch.testing.assert_close(topk(scores, 2, "index_ascending"), expected)
        padded = torch.nn.functional.pad(scores, (0, 37), value=-torch.inf)
        torch.testing.assert_close(topk(padded, 2, "index_ascending"), expected)
        with self.assertRaises(ValueError):
            topk(scores, 2, "unknown")

    def test_invalid_projection_contracts_and_empty_rows(self):
        backend = self.bind(32)
        linear, grouped = backend.function("dense_linear"), backend.function("grouped_linear")
        with self.assertRaises(ValueError):
            linear(torch.zeros(2, 4), torch.zeros(3, 5))
        with self.assertRaises(ValueError):
            linear(torch.zeros(2, 4), torch.zeros(3, 4, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            grouped(torch.zeros(1, 2, 3, 4), torch.zeros(4, 8, 4))
        self.assertEqual(linear(torch.empty(2, 0, 4), torch.empty(3, 4)).shape, (2, 0, 3))
        self.assertEqual(grouped(torch.empty(2, 0, 3, 4), torch.empty(3, 8, 4)).shape, (2, 0, 3, 8))
        with self.assertRaises(BackendSelectionError):
            self.bind(-1)


if __name__ == "__main__":
    unittest.main()

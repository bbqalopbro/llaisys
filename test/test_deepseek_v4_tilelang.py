import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
INFERENCE = Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/inference")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch.cuda.is_available() and INFERENCE.is_dir(), "GPU and published kernels required")
class TileLangBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tilelang
        from tvm import tir
        condition = tir.const(True)
        buffer = tir.decl_buffer((4,), "float32", scope="local")
        tir.Allocate(buffer.data, buffer.dtype, buffer.shape, condition, tir.Evaluate(0))
        cls.kernel = load("tilelang_baseline_test_kernel", INFERENCE / "kernel.py")
        cls.reference = load("tilelang_baseline_test_reference", ROOT / "python/llaisys/models/deepseek_v4_reference.py")
        cls.dispatch = load("tilelang_baseline_test_dispatch", ROOT / "python/llaisys/models/deepseek_v4_backends.py")
        cls.tilelang_version = tilelang.__version__

    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(908)

    def tearDown(self):
        torch.set_default_dtype(self.previous_dtype)

    def assert_numeric(self, actual, expected, relative_l2=0.005):
        self.assertTrue(torch.isfinite(actual).all().item())
        error = actual.float() - expected.float()
        relative = (error.norm() / expected.float().norm().clamp_min(1e-8)).item()
        self.assertLessEqual(relative, relative_l2)

    def test_fp8_quantization_packed_and_inplace(self):
        for rows in (1, 33):
            x = torch.randn(rows, 256, device="cuda")
            for dtype in (torch.float32, torch.float8_e8m0fnu):
                packed, scales = self.kernel.act_quant(x, 128, "ue8m0", dtype)
                decoded = (packed.float().reshape(rows, 2, 128) * scales.float().unsqueeze(-1)).reshape_as(x)
                inplace = x.clone()
                self.kernel.act_quant(inplace, 128, "ue8m0", dtype, True)
                torch.testing.assert_close(inplace, decoded.bfloat16(), atol=0, rtol=0)
                expected = self.reference.simulate_fp8_activation_quant(x, 128, power_of_two_scale=True)
                torch.testing.assert_close(inplace, expected, atol=0, rtol=0)

    def test_fp4_quantization_packed_and_inplace(self):
        x = torch.randn(33, 128, device="cuda")
        packed, scales = self.kernel.fp4_act_quant(x)
        decoded = self.reference.dequantize_fp4_e2m1(packed.view(torch.uint8), scales)
        inplace = x.clone()
        self.kernel.fp4_act_quant(inplace, 32, True)
        torch.testing.assert_close(inplace, decoded.bfloat16(), atol=0, rtol=0)

    def _gemm_inputs(self, mode, rows=5):
        x = torch.randn(rows, 256, device="cuda")
        a, sa = self.kernel.act_quant(x, 128, "ue8m0", torch.float8_e8m0fnu)
        if mode == "fp4":
            b = torch.randint(0, 256, (128, 128), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
            sb = torch.full((128, 8), 0.25, device="cuda", dtype=torch.float32).to(torch.float8_e8m0fnu)
            decoded = self.reference.dequantize_fp4_e2m1(b.view(torch.uint8), sb)
        else:
            b = torch.randn(128, 256, device="cuda").to(torch.float8_e4m3fn)
            sb = torch.full((1, 2), 0.25, device="cuda", dtype=torch.float32).to(torch.float8_e8m0fnu)
            decoded = self.reference.dequantize_fp8_blocks(b, sb)
        qa = (a.float().reshape(rows, 2, 128) * sa.float().unsqueeze(-1)).reshape(rows, 256)
        return (a, sa, b, sb, torch.float8_e8m0fnu), (qa @ decoded.float().T).bfloat16()

    def test_fp8_and_fp4_gemm_against_dequantized_oracle(self):
        for mode in ("fp8", "fp4"):
            for rows in (1, 33):
                with self.subTest(mode=mode, rows=rows):
                    arguments, expected = self._gemm_inputs(mode, rows)
                    actual = getattr(self.kernel, mode + "_gemm")(*arguments)
                    self.assert_numeric(actual, expected)

    def test_tilelang_adapter_owns_dtype_and_respects_nondefault_stream(self):
        registry = self.dispatch.OperatorRegistry()
        self.dispatch.register_tilelang(registry, self.kernel, self.tilelang_version)
        backend = registry.bind({name: "tilelang" for name in self.dispatch.CONTRACTS})
        for mode in ("fp8", "fp4"):
            arguments, _ = self._gemm_inputs(mode, 33)
            expected = getattr(self.kernel, mode + "_gemm")(*arguments)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            torch.set_default_dtype(torch.float32)
            with torch.cuda.stream(stream):
                actual = backend.function(mode + "_gemm")(*arguments)
            stream.synchronize()
            self.assertEqual(torch.get_default_dtype(), torch.float32)
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.set_default_dtype(torch.bfloat16)

    def test_sparse_attention_with_mask_and_non_aligned_candidates(self):
        for tokens in (1, 3):
            q = torch.randn(1, tokens, 64, 512, device="cuda")
            kv = torch.randn(1, 130, 512, device="cuda")
            sink = torch.randn(64, device="cuda", dtype=torch.float32)
            indices = torch.arange(65, device="cuda", dtype=torch.int32).reshape(1, 1, 65).expand(1, tokens, 65).clone()
            indices[:, :, -3:] = -1
            actual = self.kernel.sparse_attn(q, kv, sink, indices, 512 ** -0.5)
            expected = self.reference.sparse_latent_attention(q, kv, sink, indices, 512 ** -0.5)
            self.assert_numeric(actual, expected)

    def test_hyperconnection(self):
        mixes = torch.randn(1, 5, 24, device="cuda", dtype=torch.float32)
        scale = torch.randn(3, device="cuda", dtype=torch.float32)
        base = torch.randn(24, device="cuda", dtype=torch.float32)
        actual = self.kernel.hc_split_sinkhorn(mixes, scale, base)
        expected = self.reference.hyperconnection_split(mixes, scale, base, 4)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)

    def test_upstream_cuda_hadamard(self):
        module = load("tilelang_baseline_test_hadamard", ROOT / "tools/deepseek_v4_reference/fast_hadamard_transform.py")
        module.configure_backend("cuda")
        for dim in (128, 512):
            x = torch.randn(3, dim, device="cuda")
            actual = module.hadamard_transform(x, dim ** -0.5)
            expected = module.torch_hadamard_transform(x, dim ** -0.5)
            self.assert_numeric(actual, expected)
        self.assertEqual(module.backend_report()["calls"], 2)
        self.assertFalse(module.backend_report()["fallback"])

    def test_deepgemm_replacement_against_tilelang(self):
        import deep_gemm
        registry = self.dispatch.OperatorRegistry()
        self.dispatch.register_tilelang(registry, self.kernel, self.tilelang_version)
        self.dispatch.register_deepgemm(registry, deep_gemm)
        selected = {name: "tilelang" for name in self.dispatch.CONTRACTS}
        selected.update(fp8_gemm="deepgemm", fp4_gemm="deepgemm")
        backend = registry.bind(selected)
        for mode in ("fp8", "fp4"):
            arguments, _ = self._gemm_inputs(mode)
            actual = backend.function(mode + "_gemm")(*arguments)
            expected = getattr(self.kernel, mode + "_gemm")(*arguments)
            self.assert_numeric(actual, expected)
        self.assertEqual(backend.report()["fp4_gemm"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()

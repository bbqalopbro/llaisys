import ctypes
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
LIBRARY_PATH = ROOT / "python" / "llaisys" / "libllaisys" / "libllaisys.so"
REFERENCE_PATH = ROOT / "python" / "llaisys" / "models" / "deepseek_v4_reference.py"
NATIVE_PATH = ROOT / "python" / "llaisys" / "models" / "deepseek_v4_native.py"
SPEC = importlib.util.spec_from_file_location("llaisys_deepseek_v4_native_reference", REFERENCE_PATH)
REFERENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFERENCE
assert SPEC.loader is not None
SPEC.loader.exec_module(REFERENCE)
NATIVE_SPEC = importlib.util.spec_from_file_location(
    "llaisys_deepseek_v4_native_binding", NATIVE_PATH
)
NATIVE = importlib.util.module_from_spec(NATIVE_SPEC)
sys.modules[NATIVE_SPEC.name] = NATIVE
assert NATIVE_SPEC.loader is not None
NATIVE_SPEC.loader.exec_module(NATIVE)


@unittest.skipUnless(LIBRARY_PATH.is_file(), "llaisys native library is not built")
class DeepSeekV4NativeAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = ctypes.CDLL(str(LIBRARY_PATH))
        try:
            cls.operation = cls.library.llaisysDeepSeekV4SparseAttentionReference
        except AttributeError as error:
            raise unittest.SkipTest("native library predates DeepSeek-V4 reference op") from error
        cls.operation.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
        ]
        cls.operation.restype = None
        cls.router = cls.library.llaisysDeepSeekV4RouterReference
        cls.router.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int,
        ]
        cls.router.restype = None
        cls.hyperconnection = (
            cls.library.llaisysDeepSeekV4HyperconnectionSplitReference
        )
        cls.hyperconnection.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int,
        ]
        cls.hyperconnection.restype = None

    def _run_native(self, device: torch.device, device_type: int):
        torch.manual_seed(31)
        batch, sequence, heads, dimension, latent_len, topk = 2, 3, 4, 8, 7, 5
        q = torch.randn(batch, sequence, heads, dimension, device=device)
        kv = torch.randn(batch, latent_len, dimension, device=device)
        sink = torch.randn(heads, device=device)
        indices = torch.tensor(
            [
                [[0, -1, -1, -1, -1], [0, 1, -1, -1, -1], [0, 1, 2, -1, -1]],
                [[3, -1, -1, -1, -1], [3, 4, -1, -1, -1], [3, 4, 5, 6, -1]],
            ],
            dtype=torch.int32,
            device=device,
        )
        output = torch.empty_like(q)
        scale = dimension ** -0.5
        self.operation(
            output.data_ptr(), q.data_ptr(), kv.data_ptr(), sink.data_ptr(),
            indices.data_ptr(), batch, sequence, heads, dimension, latent_len,
            topk, scale, device_type,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        expected = REFERENCE.sparse_latent_attention(
            q, kv, sink, indices.long(), scale
        )
        torch.testing.assert_close(output, expected, atol=3e-6, rtol=3e-6)

    def test_cpu_matches_pytorch_oracle(self):
        self._run_native(torch.device("cpu"), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_matches_pytorch_oracle(self):
        self._run_native(torch.device("cuda"), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_typed_bf16_cuda_matches_pytorch_oracle(self):
        torch.manual_seed(34)
        query = torch.randn(1, 3, 4, 16, device="cuda", dtype=torch.bfloat16)
        latent = torch.randn(1, 7, 16, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(4, device="cuda", dtype=torch.float32)
        indices = torch.tensor(
            [[[0, -1, -1], [0, 1, -1], [2, 4, 6]]],
            device="cuda", dtype=torch.int32,
        )
        native = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY_PATH)
        actual = native.sparse_attention(query, latent, sink, indices)
        torch.cuda.synchronize()
        expected = REFERENCE.sparse_latent_attention(
            query, latent, sink, indices.long()
        )
        torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_hyperconnection_matches_pytorch_oracle(self):
        torch.manual_seed(35)
        mixes = torch.randn(2, 5, 24, device="cuda", dtype=torch.float32)
        scale = torch.randn(3, device="cuda", dtype=torch.float32)
        base = torch.randn(24, device="cuda", dtype=torch.float32)
        native = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY_PATH)
        actual = native.hyperconnection_split(mixes, scale, base, 4, 20, 1e-6)
        expected = REFERENCE.hyperconnection_split(
            mixes, scale, base, 4, iterations=20, eps=1e-6
        )
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(
                actual_value, expected_value, atol=2e-6, rtol=2e-6
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_router_binding_matches_pytorch_oracle(self):
        torch.manual_seed(37)
        logits = torch.randn(9, 256, device="cuda", dtype=torch.float32)
        bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.1
        native = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY_PATH)
        weights, indices = native.route_sqrt_softplus(logits, bias, 6, 2.5)
        expected_weights, expected_indices = REFERENCE.route_experts(
            logits, torch.eye(256, device="cuda"), 6,
            route_scale=2.5, selection_bias=bias,
        )
        torch.testing.assert_close(indices.long(), expected_indices)
        torch.testing.assert_close(weights, expected_weights, atol=2e-6, rtol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_activation_quant_matches_pytorch_oracle(self):
        torch.manual_seed(38)
        native = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY_PATH)
        for mode, block_size in (("fp8", 128), ("fp4", 32)):
            source = (
                torch.randn(3, 256, device="cuda", dtype=torch.bfloat16) * 3
            )
            actual = source.clone()
            native.quantize_activation_(
                actual, block_size, mode=mode, power_of_two_scale=True
            )
            expected = (
                REFERENCE.simulate_fp8_activation_quant(source, block_size)
                if mode == "fp8"
                else REFERENCE.simulate_fp4_activation_quant(source, block_size)
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_activation_quant_supports_strided_view(self):
        torch.manual_seed(39)
        native = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY_PATH)
        storage = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
        view = storage[:, :448]
        expected = REFERENCE.simulate_fp8_activation_quant(view.clone(), 64)
        native.quantize_activation_(view, 64, mode="fp8")
        torch.testing.assert_close(view, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_quantized_linear_matches_dequantized_oracle(self):
        torch.manual_seed(40)
        native = NATIVE.DeepSeekV4NativeReferenceOps(LIBRARY_PATH)
        input = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)

        fp8_weight = torch.randn(128, 128, device="cuda").clamp(-4, 4).to(
            torch.float8_e4m3fn
        )
        fp8_scale = torch.ones(
            1, 1, device="cuda", dtype=torch.float8_e8m0fnu
        )
        actual_fp8 = native.quantized_linear(
            input, fp8_weight, fp8_scale, mode="fp8"
        )
        expected_fp8 = torch.nn.functional.linear(
            input, (fp8_weight.float() * fp8_scale.float()).to(torch.bfloat16)
        )
        torch.testing.assert_close(actual_fp8, expected_fp8, atol=0.25, rtol=0.03)
        cublas_fp8 = native.quantized_linear(
            input, fp8_weight, fp8_scale, mode="fp8", backend="cublas"
        )
        precise_fp8 = torch.nn.functional.linear(
            input.float(), fp8_weight.float() * fp8_scale.float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(cublas_fp8, precise_fp8, atol=0, rtol=0)

        packed = torch.randint(
            -128, 128, (96, 64), device="cuda", dtype=torch.int8
        )
        fp4_weight = packed.view(torch.float4_e2m1fn_x2)
        fp4_scale = torch.ones(
            96, 4, device="cuda", dtype=torch.float8_e8m0fnu
        )
        actual_fp4 = native.quantized_linear(
            input, fp4_weight, fp4_scale, mode="fp4"
        )
        expected_fp4 = torch.nn.functional.linear(
            input,
            REFERENCE.dequantize_fp4_e2m1(packed, fp4_scale).to(torch.bfloat16),
        )
        torch.testing.assert_close(actual_fp4, expected_fp4, atol=0.5, rtol=0.03)
        cublas_fp4 = native.quantized_linear(
            input, fp4_weight, fp4_scale, mode="fp4", backend="cublas"
        )
        precise_fp4 = torch.nn.functional.linear(
            input.float(), REFERENCE.dequantize_fp4_e2m1(packed, fp4_scale)
        ).to(torch.bfloat16)
        torch.testing.assert_close(cublas_fp4, precise_fp4, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_accepts_published_head_and_topk_shape(self):
        torch.manual_seed(32)
        batch, sequence, heads, dimension = 1, 2, 64, 512
        latent_len, topk = 640, 640  # 128-token window + index_topk=512
        q = torch.randn(batch, sequence, heads, dimension, device="cuda")
        kv = torch.randn(batch, latent_len, dimension, device="cuda")
        sink = torch.randn(heads, device="cuda")
        indices = torch.arange(topk, dtype=torch.int32, device="cuda")
        indices = indices.reshape(1, 1, topk).expand(batch, sequence, topk).contiguous()
        output = torch.empty_like(q)
        scale = dimension ** -0.5
        self.operation(
            output.data_ptr(), q.data_ptr(), kv.data_ptr(), sink.data_ptr(),
            indices.data_ptr(), batch, sequence, heads, dimension, latent_len,
            topk, scale, 1,
        )
        torch.cuda.synchronize()
        expected = REFERENCE.sparse_latent_attention(q, kv, sink, indices.long(), scale)
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)

    def _run_router(self, device: torch.device, device_type: int):
        torch.manual_seed(33)
        tokens, experts, topk, route_scale = 7, 256, 6, 2.5
        logits = torch.randn(tokens, experts, device=device)
        bias = torch.randn(experts, device=device) * 0.1
        weights = torch.empty(tokens, topk, device=device)
        indices = torch.empty(tokens, topk, dtype=torch.int32, device=device)
        self.router(
            weights.data_ptr(), indices.data_ptr(), logits.data_ptr(),
            bias.data_ptr(), tokens, experts, topk, route_scale, device_type,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        expected_weights, expected_indices = REFERENCE.route_experts(
            logits, torch.eye(experts, device=device), topk,
            route_scale=route_scale, selection_bias=bias,
        )
        torch.testing.assert_close(indices.long(), expected_indices)
        torch.testing.assert_close(weights, expected_weights, atol=2e-6, rtol=2e-6)

    def test_cpu_router_matches_published_semantics(self):
        self._run_router(torch.device("cpu"), 0)

    def test_cpu_hyperconnection_matches_pytorch_oracle(self):
        torch.manual_seed(36)
        mixes = torch.randn(2, 3, 24, dtype=torch.float32)
        scale = torch.randn(3, dtype=torch.float32)
        base = torch.randn(24, dtype=torch.float32)
        pre = torch.empty(2, 3, 4, dtype=torch.float32)
        post = torch.empty_like(pre)
        combination = torch.empty(2, 3, 4, 4, dtype=torch.float32)
        self.hyperconnection(
            pre.data_ptr(), post.data_ptr(), combination.data_ptr(),
            mixes.data_ptr(), scale.data_ptr(), base.data_ptr(), 6, 4, 20,
            1e-6, 0,
        )
        expected = REFERENCE.hyperconnection_split(
            mixes, scale, base, 4, iterations=20, eps=1e-6
        )
        for actual_value, expected_value in zip(
            (pre, post, combination), expected
        ):
            torch.testing.assert_close(
                actual_value, expected_value, atol=2e-6, rtol=2e-6
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_cuda_router_matches_published_semantics(self):
        self._run_router(torch.device("cuda"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

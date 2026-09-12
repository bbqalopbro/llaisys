"""Explicit, experimental B300 W4A8 decode kernel; no default backend changes.

Build the standalone library, without Torch or llaisys build dependencies::

    nvcc -O3 -std=c++17 -arch=sm_103a -DLLAISYS_B300_STANDALONE --shared -Xcompiler=-fPIC \
      src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.cu -o /tmp/w4a8_sm103.so

Then call ``W4A8DecodeSm103('/tmp/w4a8_sm103.so')(a, sa, b, sb)``. Inputs are
already quantized. This backend does not claim TileLang bitwise equivalence:
the quantization groups are unchanged, but the FP32 reduction tree differs.
"""

from __future__ import annotations

import ctypes
from pathlib import Path


class W4A8DecodeSm103:
    def __init__(self, library_path):
        self.path = Path(library_path).resolve(strict=True)
        self._library = ctypes.CDLL(str(self.path))
        self._launch = self._library.llaisys_w4a8_decode_sm103
        self._launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
        self._launch.restype = ctypes.c_int
        version = self._library.llaisys_w4a8_decode_sm103_version
        version.argtypes = []
        version.restype = ctypes.c_char_p
        self.version = version().decode("ascii")

    def __call__(self, a, sa, b, sb, *, out=None, variant=0):
        import torch

        if type(variant) is not int or variant not in (0, 1, 2, 3):
            raise ValueError("variant must be 0, 1 (four output warps), 2 (eight output warps), or 3 (four K-split warps)")
        if a.ndim != 2 or a.dtype != torch.float8_e4m3fn:
            raise ValueError("A must be FP8 E4M3 [M,K]")
        m, k = a.shape
        if not 1 <= m <= 8 or not 128 <= k <= 32768 or k % 128:
            raise ValueError("W4A8 decode requires M in [1,8] and K in [128,32768], K%128=0")
        if b.ndim != 2 or b.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
            raise ValueError("B must be packed FP4 E2M1 or raw uint8 [N,K/2]")
        n = b.shape[0]
        if not 1 <= n <= 65536 or b.shape[1] != k // 2:
            raise ValueError("B shape must be [N,K/2], with N in [1,65536]")
        if sa.dtype != torch.float8_e8m0fnu or sb.dtype != torch.float8_e8m0fnu:
            raise ValueError("SA and SB must contain UE8M0 scales")
        if sa.shape != (m, k // 128) or sb.shape != (n, k // 32):
            raise ValueError("SA must be [M,K/128], SB must be [N,K/32]")
        if a.stride(1) != 1 or sa.stride(1) != 1 or a.stride(0) < k or sa.stride(0) < k // 128:
            raise ValueError("A/SA require contiguous columns and nonoverlapping rows")
        if not b.is_contiguous() or not sb.is_contiguous():
            raise ValueError("B and SB must be contiguous")
        if a.device.type != "cuda" or any(t.device != a.device for t in (sa, b, sb)):
            raise ValueError("all inputs must be on the same CUDA device")
        if a.data_ptr() % 16 or b.data_ptr() % 16 or a.stride(0) % 16:
            raise ValueError("A/B addresses and A row stride must be 16-byte aligned")
        if out is None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        if out.shape != (m, n) or out.dtype != torch.bfloat16 or out.device != a.device or not out.is_contiguous():
            raise ValueError("output must be contiguous BF16 [M,N] on the input device")
        with torch.cuda.device(a.device):
            stream = torch.cuda.current_stream(a.device)
            result = self._launch(a.data_ptr(), sa.data_ptr(), b.data_ptr(), sb.data_ptr(),
                                  out.data_ptr(), m, n, k, a.stride(0), sa.stride(0),
                                  stream.cuda_stream, variant)
            if result:
                raise RuntimeError(f"B300 W4A8 CUDA launch failed with cudaError_t={result}")
            # The asynchronous launch borrows inputs. Tell the Torch allocator
            # about the consumer stream, including when the caller releases an
            # input immediately after this function returns.
            for tensor in (a, sa, b, sb, out):
                tensor.record_stream(stream)
        return out

    gemm = __call__

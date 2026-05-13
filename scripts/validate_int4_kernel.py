#!/usr/bin/env python3
"""Validate fused W4A16 GEMV against dequantize_int4 + linear fallback."""

import os
import sys
from ctypes import c_void_p
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import llaisys  # noqa: E402
from llaisys import DataType, DeviceType, MemcpyKind  # noqa: E402


def load_tensor(t: llaisys.Tensor, data: torch.Tensor) -> None:
    data = data.contiguous()
    t.load(c_void_p(data.data_ptr()))


def tensor_to_torch(t: llaisys.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty(t.shape(), dtype=dtype, device="cpu")
    api = llaisys.RuntimeAPI(DeviceType.NVIDIA)
    api.memcpy_sync(
        c_void_p(out.data_ptr()),
        t.data_ptr(),
        out.numel() * out.element_size(),
        MemcpyKind.D2H,
    )
    return out


def make_case(n: int, k: int, group_size: int):
    assert k % group_size == 0
    assert k % 2 == 0
    torch.manual_seed(1000 + n + k + group_size)
    x = torch.randn((1, k), dtype=torch.float16) * 0.5
    q_lo = torch.randint(0, 16, (n, k // 2), dtype=torch.uint8)
    q_hi = torch.randint(0, 16, (n, k // 2), dtype=torch.uint8)
    weight = (q_lo & 0x0F) | (q_hi << 4)
    scale = torch.rand((n, k // group_size), dtype=torch.float32) * 0.04 + 0.001
    bias = torch.randn((n,), dtype=torch.float16) * 0.1
    residual = torch.randn((1, n), dtype=torch.float16) * 0.1
    return x, weight, scale, bias, residual


def run_case(n: int, k: int, group_size: int, out_dtype: DataType,
             with_bias: bool = False, with_residual: bool = False) -> bool:
    x_h, w_h, scale_h, bias_h, residual_h = make_case(n, k, group_size)
    torch_out_dtype = torch.float16 if out_dtype == DataType.F16 else torch.float32

    x = llaisys.Tensor((1, k), DataType.F16, DeviceType.NVIDIA)
    w = llaisys.Tensor((n, k // 2), DataType.U8, DeviceType.NVIDIA)
    scale = llaisys.Tensor((n, k // group_size), DataType.F32, DeviceType.NVIDIA)
    bias = llaisys.Tensor((n,), DataType.F16, DeviceType.NVIDIA) if with_bias else None
    residual = llaisys.Tensor((1, n), DataType.F16, DeviceType.NVIDIA) if with_residual else None
    out_fused = llaisys.Tensor((1, n), out_dtype, DeviceType.NVIDIA)
    ref_dtype = DataType.F32 if (out_dtype == DataType.F16 and with_residual) else out_dtype
    out_ref = llaisys.Tensor((1, n), ref_dtype, DeviceType.NVIDIA)
    dq = llaisys.Tensor((n, k), DataType.F32, DeviceType.NVIDIA)

    load_tensor(x, x_h)
    load_tensor(w, w_h)
    load_tensor(scale, scale_h)
    if bias is not None:
        load_tensor(bias, bias_h)
    if residual is not None:
        load_tensor(residual, residual_h)

    llaisys.Ops.linear_int4(out_fused, x, w, scale, bias, group_size, residual)
    llaisys.Ops.dequantize_int4(dq, w, scale, group_size)
    llaisys.Ops.linear(out_ref, x, dq, bias)
    llaisys.RuntimeAPI(DeviceType.NVIDIA).device_synchronize()
    a = tensor_to_torch(out_fused, torch_out_dtype).float()
    if residual is not None and out_dtype == DataType.F16:
        ref_f32 = tensor_to_torch(out_ref, torch.float32)
        res_f16 = tensor_to_torch(residual, torch.float16).float()
        b = (ref_f32 + res_f16).half().float()
    else:
        if residual is not None:
            llaisys.Ops.add(out_ref, out_ref, residual)
            llaisys.RuntimeAPI(DeviceType.NVIDIA).device_synchronize()
        b = tensor_to_torch(out_ref, torch_out_dtype).float()
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    tol = 1.0e-3 if out_dtype == DataType.F16 else 1.0e-4
    ok = max_abs <= tol
    print(
        f"N={n:<6} K={k:<6} g={group_size:<3} out={out_dtype.name:<3} "
        f"bias={with_bias} residual={with_residual} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cases = [
        (128, 512, 128, DataType.F16, False, False),
        (384, 1536, 128, DataType.F16, True, False),
        (384, 1536, 64, DataType.F16, False, True),
        (1024, 8960, 128, DataType.F16, False, False),
        (4096, 1536, 128, DataType.F32, False, False),
    ]
    passed = 0
    for case in cases:
        passed += int(run_case(*case))
    print(f"\nSummary: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

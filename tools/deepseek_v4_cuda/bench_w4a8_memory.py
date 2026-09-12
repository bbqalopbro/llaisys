"""Compare B300 W4A8 SIMT, TileLang and DeepGEMM under warm/evicted L2.

Uses real expert weight payloads and identical synthetic, officially quantized
activations. DeepGEMM scale conversions and its internal layout work are inside
each operation. Graph event intervals exclude Python overhead and cache eviction.
This is an operator benchmark, not a full-model numerical or serving acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from bench_w4a8_decode import (
    eager_timing, graph_timing, load_module, metrics, sha256, write_report,
)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def deepgemm_source_state(source):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(source), *arguments], stderr=subprocess.STDOUT)
    return dict(path=str(source.resolve()),
                commit=git("rev-parse", "HEAD").decode().strip(),
                tracked_diff_sha256=hashlib.sha256(git("diff", "HEAD", "--binary", "--no-ext-diff")).hexdigest(),
                submodules=git("submodule", "status", "--recursive").decode().splitlines())


def device_l2_bytes(torch):
    # CUDA 13 cuda.h: CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38. Query the actual
    # visible device instead of assuming the capacity from its marketing name.
    driver = ctypes.CDLL("libcuda.so.1")
    get_device = driver.cuDeviceGet
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    get_device.restype = ctypes.c_int
    get_attribute = driver.cuDeviceGetAttribute
    get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    get_attribute.restype = ctypes.c_int
    device, capacity = ctypes.c_int(), ctypes.c_int()
    error = get_device(ctypes.byref(device), torch.cuda.current_device())
    if error:
        raise RuntimeError(f"cuDeviceGet failed with CUresult={error}")
    error = get_attribute(ctypes.byref(capacity), 38, device.value)
    if error or capacity.value <= 0:
        raise RuntimeError(f"L2 capacity query failed: CUresult={error}, bytes={capacity.value}")
    return capacity.value


def cold_timing(torch, operation, expected, eviction, samples):
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    # External events become actual event record nodes inside the CUDA graph.
    # One replay queues eviction -> start -> complete adapter/GEMM -> end, so
    # eviction is outside the measured interval and Python launch gaps cannot
    # enter between its start and the operation. Allocations remain graph-owned.
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    start.record()
    end.record()
    end.synchronize()  # Initialize event handles before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eviction.add_(1)
        start.record()
        result = operation()
        end.record()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    observations = []
    for _ in range(samples):
        graph.replay()
        end.synchronize()
        observations.append(start.elapsed_time(end) * 1000)
    return dict(median_us=statistics.median(observations), min_us=min(observations),
                max_us=max(observations), samples_us=observations,
                operations_per_graph=1, capture_output=metrics(torch, result, expected),
                eviction_bytes=eviction.numel() * eviction.element_size(),
                interval="external graph start/end events; eviction excluded")


def rates(timing, m, n, k, footprint):
    duration = timing["median_us"]
    if duration <= 0:
        raise RuntimeError("nonpositive CUDA event interval")
    timing["nominal_tflops"] = 2 * m * n * k / (duration * 1e6)
    timing["footprint_gbytes_per_second"] = footprint / (duration * 1e3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deepgemm-source", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--stems", nargs="+", default=[
        "layers.0.ffn.experts.0.w1", "layers.0.ffn.experts.0.w2"])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warm-repetitions", type=int, default=32)
    parser.add_argument("--evict-l2-multiple", type=int, default=4)
    args = parser.parse_args()
    if (not 7 <= args.samples <= 15 or args.warm_repetitions <= 0
            or args.evict_l2_multiple < 2 or any(m not in range(1, 9) for m in args.rows)):
        parser.error("samples must be 7..15, repetitions positive, eviction >=2 L2, rows 1..8")
    root = Path(__file__).resolve().parents[2]
    dg_source = args.deepgemm_source or root / "third_party/DeepGEMM"
    report = dict(all_passed=False, started_utc=utc_now(), command=sys.argv,
                  scope="real-weight W4A8 operator memory experiment; no full-model acceptance",
                  cases=[], numeric_gate=dict(relative_l2_max=.005, cosine_min=.99999),
                  weights_path=str(args.weights.resolve()), activation_seed=12092026,
                  activation_kind="synthetic BF16 normal values; same A and SA for every backend",
                  activation_quantization=dict(block_size=128, scale_fmt="ue8m0", scale_dtype="float8_e8m0fnu"),
                  timing_scope=dict(
                      warm="repeated same operands in CUDA graph, warm cache",
                      cold="L2-sized independent buffer read/write before every single-operation graph event interval",
                      deepgemm="SA.float(), SB.float(), any DeepGEMM internal scale layout kernels, and GEMM included; no preconverted scale reuse",
                      excluded="weight I/O, activation quantization, JIT, graph construction, eviction, Python allocation/dispatch overhead",
                      eager="separate Python wall time includes complete adapter allocation/dispatch; warm cache",
                      bandwidth="logical operand/output footprint divided by elapsed time; not measured hardware DRAM traffic",
                      cache_limit="capacity overwrite perturbs L2; no hardware guarantee that every prior line is invalidated"))
    write_report(args.output, report)
    try:
        import torch
        import tilelang
        import deep_gemm
        from safetensors import safe_open

        torch.manual_seed(report["activation_seed"])
        torch.set_default_dtype(torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (props.major, props.minor) != (10, 3):
            raise RuntimeError("benchmark requires B300 compute capability 10.3")
        wrapper_path = root / "python/llaisys/models/deepseek_v4_w4a8_cuda.py"
        kernel_path = args.source_model / "inference/kernel.py"
        wrapper = load_module("w4a8_memory_native", wrapper_path)
        kernel = load_module("w4a8_memory_published", kernel_path)
        native = wrapper.W4A8DecodeSm103(args.library)
        source_paths = [Path(__file__), Path(__file__).with_name("bench_w4a8_decode.py"),
                        wrapper_path, kernel_path, args.library,
                        root / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.cu",
                        root / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.h",
                        Path(deep_gemm.__file__), Path(deep_gemm._C.__file__)]
        hashes = {str(p.resolve()): sha256(p) for p in source_paths}
        dg_state = deepgemm_source_state(dg_source)
        l2_bytes = device_l2_bytes(torch)
        eviction_bytes = ((l2_bytes * args.evict_l2_multiple + 3) // 4) * 4
        eviction = torch.zeros(eviction_bytes // 4, dtype=torch.int32, device="cuda")
        report.update(gpu=props.name, capability=[props.major, props.minor],
                      gpu_uuid=str(props.uuid), sm_count=props.multi_processor_count,
                      l2_bytes=l2_bytes, eviction_bytes=eviction_bytes,
                      samples=args.samples, warm_repetitions=args.warm_repetitions,
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      torch=torch.__version__, tilelang=tilelang.__version__, cuda=torch.version.cuda,
                      deepgemm=dict(version=deep_gemm.__version__, source=dg_state,
                                    package=str(Path(deep_gemm.__file__).resolve()),
                                    adapter="fp8_fp4_gemm_nt; recipe_a=(1,128), recipe_b=(1,32)",
                                    jit_environment={k: v for k, v in os.environ.items()
                                                     if k in ("DG_USE_PYTORCH_CUBLASLT_HANDLE", "DG_USE_TEMP_CUBLASLT_WORKSPACE", "DG_JIT_USE_NVRTC")}),
                      source_sha256=hashes, native_version=native.version,
                      tf32_enabled=torch.backends.cuda.matmul.allow_tf32)
        write_report(args.output, report)
        order_rng = random.Random(report["activation_seed"])
        fp4_table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                                 -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                                dtype=torch.float32, device="cuda")
        for stem in args.stems:
            with safe_open(args.weights, framework="pt", device="cpu") as checkpoint:
                weight_cpu = checkpoint.get_tensor(stem + ".weight")
                scale_cpu = checkpoint.get_tensor(stem + ".scale")
            if (weight_cpu.dtype != torch.float4_e2m1fn_x2 or scale_cpu.dtype != torch.float8_e8m0fnu
                    or weight_cpu.ndim != 2 or scale_cpu.ndim != 2 or weight_cpu.shape[1] % 64
                    or scale_cpu.shape != (weight_cpu.shape[0], weight_cpu.shape[1] // 16)):
                raise ValueError(f"invalid actual FP4/E8M0 weight contract: {stem}")
            weight_hash = hashlib.sha256(weight_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            scale_hash = hashlib.sha256(scale_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            weight, sb = weight_cpu.cuda(), scale_cpu.cuda()
            n, packed_k = weight.shape
            k = packed_k * 2
            packed = weight.view(torch.uint8)
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, k).long()
            decoded = (fp4_table[codes].reshape(n, k // 32, 32) * sb.float().unsqueeze(-1)).reshape(n, k)
            for m in args.rows:
                x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                a, sa = kernel.act_quant(x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
                qx = (a.float().reshape(m, k // 128, 128) * sa.float().unsqueeze(-1)).reshape(m, k)
                expected = (qx @ decoded.T).bfloat16()
                baseline = kernel.fp4_gemm(a, sa, weight, sb, torch.float8_e8m0fnu)
                footprint = m * k + m * (k // 128) + n * (k // 2) + n * (k // 32) + 2 * m * n

                def deepgemm_operation():
                    output = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
                    deep_gemm.fp8_fp4_gemm_nt(
                        (a, sa.float()), (weight.view(torch.int8), sb.float()), output,
                        recipe_a=(1, 128), recipe_b=(1, 32))
                    return output

                operations = dict(
                    simt_4warps=lambda: native.gemm(a, sa, weight, sb, variant=1),
                    simt_8warps=lambda: native.gemm(a, sa, weight, sb, variant=2),
                    simt_splitk4=lambda: native.gemm(a, sa, weight, sb, variant=3),
                    tilelang=lambda: kernel.fp4_gemm(a, sa, weight, sb, torch.float8_e8m0fnu),
                    deepgemm_adapter=deepgemm_operation)
                order = list(operations)
                order_rng.shuffle(order)
                row = dict(weight=stem, m=m, n=n, k=k, passed=False,
                           weight_sha256=weight_hash, scale_sha256=scale_hash,
                           weight_dtype=str(weight_cpu.dtype), scale_dtype=str(scale_cpu.dtype),
                           logical_footprint_bytes=footprint, measurement_order=order,
                           baseline_vs_oracle=metrics(torch, baseline, expected), backends={})
                report["cases"].append(row)
                write_report(args.output, report)
                for name in order:
                    operation = operations[name]
                    entry = dict(passed=False)
                    row["backends"][name] = entry
                    try:
                        actual = operation()
                        entry["vs_oracle"] = metrics(torch, actual, expected)
                        entry["vs_tilelang"] = metrics(torch, actual, baseline)
                        entry["warm_graph"] = graph_timing(torch, operation, expected, args.warm_repetitions, args.samples)
                        entry["cold_single_op"] = cold_timing(torch, operation, expected, eviction, args.samples)
                        entry["eager_warm_us"] = eager_timing(torch, operation, args.warm_repetitions)
                        for timing in (entry["warm_graph"], entry["cold_single_op"]):
                            rates(timing, m, n, k, footprint)
                        entry["passed"] = all(item["passed"] for item in (
                            entry["vs_oracle"], entry["vs_tilelang"],
                            entry["warm_graph"]["capture_output"], entry["cold_single_op"]["capture_output"]))
                    except Exception as error:
                        entry["error"] = f"{type(error).__name__}: {error}"
                    write_report(args.output, report)
                    print(json.dumps(dict(weight=stem, m=m, backend=name, passed=entry["passed"],
                                          warm_us=entry.get("warm_graph", {}).get("median_us"),
                                          cold_us=entry.get("cold_single_op", {}).get("median_us"),
                                          error=entry.get("error"))), flush=True)
                row["passed"] = row["baseline_vs_oracle"]["passed"] and all(e["passed"] for e in row["backends"].values())
                reference = row["backends"]["tilelang"]
                for entry in row["backends"].values():
                    for scope in ("warm_graph", "cold_single_op"):
                        if scope in reference and scope in entry:
                            entry[scope]["speedup_vs_tilelang"] = reference[scope]["median_us"] / entry[scope]["median_us"]
                write_report(args.output, report)
        report["source_unchanged"] = all(sha256(path) == digest for path, digest in hashes.items())
        report["deepgemm_source_unchanged"] = deepgemm_source_state(dg_source) == dg_state
        report["all_passed"] = (len(report["cases"]) == len(args.rows) * len(args.stems)
                                and report["source_unchanged"] and report["deepgemm_source_unchanged"]
                                and all(row["passed"] for row in report["cases"]))
        if not report["all_passed"]:
            raise RuntimeError("one or more numerical/capture/backend/source-identity gates failed")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = utc_now()
        write_report(args.output, report)


if __name__ == "__main__":
    main()

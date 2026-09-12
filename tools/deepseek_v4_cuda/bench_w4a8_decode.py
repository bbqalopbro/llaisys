"""B300 W4A8 decode: real checkpoint slices, independent oracle, timed graphs.

This script does not import the llaisys model or change its default backend.
Build the standalone CUDA library before running inside a Slurm GPU allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def metrics(torch, actual, expected):
    actual, expected = actual.float(), expected.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    delta = actual - expected
    relative = float(delta.norm() / expected.norm().clamp_min(1e-20))
    cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
    return dict(finite=finite, relative_l2=relative, cosine=cosine,
                max_abs=float(delta.abs().max()), different_elements=int(delta.count_nonzero()),
                passed=finite and relative <= .005 and cosine >= .99999)


def graph_timing(torch, operation, expected, repetitions, samples):
    # All input generation, weight I/O and JIT compilation happen before timing.
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            result = operation()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    observations = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        observations.append(start.elapsed_time(end) * 1000 / repetitions)
    return dict(median_us=statistics.median(observations), min_us=min(observations),
                max_us=max(observations), samples_us=observations,
                operations_per_graph=repetitions, capture_output=metrics(torch, result, expected))


def eager_timing(torch, operation, repetitions):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(repetitions):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e6 / repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--stems", nargs="+", default=[
        "layers.0.ffn.experts.0.w1", "layers.0.ffn.experts.0.w2",
        "layers.3.ffn.experts.7.w1", "layers.3.ffn.experts.7.w3"])
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.samples < 3 or any(not 1 <= m <= 8 for m in args.rows):
        parser.error("positive repetitions, samples >= 3 and rows in [1,8] required")
    import torch
    import tilelang
    from safetensors import safe_open

    root = Path(__file__).resolve().parents[2]
    wrapper_path = root / "python/llaisys/models/deepseek_v4_w4a8_cuda.py"
    kernel_path = args.source_model / "inference/kernel.py"
    wrapper = load_module("llaisys_w4a8_standalone", wrapper_path)
    kernel = load_module("published_v4_gemm_oracle", kernel_path)
    native = wrapper.W4A8DecodeSm103(args.library)
    torch.manual_seed(12092026)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("this benchmark requires the B300 SM103 target")
    sources = [Path(__file__), wrapper_path, kernel_path, args.library,
               root / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.cu",
               root / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.h"]
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    report = dict(all_passed=False, scope="W4A8 real-weight operator microbenchmark; no full-model acceptance",
                  gpu=props.name, capability=[props.major, props.minor], sm_count=props.multi_processor_count,
                  gpu_memory_bytes=props.total_memory, gpu_uuid=str(props.uuid),
                  torch=torch.__version__, tilelang=tilelang.__version__, cuda=torch.version.cuda,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  command=sys.argv, source_sha256=hashes, activation_seed=12092026,
                  weights_path=str(args.weights.resolve()),
                  numeric_gate=dict(relative_l2_max=.005, cosine_min=.99999),
                  activation_quantization=dict(block_size=128, scale_fmt="ue8m0", scale_dtype="float8_e8m0fnu"),
                  timing_scope="warm-cache CUDA graph replay per GEMM; separate eager Python wall time; no activation quantization",
                  cases=[])
    write_report(args.output, report)
    try:
        for stem in args.stems:
            with safe_open(args.weights, framework="pt", device="cpu") as file:
                weight_cpu = file.get_tensor(stem + ".weight")
                scale_cpu = file.get_tensor(stem + ".scale")
            if (weight_cpu.dtype != torch.float4_e2m1fn_x2 or scale_cpu.dtype != torch.float8_e8m0fnu
                    or weight_cpu.ndim != 2 or scale_cpu.ndim != 2
                    or weight_cpu.shape[1] % 64
                    or scale_cpu.shape != (weight_cpu.shape[0], weight_cpu.shape[1] // 16)):
                raise ValueError(f"checkpoint does not contain the declared W4A8 payload and scales: {stem}")
            weight_hash = hashlib.sha256(weight_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            scale_hash = hashlib.sha256(scale_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            weight, sb = weight_cpu.cuda(), scale_cpu.cuda()
            weight = weight.view(torch.float4_e2m1fn_x2)
            sb = sb.view(torch.float8_e8m0fnu)
            n, packed_k = weight.shape
            k = packed_k * 2
            packed = weight.view(torch.uint8)
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, k).long()
            table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda", dtype=torch.float32)
            decoded = (table[codes].reshape(n, k // 32, 32) * sb.float().unsqueeze(-1)).reshape(n, k)
            for m in args.rows:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                a, sa = kernel.act_quant(x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
                qx = (a.float().reshape(m, k // 128, 128) * sa.float().unsqueeze(-1)).reshape(m, k)
                expected = (qx @ decoded.T).bfloat16()
                baseline = kernel.fp4_gemm(a, sa, weight, sb, torch.float8_e8m0fnu)
                row = dict(weight=stem, m=m, n=n, k=k, weight_sha256=weight_hash, scale_sha256=scale_hash,
                           baseline_vs_oracle=metrics(torch, baseline, expected), variants=[])
                baseline_op = lambda: kernel.fp4_gemm(a, sa, weight, sb, torch.float8_e8m0fnu)
                row["tilelang_graph"] = graph_timing(torch, baseline_op, expected, args.repetitions, args.samples)
                row["tilelang_eager_us"] = eager_timing(torch, baseline_op, args.repetitions)
                for variant in (1, 2, 3):
                    operation = lambda: native.gemm(a, sa, weight, sb, variant=variant)
                    actual = operation()
                    entry = dict(variant=variant, vs_oracle=metrics(torch, actual, expected),
                                 vs_tilelang=metrics(torch, actual, baseline))
                    entry["graph"] = graph_timing(torch, operation, expected, args.repetitions, args.samples)
                    entry["eager_us"] = eager_timing(torch, operation, args.repetitions)
                    entry["graph_speedup"] = row["tilelang_graph"]["median_us"] / entry["graph"]["median_us"]
                    row["variants"].append(entry)
                # Different stream, padded row strides, and two in-flight outputs.
                a_pad = torch.empty(m, k + 16, dtype=a.dtype, device=a.device)
                sa_pad = torch.empty(m, k // 128 + 16, dtype=sa.dtype, device=sa.device)
                a_view, sa_view = a_pad[:, :k], sa_pad[:, :k // 128]
                a_view.copy_(a); sa_view.copy_(sa)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                row["nondefault_stream"] = []
                for variant in (0, 1, 2, 3):
                    with torch.cuda.stream(stream):
                        out1 = native.gemm(a_view, sa_view, weight, sb, variant=variant)
                        out2 = native.gemm(a, sa, weight, sb, variant=variant)
                    stream.synchronize()
                    row["nondefault_stream"].append(dict(variant=variant, strided=metrics(torch, out1, expected),
                        independent_outputs=out1.data_ptr() != out2.data_ptr() and torch.equal(out1, out2)))
                row["passed"] = (row["baseline_vs_oracle"]["passed"] and row["tilelang_graph"]["capture_output"]["passed"]
                                 and all(s["strided"]["passed"] and s["independent_outputs"] for s in row["nondefault_stream"])
                                 and all(v["vs_oracle"]["passed"] and v["vs_tilelang"]["passed"]
                                 and v["graph"]["capture_output"]["passed"] for v in row["variants"]))
                report["cases"].append(row)
                write_report(args.output, report)
                print(json.dumps(dict(weight=stem, m=m, passed=row["passed"], tilelang_us=row["tilelang_graph"]["median_us"],
                                      variants=[dict(variant=v["variant"], us=v["graph"]["median_us"], speedup=v["graph_speedup"], rel_l2=v["vs_tilelang"]["relative_l2"]) for v in row["variants"]])), flush=True)
                if not row["passed"]:
                    raise RuntimeError("W4A8 correctness gate failed")
        report["source_unchanged"] = all(sha256(p) == digest for p, digest in hashes.items())
        report["all_auto_variants_faster"] = all(c["variants"][0]["graph_speedup"] > 1 for c in report["cases"])
        report["all_passed"] = bool(report["cases"]) and report["source_unchanged"] and all(c["passed"] for c in report["cases"])
        if not report["all_passed"]:
            raise RuntimeError("source changed during benchmark")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_report(args.output, report)


if __name__ == "__main__":
    main()

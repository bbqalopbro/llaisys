"""Component-only Indexer latency; this does not measure serving TTFT/TPOT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("invalid benchmark repeat counts")
    root = Path(__file__).resolve().parents[2]
    native = load("bench_indexer_native", root / "python/llaisys/models/deepseek_v4_native.py")
    reference = load("bench_indexer_reference", root / "python/llaisys/models/deepseek_v4_reference.py")
    ops = native.DeepSeekV4NativeReferenceOps(args.library)
    torch.manual_seed(907)
    results = []

    def summarize(values):
        return {"mean": statistics.mean(values), "p50": statistics.median(values),
                "p95": sorted(values)[min(len(values)-1, int(0.95 * len(values)))]}

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        gpu, wall = [], []
        for _ in range(args.repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.perf_counter()
            start.record()
            output = fn()
            end.record()
            end.synchronize()
            gpu.append(start.elapsed_time(end))
            wall.append((time.perf_counter() - began) * 1000)
            del output
        return {"gpu_ms": summarize(gpu), "wall_ms": summarize(wall)}

    for sequence, candidates in ((1, 64), (1, 513), (1, 4096), (259, 64), (2105, 526)):
        start_pos = candidates * 4 - 1 if sequence == 1 else 0
        offset = 128 if sequence == 1 else sequence
        query = reference.simulate_fp4_activation_quant(
            torch.randn(1, sequence, 64, 128, dtype=torch.bfloat16, device="cuda"), 32)
        latent = reference.simulate_fp4_activation_quant(
            torch.randn(1, candidates, 128, dtype=torch.bfloat16, device="cuda"), 32)
        weights = torch.randn(1, sequence, 64, dtype=torch.bfloat16, device="cuda")
        valid = (torch.arange(sequence, device="cuda") + start_pos + 1) // 4
        mask = torch.arange(candidates, device="cuda")[None, None, :] >= valid[None, :, None]

        def published_scores():
            dots = torch.einsum("bshd,btd->bsht", query, latent)
            return (dots.relu() * weights.unsqueeze(-1)).sum(2).float()

        def published():
            scores = published_scores().masked_fill(mask, -torch.inf)
            indices = scores.topk(min(512, candidates), dim=-1).indices
            return torch.where(indices < valid[None, :, None], indices + offset, -1).int()

        def native_call():
            return ops.indexer_topk(ops.indexer_scores(query, latent, weights), 512, start_pos, offset)

        expected_scores = published_scores()
        torch.testing.assert_close(ops.indexer_scores(query, latent, weights), expected_scores, atol=0, rtol=0)
        expected, actual = published(), native_call()
        # Equal-score ties may use different IDs; sorted selected scores and
        # the number of valid entries must still exactly match torch.topk.
        def selected_scores(indices):
            values = expected_scores.gather(-1, (indices.long() - offset).clamp_min(0))
            return values.masked_fill(indices < 0, -torch.inf)
        torch.testing.assert_close(selected_scores(actual), selected_scores(expected), atol=0, rtol=0)
        results.append({"batch": 1, "sequence": sequence, "candidates": candidates,
                        "heads": 64, "dimension": 128, "topk": min(512, candidates),
                        "score_values_exact": True, "selected_scores_exact": True,
                        "published": measure(published), "native": measure(native_call)})
    result = {"kind": "component_latency_not_serving_ttft_tpot", "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "BF16 with FP4 QAT inputs",
              "warmup": args.warmup, "repeat": args.repeat, "cuda_graph": False,
              "native_ties": "stable lower candidate index", "published_ties": "torch.topk unspecified",
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "cases": results, "native_operation_counts": ops.operation_counts}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

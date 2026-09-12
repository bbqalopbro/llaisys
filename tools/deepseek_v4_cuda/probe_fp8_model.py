"""WIP: incomplete full-model probe for the experimental B300 FP8 kernel.

Paused at the user's request on 2026-09-12. This draft requires the not-yet
implemented llaisys.models.deepseek_v4_fp8_cuda wrapper and is not runnable.
It has not passed CPU runtime or full-model GPU validation. Existing FP8
projection benchmarks do not establish full-model numerical correctness.

One independent 43-layer model loads the real MP1 checkpoint once. Only FP8 dense
GEMM changes: flattened M>=32 uses standalone CUDA; M<32 uses the registered
TileLang implementation. This declared shape dispatch never catches a CUDA
failure to change backend. No serving-performance conclusions are supported.

Defaults to eight original published chat cases with checkpoint payload and
golden-file hash bindings. Run --suite top512 separately for the 2105-token
golden; each suite preserves its original max-seq-len/RoPE policy. This probe
requires the local independent V4 model and reference runner, which are not
included in the standalone operator-only distribution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tools/deepseek_v4_reference"
BACKEND = "experimental-fp8-sm103-prefill32-tilelang-small-m"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while data := handle.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def default_baselines(suite):
    names = {
        "short-long": [("tilelang_accuracy", "tilelang_golden"),
                       ("tilelang_long_accuracy", "tilelang_long_golden")],
        "chat": [("chat_published", "chat_golden")],
        "top512": [("tilelang_top512_accuracy", "tilelang_top512_golden")],
    }
    return [(ROOT / f"benchmark_results/deepseek_v4_b300_{report}.json",
             ROOT / f"benchmark_results/deepseek_v4_{golden}") for report, golden in names[suite]]


class ShapeDispatch:
    """The registered model operator, with actual attempts and completed calls."""

    def __init__(self, native, tilelang_gemm):
        self.native, self.tilelang_gemm = native, tilelang_gemm
        self.attempts, self.completed, self.failures = Counter(), Counter(), Counter()
        self.shapes = Counter()

    def __call__(self, a, a_s, b, b_s, scale_dtype=None):
        import torch
        from llaisys.models.deepseek_v4_backends import _validate_gemm

        if scale_dtype != torch.float8_e8m0fnu:
            raise ValueError("this explicit FP8 model probe requires UE8M0 scales")
        m, n, k = _validate_gemm("fp8", a, a_s, b, b_s, scale_dtype)
        route = "empty" if m == 0 else "cuda_prefill" if m >= 32 else "tilelang_small_m"
        self.attempts[route] += 1
        self.shapes[(route, m, n, k)] += 1
        try:
            if route == "empty":
                result = torch.empty((*a.shape[:-1], n), device=a.device, dtype=torch.bfloat16)
            elif route == "cuda_prefill":
                result = self.native.gemm(a.reshape(m, k), a_s.reshape(m, k // 128),
                                          b, b_s)
                result = result.reshape(*a.shape[:-1], n)
            else:
                result = self.tilelang_gemm(a, a_s, b, b_s, scale_dtype)
        except Exception:
            self.failures[route] += 1
            raise
        self.completed[route] += 1
        return result

    def report(self):
        return {
            "policy": "M=0: empty; M>=32: standalone CUDA; 1<=M<32: registered TileLang",
            "policy_selected_before_execution": True, "error_fallback": False,
            "minimum_native_rows": 32, "plan_cache": self.native.report(),
            "attempts": dict(self.attempts), "calls": dict(self.completed),
            "failures": dict(self.failures),
            "call_count_semantics": "Python calls returned; final synchronization verifies device completion",
            "shapes": [{"backend": route, "M": m, "N": n, "K": k, "calls": count}
                       for (route, m, n, k), count in sorted(self.shapes.items())],
        }


def run(args, report):
    source, checkpoint, library = (p.resolve(strict=True) for p in
                                   (args.source_model, args.checkpoint, args.library))
    pairs = [(Path(r).resolve(), Path(g).resolve()) for r, g in
             (args.baseline or default_baselines(args.suite))]
    baseline_reports = [json.loads(path.read_text()) for path, _ in pairs]
    lengths = {int(r["provenance"]["command"][r["provenance"]["command"].index("--max-seq-len") + 1])
               for r in baseline_reports}
    if len(lengths) != 1:
        raise ValueError("all baselines must use the same max-seq-len/RoPE cache policy")
    max_seq_len = lengths.pop()
    if args.max_seq_len is not None and args.max_seq_len != max_seq_len:
        raise ValueError("--max-seq-len differs from the original golden reports")

    # Snapshot the complete local Python package, plus the actual CUDA/runner
    # sources, selected library, tokenizer artifacts and baseline inputs.
    paths = set((ROOT / "python/llaisys").rglob("*.py"))
    paths.update((Path(__file__), REFERENCE / "run_independent.py",
                  REFERENCE / "fast_hadamard_transform.py", library,
                  ROOT / "src/ops/deepseek_v4/nvidia/fp8_prefill_sm103.cu",
                  ROOT / "src/ops/deepseek_v4/nvidia/fp8_prefill_sm103.h",
                  source / "config.json", source / "inference/config.json",
                  source / "inference/kernel.py", source / "inference/model.py"))
    paths.update(source.glob("encoding/*.py"))
    paths.update(checkpoint.parent.glob("*.json"))
    for report_path, directory in pairs:
        paths.add(report_path)
        paths.update(directory.glob("case_*.safetensors"))
    before = {str(path.resolve()): sha256(path) for path in sorted(paths)}
    report["provenance"]["file_sha256_before"] = before
    report["stage"] = "import_runtime"
    write_report(args.output, report)

    sys.path.insert(0, str(REFERENCE))
    sys.path.insert(0, str(ROOT / "python"))
    spec = importlib.util.spec_from_file_location("llaisys_fp8_probe_reference", REFERENCE / "run_independent.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    import torch
    import tilelang
    import tvm_ffi
    from transformers import AutoTokenizer
    from llaisys.models.deepseek_v4_backends import (
        CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model,
    )
    from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
    from llaisys.models.deepseek_v4_model import DeepSeekV4Model, InferenceConfig, load_converted_weights
    from llaisys.models.deepseek_v4_fp8_cuda import FP8PrefillSm103
    import fast_hadamard_transform as hadamard

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("this B300 probe requires an allocated SM103 CUDA device")
    initial_dtype = torch.get_default_dtype()
    cfg = InferenceConfig.from_directory(source, max_seq_len=max_seq_len)
    if cfg.n_layers != 43 or cfg.expert_dtype != "fp4" or cfg.weight_dtype != "fp8":
        raise ValueError("probe requires the actual 43-layer FP4-expert/FP8-dense V4 Flash config")
    report.update(model_config=asdict(cfg), gpu=torch.cuda.get_device_name(),
                  compute_capability=list(torch.cuda.get_device_capability()),
                  torch=torch.__version__, cuda=torch.version.cuda, tilelang=tilelang.__version__,
                  tvm_ffi=tvm_ffi.__version__, global_default_dtype=str(initial_dtype),
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                  stage="checkpoint_payload_hash")
    write_report(args.output, report)
    print("hashing the entire real MP1 checkpoint payload", flush=True)
    started = time.perf_counter()
    identity = checkpoint_identity(checkpoint)
    report.update(checkpoint_identity=identity, checkpoint_hash_seconds=time.perf_counter() - started)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    cases = runner.read_baselines(pairs, source, checkpoint, max_seq_len, tokenizer, cfg.vocab_size, identity)
    report["baseline_verification"] = {
        "original_published_tilelang": True, "source_model_hashes_verified": True,
        "tokenizer_and_golden_tensors_validated": True,
        "all_checkpoint_payloads_bound": all(c["checkpoint_payload_verified"] for c in cases),
        "all_golden_files_bound_in_original_reports": all(r.get("golden_file_sha256") for r in baseline_reports),
        "limitation": (None if all(c["checkpoint_payload_verified"] for c in cases) else
                       "Legacy short/134-token reports do not bind checkpoint payload or golden-file hashes. "
                       "Current payload/golden hashes and original source hashes are recorded; historical "
                       "checkpoint-byte identity cannot be retroactively established."),
        "reports": [{"path": str(path), "sha256": before[str(path)],
                     "checkpoint_payload_bound": original.get("checkpoint_identity") is not None,
                     "golden_file_hashes_bound": original.get("golden_file_sha256") is not None}
                    for (path, _), original in zip(pairs, baseline_reports)],
    }
    if not cases or not all(c["checkpoint_payload_verified"] for c in cases):
        raise ValueError("FP8 probe requires original goldens bound to the complete checkpoint payload")
    if not all(r.get("golden_file_sha256") for r in baseline_reports):
        raise ValueError("FP8 probe requires original golden-file hashes")
    report["expected_case_count"] = len(cases)
    report["expected_teacher_forced_steps"] = sum(c["tensors"]["logits"].shape[0] for c in cases)

    registry = OperatorRegistry(MODEL_CONTRACTS)
    kernel = runner.load_kernel(source / "inference/kernel.py")
    register_tilelang(registry, kernel, tilelang.__version__)
    register_torch_model(registry)
    selection = {name: "tilelang" if name in CONTRACTS else "torch" for name in MODEL_CONTRACTS}
    baseline_ops = registry.bind(selection)
    native = FP8PrefillSm103(library, workspace_mb=args.workspace_mb)
    dispatch = ShapeDispatch(native, baseline_ops.function("fp8_gemm"))
    registry.register(BACKEND, "fp8_gemm", dispatch, version=native.version,
                      contract=CONTRACTS["fp8_gemm"])
    selection["fp8_gemm"] = BACKEND
    ops = registry.bind(selection)
    hadamard.configure_backend("cuda")
    extension = str(Path(hadamard.backend_report()["extension"]).resolve())
    before[extension] = sha256(extension)
    report["operator_selection"] = selection
    report["kernel_library"] = {"path": str(library), "sha256": before[str(library)], "version": native.version}
    report["stage"] = "strict_weight_load"
    write_report(args.output, report)
    model = DeepSeekV4Model(cfg, ops, hadamard.hadamard_transform)
    started = time.perf_counter()
    report["weights"] = load_converted_weights(model, checkpoint)
    torch.cuda.synchronize()
    report["weight_load_seconds"] = time.perf_counter() - started
    report["weight_load_count"] = 1
    layer_calls = [0] * len(model.layers)
    def hook(index):
        def count(_module, _inputs, _output):
            layer_calls[index] += 1
        return count
    handles = [layer.register_forward_hook(hook(index)) for index, layer in enumerate(model.layers)]

    def live_evidence():
        report.update(operator_backends=ops.report(), fp8_dispatch=dispatch.report(),
                      registered_tilelang_backends=baseline_ops.report(), hadamard=hadamard.backend_report(),
                      layer_forward_calls=list(layer_calls))

    try:
        for index, case in enumerate(cases):
            report.update(stage="evaluate", active_case=index)
            write_report(args.output, report)
            counts_before = dict(dispatch.completed)
            result = runner.evaluate(model, tokenizer, case, args.warmup, None)
            result.pop("_logits")
            if args.free_generation:
                result["free_generation"] = runner.evaluate_free_generation(model, tokenizer, case, source, 0, None)
            native.clear()  # Synchronizes before freeing every weight/shape plan.
            result["fp8_calls"] = {key: count - counts_before.get(key, 0)
                                     for key, count in dispatch.completed.items()}
            report["cases"].append(result)
            live_evidence()
            write_report(args.output, report)
            print(f"case {index}: tokens={result['input_tokens']} steps={len(result['steps'])} "
                  f"numerical_gate={result['numerical_gate_passed']} dispatch={result['fp8_calls']}", flush=True)
        torch.cuda.synchronize()
        report["device_completion_synchronized"] = True
    finally:
        try:
            native.close()
            report["native_plans_closed"] = True
        finally:
            live_evidence()
        for handle in handles:
            handle.remove()
        after = {path: sha256(path) if Path(path).is_file() else None for path in before}
        report["provenance"]["file_sha256_after"] = after
        report["source_unchanged"] = after == before
        try:
            verify_checkpoint_unchanged(checkpoint, identity)
            report["checkpoint_unchanged"] = True
        except ValueError:
            report["checkpoint_unchanged"] = False
        report["global_default_dtype_unchanged"] = torch.get_default_dtype() == initial_dtype
        imported = [name for name, module in list(sys.modules.items())
                    if getattr(module, "__file__", None)
                    and Path(module.__file__).resolve() == source / "inference/model.py"]
        report["published_model_imported"] = bool(imported)
        report["published_model_module_names"] = imported

    report["all_43_layers_executed"] = len(layer_calls) == 43 and min(layer_calls) > 0 and len(set(layer_calls)) == 1
    report["all_numerical_gates_passed"] = len(report["cases"]) == len(cases) and all(
        c["numerical_gate_passed"] for c in report["cases"])
    report["all_free_generation_ids_match"] = (all(
        c["free_generation"]["matches_golden_ids"] for c in report["cases"])
        if args.free_generation else None)
    # Fixture-answer quality belongs to the original model, independently of
    # numerical equivalence. In particular the published chat golden already
    # contains three fixture misses; reproducing them is not GEMM regression.
    report["functional_fixture_results"] = {
        "affects_kernel_acceptance": False,
        "published_baseline_expected_match": [c.get("expected_match") for r in baseline_reports for c in r["cases"]],
        "teacher_forced_substring_expected_match": [c["expected_match"] for c in report["cases"]],
        "free_generation_expected_match": ([c["free_generation"].get("expected_match") for c in report["cases"]]
                                           if args.free_generation else None),
    }
    report["both_shape_routes_exercised"] = all(dispatch.completed.get(k, 0) > 0
                                                for k in ("cuda_prefill", "tilelang_small_m"))
    report["all_passed"] = bool(report["all_numerical_gates_passed"] and report["all_43_layers_executed"]
                                and report["all_free_generation_ids_match"] is not False
                                and report["both_shape_routes_exercised"] and report["source_unchanged"]
                                and report["checkpoint_unchanged"] and report["global_default_dtype_unchanged"]
                                and report.get("native_plans_closed", False)
                                and not report["published_model_imported"] and not dispatch.failures)
    report["stage"] = "completed"
    report["status"] = "passed" if report["all_passed"] else "failed_numerical_or_integrity_gate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors"))
    parser.add_argument("--library", type=Path, default=ROOT / "build/libv4_fp8_prefill_sm103_v2.so")
    parser.add_argument("--suite", choices=("chat", "top512"), default="chat")
    parser.add_argument("--baseline", nargs=2, action="append", metavar=("REPORT", "LOGITS_DIR"))
    parser.add_argument("--max-seq-len", type=int, help="optional assertion; otherwise inferred from golden report")
    parser.add_argument("--workspace-mb", type=int, default=64,
                        help="workspace for each weight/shape plan; plans cleared between cases")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--free-generation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workspace_mb < 0:
        parser.error("--workspace-mb must be nonnegative")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    report = {
        "status": "running", "stage": "setup", "all_passed": False,
        "executor": "llaisys-independent-43-layer-explicit-fp8-probe",
        "cases": [], "free_generation_tested": args.free_generation,
        "numerical_gate": {"min_cosine": 0.999, "max_relative_l2": 0.01, "require_same_argmax": True},
        "measurement": {"kind": "correctness_probe_only", "performance_claim": False,
                        "batch_size": 1, "warmup_per_case": args.warmup,
                        "cuda_graph": False, "error_fallback": False, "declared_shape_dispatch": True,
                        "algorithm_selection": "fixed cuBLASLt heuristic index 0, no tuning",
                        "plan_cache_lifetime": "one case, including optional free generation; clear synchronizes",
                        "graph_policy": "Python wrapper rejects CUDA graph capture before native calls"},
        "provenance": {"command": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    started = time.perf_counter()
    try:
        report["provenance"]["commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        run(args, report)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed_exception", all_passed=False,
                      failure={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Preserve an integrity result even when import, identity verification,
        # or strict weight loading fails before evaluation begins.
        if "source_unchanged" not in report and report["provenance"].get("file_sha256_before"):
            before = report["provenance"]["file_sha256_before"]
            try:
                after = {path: sha256(path) if Path(path).is_file() else None for path in before}
                report["provenance"]["file_sha256_after"] = after
                report["source_unchanged"] = before == after
            except OSError as exc:
                report["source_unchanged"] = False
                report["source_integrity_error"] = str(exc)
        write_report(args.output, report)
    print(json.dumps({"status": report["status"], "all_passed": report["all_passed"],
                      "report": str(args.output)}), flush=True)
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare CUDA and registered TileLang on every real small-M FP4 call in chat case 0.

Both operators receive exactly the same quantized activation, weight and UE8M0
scales. The model always receives the TileLang result, so this diagnostic follows
the control trajectory. It does not modify the frozen model probe or a default
backend. Two teacher-forced steps run through the complete 43-layer model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tools/deepseek_v4_reference"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while data := handle.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class ShadowFp4:
    def __init__(self, torch, native, reference, variant, report):
        self.torch, self.native, self.reference, self.variant = torch, native, reference, variant
        self.report = report
        self.names, self.fingerprints = {}, {}
        self.shape_stats = {}
        self.counts = Counter()

    def tensor_range(self, value):
        torch = self.torch
        fp = value.float()
        finite = torch.isfinite(fp)
        count = int(finite.sum().item())
        return {"dtype": str(value.dtype), "shape": list(value.shape), "stride": list(value.stride()),
                "finite_elements": count, "elements": value.numel(),
                "minimum": fp[finite].min().item() if count else None,
                "maximum": fp[finite].max().item() if count else None}

    def detail(self, row, a, sa, b, sb):
        key = b.data_ptr()
        if key not in self.fingerprints:
            self.fingerprints[key] = {
                "weight_sha256": hashlib.sha256(b.view(self.torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                "weight_scale_sha256": hashlib.sha256(sb.view(self.torch.uint8).cpu().numpy().tobytes()).hexdigest(),
            }
        return {**row, **self.fingerprints[key], "activation": self.tensor_range(a),
                "activation_scale": self.tensor_range(sa), "weight_scale": self.tensor_range(sb)}

    def __call__(self, a, sa, b, sb, scale_dtype=None):
        from llaisys.models.deepseek_v4_backends import _validate_gemm
        torch = self.torch
        if scale_dtype != torch.float8_e8m0fnu:
            raise ValueError("shadow comparison requires the real UE8M0 model contract")
        m, n, k = _validate_gemm("fp4", a, sa, b, sb, scale_dtype)
        self.counts["fp4_total"] += 1
        if not 1 <= m <= 8:
            self.counts["tilelang_large_or_empty_m"] += 1
            return self.reference(a, sa, b, sb, scale_dtype)
        self.counts["cuda_small_m_attempts"] += 1
        actual = self.native.gemm(a.reshape(m, k), sa.reshape(m, k // 128), b, sb,
                                  variant=self.variant).reshape(*a.shape[:-1], n)
        expected = self.reference(a, sa, b, sb, scale_dtype)
        self.counts["tilelang_small_m_calls"] += 1
        difference = actual.float() - expected.float()
        finite = bool(torch.isfinite(actual).all().item())
        row = {"call_index": self.counts["fp4_total"] - 1, "small_m_index": len(self.report["calls"]),
               "phase": self.report["phase"], "weight": self.names.get(b.data_ptr(), "unidentified"),
               "M": m, "N": n, "K": k, "cuda_variant": self.variant,
               "finite": finite, "different_elements": int((actual != expected).sum().item()),
               "elements": actual.numel(), "exact_equal": bool(torch.equal(actual, expected)),
               "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-30)).item() if finite else None,
               "max_abs_error": difference.abs().max().item() if finite else None,
               "cosine": torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(),
                                                                 dim=0).item() if finite else None}
        self.report["calls"].append(row)
        self.counts["cuda_small_m_completed"] += 1
        key = (m, n, k)
        stats = self.shape_stats.setdefault(key, {"M": m, "N": n, "K": k, "calls": 0,
            "nonexact_calls": 0, "nonfinite_calls": 0, "different_elements": 0,
            "max_relative_l2": 0.0, "max_abs_error": 0.0})
        stats["calls"] += 1
        stats["nonexact_calls"] += not row["exact_equal"]
        stats["nonfinite_calls"] += not finite
        stats["different_elements"] += row["different_elements"]
        if finite:
            stats["max_relative_l2"] = max(stats["max_relative_l2"], row["relative_l2"])
            stats["max_abs_error"] = max(stats["max_abs_error"], row["max_abs_error"])
        keys = []
        if "first_call" not in self.report:
            keys.append("first_call")
        if not row["exact_equal"] and "first_nonexact_call" not in self.report:
            keys.append("first_nonexact_call")
        if finite and ("worst_relative_l2_call" not in self.report or
                       row["relative_l2"] > self.report["worst_relative_l2_call"]["relative_l2"]):
            keys.append("worst_relative_l2_call")
        if finite and ("worst_absolute_error_call" not in self.report or
                       row["max_abs_error"] > self.report["worst_absolute_error_call"]["max_abs_error"]):
            keys.append("worst_absolute_error_call")
        if not finite and "first_nonfinite_call" not in self.report:
            keys.append("first_nonfinite_call")
        if keys:
            detail = self.detail(row, a, sa, b, sb)
            for name in keys:
                self.report[name] = detail
        if len(self.report["calls"]) % 256 == 0:
            print(f"shadow small-M calls={len(self.report['calls'])}; latest={row['weight']} "
                  f"relL2={row['relative_l2']} nonexact={row['different_elements']}", flush=True)
        return expected

    def finish(self):
        self.report["counts"] = dict(self.counts)
        self.report["per_shape"] = [self.shape_stats[key] for key in sorted(self.shape_stats)]
        self.report["nonexact_calls"] = sum(not row["exact_equal"] for row in self.report["calls"])


def run(args, report):
    sys.path.insert(0, str(REFERENCE))
    sys.path.insert(0, str(ROOT / "python"))
    spec = importlib.util.spec_from_file_location("llaisys_w4a8_shadow_reference", REFERENCE / "run_independent.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    import torch
    import tilelang
    from transformers import AutoTokenizer
    from llaisys.models.deepseek_v4_backends import (
        CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model,
    )
    from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
    from llaisys.models.deepseek_v4_model import DeepSeekV4Model, InferenceConfig, load_converted_weights
    from llaisys.models.deepseek_v4_w4a8_cuda import W4A8DecodeSm103
    import fast_hadamard_transform as hadamard

    source, checkpoint, library = [p.resolve(strict=True) for p in (args.source_model, args.checkpoint, args.library)]
    baseline = ROOT / "benchmark_results/deepseek_v4_b300_chat_published.json"
    golden = ROOT / "benchmark_results/deepseek_v4_chat_golden"
    paths = set((ROOT / "python/llaisys").rglob("*.py"))
    paths.update([Path(__file__), REFERENCE / "run_independent.py", REFERENCE / "fast_hadamard_transform.py",
                  source / "config.json", source / "inference/config.json", source / "inference/kernel.py",
                  source / "inference/model.py", source / "encoding/encoding_dsv4.py", library, baseline,
                  ROOT / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.cu",
                  ROOT / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.h"])
    paths.update(checkpoint.parent.glob("*.json"))
    paths.update(golden.glob("case_*.safetensors"))
    before = {str(path.resolve()): sha256(path) for path in sorted(paths)}
    report["source_sha256_before"] = before
    report["stage"] = "checkpoint_identity"
    save(args.output, report)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("an allocated SM103 B300 GPU is required")
    print("hashing complete MP1 checkpoint", flush=True)
    identity = checkpoint_identity(checkpoint)
    report["checkpoint_identity"] = identity
    cfg = InferenceConfig.from_directory(source, max_seq_len=4096)
    if cfg.n_layers != 43:
        raise ValueError("expected 43 model layers")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    case = runner.read_baselines([(baseline, golden)], source, checkpoint, 4096, tokenizer, cfg.vocab_size, identity)[0]
    if case["tensors"]["logits"].shape[0] != 2:
        raise ValueError("expected the original two-step chat case 0")
    registry = OperatorRegistry(MODEL_CONTRACTS)
    register_tilelang(registry, runner.load_kernel(source / "inference/kernel.py"), tilelang.__version__)
    register_torch_model(registry)
    selection = {name: "tilelang" if name in CONTRACTS else "torch" for name in MODEL_CONTRACTS}
    control_ops = registry.bind(selection)
    native = W4A8DecodeSm103(library)
    shadow = ShadowFp4(torch, native, control_ops.function("fp4_gemm"), args.variant, report["shadow"])
    registry.register("cuda-shadow-return-tilelang", "fp4_gemm", shadow, version=native.version,
                      contract=CONTRACTS["fp4_gemm"])
    selection["fp4_gemm"] = "cuda-shadow-return-tilelang"
    ops = registry.bind(selection)
    hadamard.configure_backend("cuda")
    extension = str(Path(hadamard.backend_report()["extension"]).resolve())
    before[extension] = sha256(extension)
    model = DeepSeekV4Model(cfg, ops, hadamard.hadamard_transform)
    report.update(stage="strict_weight_load", model_config=asdict(cfg), torch=torch.__version__,
                  cuda=torch.version.cuda, tilelang=tilelang.__version__, gpu=torch.cuda.get_device_name(),
                  global_default_dtype=str(torch.get_default_dtype()),
                  float32_matmul_precision=torch.get_float32_matmul_precision())
    save(args.output, report)
    report["weights"] = load_converted_weights(model, checkpoint)
    report["weight_load_count"] = 1
    shadow.names = {value.data_ptr(): name for name, value in model.named_parameters()}
    report["stage"] = "shadow_case_0"
    save(args.output, report)
    # Only phase metadata is attached to this new model instance. It does not
    # replace any model function or change the existing frozen probe.
    def phase(_module, inputs):
        report["shadow"]["phase"] = "prefill" if inputs[1].position == 0 else "decode"
    handle = model.register_forward_pre_hook(phase)
    try:
        result = runner.evaluate(model, tokenizer, case, 0, None)
        result.pop("_logits")
        report["control_case"] = result
        torch.cuda.synchronize()
        report["device_completion_synchronized"] = True
    finally:
        handle.remove()
        shadow.finish()
        report["operator_backends"] = ops.report()
        report["registered_tilelang_backends"] = control_ops.report()
        report["hadamard"] = hadamard.backend_report()
        report["source_sha256_after"] = {path: sha256(path) for path in before}
        report["source_unchanged"] = report["source_sha256_after"] == before
        verify_checkpoint_unchanged(checkpoint, identity)
        report["checkpoint_unchanged"] = True
    report["all_shadow_calls_exact"] = bool(report["shadow"]["calls"]) and all(
        row["exact_equal"] for row in report["shadow"]["calls"])
    report["control_numerical_gate_passed"] = report["control_case"]["numerical_gate_passed"]
    report["status"] = "completed_diagnostic"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors"))
    parser.add_argument("--library", type=Path, default=ROOT / "build/standalone_cuda/libw4a8_decode_sm103_final.so")
    parser.add_argument("--variant", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "stage": "setup", "command": sys.argv,
              "scope": "original published chat case 0, two steps, real activations, 43 layers",
              "model_result_backend": "registered TileLang for every FP4 GEMM",
              "error_fallback": False, "performance_claim": False,
              "shadow": {"calls": [], "phase": None}}
    started = time.perf_counter()
    try:
        run(args, report)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed_exception", failure={"type": type(exc).__name__, "message": str(exc),
                                                         "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save(args.output, report)
    print(json.dumps({"status": report["status"], "control_numerical_gate_passed": report.get("control_numerical_gate_passed"),
                      "all_shadow_calls_exact": report.get("all_shadow_calls_exact"), "report": str(args.output)}), flush=True)
    return 0 if report["status"] == "completed_diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())

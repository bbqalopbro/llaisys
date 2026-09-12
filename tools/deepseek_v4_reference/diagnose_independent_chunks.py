"""Trace early real-weight blocks across full and chunked execution.

This deliberately loads only a prefix of main layers. It never calls the full
model forward or declares end-to-end accuracy. Optional frozen submodule outputs
are a causal diagnostic intervention, not an inference backend or fallback.
"""

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from llaisys.models.deepseek_v4_backends import CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model
from llaisys.models.deepseek_v4_model import DeepSeekV4Model, InferenceConfig
from llaisys.models.deepseek_v4_model.weights import read_header, validate_header
from run_independent import load_kernel
import fast_hadamard_transform as hadamard


def token_rows(value):
    return value.flatten(0, 1) if value.ndim >= 3 and value.shape[0] == 1 else value


def metrics(expected, actual):
    if expected.shape != actual.shape:
        return {"shape_mismatch": [list(expected.shape), list(actual.shape)]}
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(b)
    if not torch.equal(finite, torch.isfinite(a)) or not torch.equal(a[~finite], b[~finite]):
        return {"nonfinite_values_mismatch": True, "exact_equal": False}
    a, b = torch.where(finite, a, 0), torch.where(finite, b, 0)
    error = a - b
    changed = error.reshape(error.shape[0], -1).ne(0).any(-1)
    positions = torch.where(changed)[0]
    return {"shape": list(b.shape), "dtype": str(expected.dtype), "exact_equal": torch.equal(actual, expected),
            "max_abs_error": error.abs().max().item(),
            "relative_l2": (error.norm() / b.norm().clamp_min(1e-8)).item(),
            "changed_tokens": positions.numel(), "first_changed_tokens": positions[:16].tolist(),
            "last_token_relative_l2": (error[-1].norm() / b[-1].norm().clamp_min(1e-8)).item()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=65)
    parser.add_argument("--freeze", action="append", default=[], help="exact submodule name to freeze to full outputs")
    parser.add_argument("--fixed-fp32-rows", type=int, default=0,
                        help="diagnostic intervention: pad FP32 linear calls to fixed M tiles")
    parser.add_argument("--model-linear-backend", choices=("torch", "torch-fixed32"), default="torch")
    parser.add_argument("--indexer-tie-policy", choices=("published", "index_ascending"), default="published")
    parser.add_argument("--attention-metadata-policy", choices=("published", "fixed"), default="published")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("real-weight CUDA diagnostics require an allocated Slurm GPU")
    cfg = InferenceConfig.from_directory(args.source_model, max_seq_len=args.max_seq_len)
    cfg = replace(cfg, indexer_tie_policy=args.indexer_tie_policy, attention_metadata_policy=args.attention_metadata_policy)
    if not 1 <= args.layers <= cfg.n_layers or args.chunk_size <= 0:
        raise ValueError("invalid layer count or chunk size")
    cpu_ids = load_file(str(args.tokens))["input_ids"]
    if cpu_ids.ndim != 2 or cpu_ids.shape[0] != 1 or not 0 < cpu_ids.shape[1] <= cfg.max_seq_len:
        raise ValueError("diagnostic tokens must fit one nonempty request in max-seq-len")
    import tilelang
    registry = OperatorRegistry(MODEL_CONTRACTS)
    kernel = load_kernel(args.source_model / "inference/kernel.py")
    register_tilelang(registry, kernel, tilelang.__version__)
    register_torch_model(registry, backend=args.model_linear_backend,
                         fixed_rows=32 if args.model_linear_backend == "torch-fixed32" else 0)
    selected_ops = {name: "tilelang" for name in CONTRACTS}
    selected_ops.update({name: args.model_linear_backend for name in set(MODEL_CONTRACTS)-set(CONTRACTS)})
    ops = registry.bind(selected_ops)
    hadamard.configure_backend("cuda")
    model = DeepSeekV4Model(cfg, ops, hadamard.hadamard_transform)
    header, digest = read_header(args.checkpoint)
    expected, _, _ = validate_header(model, header)
    selected = {key: value for key, value in expected.items() if key == "embed.weight" or
                (key.startswith("layers.") and int(key.split(".")[1]) < args.layers)}
    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        for key, target in selected.items():
            value = handle.get_tensor(key).to(device="cuda", dtype=target.dtype)
            prefix, _, name = key.rpartition(".")
            model.get_submodule(prefix)._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
    print(f"partial diagnostic load: {args.layers} layers, {len(selected)} tensors", flush=True)
    original_linear = torch.nn.functional.linear
    def fixed_linear(x, weight, bias=None):
        if x.dtype != torch.float32 or not args.fixed_fp32_rows:
            return original_linear(x, weight, bias)
        rows = x.reshape(-1, x.shape[-1])
        outputs = []
        for start in range(0, rows.shape[0], args.fixed_fp32_rows):
            part = rows[start:start+args.fixed_fp32_rows]
            padded = torch.nn.functional.pad(part, (0, 0, 0, args.fixed_fp32_rows-part.shape[0]))
            outputs.append(original_linear(padded, weight, bias)[:part.shape[0]])
        return torch.cat(outputs, dim=0).reshape(*x.shape[:-1], weight.shape[0])
    if args.fixed_fp32_rows:
        torch.nn.functional.linear = fixed_linear
    ids = cpu_ids.cuda()
    phases = ["full", "chunk"] + (["chunk_frozen"] if args.freeze else [])
    active = {"phase": "full", "position": 0}
    captures = {phase: defaultdict(list) for phase in phases}
    originals = {}
    hooks = []
    observable = {}
    for name, module in model.named_modules():
        if (not name.startswith("layers.") or int(name.split(".")[1]) >= args.layers
                or ".experts." in name or name.endswith(".indexer")):
            continue
        if not name or isinstance(module, torch.nn.ModuleList):
            continue
        observable[name] = module
    if set(args.freeze) - set(observable):
        raise ValueError(f"unobservable frozen modules: {set(args.freeze) - set(observable)}")

    def hook(name):
        def capture(module, inputs, output):
            phase, position = active["phase"], active["position"]
            if isinstance(inputs[0], torch.Tensor):
                captures[phase][name + ".input"].append(token_rows(inputs[0]).detach().cpu().clone())
            if isinstance(output, torch.Tensor):
                if phase == "full":
                    originals[name] = token_rows(output).detach().cpu().clone()
                elif phase == "chunk_frozen" and name in args.freeze:
                    rows = token_rows(output).shape[0]
                    output = originals[name][position:position+rows].to(output.device).reshape(output.shape)
                captures[phase][name + ".output"].append(token_rows(output).detach().cpu().clone())
            elif isinstance(output, tuple):
                for index, value in enumerate(output):
                    if isinstance(value, torch.Tensor):
                        captures[phase][name + f".output_{index}"].append(token_rows(value).detach().cpu().clone())
            return output
        return capture
    for name, module in observable.items():
        hooks.append(module.register_forward_hook(hook(name)))
    original_topk = ops.function("indexer_topk")
    def trace_topk(scores, k, policy):
        indices = original_topk(scores, k, policy)
        phase, position = active["phase"], active["position"]
        name = f"layers.{active['layer']}.attn.indexer.topk"
        width = ids.shape[1] // 4
        captures[phase][name + ".scores"].append(token_rows(
            torch.nn.functional.pad(scores, (0, width-scores.shape[-1]), value=-torch.inf)).cpu().clone())
        counts = torch.arange(position+1, position+scores.shape[1]+1, device=scores.device)[None, :, None] // 4
        visible = torch.where(indices < counts, indices, -1)
        captures[phase][name + ".indices"].append(token_rows(
            torch.nn.functional.pad(visible, (0, min(cfg.index_topk, width)-k), value=-1)).cpu().clone())
        return indices
    ops._functions["indexer_topk"] = trace_topk
    try:
        for phase in phases:
            active["phase"] = phase
            state = model.new_request()
            size = ids.shape[1] if phase == "full" else args.chunk_size
            for position in range(0, ids.shape[1], size):
                active["position"] = position
                tokens = ids[:, position:position+size]
                hidden = model.embed(tokens).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
                for index in range(args.layers):
                    active["layer"] = index
                    hidden = model.layers[index](hidden, position, state.layers[index], tokens)
            state.close()
            print(f"completed {phase}", flush=True)
    finally:
        torch.nn.functional.linear = original_linear
        ops._functions["indexer_topk"] = original_topk
        for handle in hooks:
            handle.remove()
    merged = {phase: {name: torch.cat(values, dim=0) for name, values in items.items()}
              for phase, items in captures.items()}
    results = {}
    for phase in phases[1:]:
        results[phase] = {name: metrics(expected, merged[phase][name])
                          for name, expected in merged["full"].items()}
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for phase, tensors in merged.items():
            save_file(tensors, str(args.dump_dir / f"{phase}.safetensors"))
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), args.source_model / "inference/kernel.py", args.source_model / "config.json",
             args.tokens, root / "python/llaisys/models/deepseek_v4_backends.py",
             root / "python/llaisys/models/deepseek_v4_model_ops.py",
             root / "python/llaisys/models/deepseek_v4_reference.py",
             *sorted((root / "python/llaisys/models/deepseek_v4_model").glob("*.py"))]
    result = {"scope": "partial_real_weight_layer_diagnostic_not_full_model_accuracy",
              "layers": args.layers, "loaded_tensors": len(selected), "header_sha256": digest,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "fp32_matmul_precision": torch.get_float32_matmul_precision(),
              "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "tokens": ids.shape[1], "chunk_size": args.chunk_size, "frozen_modules": args.freeze,
              "max_seq_len": cfg.max_seq_len, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "diagnostic_fixed_fp32_rows": args.fixed_fp32_rows,
              "indexer_tie_policy": cfg.indexer_tie_policy,
              "attention_metadata_policy": cfg.attention_metadata_policy,
              "command": sys.argv, "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "operator_backends": ops.report(), "comparisons": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for phase, rows in results.items():
        for name, comparison in rows.items():
            if not comparison.get("exact_equal", False):
                print(phase, name, json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()

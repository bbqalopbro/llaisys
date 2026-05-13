#!/usr/bin/env python3
"""Calibration-aware INT4 quantization for the native LLAISYS packed format.

This is an AWQ-like practical baseline, not a full AWQ implementation. It uses
calibration activations to compute per-input-channel importance, then searches a
weighted per-group INT4 scale that minimizes activation-weighted reconstruction
error. The output format is the same as scripts/quantize.py:

  weight:       uint8 [out_features, in_features / 2]
  weight.scale: float32 [out_features, in_features / group_size]

Because the runtime formula remains y = x @ dequant(W)^T, no activation
rescaling or graph surgery is needed.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import safetensors
import safetensors.torch
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


QUANTIZE_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "lm_head.weight",
]

DEFAULT_KEEP_FP16_SUFFIXES = [
    "lm_head.weight",
]

DEFAULT_CALIB_PROMPTS = [
    "What is the meaning of life?",
    "Explain paged attention in one paragraph.",
    "Write a short Python function to compute Fibonacci numbers.",
    "Summarize the difference between FP16 and INT4 inference.",
    "请用中文解释为什么大模型推理需要 KV cache。",
    "给我三个学习 CUDA kernel 优化的建议。",
    "Translate to English: 量化可以降低显存占用，但可能损失精度。",
    "Solve step by step: if a GPU generates 98 tokens/s, how long for 512 tokens?",
]


def should_quantize(name: str, keep_fp16_suffixes: Iterable[str]) -> bool:
    if any(name.endswith(s) for s in keep_fp16_suffixes):
        return False
    return any(name.endswith(s) for s in QUANTIZE_SUFFIXES)


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.exists():
        return path
    return Path(snapshot_download(model_path))


def read_prompts(path: str | None) -> List[str]:
    if path is None:
        return DEFAULT_CALIB_PROMPTS
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                prompts.append(obj.get("prompt") or obj.get("text") or obj["input"])
            else:
                prompts.append(line)
    return prompts


def module_name_to_weight_name(module_name: str) -> str:
    return module_name + ".weight"


def collect_activation_importance(model_path: Path, prompts: List[str],
                                  max_length: int, device: str) -> Dict[str, torch.Tensor]:
    print(f"==> Collecting calibration activations ({len(prompts)} prompts, max_length={max_length})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []

    def make_hook(weight_name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            if x.numel() == 0:
                return
            x = x.reshape(-1, x.shape[-1]).float().abs()
            stat = x.mean(dim=0).cpu()
            if weight_name not in sums:
                sums[weight_name] = stat
                counts[weight_name] = 1
            else:
                sums[weight_name] += stat
                counts[weight_name] += 1
        return hook

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_name = module_name_to_weight_name(module_name)
            if any(weight_name.endswith(s) for s in QUANTIZE_SUFFIXES):
                handles.append(module.register_forward_hook(make_hook(weight_name)))

    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                text = prompt
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            print(f"  calib {i + 1}/{len(prompts)}: {len(inputs['input_ids'][0])} tokens")

    for h in handles:
        h.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    importance = {}
    for name, value in sums.items():
        imp = value / max(counts[name], 1)
        imp = imp.clamp(min=1e-6)
        imp = imp / imp.mean().clamp(min=1e-6)
        importance[name] = imp.float()
        print(f"  importance {name}: channels={imp.numel()}")
    return importance


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    q = q.reshape(q.shape[0], -1)
    even = (q[:, 0::2] + 8).to(torch.uint8)
    odd = (q[:, 1::2] + 8).to(torch.uint8)
    return (odd << 4) | (even & 0x0F)


def quantize_per_group_weighted_int4(weight: torch.Tensor, importance: torch.Tensor | None,
                                     group_size: int, search_steps: int,
                                     row_chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.float().cpu()
    rows, cols = w.shape
    assert cols % group_size == 0, f"in_features {cols} cannot divide group_size {group_size}"
    assert group_size % 2 == 0, "group_size must be even"
    num_groups = cols // group_size

    if importance is None:
        imp = torch.ones(cols, dtype=torch.float32)
    else:
        imp = importance.float().cpu()
        if imp.numel() != cols:
            print(f"  WARNING: importance shape mismatch ({imp.numel()} vs {cols}), using uniform")
            imp = torch.ones(cols, dtype=torch.float32)
    imp_g = imp.reshape(1, num_groups, group_size)

    candidates = torch.linspace(1.0, 0.5, search_steps, dtype=torch.float32)
    packed_chunks = []
    scale_chunks = []

    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        wg = w[start:end].reshape(end - start, num_groups, group_size)
        base = (wg.abs().amax(dim=-1) / 7.0).clamp(min=1e-10)

        best_err = None
        best_q = None
        best_scale = None
        for factor in candidates:
            scale = (base * factor).clamp(min=1e-10)
            q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7)
            recon = q * scale.unsqueeze(-1)
            err = ((wg - recon).pow(2) * imp_g).sum(dim=-1)
            if best_err is None:
                best_err = err
                best_q = q.to(torch.int8)
                best_scale = scale
            else:
                mask = err < best_err
                best_err = torch.where(mask, err, best_err)
                best_q = torch.where(mask.unsqueeze(-1), q.to(torch.int8), best_q)
                best_scale = torch.where(mask, scale, best_scale)

        packed_chunks.append(pack_int4(best_q.reshape(end - start, cols)))
        scale_chunks.append(best_scale.float())
        print(f"    rows {start}:{end}")

    return torch.cat(packed_chunks, dim=0), torch.cat(scale_chunks, dim=0)


def quantize_model_awq_like(model_path: str, output_dir: str, group_size: int,
                            prompts_path: str | None, max_length: int,
                            device: str, keep_fp16_suffixes: List[str],
                            search_steps: int, row_chunk: int) -> None:
    model_path = resolve_model_path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(prompts_path)
    importance = collect_activation_importance(model_path, prompts, max_length, device)

    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print(f"Error: no safetensors found in {model_path}", file=sys.stderr)
        sys.exit(1)

    quantized_tensors = {}
    stats = {
        "quantized": 0,
        "kept": 0,
        "total_bytes_original": 0,
        "total_bytes_output": 0,
        "keep_fp16_suffixes": keep_fp16_suffixes,
        "search_steps": search_steps,
    }

    for file in st_files:
        print(f"\nProcessing: {file.name}")
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                orig_bytes = tensor.numel() * tensor.element_size()
                if should_quantize(name, keep_fp16_suffixes):
                    print(f"  [AWQ-LIKE INT4] {name}: {list(tensor.shape)}")
                    packed, scale = quantize_per_group_weighted_int4(
                        tensor, importance.get(name), group_size, search_steps, row_chunk
                    )
                    quantized_tensors[name] = packed
                    quantized_tensors[name + ".scale"] = scale
                    out_bytes = packed.numel() + scale.numel() * 4
                    stats["quantized"] += 1
                    stats["total_bytes_original"] += orig_bytes
                    stats["total_bytes_output"] += out_bytes
                else:
                    if tensor.dtype != torch.float16:
                        tensor = tensor.to(torch.float16 if tensor.is_floating_point() else tensor.dtype)
                    quantized_tensors[name] = tensor
                    stats["kept"] += 1
                    stats["total_bytes_original"] += orig_bytes
                    stats["total_bytes_output"] += tensor.numel() * tensor.element_size()
                    print(f"  [KEEP] {name}: {list(tensor.shape)} {tensor.dtype}")

    output_file = output_dir / "model_int4.safetensors"
    print(f"\nSaving: {output_file}")
    safetensors.torch.save_file(quantized_tensors, str(output_file))

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    quant_config = {
        "quant_method": f"awq_like_weighted_symmetric_int4_g{group_size}",
        "bits": 4,
        "group_size": group_size,
        "quantized_suffixes": [s for s in QUANTIZE_SUFFIXES if s not in keep_fp16_suffixes],
        "keep_fp16_suffixes": keep_fp16_suffixes,
        "calibration_prompts": len(prompts),
        "calibration_max_length": max_length,
        "stats": stats,
    }
    with open(output_dir / "quant_config.json", "w", encoding="utf-8") as f:
        json.dump(quant_config, f, ensure_ascii=False, indent=2)

    ratio = stats["total_bytes_original"] / max(stats["total_bytes_output"], 1)
    print("\nQuantization complete")
    print(f"  quantized tensors: {stats['quantized']}")
    print(f"  kept tensors:      {stats['kept']}")
    print(f"  output:            {output_dir}")
    print(f"  compression ratio: {ratio:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ-like weighted INT4 quantizer for LLAISYS")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--prompts", default=None, help="Calibration text/jsonl file")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-fp16-suffix", action="append", default=[],
                        help="Weight suffix to keep in FP16; repeatable")
    parser.add_argument("--quantize-lm-head", action="store_true",
                        help="Also quantize lm_head.weight")
    parser.add_argument("--search-steps", type=int, default=11)
    parser.add_argument("--row-chunk", type=int, default=128)
    args = parser.parse_args()

    keep = list(args.keep_fp16_suffix)
    if not args.quantize_lm_head:
        keep.extend(DEFAULT_KEEP_FP16_SUFFIXES)
    keep = sorted(set(keep))

    quantize_model_awq_like(
        args.model,
        args.output,
        args.group_size,
        args.prompts,
        args.max_length,
        args.device,
        keep,
        args.search_steps,
        args.row_chunk,
    )


if __name__ == "__main__":
    main()

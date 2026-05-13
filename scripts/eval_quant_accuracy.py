#!/usr/bin/env python3
"""Evaluate quantized LLAISYS generation against an FP16 LLAISYS baseline.

This script intentionally uses greedy decoding by default. Token-by-token
agreement is harsh for autoregressive generation, so the report includes both
prefix agreement and edit similarity instead of treating a single mismatch as a
complete failure.
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import llaisys  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


DEFAULT_PROMPTS = [
    "What is the meaning of life?",
    "Explain paged attention in one paragraph.",
    "Write a short Python function to compute Fibonacci numbers.",
    "Summarize the difference between FP16 and INT4 inference.",
    "请用中文解释为什么大模型推理需要 KV cache。",
    "给我三个学习 CUDA kernel 优化的建议。",
    "Translate to English: 量化可以降低显存占用，但可能损失精度。",
    "Solve step by step: if a GPU generates 98 tokens/s, how long for 512 tokens?",
]


@dataclass
class CaseResult:
    index: int
    prompt: str
    prompt_tokens: int
    baseline_new_tokens: int
    quant_new_tokens: int
    prefix_match: int
    positional_match: int
    compared_tokens: int
    positional_match_rate: float
    edit_similarity: float
    exact_match: bool
    first_divergence: int
    baseline_tok_s: float
    quant_tok_s: float
    baseline_text: str
    quant_text: str


def read_prompts(path: str | None) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS
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


def llaisys_device(name: str):
    if name == "cpu":
        return llaisys.DeviceType.CPU
    if name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    if name == "metax":
        return llaisys.DeviceType.METAX
    raise ValueError(f"unsupported device: {name}")


def clear_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def render_prompt(tokenizer, prompt: str) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        text = prompt
    return tokenizer.encode(text)


def generate_all(model_path: str, device: str, tokenizer, prompts: Iterable[str],
                 max_new_tokens: int, top_k: int, top_p: float,
                 temperature: float) -> List[dict]:
    model = llaisys.models.Qwen2(model_path, llaisys_device(device))
    rows = []
    for prompt in prompts:
        input_ids = render_prompt(tokenizer, prompt)
        t0 = time.perf_counter()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - t0
        new_ids = output_ids[len(input_ids):]
        rows.append({
            "prompt": prompt,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "new_ids": new_ids,
            "elapsed": elapsed,
            "tok_s": (len(new_ids) / elapsed) if elapsed > 0 else 0.0,
            "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        })
    del model
    clear_gpu()
    return rows


def common_prefix_len(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def levenshtein(a: List[int], b: List[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if x == y else 1),
            ))
        prev = cur
    return prev[-1]


def compare_rows(tokenizer, baseline_rows: List[dict], quant_rows: List[dict]) -> List[CaseResult]:
    results = []
    for idx, (base, quant) in enumerate(zip(baseline_rows, quant_rows)):
        b = base["new_ids"]
        q = quant["new_ids"]
        compared = min(len(b), len(q))
        prefix = common_prefix_len(b, q)
        positional = sum(1 for x, y in zip(b, q) if x == y)
        denom = max(len(b), len(q), 1)
        edit_dist = levenshtein(b, q)
        similarity = 1.0 - (edit_dist / denom)
        first_div = prefix if prefix < compared or len(b) != len(q) else -1
        results.append(CaseResult(
            index=idx,
            prompt=base["prompt"],
            prompt_tokens=len(base["input_ids"]),
            baseline_new_tokens=len(b),
            quant_new_tokens=len(q),
            prefix_match=prefix,
            positional_match=positional,
            compared_tokens=compared,
            positional_match_rate=(positional / compared) if compared else 0.0,
            edit_similarity=similarity,
            exact_match=(b == q),
            first_divergence=first_div,
            baseline_tok_s=base["tok_s"],
            quant_tok_s=quant["tok_s"],
            baseline_text=base["text"],
            quant_text=quant["text"],
        ))
    return results


def print_report(results: List[CaseResult]) -> None:
    n = len(results)
    exact = sum(r.exact_match for r in results)
    avg_prefix = sum(r.prefix_match for r in results) / max(n, 1)
    avg_pos = sum(r.positional_match_rate for r in results) / max(n, 1)
    avg_edit = sum(r.edit_similarity for r in results) / max(n, 1)
    avg_base_tps = sum(r.baseline_tok_s for r in results) / max(n, 1)
    avg_quant_tps = sum(r.quant_tok_s for r in results) / max(n, 1)

    print("\n=== Quantization Accuracy Summary ===")
    print(f"cases:                 {n}")
    print(f"exact_match_cases:     {exact}/{n} ({exact / max(n, 1):.1%})")
    print(f"avg_prefix_match:      {avg_prefix:.2f} tokens")
    print(f"avg_position_match:    {avg_pos:.1%}")
    print(f"avg_edit_similarity:   {avg_edit:.1%}")
    print(f"baseline_avg_tok_s:    {avg_base_tps:.2f}")
    print(f"quant_avg_tok_s:       {avg_quant_tps:.2f}")
    if avg_base_tps > 0:
        print(f"quant_speedup:         {avg_quant_tps / avg_base_tps:.2f}x")

    print("\n=== Per Case ===")
    header = (
        "idx  prompt_tok  new  prefix  pos_match  edit_sim  "
        "first_div  base_t/s  quant_t/s  prompt"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.index:<4} {r.prompt_tokens:<11} "
            f"{r.baseline_new_tokens}/{r.quant_new_tokens:<4} "
            f"{r.prefix_match:<7} "
            f"{r.positional_match}/{r.compared_tokens:<8} "
            f"{r.edit_similarity:<8.1%} "
            f"{r.first_divergence:<10} "
            f"{r.baseline_tok_s:<9.2f} "
            f"{r.quant_tok_s:<10.2f} "
            f"{r.prompt[:70]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLAISYS quantized generation accuracy.")
    parser.add_argument("--baseline-model", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--quant-model", default="quantized_model_int4")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path; defaults to baseline model.")
    parser.add_argument("--prompts", default=None, help="Text or JSONL prompt file.")
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "metax"])
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    prompts = read_prompts(args.prompts)
    tokenizer_path = args.tokenizer or args.baseline_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"baseline: {args.baseline_model}")
    print(f"quant:    {args.quant_model}")
    print(f"device:   {args.device}")
    print(f"prompts:  {len(prompts)}")
    print(f"decode:   greedy={args.top_k == 1 or args.temperature <= 0}, max_new={args.max_new_tokens}")

    baseline_rows = generate_all(
        args.baseline_model, args.device, tokenizer, prompts,
        args.max_new_tokens, args.top_k, args.top_p, args.temperature,
    )
    quant_rows = generate_all(
        args.quant_model, args.device, tokenizer, prompts,
        args.max_new_tokens, args.top_k, args.top_p, args.temperature,
    )

    results = compare_rows(tokenizer, baseline_rows, quant_rows)
    print_report(results)

    if args.json_out:
        payload = {
            "baseline_model": args.baseline_model,
            "quant_model": args.quant_model,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "results": [asdict(r) for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nwrote: {args.json_out}")


if __name__ == "__main__":
    main()

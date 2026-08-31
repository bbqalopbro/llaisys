#!/usr/bin/env python3
"""End-to-end serving benchmark for the Qwen2 BatchContext path.

Measures model-warm TTFT and decode TPOT separately, and validates that full,
chunked, and block-prefix-cache prefills produce the same greedy first token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import llaisys
from test_utils import llaisys_device
from transformers import AutoTokenizer
from server.engine import InferenceEngine, SamplingParams


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def synthetic_prompt(length: int) -> list[int]:
    return [(index % 1000) + 1 for index in range(length)]


def timed_full_prefill(ctx, tokens: list[int]) -> tuple[int, float]:
    ctx.slot_reset(0)
    start = time.perf_counter()
    token = ctx.prefill(0, tokens, temperature=0.0, top_k=1, top_p=1.0)
    return int(token), (time.perf_counter() - start) * 1000.0


def timed_chunked_prefill(ctx, tokens: list[int], chunk_size: int) -> tuple[int, float]:
    ctx.slot_reset(0)
    start = time.perf_counter()
    output = None
    cursor = 0
    while cursor < len(tokens):
        end = min(cursor + chunk_size, len(tokens))
        output = ctx.prefill_chunk(
            0, tokens[cursor:end], start_pos=cursor,
            is_last_chunk=end == len(tokens),
            temperature=0.0, top_k=1, top_p=1.0,
        )
        cursor = end
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if output is None:
        raise RuntimeError("chunked prefill did not produce an output token")
    return int(output), elapsed_ms


def timed_prefix_hit(ctx, tokens: list[int]) -> tuple[int, int, float]:
    ctx.slot_reset(0)
    start = time.perf_counter()
    matched = ctx.prefix_lookup(0, tokens)
    output = None
    cursor = matched
    while cursor < len(tokens):
        end = len(tokens)
        output = ctx.prefill_chunk(
            0, tokens[cursor:end], start_pos=cursor, is_last_chunk=True,
            temperature=0.0, top_k=1, top_p=1.0,
        )
        cursor = end
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if output is None:
        raise RuntimeError("prefix-hit prefill did not produce an output token")
    return int(output), int(matched), elapsed_ms


def timed_decode(ctx, first_token: int, count: int) -> tuple[list[int], list[float]]:
    current = first_token
    outputs: list[int] = []
    latencies_ms: list[float] = []
    for _ in range(count):
        start = time.perf_counter()
        result = ctx.decode_per_request(
            [0], [current], [0.0], [1], [1.0]
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        current = int(result[0])
        outputs.append(current)
    return outputs, latencies_ms


async def timed_scheduler_request(model, prompt_ids: list[int], count: int) -> dict:
    engine = InferenceEngine(
        model=model, max_batch_size=1, max_seq_per_slot=2048,
        prefill_chunk_size=256, enable_block_prefix_cache=True,
    )
    engine.start()
    timestamps: list[float] = []
    tokens: list[int] = []
    start = time.perf_counter()
    try:
        request = engine.submit(
            input_ids=prompt_ids,
            params=SamplingParams(
                temperature=0.0, top_k=1, top_p=1.0, max_tokens=count
            ),
            session_id="e2e-benchmark", stream=True,
            loop=asyncio.get_running_loop(),
        )
        async for token in request.stream_tokens():
            timestamps.append(time.perf_counter())
            tokens.append(int(token))
        await request.future
    finally:
        engine.stop()
    intervals = [
        (right - left) * 1000.0
        for left, right in zip(timestamps, timestamps[1:])
    ]
    return {
        "generated_tokens": tokens,
        "ttft_ms": (timestamps[0] - start) * 1000.0,
        "tpot_mean_ms": statistics.mean(intervals) if intervals else 0.0,
        "tpot_p95_ms": percentile(intervals, 0.95),
        "end_to_end_ms": (timestamps[-1] - start) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] /
                    "models/DeepSeek-R1-Distill-Qwen-1.5B"),
    )
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"])
    parser.add_argument("--lengths", default="32,256")
    parser.add_argument("--decode-tokens", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--prefix-length", type=int, default=257)
    parser.add_argument("--correctness-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="What is 2+3?")
    parser.add_argument("--scheduler-long", action="store_true")
    parser.add_argument(
        "--paged-prefill", choices=("auto", "require", "fallback"),
        default="auto",
        help=("select direct paged prefill automatically, require it (fail if "
              "unavailable), or force the contiguous gather fallback"),
    )
    parser.add_argument("--json-output")
    args = parser.parse_args()

    if args.paged_prefill == "require":
        os.environ.pop("LLAISYS_DISABLE_FLASHINFER_PREFILL", None)
        os.environ["LLAISYS_REQUIRE_PAGED_PREFILL"] = "1"
    elif args.paged_prefill == "fallback":
        os.environ.pop("LLAISYS_REQUIRE_PAGED_PREFILL", None)
        os.environ["LLAISYS_DISABLE_FLASHINFER_PREFILL"] = "1"

    lengths = [int(item) for item in args.lengths.split(",") if item]
    load_start = time.perf_counter()
    model = llaisys.models.Qwen2(args.model, llaisys_device(args.device))
    model_load_s = time.perf_counter() - load_start
    ctx = model.create_batch_context(max_batch_size=1, max_seq_per_slot=2048)
    binding = "pybind11" if getattr(ctx, "_native", None) is not None else "ctypes"

    report: dict = {
        "model": os.path.abspath(args.model),
        "device": args.device,
        "binding": binding,
        "paged_prefill_request": args.paged_prefill,
        "model_load_s": model_load_s,
        "full_prefill": [],
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer.encode(prompt_text)
    # Exercise BatchContext first so this is a genuine post-load cold TTFT.
    batch_first, batch_cold_ttft = timed_full_prefill(ctx, prompt_ids)
    batch_tail, _ = timed_decode(ctx, batch_first, args.correctness_tokens - 1)
    batch_output = [batch_first, *batch_tail]
    reference = model.generate(
        prompt_ids, max_new_tokens=args.correctness_tokens,
        temperature=0.0, top_k=1, top_p=1.0,
    )[-args.correctness_tokens:]
    if batch_output != reference:
        raise AssertionError(
            f"BatchContext differs from single-model greedy output: "
            f"reference={reference}, batch={batch_output}"
        )
    report["correctness"] = {
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": batch_output,
        "decoded_text": tokenizer.decode(batch_output, skip_special_tokens=True),
        "matches_single_model": True,
        "batch_cold_ttft_ms": batch_cold_ttft,
    }

    for length in lengths:
        tokens = synthetic_prompt(length)
        runs = []
        first_tokens = []
        decode_samples: list[float] = []
        decode_steady_samples: list[float] = []
        for _ in range(args.repeat):
            first_token, ttft_ms = timed_full_prefill(ctx, tokens)
            _, decode_ms = timed_decode(ctx, first_token, args.decode_tokens)
            runs.append(ttft_ms)
            first_tokens.append(first_token)
            decode_samples.extend(decode_ms)
            decode_steady_samples.extend(decode_ms[2:] or decode_ms)
        if len(set(first_tokens)) != 1:
            raise AssertionError(f"non-deterministic greedy first token: {first_tokens}")
        report["full_prefill"].append({
            "prompt_tokens": length,
            "first_token": first_tokens[0],
            "ttft_first_ms": runs[0],
            "ttft_warm_mean_ms": statistics.mean(runs[1:] or runs),
            "ttft_warm_min_ms": min(runs[1:] or runs),
            "tpot_mean_ms": statistics.mean(decode_samples),
            "tpot_p50_ms": percentile(decode_samples, 0.50),
            "tpot_p95_ms": percentile(decode_samples, 0.95),
            "tpot_steady_mean_ms": statistics.mean(decode_steady_samples),
            "tpot_steady_p95_ms": percentile(decode_steady_samples, 0.95),
            "decode_tokens_per_s": 1000.0 / statistics.mean(decode_samples),
            "decode_steady_tokens_per_s": 1000.0 / statistics.mean(decode_steady_samples),
        })

    prefix_tokens = synthetic_prompt(args.prefix_length)
    baseline_token, baseline_ttft = timed_full_prefill(ctx, prefix_tokens)
    if not ctx.prefix_publish(0, prefix_tokens):
        raise RuntimeError("failed to publish prefix blocks")
    hit_token, matched_tokens, hit_ttft = timed_prefix_hit(ctx, prefix_tokens)
    if hit_token != baseline_token:
        raise AssertionError(
            f"prefix hit changed greedy token: full={baseline_token}, hit={hit_token}"
        )
    report["prefix_cache"] = {
        "prompt_tokens": len(prefix_tokens),
        "matched_tokens": matched_tokens,
        "full_ttft_ms": baseline_ttft,
        "hit_ttft_ms": hit_ttft,
        "speedup": baseline_ttft / hit_ttft if hit_ttft else 0.0,
        "first_token": hit_token,
    }

    chunk_tokens = synthetic_prompt(max(lengths))
    full_token, full_ttft = timed_full_prefill(ctx, chunk_tokens)
    long_tail, _ = timed_decode(ctx, full_token, args.correctness_tokens - 1)
    long_reference = [full_token, *long_tail]
    chunk_token, chunk_ttft = timed_chunked_prefill(ctx, chunk_tokens, args.chunk_size)
    if chunk_token != full_token:
        raise AssertionError(
            f"chunked prefill changed greedy token: full={full_token}, chunk={chunk_token}"
        )
    report["chunked_prefill"] = {
        "prompt_tokens": len(chunk_tokens),
        "chunk_size": args.chunk_size,
        "full_ttft_ms": full_ttft,
        "chunked_ttft_ms": chunk_ttft,
        "slowdown": chunk_ttft / full_ttft if full_ttft else 0.0,
        "first_token": chunk_token,
    }

    scheduler_result = asyncio.run(
        timed_scheduler_request(model, prompt_ids, args.correctness_tokens)
    )
    if scheduler_result["generated_tokens"] != reference:
        raise AssertionError(
            "InferenceEngine output differs from the greedy reference: "
            f"reference={reference}, engine={scheduler_result['generated_tokens']}"
        )
    scheduler_result["matches_reference"] = True
    report["scheduler_e2e"] = scheduler_result
    if args.scheduler_long:
        scheduler_long_result = asyncio.run(
            timed_scheduler_request(
                model, chunk_tokens, args.correctness_tokens
            )
        )
        if scheduler_long_result["generated_tokens"] != long_reference:
            raise AssertionError(
                "Long-prompt InferenceEngine output differs from full prefill: "
                f"reference={long_reference}, "
                f"engine={scheduler_long_result['generated_tokens']}"
            )
        scheduler_long_result["prompt_tokens"] = len(chunk_tokens)
        scheduler_long_result["chunk_size"] = args.chunk_size
        scheduler_long_result["matches_full_prefill"] = True
        report["scheduler_long_prompt"] = scheduler_long_result

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

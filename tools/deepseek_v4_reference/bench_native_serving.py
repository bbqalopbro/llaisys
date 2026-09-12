"""Warm, fixed-input native InferenceEngine latency/throughput benchmark.

No golden/observer/logits D2H. Includes Python queueing and streaming delivery,
excludes model load and tokenization. EOS is honored; actual lengths are saved.
Memory is device-wide nvidia-smi sampling (includes native allocations), not an
exact allocator peak, and is reported separately with its sampling resolution.
"""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import torch
from transformers import AutoTokenizer
from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
from llaisys.models.deepseek_v4_model.chat import ChatCodec
from llaisys.models.deepseek_v4_model.native_session import load_native_session, _digest
from llaisys.models.deepseek_v4_model.native_batch import DeepSeekV4NativeServingModel
from server.engine import InferenceEngine, SamplingParams


def summarize(values):
    ordered = sorted(values)
    if not ordered:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, min=None, max=None)
    def percentile(q):
        pos = (len(ordered) - 1) * q; low = math.floor(pos); high = math.ceil(pos)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
    return dict(count=len(values), mean=statistics.fmean(values), p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99), min=ordered[0], max=ordered[-1])


class MemorySampler:
    def __init__(self, uuid):
        self.samples, self.errors = [], []
        self.stopping = False
        self.ready = threading.Event()
        self.process = subprocess.Popen(["nvidia-smi", "-i", uuid, "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True); self.thread.start()
        if not self.ready.wait(15):
            self.close(); raise RuntimeError("GPU memory sampler did not become ready")
        if self.errors or not self.samples:
            self.close(); raise RuntimeError("GPU memory sampler failed")

    def _read(self):
        try:
            for line in self.process.stdout:
                used, total = [int(v.strip()) for v in line.split(",")]
                self.samples.append((time.perf_counter(), used * 1024**2, total * 1024**2))
                self.ready.set()
        except Exception as error:
            self.errors.append(str(error))
        finally:
            if not self.stopping:
                self.errors.append("GPU memory sampler ended before explicit shutdown")
            self.ready.set()

    def report(self, start, end):
        values = [s for s in self.samples if start <= s[0] <= end]
        timestamps = [start] + [s[0] for s in values] + [end]
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        max_gap = max(gaps)
        return dict(method="nvidia-smi device-wide sampled memory.used", interval_ms=100, sample_count=len(values),
            peak_used_bytes=max((v[1] for v in values), default=None),
            total_bytes=values[-1][2] if values else None, includes_native_allocations=True,
            exact_allocator_peak=False, includes_other_gpu_processes=True,
            first_sample_delay_ms=(values[0][0] - start) * 1000 if values else None,
            last_sample_age_ms=(end - values[-1][0]) * 1000 if values else None,
            maximum_gap_ms=max_gap * 1000, coverage_limit_ms=500,
            coverage_valid=bool(values) and max_gap <= .5 and not self.errors and self.thread.is_alive()
                           and self.process.poll() is None)

    def close(self):
        self.stopping = True
        if self.process.poll() is None: self.process.terminate()
        try: self.process.wait(5)
        except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(5)
        self.thread.join(5)
        self.process.stdout.close(); self.process.stderr.close()


def prompts(tokenizer, lengths, concurrency):
    # Deliberately synthetic raw completions, not task-quality/chat evaluation.
    result = {}
    for size in lengths:
        rows = []
        for i in range(concurrency):
            prefix = tokenizer.encode(f"Document {i}: Continue the following numbered explanation in detail. ", add_special_tokens=False)
            body = tokenizer.encode("An inference engine loads weights, schedules requests, and executes attention and experts. ", add_special_tokens=False)
            suffix = tokenizer.encode("\nDetailed explanation:\n1.", add_special_tokens=False)
            if size < len(suffix) + 2: raise ValueError("prompt length too short for benchmark template")
            content = (prefix + body * (size // len(body) + 2))[:size - 1 - len(suffix)]
            rows.append([tokenizer.bos_token_id] + content + suffix)
        result[size] = rows
    return result


async def cohort(engine, inputs, output_budget):
    starts, requests = [], []
    for i, ids in enumerate(inputs):
        starts.append(time.perf_counter())
        requests.append(engine.submit(ids, SamplingParams(temperature=0, top_k=1, top_p=1, max_tokens=output_budget),
                                      f"bench-{i}", stream=True))
    async def consume(i, request):
        times, ids = [], []
        async for token in request.stream_tokens():
            times.append(time.perf_counter()); ids.append(token)
        result = await request.future
        completed = time.perf_counter()
        if (not ids or ids != result or len(ids) > output_budget
                or (len(ids) < output_budget and ids[-1] != engine._eos_token_id)):
            raise RuntimeError("stream/future/output length results disagree")
        intervals = [1000 * (b - a) for a, b in zip(times, times[1:])]
        return dict(input_tokens=len(inputs[i]), output_tokens=len(ids), generated_ids=ids,
            finish_reason="stop" if ids[-1] == engine._eos_token_id else "length",
            ttft_ms=(times[0] - starts[i]) * 1000, tpot_ms=statistics.fmean(intervals) if intervals else None,
            inter_token_ms=intervals, latency_to_last_token_ms=(times[-1] - starts[i]) * 1000,
            completion_latency_ms=(completed - starts[i]) * 1000,
            effective_prompt_tokens_per_second=len(inputs[i]) / (times[0] - starts[i]),
            decode_tokens_per_second=(len(ids) - 1) / (times[-1] - times[0]) if len(ids) > 1 else None)
    rows = await asyncio.wait_for(asyncio.gather(*(consume(i, r) for i, r in enumerate(requests))), 900)
    elapsed = time.perf_counter() - min(starts)
    return dict(requests=rows, cohort_seconds=elapsed, output_tokens_per_second=sum(r["output_tokens"] for r in rows) / elapsed,
                requests_per_second=len(rows) / elapsed)


async def measure(session, eos, capacity, inputs, output_budget, warmup, repeats, sampler):
    facade = DeepSeekV4NativeServingModel(session, eos_id=eos)
    engine = InferenceEngine(facade, max_batch_size=len(inputs), max_seq_per_slot=capacity,
                             prefill_chunk_size=capacity, enable_block_prefix_cache=False, block_watermark=0)
    measurements = []
    try:
        engine.start()
        warm = [await cohort(engine, inputs, output_budget) for _ in range(warmup)]
        start = time.perf_counter()
        for _ in range(repeats):
            measurements.append(await cohort(engine, inputs, output_budget))
        end = time.perf_counter()
    finally:
        await asyncio.to_thread(engine.stop, 120)
    if engine._worker_error is not None:
        raise RuntimeError("native benchmark worker cleanup failed") from engine._worker_error
    runtime = facade.last_report
    if (runtime is None or runtime["fallback"] or runtime["logits_observer"]
            or runtime["runtime"]["active_slots"] or runtime["runtime"]["model_failures"]
            or any(op["failures"] or op["fallback"] for op in runtime["runtime"]["operators"])):
        raise RuntimeError("native benchmark runtime/failure accounting invalid")
    rows = [r for trial in measurements for r in trial["requests"]]
    ids = [[r["generated_ids"] for r in trial["requests"]] for trial in warm + measurements]
    if any(value != ids[0] for value in ids[1:]): raise AssertionError("greedy output changed across identical benchmark repeats")
    if session.info()["active_slots"]: raise AssertionError("benchmark left live request slots")
    return dict(input_tokens=len(inputs[0]), input_ids=inputs, requested_output_tokens=output_budget, concurrency=len(inputs),
        warmup=warmup, repeats=repeats, raw_trials=measurements, repeated_greedy_ids_equal=True,
        ttft_ms=summarize([r["ttft_ms"] for r in rows]), tpot_ms=summarize([r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]),
        inter_token_ms=summarize([v for r in rows for v in r["inter_token_ms"]]),
        effective_prompt_tokens_per_second=summarize([r["effective_prompt_tokens_per_second"] for r in rows]),
        per_request_decode_tokens_per_second=summarize([r["decode_tokens_per_second"] for r in rows if r["decode_tokens_per_second"] is not None]),
        aggregate_output_tokens_per_second=summarize([trial["output_tokens_per_second"] for trial in measurements]),
        aggregate_requests_per_second=summarize([trial["requests_per_second"] for trial in measurements]),
        memory=sampler.report(start, end), runtime=facade.last_report)


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native serving benchmark requires Slurm GPU allocation")
    if args.warmup < 1 or args.repeats < 2 or args.output_tokens < 2 or any(c < 1 for c in args.concurrency):
        raise ValueError("benchmark requires warmup>=1, repeats>=2, output>=2 and positive concurrency")
    root = Path(__file__).resolve().parents[2]
    source, checkpoint = args.source_model.resolve(), args.checkpoint.resolve()
    bundle = json.loads(args.bundle.read_text()); capacity = bundle["configuration"]["max_seq_len"]
    if any(n < 1 or n + args.output_tokens - 1 > capacity for n in args.lengths):
        raise ValueError("benchmark prompt/output lengths exceed capacity")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    codec = ChatCodec(source, tokenizer); inputs = prompts(tokenizer, args.lengths, max(args.concurrency))
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid)
    if not uuid.startswith(("GPU-", "MIG-")): uuid = "GPU-" + uuid
    paths = [Path(__file__), args.module.resolve(), args.bundle.resolve(), root / "python/server/engine.py",
             root / "python/bindings/v4_native.cpp", root / "src/backends/v4_reference.cpp", root / "src/backends/v4_reference.hpp",
             root / "xmake/native_model.lua", root / "python/llaisys/libllaisys/libllaisys.so", root / "python/llaisys/_C.so",
             Path(checkpoint_identity.__code__.co_filename), codec.path]
    paths.extend(sorted((root / "python/llaisys/models/deepseek_v4_model").glob("*.py")))
    for directory in ("src/models/deepseek_v4", "src/backends/native", "src/backends/aten", "src/backends/tilelang", "src/backends/hadamard"):
        paths.extend(sorted((root / directory).glob("*.[ch]pp")))
    paths.extend(args.bundle.resolve().parent / k["library"] for k in bundle["kernels"].values())
    paths.extend(source / p for p in ("config.json", "inference/config.json", "inference/kernel.py"))
    hashes = {str(p.resolve()): _digest(p) for p in paths}
    print("verifying entire benchmark checkpoint payload", flush=True)
    identity = checkpoint_identity(checkpoint)
    sampler = MemorySampler(uuid); session = None; results = []
    args._report = dict(all_passed=False, scope="native serving benchmark in progress", cases=results,
                        source_model=str(source), checkpoint_identity=identity,
                        slurm_job_id=os.environ["SLURM_JOB_ID"], provenance=dict(command=sys.argv, file_sha256=hashes))
    try:
        begin_load = time.perf_counter()
        session = load_native_session(source, checkpoint, args.bundle, args.module, slots=max(args.concurrency))
        load_seconds = time.perf_counter() - begin_load
        print("model loaded; warm/measure without logits observer", flush=True)
        for concurrency in args.concurrency:
            for length in args.lengths:
                result = asyncio.run(measure(session, codec.eos_id, capacity, inputs[length][:concurrency], args.output_tokens,
                                             args.warmup, args.repeats, sampler))
                results.append(result)
                print(json.dumps({"input": length, "concurrency": concurrency,
                    "ttft_mean_ms": result["ttft_ms"]["mean"], "tpot_mean_ms": result["tpot_ms"]["mean"],
                    "outputs": [r["output_tokens"] for t in result["raw_trials"] for r in t["requests"]]}), flush=True)
        final = session.info()
    finally:
        try:
            if session is not None: session.close()
        finally:
            sampler.close()
    verify_checkpoint_unchanged(checkpoint, identity)
    unchanged = hashes == {str(p.resolve()): _digest(p) for p in paths}
    valid = unchanged and not sampler.errors and all(row["memory"]["coverage_valid"] for row in results)
    versions = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    report = dict(all_passed=bool(valid), scope="warm native InferenceEngine serving baseline, not optimized/fused/HTTP throughput",
        source_model=str(source), checkpoint_identity=identity, source_unchanged=unchanged, load_seconds_excluded=load_seconds,
        cases=results, final_runtime=final, memory_sampler_errors=sampler.errors, slurm_job_id=os.environ["SLURM_JOB_ID"],
        hardware=dict(nvidia_smi=versions, gpu_count=torch.cuda.device_count(), capability=list(torch.cuda.get_device_capability(0))),
        software=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, tilelang=bundle["tilelang"],
                      tvm_ffi=bundle["tvm_ffi"], nccl=torch.cuda.nccl.version()),
        precision=dict(dense="W8A8 E4M3", routed_experts="W4A8 E2M1/FP8", scale="UE8M0", hidden_cache="BF16", structural="FP32/BF16 per contract"),
        measurement=dict(clock="time.perf_counter", ttft="submit to first asyncio stream token; includes queueing",
            tpot="mean inter-token arrival interval per request, excludes first token", percentiles="linear interpolation; small-sample descriptive only",
            prefill_throughput="input_tokens/TTFT; effective serving rate, not isolated kernel prefill throughput",
            decode_throughput="(output_tokens-1)/(last-first); per request, do not sum to obtain aggregate throughput",
            memory="device-wide 100ms sampled peak, includes native allocations; not exact allocator high-water mark",
            gpu_batch_size=1, fused_batch=False, paged=False, prefix_cache=False, cache_state="fresh request cache each trial",
            cuda_graph=False, fallback=False, logits_observer=False, eos_honored=True, workload="deterministic synthetic raw completion",
            warmup_per_case=args.warmup, repeats_per_case=args.repeats),
        provenance=dict(command=sys.argv, file_sha256=hashes,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())))
    args._report = report
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if not valid: raise RuntimeError("benchmark provenance or memory sampling validation failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-model", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True); p.add_argument("--module", type=Path, required=True)
    p.add_argument("--lengths", nargs="+", type=int, default=[32, 256, 2105]); p.add_argument("--concurrency", nargs="+", type=int, default=[1, 2])
    p.add_argument("--output-tokens", type=int, default=16); p.add_argument("--warmup", type=int, default=1); p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    try: main(args)
    except Exception as error:
        report = getattr(args, "_report", dict(scope="native serving benchmark incomplete", command=sys.argv,
                                             slurm_job_id=os.environ.get("SLURM_JOB_ID")))
        report.update(all_passed=False, error=f"{type(error).__name__}: {error}")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        raise

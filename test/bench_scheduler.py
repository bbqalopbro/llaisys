"""
Performance Benchmark: Scheduler & Engine Throughput

Measures:
  1. RequestQueue submit/get throughput
  2. Dynamic admission decision throughput
  3. Preemption cycle overhead
  4. End-to-end engine throughput (mock model, varying concurrency)
  5. Block utilization tracking under load
"""

import sys
import os
import time
import threading
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from server.engine import (
    InferenceEngine,
    InferenceRequest,
    SamplingParams,
    RequestQueue,
    RequestStatus,
)


# ── Mock Model ────────────────────────────────────────────────────

class PerfMockBatchContext:
    """High-fidelity mock with block tracking and configurable latency."""

    def __init__(self, block_size=16, total_blocks=256, decode_latency_us=0):
        self._block_size = block_size
        self._total_blocks = total_blocks
        self._used_blocks = 0
        self._slots = {}
        self._decode_count = 0
        self._decode_latency_us = decode_latency_us
        self._peak_used = 0
        self._utilization_samples = []

    def get_block_size(self):
        return self._block_size

    def get_free_blocks(self):
        return self._total_blocks - self._used_blocks

    def get_total_blocks(self):
        return self._total_blocks

    def slot_reset(self, slot_id):
        if slot_id in self._slots:
            self._used_blocks -= self._slots[slot_id]["blocks"]
            del self._slots[slot_id]

    def prefill(self, slot_id, token_ids, temperature=0.8, top_k=50, top_p=0.9):
        n = len(token_ids)
        blocks = (n + self._block_size - 1) // self._block_size
        self._used_blocks += blocks
        self._peak_used = max(self._peak_used, self._used_blocks)
        self._slots[slot_id] = {"pos": n, "blocks": blocks}
        return 42

    def decode_per_request(self, active_slots, current_tokens,
                           temperatures, top_ks, top_ps):
        self._decode_count += 1
        if self._decode_latency_us > 0:
            time.sleep(self._decode_latency_us / 1e6)

        self._utilization_samples.append(self._used_blocks / max(1, self._total_blocks))

        results = []
        for sid in active_slots:
            slot = self._slots.get(sid, {"pos": 0, "blocks": 0})
            slot["pos"] += 1
            if slot["pos"] % self._block_size == 0:
                slot["blocks"] += 1
                self._used_blocks += 1
                self._peak_used = max(self._peak_used, self._used_blocks)
            results.append(100 + self._decode_count)
        return results

    def slot_save(self, slot_id):
        return {"slot": slot_id, "data": "snapshot"}

    def slot_restore(self, slot_id, snapshot):
        self._slots[slot_id] = {"pos": 1, "blocks": 1}
        self._used_blocks += 1


class PerfMockModel:
    _end_token = 151643

    def __init__(self, block_size=16, total_blocks=256, decode_latency_us=0):
        self._block_size = block_size
        self._total_blocks = total_blocks
        self._decode_latency_us = decode_latency_us
        self._last_ctx = None

    def create_batch_context(self, max_batch_size, max_seq_per_slot):
        ctx = PerfMockBatchContext(
            block_size=self._block_size,
            total_blocks=self._total_blocks,
            decode_latency_us=self._decode_latency_us,
        )
        self._last_ctx = ctx
        return ctx


# ── Benchmark 1: RequestQueue Throughput ──────────────────────────

def bench_request_queue(n_ops=100000):
    q = RequestQueue(max_size=n_ops + 1)

    requests = [
        InferenceRequest(
            request_id=f"r{i}",
            input_ids=list(range(32)),
            params=SamplingParams(),
            session_id="s",
        )
        for i in range(n_ops)
    ]

    # Submit
    t0 = time.perf_counter()
    for req in requests:
        q.submit(req)
    submit_time = time.perf_counter() - t0

    # Get pending
    t0 = time.perf_counter()
    q.get_pending(max_count=n_ops)
    get_time = time.perf_counter() - t0

    submit_ops = n_ops / submit_time
    get_ops = n_ops / get_time

    print(f"  Submit: {n_ops} ops in {submit_time*1000:.1f}ms "
          f"({submit_ops/1e6:.2f} M ops/s)")
    print(f"  Get:    {n_ops} ops in {get_time*1000:.1f}ms "
          f"({get_ops/1e6:.2f} M ops/s)")


# ── Benchmark 2: Admission Decision Throughput ────────────────────

def bench_admission_decisions(n_ops=100000):
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.block_watermark = 0.1

    ctx = PerfMockBatchContext(block_size=16, total_blocks=1024)

    t0 = time.perf_counter()
    for _ in range(n_ops):
        engine._can_admit(ctx, prompt_len=64, max_tokens=256)
    elapsed = time.perf_counter() - t0

    ops_per_sec = n_ops / elapsed
    print(f"  Admission checks: {n_ops} ops in {elapsed*1000:.1f}ms "
          f"({ops_per_sec/1e6:.2f} M ops/s)")


# ── Benchmark 3: Preemption Cycle ─────────────────────────────────

def bench_preemption_cycle(n_ops=10000):
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.queue = RequestQueue(max_size=n_ops * 2)
    engine._total_preemptions = 0

    ctx = PerfMockBatchContext(block_size=16, total_blocks=1024)

    t0 = time.perf_counter()
    for i in range(n_ops):
        running = {}
        free_slots = []

        req = InferenceRequest(
            request_id=f"victim_{i}",
            input_ids=list(range(32)),
            params=SamplingParams(),
            session_id="s",
        )
        req.generated_tokens = list(range(50))
        req.status = RequestStatus.DECODING
        running[0] = req
        ctx._slots[0] = {"pos": 50, "blocks": 4}
        ctx._used_blocks = 4

        engine._try_preempt(ctx, running, free_slots, 5)

    elapsed = time.perf_counter() - t0
    ops_per_sec = n_ops / elapsed
    print(f"  Preemption cycles: {n_ops} ops in {elapsed*1000:.1f}ms "
          f"({ops_per_sec/1e3:.1f} K ops/s)")


# ── Benchmark 4: End-to-End Engine Throughput ─────────────────────

def bench_engine_throughput(max_batch_size, num_requests, prompt_len,
                             max_tokens, total_blocks, decode_latency_us=0):
    model = PerfMockModel(
        block_size=16,
        total_blocks=total_blocks,
        decode_latency_us=decode_latency_us,
    )
    engine = InferenceEngine(
        model=model,
        max_batch_size=max_batch_size,
        max_seq_per_slot=4096,
        max_queue_size=num_requests + 100,
        block_watermark=0.05,
    )

    engine.start()

    # Submit all requests
    import asyncio
    loop = asyncio.new_event_loop()

    submitted = []
    for i in range(num_requests):
        req = engine.submit(
            input_ids=list(range(prompt_len)),
            params=SamplingParams(max_tokens=max_tokens),
            session_id=f"bench_s{i}",
            loop=loop,
        )
        submitted.append(req)

    # Wait for all to finish
    t0 = time.perf_counter()
    timeout = 60.0
    while time.perf_counter() - t0 < timeout:
        if all(r.is_finished for r in submitted):
            break
        time.sleep(0.01)

    elapsed = time.perf_counter() - t0
    engine.stop()
    loop.close()

    finished = sum(1 for r in submitted if r.status == RequestStatus.DONE)
    total_tokens = sum(len(r.generated_tokens) for r in submitted if r.status == RequestStatus.DONE)
    token_throughput = total_tokens / elapsed if elapsed > 0 else 0

    stats = engine.get_stats()
    ctx = model._last_ctx

    util_avg = statistics.mean(ctx._utilization_samples) if ctx._utilization_samples else 0
    util_max = max(ctx._utilization_samples) if ctx._utilization_samples else 0

    print(f"  B={max_batch_size} reqs={num_requests} prompt={prompt_len} "
          f"max_tok={max_tokens} blocks={total_blocks} "
          f"latency={decode_latency_us}us")
    print(f"    Finished: {finished}/{num_requests} in {elapsed:.2f}s")
    print(f"    Token throughput: {token_throughput:.0f} tok/s "
          f"({total_tokens} tokens)")
    print(f"    Preemptions: {stats['total_preemptions']}")
    print(f"    Block util: avg={util_avg:.1%} peak={util_max:.1%} "
          f"(peak_used={ctx._peak_used}/{ctx._total_blocks})")
    print()


# ── Benchmark 5: Block Utilization Under Pressure ─────────────────

def bench_block_utilization():
    """Simulate workload mix: short & long requests competing for blocks."""
    model = PerfMockModel(block_size=16, total_blocks=128)
    engine = InferenceEngine(
        model=model,
        max_batch_size=8,
        max_seq_per_slot=4096,
        max_queue_size=1024,
        block_watermark=0.1,
    )
    engine.start()

    import asyncio
    loop = asyncio.new_event_loop()

    # Mix: 20 short (prompt=16, gen=16) + 10 long (prompt=64, gen=128)
    submitted = []
    for i in range(20):
        req = engine.submit(
            input_ids=list(range(16)),
            params=SamplingParams(max_tokens=16),
            session_id=f"short_{i}",
            loop=loop,
        )
        submitted.append(("short", req))

    for i in range(10):
        req = engine.submit(
            input_ids=list(range(64)),
            params=SamplingParams(max_tokens=128),
            session_id=f"long_{i}",
            loop=loop,
        )
        submitted.append(("long", req))

    t0 = time.perf_counter()
    timeout = 30.0
    while time.perf_counter() - t0 < timeout:
        if all(r.is_finished for _, r in submitted):
            break
        time.sleep(0.01)

    elapsed = time.perf_counter() - t0
    engine.stop()
    loop.close()

    stats = engine.get_stats()
    ctx = model._last_ctx

    short_done = sum(1 for t, r in submitted if t == "short" and r.status == RequestStatus.DONE)
    long_done = sum(1 for t, r in submitted if t == "long" and r.status == RequestStatus.DONE)
    short_tokens = sum(len(r.generated_tokens) for t, r in submitted
                       if t == "short" and r.status == RequestStatus.DONE)
    long_tokens = sum(len(r.generated_tokens) for t, r in submitted
                      if t == "long" and r.status == RequestStatus.DONE)

    print(f"  Workload: 20 short + 10 long requests, 128 blocks")
    print(f"    Short: {short_done}/20 done, {short_tokens} tokens")
    print(f"    Long:  {long_done}/10 done, {long_tokens} tokens")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    Preemptions: {stats['total_preemptions']}")
    if ctx._utilization_samples:
        util_avg = statistics.mean(ctx._utilization_samples)
        util_p50 = statistics.median(ctx._utilization_samples)
        util_max = max(ctx._utilization_samples)
        print(f"    Block util: avg={util_avg:.1%} p50={util_p50:.1%} peak={util_max:.1%}")
    print()


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  LLAISYS Scheduler Performance Benchmark")
    print("=" * 64)
    print()

    print("--- 1. RequestQueue Throughput ---")
    bench_request_queue(100000)
    print()

    print("--- 2. Admission Decision Throughput ---")
    bench_admission_decisions(100000)
    print()

    print("--- 3. Preemption Cycle Overhead ---")
    bench_preemption_cycle(10000)
    print()

    print("--- 4. End-to-End Engine Throughput ---")
    # Small batch, few requests
    bench_engine_throughput(max_batch_size=1, num_requests=10,
                            prompt_len=32, max_tokens=32,
                            total_blocks=256)
    # Moderate batch
    bench_engine_throughput(max_batch_size=4, num_requests=20,
                            prompt_len=32, max_tokens=64,
                            total_blocks=256)
    # Large batch
    bench_engine_throughput(max_batch_size=8, num_requests=40,
                            prompt_len=64, max_tokens=64,
                            total_blocks=512)
    # Memory pressure (limited blocks)
    bench_engine_throughput(max_batch_size=8, num_requests=20,
                            prompt_len=64, max_tokens=128,
                            total_blocks=64)
    # With simulated decode latency (100us per step)
    bench_engine_throughput(max_batch_size=4, num_requests=20,
                            prompt_len=32, max_tokens=64,
                            total_blocks=256,
                            decode_latency_us=100)

    print("--- 5. Block Utilization Under Mixed Workload ---")
    bench_block_utilization()

    print("=" * 64)
    print("  Benchmark complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()

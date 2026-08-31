"""
Phase 4 Scheduler Tests — validates dynamic admission, preemption, and per-request sampling.

These tests verify the scheduling logic at the Python level using mock objects,
without requiring a real model or GPU.
"""

import sys
import os
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from server.engine import (
    InferenceEngine,
    InferenceRequest,
    SamplingParams,
    RequestQueue,
    RequestStatus,
)


class MockBatchContext:
    """Simulates BatchContext with block accounting."""

    def __init__(self, block_size=16, total_blocks=64):
        self._block_size = block_size
        self._total_blocks = total_blocks
        self._used_blocks = 0
        self._slots = {}  # slot_id -> {"pos": int, "blocks": int}
        self._decode_count = 0

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
        self._slots[slot_id] = {"pos": n, "blocks": blocks}
        return 42  # dummy token

    def decode_per_request(self, active_slots, current_tokens,
                           temperatures, top_ks, top_ps):
        self._decode_count += 1
        results = []
        for sid in active_slots:
            slot = self._slots.get(sid, {"pos": 0, "blocks": 0})
            slot["pos"] += 1
            if slot["pos"] % self._block_size == 0:
                slot["blocks"] += 1
                self._used_blocks += 1
            results.append(100 + self._decode_count)
        return results

    def slot_save(self, slot_id):
        return f"snapshot_{slot_id}"

    def slot_restore(self, slot_id, snapshot):
        self._slots[slot_id] = {"pos": 1, "blocks": 1}
        self._used_blocks += 1


class MockModel:
    """Minimal model mock for engine tests."""
    _end_token = 151643

    def create_batch_context(self, max_batch_size, max_seq_per_slot):
        return MockBatchContext(block_size=16, total_blocks=64)


# ── Unit Tests ──

class TestRequestQueue(unittest.TestCase):
    def test_submit_and_get(self):
        q = RequestQueue(max_size=10)
        req = InferenceRequest(
            request_id="r1", input_ids=[1, 2, 3],
            params=SamplingParams(), session_id="s1"
        )
        assert q.submit(req)
        assert q.waiting_count == 1
        batch = q.get_pending(max_count=5)
        assert len(batch) == 1
        assert batch[0].request_id == "r1"
        assert q.waiting_count == 0

    def test_max_size(self):
        q = RequestQueue(max_size=2)
        for i in range(3):
            req = InferenceRequest(
                request_id=f"r{i}", input_ids=[i],
                params=SamplingParams(), session_id="s"
            )
            result = q.submit(req)
            if i < 2:
                assert result
            else:
                assert not result


class TestDynamicAdmission(unittest.TestCase):
    def test_can_admit_check(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.block_watermark = 0.1

        ctx = MockBatchContext(block_size=16, total_blocks=100)

        # Small request should be admitted
        assert engine._can_admit(ctx, prompt_len=10, max_tokens=20)

        # Use up most blocks
        ctx._used_blocks = 95
        # Free = 5, watermark = 10, needed ~ 2 blocks → 5 < 2 + 10 → reject
        assert not engine._can_admit(ctx, prompt_len=10, max_tokens=20)

    def test_estimate_blocks(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        # 100 tokens with block_size=16 → ceil(100/16) = 7 blocks
        assert engine._estimate_blocks_needed(50, 50, 16) == 7
        # Exact multiple
        assert engine._estimate_blocks_needed(16, 16, 16) == 2


class TestPreemption(unittest.TestCase):
    def test_preempt_longest(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.queue = RequestQueue()
        engine._total_preemptions = 0

        ctx = MockBatchContext(block_size=16, total_blocks=100)

        running = {}
        free_slots = []

        # Add two running requests with different lengths
        req_short = InferenceRequest(
            request_id="short", input_ids=[1, 2],
            params=SamplingParams(), session_id="s"
        )
        req_short.generated_tokens = [10, 11]
        req_short.status = RequestStatus.DECODING
        running[0] = req_short

        req_long = InferenceRequest(
            request_id="long", input_ids=[1, 2, 3, 4, 5],
            params=SamplingParams(), session_id="s"
        )
        req_long.generated_tokens = [10, 11, 12, 13, 14, 15, 16, 17]
        req_long.status = RequestStatus.DECODING
        running[1] = req_long

        # Preempt should evict the longer one
        result = engine._try_preempt(ctx, running, free_slots, 5)
        assert result
        assert 1 not in running  # slot 1 freed
        assert 0 in running      # slot 0 kept
        assert 1 in free_slots
        assert engine._total_preemptions == 1


class TestPerRequestSampling(unittest.TestCase):
    def test_different_params(self):
        ctx = MockBatchContext(block_size=16, total_blocks=100)
        ctx._slots = {0: {"pos": 5, "blocks": 1}, 1: {"pos": 10, "blocks": 1}}
        ctx._used_blocks = 2

        results = ctx.decode_per_request(
            active_slots=[0, 1],
            current_tokens=[42, 43],
            temperatures=[0.5, 1.2],
            top_ks=[1, 50],
            top_ps=[0.9, 0.95],
        )
        assert len(results) == 2


class TestChunkedPrefill(unittest.TestCase):
    def test_scheduler_splits_prompt_and_tracks_positions(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.prefill_chunk_size = 3
        calls = []

        class ChunkContext(MockBatchContext):
            def prefill_chunk(self, slot_id, token_ids, start_pos,
                              is_last_chunk, **sampling):
                calls.append((list(token_ids), start_pos, is_last_chunk))
                return 77 if is_last_chunk else None

        req = InferenceRequest(
            request_id="chunk", input_ids=list(range(8)),
            params=SamplingParams(), session_id="s"
        )
        result = engine._prefill_prompt(ChunkContext(), 0, req)
        assert result == 77
        assert calls == [
            ([0, 1, 2], 0, False),
            ([3, 4, 5], 3, False),
            ([6, 7], 6, True),
        ]


if __name__ == "__main__":
    unittest.main(verbosity=2)

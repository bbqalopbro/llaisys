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

    def create_cache_pool(self):
        return None


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


class TestGraphStartup(unittest.TestCase):
    def test_capture_failure_is_reported_to_start(self):
        class BrokenContext(MockBatchContext):
            def prepare_graphs(self):
                raise RuntimeError("injected capture failure")
        class BrokenModel(MockModel):
            def create_batch_context(self, *args):
                return BrokenContext()
        engine = InferenceEngine(BrokenModel())
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            engine.start()
        self.assertFalse(engine._running)
        engine.stop()



class ChunkContext:
    """Tracks the actual KV token sequence, pages and snapshot ownership."""
    def __init__(self, model, total=128):
        self.model = model
        self.total = total
        self.slots = {}
        self.events = []
        self.chunks = 0
        self.replays = 0
        self.on_chunk = None

    def get_block_size(self): return 16
    def get_total_blocks(self): return self.total
    def get_free_blocks(self): return self.total - sum((len(x)+15)//16 for x in self.slots.values())
    def slot_get_pos(self, sid): return len(self.slots.get(sid, []))
    def slot_reset(self, sid): self.slots.pop(sid, None)
    def slot_truncate(self, sid, pos): self.slots[sid] = self.slots[sid][:pos]
    def prepare_graphs(self): pass
    def graph_stats(self): return dict(replays=self.replays)
    @staticmethod
    def next_token(tokens): return sum(tokens) % 997 + 1
    def prefill_chunk(self, sid, tokens, *, final=False, **kwargs):
        old = self.slots.setdefault(sid, [])
        extra = (len(old)+len(tokens)+15)//16 - (len(old)+15)//16
        if extra > self.get_free_blocks(): raise RuntimeError("pool exhausted")
        old.extend(tokens)
        self.chunks += 1
        self.events.append(("prefill", sid, len(tokens), final))
        if self.on_chunk: self.on_chunk(self, sid, final)
        time.sleep(.001)
        return self.next_token(old) if final else None
    def decode_per_request(self, slots, tokens, *args):
        self.replays += 1
        self.events.append(("decode", len(slots)))
        values = []
        for sid, token in zip(slots, tokens):
            self.slots[sid].append(token)
            values.append(self.next_token(self.slots[sid]))
        return values
    def slot_save(self, sid):
        snap = tuple(self.slots.get(sid, []))
        self.model.owned.append(snap)
        return snap
    def slot_restore(self, sid, snapshot):
        assert (len(snapshot)+15)//16 <= self.get_free_blocks()
        self.slots[sid] = list(snapshot)


class ChunkModel:
    _end_token = -1
    def __init__(self, total=128):
        self.total = total
        self.owned = []
        self.pool = {}
    def create_batch_context(self, *args):
        self.ctx = ChunkContext(self, self.total)
        return self.ctx
    def create_cache_pool(self): return self.pool
    def cache_pool_lookup(self, pool, tokens):
        keys = [key for key in pool if tuple(tokens[:len(key)]) == key]
        if not keys: return None, 0
        key = max(keys, key=len)
        return pool[key], len(key)
    def cache_pool_insert(self, pool, tokens, snapshot):
        assert tuple(tokens) == snapshot, "cache key includes an uncomputed token"
        self.owned.remove(snapshot)
        pool[tuple(tokens)] = snapshot
    def destroy_snapshot(self, snap): self.owned.remove(snap)


class TestChunkScheduler(unittest.TestCase):
    def run_case(self, coroutine):
        import asyncio
        asyncio.run(asyncio.wait_for(coroutine, 10))

    def test_budget_interleave_and_reference(self):
        async def run():
            import asyncio
            model = ChunkModel()
            engine = InferenceEngine(model, max_batch_size=3, max_seq_per_slot=256,
                prefill_chunk_size=17, max_num_batched_tokens=24, scheduler_trace=True)
            engine.start()
            try:
                reqs = [engine.submit(list(range(1, n+1)), SamplingParams(max_tokens=8), str(n), stream=True)
                        for n in [5, 73, 99]]
                values = await asyncio.gather(*(r.future for r in reqs))
                for req, value in zip(reqs, values):
                    tokens = req.input_ids[:]; expected = []
                    for _ in range(8):
                        token = ChunkContext.next_token(tokens); expected.append(token); tokens.append(token)
                    self.assertEqual(value, expected)
                    streamed = [t async for t in req.stream_tokens()]
                    self.assertEqual(streamed, expected)
                trace = engine.get_stats()['scheduler_trace']; usage = {}
                for e in trace:
                    if e['kind'] in ('decode', 'prefill'):
                        usage[e['iteration']] = usage.get(e['iteration'], 0) + e['tokens']
                self.assertTrue(all(n <= 24 for n in usage.values()))
                self.assertTrue(any(e['kind']=='prefill' and not e['final'] for e in trace))
                self.assertTrue(any(e['kind']=='decode' and any(p['kind']=='prefill' and p['iteration']==e['iteration'] for p in trace) for e in trace))
                for it in usage:
                    kinds = [e['kind'] for e in trace if e['iteration']==it and e['kind'] in ('decode','prefill')]
                    if 'decode' in kinds: self.assertEqual(kinds[0], 'decode')
                self.assertEqual(model.ctx.get_free_blocks(), model.ctx.total)
            finally: engine.stop()
        self.run_case(run())

    def test_full_and_partial_prefix_hit(self):
        async def run():
            model = ChunkModel()
            # Truthy sentinel is not a matching prefix.
            model.pool[(-5,)] = (-5,)
            engine = InferenceEngine(model, max_batch_size=2, max_seq_per_slot=128,
                prefill_chunk_size=7, max_num_batched_tokens=16)
            engine.start()
            try:
                prompt = list(range(1, 34))
                async def get(ids):
                    r = engine.submit(ids, SamplingParams(max_tokens=1), 's')
                    return await r.future
                expected = await get(prompt)
                before = engine._prefill_tokens
                self.assertEqual(await get(prompt), expected)
                self.assertEqual(engine._prefill_tokens-before, 1)
                extended = prompt + [88, 99]
                self.assertEqual(await get(extended), [ChunkContext.next_token(extended)])
                self.assertEqual(engine._prefix_hits, 2)
                self.assertFalse(model.owned)
            finally: engine.stop()
        self.run_case(run())

    def test_preempt_partial_prefill_restores_cursor(self):
        async def run():
            import asyncio
            model = ChunkModel(total=4)
            engine = InferenceEngine(model, max_batch_size=2, max_seq_per_slot=80,
                prefill_chunk_size=8, max_num_batched_tokens=16, block_watermark=0,
                scheduler_trace=True)
            engine.start()
            try:
                reqs = [engine.submit(list(range(1, n+1)), SamplingParams(max_tokens=3), str(n)) for n in [45, 47]]
                values = await asyncio.gather(*(r.future for r in reqs))
                self.assertGreater(engine._total_preemptions, 0)
                for req, value in zip(reqs, values):
                    tokens = req.input_ids[:]; expected = []
                    for _ in range(3):
                        t = ChunkContext.next_token(tokens); expected.append(t); tokens.append(t)
                    self.assertEqual(value, expected)
                self.assertFalse(model.owned)
                self.assertEqual(model.ctx.get_free_blocks(), 4)
            finally: engine.stop()
        self.run_case(run())

    def test_cancel_during_prefill_and_waiting_stream_terminate(self):
        async def run():
            import asyncio
            model = ChunkModel(); engine = InferenceEngine(model, max_batch_size=1,
                max_seq_per_slot=256, prefill_chunk_size=8, max_num_batched_tokens=8)
            engine.start()
            try:
                hit = threading.Event(); release = threading.Event()
                def pause(ctx, sid, final):
                    hit.set(); release.wait(2); ctx.on_chunk = None
                model.ctx.on_chunk = pause
                active = engine.submit(list(range(1, 100)), SamplingParams(max_tokens=4), 'a', stream=True)
                await asyncio.to_thread(hit.wait, 2)
                waiting = engine.submit([1,2], SamplingParams(max_tokens=4), 'b', stream=True)
                engine.queue.cancel(active.request_id); engine.queue.cancel(waiting.request_id)
                release.set()
                await asyncio.gather(active.future, waiting.future, return_exceptions=True)
                self.assertEqual(active.status, RequestStatus.CANCELLED)
                self.assertEqual(waiting.status, RequestStatus.CANCELLED)
                self.assertEqual([t async for t in active.stream_tokens()], [])
                self.assertEqual([t async for t in waiting.stream_tokens()], [])
                self.assertEqual(model.ctx.get_free_blocks(), model.ctx.total)
            finally: engine.stop()
        self.run_case(run())

    def test_failure_reclaims_slot_and_shutdown_settles_waiters(self):
        async def run():
            import asyncio
            model = ChunkModel(); engine = InferenceEngine(model, max_batch_size=1,
                max_seq_per_slot=256, prefill_chunk_size=8, max_num_batched_tokens=8)
            engine.start()
            def fail(ctx, sid, final):
                ctx.on_chunk = None
                raise RuntimeError('injected chunk failure')
            model.ctx.on_chunk = fail
            req = engine.submit(list(range(1,50)), SamplingParams(max_tokens=3), 'fail', stream=True)
            with self.assertRaisesRegex(RuntimeError, 'injected chunk failure'): await req.future
            self.assertEqual([t async for t in req.stream_tokens()], [])
            self.assertEqual(model.ctx.get_free_blocks(), model.ctx.total)
            reqs = [engine.submit(list(range(1,100)), SamplingParams(max_tokens=20), str(i), stream=True) for i in range(3)]
            engine.stop()
            await asyncio.gather(*(r.future for r in reqs), return_exceptions=True)
            self.assertTrue(all(r.status==RequestStatus.CANCELLED for r in reqs))
            self.assertFalse(model.owned)
        self.run_case(run())

    def test_cancel_during_prefix_restore_is_not_overwritten(self):
        async def run():
            import asyncio
            model=ChunkModel();prompt=list(range(1,34));model.pool[tuple(prompt)]=tuple(prompt)
            engine=InferenceEngine(model,max_batch_size=1,max_seq_per_slot=128)
            engine.start()
            entered=threading.Event();release=threading.Event()
            restore=model.ctx.slot_restore
            def blocked(sid,snapshot):
                entered.set();release.wait(2);restore(sid,snapshot)
            model.ctx.slot_restore=blocked
            try:
                req=engine.submit(prompt,SamplingParams(max_tokens=3),'prefix-cancel',stream=True)
                await asyncio.to_thread(entered.wait,2)
                engine.queue.cancel(req.request_id);release.set()
                await asyncio.gather(req.future,return_exceptions=True)
                self.assertEqual(req.status,RequestStatus.CANCELLED)
                self.assertEqual([t async for t in req.stream_tokens()],[])
                self.assertEqual(model.ctx.chunks,0)
                self.assertEqual(model.ctx.get_free_blocks(),model.ctx.total)
            finally:release.set();engine.stop()
        self.run_case(run())

    def test_invalid_budget_and_oversize_request(self):
        with self.assertRaises(ValueError): InferenceEngine(ChunkModel(), max_batch_size=8, max_num_batched_tokens=4)
        async def run():
            engine = InferenceEngine(ChunkModel(), max_seq_per_slot=16)
            engine.start()
            try:
                req = engine.submit([1]*16, SamplingParams(max_tokens=2), 'bad', stream=True)
                with self.assertRaises(ValueError): await req.future
                self.assertEqual([t async for t in req.stream_tokens()], [])
            finally: engine.stop()
        self.run_case(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)

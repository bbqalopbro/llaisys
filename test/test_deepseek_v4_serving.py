"""Actual V4 reference slots/C++ pools and the existing threaded scheduler.

CPU small-model tests always run. CUDA coverage requires an allocated Slurm GPU;
these tests do not imply full-checkpoint or statistical performance acceptance.
"""

import asyncio
import gc
import os
import threading
import unittest
from unittest.mock import patch

import torch

from llaisys import _C
from llaisys.models.deepseek_v4_model.batch import DeepSeekV4ServingModel
from llaisys.models.deepseek_v4_model.generation import greedy_generate
from server.engine import InferenceEngine, SamplingParams, RequestStatus
from test_deepseek_v4_model import initialized_model


def fixture(device="cpu", **options):
    torch.manual_seed(730)
    model = initialized_model(fixed_rows=32, indexer_tie_policy="index_ascending",
                              attention_metadata_policy="fixed", max_seq_len=512).to(device)
    return model, DeepSeekV4ServingModel(model, eos_id=1, **options)


def devices():
    return ("cpu", "cuda") if os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() else ("cpu",)


class V4BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @torch.inference_mode()
    def test_serial_slots_match_independent_generation_and_release(self):
        for device in devices():
            model, facade = fixture(device)
            before = _C.v4_storage_counters()
            prompts = [[i % 32 for i in range(137)], [3, 7, 9]]
            expected = [greedy_generate(model, ids, max_new_tokens=4, eos_id=1)["generated_ids"] for ids in prompts]
            ctx = facade.create_batch_context(2, 512)
            generated = [[ctx.prefill(i, prompt, temperature=0, top_k=1)] for i, prompt in enumerate(prompts)]
            for _ in range(3):
                active = [i for i in range(2) if generated[i][-1] != 1]
                tokens = ctx.decode_per_request(active, [generated[i][-1] for i in active],
                                                [0] * len(active), [1] * len(active), [1] * len(active))
                for i, token in zip(active, tokens):
                    generated[i].append(token)
            self.assertEqual(generated, expected)
            ctx.close()
            gc.collect()
            self.assertEqual(_C.v4_storage_counters(), before)
            self.assertFalse(facade.last_report["fused_gpu_batch"])
            self.assertEqual(facade.last_report["pool"]["num_free"], 8)

    @torch.inference_mode()
    def test_host_snapshot_roundtrip_preserves_logits_compressors_and_rng(self):
        for device in devices():
            captured = []
            model, facade = fixture(device, observer=lambda *args: captured.append(args[-1].cpu().clone()),
                                    enable_prefix_cache=True, allow_experimental_chunking=True)
            ctx = facade.create_batch_context(2, 512)
            tokens = [i % 32 for i in range(133)]
            first = ctx.prefill(0, tokens, temperature=0.7, top_k=8)
            snapshot = ctx.slot_save(0)
            self.assertGreater(snapshot.state.host_bytes, 0)
            self.assertTrue(all(t.device.type == "cpu" for layer in snapshot.state.layers for t in layer.values()))
            original_ids = ctx.slots[0].lease.block_ids
            expected = ctx.decode_per_request([0], [first], [0.7], [8], [0.9])
            expected_logits = captured[-1]
            ctx.slot_reset(0)
            self.assertEqual(ctx.get_free_blocks(), ctx.get_total_blocks())
            ctx.slot_restore(1, snapshot)
            self.assertNotEqual(ctx.slots[1].lease.block_ids, original_ids)
            self.assertEqual(ctx.slot_get_pos(1), 133)
            actual = ctx.decode_per_request([1], [first], [0.7], [8], [0.9])
            self.assertEqual(actual, expected)
            torch.testing.assert_close(captured[-1], expected_logits, atol=0, rtol=0)
            snapshot.close()
            self.assertEqual(snapshot.state.host_bytes, 0)
            ctx.close()

    @torch.inference_mode()
    def test_snapshot_cross_pool_same_model_and_foreign_or_closed_rejection(self):
        model, facade = fixture()
        ctx = facade.create_batch_context(1, 512)
        token = ctx.prefill(0, [4, 8, 9], temperature=0)
        snapshot = ctx.slot_save(0)
        ctx.close()
        ctx = facade.create_batch_context(1, 512)
        ctx.slot_restore(0, snapshot)
        self.assertEqual(ctx.pending[0], token)
        ctx.slot_reset(0)
        _, foreign = fixture()
        other = foreign.create_batch_context(1, 512)
        with self.assertRaisesRegex(ValueError, "foreign"):
            other.slot_restore(0, snapshot)
        other.close()
        snapshot.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            ctx.slot_restore(0, snapshot)
        ctx.close()

    @torch.inference_mode()
    def test_failed_restore_returns_blocks_and_preserves_snapshot(self):
        _, facade = fixture()
        ctx = facade.create_batch_context(1, 512)
        ctx.prefill(0, list(range(20)), temperature=0)
        snapshot = ctx.slot_save(0)
        ctx.slot_reset(0)
        free = ctx.get_free_blocks()
        with patch.dict(ctx.pool.ops._functions, cache_write=lambda *a: (_ for _ in ()).throw(RuntimeError("copy failed"))):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                ctx.slot_restore(0, snapshot)
        self.assertEqual(ctx.get_free_blocks(), free)
        self.assertIsNone(ctx.slots[0])
        self.assertFalse(snapshot.state.closed)
        ctx.slot_restore(0, snapshot)
        snapshot.close()
        ctx.close()

    def test_owner_thread_and_single_context_are_enforced(self):
        _, facade = fixture()
        ctx = facade.create_batch_context(1, 512)
        errors = []
        def wrong_thread():
            try:
                ctx.get_free_blocks()
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=wrong_thread)
        thread.start()
        thread.join()
        self.assertRegex(str(errors[0]), "owning worker")
        with self.assertRaisesRegex(RuntimeError, "active serving"):
            facade.create_batch_context(1, 512)
        ctx.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            ctx.prefill(0, [1, 2])
        facade.create_batch_context(1, 512).close()

    def test_default_rejects_shape_sensitive_prefill_and_invalid_sampling(self):
        model, facade = fixture()
        with self.assertRaisesRegex(ValueError, "opt-in"):
            DeepSeekV4ServingModel(model, eos_id=1, enable_prefix_cache=True)
        ctx = facade.create_batch_context(2, 512)
        with self.assertRaisesRegex(RuntimeError, "opt-in"):
            ctx.prefill_chunk(0, [2], 0, False)
        for params in ((-1, 3, .9), (float("nan"), 3, .9), (.5, -1, .9), (.5, 3, 0)):
            with self.assertRaises(ValueError):
                ctx.prefill(0, [2], *params)
        first = ctx.prefill(0, [3, 4], temperature=0)
        position = ctx.slot_get_pos(0)
        with self.assertRaises(ValueError):
            ctx.decode_per_request([0, 1], [first, 1], [0, 0], [1, 1], [1, 1])
        self.assertEqual(ctx.slot_get_pos(0), position)
        ctx.close()

    @torch.inference_mode()
    def test_explicit_chunk_prefix_and_phase_contract(self):
        captured = []
        _, facade = fixture(enable_prefix_cache=True, allow_experimental_chunking=True,
                            observer=lambda *args: captured.append(args[-1].clone() if args[-1] is not None else None))
        ctx = facade.create_batch_context(2, 512)
        tokens = [i % 32 for i in range(137)]
        expected = ctx.prefill(0, tokens, temperature=0)
        logits = captured[-1]
        self.assertTrue(ctx.prefix_publish(0, tokens))
        self.assertEqual(ctx.prefix_lookup(1, tokens), 128)
        self.assertIsNone(ctx.prefill_chunk(1, tokens[128:133], 128, False, temperature=0))
        self.assertIsNone(captured[-1])
        with self.assertRaises(ValueError):
            ctx.prefill_chunk(1, tokens[133:], 128, True, temperature=0)
        self.assertEqual(ctx.prefill_chunk(1, tokens[133:], 133, True, temperature=0), expected)
        torch.testing.assert_close(captured[-1], logits, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            ctx.prefill_chunk(1, [7], 137, True)
        with self.assertRaises(ValueError):
            ctx.prefix_publish(1, tokens[:-1] + [9])
        ctx.close()


class V4ServingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.engines = []

    async def asyncTearDown(self):
        for engine in self.engines:
            await asyncio.to_thread(engine.stop)
        torch.set_num_threads(self.threads)

    def engine(self, facade, **kwargs):
        engine = InferenceEngine(facade, max_batch_size=2, max_seq_per_slot=512,
                                 block_watermark=0, prefill_chunk_size=kwargs.pop("prefill_chunk_size", 512), **kwargs)
        self.engines.append(engine)
        return engine

    async def test_existing_engine_real_worker_streams_and_matches_generation(self):
        for device in devices():
            model, facade = fixture(device)
            prompts = [[3, 4, 5], [i % 32 for i in range(137)]]
            expected = [greedy_generate(model, ids, max_new_tokens=4, eos_id=1)["generated_ids"] for ids in prompts]
            engine = self.engine(facade)
            requests = [engine.submit(ids, SamplingParams(temperature=0, top_k=1, max_tokens=4), "test", stream=True)
                        for ids in prompts]
            engine.start()
            actual = await asyncio.wait_for(asyncio.gather(*(r.future for r in requests)), timeout=45)
            self.assertEqual(actual, expected)
            for req, tokens in zip(requests, expected):
                self.assertEqual([token async for token in req.stream_tokens()], tokens)
                self.assertEqual(req.status, RequestStatus.DONE)
            await asyncio.to_thread(engine.stop)
            self.assertEqual(engine.queue.active_count, 0)
            self.assertEqual(facade.last_report["pool"]["num_free"], 8)
            if device == "cuda":
                self.assertNotEqual(facade.last_report["worker_owned_stream"], torch.cuda.current_stream().cuda_stream)

    async def test_cancel_waiting_and_mid_chunk_completes_future_and_releases(self):
        engine = None
        active_req = None
        def cancel_after_chunk(stage, *args):
            if stage == "prefill_chunk":
                engine.cancel(active_req.request_id)
        _, facade = fixture(allow_experimental_chunking=True, observer=cancel_after_chunk)
        engine = self.engine(facade, prefill_chunk_size=65)
        waiting = engine.submit([4, 5], SamplingParams(max_tokens=4), "wait", stream=True)
        self.assertTrue(engine.cancel(waiting.request_id))
        active_req = engine.submit([i % 32 for i in range(137)], SamplingParams(max_tokens=4), "active", stream=True)
        engine.start()
        for req in (waiting, active_req):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(req.future, timeout=30)
            self.assertEqual([token async for token in req.stream_tokens()], [])
            self.assertEqual(req.status, RequestStatus.CANCELLED)
        await asyncio.to_thread(engine.stop)
        self.assertEqual(facade.last_report["calls"]["prefill_chunk"], 1)
        self.assertEqual(facade.last_report["pool"]["num_free"], 8)

    async def test_prefill_failure_releases_slot_then_next_request_works(self):
        failed = False
        def fail_once(*args):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("injected execution failure")
        model, facade = fixture(observer=fail_once)
        expected = greedy_generate(model, [4, 5], max_new_tokens=2, eos_id=1)["generated_ids"]
        engine = self.engine(facade)
        bad = engine.submit([2, 3], SamplingParams(temperature=0, max_tokens=2), "bad", stream=True)
        good = engine.submit([4, 5], SamplingParams(temperature=0, max_tokens=2), "good")
        engine.start()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            await asyncio.wait_for(bad.future, timeout=30)
        self.assertEqual([token async for token in bad.stream_tokens()], [])
        self.assertEqual(await asyncio.wait_for(good.future, timeout=30), expected)
        await asyncio.to_thread(engine.stop)
        self.assertEqual(facade.last_report["pool"]["num_free"], 8)

    async def test_preemption_real_snapshot_under_block_pressure_makes_progress(self):
        for device in devices():
            model, facade = fixture(device, num_blocks=2)
            prompt = [i % 32 for i in range(137)]
            expected = greedy_generate(model, prompt, max_new_tokens=3, eos_id=1)["generated_ids"]
            engine = self.engine(facade, enable_block_prefix_cache=False)
            reqs = [engine.submit(prompt, SamplingParams(temperature=0, top_k=1, max_tokens=3), str(i)) for i in range(2)]
            engine.start()
            results = await asyncio.wait_for(asyncio.gather(*(r.future for r in reqs)), timeout=45)
            self.assertEqual(results, [expected, expected])
            await asyncio.to_thread(engine.stop)
            self.assertGreater(engine._total_preemptions, 0)
            self.assertEqual(facade.last_report["calls"]["prefill"], 2)
            self.assertGreater(facade.last_report["calls"]["restore"], 0)
            self.assertEqual(facade.last_report["pool"]["num_free"], 2)

    async def test_invalid_or_impossible_requests_fail_without_queue_livelock(self):
        _, facade = fixture(num_blocks=1)
        engine = self.engine(facade)
        for ids, count in (([], 2), ([999], 2), ([1], 0), ([2] * 510, 4)):
            req = engine.submit(ids, SamplingParams(max_tokens=count), "invalid", stream=True)
            with self.assertRaises(ValueError):
                await req.future
            self.assertEqual([token async for token in req.stream_tokens()], [])
        impossible = engine.submit([2] * 137, SamplingParams(max_tokens=3), "impossible")
        engine.start()
        with self.assertRaisesRegex(ValueError, "total cache"):
            await asyncio.wait_for(impossible.future, timeout=30)
        self.assertEqual(engine.queue.waiting_count, 0)

    async def test_initialization_failure_completes_queued_request(self):
        _, facade = fixture(num_blocks=0)
        engine = self.engine(facade)
        request = engine.submit([3, 4], SamplingParams(max_tokens=2), "init", stream=True)
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            engine.start()
        with self.assertRaises(Exception):
            await asyncio.wait_for(request.future, timeout=10)
        self.assertEqual([token async for token in request.stream_tokens()], [])

    async def test_stop_keeps_live_handle_then_drains_active_and_waiting(self):
        entered, release = threading.Event(), threading.Event()
        def hold(*args):
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release worker")
        _, facade = fixture(observer=hold)
        engine = self.engine(facade)
        requests = [engine.submit([3, 4], SamplingParams(temperature=0, max_tokens=4), str(i), stream=True)
                    for i in range(3)]
        engine.start()
        self.assertTrue(await asyncio.to_thread(entered.wait, 10))
        try:
            with self.assertRaises(TimeoutError):
                engine.stop(timeout=0.001)
            self.assertTrue(engine._worker_thread.is_alive())
        finally:
            release.set()
        await asyncio.to_thread(engine.stop)
        for req in requests:
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                await req.future
            tokens = [token async for token in req.stream_tokens()]
            self.assertLessEqual(len(tokens), 1)
        self.assertEqual(facade.last_report["calls"]["decode"], 0)
        self.assertEqual(engine.queue.waiting_count + engine.queue.active_count, 0)

    async def test_client_future_cancel_propagates_and_no_invalid_state_callback(self):
        _, facade = fixture()
        engine = self.engine(facade)
        request = engine.submit([3, 4], SamplingParams(max_tokens=4), "client", stream=True)
        request.future.cancel()
        await asyncio.sleep(0)
        engine.start()
        self.assertEqual([token async for token in request.stream_tokens()], [])
        self.assertEqual(request.status, RequestStatus.CANCELLED)

    async def test_snapshot_failure_does_not_discard_running_history(self):
        model, facade = fixture(num_blocks=2)
        prompt = [i % 32 for i in range(137)]
        expected = greedy_generate(model, prompt, max_new_tokens=3, eos_id=1)["generated_ids"]
        engine = self.engine(facade)
        requests = [engine.submit(prompt, SamplingParams(temperature=0, max_tokens=3), str(i)) for i in range(2)]
        with patch("llaisys.models.deepseek_v4_model.batch.save_request", side_effect=RuntimeError("injected snapshot failure")):
            engine.start()
            actual = await asyncio.wait_for(asyncio.gather(*(r.future for r in requests)), timeout=30)
        self.assertEqual(actual, [expected, expected])
        self.assertEqual(engine._total_preemptions, 0)


if __name__ == "__main__":
    unittest.main()

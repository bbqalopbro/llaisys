"""CPU protocol/ownership tests; numerical acceptance uses real native weights."""
import asyncio
import copy
import threading
import unittest

from llaisys.models.deepseek_v4_model.native_batch import DeepSeekV4NativeServingModel
from server.engine import InferenceEngine, SamplingParams, RequestStatus


class RecordingSession:
    """Protocol double, explicitly not evidence of CUDA/model correctness."""
    def __init__(self, chunks=False):
        self.positions = [0, 0]
        self.plans = []
        self.chunks = chunks
        self.fail = False
        self.bad_response = False

    def info(self):
        return dict(vocabulary=100, capacity=32, slots=2, experimental_chunking=self.chunks,
                    active_slots=sum(p > 0 for p in self.positions), steps=len(self.plans), operators=[])

    def execute(self, plan, capture_logits=False):
        self.plans.append(copy.deepcopy(plan))
        for sid in plan.get("reset_slots", []):
            self.positions[sid] = 0
        rows = []
        for item in plan.get("prefills", []) + plan.get("decodes", []):
            sid = item["slot_id"]
            tokens = item.get("token_ids", [item.get("token_id")])
            self.positions[sid] += len(tokens)
            if self.fail:
                raise RuntimeError("injected native failure after first slot advanced")
            if item.get("is_last_chunk", True):
                token = (tokens[-1] + 1) % 100
                rows.append(dict(slot_id=sid, request_id=sid, token_id=token,
                                 logits=[float(token)] * 100 if capture_logits else None))
        return dict(step_id=plan["step_id"] + int(self.bad_response), outputs=rows)


class NativeBatchTests(unittest.TestCase):
    def fixture(self, **kwargs):
        session = RecordingSession(kwargs.pop("chunks", False))
        facade = DeepSeekV4NativeServingModel(session, eos_id=1, **kwargs)
        return session, facade, facade.create_batch_context(2, 32)

    def test_prefill_and_two_slot_decode_are_coarse_native_calls(self):
        session, facade, ctx = self.fixture()
        self.assertEqual(ctx.prefill(0, [3, 4]), 5)
        self.assertEqual(ctx.prefill(1, [20]), 21)
        self.assertEqual(ctx.decode_per_request([1, 0], [21, 5], [0, 0], [1, 1], [1, 1]), [22, 6])
        self.assertEqual(len(session.plans), 3)
        self.assertEqual(len(session.plans[-1]["decodes"]), 2)
        self.assertEqual([ctx.slot_get_pos(i) for i in range(2)], [3, 2])
        ctx.close()
        self.assertEqual(session.info()["active_slots"], 0)
        self.assertTrue(facade.last_report["cpp_model_execution"])
        self.assertFalse(facade.last_report["paged"])

    def test_invalid_decode_batch_is_rejected_before_any_native_call(self):
        session, _, ctx = self.fixture()
        ctx.prefill(0, [3])
        before = len(session.plans)
        for args in (([0, 1], [4, 2], [0, 0], [1, 1], [1, 1]),
                     ([0, 0], [4, 4], [0, 0], [1, 1], [1, 1]),
                     ([True], [4], [0], [1], [1]), ([0], [5], [0], [1], [1]),
                     ([0], [4], [0], [], [1])):
            with self.assertRaises(ValueError):
                ctx.decode_per_request(*args)
        self.assertEqual(len(session.plans), before)
        self.assertEqual(ctx.pending[0], 4)
        ctx.close()

    def test_sampling_token_and_capacity_validation_preserve_state(self):
        session, _, ctx = self.fixture()
        for tokens in ([], [-1], [100], [True], [1.0], [2] * 33):
            with self.assertRaises(ValueError):
                ctx.prefill(0, tokens)
        for sampling in ((.8, 50, .9), (-1, 1, 1), (True, 1, 1), (0, True, 1),
                         (float("nan"), 1, 1), (0, 1, float("inf")), (0, 1, 0)):
            with self.assertRaises(ValueError):
                ctx.prefill(0, [2], *sampling)
        self.assertEqual(session.plans, [])
        ctx.prefill(0, [2] * 32)
        with self.assertRaises(ValueError):
            ctx.decode_per_request([0], [3], [0], [1], [1])
        ctx.close()

    def test_single_context_thread_ownership_and_restart(self):
        _, facade, ctx = self.fixture()
        with self.assertRaises(RuntimeError):
            facade.create_batch_context(1, 32)
        errors = []
        def foreign():
            try:
                ctx.slot_reset(0)
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=foreign)
        thread.start(); thread.join()
        self.assertIn("owning worker", errors[0])
        ctx.close()
        with self.assertRaises(RuntimeError):
            ctx.get_block_size()
        facade.create_batch_context(1, 16).close()

    def test_explicit_incremental_prefill_phase_and_no_intermediate_token(self):
        _, _, ctx = self.fixture(chunks=True)
        self.assertIsNone(ctx.prefill_chunk(0, [2, 3], 0, False))
        with self.assertRaises(ValueError):
            ctx.prefill_chunk(0, [4], 1, True)
        self.assertEqual(ctx.prefill_chunk(0, [4], 2, True), 5)
        with self.assertRaises(ValueError):
            ctx.prefill_chunk(0, [5], 3, True)
        ctx.close()

    def test_unsupported_policies_fail_explicitly(self):
        _, facade, ctx = self.fixture()
        with self.assertRaises(RuntimeError):
            ctx.prefill_chunk(0, [2], 0, True)
        for method, args in ((ctx.slot_save, (0,)), (ctx.slot_restore, (0, object()))):
            with self.assertRaises(NotImplementedError):
                method(*args)
        ctx.close()
        for options in (dict(enable_block_prefix_cache=True, prefill_chunk_size=32),
                        dict(enable_block_prefix_cache=False, prefill_chunk_size=16)):
            with self.assertRaises(ValueError):
                InferenceEngine(facade, max_batch_size=2, max_seq_per_slot=32, **options)

    def test_native_partial_failure_discards_all_touched_slots(self):
        session, _, ctx = self.fixture()
        ctx.prefill(0, [3]); ctx.prefill(1, [4])
        session.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            ctx.decode_per_request([0, 1], [4, 5], [0, 0], [1, 1], [1, 1])
        self.assertEqual(session.positions, [0, 0])
        self.assertEqual(ctx.pending, [None, None])
        session.fail = False
        self.assertEqual(ctx.prefill(1, [7]), 8)
        ctx.close()

    def test_observer_failure_and_bad_response_reset_advanced_state(self):
        for bad_response in (False, True):
            def observe(*args):
                raise RuntimeError("observer failed")
            session, _, ctx = self.fixture(observer=None if bad_response else observe)
            session.bad_response = bad_response
            with self.assertRaises(RuntimeError):
                ctx.prefill(0, [3])
            self.assertEqual(session.positions, [0, 0])
            session.bad_response = False
            ctx.close()


class NativeEngineTests(unittest.IsolatedAsyncioTestCase):
    def engine(self, *, observer=None):
        session = RecordingSession()
        facade = DeepSeekV4NativeServingModel(session, eos_id=1, observer=observer)
        engine = InferenceEngine(facade, max_batch_size=2, max_seq_per_slot=32,
                                 prefill_chunk_size=32, enable_block_prefix_cache=False)
        self.addAsyncCleanup(asyncio.to_thread, engine.stop)
        return session, facade, engine

    async def test_different_requests_stream_eos_length_and_queue_reuse(self):
        session, facade, engine = self.engine()
        reqs = [engine.submit([token], SamplingParams(temperature=0, top_k=1, max_tokens=3), str(token), stream=True)
                for token in (3, 20, 0, 40)]
        engine.start()
        actual = await asyncio.wait_for(asyncio.gather(*(r.future for r in reqs)), 10)
        self.assertEqual(actual, [[4, 5, 6], [21, 22, 23], [1], [41, 42, 43]])
        self.assertEqual([[t async for t in r.stream_tokens()] for r in reqs], actual)
        await asyncio.to_thread(engine.stop)
        self.assertEqual(session.info()["active_slots"], 0)
        self.assertEqual(facade.last_report["max_decode_batch"], 2)
        self.assertEqual(engine.get_stats()["total_preemptions"], 0)

    async def test_invalid_request_and_waiting_cancel_do_not_execute(self):
        session, _, engine = self.engine()
        invalid = engine.submit([2], SamplingParams(), "sampling", stream=True)
        with self.assertRaises(ValueError):
            await invalid.future
        cancelled = engine.submit([3], SamplingParams(temperature=0, max_tokens=3), "cancel", stream=True)
        self.assertTrue(engine.cancel(cancelled.request_id))
        engine.start()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled.future
        self.assertEqual([t async for t in cancelled.stream_tokens()], [])
        await asyncio.to_thread(engine.stop)
        self.assertTrue(all(not p.get("prefills") and not p.get("decodes") for p in session.plans))

    async def test_active_cancel_and_subsequent_request_recover_slot(self):
        entered, release = threading.Event(), threading.Event()
        def observe(*args):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test failed to release observer")
        session, facade, engine = self.engine(observer=observe)
        request = engine.submit([3], SamplingParams(temperature=0, max_tokens=3), "cancel", stream=True)
        engine.start()
        self.assertTrue(await asyncio.to_thread(entered.wait, 10))
        try:
            self.assertTrue(engine.cancel(request.request_id))
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await request.future
        self.assertEqual(request.status, RequestStatus.CANCELLED)
        recovered = engine.submit([20], SamplingParams(temperature=0, max_tokens=2), "recovered")
        self.assertEqual(await asyncio.wait_for(recovered.future, 10), [21, 22])
        await asyncio.to_thread(engine.stop)
        self.assertEqual(session.info()["active_slots"], 0)
        self.assertTrue(facade.last_report["logits_observer"])

    async def test_observer_exception_errors_request_and_returns_slot(self):
        count = [0]
        def observe(*args):
            count[0] += 1
            if count[0] == 1:
                raise RuntimeError("injected observer failure")
        session, _, engine = self.engine(observer=observe)
        failed = engine.submit([3], SamplingParams(temperature=0, max_tokens=2), "bad", stream=True)
        good = engine.submit([20], SamplingParams(temperature=0, max_tokens=2), "good")
        engine.start()
        with self.assertRaisesRegex(RuntimeError, "observer"):
            await failed.future
        self.assertEqual(await good.future, [21, 22])
        self.assertEqual([t async for t in failed.stream_tokens()], [])
        await asyncio.to_thread(engine.stop)
        self.assertEqual(session.info()["active_slots"], 0)


if __name__ == "__main__":
    unittest.main()

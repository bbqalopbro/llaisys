"""CPU-only benchmark contracts; doubles do not prove GPU/model correctness."""
import asyncio
import copy
import importlib.util
import io
import itertools
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


_path = Path(__file__).resolve().parents[1] / "tools/deepseek_v4_reference/bench_native_serving.py"
_spec = importlib.util.spec_from_file_location("llaisys_test_native_benchmark", _path)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class SummaryTests(unittest.TestCase):
    def test_empty_singleton_and_linear_interpolation(self):
        empty = bench.summarize([])
        self.assertEqual(empty["count"], 0)
        self.assertTrue(all(value is None for key, value in empty.items() if key != "count"))
        single = bench.summarize([7])
        self.assertEqual(single, dict(count=1, mean=7, p50=7, p95=7, p99=7, min=7, max=7))
        values = [30, 0, 20, 10]
        result = bench.summarize(values)
        self.assertEqual(values, [30, 0, 20, 10])
        self.assertEqual(result["mean"], 15)
        self.assertEqual(result["p50"], 15)
        self.assertAlmostEqual(result["p95"], 28.5)
        self.assertAlmostEqual(result["p99"], 29.7)
        self.assertEqual((result["min"], result["max"]), (0, 30))


class MemorySamplerTests(unittest.TestCase):
    def sampler(self, samples=()):
        # Do not invoke __init__: no nvidia-smi process or GPU is started.
        sampler = bench.MemorySampler.__new__(bench.MemorySampler)
        sampler.samples, sampler.errors = list(samples), []
        sampler.stopping = False
        sampler.ready = threading.Event()
        sampler.process = Mock()
        sampler.process.stdout = io.StringIO("100, 1000\n")
        sampler.process.stderr = io.StringIO()
        sampler.process.poll.return_value = None
        sampler.thread = Mock()
        sampler.thread.is_alive.return_value = True
        return sampler

    def test_unexpected_eof_after_valid_sample_is_an_error(self):
        sampler = self.sampler()
        sampler._read()
        self.assertEqual(len(sampler.samples), 1)
        self.assertEqual(sampler.samples[0][1:], (100 * 1024**2, 1000 * 1024**2))
        self.assertTrue(sampler.ready.is_set())
        self.assertTrue(any("before explicit shutdown" in error for error in sampler.errors))
        now = sampler.samples[0][0]
        self.assertFalse(sampler.report(now - .1, now + .1)["coverage_valid"])

    def test_explicit_close_marks_shutdown_before_reader_eof(self):
        sampler = self.sampler()
        def terminate():
            self.assertTrue(sampler.stopping)
            sampler._read()
        sampler.process.terminate.side_effect = terminate
        sampler.close()
        self.assertEqual(sampler.errors, [])
        sampler.process.terminate.assert_called_once_with()
        sampler.process.wait.assert_called_once_with(5)
        sampler.thread.join.assert_called_once_with(5)
        self.assertTrue(sampler.process.stdout.closed)
        self.assertTrue(sampler.process.stderr.closed)

    def test_complete_window_reports_sampled_not_allocator_peak(self):
        sampler = self.sampler([(10.1, 100, 1000), (10.4, 300, 1000), (10.8, 200, 1000)])
        result = sampler.report(10, 11)
        self.assertTrue(result["coverage_valid"])
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["peak_used_bytes"], 300)
        self.assertEqual(result["total_bytes"], 1000)
        self.assertAlmostEqual(result["first_sample_delay_ms"], 100)
        self.assertAlmostEqual(result["last_sample_age_ms"], 200)
        self.assertAlmostEqual(result["maximum_gap_ms"], 400)
        self.assertFalse(result["exact_allocator_peak"])
        self.assertTrue(result["includes_other_gpu_processes"])

    def test_empty_edge_and_internal_gaps_reject_coverage(self):
        cases = {
            "empty": [], "outside": [9.9, 11.1], "late_start": [10.6, 10.9],
            "early_end": [10.1, 10.4], "internal_gap": [10.1, 10.8, 10.9],
        }
        for name, timestamps in cases.items():
            with self.subTest(name=name):
                sampler = self.sampler([(t, 100, 1000) for t in timestamps])
                self.assertFalse(sampler.report(10, 11)["coverage_valid"])

    def test_dead_sampler_or_error_rejects_otherwise_complete_window(self):
        for failure in ("thread", "process", "error"):
            with self.subTest(failure=failure):
                sampler = self.sampler([(10.1, 100, 1000), (10.5, 100, 1000), (10.9, 100, 1000)])
                if failure == "thread": sampler.thread.is_alive.return_value = False
                elif failure == "process": sampler.process.poll.return_value = 1
                else: sampler.errors.append("malformed sample")
                self.assertFalse(sampler.report(10, 11)["coverage_valid"])


class StreamRequest:
    def __init__(self, tokens, result=None):
        self.tokens = tokens
        self.future = asyncio.get_running_loop().create_future()
        self.future.set_result(tokens if result is None else result)

    async def stream_tokens(self):
        for token in self.tokens:
            yield token


class CohortTests(unittest.IsolatedAsyncioTestCase):
    async def run_cohort(self, tokens, budget, result=None):
        engine = Mock()
        engine._eos_token_id = 9
        engine.submit.return_value = StreamRequest(tokens, result)
        # Patch the benchmark's clock object, not time.perf_counter globally:
        # asyncio's own timeout clock must keep its real monotonic behavior.
        clock = SimpleNamespace(perf_counter=Mock(side_effect=itertools.count(10, .01)))
        with patch.object(bench, "time", clock):
            report = await bench.cohort(engine, [[1, 2, 3]], budget)
        params = engine.submit.call_args.args[1]
        self.assertEqual((params.temperature, params.top_k, params.max_tokens), (0, 1, budget))
        self.assertTrue(engine.submit.call_args.kwargs["stream"])
        return report

    async def test_eos_records_actual_length_and_not_requested_budget(self):
        report = await self.run_cohort([7, 9], 16)
        row = report["requests"][0]
        self.assertEqual((row["output_tokens"], row["finish_reason"]), (2, "stop"))
        self.assertAlmostEqual(row["ttft_ms"], 10)
        self.assertAlmostEqual(row["tpot_ms"], 10)
        self.assertAlmostEqual(row["decode_tokens_per_second"], 100)
        self.assertAlmostEqual(report["output_tokens_per_second"], 50)

    async def test_single_eos_has_no_decode_statistic(self):
        row = (await self.run_cohort([9], 16))["requests"][0]
        self.assertEqual(row["output_tokens"], 1)
        self.assertEqual(row["finish_reason"], "stop")
        self.assertIsNone(row["tpot_ms"])
        self.assertIsNone(row["decode_tokens_per_second"])
        self.assertEqual(row["inter_token_ms"], [])

    async def test_budget_exhaustion_records_length(self):
        row = (await self.run_cohort([6, 7], 2))["requests"][0]
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["generated_ids"], [6, 7])

    async def test_empty_mismatch_over_budget_and_premature_non_eos_rejected(self):
        for tokens, budget, result in (([], 2, None), ([7, 9], 2, [7, 8]),
                                       ([6, 7, 9], 2, None), ([7], 2, None)):
            with self.subTest(tokens=tokens, budget=budget, result=result):
                with self.assertRaisesRegex(RuntimeError, "results disagree"):
                    await self.run_cohort(tokens, budget, result)


class MeasurementCleanupTests(unittest.IsolatedAsyncioTestCase):
    def runtime_report(self):
        return dict(fallback=False, logits_observer=False,
                    runtime=dict(active_slots=0, model_failures=0,
                                 operators=[dict(failures=0, fallback=False)]))

    def trial(self):
        return dict(requests=[dict(generated_ids=[7, 9], ttft_ms=10, tpot_ms=20,
            inter_token_ms=[20], effective_prompt_tokens_per_second=100,
            decode_tokens_per_second=50)], output_tokens_per_second=60, requests_per_second=30)

    async def run_measure(self, runtime, *, startup_error=None, cleanup_error=None):
        facade = SimpleNamespace(last_report=None)
        engine, session, sampler = Mock(), Mock(), Mock()
        engine._worker_error = None
        engine.start.side_effect = startup_error
        def stop(timeout):
            facade.last_report = runtime
            engine._worker_error = cleanup_error
        engine.stop.side_effect = stop
        session.info.return_value = dict(active_slots=0)
        sampler.report.return_value = dict(coverage_valid=True)
        self.engine, self.cohorts = engine, AsyncMock(return_value=self.trial())
        with patch.object(bench, "DeepSeekV4NativeServingModel", return_value=facade), \
             patch.object(bench, "InferenceEngine", return_value=engine), \
             patch.object(bench, "cohort", self.cohorts):
            return await bench.measure(session, 9, 32, [[1]], 2, 1, 2, sampler)

    async def test_startup_failure_still_stops_worker(self):
        with self.assertRaisesRegex(TimeoutError, "startup"):
            await self.run_measure(self.runtime_report(), startup_error=TimeoutError("startup"))
        self.engine.stop.assert_called_once_with(120)
        self.cohorts.assert_not_awaited()

    async def test_cleanup_error_is_not_reported_as_success(self):
        failure = RuntimeError("injected reset failure")
        with self.assertRaisesRegex(RuntimeError, "worker cleanup") as raised:
            await self.run_measure(self.runtime_report(), cleanup_error=failure)
        self.assertIs(raised.exception.__cause__, failure)
        self.engine.stop.assert_called_once_with(120)

    async def test_missing_report_and_failure_accounting_rejected(self):
        cases = [("missing report", None)]
        for key in ("fallback", "logits_observer"):
            report = self.runtime_report(); report[key] = True
            cases.append((key, report))
        for key in ("active_slots", "model_failures"):
            report = self.runtime_report(); report["runtime"][key] = 1
            cases.append((key, report))
        for key in ("failures", "fallback"):
            report = self.runtime_report(); report["runtime"]["operators"][0][key] = 1
            cases.append(("operator " + key, report))
        for name, report in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "accounting invalid"):
                    await self.run_measure(copy.deepcopy(report))
                self.engine.stop.assert_called_once_with(120)

    async def test_success_includes_only_measured_trials_and_closed_runtime(self):
        runtime = self.runtime_report()
        report = await self.run_measure(runtime)
        self.assertEqual(self.cohorts.await_count, 3)
        self.assertEqual(len(report["raw_trials"]), 2)
        self.assertEqual(report["ttft_ms"]["count"], 2)
        self.assertEqual(report["runtime"], runtime)
        self.assertTrue(report["repeated_greedy_ids_equal"])
        self.engine.stop.assert_called_once_with(120)


if __name__ == "__main__":
    unittest.main()

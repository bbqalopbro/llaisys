import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from llaisys.models.deepseek_v4_model.chat import ChatCodec, check_chat_answer
from llaisys.models.deepseek_v4_model.generation import greedy_generate


SOURCE = Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731")


class ChatEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True, trust_remote_code=False)
        cls.codec = ChatCodec(SOURCE, cls.tokenizer)
        spec = importlib.util.spec_from_file_location("published_encoding_test", SOURCE / "encoding/encoding_dsv4.py")
        cls.published = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.published)

    def test_exact_basic_format_and_single_bos(self):
        messages = [{"role": "system", "content": "简短回答。"}, {"role": "user", "content": "你好"}]
        encoded = self.codec.encode(messages)
        self.assertEqual(encoded["text"], "<｜begin▁of▁sentence｜>简短回答。<｜User｜>你好<｜Assistant｜></think>")
        self.assertEqual(encoded["input_ids"], self.tokenizer.encode(encoded["text"]))
        self.assertEqual(encoded["input_ids"].count(self.tokenizer.bos_token_id), 1)

    def test_multiturn_thinking_effort_matches_published_without_mutation(self):
        messages = [{"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B", "reasoning_content": "old reasoning"},
                    {"role": "user", "content": "C"}]
        original = copy.deepcopy(messages)
        for mode in ("chat", "thinking"):
            for drop in (False, True):
                for effort in ("low", "high", "max"):
                    options = dict(thinking_mode=mode, drop_thinking=drop, reasoning_effort=effort)
                    actual = self.codec.encode(messages, **options)
                    expected = self.published.encode_messages(messages, **options)
                    self.assertEqual(actual["text"], expected)
                    self.assertEqual(actual["input_ids"], self.tokenizer.encode(expected))
        self.assertEqual(messages, original)

    def test_tool_results_sorted_like_published_and_never_executed(self):
        messages = [{"role": "system", "tools": [{"type": "function", "function": {
                        "name": "lookup", "parameters": {"type": "object"}}}]},
                    {"role": "user", "content": "Compare A and B"},
                    {"role": "assistant", "tool_calls": [
                        {"id": name, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"key": name})}}
                        for name in ("a", "b")]},
                    {"role": "tool", "tool_call_id": "b", "content": "SECOND"},
                    {"role": "tool", "tool_call_id": "a", "content": "FIRST"}]
        original = copy.deepcopy(messages)
        actual = self.codec.encode(messages, thinking_mode="thinking")
        self.assertEqual(actual["text"], self.published.encode_messages(messages, thinking_mode="thinking"))
        self.assertLess(actual["text"].index("<tool_result>FIRST"), actual["text"].index("<tool_result>SECOND"))
        self.assertEqual(messages, original)
        self.assertFalse(self.codec.report()["tool_execution"])

    def test_decode_complete_truncated_and_malformed_without_repair(self):
        eos = self.published.eos_token
        def decode(text, mode="chat", reason="stop"):
            return self.codec.decode_completion(self.tokenizer.encode(text, add_special_tokens=False),
                                                thinking_mode=mode, finish_reason=reason)
        self.assertEqual(decode("答案" + eos)["message"]["content"], "答案")
        thinking = decode("arithmetic</think>391" + eos, "thinking")
        self.assertEqual(thinking["message"]["content"], "391")
        self.assertTrue(check_chat_answer(thinking, {"expected_exact": "391"}))
        partial = decode("unfinished", reason="length")
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["message"])
        self.assertEqual(partial["raw_text"], "unfinished")
        malformed = decode("missing closing marker" + eos, "thinking")
        self.assertFalse(malformed["complete"])
        self.assertIsNotNone(malformed["parse_error"])
        with self.assertRaises(ValueError):
            decode("missing EOS")
        with self.assertRaises(ValueError):
            decode("x" + eos, reason="length")

    def test_tools_parse_matches_reference(self):
        text = '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n<｜DSML｜parameter name="key" string="true">a</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>'
        actual = self.codec.decode_completion(self.tokenizer.encode(text, add_special_tokens=False), finish_reason="stop")
        self.assertTrue(actual["complete"])
        self.assertEqual(actual["message"], self.published.parse_message_from_completion_text(text, thinking_mode="chat"))

    def test_invalid_inputs_and_tokenizer_mismatch_fail_explicitly(self):
        for messages in ([], "hello", [{"role": "unknown"}], [{"role": "user", "content": ["image"]}],
                         [{"role": "user", "content_blocks": []}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.codec.encode(messages)
        for options in ({"thinking_mode": "auto"}, {"drop_thinking": 1}, {"reasoning_effort": "medium"}):
            with self.assertRaises(ValueError):
                self.codec.encode([{"role": "user", "content": "x"}], **options)
        incompatible = types.SimpleNamespace(eos_token_id=3, bos_token_id=self.tokenizer.bos_token_id,
                                             encode=self.tokenizer.encode)
        with self.assertRaises(ValueError):
            ChatCodec(SOURCE, incompatible)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            ChatCodec(directory, self.tokenizer)

    def test_baseline_rechecks_messages_encoder_tokenizer_and_completion(self):
        from test_deepseek_v4_independent_runner import RUNNER
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "encoding").mkdir()
            (source / "inference").mkdir()
            (source / "encoding/encoding_dsv4.py").write_bytes((SOURCE / "encoding/encoding_dsv4.py").read_bytes())
            paths = [source / "inference" / name for name in ("config.json", "model.py", "kernel.py")]
            for path in paths:
                path.write_text("fixture")
            codec = ChatCodec(source, self.tokenizer)
            messages = [{"role": "user", "content": "Reply OK"}]
            encoded = codec.encode(messages)
            generated = self.tokenizer.encode("OK", add_special_tokens=False) + [codec.eos_id]
            logits = torch.zeros(len(generated), 1, len(self.tokenizer))
            for i, token in enumerate(generated):
                logits[i, 0, token] = 1
            save_file({"input_ids": torch.tensor([encoded["input_ids"]]),
                       "generated_ids": torch.tensor(generated), "logits": logits}, str(source / "case_0.safetensors"))
            case = {"prompt": encoded["text"], "input_format": "chat", "stop_on_eos": True,
                    "finish_reason": "stop", "generated_ids": generated,
                    "chat": {"messages": messages, "thinking_mode": "chat", "reasoning_effort": "low",
                             "drop_thinking": True, "codec": codec.report()},
                    "completion": codec.decode_completion(generated, finish_reason="stop")}
            original = {"backend": "published-tilelang", "source_model": str(source), "converted_model": str(source),
                        "measurement": {"fallback": False}, "provenance": {"command": ["--max-seq-len", "64"],
                        "file_sha256": {str(path): RUNNER.sha256(path) for path in paths}}, "cases": [case]}
            report = source / "report.json"
            def read():
                return RUNNER.read_baselines([(report, source)], source, source / "model0-mp1.safetensors",
                                             64, self.tokenizer, len(self.tokenizer))
            report.write_text(json.dumps(original))
            self.assertEqual(read()[0]["chat"]["messages"], messages)
            for corruption in ("messages", "encoding", "tokenizer", "metadata", "completion"):
                changed = copy.deepcopy(original)
                row = changed["cases"][0]
                if corruption == "messages":
                    row["chat"]["messages"][0]["content"] = "different"
                elif corruption in ("encoding", "tokenizer"):
                    key = "encoding_sha256" if corruption == "encoding" else "tokenizer_backend_sha256"
                    row["chat"]["codec"][key] = "0" * 64
                elif corruption == "metadata":
                    row["chat"] = None
                else:
                    row["completion"]["message"]["content"] = "wrong"
                report.write_text(json.dumps(changed))
                with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                    read()


class FakeState:
    def __init__(self):
        self.position, self.closed = 0, False

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self, sequence=(4, 5, 1)):
        self.config = types.SimpleNamespace(max_seq_len=16, vocab_size=8)
        self.embed = types.SimpleNamespace(weight=torch.empty(1))
        self.sequence, self.calls, self.states, self.emitted = sequence, [], [], 0

    def new_request(self, **kwargs):
        self.emitted = 0
        state = FakeState()
        self.states.append(state)
        return state

    def __call__(self, ids, state, emit_logits=True):
        self.calls.append((state.position, ids.tolist(), emit_logits))
        state.position += ids.shape[1]
        logits = None
        if emit_logits:
            logits = torch.zeros((1, 8))
            logits[0, self.sequence[min(self.emitted, len(self.sequence)-1)]] = 2
            self.emitted += 1
        return types.SimpleNamespace(logits=logits)


class GreedyGenerationTests(unittest.TestCase):
    def run_model(self, model, **kwargs):
        return greedy_generate(model, [0, 2, 3, 6, 7], max_new_tokens=kwargs.pop("max_new_tokens", 8),
                               eos_id=1, **kwargs)

    def test_eos_no_extra_decode_and_actual_feedback(self):
        model = FakeModel()
        result = self.run_model(model, retain_logits=True)
        self.assertEqual(result["generated_ids"], [4, 5, 1])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(model.calls, [(0, [[0, 2, 3, 6, 7]], True), (5, [[4]], True), (6, [[5]], True)])
        self.assertEqual(result["logits"].shape, (3, 1, 8))
        self.assertTrue(result["request_closed"])

    def test_chunk_skips_intermediate_heads_and_covers_tail(self):
        model = FakeModel()
        result = self.run_model(model, chunk_size=2)
        self.assertEqual(model.calls[:3], [(0, [[0, 2]], False), (2, [[3, 6]], False), (4, [[7]], True)])
        self.assertEqual(result["generated_ids"], [4, 5, 1])

    def test_max_tokens_context_capacity_and_first_token_eos(self):
        for limit, reason, budget in ((2, "max_new_tokens", 2), (20, "context_length", 11)):
            result = self.run_model(FakeModel((4,)), max_new_tokens=limit)
            self.assertEqual(result["output_tokens"], budget)
            self.assertEqual(result["stop_reason"], reason)
            self.assertEqual(result["finish_reason"], "length")
        model = FakeModel((1,))
        self.assertEqual(self.run_model(model)["generated_ids"], [1])
        self.assertEqual(len(model.calls), 1)

    def test_cancellation_before_prefill_and_after_token_releases_state(self):
        model = FakeModel()
        result = self.run_model(model, cancelled=lambda: True)
        self.assertEqual(result["generated_ids"], [])
        self.assertEqual(result["finish_reason"], "cancelled")
        self.assertEqual(model.calls, [])
        self.assertTrue(model.states[-1].closed)
        seen = []
        result = self.run_model(model, cancelled=lambda: bool(seen), on_token=seen.append)
        self.assertEqual(result["generated_ids"], [4])
        self.assertEqual(result["finish_reason"], "cancelled")
        self.assertTrue(model.states[-1].closed)

    def test_execution_and_callback_errors_release_state(self):
        def fail(*args, **kwargs):
            raise RuntimeError("injected")
        model = FakeModel()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_model(model, on_token=fail)
        self.assertTrue(model.states[-1].closed)
        with patch.object(FakeModel, "__call__", fail), self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_model(model)
        self.assertTrue(model.states[-1].closed)

    def test_invalid_inputs_do_not_allocate_state(self):
        model = FakeModel()
        for ids in ([], [True], [-1], [8], [0] * 16):
            with self.assertRaises(ValueError):
                greedy_generate(model, ids, max_new_tokens=2, eos_id=1)
        for options in ({"max_new_tokens": 0}, {"chunk_size": -1}, {"max_new_tokens": True}):
            with self.assertRaises(ValueError):
                self.run_model(model, **options)
        self.assertEqual(model.states, [])

    def test_real_small_model_generation_matches_manual_and_releases_cpp_blocks(self):
        from test_deepseek_v4_model import initialized_model
        from llaisys.models.deepseek_v4_model.paged import PagedCachePool
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = initialized_model()
            pool = PagedCachePool(model, num_blocks=2, block_size=128)
            ids = [1, 2, 3, 4, 5]
            state = model.new_request(cache_pool=pool)
            expected = []
            with torch.inference_mode():
                current = torch.tensor([ids])
                for _ in range(4):
                    token = model(current, state).logits.argmax(-1).item()
                    expected.append(token)
                    if token == 0:
                        break
                    current = torch.tensor([[token]])
            state.close()
            result = greedy_generate(model, ids, max_new_tokens=4, eos_id=0, cache_pool=pool)
            self.assertEqual(result["generated_ids"], expected)
            self.assertEqual(pool.report()["num_free"], 2)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()

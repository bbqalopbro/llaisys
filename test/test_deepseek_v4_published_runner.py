import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("published_runner_test", ROOT / "tools/deepseek_v4_reference/run_single_gpu.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Tokenizer:
    eos_token_id = 2
    def encode(self, prompt, return_tensors=None):
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids):
        return "fixture"


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs, position):
        self.calls.append((position, inputs.shape[1]))
        return torch.tensor([2]), torch.tensor([[0., 1., 3., 2.]]), None


class PublishedRunnerModesTests(unittest.TestCase):
    def evaluate(self, *, replay, dump=None):
        model = Model()
        # CPU-only harness for orchestration; this makes no kernel/GPU claims.
        with patch.object(torch.Tensor, "cuda", lambda value: value), patch("torch.cuda.synchronize"):
            report = RUNNER._evaluate_case(model, Tokenizer(), "fixture", 2, 8,
                                           dump_logits=dump, token_replay=replay)
        return model, report

    def test_full_only_does_not_claim_token_replay_or_its_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.safetensors"
            model, report = self.evaluate(replay=False, dump=path)
            self.assertEqual(model.calls, [(0, 3), (0, 3), (3, 1)])
            self.assertFalse(report["token_replay_tested"])
            self.assertIsNone(report["accuracy"])
            data = load_file(str(path))
            self.assertEqual(data["logits"].shape, (2, 1, 4))
            self.assertEqual(data["generated_ids"].tolist(), [2, 2])
            self.assertEqual([step["input_tokens"] for step in report["steps"]], [3, 1])

    def test_full_and_replay_keeps_the_existing_check(self):
        model, report = self.evaluate(replay=True)
        self.assertEqual(model.calls, [(0, 3), (0, 1), (1, 1), (2, 1), (0, 3), (3, 1)])
        self.assertTrue(report["token_replay_tested"])
        self.assertTrue(report["accuracy"]["argmax_equal"])
        self.assertEqual(report["accuracy"]["logits_max_abs_error"], 0)

    def test_chat_eos_stops_without_appending_synthetic_tokens(self):
        model = Model()
        with patch.object(torch.Tensor, "cuda", lambda value: value), patch("torch.cuda.synchronize"):
            report = RUNNER._evaluate_case(model, Tokenizer(), "encoded", 4, 8,
                                           token_replay=False, encoded_ids=[0, 3], stop_on_eos=True)
        self.assertEqual(report["generated_ids"], [2])
        self.assertEqual(report["finish_reason"], "stop")
        self.assertEqual(model.calls, [(0, 2), (0, 2)])

    def test_chat_length_does_not_fake_eos(self):
        model = Model()
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = 1
        with patch.object(torch.Tensor, "cuda", lambda value: value), patch("torch.cuda.synchronize"):
            report = RUNNER._evaluate_case(model, tokenizer, "encoded", 2, 8,
                                           token_replay=False, encoded_ids=[0, 3], stop_on_eos=True)
        self.assertEqual(report["generated_ids"], [2, 2])
        self.assertEqual(report["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file
from llaisys.models.deepseek_v4_evidence import checkpoint_identity, require_same_checkpoint, verify_checkpoint_unchanged


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("independent_runner_test", ROOT / "tools/deepseek_v4_reference/run_independent.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TokenizerFixture:
    vocab_size = 2  # Added tokens must not be omitted from the model output check.

    def __len__(self):
        return 4

    def encode(self, prompt, return_tensors=None):
        return torch.tensor([[1, 2]])


class IndependentRunnerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)
        self.checkpoint = self.source / "model0-mp1.safetensors"
        self.report = self.source / "baseline.json"
        paths = [self.source / "inference" / name for name in ("kernel.py", "config.json", "model.py")]
        paths[0].parent.mkdir()
        for path in paths:
            path.write_text("fixture")
        self.report.write_text(json.dumps({
            "backend": "published-tilelang", "measurement": {"fallback": False},
            "source_model": str(self.source), "converted_model": str(self.source),
            "provenance": {"command": ["runner", "--max-seq-len", "8"],
                           "file_sha256": {str(path): RUNNER.sha256(path) for path in paths}},
            "cases": [{"prompt": "fixture", "generated_ids": [2, 3]}],
        }))
        self.golden = self.source / "case_0.safetensors"
        save_file({"input_ids": torch.tensor([[1, 2]]), "generated_ids": torch.tensor([2, 3]),
                   "logits": torch.tensor([[[0., 1., 3., 2.]], [[0., 1., 2., 3.]]])}, str(self.golden))

    def read(self, **kwargs):
        return RUNNER.read_baselines([(str(self.report), str(self.source))], self.source,
                                     self.checkpoint, kwargs.get("max_seq_len", 8), TokenizerFixture(),
                                     kwargs.get("model_vocab_size", 4), kwargs.get("weight_identity"),
                                     kwargs.get("reference_profile", "published"))

    def test_numerical_profiles_cannot_be_mixed_or_relabelled(self):
        with self.assertRaises(ValueError):
            self.read(reference_profile="fixed32-reference-v1")
        report = json.loads(self.report.read_text())
        report["backend"] = "published-tilelang-fixed32-reference-v1"
        report["reference_profile"] = "fixed32-reference-v1"
        self.report.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            self.read()
        from test_deepseek_v4_fixed_profile import PROFILE, SOURCE
        actual_source = self.source / "inference/model.py"
        actual_source.write_bytes(SOURCE.read_bytes())
        report["provenance"]["file_sha256"][str(actual_source)] = RUNNER.sha256(actual_source)
        report["profile_adapter"] = PROFILE.prepare_source(actual_source)[1]
        self.report.write_text(json.dumps(report))
        cases = self.read(reference_profile="fixed32-reference-v1")
        self.assertEqual(cases[0]["reference_profile"], "fixed32-reference-v1")
        report["profile_adapter"]["adapter_sha256"] = "0" * 64
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "profile adapter"):
            self.read(reference_profile="fixed32-reference-v1")

    def test_payload_identity_rejects_same_size_changed_checkpoint(self):
        self.checkpoint.write_bytes(b"original payload")
        identity = checkpoint_identity(self.checkpoint, chunk_bytes=3)
        report = json.loads(self.report.read_text())
        report["checkpoint_identity"] = identity
        report["golden_file_sha256"] = {self.golden.name: RUNNER.sha256(self.golden)}
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "requires checkpoint payload"):
            self.read()
        cases = self.read(weight_identity=checkpoint_identity(self.checkpoint))
        self.assertTrue(cases[0]["checkpoint_payload_verified"])
        self.checkpoint.write_bytes(b"modified payload")
        # Same-size writes may share one filesystem timestamp tick. Make the
        # metadata change explicit for this separate end-of-run signature test;
        # the full-payload hash assertion below independently detects the bytes.
        os.utime(self.checkpoint, ns=(self.checkpoint.stat().st_atime_ns, identity["signature"]["st_mtime_ns"] + 10**9))
        self.assertEqual(self.checkpoint.stat().st_size, identity["size_bytes"])
        with self.assertRaisesRegex(ValueError, "payload identity differs"):
            self.read(weight_identity=checkpoint_identity(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            verify_checkpoint_unchanged(self.checkpoint, identity)

    def test_golden_hash_rejects_changed_nonwinning_logit(self):
        report = json.loads(self.report.read_text())
        report["golden_file_sha256"] = {self.golden.name: RUNNER.sha256(self.golden)}
        self.report.write_text(json.dumps(report))
        data = load_file(str(self.golden))
        data["logits"][0, 0, 0] += 0.125  # Same input, shape, generated IDs and argmax.
        save_file(data, str(self.golden))
        with self.assertRaisesRegex(ValueError, "golden artifact hash"):
            self.read()

    def test_payload_hash_is_streamed_and_copies_match(self):
        payload = b"checkpoint tensor bytes" * 131
        self.checkpoint.write_bytes(payload)
        identity = checkpoint_identity(self.checkpoint, chunk_bytes=7)
        self.assertEqual(identity["sha256"], RUNNER.sha256(self.checkpoint))
        verify_checkpoint_unchanged(self.checkpoint, identity)
        copy = self.source / "copy.safetensors"
        copy.write_bytes(payload)
        require_same_checkpoint(checkpoint_identity(copy), identity)
        with self.assertRaisesRegex(ValueError, "chunk size"):
            checkpoint_identity(copy, chunk_bytes=0)
        with self.assertRaisesRegex(ValueError, "invalid checkpoint"):
            require_same_checkpoint(identity, {**identity, "sha256": None})

    def test_accepts_added_token_vocabulary_and_records_artifact_hashes(self):
        cases = self.read()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["golden_sha256"], RUNNER.sha256(self.golden))
        self.assertEqual(cases[0]["baseline_report_sha256"], RUNNER.sha256(self.report))

    def test_rejects_changed_source_or_sequence_configuration(self):
        with self.assertRaisesRegex(ValueError, "cache/RoPE"):
            self.read(max_seq_len=16)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.read(model_vocab_size=2)
        (self.source / "inference/kernel.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.read()

    def test_rejects_nonfinite_logits_and_mismatched_history(self):
        original = load_file(str(self.golden))
        for corrupt in ("nan", "input", "generated", "shape"):
            data = {key: value.clone() for key, value in original.items()}
            if corrupt == "nan":
                data["logits"][0, 0, 0] = torch.nan
            elif corrupt == "input":
                data["input_ids"][0, 0] = 0
            elif corrupt == "generated":
                data["generated_ids"][0] = 0
            else:
                data["logits"] = data["logits"][..., :2].contiguous()
            save_file(data, str(self.golden))
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, "golden tensors"):
                self.read()


if __name__ == "__main__":
    unittest.main()

"""Independent published-control-flow oracle for an explicit numerical policy."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from llaisys.models.deepseek_v4_backends import (
    CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model,
)
from llaisys.models.deepseek_v4_model import PagedCachePool
from test_deepseek_v4_model import initialized_model, hadamard, torch_ops


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("published_fixed_profile_test", ROOT / "tools/deepseek_v4_reference/published_fixed_profile.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)
SOURCE = Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py")


class FixedProfileOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def load_source(self, name, ops=None):
        kernel, ops = types.ModuleType("kernel"), ops if ops is not None else torch_ops()
        for key in CONTRACTS:
            setattr(kernel, key, ops.function(key))
        with patch.dict(sys.modules, {"kernel": kernel}):
            module = PROFILE.load_profiled_model(name, SOURCE)
        self.addCleanup(sys.modules.pop, name, None)
        module.rotate_activation = lambda x: hadamard(x, x.shape[-1] ** -0.5)
        return module

    def test_source_transform_is_audited_and_never_edits_shared_source(self):
        before = SOURCE.read_bytes()
        _, manifest = PROFILE.prepare_source(SOURCE)
        self.assertEqual(manifest["transforms"], PROFILE.EXPECTED)
        self.assertFalse(manifest["unmodified_published_baseline"])
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "model.py"
            altered.write_text(before.decode().replace("q.square().mean(-1, keepdim=True)", "q.square().sum(-1, keepdim=True)"))
            with self.assertRaisesRegex(ValueError, "inverse RMS expression changed"):
                PROFILE.prepare_source(altered)
            altered.write_text(before.decode().replace("topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)",
                                                       "topk_idxs = other_window(win, bsz, seqlen, start_pos)"))
            with self.assertRaisesRegex(ValueError, "call-site audit"):
                PROFILE.prepare_source(altered)
        self.assertEqual(SOURCE.read_bytes(), before)

    def test_profile_is_module_local_and_cannot_silently_execute_mtp(self):
        linear, einsum, rsqrt = torch.nn.functional.linear, torch.einsum, torch.rsqrt
        source = self.load_source("fixed_profile_scope_test")
        self.assertIs(torch.nn.functional.linear, linear)
        self.assertIs(torch.einsum, einsum)
        self.assertIs(torch.rsqrt, rsqrt)
        with self.assertRaisesRegex(RuntimeError, "MTP"):
            source.Transformer.forward_spec(None)
        other = self.load_source("fixed_profile_scope_other_test")
        source._profile_linear(torch.randn(3, 8), torch.randn(4, 8))
        self.assertEqual(source._reference_profile_counts["linear"], 1)
        self.assertEqual(other._reference_profile_counts["linear"], 0)

    def test_published_full_replay_and_independent_paged_chunks_match(self):
        devices = ["cpu", "cuda"] if os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() else ["cpu"]
        for device in devices:
            torch.manual_seed(912)
            model = initialized_model(fixed_rows=32, indexer_tie_policy="index_ascending",
                                      attention_metadata_policy="fixed").to(device)
            if device == "cuda":
                # The selected GPU profile uses the actual TileLang kernels.
                # A torch.bmm-based sparse oracle is a different numerical
                # backend and is not assumed batch-shape invariant on CUDA.
                import tilelang
                spec = importlib.util.spec_from_file_location("fixed_profile_gpu_kernels", SOURCE.with_name("kernel.py"))
                kernels = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(kernels)
                registry = OperatorRegistry(MODEL_CONTRACTS)
                register_tilelang(registry, kernels, tilelang.__version__)
                register_torch_model(registry, backend="torch-fixed32", fixed_rows=32)
                ops = registry.bind({key: "tilelang" if key in CONTRACTS else "torch-fixed32" for key in MODEL_CONTRACTS})
                for module in model.modules():
                    if hasattr(module, "ops"):
                        module.ops = ops
            else:
                ops = model.ops
            source = self.load_source(f"fixed_profile_model_{device}", ops)
            fields = source.ModelArgs.__dataclass_fields__
            values = {key: value for key, value in vars(model.config).items() if key in fields}
            values.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0, max_batch_size=1,
                          dspark_block_size=0, temperature=0)
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with torch.device(device):
                    published = source.Transformer(source.ModelArgs(**values))
                published.load_state_dict(model.state_dict(), strict=True)
                inputs = torch.randint(0, model.config.vocab_size, (1, 137), device=device)
                # The published helpers create arange/masks on the current
                # default device; use a scoped context, like its CUDA runner.
                with torch.inference_mode(), torch.device(device):
                    _, expected, _ = published(inputs, 0)
                    independent_full = model.new_request()
                    try:
                        torch.testing.assert_close(model(inputs, independent_full).logits, expected, atol=0, rtol=0)
                    finally:
                        independent_full.close()
                    for name, buffer in published.named_buffers():
                        if name.endswith("score_state"):
                            buffer.fill_(-torch.inf)
                        elif name.endswith(("kv_cache", "kv_state")):
                            buffer.zero_()
                    for position in range(inputs.shape[1]):
                        _, replay, _ = published(inputs[:, position:position+1], position)
                    torch.testing.assert_close(replay, expected, atol=0, rtol=0)
                    pool = PagedCachePool(model, num_blocks=3)
                    for cache in (None, pool):
                        with self.subTest(device=device, paged=cache is not None):
                            state = model.new_request(cache_pool=cache)
                            start = 0
                            try:
                                for count in (3, 2, 7, 115, 10):
                                    final = start + count == inputs.shape[1]
                                    actual = model(inputs[:, start:start+count], state, emit_logits=final)
                                    if not final:
                                        self.assertIsNone(actual.logits)
                                    start += count
                                torch.testing.assert_close(actual.logits, expected, atol=0, rtol=0)
                                # Compare independent decode with the official control flow.
                                for _ in range(3):
                                    tokens = actual.logits.argmax(-1).unsqueeze(1)
                                    _, oracle, _ = published(tokens, state.position)
                                    actual = model(tokens, state)
                                    torch.testing.assert_close(actual.logits, oracle, atol=0, rtol=0)
                            finally:
                                state.close()
                            # Restore published request before evaluating the next layout.
                            for name, buffer in published.named_buffers():
                                if name.endswith("score_state"):
                                    buffer.fill_(-torch.inf)
                                elif name.endswith(("kv_cache", "kv_state")):
                                    buffer.zero_()
                            published(inputs, 0)
            finally:
                torch.set_default_dtype(previous)
            self.assertTrue(all(source._reference_profile_counts[key] > 0 for key in PROFILE.EXPECTED))


if __name__ == "__main__":
    unittest.main()

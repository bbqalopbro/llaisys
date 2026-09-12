"""Cache-only intermediate chunks must not execute an unused output head."""

import os
import unittest
from unittest.mock import patch

import torch

from llaisys.models.deepseek_v4_model import PagedCachePool
from test_deepseek_v4_model import initialized_model


class IntermediatePrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_omitted_heads_preserve_chunks_captures_and_following_decode(self):
        devices = ["cpu", "cuda"] if os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() else ["cpu"]
        for device in devices:
            torch.manual_seed(20260908)
            model = initialized_model().to(device)
            pool = PagedCachePool(model, num_blocks=5)
            inputs = torch.randint(0, model.config.vocab_size, (1, 137), device=device)
            for cache in (None, pool):
                with self.subTest(device=device, paged=cache is not None):
                    baseline, selected = model.new_request(cache_pool=cache), model.new_request(cache_pool=cache)
                    try:
                        start = 0
                        for count in (3, 2, 7, 115, 10):
                            ids = inputs[:, start:start+count]
                            expected = model(ids, baseline, capture_layers=True)
                            final = start + count == inputs.shape[1]
                            before = model.ops.report()
                            with patch.object(model.head, "forward", wraps=model.head.forward) as head, \
                                    patch.object(model.norm, "forward", wraps=model.norm.forward) as norm:
                                actual = model(ids, selected, capture_layers=True, emit_logits=final)
                            after = model.ops.report()
                            self.assertEqual(head.call_count, int(final))
                            self.assertEqual(norm.call_count, int(final))
                            # All transformer layers still execute on cache-only chunks.
                            self.assertEqual(after["sparse_attn"]["calls"] - before["sparse_attn"]["calls"],
                                             model.config.n_layers)
                            self.assertEqual(selected.position, baseline.position)
                            self.assertTrue(selected.valid)
                            for lhs, rhs in zip(actual.layer_hiddens, expected.layer_hiddens):
                                torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
                            if final:
                                torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                            else:
                                self.assertIsNone(actual.logits)
                            start += count
                        for _ in range(3):
                            ids = expected.logits.argmax(-1).unsqueeze(1)
                            expected, actual = model(ids, baseline), model(ids, selected)
                            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                    finally:
                        baseline.close()
                        selected.close()
                    self.assertEqual(pool.blocks.num_free, pool.blocks.num_total)

    def test_output_policy_validation_precedes_any_state_mutation(self):
        model = initialized_model()
        pool = PagedCachePool(model, num_blocks=2)
        state = model.new_request(cache_pool=pool)
        try:
            for policy in (0, 1, None, "false", torch.tensor(False)):
                with self.assertRaisesRegex(ValueError, "explicit boolean"):
                    model(torch.tensor([[1, 2]]), state, emit_logits=policy)
                self.assertEqual(state.position, 0)
                self.assertTrue(state.valid)
                self.assertEqual(state.lease.block_ids, [])
        finally:
            state.close()

    def test_cache_only_failure_invalidates_and_reset_recovers_request(self):
        model = initialized_model()
        state = model.new_request()
        with patch.object(model.layers[1], "forward", side_effect=RuntimeError("injected layer error")), \
                patch.object(model.head, "forward", wraps=model.head.forward) as head:
            with self.assertRaisesRegex(RuntimeError, "injected layer error"):
                model(torch.tensor([[1, 2]]), state, emit_logits=False)
        self.assertEqual(head.call_count, 0)
        self.assertFalse(state.valid)
        self.assertEqual(state.position, 0)
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            model(torch.tensor([[3]]), state)
        state.reset()
        self.assertIsNone(model(torch.tensor([[1, 2]]), state, emit_logits=False).logits)
        self.assertEqual(state.position, 2)
        self.assertIsNotNone(model(torch.tensor([[3]]), state).logits)
        state.close()


if __name__ == "__main__":
    unittest.main()

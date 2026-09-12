"""Actual C++ block ownership plus V4 overlap-state/payload correctness."""

import gc
import os
import unittest
from unittest.mock import patch

import torch

from llaisys import _C
from llaisys.models.deepseek_v4_model.paged import PagedCachePool, torch_cache_ops
from test_deepseek_v4_model import initialized_model


class V4PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def devices(self):
        return ("cpu", "cuda") if os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() else ("cpu",)

    def fixture(self, device="cpu", blocks=8):
        torch.manual_seed(726)
        model = initialized_model(fixed_rows=32, indexer_tie_policy="index_ascending",
                                  attention_metadata_policy="fixed", max_seq_len=512).to(device)
        pool = PagedCachePool(model, num_blocks=blocks, enable_prefix_cache=True)
        tokens = torch.randint(0, model.config.vocab_size, (1, 267), device=device)
        return model, pool, tokens

    @torch.inference_mode()
    def test_full_prefill_captures_each_boundary_and_real_hit_skips_tokens(self):
        for device in self.devices():
            model, pool, tokens = self.fixture(device)
            seed = model.new_request(cache_pool=pool)
            expected = model(tokens, seed).logits
            self.assertEqual(set(seed.prefix_snapshots), {128, 256})
            self.assertTrue(seed.publish_prefix())
            ids = seed.lease.block_ids[:2]
            seed.close()
            hit = model.new_request(cache_pool=pool)
            self.assertEqual(hit.attach_prefix(tokens[0].tolist()), 256)
            self.assertEqual(hit.lease.block_ids, ids)
            self.assertEqual(hit.position, 256)
            actual = model(tokens[:, 256:], hit).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for _ in range(3):
                next_ids = expected.argmax(-1).unsqueeze(1)
                # Independent cold full history for this selected fixed profile.
                tokens = torch.cat((tokens, next_ids), dim=1)
                reference = model.new_request()
                expected = model(tokens, reference).logits
                reference.close()
                actual = model(next_ids, hit).logits
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(pool.prefix.report()["hit_tokens"], 256)
            hit.close()
            pool.prefix.clear()
            self.assertEqual(pool.blocks.num_free, pool.blocks.num_total)
            self.assertEqual(pool.blocks.num_cached, 0)

    @torch.inference_mode()
    def test_snapshots_across_non_aligned_chunks_and_single_token_boundary(self):
        for device in self.devices():
            for chunks in ((267,), (127, 1, 139), (125, 3, 128, 11), (65, 65, 65, 65, 7)):
                with self.subTest(device=device, chunks=chunks):
                    model, pool, tokens = self.fixture(device)
                    seed = model.new_request(cache_pool=pool)
                    position = 0
                    for count in chunks:
                        expected = model(tokens[:, position:position+count], seed).logits
                        position += count
                    self.assertTrue(seed.publish_prefix())
                    hit = model.new_request(cache_pool=pool)
                    self.assertEqual(hit.attach_prefix(tokens[0].tolist()), 256)
                    actual = model(tokens[:, 256:], hit).logits
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    seed.close()
                    hit.close()
                    pool.prefix.clear()

    @torch.inference_mode()
    def test_shorter_prefix_restores_its_own_overlap_not_final_request_state(self):
        model, pool, tokens = self.fixture()
        seed = model.new_request(cache_pool=pool)
        model(tokens, seed)
        seed.publish_prefix()
        changed = torch.cat((tokens[:, :128], 31 - tokens[:, 128:137]), dim=1)
        hit, cold = model.new_request(cache_pool=pool), model.new_request()
        self.assertEqual(hit.attach_prefix(changed[0].tolist()), 128)
        torch.testing.assert_close(model(changed[:, 128:], hit).logits, model(changed, cold).logits, atol=0, rtol=0)
        seed.close()
        hit.close()
        cold.close()

    @torch.inference_mode()
    def test_exact_block_prompt_leaves_one_block_for_logits(self):
        model, pool, tokens = self.fixture()
        seed = model.new_request(cache_pool=pool)
        model(tokens[:, :256], seed)
        seed.publish_prefix()
        for length, match in ((1, 0), (127, 0), (128, 0), (129, 128), (256, 128), (257, 256)):
            hit = model.new_request(cache_pool=pool)
            self.assertEqual(hit.attach_prefix(tokens[0, :length].tolist()), match)
            hit.close()
        seed.close()

    @torch.inference_mode()
    def test_active_shared_prefix_is_immutable_and_clear_does_not_break_readers(self):
        for device in self.devices():
            model, pool, tokens = self.fixture(device)
            seed = model.new_request(cache_pool=pool)
            model(tokens[:, :128], seed)
            seed.publish_prefix()
            first, second = model.new_request(cache_pool=pool), model.new_request(cache_pool=pool)
            for state in (first, second):
                self.assertEqual(state.attach_prefix(tokens[0, :137].tolist()), 128)
            block = seed.lease.block_ids[0]
            self.assertEqual(pool.blocks.metadata(block).ref_count, 3)
            original = [value.clone() for value in pool.payloads]
            pool.prefix.clear()
            self.assertEqual(pool.blocks.metadata(block).ref_count, 3)
            for state, suffix in ((first, tokens[:, 128:137]), (second, 31 - tokens[:, 128:137])):
                model(suffix, state)
            for layer, ratio in enumerate(model.config.compress_ratios):
                stride = 128 + (128 // ratio if ratio else 0)
                torch.testing.assert_close(pool.payloads[layer][:, block*stride:(block+1)*stride],
                                           original[layer][:, block*stride:(block+1)*stride], atol=0, rtol=0)
            for state in (seed, first, second):
                state.close()
            self.assertEqual(pool.blocks.num_free, pool.blocks.num_total)

    @torch.inference_mode()
    def test_partial_fork_cow_preserves_parent_and_copies_every_component(self):
        for device in self.devices():
            model, pool, tokens = self.fixture(device)
            parent = model.new_request(cache_pool=pool)
            model(tokens[:, :133], parent)
            child = parent.fork()
            shared_ids = parent.lease.block_ids
            self.assertEqual(child.lease.block_ids, shared_ids)
            child_input, parent_input = tokens[:, 133:137], 31 - tokens[:, 133:137]
            actual_child = model(child_input, child).logits
            self.assertEqual(pool.cow_copies, 1)
            self.assertEqual(child.lease.block_ids[0], shared_ids[0])
            self.assertNotEqual(child.lease.block_ids[1], shared_ids[1])
            actual_parent = model(parent_input, parent).logits
            self.assertEqual(pool.cow_copies, 1)
            for suffix, actual in ((child_input, actual_child), (parent_input, actual_parent)):
                cold = model.new_request()
                expected = model(torch.cat((tokens[:, :133], suffix), dim=1), cold).logits
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                cold.close()
            self.assertGreater(pool.report()["operators"]["cache_copy_block"]["calls"], len(model.layers))
            parent.close()
            child.close()
            self.assertEqual(pool.blocks.num_free, pool.blocks.num_total)

    @torch.inference_mode()
    def test_cow_exhaustion_and_copy_error_leave_original_aliases_intact(self):
        model, pool, tokens = self.fixture(blocks=2)
        parent = model.new_request(cache_pool=pool)
        model(tokens[:, :133], parent)
        child = parent.fork()
        before = child.lease.block_ids
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            model(tokens[:, 133:134], child)
        self.assertEqual(child.lease.block_ids, before)
        self.assertTrue(child.valid)
        parent.close()
        child.close()
        model, pool, tokens = self.fixture()
        parent = model.new_request(cache_pool=pool)
        model(tokens[:, :133], parent)
        child = parent.fork()
        before, free = child.lease.block_ids, pool.blocks.num_free
        def fail(*args):
            raise RuntimeError("copy failed")
        with patch.dict(pool.ops._functions, cache_copy_block=fail), self.assertRaisesRegex(RuntimeError, "copy failed"):
            model(tokens[:, 133:134], child)
        self.assertEqual(child.lease.block_ids, before)
        self.assertEqual(pool.blocks.num_free, free)
        self.assertEqual(pool.cow_copies, 0)
        parent.close()
        child.close()

    @torch.inference_mode()
    def test_lru_recycling_cannot_restore_stale_compressor_state(self):
        model, pool, tokens = self.fixture(blocks=3)
        seed = model.new_request(cache_pool=pool)
        model(tokens[:, :256], seed)
        seed.publish_prefix()
        seed.close()
        occupied = pool.blocks.allocate(3)
        self.assertEqual(pool.blocks.num_cached, 0)
        occupied.close()
        hit = model.new_request(cache_pool=pool)
        self.assertEqual(hit.attach_prefix(tokens[0].tolist()), 0)
        self.assertEqual(pool.prefix.report()["record_count"], 0)
        hit.close()

    @torch.inference_mode()
    def test_failure_cannot_publish_partial_layers_and_reset_cleans_metadata(self):
        model, pool, tokens = self.fixture()
        state = model.new_request(cache_pool=pool)
        with patch.object(model.layers[2], "forward", side_effect=RuntimeError("layer failed")):
            with self.assertRaises(RuntimeError):
                model(tokens, state)
        with self.assertRaises(ValueError):
            state.publish_prefix()
        self.assertEqual(pool.blocks.num_cached, 0)
        state.reset()
        self.assertEqual(state.prefix_snapshots, {})
        model(tokens[:, :128], state)
        self.assertTrue(state.publish_prefix())
        state.close()

    @torch.inference_mode()
    def test_exact_tokens_and_full_resume_records_are_required(self):
        model, pool, tokens = self.fixture()
        seed = model.new_request(cache_pool=pool)
        model(tokens, seed)
        snapshot = seed.prefix_snapshots.pop(128)
        with self.assertRaisesRegex(RuntimeError, "missing complete"):
            seed.publish_prefix()
        seed.prefix_snapshots[128] = snapshot
        seed.publish_prefix()
        first = seed.lease.block_ids[0]
        record = pool.prefix.records[first]
        from dataclasses import replace
        pool.prefix.records[first] = replace(record, tokens=(99,) * 128)
        hit = model.new_request(cache_pool=pool)
        self.assertEqual(hit.attach_prefix(tokens[0].tolist()), 0)
        hit.close()
        seed.close()

    def test_block_copy_contract_and_explicit_opt_in(self):
        ops = torch_cache_ops()
        payload = torch.arange(32).reshape(1, 8, 4).bfloat16()
        ops.function("cache_copy_block")(payload, 0, 1, 4)
        torch.testing.assert_close(payload[:, :4], payload[:, 4:], atol=0, rtol=0)
        for args in ((0, 0, 4), (0, 2, 4), (-1, 1, 4), (0, 1, 0)):
            with self.assertRaises(ValueError):
                ops.function("cache_copy_block")(payload, *args)
        model = initialized_model()
        pool = PagedCachePool(model, num_blocks=2)
        state = model.new_request(cache_pool=pool)
        with self.assertRaisesRegex(RuntimeError, "explicitly enabled"):
            state.attach_prefix([1, 2])
        with self.assertRaises(RuntimeError):
            state.publish_prefix()
        state.close()

    def test_generation_reuses_real_prefix_and_cancellation_releases_hit(self):
        from llaisys.models.deepseek_v4_model.generation import greedy_generate
        model, pool, tokens = self.fixture()
        ids = tokens[0].tolist()
        first = greedy_generate(model, ids, max_new_tokens=3, eos_id=0, cache_pool=pool, reuse_prefix=True)
        second = greedy_generate(model, ids, max_new_tokens=3, eos_id=0, cache_pool=pool, reuse_prefix=True)
        self.assertEqual(first["prefix_hit_tokens"], 0)
        self.assertEqual(second["prefix_hit_tokens"], 256)
        self.assertEqual(first["generated_ids"], second["generated_ids"])
        cancelled = greedy_generate(model, ids, max_new_tokens=3, eos_id=0, cache_pool=pool,
                                    reuse_prefix=True, cancelled=lambda: True)
        self.assertEqual(cancelled["finish_reason"], "cancelled")
        self.assertEqual(cancelled["generated_ids"], [])
        self.assertEqual(pool.blocks.num_free, pool.blocks.num_total)
        pool.prefix.clear()


if __name__ == "__main__":
    unittest.main()

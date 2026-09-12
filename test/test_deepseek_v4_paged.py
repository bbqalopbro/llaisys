"""Real C++ block pool plus V4 payload/model tests; no Qwen execution."""

import gc
import os
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from llaisys.models.deepseek_v4_backends import OperatorRegistry
from llaisys.models.deepseek_v4_model.paged import CACHE_CONTRACTS, PagedCachePool, torch_cache_ops
from test_deepseek_v4_model import initialized_model


class PagedV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def devices(self):
        # No CUDA work on the login host, including when a driver is visible.
        return ["cpu", "cuda"] if os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() else ["cpu"]

    def test_slot_mapping_and_scatter_are_component_specific(self):
        ops = torch_cache_ops()
        table = torch.tensor([3, 0, 5])
        indices = torch.tensor([[-1, 0, 127, 128, 255, 256]])
        expected = torch.tensor([[-1, 480, 607, 0, 127, 800]])
        torch.testing.assert_close(ops.function("cache_map_slots")(indices, table, 128, 160, 0), expected)
        groups = torch.tensor([0, 31, 32, 63, 64])
        compressed = ops.function("cache_map_slots")(groups, table, 32, 160, 128)
        torch.testing.assert_close(compressed, torch.tensor([608, 639, 128, 159, 928]))
        pool = torch.zeros(1, 960, 128, dtype=torch.bfloat16)
        values = torch.randn(1, 5, 128).bfloat16()
        ops.function("cache_write")(pool, values, compressed)
        torch.testing.assert_close(ops.function("cache_read")(pool, compressed), values, atol=0, rtol=0)
        self.assertEqual(pool[:, :128].count_nonzero().item(), 0)
        with self.assertRaises(ValueError):
            ops.function("cache_write")(pool, values.float(), compressed)

    def test_full_and_decode_match_contiguous_with_fragmented_physical_blocks(self):
        for device in self.devices():
            torch.manual_seed(184)
            model = initialized_model().to(device)
            pool = PagedCachePool(model, num_blocks=6)
            held = pool.blocks.allocate(2)
            for length in (1, 7, 8, 9, 127, 128, 137):
                with self.subTest(device=device, length=length):
                    baseline, paged = model.new_request(), model.new_request(cache_pool=pool)
                    tokens = torch.randint(0, model.config.vocab_size, (1, length), device=device)
                    try:
                        for step in range(4):
                            expected = model(tokens, baseline, capture_layers=True)
                            actual = model(tokens, paged, capture_layers=True)
                            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                            for lhs, rhs in zip(actual.layer_hiddens, expected.layer_hiddens):
                                torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
                            self.assertTrue(all(layer.kv_cache is None and layer.index_cache is None for layer in paged.layers))
                            tokens = expected.logits.argmax(-1).unsqueeze(1)
                        self.assertEqual(paged.position, length + 3)
                    finally:
                        paged.close()
                        baseline.close()
                    self.assertEqual(pool.blocks.num_free, 4)
            held.close()

    def test_non_aligned_chunks_match_the_same_contiguous_schedule(self):
        # This isolates cache layout, not the still-failing default chunk/full
        # numerical policy. Both sides receive exactly the same chunk shapes.
        for device in self.devices():
            torch.manual_seed(909)
            model = initialized_model().to(device)
            pool = PagedCachePool(model, num_blocks=5)
            tokens = torch.randint(0, model.config.vocab_size, (1, 137), device=device)
            for chunks in ((3, 2, 7, 115, 10), (1, 130, 6), (127, 2, 8), (65, 65, 7)):
                with self.subTest(device=device, chunks=chunks):
                    first, second = model.new_request(), model.new_request(cache_pool=pool)
                    position = 0
                    try:
                        for count in chunks:
                            ids = tokens[:, position:position+count]
                            expected, actual = model(ids, first), model(ids, second)
                            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                            position += count
                        for _ in range(3):
                            ids = expected.logits.argmax(-1).unsqueeze(1)
                            expected, actual = model(ids, first), model(ids, second)
                            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                    finally:
                        first.close()
                        second.close()

    def test_attention_receives_pool_pointer_and_only_indexer_gathers(self):
        model = initialized_model()
        pool = PagedCachePool(model, num_blocks=4)
        state = model.new_request(cache_pool=pool)
        function = model.ops._functions["sparse_attn"]
        pointers = []

        def inspect(query, latent, *args):
            pointers.append(latent.data_ptr())
            return function(query, latent, *args)

        with patch.dict(model.ops._functions, sparse_attn=inspect):
            model(torch.tensor([[1, 2, 3, 4, 5]]), state)
        self.assertEqual(pointers, [value.data_ptr() for value in pool.payloads])
        report = pool.report()
        self.assertEqual(report["operators"]["cache_read"]["calls"], 1)
        self.assertTrue(report["attention_reads_pool_directly"])
        self.assertFalse(report["attention_history_gather"])
        self.assertTrue(report["indexer_history_gather"])
        self.assertFalse(report["fallback"])
        state.close()

    def test_interleaving_requests_keep_disjoint_payload_and_noncontiguous_tables(self):
        torch.manual_seed(911)
        model = initialized_model()
        pool = PagedCachePool(model, num_blocks=5)
        first, second = model.new_request(cache_pool=pool), model.new_request(cache_pool=pool)
        reference = model.new_request()
        tokens = torch.randint(0, model.config.vocab_size, (1, 137))
        try:
            model(tokens[:, :65], first)
            model(tokens[:, :65], reference)
            model(torch.tensor([[3, 4, 2]]), second)
            held_payload = [value.clone() for value in pool.payloads]
            actual = model(tokens[:, 65:], first)
            expected = model(tokens[:, 65:], reference)
            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
            self.assertEqual(first.lease.block_ids, [0, 2])
            self.assertEqual(second.lease.block_ids, [1])
            for index, ratio in enumerate(model.config.compress_ratios):
                width = 128 + (128 // ratio if ratio else 0)
                torch.testing.assert_close(pool.payloads[index][:, width:2*width],
                                           held_payload[index][:, width:2*width], atol=0, rtol=0)
        finally:
            first.close()
            second.close()
            reference.close()
        self.assertEqual(pool.blocks.num_free, 5)

    def test_exhaustion_does_not_advance_state_and_reset_releases_blocks(self):
        model = initialized_model()
        pool = PagedCachePool(model, num_blocks=1)
        state = model.new_request(cache_pool=pool)
        model(torch.ones(1, 127, dtype=torch.int64), state)
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            model(torch.ones(1, 2, dtype=torch.int64), state)
        self.assertTrue(state.valid)
        self.assertEqual(state.position, 127)
        self.assertEqual(pool.blocks.metadata(0).num_tokens, 127)
        state.reset()
        self.assertEqual(pool.blocks.num_free, 1)
        self.assertEqual(state.position, 0)
        self.assertEqual(state.lease.block_ids, [])
        state.close()
        state.close()

    def test_failure_invalidates_request_and_reset_recovers_without_leaks(self):
        model = initialized_model()
        pool = PagedCachePool(model, num_blocks=3)
        state = model.new_request(cache_pool=pool)
        with patch.object(model.layers[1], "forward", side_effect=RuntimeError("kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                model(torch.tensor([[1, 2, 3]]), state)
        self.assertFalse(state.valid)
        self.assertEqual(state.position, 0)
        self.assertFalse(pool.blocks.metadata(0).computed)
        with self.assertRaises(RuntimeError):
            model(torch.tensor([[4]]), state)
        state.reset()
        self.assertEqual(pool.blocks.num_free, 3)
        model(torch.tensor([[1, 2, 3]]), state)
        del state
        gc.collect()
        self.assertEqual(pool.blocks.num_free, 3)

    def test_invalid_layout_owner_batch_and_weight_generation_are_rejected(self):
        model = initialized_model()
        for block_size in (0, 16, 129):
            with self.assertRaises(ValueError):
                PagedCachePool(model, num_blocks=2, block_size=block_size)
        pool = PagedCachePool(model, num_blocks=2)
        with self.assertRaises(ValueError):
            initialized_model().new_request(cache_pool=pool)
        with self.assertRaises(ValueError):
            model.new_request(2, cache_pool=pool)
        state = model.new_request(cache_pool=pool)
        model._state_owner = object()
        with self.assertRaises(ValueError):
            model(torch.tensor([[1]]), state)
        with self.assertRaises(ValueError):
            model.new_request(cache_pool=pool)
        state.close()

    def test_cache_operator_library_is_replaceable_and_failure_never_falls_back(self):
        defaults, registry = torch_cache_ops(), OperatorRegistry(CACHE_CONTRACTS)
        for name in CACHE_CONTRACTS:
            function = defaults.function(name)
            if name == "cache_write":
                def function(*args):
                    raise RuntimeError("selected slot writer failed")
            registry.register("user-library", name, function, version="test", contract=CACHE_CONTRACTS[name])
        ops = registry.bind({name: "user-library" for name in CACHE_CONTRACTS})
        model = initialized_model()
        pool = PagedCachePool(model, num_blocks=2, ops=ops)
        state = model.new_request(cache_pool=pool)
        with self.assertRaisesRegex(RuntimeError, "selected slot writer failed"):
            model(torch.tensor([[1, 2]]), state)
        self.assertFalse(state.valid)
        report = pool.report()["operators"]["cache_write"]
        self.assertEqual(report["failures"], 1)
        self.assertFalse(report["fallback"])
        state.close()

    def test_same_operator_names_do_not_bypass_contract_validation(self):
        contracts = dict(CACHE_CONTRACTS)
        contracts["cache_write"] = replace(contracts["cache_write"], revision=2)
        registry, defaults = OperatorRegistry(contracts), torch_cache_ops()
        for name, contract in contracts.items():
            registry.register("incompatible", name, defaults.function(name), version="2", contract=contract)
        ops = registry.bind({name: "incompatible" for name in contracts})
        with self.assertRaisesRegex(ValueError, "exactly"):
            PagedCachePool(initialized_model(), num_blocks=2, ops=ops)

    def test_cuda_stream_affinity_is_explicit(self):
        if "cuda" not in self.devices():
            # CPU behavior is tested, GPU rejection is separately executed in
            # the Slurm regression; no GPU verification is claimed on CPU.
            self.assertIsNone(PagedCachePool(initialized_model(), num_blocks=1).stream)
            return
        model = initialized_model().cuda()
        pool = PagedCachePool(model, num_blocks=2)
        state = model.new_request(cache_pool=pool)
        ids = torch.tensor([[1, 2, 3]], device="cuda")
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "owning CUDA stream"):
                model(ids, state)
        self.assertEqual(state.position, 0)
        model(ids, state)
        state.close()


if __name__ == "__main__":
    unittest.main()

"""Actual C++ block ownership tests, without model inference or payload claims."""

import gc
import unittest

from llaisys import _C


class NativeCacheBindingTests(unittest.TestCase):
    def test_prefix_lease_and_replacement_transfer_only_selected_references(self):
        pool = _C.CacheBlockPool(5)
        source = pool.allocate(3)
        prefix = source.prefix(2)
        self.assertEqual(prefix.block_ids, source.block_ids[:2])
        self.assertEqual([pool.metadata(i).ref_count for i in source.block_ids], [2, 2, 1])
        replacement = pool.allocate()
        old, new = prefix.block_ids[1], replacement.block_ids[0]
        prefix.replace(1, replacement)
        self.assertTrue(replacement.closed)
        self.assertEqual(pool.metadata(old).ref_count, 1)
        self.assertEqual(prefix.block_ids[1], new)
        with self.assertRaises(ValueError):
            source.prefix(4)
        with self.assertRaises(ValueError):
            source.replace(0, source.prefix(1))
        prefix.close()
        source.close()
        self.assertEqual(pool.num_free, 5)

    def test_uncache_does_not_release_active_request_ownership(self):
        pool = _C.CacheBlockPool(2)
        index = _C.CachePrefixIndex(pool, 4)
        lease = pool.allocate()
        lease.mark_computed(4)
        index.publish([1, 2, 3, 4], lease)
        block = lease.block_ids[0]
        self.assertNotEqual(pool.metadata(block).block_hash, 0)
        self.assertTrue(pool.uncache(block))
        self.assertFalse(pool.metadata(block).cached)
        self.assertEqual(pool.metadata(block).ref_count, 1)
        self.assertEqual(index.lookup([1, 2, 3, 4]).matched_tokens, 0)
        lease.close()
        self.assertEqual(pool.num_free, 2)

    def test_append_transfers_ownership_and_rejects_aliases_atomically(self):
        pool = _C.CacheBlockPool(5)
        first, second = pool.allocate(2), pool.allocate(2)
        ids = first.block_ids + second.block_ids
        first.append(second)
        self.assertTrue(second.closed)
        self.assertEqual(first.block_ids, ids)
        self.assertEqual([pool.metadata(i).ref_count for i in ids], [1] * 4)
        shared = first.share()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            first.append(shared)
        with self.assertRaises(ValueError):
            first.append(first)
        with self.assertRaises(ValueError):
            first.append(_C.CacheBlockPool(1).allocate())
        self.assertEqual(first.block_ids, ids)
        self.assertFalse(shared.closed)
        shared.close()
        first.close()
        self.assertEqual(pool.num_free, 5)

    def test_per_block_validity_prevalidates_every_count(self):
        pool = _C.CacheBlockPool(3)
        lease = pool.allocate(3)
        ids = lease.block_ids
        for counts in ([4, 4], [4, 0, 1]):
            with self.assertRaises(ValueError):
                lease.mark_computed_counts(counts)
            self.assertTrue(all(not pool.metadata(i).computed for i in ids))
        lease.mark_computed_counts([4, 4, 1])
        self.assertEqual([pool.metadata(i).num_tokens for i in ids], [4, 4, 1])

    def test_batched_allocation_sharing_close_and_metadata_snapshots(self):
        pool = _C.CacheBlockPool(6)
        lease = pool.allocate(2)
        ids = lease.block_ids
        self.assertEqual(pool.num_free, 4)
        shared = lease.share()
        snapshot = pool.metadata(ids[0])
        self.assertEqual(snapshot.ref_count, 2)
        lease.close()
        lease.close()
        self.assertTrue(lease.closed)
        self.assertEqual(lease.block_ids, [])
        self.assertEqual(pool.metadata(ids[0]).ref_count, 1)
        self.assertEqual(snapshot.ref_count, 2)
        shared.close()
        self.assertEqual(pool.num_free, 6)
        self.assertFalse(pool.metadata(ids[0]).allocated)
        with self.assertRaises(RuntimeError):
            lease.share()

    def test_allocation_failure_does_not_partially_consume_blocks(self):
        pool = _C.CacheBlockPool(3)
        lease = pool.allocate(2)
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            pool.allocate(2)
        self.assertEqual(pool.num_free, 1)
        self.assertEqual(pool.metadata(lease.block_ids[0]).ref_count, 1)
        empty = pool.allocate(0)
        self.assertEqual(empty.block_ids, [])

    def test_gc_and_context_manager_release_references(self):
        pool = _C.CacheBlockPool(4)
        lease = pool.allocate(3)
        del lease
        gc.collect()
        self.assertEqual(pool.num_free, 4)
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            with pool.allocate(4) as current:
                self.assertEqual(pool.num_free, 0)
                raise RuntimeError("request failed")
        self.assertTrue(current.closed)
        self.assertEqual(pool.num_free, 4)

    def test_lease_keeps_cpp_pool_alive_without_python_pool_object(self):
        pool = _C.CacheBlockPool(2)
        lease = pool.allocate(2)
        del pool
        gc.collect()
        shared = lease.share()
        lease.close()
        self.assertEqual(len(shared.block_ids), 2)
        shared.close()

    def test_prefix_lookup_retains_blocks_and_lru_reclaims_only_unreferenced(self):
        pool = _C.CacheBlockPool(4)
        index = _C.CachePrefixIndex(pool, 4, 123)
        source = pool.allocate(2)
        source.mark_computed(4)
        self.assertTrue(index.publish(list(range(8)), source))
        match = index.lookup(list(range(11)))
        self.assertEqual(match.matched_tokens, 8)
        self.assertEqual(match.block_ids, source.block_ids)
        self.assertNotEqual(match.terminal_hash, 0)
        source.close()
        self.assertEqual(pool.num_free, 2)
        with self.assertRaises(RuntimeError):
            pool.allocate(3)
        match.close()
        self.assertEqual(pool.num_free, 4)
        self.assertEqual(pool.num_cached, 2)
        replacement = pool.allocate(4)
        self.assertEqual(pool.num_cached, 0)
        self.assertEqual(index.lookup(list(range(8))).matched_tokens, 0)
        replacement.close()

    def test_publish_requires_complete_computed_blocks_and_correct_owner(self):
        pool = _C.CacheBlockPool(4)
        index = _C.CachePrefixIndex(pool, 4)
        lease = pool.allocate()
        for tokens in ([1, 2, 3], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                index.publish(tokens, lease)
        lease.mark_computed(3)
        with self.assertRaisesRegex(ValueError, "complete computed"):
            index.publish([1, 2, 3, 4], lease)
        other = _C.CacheBlockPool(2).allocate()
        other.mark_computed(4)
        with self.assertRaisesRegex(ValueError, "another block pool"):
            index.publish([1, 2, 3, 4], other)
        lease.mark_computed(4)
        self.assertTrue(index.publish([1, 2, 3, 4], lease))
        with self.assertRaisesRegex(ValueError, "validity"):
            lease.mark_computed(3)

    def test_cache_salt_collision_and_payload_rebinding_are_explicit(self):
        pool = _C.CacheBlockPool(4)
        index = _C.CachePrefixIndex(pool, 4, 1)
        first, second = pool.allocate(), pool.allocate()
        first.mark_computed(4)
        second.mark_computed(4)
        self.assertTrue(index.publish([1, 2, 3, 4], first))
        self.assertFalse(index.publish([1, 2, 3, 4], second))
        self.assertEqual(pool.num_cached, 1)
        with self.assertRaisesRegex(ValueError, "different prefix"):
            index.publish([5, 6, 7, 8], first)
        different_salt = _C.CachePrefixIndex(pool, 4, 2)
        self.assertEqual(different_salt.lookup([1, 2, 3, 4]).matched_tokens, 0)
        self.assertEqual(index.lookup([1, 2, 3, 9]).matched_tokens, 0)

    def test_invalid_capacities_and_no_arbitrary_lease_adoption(self):
        for count in (0, 2**31):
            with self.assertRaises(ValueError):
                _C.CacheBlockPool(count)
        pool = _C.CacheBlockPool(1)
        for size in (0, 2**32):
            with self.assertRaises(ValueError):
                _C.CachePrefixIndex(pool, size)
        with self.assertRaises(ValueError):
            _C.CachePrefixIndex(None, 4)
        with self.assertRaises(IndexError):
            pool.metadata(1)
        with self.assertRaises(TypeError):
            _C.CacheBlockLease()


if __name__ == "__main__":
    unittest.main()

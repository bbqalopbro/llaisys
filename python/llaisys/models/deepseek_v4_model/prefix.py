"""V4 compressor boundary state accompanying the existing C++ block index.

No independent prefix search tree or block allocator lives here. Records contain
only model-specific resume state and exact token blocks (collision validation).
The C++ index performs lookup/refcounts/LRU. One immutable weight generation,
one arithmetic profile and one CUDA stream are scoped by the owning pool.
"""

from dataclasses import dataclass
import weakref

import torch


@dataclass(frozen=True)
class PrefixRecord:
    block_hash: int
    tokens: tuple[int, ...]
    resume: dict


class V4PrefixCache:
    def __init__(self, pool):
        from llaisys import _C
        self._pool = weakref.ref(pool)
        self.index = _C.CachePrefixIndex(pool.blocks, pool.block_size)
        self.records = {}
        self.required = {(layer, component) for layer, ratio in enumerate(pool.config.compress_ratios)
                         if ratio == 4 for component in ("compressed", "index")}
        self.hits = self.hit_tokens = self.misses = self.published_blocks = 0

    @property
    def pool(self):
        pool = self._pool()
        if pool is None:
            raise RuntimeError("prefix cache pool was released")
        pool.check_stream()
        return pool

    def discard_recycled(self, ids):
        for block in ids:
            self.records.pop(block, None)

    def publish(self, state):
        pool = self.pool
        if state.pool is not pool or state.closed or not state.valid:
            raise ValueError("publish requires a valid request from this pool")
        count = state.position // pool.block_size
        if not count:
            return False
        # One host transfer at publication, never per decode token. Actual
        # executed input IDs are recorded by the model, not asserted by caller.
        tokens = state.token_history[:count * pool.block_size].tolist()
        snapshots = [state.prefix_snapshots.get((i + 1) * pool.block_size) for i in range(count)]
        if any(snapshot is None or set(snapshot) != self.required for snapshot in snapshots):
            raise RuntimeError("prefix is missing complete compressor boundary state")
        ids = state.lease.block_ids[:count]
        # A duplicate existing prefix is an explicit no-op; never replace its
        # immutable payload/resume state with another execution's arithmetic.
        if not self.index.publish(tokens, state.lease):
            return False
        for i, (block, snapshot) in enumerate(zip(ids, snapshots)):
            self.records[block] = PrefixRecord(pool.blocks.metadata(block).block_hash,
                tuple(tokens[i * pool.block_size:(i + 1) * pool.block_size]), dict(snapshot))
        self.published_blocks += count
        return True

    def attach(self, state, tokens):
        pool = self.pool
        if state.pool is not pool or state.closed or not state.valid or state.position != 0 or state.lease.block_ids:
            raise ValueError("prefix attach requires an empty valid request from this pool")
        if not isinstance(tokens, (list, tuple)) or not tokens or len(tokens) > state.max_seq_len or any(
                type(token) is not int or not 0 <= token < pool.config.vocab_size for token in tokens):
            raise ValueError("invalid prompt token IDs")
        # Always leave at least one prompt token to compute the final logits.
        # Cached latent blocks alone do not contain a final hidden/head output.
        limit = (len(tokens) - 1) // pool.block_size * pool.block_size
        matched = self.index.lookup(list(tokens[:limit]))
        lease = None
        try:
            valid = 0
            for i, block in enumerate(matched.block_ids):
                record = self.records.get(block)
                if (record is None or record.block_hash != pool.blocks.metadata(block).block_hash
                        or record.tokens != tuple(tokens[i * pool.block_size:(i + 1) * pool.block_size])):
                    break
                valid += 1
            count = valid * pool.block_size
            if not count:
                self.misses += 1
                return 0
            lease = matched.prefix(valid)
            table = torch.tensor(lease.block_ids, dtype=torch.int64, device=pool.device)
            terminal = self.records[lease.block_ids[-1]]
            for (layer, component), (kv, scores) in terminal.resume.items():
                destination = state.layers[layer].compressor if component == "compressed" else state.layers[layer].index_compressor
                destination.kv[:, :4, :kv.shape[-1]].copy_(kv)
                destination.scores[:, :4, :scores.shape[-1]].copy_(scores)
            state.token_history[:count].copy_(torch.tensor(tokens[:count], dtype=torch.int64, device=pool.device))
            snapshots = {(i + 1) * pool.block_size: dict(self.records[block].resume)
                         for i, block in enumerate(lease.block_ids)}
            state.lease.close()
            state.lease, lease = lease, None
            state.block_table, state.position, state.prefix_snapshots = table, count, snapshots
            state.prefix_hit_tokens = count
            self.hits += 1
            self.hit_tokens += count
            return count
        except Exception:
            state.valid = False
            raise
        finally:
            if lease is not None:
                lease.close()
            matched.close()

    def clear(self):
        pool = self.pool
        for block, record in self.records.items():
            metadata = pool.blocks.metadata(block)
            if metadata.cached and metadata.block_hash == record.block_hash:
                pool.blocks.uncache(block)
        self.records.clear()

    def report(self):
        pool = self.pool
        # Also tolerate direct low-level pool allocation by pruning stale IDs.
        stale = [block for block, record in self.records.items()
                 if not pool.blocks.metadata(block).cached or pool.blocks.metadata(block).block_hash != record.block_hash]
        self.discard_recycled(stale)
        return {"backend": "existing-cpp-block-prefix-cache", "record_count": len(self.records),
                "resume_state_bytes": sum(t.numel() * t.element_size() for record in self.records.values()
                                          for pair in record.resume.values() for t in pair),
                "hits": self.hits, "hit_tokens": self.hit_tokens, "misses": self.misses,
                "published_blocks": self.published_blocks, "exact_token_block_validation": True,
                "fallback": False}

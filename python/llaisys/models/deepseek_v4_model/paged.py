"""V4 paged latent execution using the existing C++ block allocator.

The attention kernel reads the pool directly through physical slot indices.
This first layout retains full-resolution slots until request release (no window
eviction yet). Indexer uses an explicit gathered compressed-key reference view.
C++ PagedCacheStorage owns the default payload; Python/Torch views and launches
are still a bridge, not the final C++ batch executor.
"""

from types import MappingProxyType
from dataclasses import asdict
import weakref

import torch

from ..deepseek_v4_backends import OperatorContract, OperatorRegistry
from .state import CompressorState, LayerState, RequestState


CACHE_CONTRACTS = MappingProxyType({
    "cache_map_slots": OperatorContract(1,
        "integer logical slots, int64 device block table, component width, physical block stride, offset",
        "int64 physical slots in the same shape; negative logical slots remain -1"),
    "cache_write": OperatorContract(1,
        "BF16 pool[1,L,D], BF16 values[1,T,D], unique int64 physical slots[T]; same device",
        "none; scatter values into owned physical slots", "pool only at the specified slots"),
    "cache_read": OperatorContract(1,
        "BF16 pool[1,L,D], valid int64 physical slots[T]; same device",
        "owned BF16 [1,T,D] gathered reference view; used by Indexer, not attention"),
    "cache_copy_block": OperatorContract(1,
        "BF16 pool[1,L,D], source and destination physical block IDs, component stride in slots",
        "none; copy one complete component block to a distinct owned destination", "destination block only"),
})


def torch_cache_ops():
    def map_slots(logical, table, width, stride, offset):
        if width <= 0 or stride < width + offset or offset < 0:
            raise ValueError("invalid cache component layout")
        safe = logical.clamp_min(0).long()
        slots = table[safe // width] * stride + safe % width + offset
        return torch.where(logical >= 0, slots, -1)

    def write(pool, values, slots):
        if (pool.ndim != 3 or values.ndim != 3 or pool.shape[0] != 1 or values.shape[0] != 1
                or pool.shape[2] != values.shape[2] or slots.ndim != 1
                or slots.numel() != values.shape[1] or slots.dtype != torch.int64
                or pool.dtype != torch.bfloat16 or values.dtype != pool.dtype
                or values.device != pool.device or slots.device != pool.device):
            raise ValueError("cache write requires matching single-sequence BF16 tensors and int64 slots")
        pool.index_copy_(1, slots, values)

    def read(pool, slots):
        if slots.ndim != 1 or slots.dtype != torch.int64 or slots.device != pool.device:
            raise ValueError("cache read requires device int64 slots")
        return pool.index_select(1, slots)

    def copy_block(pool, source, destination, stride):
        if (pool.ndim != 3 or pool.shape[0] != 1 or pool.dtype != torch.bfloat16
                or any(type(value) is not int for value in (source, destination, stride))
                or stride <= 0 or source < 0 or destination < 0 or source == destination
                or (max(source, destination) + 1) * stride > pool.shape[1]):
            raise ValueError("block copy requires valid distinct BF16 component blocks")
        pool[:, destination * stride:(destination + 1) * stride].copy_(pool[:, source * stride:(source + 1) * stride])

    registry = OperatorRegistry(CACHE_CONTRACTS)
    for name, function in (("cache_map_slots", map_slots), ("cache_write", write), ("cache_read", read),
                           ("cache_copy_block", copy_block)):
        registry.register("torch-paged", name, function, version=str(torch.__version__), contract=CACHE_CONTRACTS[name])
    return registry.bind({name: "torch-paged" for name in CACHE_CONTRACTS})


class PagedCachePool:
    """One model/weight generation and one CUDA stream, multiple serial requests.

    Block IDs and references belong to C++ CacheBlockPool. Only this model-specific
    layer knows the full-resolution, compressed and Indexer payload dimensions.
    Prefix sharing is explicit; batch > 1 and cross-stream concurrency are not enabled.
    """

    def __init__(self, model, *, num_blocks, block_size=128, ops=None, storage_backend="cpp", enable_prefix_cache=False):
        from llaisys import _C
        if type(num_blocks) is not int or num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if storage_backend not in ("cpp", "torch"):
            raise ValueError("select explicit cpp or torch paged storage")
        if type(enable_prefix_cache) is not bool:
            raise ValueError("enable_prefix_cache must be an explicit boolean")
        cfg = model.config
        if any(r and block_size % r for r in cfg.compress_ratios):
            raise ValueError("block_size must be divisible by every layer's compression ratio")
        if model._load_failed or model.embed.weight.is_meta:
            raise RuntimeError("load valid model weights before allocating a paged pool")
        if num_blocks * (block_size + block_size // 4) >= 2**31:
            raise ValueError("physical latent indices must fit int32 attention metadata")
        self.owner, self.config = model._state_owner, cfg
        self.device = model.embed.weight.device
        self.stream = torch.cuda.current_stream(self.device).cuda_stream if self.device.type == "cuda" else None
        self.blocks = _C.CacheBlockPool(num_blocks)
        self.block_size, self.ops = block_size, ops if ops is not None else torch_cache_ops()
        self.storage_backend, self.native_storage = storage_backend, None
        selected = self.ops.report()
        if (set(selected) != set(CACHE_CONTRACTS)
                or any(selected[name]["contract"] != asdict(contract) for name, contract in CACHE_CONTRACTS.items())):
            raise ValueError("select exactly the V4 cache operator contracts")
        self.payloads, self.index_payloads = [], []
        if storage_backend == "cpp":
            if not getattr(_C, "v4_native_storage_available", False):
                raise RuntimeError("native V4 storage is unavailable; rebuild pybind with --dlpack-include, or explicitly select torch storage")
            self.native_storage = _C.V4PagedStorage(num_blocks, block_size, cfg.head_dim, cfg.index_head_dim,
                                                   list(cfg.compress_ratios), self.device.type,
                                                   self.device.index or 0, self.stream or 0)
        for index, ratio in enumerate(cfg.compress_ratios):
            stride = block_size + (block_size // ratio if ratio else 0)
            if self.native_storage is not None:
                self.payloads.append(torch.utils.dlpack.from_dlpack(self.native_storage.view(index, "latent")).zero_())
                self.index_payloads.append(torch.utils.dlpack.from_dlpack(self.native_storage.view(index, "index")).zero_()
                                           if ratio == 4 else None)
            else:
                self.payloads.append(torch.zeros(1, num_blocks * stride, cfg.head_dim,
                                                 dtype=torch.bfloat16, device=self.device))
                self.index_payloads.append(torch.zeros(1, num_blocks * (block_size // 4), cfg.index_head_dim,
                                                       dtype=torch.bfloat16, device=self.device) if ratio == 4 else None)
        from .model import rope_frequencies
        self.freqs = {compressed: rope_frequencies(cfg, compressed, self.device)
                      for compressed in {bool(r) for r in cfg.compress_ratios}}
        from .prefix import V4PrefixCache
        self.prefix = V4PrefixCache(self) if enable_prefix_cache else None
        self.cow_copies = 0

    def validate(self, model):
        if self.owner is not model._state_owner or self.config != model.config or self.device != model.embed.weight.device:
            raise ValueError("paged pool belongs to another model, weight generation, configuration or device")
        self.check_stream()

    def check_stream(self):
        if self.stream is not None and torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("paged pool requires its owning CUDA stream; cross-stream sharing is not enabled")

    def new_request(self):
        self.check_stream()
        return PagedRequestState(self)

    def report(self):
        payload_bytes = sum(t.numel() * t.element_size() for t in (*self.payloads, *self.index_payloads) if t is not None)
        return {"layout": "v4-interleaved-full-and-compressed-latent-v1", "block_size": self.block_size,
                "num_blocks": self.blocks.num_total, "num_free": self.blocks.num_free,
                "num_cached": self.blocks.num_cached, "payload_bytes": payload_bytes,
                "allocator": "existing-cpp-block-manager-pybind",
                "tensor_storage": "cpp-paged-cache-storage" if self.native_storage is not None else "torch-device-pool",
                "storage_backend": self.storage_backend,
                "native_storage": self.native_storage.report() if self.native_storage is not None else None,
                "attention_reads_pool_directly": True, "attention_history_gather": False,
                "indexer_history_gather": True, "full_resolution_window_eviction": False,
                "prefix_cache": self.prefix.report() if self.prefix is not None else False,
                "cow_block_copies": self.cow_copies, "fallback": False, "operators": self.ops.report()}


class PagedLayer:
    def __init__(self, request, index, ratio):
        self._request, self.index, self.ratio = weakref.ref(request), index, ratio

    @property
    def request(self):
        request = self._request()
        if request is None or request.closed:
            raise RuntimeError("paged layer belongs to a released request")
        return request

    @property
    def pool(self):
        return self.request.pool.payloads[self.index]

    def slots(self, logical, component="window"):
        cache, ratio = self.request.pool, self.ratio
        block_size = cache.block_size
        if component not in ("window", "compressed", "index") or (component != "window" and not ratio):
            raise ValueError("invalid V4 cache component")
        width = block_size if component == "window" else block_size // ratio
        stride = block_size + (block_size // ratio if ratio else 0)
        offset = block_size if component == "compressed" else 0
        if component == "index":
            if ratio != 4:
                raise ValueError("only ratio-4 layers have an Indexer cache")
            stride = width
        return cache.ops.function("cache_map_slots")(logical, self.request.block_table, width, stride, offset)

    def write(self, start, values, component="window"):
        logical = torch.arange(start, start + values.shape[1], device=values.device)
        target = self.request.pool.index_payloads[self.index] if component == "index" else self.pool
        self.request.pool.ops.function("cache_write")(target, values, self.slots(logical, component))

    def write_compressed(self, start, values):
        self.write(start, values, "compressed")

    def write_index(self, start, values):
        self.write(start, values, "index")

    def read_index(self, count):
        logical = torch.arange(count, device=self.pool.device)
        return self.request.pool.ops.function("cache_read")(
            self.request.pool.index_payloads[self.index], self.slots(logical, "index"))

    def capture_compressor(self, position, kv, scores, ape, *, component):
        request, ratio = self.request, self.ratio
        if request.pool.prefix is None or ratio != 4:
            return
        if component not in ("compressed", "index"):
            raise ValueError("invalid compressor snapshot component")
        dim = kv.shape[-1] // 2
        state = request.layers[self.index].compressor if component == "compressed" else request.layers[self.index].index_compressor
        size, end = request.pool.block_size, position + kv.shape[1]
        for boundary in range((position // size + 1) * size, end + 1, size):
            offset = boundary - position
            if offset >= ratio:
                values = kv[:, offset-ratio:offset, :dim].clone()
                logits = scores[:, offset-ratio:offset, :dim] + ape[:, :dim]
            else:
                partial = ratio - offset
                values = torch.cat((state.kv[:, ratio:ratio+partial, :dim], kv[:, :offset, :dim]), dim=1)
                logits = torch.cat((state.scores[:, ratio:ratio+partial, :dim],
                                    scores[:, :offset, :dim] + ape[partial:, :dim]), dim=1)
            request.prefix_snapshots.setdefault(boundary, {})[(self.index, component)] = (values, logits)


class PagedRequestState(RequestState):
    def __init__(self, pool):
        cfg = pool.config
        self.pool = pool
        self.lease = pool.blocks.allocate(0)
        self.block_table = torch.empty(0, dtype=torch.int64, device=pool.device)
        self.token_history = (torch.empty(cfg.max_seq_len, dtype=torch.int64, device=pool.device)
                              if pool.prefix is not None else None)
        self.prefix_snapshots, self.prefix_hit_tokens = {}, 0
        super().__init__(pool.owner, [], 1, cfg.max_seq_len)
        for index, ratio in enumerate(cfg.compress_ratios):
            self.layers.append(LayerState(
                None, pool.freqs[bool(ratio)],
                CompressorState.allocate(1, ratio, cfg.head_dim, pool.device) if ratio else None,
                None, CompressorState.allocate(1, 4, cfg.index_head_dim, pool.device) if ratio == 4 else None,
                PagedLayer(self, index, ratio)))

    def prepare(self, count):
        self.pool.check_stream()
        end = self.position + count
        needed = (end + self.pool.block_size - 1) // self.pool.block_size
        ids = self.lease.block_ids
        partial = self.position % self.pool.block_size
        cow = bool(partial and self.pool.blocks.metadata(ids[-1]).ref_count > 1)
        if needed - len(ids) + int(cow) > self.pool.blocks.num_free:
            raise RuntimeError("insufficient free cache blocks including copy-on-write")
        if cow:
            replacement = self.pool.blocks.allocate(1)
            try:
                new_id, old_id = replacement.block_ids[0], ids[-1]
                if self.pool.prefix is not None:
                    self.pool.prefix.discard_recycled([new_id])
                table = torch.tensor(ids[:-1] + [new_id], dtype=torch.int64, device=self.pool.device)
                copy_block = self.pool.ops.function("cache_copy_block")
                for payload, index_payload, ratio in zip(self.pool.payloads, self.pool.index_payloads, self.pool.config.compress_ratios):
                    copy_block(payload, old_id, new_id, self.pool.block_size + (self.pool.block_size // ratio if ratio else 0))
                    if index_payload is not None:
                        copy_block(index_payload, old_id, new_id, self.pool.block_size // 4)
                self.lease.replace(len(ids) - 1, replacement)
                ids[-1] = new_id
                self.block_table = table
                self.pool.cow_copies += 1
            finally:
                replacement.close()
        if needed > len(ids):
            extra = self.pool.blocks.allocate(needed - len(ids))
            try:
                if self.pool.prefix is not None:
                    self.pool.prefix.discard_recycled(extra.block_ids)
                table = torch.tensor(ids + extra.block_ids, dtype=torch.int64, device=self.pool.device)
                self.lease.append(extra)
                self.block_table = table
            finally:
                extra.close()

    def commit(self, count):
        end, size = self.position + count, self.pool.block_size
        counts = [min(size, end - start) for start in range(0, end, size)]
        self.lease.mark_computed_counts(counts)
        self.position = end
        if self.pool.prefix is not None:
            # Layers without overlap need no resume payload at aligned boundaries.
            for boundary in range(size, end + 1, size):
                self.prefix_snapshots.setdefault(boundary, {})

    def record_inputs(self, input_ids):
        if self.token_history is not None:
            self.token_history[self.position:self.position + input_ids.shape[1]].copy_(input_ids[0])

    def attach_prefix(self, tokens):
        if self.pool.prefix is None:
            raise RuntimeError("Prefix Cache was not explicitly enabled")
        return self.pool.prefix.attach(self, tokens)

    def publish_prefix(self):
        if self.pool.prefix is None:
            raise RuntimeError("Prefix Cache was not explicitly enabled")
        return self.pool.prefix.publish(self)

    def fork(self):
        if self.closed or not self.valid:
            raise RuntimeError("cannot fork a closed or invalid request")
        self.pool.check_stream()
        child = self.pool.new_request()
        try:
            shared = self.lease.share()
            child.lease.close()
            child.lease = shared
            child.block_table = self.block_table.clone()
            for source, destination in zip(self.layers, child.layers):
                for name in ("compressor", "index_compressor"):
                    current, target = getattr(source, name), getattr(destination, name)
                    if current is not None:
                        target.kv.copy_(current.kv)
                        target.scores.copy_(current.scores)
            if self.token_history is not None:
                child.token_history[:self.position].copy_(self.token_history[:self.position])
            child.prefix_snapshots = {boundary: dict(values) for boundary, values in self.prefix_snapshots.items()}
            child.position, child.prefix_hit_tokens = self.position, self.prefix_hit_tokens
            return child
        except Exception:
            child.close()
            raise

    def reset(self):
        if self.closed:
            raise RuntimeError("cannot reset a closed request")
        self.pool.check_stream()
        self.lease.close()
        self.lease = self.pool.blocks.allocate(0)
        self.block_table = torch.empty(0, dtype=torch.int64, device=self.pool.device)
        for layer in self.layers:
            if layer.compressor is not None:
                layer.compressor.reset()
            if layer.index_compressor is not None:
                layer.index_compressor.reset()
        self.position, self.valid = 0, True
        self.prefix_snapshots.clear()
        self.prefix_hit_tokens = 0

    def close(self):
        if self.closed:
            return
        self.pool.check_stream()
        self.lease.close()
        self.block_table = torch.empty(0, dtype=torch.int64, device=self.pool.device)
        self.token_history = None
        self.prefix_snapshots.clear()
        super().close()
        # A retained closed request (e.g. in an exception traceback) must not
        # keep the whole native pool alive after serving shutdown.
        self.pool = None

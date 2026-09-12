"""Existing serving protocol backed by the explicit Python V4 reference executor.

One worker owns one stream and a C++ paged pool. Slots execute serially, NOT as
a fused GPU batch. This adapter is an integration/reference backend, not the
final C++ model runtime. Queueing, chunk sizes and preemption policy stay in the
existing server.engine.InferenceEngine.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
import math
import threading

import torch

from .paged import PagedCachePool
from .snapshot import save_request, restore_request


def _sampling(temperature, top_k, top_p):
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0
            or type(top_k) is not int or top_k < 0
            or isinstance(top_p, bool) or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("invalid temperature, top_k or top_p")


def _owned(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        if self.closed or threading.get_ident() != self.thread:
            raise RuntimeError("V4 batch context is closed or accessed outside its owning worker thread")
        device = torch.cuda.device(self.device) if self.stream is not None else nullcontext()
        stream = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with device, stream, torch.inference_mode():
            return function(self, *args, **kwargs)
    return call


@dataclass
class BatchSnapshot:
    state: object
    pending_token: int
    rng_state: torch.Tensor

    def close(self):
        self.state.close()


class DeepSeekV4ServingModel:
    """Explicit serving facade for an already loaded independent V4 model."""

    def __init__(self, model, *, eos_id, num_blocks=None, block_size=128,
                 enable_prefix_cache=False, allow_experimental_chunking=False,
                 seed=0, observer=None):
        if model._load_failed or model.embed.weight.is_meta:
            raise RuntimeError("load valid model weights before creating the serving adapter")
        if type(eos_id) is not int or not 0 <= eos_id < model.config.vocab_size:
            raise ValueError("an explicit model-specific EOS token is required")
        if type(allow_experimental_chunking) is not bool or type(enable_prefix_cache) is not bool:
            raise ValueError("cache and experimental chunking flags must be explicit booleans")
        if enable_prefix_cache and not allow_experimental_chunking:
            raise ValueError("V4 prefix reuse requires explicit experimental shape-sensitive prefill opt-in")
        self.model, self._end_token = model, eos_id
        self.num_blocks, self.block_size = num_blocks, block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.allow_experimental_chunking = allow_experimental_chunking
        self.seed, self.observer = seed, observer
        self._lock = threading.Lock()
        self.last_report = None
        self._ready = None
        if model.embed.weight.device.type == "cuda":
            # Record the producer stream after weight loading. The worker waits
            # on this event before touching weights from its own stream.
            with torch.cuda.device(model.embed.weight.device):
                self._ready = torch.cuda.Event()
                self._ready.record(torch.cuda.current_stream(model.embed.weight.device))

    def validate_request(self, input_ids, params, max_seq_per_slot):
        if (not input_ids or any(type(value) is not int or not 0 <= value < self.model.config.vocab_size
                                 for value in input_ids)):
            raise ValueError("input_ids must contain valid model token IDs")
        _sampling(params.temperature, params.top_k, params.top_p)
        if type(params.max_tokens) is not int or params.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if len(input_ids) + params.max_tokens - 1 > max_seq_per_slot:
            raise ValueError("prompt and decode budget exceed the slot sequence capacity")

    def create_batch_context(self, max_batch_size, max_seq_per_slot):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("this reference model already has an active serving context")
        try:
            return V4BatchContext(self, max_batch_size, max_seq_per_slot)
        except Exception:
            self._lock.release()
            raise


class V4BatchContext:
    def __init__(self, facade, max_batch_size, max_seq_per_slot):
        cfg = facade.model.config
        if (type(max_batch_size) is not int or max_batch_size <= 0
                or type(max_seq_per_slot) is not int or not 0 < max_seq_per_slot <= cfg.max_seq_len):
            raise ValueError("invalid serving slot count or sequence capacity")
        self.facade, self.model = facade, facade.model
        self.thread, self.closed = threading.get_ident(), False
        self.device = self.model.embed.weight.device
        self.max_seq_per_slot = max_seq_per_slot
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.slots, self.pending = [None] * max_batch_size, [None] * max_batch_size
        self.calls = dict(prefill=0, prefill_chunk=0, decode=0, save=0, restore=0)
        self.prefill_tokens = self.prefix_hit_tokens = 0
        self.generators = [torch.Generator(device=self.device).manual_seed(facade.seed + i)
                           for i in range(max_batch_size)]
        self.pool = None
        self._initialize()

    @_owned
    def _initialize(self):
        if self.stream is not None:
            self.stream.wait_event(self.facade._ready)
        count = self.facade.num_blocks
        if count is None:
            count = len(self.slots) * math.ceil(self.max_seq_per_slot / self.facade.block_size)
        self.pool = PagedCachePool(self.model, num_blocks=count, block_size=self.facade.block_size,
                                  storage_backend="cpp", enable_prefix_cache=self.facade.enable_prefix_cache)

    def _slot(self, slot_id):
        if type(slot_id) is not int or not 0 <= slot_id < len(self.slots):
            raise ValueError("invalid V4 slot ID")
        return self.slots[slot_id]

    def _reset(self, slot_id):
        state = self._slot(slot_id)
        if state is not None:
            state.close()
        self.slots[slot_id], self.pending[slot_id] = None, None

    @_owned
    def slot_reset(self, slot_id):
        self._reset(slot_id)

    def _sample(self, logits, slot_id, temperature, top_k, top_p):
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("non-finite V4 logits")
        if temperature == 0 or top_k == 1:
            return int(logits.argmax(-1).item())
        scores = logits[0].float() / temperature
        if top_k:
            values, indices = torch.topk(scores, min(top_k, scores.numel()))
        else:
            values, indices = torch.sort(scores, descending=True)
        probs = torch.softmax(values, -1)
        # Retain the token crossing the nucleus threshold, including at p=1.
        probs = probs.masked_fill(probs.cumsum(-1) - probs >= top_p, 0)
        choice = torch.multinomial(probs, 1, generator=self.generators[slot_id])
        return int(indices[choice].item())

    def _execute(self, slot_id, token_ids, stage, emit_logits, sampling):
        state = self._slot(slot_id)
        if state is None:
            state = self.pool.new_request()
            self.slots[slot_id] = state
        if not token_ids or any(type(value) is not int or not 0 <= value < self.model.config.vocab_size
                                for value in token_ids):
            raise ValueError("invalid input token IDs")
        if state.position + len(token_ids) > self.max_seq_per_slot:
            raise ValueError("execution exceeds slot capacity")
        position = state.position
        try:
            output = self.model(torch.tensor([token_ids], device=self.device, dtype=torch.int64), state,
                                emit_logits=emit_logits)
            self.calls[stage] += 1
            if stage != "decode":
                self.prefill_tokens += len(token_ids)
            if self.facade.observer is not None:
                self.facade.observer(stage, slot_id, position, list(token_ids), output.logits)
            if not emit_logits:
                return None
            result = self._sample(output.logits, slot_id, *sampling)
            self.pending[slot_id] = result
            return result
        except Exception:
            self._reset(slot_id)
            raise

    @_owned
    def prefill(self, slot_id, token_ids, temperature=0.8, top_k=50, top_p=0.9):
        _sampling(temperature, top_k, top_p)
        self._reset(slot_id)
        return self._execute(slot_id, list(token_ids), "prefill", True, (temperature, top_k, top_p))

    @_owned
    def prefill_chunk(self, slot_id, token_ids, start_pos, is_last_chunk,
                      temperature=0.8, top_k=50, top_p=0.9):
        _sampling(temperature, top_k, top_p)
        if not self.facade.allow_experimental_chunking:
            raise RuntimeError("V4 chunk/prefix numerics require explicit experimental opt-in; use full prefill otherwise")
        state = self._slot(slot_id)
        if (type(start_pos) is not int or type(is_last_chunk) is not bool
                or start_pos != (state.position if state is not None else 0)
                or self.pending[slot_id] is not None):
            raise ValueError("prefill chunk position or request phase mismatch")
        return self._execute(slot_id, list(token_ids), "prefill_chunk", is_last_chunk, (temperature, top_k, top_p))

    @_owned
    def decode_per_request(self, active_slots, current_tokens, temperatures, top_ks, top_ps):
        count = len(active_slots)
        if any(len(values) != count for values in (current_tokens, temperatures, top_ks, top_ps)) or len(set(active_slots)) != count:
            raise ValueError("decode vectors must match and slot IDs must be unique")
        # Validate the entire batch before changing any slot.
        for sid, token, temperature, top_k, top_p in zip(active_slots, current_tokens, temperatures, top_ks, top_ps):
            _sampling(temperature, top_k, top_p)
            state = self._slot(sid)
            if (state is None or not state.valid or self.pending[sid] is None
                    or type(token) is not int or token != self.pending[sid]
                    or state.position >= self.max_seq_per_slot):
                raise ValueError("decode requires a valid slot, matching pending token and remaining capacity")
        return [self._execute(sid, [token], "decode", True, (temperature, top_k, top_p))
                for sid, token, temperature, top_k, top_p in zip(active_slots, current_tokens, temperatures, top_ks, top_ps)]

    @_owned
    def prefix_lookup(self, slot_id, token_ids):
        self._reset(slot_id)
        if self.pool.prefix is None:
            return 0
        state = self.pool.new_request()
        self.slots[slot_id] = state
        count = state.attach_prefix(token_ids)
        self.prefix_hit_tokens += count
        return count

    @_owned
    def prefix_publish(self, slot_id, token_ids):
        state = self._slot(slot_id)
        if self.pool.prefix is None:
            return False
        if (state is None or self.pending[slot_id] is None or len(token_ids) != state.position
                or state.token_history[:state.position].tolist() != list(token_ids)):
            raise ValueError("publish must describe the actual completed prompt")
        return state.publish_prefix()

    @_owned
    def slot_get_pos(self, slot_id):
        state = self._slot(slot_id)
        return state.position if state is not None else 0

    @_owned
    def slot_save(self, slot_id):
        state = self._slot(slot_id)
        if state is None or self.pending[slot_id] is None:
            raise RuntimeError("preemption requires a completed prefill/decode step")
        snapshot = BatchSnapshot(save_request(state), self.pending[slot_id], self.generators[slot_id].get_state().clone())
        self.calls["save"] += 1
        return snapshot

    @_owned
    def slot_restore(self, slot_id, snapshot):
        if (not isinstance(snapshot, BatchSnapshot) or self._slot(slot_id) is not None
                or snapshot.state.position > self.max_seq_per_slot
                or type(snapshot.pending_token) is not int
                or not 0 <= snapshot.pending_token < self.model.config.vocab_size):
            raise ValueError("restore requires an empty slot and V4 snapshot")
        state = restore_request(self.pool, snapshot.state)
        try:
            self.generators[slot_id].set_state(snapshot.rng_state)
        except Exception:
            state.close()
            raise
        self.slots[slot_id], self.pending[slot_id] = state, snapshot.pending_token
        self.calls["restore"] += 1

    @_owned
    def get_free_blocks(self):
        return self.pool.blocks.num_free

    @_owned
    def get_total_blocks(self):
        return self.pool.blocks.num_total

    @_owned
    def get_block_size(self):
        return self.pool.block_size

    @_owned
    def close(self):
        for slot_id in range(len(self.slots)):
            self._reset(slot_id)
        if self.pool.prefix is not None:
            self.pool.prefix.clear()
        if self.stream is not None:
            self.stream.synchronize()
        self.facade.last_report = {
            "executor": "independent-python-reference-serving-adapter",
            "fused_gpu_batch": False, "cpp_model_execution": False,
            "cuda_graph": False, "fallback": False, "mtp": False,
            "worker_owned_stream": self.stream.cuda_stream if self.stream is not None else None,
            "calls": dict(self.calls), "prefill_tokens": self.prefill_tokens,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "experimental_chunking": self.facade.allow_experimental_chunking,
            "logits_observer": self.facade.observer is not None, "pool": self.pool.report()}
        self.pool = None
        self.closed = True
        self.facade._lock.release()

"""Existing InferenceEngine protocol over native C++ SchedulePlan execution.

No tensor arithmetic or per-operator Python callbacks occur here. The facade
owns the Session's use while alive; callers must not execute/close that Session
concurrently. Current native storage is continuous/ring, NOT paged/Prefix.
Python continues to choose admission, chunk boundaries and cancellation.
"""
from functools import wraps
import math
import threading


def _sampling(temperature, top_k, top_p):
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0
            or type(top_k) is not int or top_k < 0
            or isinstance(top_p, bool) or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p) or not 0 < top_p <= 1
            or not (temperature == 0 or top_k == 1)):
        raise ValueError("native base model requires explicit valid greedy sampling")
    return dict(temperature=temperature, top_k=top_k, top_p=top_p)


def _owned(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        if self.closed or threading.get_ident() != self.thread:
            raise RuntimeError("native batch context is closed or outside its owning worker thread")
        return function(self, *args, **kwargs)
    return call


class DeepSeekV4NativeServingModel:
    """Borrow a loaded native Session; close the Session after stopping engines.

    An optional step observer receives CPU logits for numerical diagnostics.
    With no observer, only token IDs cross the execution boundary. This facade
    does not import the reference model or construct Torch tensors.
    """
    def __init__(self, session, *, eos_id, observer=None):
        info = session.info()
        if (type(eos_id) is not int or not 0 <= eos_id < info["vocabulary"]
                or info["active_slots"] != 0):
            raise ValueError("native facade requires a valid EOS and an idle Session")
        if observer is not None and not callable(observer):
            raise ValueError("observer must be callable or None")
        self.session, self._end_token, self.observer = session, eos_id, observer
        self.capacity, self.vocabulary, self.slots = info["capacity"], info["vocabulary"], info["slots"]
        self.experimental_chunking = info["experimental_chunking"]
        self._lock = threading.Lock()
        self.last_report = None

    def validate_engine_config(self, *, max_batch_size, max_seq_per_slot,
                               prefill_chunk_size, enable_block_prefix_cache):
        if (type(max_batch_size) is not int or not 0 < max_batch_size <= self.slots
                or type(max_seq_per_slot) is not int or not 0 < max_seq_per_slot <= self.capacity):
            raise ValueError("engine slots/capacity exceed the native Session")
        if enable_block_prefix_cache is not False:
            raise ValueError("native Session has no paged Prefix Cache; explicitly disable it")
        if not self.experimental_chunking and prefill_chunk_size < max_seq_per_slot:
            raise ValueError("native published baseline requires full prefill; set chunk size to slot capacity")

    def validate_request(self, input_ids, params, max_seq_per_slot):
        if not input_ids or any(type(t) is not int or not 0 <= t < self.vocabulary for t in input_ids):
            raise ValueError("invalid native model token IDs")
        _sampling(params.temperature, params.top_k, params.top_p)
        if (type(params.max_tokens) is not int or params.max_tokens <= 0
                or len(input_ids) + params.max_tokens - 1 > min(max_seq_per_slot, self.capacity)):
            raise ValueError("request exceeds native slot capacity")

    def create_batch_context(self, max_batch_size, max_seq_per_slot):
        self.validate_engine_config(max_batch_size=max_batch_size, max_seq_per_slot=max_seq_per_slot,
                                    prefill_chunk_size=max_seq_per_slot, enable_block_prefix_cache=False)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("native Session already has an active serving context")
        try:
            if self.session.info()["active_slots"]:
                raise RuntimeError("native Session was modified outside its serving facade")
            return NativeBatchContext(self, max_batch_size, max_seq_per_slot)
        except Exception:
            self._lock.release()
            raise


class NativeBatchContext:
    def __init__(self, facade, slots, capacity):
        self.facade, self.session = facade, facade.session
        self.capacity = capacity
        self.thread, self.closed = threading.get_ident(), False
        self.positions, self.pending = [0] * slots, [None] * slots
        self.step = 0
        self.calls = dict(prefill=0, prefill_chunk=0, decode=0, reset=0)
        self.prefill_tokens = 0
        self.max_decode_batch = 0

    def _slot(self, slot):
        if type(slot) is not int or not 0 <= slot < len(self.positions):
            raise ValueError("invalid native serving slot")

    def _tokens(self, tokens):
        values = list(tokens)
        if not values or any(type(t) is not int or not 0 <= t < self.facade.vocabulary for t in values):
            raise ValueError("invalid native serving token IDs")
        return values

    def _reset_many(self, slots):
        self.step += 1
        self.session.execute(dict(step_id=self.step, reset_slots=list(slots)))
        for sid in slots:
            self.positions[sid], self.pending[sid] = 0, None
        self.calls["reset"] += len(slots)

    def _execute(self, plan, updates):
        self.step += 1
        plan["step_id"] = self.step
        try:
            response = self.session.execute(plan, capture_logits=self.facade.observer is not None)
            expected = [sid for sid, _, _, _, emit in updates if emit]
            rows = response["outputs"]
            if (response["step_id"] != self.step or [r["slot_id"] for r in rows] != expected
                    or any(r["request_id"] != r["slot_id"] or type(r["token_id"]) is not int
                           or not 0 <= r["token_id"] < self.facade.vocabulary for r in rows)):
                raise RuntimeError("native execution response does not match the supplied SchedulePlan")
            outputs = {row["slot_id"]: row for row in rows}
            for sid, position, tokens, stage, emit in updates:
                self.positions[sid] = position + len(tokens)
                self.pending[sid] = outputs[sid]["token_id"] if emit else None
                self.calls[stage] += 1
                if stage != "decode":
                    self.prefill_tokens += len(tokens)
                if self.facade.observer is not None:
                    self.facade.observer(stage, sid, position, tokens, outputs[sid]["logits"] if emit else None)
            return [outputs[sid]["token_id"] for sid in expected]
        except Exception:
            # C++ rejects plans atomically or drops all partially executed slots.
            # Also reset after a diagnostic callback/response validation fails.
            self._reset_many([item[0] for item in updates])
            raise

    @_owned
    def slot_reset(self, slot_id):
        self._slot(slot_id)
        self._reset_many([slot_id])

    @_owned
    def prefill(self, slot_id, token_ids, temperature=0, top_k=1, top_p=1):
        self._slot(slot_id)
        tokens, sampling = self._tokens(token_ids), _sampling(temperature, top_k, top_p)
        if len(tokens) > self.capacity:
            raise ValueError("prefill exceeds native slot capacity")
        plan = dict(reset_slots=[slot_id], prefills=[dict(slot_id=slot_id, token_ids=tokens, sampling=sampling)])
        return self._execute(plan, [(slot_id, 0, tokens, "prefill", True)])[0]

    @_owned
    def prefill_chunk(self, slot_id, token_ids, start_pos, is_last_chunk, temperature=0, top_k=1, top_p=1):
        self._slot(slot_id)
        if not self.facade.experimental_chunking:
            raise RuntimeError("native chunked prefill requires explicit experimental opt-in")
        tokens, sampling = self._tokens(token_ids), _sampling(temperature, top_k, top_p)
        if (type(start_pos) is not int or type(is_last_chunk) is not bool or start_pos != self.positions[slot_id]
                or self.pending[slot_id] is not None or start_pos + len(tokens) > self.capacity):
            raise ValueError("prefill chunk phase/position/capacity mismatch")
        result = self._execute(dict(prefills=[dict(slot_id=slot_id, token_ids=tokens, start_pos=start_pos,
            is_last_chunk=is_last_chunk, sampling=sampling)]), [(slot_id, start_pos, tokens, "prefill_chunk", is_last_chunk)])
        return result[0] if result else None

    @_owned
    def decode_per_request(self, active_slots, current_tokens, temperatures, top_ks, top_ps):
        size = len(active_slots)
        if any(len(v) != size for v in (current_tokens, temperatures, top_ks, top_ps)):
            raise ValueError("native decode vectors must have identical lengths")
        for sid in active_slots:
            self._slot(sid)
        if len(set(active_slots)) != size:
            raise ValueError("duplicate native decode slot")
        items, updates = [], []
        for sid, token, temperature, top_k, top_p in zip(active_slots, current_tokens, temperatures, top_ks, top_ps):
            sampling = _sampling(temperature, top_k, top_p)
            if (type(token) is not int or self.pending[sid] is None or token != self.pending[sid]
                    or self.positions[sid] >= self.capacity):
                raise ValueError("native decode requires matching pending token and remaining capacity")
            items.append(dict(slot_id=sid, token_id=token, sampling=sampling))
            updates.append((sid, self.positions[sid], [token], "decode", True))
        if not items:
            return []
        self.max_decode_batch = max(self.max_decode_batch, size)
        return self._execute(dict(decodes=items), updates)

    @_owned
    def slot_get_pos(self, slot_id):
        self._slot(slot_id)
        return self.positions[slot_id]

    @_owned
    def get_block_size(self):
        return 0  # Existing engine's explicit non-paged capability; no fake blocks.

    @_owned
    def get_free_blocks(self):
        return 0

    @_owned
    def get_total_blocks(self):
        return 0

    @_owned
    def slot_save(self, slot_id):
        self._slot(slot_id)
        raise NotImplementedError("native cache snapshots/preemption are not connected yet")

    @_owned
    def slot_restore(self, slot_id, snapshot):
        self._slot(slot_id)
        raise NotImplementedError("native cache snapshots/preemption are not connected yet")

    @_owned
    def close(self):
        try:
            self._reset_many(list(range(len(self.positions))))
            self.facade.last_report = dict(executor="native-cpp-session-serving-adapter", cpp_model_execution=True,
                cache_backend="continuous-ring", paged=False, prefix_cache=False, preemption=False,
                fused_gpu_batch=False, cuda_graph=False, fallback=False, mtp=False,
                experimental_chunking=self.facade.experimental_chunking, logits_observer=self.facade.observer is not None,
                calls=dict(self.calls), prefill_tokens=self.prefill_tokens, max_decode_batch=self.max_decode_batch,
                runtime=self.session.info())
        finally:
            self.closed = True
            self.facade._lock.release()

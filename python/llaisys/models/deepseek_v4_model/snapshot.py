"""Host-owned V4 preemption snapshots; no GPU block references are retained.

These are internal, in-process runtime objects, not a checkpoint file format.
Only the model layer knows compressor/latent layouts. The block manager remains
responsible solely for logical ownership, allocation and computed-token counts.
"""

from dataclasses import dataclass

import torch


@dataclass
class PagedSnapshot:
    owner: object
    config: object
    position: int
    layers: list
    token_history: torch.Tensor | None
    prefix_snapshots: dict
    closed: bool = False

    @property
    def host_bytes(self):
        tensors = [value for layer in self.layers for value in layer.values() if value is not None]
        tensors += [value for record in self.prefix_snapshots.values() for pair in record.values() for value in pair]
        if self.token_history is not None:
            tensors.append(self.token_history)
        return sum(value.numel() * value.element_size() for value in tensors)

    def close(self):
        self.layers.clear()
        self.prefix_snapshots.clear()
        self.token_history = None
        self.closed = True


def _host(tensor):
    return tensor.detach().to(device="cpu", copy=True)


@torch.inference_mode()
def save_request(state):
    if state.closed or not state.valid or state.position <= 0:
        raise RuntimeError("cannot snapshot an empty, closed or invalid V4 request")
    pool = state.pool
    pool.check_stream()
    layers = []
    for layer, ratio in zip(state.layers, pool.config.compress_ratios):
        record = {}
        for component, count in (("window", state.position),
                                 ("compressed", state.position // ratio if ratio else 0),
                                 ("index", state.position // 4 if ratio == 4 else 0)):
            if component != "window" and not count:
                continue
            logical = torch.arange(count, device=pool.device)
            payload = pool.index_payloads[layer.paged.index] if component == "index" else layer.paged.pool
            record[component] = _host(pool.ops.function("cache_read")(payload, layer.paged.slots(logical, component)))
        for name in ("compressor", "index_compressor"):
            compressor = getattr(layer, name)
            if compressor is not None:
                record[name + "_kv"] = _host(compressor.kv)
                record[name + "_scores"] = _host(compressor.scores)
        layers.append(record)
    history = _host(state.token_history[:state.position]) if state.token_history is not None else None
    records = {boundary: {key: tuple(_host(value) for value in pair) for key, pair in values.items()}
               for boundary, values in state.prefix_snapshots.items()}
    return PagedSnapshot(pool.owner, pool.config, state.position, layers, history, records)


@torch.inference_mode()
def restore_request(pool, snapshot):
    """Build a new state transactionally; failure leaves snapshot and callers intact."""
    pool.check_stream()
    if (not isinstance(snapshot, PagedSnapshot) or snapshot.closed
            or snapshot.owner is not pool.owner or snapshot.config != pool.config
            or not 0 < snapshot.position <= pool.config.max_seq_len
            or len(snapshot.layers) != len(pool.config.compress_ratios)
            or (snapshot.token_history is not None) != (pool.prefix is not None)):
        raise ValueError("incompatible, closed or foreign V4 preemption snapshot")
    state = pool.new_request()
    try:
        state.prepare(snapshot.position)
        for layer, record in zip(state.layers, snapshot.layers):
            for component in ("window", "compressed", "index"):
                if component in record:
                    layer.paged.write(0, record[component].to(pool.device), component)
            for name in ("compressor", "index_compressor"):
                compressor = getattr(layer, name)
                if compressor is not None:
                    compressor.kv.copy_(record[name + "_kv"])
                    compressor.scores.copy_(record[name + "_scores"])
        if state.token_history is not None:
            state.token_history[:snapshot.position].copy_(snapshot.token_history)
        state.prefix_snapshots = {
            boundary: {key: tuple(value.to(device=pool.device, copy=True) for value in pair)
                       for key, pair in values.items()}
            for boundary, values in snapshot.prefix_snapshots.items()}
        state.commit(snapshot.position)
        return state
    except Exception:
        state.close()
        raise

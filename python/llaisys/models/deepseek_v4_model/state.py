from dataclasses import dataclass

import torch


@dataclass
class CompressorState:
    kv: torch.Tensor
    scores: torch.Tensor

    @classmethod
    def allocate(cls, batch, ratio, dim, device):
        overlap = 2 if ratio == 4 else 1
        shape = (batch, overlap * ratio, overlap * dim)
        return cls(torch.zeros(shape, dtype=torch.float32, device=device),
                   torch.full(shape, -torch.inf, dtype=torch.float32, device=device))

    def reset(self):
        self.kv.zero_()
        self.scores.fill_(-torch.inf)


@dataclass
class LayerState:
    kv_cache: torch.Tensor | None
    freqs: torch.Tensor
    compressor: CompressorState | None
    index_cache: torch.Tensor | None
    index_compressor: CompressorState | None
    paged: object | None = None

    def reset(self):
        self.kv_cache.zero_()
        if self.compressor is not None:
            self.compressor.reset()
        if self.index_cache is not None:
            self.index_cache.zero_()
            self.index_compressor.reset()


class RequestState:
    def __init__(self, owner, layers, batch_size, max_seq_len):
        self.owner = owner
        self.layers = layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.position = 0
        self.valid = True
        self.closed = False

    def reset(self):
        if self.closed:
            raise RuntimeError("cannot reset a closed request")
        for layer in self.layers:
            layer.reset()
        self.position, self.valid = 0, True

    def close(self):
        self.layers.clear()
        self.closed, self.valid = True, False

    def prepare(self, count):
        pass

    def record_inputs(self, input_ids):
        pass

    def commit(self, count):
        self.position += count

    def validate(self, owner, input_ids):
        if self.owner is not owner:
            raise ValueError("request cache belongs to a different model or weight generation")
        if self.closed or not self.valid:
            raise RuntimeError("request cache is closed or invalid after an execution error")
        if input_ids.ndim != 2 or input_ids.shape[0] != self.batch_size or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, tokens] tensor matching the request")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input token IDs must be int32 or int64")
        if self.position + input_ids.shape[1] > self.max_seq_len:
            raise ValueError("request exceeds its allocated sequence capacity")

"""Independent single-device reference executor; scheduling stays outside it."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Attention, MoE, RMSNorm, parameter
from .state import CompressorState, LayerState, RequestState


def rope_frequencies(cfg, compressed, device):
    dim = cfg.rope_head_dim
    base = cfg.compress_rope_theta if compressed else cfg.rope_theta
    indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (base ** (indices / dim))
    if compressed and cfg.original_seq_len > 0:
        def correction(rotations):
            return dim * math.log(cfg.original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction(cfg.beta_fast)), 0)
        high = min(math.ceil(correction(cfg.beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / cfg.rope_factor * (1 - smooth) + freqs * smooth
    phase = torch.outer(torch.arange(cfg.max_seq_len, device=device), freqs)
    return torch.polar(torch.ones_like(phase), phase)


class Block(nn.Module):
    def __init__(self, layer, cfg, ops, hadamard):
        super().__init__()
        self.cfg, self.ops = cfg, ops
        self.attn = Attention(layer, cfg, ops, hadamard)
        self.ffn = MoE(layer, cfg, ops)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps, ops)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps, ops)
        width = (2 + cfg.hc_mult) * cfg.hc_mult
        for name in ("attn", "ffn"):
            setattr(self, f"hc_{name}_fn", parameter(width, cfg.hc_mult * cfg.dim, dtype=torch.float32))
            setattr(self, f"hc_{name}_base", parameter(width, dtype=torch.float32))
            setattr(self, f"hc_{name}_scale", parameter(3, dtype=torch.float32))

    def hc_pre(self, x, name):
        value = x.flatten(2).float()
        inv_rms = self.ops.function("row_inv_rms")(value, self.cfg.norm_eps)
        mixes = self.ops.function("dense_linear")(value, getattr(self, f"hc_{name}_fn")) * inv_rms
        pre, post, combination = self.ops.function("hc_split_sinkhorn")(
            mixes, getattr(self, f"hc_{name}_scale"), getattr(self, f"hc_{name}_base"),
            self.cfg.hc_mult, self.cfg.hc_sinkhorn_iters, self.cfg.hc_eps)
        output = self.ops.function("hc_sum")(pre.unsqueeze(-1) * value.view(x.shape), 2)
        return output.to(x.dtype), post, combination

    def hc_post(self, x, residual, post, combination):
        output = post.unsqueeze(-1) * x.unsqueeze(-2)
        output += self.ops.function("hc_sum")(combination.unsqueeze(-1) * residual.unsqueeze(-2), 2)
        return output.to(x.dtype)

    def forward(self, x, position, state, input_ids):
        value, post, combination = self.hc_pre(x, "attn")
        value = self.attn(self.attn_norm(value), position, state)
        x = self.hc_post(value, x, post, combination)
        value, post, combination = self.hc_pre(x, "ffn")
        value = self.ffn(self.ffn_norm(value), input_ids)
        return self.hc_post(value, x, post, combination)


class Head(nn.Module):
    def __init__(self, cfg, ops):
        super().__init__()
        self.ops = ops
        self.weight = parameter(cfg.vocab_size, cfg.dim, dtype=torch.float32)

    def forward(self, x):
        return self.ops.function("dense_linear")(x[:, -1].float(), self.weight)


@dataclass
class ModelOutput:
    # Intermediate prefill chunks commit cache state without producing logits.
    logits: torch.Tensor | None
    layer_hiddens: tuple[torch.Tensor, ...] = ()


class DeepSeekV4Model(nn.Module):
    def __init__(self, cfg, ops, hadamard, *, device="meta"):
        super().__init__()
        self.config, self.ops = cfg, ops
        self._state_owner = object()
        self._load_failed = False
        with torch.device(device):
            self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, dtype=torch.bfloat16)
            self.embed.requires_grad_(False)
            self.layers = nn.ModuleList([Block(i, cfg, ops, hadamard) for i in range(cfg.n_layers)])
            self.norm = RMSNorm(cfg.dim, cfg.norm_eps, ops)
            self.head = Head(cfg, ops)
            self.hc_head_fn = parameter(cfg.hc_mult, cfg.hc_mult * cfg.dim, dtype=torch.float32)
            self.hc_head_base = parameter(cfg.hc_mult, dtype=torch.float32)
            self.hc_head_scale = parameter(1, dtype=torch.float32)
        self.eval()

    def new_request(self, batch_size=1, *, cache_pool=None):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self._load_failed or self.embed.weight.is_meta:
            raise RuntimeError("load valid model weights before allocating request state")
        if cache_pool is not None:
            if batch_size != 1:
                raise ValueError("paged execution currently requires one sequence per request")
            cache_pool.validate(self)
            return cache_pool.new_request()
        cfg, device = self.config, self.embed.weight.device
        freqs = {compressed: rope_frequencies(cfg, compressed, device)
                 for compressed in {bool(r) for r in cfg.compress_ratios}}
        layers = []
        for ratio in cfg.compress_ratios:
            size = cfg.window_size + (cfg.max_seq_len // ratio if ratio else 0)
            layers.append(LayerState(
                torch.zeros(batch_size, size, cfg.head_dim, dtype=torch.bfloat16, device=device),
                freqs[bool(ratio)],
                CompressorState.allocate(batch_size, ratio, cfg.head_dim, device) if ratio else None,
                torch.zeros(batch_size, cfg.max_seq_len // 4, cfg.index_head_dim,
                            dtype=torch.bfloat16, device=device) if ratio == 4 else None,
                CompressorState.allocate(batch_size, 4, cfg.index_head_dim, device) if ratio == 4 else None,
            ))
        return RequestState(self._state_owner, layers, batch_size, cfg.max_seq_len)

    @torch.inference_mode()
    def forward(self, input_ids, state, *, capture_layers=False, emit_logits=True):
        """Execute one supplied chunk; the caller owns chunk/final-token policy.

        ``emit_logits=False`` skips only HC head, final norm and vocab projection.
        All transformer layers and cache commits still run. It never samples or
        implicitly replays a token to recover an omitted output.
        """
        if type(emit_logits) is not bool:
            raise ValueError("emit_logits must be an explicit boolean")
        if self._load_failed:
            raise RuntimeError("model is invalid after an incomplete weight load")
        state.validate(self._state_owner, input_ids)
        if input_ids.device != self.embed.weight.device:
            raise ValueError("token IDs must be on the model device")
        if not bool(((input_ids >= 0) & (input_ids < self.config.vocab_size)).all()):
            raise ValueError("token ID outside the vocabulary")
        if hasattr(state, "pool"):
            state.pool.validate(self)
        state.prepare(input_ids.shape[1])
        captures = []
        try:
            state.record_inputs(input_ids)
            hidden = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
            for layer, cache in zip(self.layers, state.layers):
                hidden = layer(hidden, state.position, cache, input_ids)
                if capture_layers:
                    captures.append(hidden[:, -1].detach().clone())
            logits = None
            if emit_logits:
                value = hidden.flatten(2).float()
                inv_rms = self.ops.function("row_inv_rms")(value, self.config.norm_eps)
                mixes = self.ops.function("dense_linear")(value, self.hc_head_fn) * inv_rms
                pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
                value = self.ops.function("hc_sum")(pre.unsqueeze(-1) * value.view(hidden.shape), 2).to(hidden.dtype)
                logits = self.head(self.norm(value))
            state.commit(input_ids.shape[1])
        except Exception:
            state.valid = False
            raise
        return ModelOutput(logits, tuple(captures))

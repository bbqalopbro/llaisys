"""Dequantized DeepSeek-V4-Flash correctness oracle.

The production checkpoint uses FP8/FP4 TileLang kernels.  This module keeps a
small, ordinary-PyTorch implementation of the model-specific math so new CUDA
kernels can be checked independently.  It is intentionally not a serving or
performance fallback and never silently replaces a production backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F


FP4_E2M1_TABLE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def dequantize_fp8_blocks(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Dequantize an aligned E4M3/UE8M0 weight crop to FP32."""
    rows, columns = weight.shape
    block_rows, block_columns = block_size
    expected = (math.ceil(rows / block_rows), math.ceil(columns / block_columns))
    if tuple(scale.shape) != expected:
        raise ValueError(f"scale shape {tuple(scale.shape)} does not match {expected}")
    expanded = scale.float().repeat_interleave(block_rows, 0).repeat_interleave(
        block_columns, 1
    )[:rows, :columns]
    return weight.float() * expanded


def dequantize_fp4_e2m1(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """Unpack the checkpoint's two E2M1 values per I8 byte and apply scales."""
    if packed_weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP4 weight and scale must be matrices")
    logical_columns = packed_weight.shape[1] * 2
    expected_groups = math.ceil(logical_columns / group_size)
    if scale.shape != (packed_weight.shape[0], expected_groups):
        raise ValueError("FP4 scale shape does not match packed weight")
    byte = packed_weight.view(torch.uint8)
    table = torch.tensor(FP4_E2M1_TABLE, dtype=torch.float32, device=byte.device)
    low, high = table[(byte & 0x0F).long()], table[((byte >> 4) & 0x0F).long()]
    value = torch.stack((low, high), dim=-1).flatten(1)
    expanded_scale = scale.float().repeat_interleave(group_size, 1)[:, :logical_columns]
    return value * expanded_scale


def simulate_fp8_activation_quant(
    x: torch.Tensor,
    block_size: int = 128,
    *,
    power_of_two_scale: bool = True,
) -> torch.Tensor:
    """Quantize/dequantize activations like the checkpoint's inplace QAT path."""
    if x.shape[-1] % block_size:
        raise ValueError("activation dimension must be divisible by block size")
    dtype = x.dtype
    grouped = x.float().reshape(*x.shape[:-1], -1, block_size)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    if power_of_two_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    quantized = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).reshape_as(x).to(dtype)


def simulate_fp4_activation_quant(
    x: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Quantize/dequantize activations using E2M1 and UE8M0-style scales."""
    if x.shape[-1] % block_size:
        raise ValueError("activation dimension must be divisible by block size")
    dtype = x.dtype
    grouped = x.float().reshape(*x.shape[:-1], -1, block_size)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126) / 6.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    normalized = (grouped / scale).clamp(-6, 6)
    table = torch.tensor(FP4_E2M1_TABLE, device=x.device)
    distance = (normalized.unsqueeze(-1) - table).abs()
    quantized = table[distance.argmin(-1)]
    return (quantized * scale).reshape_as(x).to(dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


def apply_rotary(
    x: torch.Tensor, freqs_cis: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    """Apply the checkpoint's adjacent-pair RoPE without mutating ``x``."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE dimension must be even")
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if x.ndim == 3:
        freqs = freqs.reshape(1, x.shape[1], -1)
    elif x.ndim == 4:
        freqs = freqs.reshape(1, x.shape[1], 1, -1)
    else:
        raise ValueError("RoPE expects [batch, sequence, (heads), dimension]")
    return torch.view_as_real(complex_x * freqs).flatten(-2).to(x.dtype)


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    *,
    base: float = 10_000.0,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if dim % 2:
        raise ValueError("RoPE dimension must be even")
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def correction(rotations: int) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (
                2 * math.log(base)
            )

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    phase = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(phase), phase).to(device=device)


def window_indices(
    window_size: int,
    batch_size: int,
    sequence_length: int,
    start_pos: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Indices used by the published circular sliding-window cache."""
    if start_pos >= window_size - 1:
        cursor = start_pos % window_size
        row = torch.cat(
            [torch.arange(cursor + 1, window_size), torch.arange(cursor + 1)]
        )
    elif start_pos > 0:
        row = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(sequence_length).unsqueeze(1)
        row = (base - window_size + 1).clamp(0) + torch.arange(
            min(sequence_length, window_size)
        )
        row = torch.where(row > base, -1, row)
    return row.to(device=device, dtype=torch.int64).unsqueeze(0).expand(
        batch_size, -1, -1
    ).contiguous()


def compressed_indices(
    ratio: int,
    batch_size: int,
    sequence_length: int,
    start_pos: int,
    offset: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if ratio <= 0:
        raise ValueError("compression ratio must be positive")
    if start_pos > 0:
        row = torch.arange((start_pos + 1) // ratio) + offset
    else:
        row = torch.arange(sequence_length // ratio).repeat(sequence_length, 1)
        mask = row >= torch.arange(1, sequence_length + 1).unsqueeze(1) // ratio
        row = torch.where(mask, -1, row + offset)
    return row.to(device=device, dtype=torch.int64).unsqueeze(0).expand(
        batch_size, -1, -1
    ).contiguous()


def sparse_latent_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference for sparse_attn, including ``-1`` masks and the zero-value sink."""
    if q.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("expected q[B,S,H,D], kv[B,N,D], indices[B,S,K]")
    if q.shape[0] != kv.shape[0] or q.shape[:2] != indices.shape[:2]:
        raise ValueError("batch/sequence dimensions do not agree")
    valid = indices >= 0
    safe = indices.clamp_min(0)
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    selected = kv[batch, safe]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), selected.float())
    scores *= scale if scale is not None else q.shape[-1] ** -0.5
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().reshape(1, 1, -1, 1).expand(*scores.shape[:-1], 1)
    probabilities = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probabilities, selected.float()).to(q.dtype)


def compress_projected_prefill(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    ratio: int,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Learned compression after wkv/wgate projection, before RoPE/quantization.

    Only complete groups are returned.  For ratio 4 the checkpoint's overlap
    rule combines the previous group's first half with the current group's
    second half.
    """
    overlap = ratio == 4
    if ratio not in (4, 128):
        raise ValueError("published DeepSeek-V4 ratios are 4 and 128")
    dimension = norm_weight.numel()
    coefficient = 2 if overlap else 1
    expected = coefficient * dimension
    if kv.shape != score.shape or kv.shape[-1] != expected:
        raise ValueError("projected kv/score shape does not match compression mode")
    groups = kv.shape[1] // ratio
    if groups == 0:
        return kv.new_empty(kv.shape[0], 0, dimension)
    kv = kv[:, : groups * ratio].float().unflatten(1, (groups, ratio))
    score = score[:, : groups * ratio].float().unflatten(1, (groups, ratio)) + ape.float()
    if overlap:
        expanded_kv = kv.new_zeros(kv.shape[0], groups, 2 * ratio, dimension)
        expanded_score = score.new_full(expanded_kv.shape, float("-inf"))
        expanded_kv[:, :, ratio:] = kv[..., dimension:]
        expanded_score[:, :, ratio:] = score[..., dimension:]
        expanded_kv[:, 1:, :ratio] = kv[:, :-1, :, :dimension]
        expanded_score[:, 1:, :ratio] = score[:, :-1, :, :dimension]
        kv, score = expanded_kv, expanded_score
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    return rms_norm(pooled.to(norm_weight.dtype), norm_weight, eps)


ScoreFunction = Literal["softmax", "sigmoid", "sqrtsoftplus"]


def route_experts(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    topk: int,
    *,
    score_function: ScoreFunction = "sqrtsoftplus",
    route_scale: float = 1.0,
    selection_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    token_to_expert: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(x.float(), gate_weight.float())
    if score_function == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_function == "sigmoid":
        scores = scores.sigmoid()
    elif score_function == "sqrtsoftplus":
        scores = F.softplus(scores).sqrt()
    else:
        raise ValueError(f"unknown score function: {score_function}")
    original = scores
    if token_to_expert is not None:
        if input_ids is None:
            raise ValueError("hash routing requires input_ids")
        indices = token_to_expert[input_ids]
    else:
        ranked = scores if selection_bias is None else scores + selection_bias.float()
        indices = ranked.topk(topk, dim=-1).indices
    weights = original.gather(-1, indices)
    if score_function != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * route_scale, indices


def swiglu_expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    *,
    limit: float = 0.0,
) -> torch.Tensor:
    gate, up = F.linear(x, w1).float(), F.linear(x, w3).float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.linear(F.silu(gate) * up, w2.float()).to(x.dtype)


def moe_reference(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    expert_w1: torch.Tensor,
    expert_w2: torch.Tensor,
    expert_w3: torch.Tensor,
    shared_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Single-rank routed + shared expert combine for dequantized weights."""
    original_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    weights, indices = weights.reshape(flat.shape[0], -1), indices.reshape(flat.shape[0], -1)
    output = torch.zeros_like(flat, dtype=torch.float32)
    for expert_id in range(expert_w1.shape[0]):
        token, slot = torch.where(indices == expert_id)
        if token.numel():
            value = swiglu_expert(
                flat[token], expert_w1[expert_id], expert_w2[expert_id],
                expert_w3[expert_id], limit=swiglu_limit,
            )
            output[token] += value.float() * weights[token, slot, None]
    output += swiglu_expert(flat, *shared_weights, limit=swiglu_limit).float()
    return output.to(x.dtype).reshape(original_shape)


def hyperconnection_split(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    *,
    iterations: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch oracle for the checkpoint's HC split and Sinkhorn balancing."""
    pre_logits, post_logits, combination = torch.split(
        mixes, (hc_mult, hc_mult, hc_mult * hc_mult), dim=-1
    )
    pre = torch.sigmoid(pre_logits * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        post_logits * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    combination = combination * scale[2] + base[2 * hc_mult :]
    combination = combination.unflatten(-1, (hc_mult, hc_mult))
    combination = combination.softmax(dim=-1) + eps
    combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        combination = combination / (
            combination.sum(dim=-1, keepdim=True) + eps
        )
        combination = combination / (
            combination.sum(dim=-2, keepdim=True) + eps
        )
    return pre, post, combination


def hyperconnection_pre(
    x: torch.Tensor,
    projection: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float = 1e-6,
    sinkhorn_iterations: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape, dtype = x.shape, x.dtype
    flat = x.flatten(2).float()
    mixes = F.linear(flat, projection.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    pre, post, combination = hyperconnection_split(
        mixes, scale.float(), base.float(), shape[2],
        iterations=sinkhorn_iterations, eps=hc_eps,
    )
    reduced = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
    return reduced.to(dtype), post, combination


def hyperconnection_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combination: torch.Tensor,
) -> torch.Tensor:
    value = post.unsqueeze(-1) * x.unsqueeze(-2)
    value += torch.sum(
        combination.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return value.to(x.dtype)


@dataclass(frozen=True)
class DeepSeekV4CacheSpec:
    """Model-owned cache payload; block allocation remains runtime-owned."""

    block_size: int
    latent_dim: int
    index_dim: int
    element_size: int
    compression_ratios: tuple[int, ...]

    def layer_bytes(self, component: str, layer: int) -> int:
        ratio = self.compression_ratios[layer]
        if component == "window_latent":
            tokens, dimension = self.block_size, self.latent_dim
        elif component == "compressed_latent":
            tokens, dimension = (0 if ratio == 0 else math.ceil(self.block_size / ratio)), self.latent_dim
        elif component == "index_latent":
            tokens, dimension = (math.ceil(self.block_size / 4) if ratio == 4 else 0), self.index_dim
        else:
            raise KeyError(component)
        return tokens * dimension * self.element_size

"""Model-specific operator contracts implemented with explicit PyTorch math.

These are correctness backends, not hidden fallbacks. In particular, the expert
executor and reference combine read device offsets on the host. A fused library
can consume the same packed device tensors without changing router semantics,
weight ownership, or the Python serving scheduler.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .deepseek_v4_reference import apply_rotary


@dataclass(frozen=True)
class ExpertDispatch:
    hidden: torch.Tensor
    route_weights: torch.Tensor
    token_indices: torch.Tensor
    expert_offsets: torch.Tensor
    num_tokens: int


def register_torch_structure(registry, *, backend, contracts, version, inv_rms):
    def same_device(*values):
        if len({value.device for value in values if value is not None}) != 1:
            raise ValueError("operator tensors must be on the same device")

    def norm(x, weight, eps):
        if (x.ndim < 2 or x.dtype not in (torch.bfloat16, torch.float32)
                or weight.dtype != torch.float32 or weight.shape != (x.shape[-1],)):
            raise ValueError("RMSNorm requires BF16/FP32 rows and matching FP32 weight")
        same_device(x, weight)
        value = x.float()
        value = value * inv_rms(value, eps)
        return (weight * value).to(x.dtype)

    def rotary(x, freqs, inverse=False):
        if (x.ndim not in (3, 4) or x.dtype not in (torch.bfloat16, torch.float32)
                or x.shape[-1] % 2 or freqs.dtype != torch.complex64
                or freqs.shape != (x.shape[1], x.shape[-1] // 2) or type(inverse) is not bool):
            raise ValueError("RoPE requires [B,S,(H),R] and complex64 [S,R/2] frequencies")
        same_device(x, freqs)
        x.copy_(apply_rotary(x, freqs, inverse=inverse))
        return x

    def index_scores(query, cache, weights):
        if (query.ndim != 4 or cache.ndim != 3 or weights.shape != query.shape[:-1]
                or query.shape[0] != cache.shape[0] or query.shape[-1] != cache.shape[-1]
                or any(value.dtype != torch.bfloat16 for value in (query, cache, weights))):
            raise ValueError("Indexer requires BF16 Q[B,S,H,D], K[B,C,D], weights[B,S,H]")
        same_device(query, cache, weights)
        scores = torch.einsum("bshd,btd->bsht", query, cache)
        return (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)

    def router(logits, bias, hash_ids, topk, score_func, scale):
        if (logits.ndim != 2 or logits.dtype != torch.float32 or type(topk) is not int
                or not 1 <= topk <= logits.shape[-1] or score_func not in ("softmax", "sigmoid", "sqrtsoftplus")):
            raise ValueError("router requires FP32 [T,E], a valid top-k and score function")
        if hash_ids is not None:
            if (bias is not None or hash_ids.shape != (logits.shape[0], topk)
                    or hash_ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError("hash routing requires integer [T,K] IDs and no bias")
        elif bias is None or bias.dtype != torch.float32 or bias.shape != (logits.shape[1],):
            raise ValueError("learned routing requires FP32 bias[E]")
        same_device(logits, bias, hash_ids)
        scores = (logits.softmax(-1) if score_func == "softmax" else logits.sigmoid()
                  if score_func == "sigmoid" else F.softplus(logits).sqrt())
        indices = hash_ids if hash_ids is not None else (scores + bias).topk(topk, dim=-1)[1]
        weights = scores.gather(1, indices.long())
        if score_func != "softmax":
            weights /= weights.sum(-1, keepdim=True)
        return weights * scale, indices

    def pool(values, scores):
        if (values.ndim not in (3, 4) or values.shape != scores.shape or values.shape[-2] == 0
                or values.dtype != torch.float32 or scores.dtype != torch.float32):
            raise ValueError("compressor pooling requires matching FP32 [B,(G),R,D]")
        same_device(values, scores)
        return (values * scores.softmax(dim=-2)).sum(dim=-2, keepdim=values.ndim == 3)

    def activation(gate, up, limit, weights=None):
        if (gate.ndim != 2 or gate.shape != up.shape
                or gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16):
            raise ValueError("expert activation requires matching BF16 [T,I] projections")
        if weights is not None and (weights.shape != (gate.shape[0], 1) or weights.dtype != torch.float32):
            raise ValueError("expert routing weights must be FP32 [T,1]")
        same_device(gate, up, weights)
        value_gate, value_up = gate.float(), up.float()
        if limit > 0:
            value_up, value_gate = value_up.clamp(-limit, limit), value_gate.clamp(max=limit)
        value = F.silu(value_gate) * value_up
        if weights is not None:
            value = weights * value
        return value.to(gate.dtype)

    def dispatch(x, weights, indices, num_experts):
        if (x.ndim != 2 or x.dtype != torch.bfloat16 or indices.ndim != 2
                or indices.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32
                or weights.shape != indices.shape or indices.shape[0] != x.shape[0]
                or type(num_experts) is not int or num_experts <= 0
                or not 1 <= indices.shape[1] <= num_experts):
            raise ValueError("dispatch requires BF16 X[T,D], FP32 weights[T,K], integer IDs[T,K], E>=K")
        same_device(x, weights, indices)
        flat = indices.flatten().long()
        # Stable sorting preserves ascending input row / choice within each
        # expert, exactly as the original torch.where(indices == expert_id).
        order = flat.argsort(stable=True)
        rows = order // indices.shape[1]
        counts = torch.bincount(flat, minlength=num_experts)
        if counts.numel() != num_experts:
            raise ValueError("dispatch expert ID out of range")
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        return ExpertDispatch(x[rows], weights.flatten()[order, None], rows, offsets, x.shape[0])

    def combine(outputs, packed):
        if (not isinstance(packed, ExpertDispatch) or outputs.dtype != torch.bfloat16
                or outputs.shape != packed.hidden.shape):
            raise ValueError("combine requires BF16 outputs in the dispatch packed layout")
        same_device(outputs, packed.hidden, packed.token_indices, packed.expert_offsets)
        result = torch.zeros(packed.num_tokens, outputs.shape[1], dtype=torch.float32, device=outputs.device)
        # Explicit reference host synchronization. Preserve FP32 accumulation
        # order instead of nondeterministic atomics across a token's experts.
        offsets = packed.expert_offsets.tolist()
        for start, end in zip(offsets, offsets[1:]):
            if start != end:
                result[packed.token_indices[start:end]] += outputs[start:end]
        return result

    for name, function in (("rms_norm", norm), ("rotary_inplace", rotary), ("indexer_scores", index_scores),
                           ("router", router), ("compressor_pool", pool), ("expert_activation", activation),
                           ("moe_dispatch", dispatch), ("moe_combine", combine)):
        registry.register(backend, name, function, version=version, contract=contracts[name])

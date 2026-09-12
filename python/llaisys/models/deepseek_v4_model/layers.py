"""Model-owned V4 layers; upstream attribution and license are in NOTICE."""

import torch
from torch import nn
from torch.nn import functional as F

from ..deepseek_v4_reference import compressed_indices, window_indices


DTYPES = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn, "fp4": torch.float4_e2m1fn_x2}


def parameter(*shape, dtype=torch.bfloat16):
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class Linear(nn.Module):
    def __init__(self, cfg, ops, in_dim, out_dim, dtype=None):
        super().__init__()
        self.ops = ops
        dtype = dtype or DTYPES[cfg.weight_dtype]
        packed = dtype == torch.float4_e2m1fn_x2
        self.weight = parameter(out_dim, in_dim // 2 if packed else in_dim, dtype=dtype)
        if dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
            if in_dim % 128:
                raise ValueError("quantized Linear requires K divisible by 128")
            shape = (out_dim, in_dim // 32) if packed else ((out_dim + 127) // 128, in_dim // 128)
            self.scale = parameter(*shape, dtype=torch.float8_e8m0fnu)
        else:
            self.register_parameter("scale", None)

    def forward(self, x):
        if self.scale is None:
            return self.ops.function("dense_linear")(x, self.weight)
        a, scales = self.ops.function("act_quant")(x, 128, "ue8m0", torch.float8_e8m0fnu)
        name = "fp4_gemm" if self.weight.dtype == torch.float4_e2m1fn_x2 else "fp8_gemm"
        return self.ops.function(name)(a, scales, self.weight, self.scale, torch.float8_e8m0fnu)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps, ops=None):
        super().__init__()
        self.weight = parameter(dim, dtype=torch.float32)
        self.eps = eps
        self.ops = ops

    def forward(self, x):
        return self.ops.function("rms_norm")(x, self.weight, self.eps)


class Compressor(nn.Module):
    def __init__(self, cfg, ops, ratio, dim, hadamard, rotate=False):
        super().__init__()
        self.cfg, self.ops = cfg, ops
        self.ratio, self.dim, self.rotate = ratio, dim, rotate
        self.hadamard = hadamard
        coff = 2 if ratio == 4 else 1
        self.ape = parameter(ratio, coff * dim, dtype=torch.float32)
        self.wkv = Linear(cfg, ops, cfg.dim, coff * dim, torch.float32)
        self.wgate = Linear(cfg, ops, cfg.dim, coff * dim, torch.float32)
        self.norm = RMSNorm(dim, cfg.norm_eps, ops)

    def forward(self, x, position, state, cache, freqs, *, cache_write=None, capture_boundary=None):
        ratio, dim, overlap = self.ratio, self.dim, self.ratio == 4
        batch, sequence = x.shape[:2]
        kv, scores = self.wkv(x.float()), self.wgate(x.float())
        if capture_boundary is not None:
            capture_boundary(position, kv, scores, self.ape)
        if position == 0:
            cutoff = sequence - sequence % ratio
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                state.kv[:, :ratio] = kv[:, cutoff-ratio:cutoff]
                state.scores[:, :ratio] = scores[:, cutoff-ratio:cutoff] + self.ape
            if sequence > cutoff:
                state.kv[:, offset:offset+sequence-cutoff] = kv[:, cutoff:]
                state.scores[:, offset:offset+sequence-cutoff] = scores[:, cutoff:] + self.ape[:sequence-cutoff]
            if cutoff == 0:
                return None
            kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
            scores = scores[:, :cutoff].unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                shape = (batch, cutoff // ratio, ratio * 2, dim)
                values = kv.new_zeros(shape)
                logits = scores.new_full(shape, -torch.inf)
                values[:, :, ratio:], logits[:, :, ratio:] = kv[..., dim:], scores[..., dim:]
                values[:, 1:, :ratio], logits[:, 1:, :ratio] = kv[:, :-1, :, :dim], scores[:, :-1, :, :dim]
                kv, scores = values, logits
            pooled = self.ops.function("compressor_pool")(kv, scores)
            selected_freqs = freqs[:cutoff:ratio]
            output_start = 0
        elif sequence > 1:
            # Join only the unfinished compression group with this chunk's
            # projected tokens. Previously emitted latent blocks stay in cache.
            partial, offset = position % ratio, ratio if overlap else 0
            scores = scores + self.ape[(torch.arange(sequence, device=x.device) + position) % ratio]
            kv = torch.cat((state.kv[:, offset:offset+partial], kv), dim=1)
            scores = torch.cat((state.scores[:, offset:offset+partial], scores), dim=1)
            complete = kv.shape[1] // ratio
            cutoff, remainder = complete * ratio, kv.shape[1] % ratio
            if remainder:
                state.kv[:, offset:offset+remainder] = kv[:, cutoff:]
                state.scores[:, offset:offset+remainder] = scores[:, cutoff:]
            if complete == 0:
                return None
            grouped_kv = kv[:, :cutoff].unflatten(1, (complete, ratio))
            grouped_scores = scores[:, :cutoff].unflatten(1, (complete, ratio))
            if overlap:
                shape = (batch, complete, 2 * ratio, dim)
                values, logits = kv.new_empty(shape), scores.new_empty(shape)
                values[:, :, ratio:], logits[:, :, ratio:] = grouped_kv[..., dim:], grouped_scores[..., dim:]
                values[:, 0, :ratio], logits[:, 0, :ratio] = state.kv[:, :ratio, :dim], state.scores[:, :ratio, :dim]
                values[:, 1:, :ratio], logits[:, 1:, :ratio] = grouped_kv[:, :-1, :, :dim], grouped_scores[:, :-1, :, :dim]
                state.kv[:, :ratio], state.scores[:, :ratio] = grouped_kv[:, -1], grouped_scores[:, -1]
            else:
                values, logits = grouped_kv, grouped_scores
            pooled = self.ops.function("compressor_pool")(values, logits)
            output_start = position // ratio
            selected_freqs = freqs[output_start*ratio:(output_start+complete)*ratio:ratio]
        else:
            slot = (ratio if overlap else 0) + position % ratio
            state.kv[:, slot] = kv.squeeze(1)
            state.scores[:, slot] = (scores + self.ape[position % ratio]).squeeze(1)
            if (position + 1) % ratio:
                return None
            if overlap:
                values = torch.cat((state.kv[:, :ratio, :dim], state.kv[:, ratio:, dim:]), dim=1)
                logits = torch.cat((state.scores[:, :ratio, :dim], state.scores[:, ratio:, dim:]), dim=1)
            else:
                values, logits = state.kv, state.scores
            pooled = self.ops.function("compressor_pool")(values, logits)
            if overlap:
                state.kv[:, :ratio] = state.kv[:, ratio:]
                state.scores[:, :ratio] = state.scores[:, ratio:]
            selected_freqs = freqs[position + 1 - ratio].unsqueeze(0)
            output_start = position // ratio
        pooled = self.norm(pooled.to(x.dtype))
        rd = self.cfg.rope_head_dim
        self.ops.function("rotary_inplace")(pooled[..., -rd:], selected_freqs)
        if self.rotate:
            pooled = self.hadamard(pooled, dim ** -0.5)
            self.ops.function("fp4_act_quant")(pooled, 32, True)
        else:
            self.ops.function("act_quant")(pooled[..., :-rd], 64, "ue8m0", torch.float8_e8m0fnu, True)
        if cache_write is None:
            cache[:, output_start:output_start+pooled.shape[1]] = pooled
        else:
            cache_write(output_start, pooled)
        return pooled


class Indexer(nn.Module):
    def __init__(self, cfg, ops, hadamard):
        super().__init__()
        self.cfg, self.ops, self.hadamard = cfg, ops, hadamard
        self.wq_b = Linear(cfg, ops, cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim)
        self.weights_proj = Linear(cfg, ops, cfg.dim, cfg.index_n_heads, torch.bfloat16)
        self.compressor = Compressor(cfg, ops, 4, cfg.index_head_dim, hadamard, True)

    def forward(self, x, qr, position, offset, state):
        cfg = self.cfg
        sequence, end = x.shape[1], position + x.shape[1]
        query = self.wq_b(qr).unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim))
        self.ops.function("rotary_inplace")(query[..., -cfg.rope_head_dim:], state.freqs[position:end])
        query = self.hadamard(query, cfg.index_head_dim ** -0.5)
        self.ops.function("fp4_act_quant")(query, 32, True)
        paged = state.paged
        self.compressor(x, position, state.index_compressor, state.index_cache, state.freqs,
                        cache_write=paged.write_index if paged is not None else None,
                        capture_boundary=(lambda *args: paged.capture_compressor(*args, component="index"))
                        if paged is not None and paged.request.pool.prefix is not None else None)
        weights = self.weights_proj(x) * (cfg.index_head_dim ** -0.5 * cfg.index_n_heads ** -0.5)
        keys = paged.read_index(end // 4) if paged is not None else state.index_cache[:, :end // 4]
        scores = self.ops.function("indexer_scores")(query, keys, weights)
        if position == 0 or sequence > 1:
            counts = torch.arange(position + 1, end + 1, device=x.device).unsqueeze(1) // 4
            mask = torch.arange(end // 4, device=x.device).repeat(sequence, 1) >= counts
            scores += torch.where(mask, -torch.inf, 0)
        indices = self.ops.function("indexer_topk")(scores, min(cfg.index_topk, end // 4), cfg.indexer_tie_policy)
        if position == 0 or sequence > 1:
            indices = torch.where(indices >= counts, -1, indices + offset)
        else:
            indices += offset
        return indices.int()


class Attention(nn.Module):
    def __init__(self, layer, cfg, ops, hadamard):
        super().__init__()
        self.cfg, self.ops = cfg, ops
        self.ratio = cfg.compress_ratios[layer]
        self.attn_sink = parameter(cfg.n_heads, dtype=torch.float32)
        self.wq_a = Linear(cfg, ops, cfg.dim, cfg.q_lora_rank)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps, ops)
        self.wq_b = Linear(cfg, ops, cfg.q_lora_rank, cfg.n_heads * cfg.head_dim)
        self.wkv = Linear(cfg, ops, cfg.dim, cfg.head_dim)
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, ops)
        self.wo_a = Linear(cfg, ops, cfg.n_heads * cfg.head_dim // cfg.o_groups,
                           cfg.o_groups * cfg.o_lora_rank, torch.bfloat16)
        self.wo_b = Linear(cfg, ops, cfg.o_groups * cfg.o_lora_rank, cfg.dim)
        self.compressor = Compressor(cfg, ops, self.ratio, cfg.head_dim, hadamard) if self.ratio else None
        self.indexer = Indexer(cfg, ops, hadamard) if self.ratio == 4 else None

    def forward(self, x, position, state):
        cfg, ratio = self.cfg, self.ratio
        batch, sequence = x.shape[:2]
        freqs = state.freqs[position:position+sequence]
        qr = self.q_norm(self.wq_a(x))
        query = self.wq_b(qr).unflatten(-1, (cfg.n_heads, cfg.head_dim))
        query *= self.ops.function("row_inv_rms")(query, cfg.norm_eps)
        self.ops.function("rotary_inplace")(query[..., -cfg.rope_head_dim:], freqs)
        latent = self.kv_norm(self.wkv(x))
        self.ops.function("rotary_inplace")(latent[..., -cfg.rope_head_dim:], freqs)
        self.ops.function("act_quant")(latent[..., :-cfg.rope_head_dim], 64, "ue8m0", torch.float8_e8m0fnu, True)
        if state.paged is not None:
            return self.paged_attention(x, qr, query, latent, position, state, freqs)
        if position > 0 and sequence > 1:
            return self.chunk_attention(x, qr, query, latent, position, state, freqs)
        indices = window_indices(cfg.window_size, batch, sequence, position, device=x.device).int()
        extra = None
        if ratio:
            offset = sequence if position == 0 else cfg.window_size
            extra = (self.indexer(x, qr, position, offset, state) if self.indexer is not None else
                     compressed_indices(ratio, batch, sequence, position, offset, device=x.device).int())
        indices = self.combine_indices(indices, extra)
        if position == 0:
            if sequence <= cfg.window_size:
                state.kv_cache[:, :sequence] = latent
            else:
                cursor = sequence % cfg.window_size
                state.kv_cache[:, cursor:cfg.window_size], state.kv_cache[:, :cursor] = (
                    latent[:, -cfg.window_size:].split([cfg.window_size-cursor, cursor], dim=1))
            if ratio:
                compressed = self.compressor(x, position, state.compressor,
                                             state.kv_cache[:, cfg.window_size:], state.freqs)
                if compressed is not None:
                    latent = torch.cat((latent, compressed), dim=1)
        else:
            state.kv_cache[:, position % cfg.window_size] = latent.squeeze(1)
            if ratio:
                self.compressor(x, position, state.compressor, state.kv_cache[:, cfg.window_size:], state.freqs)
            latent = state.kv_cache
        output = self.ops.function("sparse_attn")(query, latent, self.attn_sink, indices, cfg.head_dim ** -0.5)
        return self.project_output(output, freqs)

    def paged_attention(self, x, qr, query, latent, position, state, freqs):
        """Preserve candidate ordering, remap metadata, read latent pool directly.

        No old/new attention payload concatenation or history gather. Indexer
        still explicitly gathers its compressed scoring keys in its own path.
        """
        cfg, ratio, paged = self.cfg, self.ratio, state.paged
        batch, sequence = x.shape[:2]
        end = position + sequence
        paged.write(position, latent)
        positions = torch.arange(position, end, device=x.device)
        width = min(sequence, cfg.window_size) if position == 0 else cfg.window_size
        logical = (positions[:, None] - cfg.window_size + 1).clamp_min(0) + torch.arange(width, device=x.device)
        logical = torch.where(logical <= positions[:, None], logical, -1)
        indices = paged.slots(logical).unsqueeze(0).expand(batch, -1, -1).int().contiguous()
        extra = None
        if ratio:
            if self.indexer is not None:
                groups = self.indexer(x, qr, position, 0, state)
            else:
                groups = torch.arange(end // ratio, device=x.device)
                groups = torch.where(groups[None] < (positions[:, None] + 1) // ratio, groups[None], -1)
                groups = groups.unsqueeze(0).expand(batch, -1, -1)
            extra = paged.slots(groups, "compressed").int().contiguous()
            self.compressor(x, position, state.compressor, None, state.freqs,
                            cache_write=paged.write_compressed,
                            capture_boundary=(lambda *args: paged.capture_compressor(*args, component="compressed"))
                            if paged.request.pool.prefix is not None else None)
        indices = self.combine_indices(indices, extra)
        output = self.ops.function("sparse_attn")(query, paged.pool, self.attn_sink, indices, cfg.head_dim ** -0.5)
        return self.project_output(output, freqs)

    def chunk_attention(self, x, qr, query, latent, position, state, freqs):
        """Vectorized incremental reference path; never replays a chunk token by token.

        The temporary attention view holds at most window-1 old full-resolution
        slots, new slots, and existing compressed slots. This contiguous reference
        view is explicit and is not claimed to be a paged-kernel implementation.
        """
        cfg, ratio = self.cfg, self.ratio
        batch, sequence = x.shape[:2]
        end = position + sequence
        history = min(position, cfg.window_size - 1)
        old_slots = torch.arange(position-history, position, device=x.device) % cfg.window_size
        payload = torch.cat((state.kv_cache[:, old_slots], latent), dim=1)
        query_positions = torch.arange(position, end, device=x.device)
        # Preserve the published full-prefill candidate order: valid historical
        # slots first, then -1 padding. Leading padding changes TileLang's
        # 64-candidate online-softmax reduction groups and BF16 rounding.
        window_positions = (query_positions[:, None] - cfg.window_size + 1).clamp_min(0)
        window_positions = window_positions + torch.arange(cfg.window_size, device=x.device)
        window = torch.where(window_positions <= query_positions[:, None],
                             window_positions - (position-history), -1)
        indices = window.unsqueeze(0).expand(batch, -1, -1).int().contiguous()
        extra = None
        if ratio:
            offset = history + sequence
            if self.indexer is not None:
                extra = self.indexer(x, qr, position, offset, state)
            else:
                groups = torch.arange(end // ratio, device=x.device)
                extra = torch.where(groups[None] < (query_positions[:, None] + 1) // ratio,
                                    groups[None] + offset, -1)
                extra = extra.unsqueeze(0).expand(batch, -1, -1).int().contiguous()
            self.compressor(x, position, state.compressor, state.kv_cache[:, cfg.window_size:], state.freqs)
            payload = torch.cat((payload, state.kv_cache[:, cfg.window_size:cfg.window_size+end//ratio]), dim=1)
        indices = self.combine_indices(indices, extra)
        # Assign unique physical slots: if a chunk exceeds the window only its
        # final window survives, avoiding duplicate scatter destinations.
        retained = min(sequence, cfg.window_size)
        slots = torch.arange(end-retained, end, device=x.device) % cfg.window_size
        state.kv_cache[:, slots] = latent[:, -retained:]
        output = self.ops.function("sparse_attn")(query, payload, self.attn_sink, indices, cfg.head_dim ** -0.5)
        return self.project_output(output, freqs)

    def combine_indices(self, window, compressed):
        if self.cfg.attention_metadata_policy == "fixed":
            # Fixed segments preserve online-softmax tile grouping when a short
            # first chunk otherwise places compressed entries inside a window
            # tile. Payload slots are not duplicated or expanded.
            window = F.pad(window, (0, self.cfg.window_size-window.shape[-1]), value=-1)
            if compressed is not None:
                capacity = self.cfg.max_seq_len // self.ratio
                if self.ratio == 4:
                    capacity = min(capacity, self.cfg.index_topk)
                compressed = F.pad(compressed, (0, capacity-compressed.shape[-1]), value=-1)
        return torch.cat((window, compressed), dim=-1) if compressed is not None else window

    def project_output(self, output, freqs):
        cfg = self.cfg
        batch, sequence = output.shape[:2]
        self.ops.function("rotary_inplace")(output[..., -cfg.rope_head_dim:], freqs, True)
        grouped = output.view(batch, sequence, cfg.o_groups, -1)
        weights = self.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, -1)
        return self.wo_b(self.ops.function("grouped_linear")(grouped, weights).flatten(2))


class Gate(nn.Module):
    def __init__(self, layer, cfg, ops):
        super().__init__()
        self.cfg, self.ops = cfg, ops
        self.weight = parameter(cfg.n_routed_experts, cfg.dim)
        if layer < cfg.n_hash_layers:
            self.tid2eid = parameter(cfg.vocab_size, cfg.n_activated_experts, dtype=torch.int32)
            self.register_parameter("bias", None)
        else:
            self.register_parameter("tid2eid", None)
            self.bias = parameter(cfg.n_routed_experts, dtype=torch.float32)

    def forward(self, x, input_ids):
        cfg = self.cfg
        logits = self.ops.function("dense_linear")(x.float(), self.weight.float())
        hash_ids = self.tid2eid[input_ids] if self.tid2eid is not None else None
        return self.ops.function("router")(logits, self.bias, hash_ids, cfg.n_activated_experts,
                                           cfg.score_func, cfg.route_scale)


class Expert(nn.Module):
    def __init__(self, cfg, ops, dtype):
        super().__init__()
        self.ops = ops
        self.limit = cfg.swiglu_limit
        self.w1 = Linear(cfg, ops, cfg.dim, cfg.moe_inter_dim, dtype)
        self.w2 = Linear(cfg, ops, cfg.moe_inter_dim, cfg.dim, dtype)
        self.w3 = Linear(cfg, ops, cfg.dim, cfg.moe_inter_dim, dtype)

    def forward(self, x, weights=None):
        value = self.ops.function("expert_activation")(self.w1(x), self.w3(x), self.limit, weights)
        return self.w2(value)


class MoE(nn.Module):
    def __init__(self, layer, cfg, ops):
        super().__init__()
        self.cfg, self.ops = cfg, ops
        self.gate = Gate(layer, cfg, ops)
        self.experts = nn.ModuleList([Expert(cfg, ops, DTYPES[cfg.expert_dtype]) for _ in range(cfg.n_routed_experts)])
        self.shared_experts = Expert(cfg, ops, DTYPES[cfg.weight_dtype])

    def forward(self, x, input_ids):
        shape = x.shape
        value = x.view(-1, self.cfg.dim)
        weights, indices = self.gate(value, input_ids.flatten())
        packed = self.ops.function("moe_dispatch")(value, weights, indices, len(self.experts))
        outputs = torch.empty_like(packed.hidden)
        # Reference execution only: device metadata is reusable by a future
        # grouped-expert backend; Python serving policy is not involved here.
        offsets = packed.expert_offsets.tolist()
        for expert_id, (start, end) in enumerate(zip(offsets, offsets[1:])):
            if start != end:
                outputs[start:end] = self.experts[expert_id](packed.hidden[start:end], packed.route_weights[start:end])
        result = self.ops.function("moe_combine")(outputs, packed)
        result += self.shared_experts(value)
        return result.to(x.dtype).view(shape)

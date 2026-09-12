"""Explicit bindings for DeepSeek-V4 native correctness operators.

These operations are opt-in bring-up backends.  They never replace a missing
production kernel silently.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch


class DeepSeekV4NativeError(RuntimeError):
    pass


class DeepSeekV4NativeReferenceOps:
    def __init__(self, library_path: str | Path) -> None:
        path = Path(library_path)
        if not path.is_file():
            raise DeepSeekV4NativeError(f"native library not found: {path}")
        self.library_path = path
        self.operation_counts = {
            "sparse_attention": 0,
            "hyperconnection": 0,
            "sqrt_softplus_router": 0,
            "activation_quant": 0,
            "fp8_linear": 0,
            "fp4_linear": 0,
            "fp8_cublas_linear": 0,
            "fp4_cublas_linear": 0,
            "compress_projected": 0,
            "indexer_scores_cublas": 0,
            "indexer_topk_cub": 0,
        }
        self._library = ctypes.CDLL(str(path))
        try:
            operation = self._library.llaisysDeepSeekV4SparseAttentionReferenceTyped
        except AttributeError as error:
            raise DeepSeekV4NativeError(
                "native library does not contain the typed DeepSeek-V4 operation"
            ) from error
        operation.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int, ctypes.c_int,
        ]
        operation.restype = None
        self._sparse_attention = operation
        try:
            hyperconnection = (
                self._library.llaisysDeepSeekV4HyperconnectionSplitReference
            )
        except AttributeError as error:
            raise DeepSeekV4NativeError(
                "native library does not contain the DeepSeek-V4 HC operation"
            ) from error
        hyperconnection.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int,
        ]
        hyperconnection.restype = None
        self._hyperconnection = hyperconnection
        router = self._library.llaisysDeepSeekV4RouterReference
        router.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_int,
        ]
        router.restype = None
        self._router = router
        activation_quant = (
            self._library.llaisysDeepSeekV4ActivationQuantReference
        )
        activation_quant.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        activation_quant.restype = None
        self._activation_quant = activation_quant
        quantized_linear = (
            self._library.llaisysDeepSeekV4QuantizedLinearReference
        )
        quantized_linear.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        quantized_linear.restype = None
        self._quantized_linear = quantized_linear
        cublas_linear = self._library.llaisysDeepSeekV4QuantizedLinearCublas
        cublas_linear.argtypes = quantized_linear.argtypes
        cublas_linear.restype = None
        self._cublas_linear = cublas_linear

    def indexer_scores(self, query, latent, weights):
        """Published H=64,D=128 BF16 dot/product/sum boundaries via cuBLAS."""
        if query.ndim != 4 or query.shape[-2:] != (64, 128):
            raise DeepSeekV4NativeError("Indexer requires query[B,S,64,128]")
        batch, sequence = query.shape[:2]
        if (latent.ndim != 3 or latent.shape[0] != batch or latent.shape[2] != 128
                or weights.shape != (batch, sequence, 64) or min(batch, sequence) <= 0):
            raise DeepSeekV4NativeError("Indexer latent/weight shape mismatch")
        if query.device.type != "cuda" or any(
            t.device != query.device or t.dtype != torch.bfloat16 or not t.is_contiguous()
            for t in (query, latent, weights)
        ):
            raise DeepSeekV4NativeError("Indexer requires contiguous CUDA BF16 inputs")
        candidates = latent.shape[1]
        if max(query.numel(), batch * sequence * 64 * candidates) >= 2**31:
            raise DeepSeekV4NativeError("Indexer exceeds int32 indexing capacity")
        scores = torch.empty(batch, sequence, candidates, device=query.device, dtype=torch.float32)
        if candidates == 0:
            return scores
        dots = torch.empty(batch, sequence, 64, candidates, device=query.device, dtype=torch.bfloat16)
        operation = getattr(self, "_indexer_scores", None)
        if operation is None:
            try:
                operation = self._library.llaisysDeepSeekV4IndexerScoresCublas
            except AttributeError as error:
                raise DeepSeekV4NativeError("native library lacks Indexer score operation") from error
            operation.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
            operation.restype = ctypes.c_int
            self._indexer_scores = operation
        with torch.cuda.device(query.device):
            status = operation(
                scores.data_ptr(), dots.data_ptr(), query.data_ptr(), latent.data_ptr(), weights.data_ptr(),
                batch, sequence, candidates, torch.cuda.current_stream(query.device).cuda_stream,
            )
        if status:
            raise DeepSeekV4NativeError("native Indexer cuBLAS scoring failed")
        self.operation_counts["indexer_scores_cublas"] += 1
        return scores

    def indexer_topk(self, scores, topk, start_pos, index_offset, ratio=4):
        """Causal stable Top-K: score descending, lower index first on ties."""
        if (scores.ndim != 3 or min(scores.shape[:2]) <= 0 or scores.device.type != "cuda"
                or scores.dtype != torch.float32 or not scores.is_contiguous()):
            raise DeepSeekV4NativeError("Indexer Top-K requires contiguous CUDA FP32 scores[B,S,T]")
        batch, sequence, candidates = scores.shape
        if topk <= 0 or start_pos < 0 or index_offset < 0 or ratio != 4:
            raise DeepSeekV4NativeError("invalid Indexer Top-K parameters")
        if max(start_pos + sequence, index_offset + candidates, scores.numel() * 64) >= 2**31:
            raise DeepSeekV4NativeError("Indexer Top-K exceeds int32 indexing capacity")
        count = min(topk, candidates)
        indices = torch.empty(batch, sequence, count, device=scores.device, dtype=torch.int32)
        if count == 0:
            return indices
        operation = getattr(self, "_indexer_topk", None)
        if operation is None:
            try:
                operation = self._library.llaisysDeepSeekV4IndexerTopKCub
                size_operation = self._library.llaisysDeepSeekV4IndexerTopKWorkspaceSize
            except AttributeError as error:
                raise DeepSeekV4NativeError("native library lacks Indexer Top-K operation") from error
            operation.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] + [ctypes.c_int] * 7 + [ctypes.c_void_p]
            operation.restype = ctypes.c_int
            size_operation.argtypes = [ctypes.c_int, ctypes.c_int]
            size_operation.restype = ctypes.c_size_t
            self._indexer_topk = operation
            self._indexer_workspace_size = size_operation
            self._indexer_workspace_sizes = {}
        with torch.cuda.device(scores.device):
            key = (scores.device.index, batch * sequence, candidates)
            size = self._indexer_workspace_sizes.get(key)
            if size is None:
                size = self._indexer_workspace_size(batch * sequence, candidates)
                if not size:
                    raise DeepSeekV4NativeError("Indexer sort workspace query failed")
                self._indexer_workspace_sizes[key] = size
            workspace = torch.empty(size, dtype=torch.uint8, device=scores.device)
            status = operation(
                indices.data_ptr(), scores.data_ptr(), workspace.data_ptr(), size,
                batch, sequence, candidates, count, start_pos, ratio, index_offset,
                torch.cuda.current_stream(scores.device).cuda_stream,
            )
        if status:
            raise DeepSeekV4NativeError("native Indexer CUB Top-K failed")
        self.operation_counts["indexer_topk_cub"] += 1
        return indices

    def compress_projected(
        self, kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor,
        kv_state: torch.Tensor, score_state: torch.Tensor,
        ratio: int, start_pos: int,
    ) -> torch.Tensor:
        """Append projected tokens and return completed FP32 pooled groups.

        The caller owns request-local state and supplies consecutive positions.
        start_pos=0 resets state, including the masked previous ratio-4 group.
        RMSNorm, RoPE and activation quantization follow this operation.
        """
        if ratio not in (4, 128) or start_pos < 0:
            raise DeepSeekV4NativeError("invalid compressor ratio/start position")
        tensors = (kv, score, ape, kv_state, score_state)
        if kv.device.type != "cuda" or any(
            t.device != kv.device or t.dtype != torch.float32 or not t.is_contiguous()
            for t in tensors
        ):
            raise DeepSeekV4NativeError("compressor requires contiguous CUDA FP32 tensors")
        if kv.ndim != 3 or score.shape != kv.shape or min(kv.shape) <= 0:
            raise DeepSeekV4NativeError("expected nonempty kv/score[B,S,C*D]")
        batch, sequence, width = kv.shape
        coefficient = 2 if ratio == 4 else 1
        if width % coefficient or ape.shape != (ratio, width):
            raise DeepSeekV4NativeError("invalid compressor APE/projected width")
        state_shape = (batch, coefficient * ratio, width)
        if kv_state.shape != state_shape or score_state.shape != state_shape:
            raise DeepSeekV4NativeError("invalid compressor state shape")
        if start_pos + sequence >= 2**31 or max(kv.numel(), kv_state.numel()) >= 2**31:
            raise DeepSeekV4NativeError("compressor exceeds int32 indexing capacity")
        # Output is distinct; reject overlapping state/input storage before a
        # kernel could overwrite values another feature still needs to read.
        def overlaps(a, b):
            return (a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
                    and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size())
        if overlaps(kv_state, score_state) or any(
            overlaps(state, source)
            for state in (kv_state, score_state) for source in (kv, score, ape)
        ):
            raise DeepSeekV4NativeError("compressor state must not alias inputs or other state")
        operation = getattr(self, "_compress_projected", None)
        if operation is None:
            try:
                operation = self._library.llaisysDeepSeekV4CompressProjectedReference
            except AttributeError as error:
                raise DeepSeekV4NativeError("native library lacks the compressor operation") from error
            operation.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
            operation.restype = ctypes.c_int
            self._compress_projected = operation
        dimension = width // coefficient
        groups = (start_pos + sequence) // ratio - start_pos // ratio
        output = torch.empty(batch, groups, dimension, dtype=torch.float32, device=kv.device)
        with torch.cuda.device(kv.device):
            status = operation(
                output.data_ptr(), kv_state.data_ptr(), score_state.data_ptr(),
                kv.data_ptr(), score.data_ptr(), ape.data_ptr(),
                batch, sequence, dimension, ratio, start_pos,
                torch.cuda.current_stream(kv.device).cuda_stream,
            )
        if status != 0:
            raise DeepSeekV4NativeError("native projected compressor launch failed")
        self.operation_counts["compress_projected"] += 1
        return output

    def sparse_attention(
        self,
        query: torch.Tensor,
        latent: torch.Tensor,
        attn_sink: torch.Tensor,
        indices: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        if query.device.type != "cuda":
            raise DeepSeekV4NativeError("native BF16 sparse attention requires CUDA")
        if query.dtype not in (torch.float32, torch.bfloat16):
            raise DeepSeekV4NativeError("query dtype must be FP32 or BF16")
        if latent.dtype != query.dtype or latent.device != query.device:
            raise DeepSeekV4NativeError("latent dtype/device must match query")
        if attn_sink.dtype != torch.float32 or attn_sink.device != query.device:
            raise DeepSeekV4NativeError("attention sink must be CUDA FP32")
        if indices.dtype != torch.int32 or indices.device != query.device:
            raise DeepSeekV4NativeError("indices must be CUDA I32")
        if query.ndim != 4 or latent.ndim != 3 or indices.ndim != 3:
            raise DeepSeekV4NativeError(
                "expected query[B,S,H,D], latent[B,N,D], indices[B,S,K]"
            )
        batch, sequence, heads, dimension = query.shape
        if latent.shape[0] != batch or latent.shape[2] != dimension:
            raise DeepSeekV4NativeError("latent shape does not match query")
        if indices.shape[:2] != (batch, sequence):
            raise DeepSeekV4NativeError("index shape does not match query")
        if attn_sink.numel() != heads:
            raise DeepSeekV4NativeError("attention sink size does not match heads")
        tensors = (query, latent, attn_sink, indices)
        if not all(value.is_contiguous() for value in tensors):
            raise DeepSeekV4NativeError("native sparse attention requires contiguous tensors")

        output = torch.empty_like(query)
        dtype = 13 if query.dtype == torch.float32 else 19
        value_scale = dimension ** -0.5 if scale is None else float(scale)
        self.operation_counts["sparse_attention"] += 1
        self._sparse_attention(
            output.data_ptr(), query.data_ptr(), latent.data_ptr(),
            attn_sink.data_ptr(), indices.data_ptr(), batch, sequence, heads,
            dimension, latent.shape[1], indices.shape[2], value_scale,
            dtype, 1,
        )
        return output

    def hyperconnection_split(
        self,
        mixes: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        hc_mult: int = 4,
        iterations: int = 20,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mixes.device.type != "cuda" or mixes.dtype != torch.float32:
            raise DeepSeekV4NativeError("native Hyper-Connection requires CUDA FP32")
        if scale.device != mixes.device or scale.dtype != torch.float32:
            raise DeepSeekV4NativeError("HC scale must be CUDA FP32")
        if base.device != mixes.device or base.dtype != torch.float32:
            raise DeepSeekV4NativeError("HC base must be CUDA FP32")
        expected_width = (2 + hc_mult) * hc_mult
        if mixes.shape[-1] != expected_width or scale.numel() != 3:
            raise DeepSeekV4NativeError("invalid Hyper-Connection shape")
        if base.numel() != expected_width:
            raise DeepSeekV4NativeError("HC base size does not match hc_mult")
        if not all(value.is_contiguous() for value in (mixes, scale, base)):
            raise DeepSeekV4NativeError("native Hyper-Connection requires contiguous tensors")
        output_shape = (*mixes.shape[:-1], hc_mult)
        pre = torch.empty(output_shape, device=mixes.device, dtype=torch.float32)
        post = torch.empty_like(pre)
        combination = torch.empty(
            (*mixes.shape[:-1], hc_mult, hc_mult),
            device=mixes.device, dtype=torch.float32,
        )
        rows = mixes.numel() // expected_width
        self.operation_counts["hyperconnection"] += 1
        self._hyperconnection(
            pre.data_ptr(), post.data_ptr(), combination.data_ptr(),
            mixes.data_ptr(), scale.data_ptr(), base.data_ptr(), rows, hc_mult,
            iterations, float(eps), 1,
        )
        return pre, post, combination

    def route_sqrt_softplus(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        topk: int,
        route_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.device.type != "cuda" or logits.dtype != torch.float32:
            raise DeepSeekV4NativeError("native router requires CUDA FP32 logits")
        if selection_bias.device != logits.device or selection_bias.dtype != torch.float32:
            raise DeepSeekV4NativeError("router selection bias must be CUDA FP32")
        if logits.ndim != 2 or selection_bias.shape != (logits.shape[1],):
            raise DeepSeekV4NativeError("invalid router shape")
        if not logits.is_contiguous() or not selection_bias.is_contiguous():
            raise DeepSeekV4NativeError("native router requires contiguous tensors")
        weights = torch.empty(
            logits.shape[0], topk, device=logits.device, dtype=torch.float32
        )
        indices = torch.empty(
            logits.shape[0], topk, device=logits.device, dtype=torch.int32
        )
        self.operation_counts["sqrt_softplus_router"] += 1
        self._router(
            weights.data_ptr(), indices.data_ptr(), logits.data_ptr(),
            selection_bias.data_ptr(), logits.shape[0], logits.shape[1], topk,
            float(route_scale), 1,
        )
        return weights, indices

    def quantize_activation_(
        self,
        value: torch.Tensor,
        block_size: int,
        *,
        mode: str,
        power_of_two_scale: bool = True,
    ) -> torch.Tensor:
        if value.device.type != "cuda" or value.dtype != torch.bfloat16:
            raise DeepSeekV4NativeError("native activation quant requires CUDA BF16")
        if value.ndim == 0 or value.shape[-1] % block_size:
            raise DeepSeekV4NativeError("activation dimension is not block aligned")
        if mode not in ("fp8", "fp4"):
            raise DeepSeekV4NativeError(f"unknown activation quant mode: {mode}")
        contiguous = value if value.is_contiguous() else value.contiguous()
        columns = contiguous.shape[-1]
        rows = contiguous.numel() // columns
        self.operation_counts["activation_quant"] += 1
        self._activation_quant(
            contiguous.data_ptr(), rows, columns, block_size,
            0 if mode == "fp8" else 1, int(power_of_two_scale), 19, 1,
        )
        if contiguous.data_ptr() != value.data_ptr():
            value.copy_(contiguous)
        return value

    def quantized_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        mode: str,
        backend: str = "reference",
    ) -> torch.Tensor:
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise DeepSeekV4NativeError("native quantized linear requires CUDA BF16 input")
        if weight.device != input.device or scale.device != input.device:
            raise DeepSeekV4NativeError("weight and scale must share the input device")
        if mode == "fp8":
            out_features, in_features = weight.shape
            expected_scale = (out_features // 128, in_features // 128)
        elif mode == "fp4":
            out_features, packed_features = weight.shape
            in_features = packed_features * 2
            expected_scale = (out_features, in_features // 32)
        else:
            raise DeepSeekV4NativeError(f"unknown weight quant mode: {mode}")
        if input.shape[-1] != in_features or tuple(scale.shape) != expected_scale:
            raise DeepSeekV4NativeError("quantized linear shape/scale mismatch")
        if not all(value.is_contiguous() for value in (input, weight, scale)):
            raise DeepSeekV4NativeError("native quantized linear requires contiguous tensors")
        rows = input.numel() // in_features
        output = torch.empty(
            (*input.shape[:-1], out_features),
            device=input.device, dtype=torch.bfloat16,
        )
        if backend == "reference":
            operation = self._quantized_linear
            counter = f"{mode}_linear"
        elif backend == "cublas":
            operation = self._cublas_linear
            counter = f"{mode}_cublas_linear"
        else:
            raise DeepSeekV4NativeError(
                f"unknown quantized linear backend: {backend}"
            )
        self.operation_counts[counter] += 1
        operation(
            output.data_ptr(), input.data_ptr(), weight.data_ptr(),
            scale.data_ptr(), rows, out_features, in_features,
            0 if mode == "fp8" else 1, 19, 1,
        )
        return output


def make_native_compressor_forward(published, native_ops):
    """Explicit adapter for the published compressor's projection/postprocess.

    Only pooling and state updates are replaced. Keep the published BF16 cast
    before RMSNorm, group-start RoPE and separate index/attention QAT formats.
    This adapter alone does not enable chunked model attention or serving.
    """
    def forward(self, x, start_pos):
        if self.kv_cache is None or self.freqs_cis is None:
            raise DeepSeekV4NativeError("compressor cache and RoPE must be attached")
        batch, sequence, _ = x.shape
        ratio = self.compress_ratio
        if start_pos and start_pos != getattr(self, "_llaisys_next_pos", None):
            raise DeepSeekV4NativeError("compressor positions must be consecutive")
        end_pos = start_pos + sequence
        if end_pos > self.freqs_cis.shape[0] or end_pos // ratio > self.kv_cache.shape[1]:
            raise DeepSeekV4NativeError("compressor exceeds cache/position capacity")
        projected = x.float()
        pooled = native_ops.compress_projected(
            self.wkv(projected).contiguous(), self.wgate(projected).contiguous(),
            self.ape, self.kv_state[:batch], self.score_state[:batch],
            ratio, start_pos,
        )
        self._llaisys_next_pos = end_pos
        if pooled.shape[1] == 0:
            return None
        kv = self.norm(pooled.to(x.dtype))
        first_group = start_pos // ratio
        last_group = end_pos // ratio
        freqs = self.freqs_cis[first_group * ratio:last_group * ratio:ratio]
        published.apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs)
        if self.rotate:
            kv = published.rotate_activation(kv)
            published.fp4_act_quant(kv, published.fp4_block_size, True)
        else:
            published.act_quant(
                kv[..., :-self.rope_head_dim], 64,
                published.scale_fmt, published.scale_dtype, True,
            )
        self.kv_cache[:batch, first_group:last_group] = kv
        return kv
    return forward


def make_native_indexer_forward(published, native_ops):
    """Replace single-rank scoring/selection; retain projections, RoPE and QAT."""
    def forward(self, x, qr, start_pos, offset):
        if published.world_size != 1:
            raise DeepSeekV4NativeError("native Indexer has not implemented distributed score reduction")
        if self.n_heads != 64 or self.head_dim != 128 or self.compress_ratio != 4:
            raise DeepSeekV4NativeError("unsupported native Indexer model configuration")
        batch, sequence, _ = x.shape
        if start_pos > 0 and sequence != 1:
            raise DeepSeekV4NativeError("model Indexer incremental adapter requires one token")
        end_pos = start_pos + sequence
        freqs = self.freqs_cis[start_pos:end_pos]
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        query = self.wq_b(qr).unflatten(-1, (64, 128))
        published.apply_rotary_emb(query[..., -self.rope_head_dim:], freqs)
        query = published.rotate_activation(query)
        published.fp4_act_quant(query, published.fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        latent = self.kv_cache[:batch, :end_pos // 4].contiguous()
        scores = native_ops.indexer_scores(query.contiguous(), latent, weights.contiguous())
        return native_ops.indexer_topk(scores, self.index_topk, start_pos, offset)
    return forward


__all__ = ["DeepSeekV4NativeError", "DeepSeekV4NativeReferenceOps",
           "make_native_compressor_forward", "make_native_indexer_forward"]

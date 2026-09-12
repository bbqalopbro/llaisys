#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H

#include "tensor.h"

__C {
    __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
    __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
    __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
    __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
    __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
    __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
    __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
    __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
    __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
    __export void llaisysSample(llaisysTensor_t out_idx, llaisysTensor_t logits, float temperature, int top_k, float top_p, uint64_t seed);
    __export void llaisysDequantize(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale);
    __export void llaisysDequantizeInt4(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale, int group_size);
    __export void llaisysLinearInt4(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight,
                                    llaisysTensor_t scale, llaisysTensor_t bias,
                                    int group_size, llaisysTensor_t residual);

    // Correctness-only DeepSeek-V4 sparse latent attention. All tensors are
    // contiguous FP32 except indices (I32). This is an observable bring-up
    // oracle, not a production fallback or serving hot-path kernel.
    __export void llaisysDeepSeekV4SparseAttentionReference(
        float *output, const float *query, const float *latent,
        const float *attn_sink, const int *indices,
        int batch_size, int query_len, int num_heads, int head_dim,
        int latent_len, int topk, float scale,
        llaisysDeviceType_t device_type);

    // Typed variant used by the real model bring-up path. F32 is available on
    // CPU/NVIDIA; BF16 is available on NVIDIA. The operation remains an
    // explicitly selected correctness backend, never an implicit fallback.
    __export void llaisysDeepSeekV4SparseAttentionReferenceTyped(
        void *output, const void *query, const void *latent,
        const float *attn_sink, const int *indices,
        int batch_size, int query_len, int num_heads, int head_dim,
        int latent_len, int topk, float scale, llaisysDataType_t dtype,
        llaisysDeviceType_t device_type);

    // Correctness-only non-hash router for the checkpoint's sqrt(softplus)
    // scoring. Selection bias changes top-k membership but not output weights.
    __export void llaisysDeepSeekV4RouterReference(
        float *weights, int *indices, const float *logits,
        const float *selection_bias, int num_tokens, int num_experts,
        int topk, float route_scale, llaisysDeviceType_t device_type);

    __export void llaisysDeepSeekV4HyperconnectionSplitReference(
        float *pre, float *post, float *combination,
        const float *mixes, const float *scale, const float *base,
        int rows, int hc_mult, int iterations, float eps,
        llaisysDeviceType_t device_type);

    // In-place QAT simulation on BF16 activations. quant_mode: 0=FP8 E4M3,
    // 1=FP4 E2M1. This exposes the checkpoint's quantization boundary while
    // native quantized GEMM is being implemented.
    __export void llaisysDeepSeekV4ActivationQuantReference(
        void *data, int rows, int columns, int block_size,
        int quant_mode, int power_of_two_scale,
        llaisysDataType_t dtype, llaisysDeviceType_t device_type);

    // Fused correctness GEMM: BF16 activation times FP8-E4M3 or packed
    // FP4-E2M1 checkpoint weight with E8M0 scales. quant_mode: 0=FP8, 1=FP4.
    __export void llaisysDeepSeekV4QuantizedLinearReference(
        void *output, const void *input, const void *weight, const void *scale,
        int rows, int out_features, int in_features, int quant_mode,
        llaisysDataType_t dtype, llaisysDeviceType_t device_type);

    // Correctness backend using the operator library for GEMM. Quantized
    // checkpoint weights are decoded to FP32, then multiplied by cuBLAS.
    // This is explicit and never selected as a silent fallback.
    __export void llaisysDeepSeekV4QuantizedLinearCublas(
        void *output, const void *input, const void *weight, const void *scale,
        int rows, int out_features, int in_features, int quant_mode,
        llaisysDataType_t dtype, llaisysDeviceType_t device_type);

    // Stateful projected compressor, CUDA FP32 correctness path. Emits
    // floor((start_pos+sequence)/ratio)-floor(start_pos/ratio) groups.
    // stream is the caller's CUDA stream. Returns 0 on success, -1 on error.
    __export int llaisysDeepSeekV4CompressProjectedReference(
        float *output, float *kv_state, float *score_state,
        const float *kv, const float *score, const float *ape,
        int batch, int sequence, int dimension, int ratio, int start_pos,
        void *stream);

    // DeepSeek-V4 Indexer: contiguous BF16 Q[B,S,64,128], K[B,T,128],
    // weights[B,S,64]. dots[B,S,64,T] is caller-owned BF16 workspace;
    // scores[B,S,T] is FP32 carrying the published BF16-rounded head sum.
    __export int llaisysDeepSeekV4IndexerScoresCublas(
        float *scores, void *dots, const void *query, const void *latent,
        const void *weights, int batch, int sequence, int candidates, void *stream);
    __export size_t llaisysDeepSeekV4IndexerTopKWorkspaceSize(int rows, int candidates);
    // Stable descending score order, ascending candidate index for ties.
    // Causal mask is applied before selection; unavailable entries are -1.
    __export int llaisysDeepSeekV4IndexerTopKCub(
        int *indices, const float *scores, void *workspace, size_t workspace_bytes,
        int batch, int sequence, int candidates, int topk, int start_pos,
        int ratio, int index_offset, void *stream);

    // Paged Attention: Q·K^T→softmax→·V with block-pooled KV-Cache
    // block_tables: [batch_size, max_blocks_per_seq] CPU-side int array
    // seq_lens: [batch_size] CPU-side int array
    // kv_quant: 0=FP32, 1=INT8, 2=INT4
    __export void llaisysPagedAttention(
        float *output, const float *query,
        void *k_pool, void *v_pool,
        int *block_tables, int *seq_lens,
        int batch_size, int num_heads, int num_kv_heads, int head_dim,
        int block_size, int max_blocks_per_seq,
        size_t pool_block_stride, size_t pool_layer_stride,
        int layer_idx, float scale,
        llaisysDeviceType_t device_type,
        int kv_quant);
}

#endif

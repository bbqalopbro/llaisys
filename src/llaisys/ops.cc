#include "llaisys/ops.h"

#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/sample/op.hpp"
#include "../ops/dequantize/op.hpp"
#include "../ops/deepseek_v4/sparse_latent_attention.hpp"
#include "../ops/deepseek_v4/hyperconnection.hpp"
#include "../ops/deepseek_v4/activation_quant.hpp"
#include "../ops/deepseek_v4/quantized_linear.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../ops/deepseek_v4/nvidia/compressor.cuh"
#include "../ops/deepseek_v4/nvidia/indexer.cuh"
#endif
#include <climits>
#include "../ops/self_attention/paged_attention.hpp"

__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias ? bias->tensor : nullptr);
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        llaisys::ops::rearrange(out->tensor, in->tensor);
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }
    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    }
    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }
    void llaisysSample(llaisysTensor_t out_idx, llaisysTensor_t logits, float temperature, int top_k, float top_p, uint64_t seed) {
        llaisys::ops::sample(out_idx->tensor, logits->tensor, temperature, top_k, top_p, seed);
    }
    void llaisysDequantize(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale) {
        llaisys::ops::dequantize(out->tensor, weight->tensor, scale->tensor);
    }
    void llaisysDequantizeInt4(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale, int group_size) {
        llaisys::ops::dequantize_int4(out->tensor, weight->tensor, scale->tensor, group_size);
    }
    void llaisysLinearInt4(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight,
                           llaisysTensor_t scale, llaisysTensor_t bias,
                           int group_size, llaisysTensor_t residual) {
        llaisys::ops::linear_int4(out->tensor, in->tensor, weight->tensor, scale->tensor,
                                  bias ? bias->tensor : nullptr, group_size,
                                  residual ? residual->tensor : nullptr);
    }
    void llaisysDeepSeekV4SparseAttentionReference(
        float *output, const float *query, const float *latent,
        const float *attn_sink, const int *indices,
        int batch_size, int query_len, int num_heads, int head_dim,
        int latent_len, int topk, float scale,
        llaisysDeviceType_t device_type) {
        llaisys::ops::deepseek_v4_sparse_attention_reference(
            output, query, latent, attn_sink, indices, batch_size, query_len,
            num_heads, head_dim, latent_len, topk, scale, device_type);
    }
    void llaisysDeepSeekV4SparseAttentionReferenceTyped(
        void *output, const void *query, const void *latent,
        const float *attn_sink, const int *indices,
        int batch_size, int query_len, int num_heads, int head_dim,
        int latent_len, int topk, float scale, llaisysDataType_t dtype,
        llaisysDeviceType_t device_type) {
        llaisys::ops::deepseek_v4_sparse_attention_reference_typed(
            output, query, latent, attn_sink, indices, batch_size, query_len,
            num_heads, head_dim, latent_len, topk, scale, dtype, device_type);
    }
    void llaisysDeepSeekV4RouterReference(
        float *weights, int *indices, const float *logits,
        const float *selection_bias, int num_tokens, int num_experts,
        int topk, float route_scale, llaisysDeviceType_t device_type) {
        llaisys::ops::deepseek_v4_router_reference(
            weights, indices, logits, selection_bias, num_tokens,
            num_experts, topk, route_scale, device_type);
    }
    void llaisysDeepSeekV4HyperconnectionSplitReference(
        float *pre, float *post, float *combination,
        const float *mixes, const float *scale, const float *base,
        int rows, int hc_mult, int iterations, float eps,
        llaisysDeviceType_t device_type) {
        llaisys::ops::deepseek_v4_hyperconnection_split_reference(
            pre, post, combination, mixes, scale, base, rows, hc_mult,
            iterations, eps, device_type);
    }
    void llaisysDeepSeekV4ActivationQuantReference(
        void *data, int rows, int columns, int block_size,
        int quant_mode, int power_of_two_scale,
        llaisysDataType_t dtype, llaisysDeviceType_t device_type) {
        llaisys::ops::deepseek_v4_activation_quant_reference(
            data, rows, columns, block_size, quant_mode, power_of_two_scale,
            dtype, device_type);
    }
    void llaisysDeepSeekV4QuantizedLinearReference(
        void *output, const void *input, const void *weight, const void *scale,
        int rows, int out_features, int in_features, int quant_mode,
        llaisysDataType_t dtype, llaisysDeviceType_t device_type) {
        llaisys::ops::deepseek_v4_quantized_linear_reference(
            output, input, weight, scale, rows, out_features, in_features,
            quant_mode, dtype, device_type);
    }
    void llaisysDeepSeekV4QuantizedLinearCublas(
        void *output, const void *input, const void *weight, const void *scale,
        int rows, int out_features, int in_features, int quant_mode,
        llaisysDataType_t dtype, llaisysDeviceType_t device_type) {
        llaisys::ops::deepseek_v4_quantized_linear_cublas(
            output, input, weight, scale, rows, out_features, in_features,
            quant_mode, dtype, device_type);
    }
    int llaisysDeepSeekV4CompressProjectedReference(
        float *output, float *kv_state, float *score_state,
        const float *kv, const float *score, const float *ape,
        int batch, int sequence, int dimension, int ratio, int start_pos,
        void *stream) {
        if (!kv_state || !score_state || !kv || !score || !ape ||
            batch <= 0 || sequence <= 0 || dimension <= 0 || start_pos < 0 ||
            (ratio != 4 && ratio != 128) ||
            (!output && (start_pos + sequence) / ratio > start_pos / ratio)) return -1;
#ifdef ENABLE_NVIDIA_API
        try {
            llaisys::ops::nvidia::deepseek_v4_compress_projected_reference(
                output, kv_state, score_state, kv, score, ape,
                batch, sequence, dimension, ratio, start_pos, stream);
            return 0;
        } catch (...) {
            return -1;
        }
#else
        (void)stream;
        return -1;
#endif
    }
    int llaisysDeepSeekV4IndexerScoresCublas(
        float *scores, void *dots, const void *query, const void *latent,
        const void *weights, int batch, int sequence, int candidates, void *stream) {
        if (!scores || !dots || !query || !latent || !weights ||
            batch <= 0 || sequence <= 0 || candidates <= 0 ||
            static_cast<long long>(batch) * sequence > INT_MAX / 8192 ||
            static_cast<long long>(batch) * sequence * candidates > INT_MAX / 64) return -1;
#ifdef ENABLE_NVIDIA_API
        try {
            llaisys::ops::nvidia::deepseek_v4_indexer_scores_cublas(
                scores, dots, query, latent, weights, batch, sequence, candidates, stream);
            return 0;
        } catch (...) { return -1; }
#else
        (void)stream;
        return -1;
#endif
    }
    size_t llaisysDeepSeekV4IndexerTopKWorkspaceSize(int rows, int candidates) {
        if (rows <= 0 || candidates <= 0 ||
            static_cast<long long>(rows) * candidates > INT_MAX / 64) return 0;
#ifdef ENABLE_NVIDIA_API
        try { return llaisys::ops::nvidia::deepseek_v4_indexer_topk_workspace(rows, candidates); }
        catch (...) { return 0; }
#else
        return 0;
#endif
    }
    int llaisysDeepSeekV4IndexerTopKCub(
        int *indices, const float *scores, void *workspace, size_t workspace_bytes,
        int batch, int sequence, int candidates, int topk, int start_pos,
        int ratio, int index_offset, void *stream) {
        if (!indices || !scores || !workspace || batch <= 0 || sequence <= 0 ||
            candidates <= 0 || topk <= 0 || topk > candidates || start_pos < 0 ||
            ratio != 4 || index_offset < 0 ||
            static_cast<long long>(start_pos) + sequence >= INT_MAX ||
            static_cast<long long>(index_offset) + candidates >= INT_MAX ||
            static_cast<long long>(batch) * sequence > INT_MAX / candidates / 64) return -1;
#ifdef ENABLE_NVIDIA_API
        try {
            llaisys::ops::nvidia::deepseek_v4_indexer_topk_cub(
                indices, scores, workspace, workspace_bytes, batch, sequence,
                candidates, topk, start_pos, ratio, index_offset, stream);
            return 0;
        } catch (...) { return -1; }
#else
        (void)workspace_bytes;
        (void)stream;
        return -1;
#endif
    }
    void llaisysPagedAttention(
        float *output, const float *query,
        void *k_pool, void *v_pool,
        int *block_tables, int *seq_lens,
        int batch_size, int num_heads, int num_kv_heads, int head_dim,
        int block_size, int max_blocks_per_seq,
        size_t pool_block_stride, size_t pool_layer_stride,
        int layer_idx, float scale,
        llaisysDeviceType_t device_type,
        int kv_quant) {
        llaisys::ops::paged_attention(output, query, k_pool, v_pool,
                                       block_tables, seq_lens,
                                       batch_size, num_heads, num_kv_heads, head_dim,
                                       block_size, max_blocks_per_seq,
                                       pool_block_stride, pool_layer_stride,
                                       layer_idx, scale, device_type,
                                       static_cast<llaisys::ops::KVQuantMode>(kv_quant));
    }
}

#include "sparse_latent_attention.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/sparse_latent_attention.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace llaisys::ops {

static void cpu_reference(
    float *output, const float *query, const float *latent,
    const float *attn_sink, const int *indices,
    int batch_size, int query_len, int num_heads, int head_dim,
    int latent_len, int topk, float scale) {
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < query_len; ++s) {
            const int *row_indices = indices + (b * query_len + s) * topk;
            for (int h = 0; h < num_heads; ++h) {
                const float *q = query +
                    ((b * query_len + s) * num_heads + h) * head_dim;
                float maximum = attn_sink[h];
                for (int k = 0; k < topk; ++k) {
                    const int position = row_indices[k];
                    if (position < 0) continue;
                    if (position >= latent_len) {
                        throw std::out_of_range("DeepSeek-V4 latent index out of range");
                    }
                    const float *kv = latent + (b * latent_len + position) * head_dim;
                    float score = 0.0F;
                    for (int d = 0; d < head_dim; ++d) score += q[d] * kv[d];
                    maximum = std::max(maximum, score * scale);
                }
                float denominator = std::exp(attn_sink[h] - maximum);
                for (int k = 0; k < topk; ++k) {
                    const int position = row_indices[k];
                    if (position < 0) continue;
                    const float *kv = latent + (b * latent_len + position) * head_dim;
                    float score = 0.0F;
                    for (int d = 0; d < head_dim; ++d) score += q[d] * kv[d];
                    denominator += std::exp(score * scale - maximum);
                }
                float *out = output +
                    ((b * query_len + s) * num_heads + h) * head_dim;
                std::fill(out, out + head_dim, 0.0F);
                for (int k = 0; k < topk; ++k) {
                    const int position = row_indices[k];
                    if (position < 0) continue;
                    const float *kv = latent + (b * latent_len + position) * head_dim;
                    float score = 0.0F;
                    for (int d = 0; d < head_dim; ++d) score += q[d] * kv[d];
                    const float probability = std::exp(score * scale - maximum) / denominator;
                    for (int d = 0; d < head_dim; ++d) out[d] += probability * kv[d];
                }
            }
        }
    }
}

static float sqrt_softplus(float value) {
    const float softplus = std::max(value, 0.0F) +
        std::log1p(std::exp(-std::abs(value)));
    return std::sqrt(softplus);
}

static void cpu_router_reference(
    float *weights, int *indices, const float *logits,
    const float *selection_bias, int num_tokens, int num_experts,
    int topk, float route_scale) {
    for (int token = 0; token < num_tokens; ++token) {
        float *token_weights = weights + token * topk;
        int *token_indices = indices + token * topk;
        std::fill(token_weights, token_weights + topk,
                  -std::numeric_limits<float>::infinity());
        std::fill(token_indices, token_indices + topk, -1);
        for (int expert = 0; expert < num_experts; ++expert) {
            const float original = sqrt_softplus(
                logits[token * num_experts + expert]);
            const float ranked = original + selection_bias[expert];
            int slot = topk;
            while (slot > 0 && ranked > token_weights[slot - 1]) --slot;
            if (slot == topk) continue;
            for (int move = topk - 1; move > slot; --move) {
                token_weights[move] = token_weights[move - 1];
                token_indices[move] = token_indices[move - 1];
            }
            token_weights[slot] = ranked;
            token_indices[slot] = expert;
        }
        float sum = 0.0F;
        for (int slot = 0; slot < topk; ++slot) {
            token_weights[slot] = sqrt_softplus(
                logits[token * num_experts + token_indices[slot]]);
            sum += token_weights[slot];
        }
        for (int slot = 0; slot < topk; ++slot) {
            token_weights[slot] = token_weights[slot] / sum * route_scale;
        }
    }
}

void deepseek_v4_sparse_attention_reference(
    float *output, const float *query, const float *latent,
    const float *attn_sink, const int *indices,
    int batch_size, int query_len, int num_heads, int head_dim,
    int latent_len, int topk, float scale,
    llaisysDeviceType_t device_type) {
    if (!output || !query || !latent || !attn_sink || !indices ||
        batch_size <= 0 || query_len <= 0 || num_heads <= 0 ||
        head_dim <= 0 || latent_len <= 0 || topk <= 0) {
        throw std::invalid_argument("invalid DeepSeek-V4 sparse attention arguments");
    }
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::deepseek_v4_sparse_attention_reference(
            output, query, latent, attn_sink, indices, batch_size, query_len,
            num_heads, head_dim, latent_len, topk, scale);
    }
#endif
    if (device_type != LLAISYS_DEVICE_CPU) {
        throw std::invalid_argument("DeepSeek-V4 sparse attention device is unavailable");
    }
    cpu_reference(output, query, latent, attn_sink, indices, batch_size,
                  query_len, num_heads, head_dim, latent_len, topk, scale);
}

void deepseek_v4_sparse_attention_reference_typed(
    void *output, const void *query, const void *latent,
    const float *attn_sink, const int *indices,
    int batch_size, int query_len, int num_heads, int head_dim,
    int latent_len, int topk, float scale, llaisysDataType_t dtype,
    llaisysDeviceType_t device_type) {
    if (!output || !query || !latent || !attn_sink || !indices ||
        batch_size <= 0 || query_len <= 0 || num_heads <= 0 ||
        head_dim <= 0 || latent_len <= 0 || topk <= 0) {
        throw std::invalid_argument("invalid typed DeepSeek-V4 sparse attention arguments");
    }
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::deepseek_v4_sparse_attention_reference_typed(
            output, query, latent, attn_sink, indices, batch_size, query_len,
            num_heads, head_dim, latent_len, topk, scale, dtype);
    }
#endif
    if (device_type != LLAISYS_DEVICE_CPU || dtype != LLAISYS_DTYPE_F32) {
        throw std::invalid_argument(
            "typed DeepSeek-V4 sparse attention backend/dtype is unavailable");
    }
    cpu_reference(static_cast<float *>(output),
                  static_cast<const float *>(query),
                  static_cast<const float *>(latent), attn_sink, indices,
                  batch_size, query_len, num_heads, head_dim, latent_len,
                  topk, scale);
}

void deepseek_v4_router_reference(
    float *weights, int *indices, const float *logits,
    const float *selection_bias, int num_tokens, int num_experts,
    int topk, float route_scale, llaisysDeviceType_t device_type) {
    if (!weights || !indices || !logits || !selection_bias ||
        num_tokens <= 0 || num_experts <= 0 || topk <= 0 ||
        topk > num_experts || topk > 16) {
        throw std::invalid_argument("invalid DeepSeek-V4 router arguments");
    }
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::deepseek_v4_router_reference(
            weights, indices, logits, selection_bias, num_tokens,
            num_experts, topk, route_scale);
    }
#endif
    if (device_type != LLAISYS_DEVICE_CPU) {
        throw std::invalid_argument("DeepSeek-V4 router device is unavailable");
    }
    cpu_router_reference(weights, indices, logits, selection_bias, num_tokens,
                         num_experts, topk, route_scale);
}

} // namespace llaisys::ops

#include "sparse_latent_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

template <typename T>
__device__ static float load_value(const T *value) {
    return static_cast<float>(*value);
}

template <>
__device__ float load_value<__nv_bfloat16>(const __nv_bfloat16 *value) {
    return __bfloat162float(*value);
}

template <typename T>
__device__ static T store_value(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __nv_bfloat16 store_value<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
__global__ static void sparse_latent_reference_kernel(
    T *output, const T *query, const T *latent,
    const float *attn_sink, const int *indices,
    int query_len, int num_heads, int head_dim, int latent_len,
    int topk, float scale) {
    const int row = blockIdx.x;
    const int head = row % num_heads;
    const int sequence = (row / num_heads) % query_len;
    const int batch = row / (num_heads * query_len);
    const T *q = query + row * head_dim;
    const int *row_indices = indices + (batch * query_len + sequence) * topk;
    extern __shared__ float probabilities[];

    if (threadIdx.x == 0) {
        float maximum = attn_sink[head];
        for (int k = 0; k < topk; ++k) {
            const int position = row_indices[k];
            float score = -INFINITY;
            if (position >= 0 && position < latent_len) {
                const T *kv = latent + (batch * latent_len + position) * head_dim;
                score = 0.0F;
                    for (int d = 0; d < head_dim; ++d) {
                        score += load_value(q + d) * load_value(kv + d);
                    }
                score *= scale;
                maximum = fmaxf(maximum, score);
            }
            probabilities[k] = score;
        }
        float denominator = expf(attn_sink[head] - maximum);
        for (int k = 0; k < topk; ++k) {
            const float probability = probabilities[k] == -INFINITY
                ? 0.0F : expf(probabilities[k] - maximum);
            probabilities[k] = probability;
            denominator += probability;
        }
        for (int k = 0; k < topk; ++k) probabilities[k] /= denominator;
    }
    __syncthreads();

    T *out = output + row * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = 0.0F;
        for (int k = 0; k < topk; ++k) {
            const int position = row_indices[k];
            if (position >= 0 && position < latent_len) {
                value += probabilities[k] * load_value(
                    latent + (batch * latent_len + position) * head_dim + d);
            }
        }
        out[d] = store_value<T>(value);
    }
}

void deepseek_v4_sparse_attention_reference(
    float *output, const float *query, const float *latent,
    const float *attn_sink, const int *indices,
    int batch_size, int query_len, int num_heads, int head_dim,
    int latent_len, int topk, float scale) {
    const int rows = batch_size * query_len * num_heads;
    sparse_latent_reference_kernel<float><<<rows, 256, topk * sizeof(float)>>>(
        output, query, latent, attn_sink, indices, query_len, num_heads,
        head_dim, latent_len, topk, scale);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void deepseek_v4_sparse_attention_reference_typed(
    void *output, const void *query, const void *latent,
    const float *attn_sink, const int *indices,
    int batch_size, int query_len, int num_heads, int head_dim,
    int latent_len, int topk, float scale, llaisysDataType_t dtype) {
    const int rows = batch_size * query_len * num_heads;
    if (dtype == LLAISYS_DTYPE_F32) {
        sparse_latent_reference_kernel<float><<<rows, 256, topk * sizeof(float)>>>(
            static_cast<float *>(output), static_cast<const float *>(query),
            static_cast<const float *>(latent), attn_sink, indices, query_len,
            num_heads, head_dim, latent_len, topk, scale);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        sparse_latent_reference_kernel<__nv_bfloat16><<<
            rows, 256, topk * sizeof(float)>>>(
            static_cast<__nv_bfloat16 *>(output),
            static_cast<const __nv_bfloat16 *>(query),
            static_cast<const __nv_bfloat16 *>(latent), attn_sink, indices,
            query_len, num_heads, head_dim, latent_len, topk, scale);
    } else {
        throw std::invalid_argument(
            "DeepSeek-V4 typed sparse attention supports F32 and BF16 only");
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__device__ static float sqrt_softplus(float value) {
    return sqrtf(fmaxf(value, 0.0F) + log1pf(expf(-fabsf(value))));
}

__global__ static void router_reference_kernel(
    float *weights, int *indices, const float *logits,
    const float *selection_bias, int num_experts, int topk,
    float route_scale) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    float ranked_top[16];
    int expert_top[16];
    for (int slot = 0; slot < topk; ++slot) {
        ranked_top[slot] = -INFINITY;
        expert_top[slot] = -1;
    }
    for (int expert = 0; expert < num_experts; ++expert) {
        const float original = sqrt_softplus(logits[token * num_experts + expert]);
        const float ranked = original + selection_bias[expert];
        int slot = topk;
        while (slot > 0 && ranked > ranked_top[slot - 1]) --slot;
        if (slot == topk) continue;
        for (int move = topk - 1; move > slot; --move) {
            ranked_top[move] = ranked_top[move - 1];
            expert_top[move] = expert_top[move - 1];
        }
        ranked_top[slot] = ranked;
        expert_top[slot] = expert;
    }
    float sum = 0.0F;
    for (int slot = 0; slot < topk; ++slot) {
        const float value = sqrt_softplus(
            logits[token * num_experts + expert_top[slot]]);
        weights[token * topk + slot] = value;
        indices[token * topk + slot] = expert_top[slot];
        sum += value;
    }
    for (int slot = 0; slot < topk; ++slot) {
        weights[token * topk + slot] =
            weights[token * topk + slot] / sum * route_scale;
    }
}

void deepseek_v4_router_reference(
    float *weights, int *indices, const float *logits,
    const float *selection_bias, int num_tokens, int num_experts,
    int topk, float route_scale) {
    // One thread owns one token. The production EP path will replace this
    // bring-up kernel with device-side dispatch metadata and collectives.
    router_reference_kernel<<<num_tokens, 1>>>(
        weights, indices, logits, selection_bias, num_experts, topk,
        route_scale);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia

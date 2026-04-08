#include "paged_attention.hpp"
#include "../../../src/core/kv_quant.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/paged_attention_nvidia.cuh"
#include "nvidia/flashinfer_adapter.cuh"
#endif

#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

namespace llaisys::ops {

// ── FP32 CPU kernel ────────────────────────────────────────────────

static void paged_attention_cpu_fp32(
    float *output, const float *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    int group_size = num_heads / num_kv_heads;

    for (int b = 0; b < batch_size; ++b) {
        int seq_len = seq_lens[b];
        int num_blocks = (seq_len + block_size - 1) / block_size;

        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / group_size;

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            std::vector<float> acc(head_dim, 0.0f);

            for (int bi = 0; bi < num_blocks; ++bi) {
                int block_id = block_tables[b * max_blocks_per_seq + bi];
                int tokens_in_block = std::min(block_size, seq_len - bi * block_size);

                const float *k_base = reinterpret_cast<const float *>(
                    static_cast<const std::byte *>(k_pool) +
                    static_cast<size_t>(block_id) * pool_block_stride +
                    static_cast<size_t>(layer_idx) * pool_layer_stride);

                const float *v_base = reinterpret_cast<const float *>(
                    static_cast<const std::byte *>(v_pool) +
                    static_cast<size_t>(block_id) * pool_block_stride +
                    static_cast<size_t>(layer_idx) * pool_layer_stride);

                for (int t = 0; t < tokens_in_block; ++t) {
                    const float *k_vec = k_base + t * num_kv_heads * head_dim + kv_h * head_dim;
                    const float *v_vec = v_base + t * num_kv_heads * head_dim + kv_h * head_dim;
                    const float *q_vec = query + b * num_heads * head_dim + h * head_dim;

                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        score += q_vec[d] * k_vec[d];
                    score *= scale;

                    float m_new = std::max(m, score);
                    float p = std::exp(score - m_new);
                    float correction = std::exp(m - m_new);
                    l = correction * l + p;
                    for (int d = 0; d < head_dim; ++d)
                        acc[d] = correction * acc[d] + p * v_vec[d];
                    m = m_new;
                }
            }

            float *out_vec = output + b * num_heads * head_dim + h * head_dim;
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            for (int d = 0; d < head_dim; ++d)
                out_vec[d] = acc[d] * inv_l;
        }
    }
}

// ── INT8 quantized KV CPU kernel ──────────────────────────────────
// Block layout: [block_size * nkvh * dh] int8 data + [block_size * nkvh] float scales

static void paged_attention_cpu_int8(
    float *output, const float *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    int group_size = num_heads / num_kv_heads;
    size_t data_size = (size_t)block_size * num_kv_heads * head_dim;
    // Scales follow the int8 data within each block-layer region

    std::vector<float> k_buf(head_dim), v_buf(head_dim);

    for (int b = 0; b < batch_size; ++b) {
        int seq_len = seq_lens[b];
        int num_blocks = (seq_len + block_size - 1) / block_size;

        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / group_size;

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            std::vector<float> acc(head_dim, 0.0f);

            for (int bi = 0; bi < num_blocks; ++bi) {
                int block_id = block_tables[b * max_blocks_per_seq + bi];
                int tokens_in_block = std::min(block_size, seq_len - bi * block_size);

                const std::byte *k_block = static_cast<const std::byte *>(k_pool) +
                    static_cast<size_t>(block_id) * pool_block_stride +
                    static_cast<size_t>(layer_idx) * pool_layer_stride;
                const std::byte *v_block = static_cast<const std::byte *>(v_pool) +
                    static_cast<size_t>(block_id) * pool_block_stride +
                    static_cast<size_t>(layer_idx) * pool_layer_stride;

                const int8_t *k_data = reinterpret_cast<const int8_t *>(k_block);
                const float *k_scales = reinterpret_cast<const float *>(k_block + data_size);
                const int8_t *v_data = reinterpret_cast<const int8_t *>(v_block);
                const float *v_scales = reinterpret_cast<const float *>(v_block + data_size);

                for (int t = 0; t < tokens_in_block; ++t) {
                    const int8_t *k_vec = k_data + t * num_kv_heads * head_dim + kv_h * head_dim;
                    float k_scale = k_scales[t * num_kv_heads + kv_h];
                    llaisys::core::dequantize_int8_to_fp32(k_buf.data(), k_vec, k_scale, head_dim);

                    const float *q_vec = query + b * num_heads * head_dim + h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        score += q_vec[d] * k_buf[d];
                    score *= scale;

                    float m_new = std::max(m, score);
                    float p = std::exp(score - m_new);
                    float correction = std::exp(m - m_new);
                    l = correction * l + p;

                    const int8_t *v_vec = v_data + t * num_kv_heads * head_dim + kv_h * head_dim;
                    float v_scale = v_scales[t * num_kv_heads + kv_h];
                    llaisys::core::dequantize_int8_to_fp32(v_buf.data(), v_vec, v_scale, head_dim);

                    for (int d = 0; d < head_dim; ++d)
                        acc[d] = correction * acc[d] + p * v_buf[d];
                    m = m_new;
                }
            }

            float *out_vec = output + b * num_heads * head_dim + h * head_dim;
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            for (int d = 0; d < head_dim; ++d)
                out_vec[d] = acc[d] * inv_l;
        }
    }
}

// ── INT4 quantized KV CPU kernel ──────────────────────────────────

static void paged_attention_cpu_int4(
    float *output, const float *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    int group_size = num_heads / num_kv_heads;
    size_t packed_dh = (head_dim + 1) / 2;
    size_t data_size = (size_t)block_size * num_kv_heads * packed_dh;

    std::vector<float> k_buf(head_dim), v_buf(head_dim);

    for (int b = 0; b < batch_size; ++b) {
        int seq_len = seq_lens[b];
        int num_blocks = (seq_len + block_size - 1) / block_size;

        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / group_size;

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            std::vector<float> acc(head_dim, 0.0f);

            for (int bi = 0; bi < num_blocks; ++bi) {
                int block_id = block_tables[b * max_blocks_per_seq + bi];
                int tokens_in_block = std::min(block_size, seq_len - bi * block_size);

                const std::byte *k_block = static_cast<const std::byte *>(k_pool) +
                    static_cast<size_t>(block_id) * pool_block_stride +
                    static_cast<size_t>(layer_idx) * pool_layer_stride;
                const std::byte *v_block = static_cast<const std::byte *>(v_pool) +
                    static_cast<size_t>(block_id) * pool_block_stride +
                    static_cast<size_t>(layer_idx) * pool_layer_stride;

                const uint8_t *k_data = reinterpret_cast<const uint8_t *>(k_block);
                const float *k_scales = reinterpret_cast<const float *>(k_block + data_size);
                const uint8_t *v_data = reinterpret_cast<const uint8_t *>(v_block);
                const float *v_scales = reinterpret_cast<const float *>(v_block + data_size);

                for (int t = 0; t < tokens_in_block; ++t) {
                    const uint8_t *k_vec = k_data + t * num_kv_heads * packed_dh + kv_h * packed_dh;
                    float k_scale = k_scales[t * num_kv_heads + kv_h];
                    llaisys::core::dequantize_int4_to_fp32(k_buf.data(), k_vec, k_scale, head_dim);

                    const float *q_vec = query + b * num_heads * head_dim + h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        score += q_vec[d] * k_buf[d];
                    score *= scale;

                    float m_new = std::max(m, score);
                    float p = std::exp(score - m_new);
                    float correction = std::exp(m - m_new);
                    l = correction * l + p;

                    const uint8_t *v_vec = v_data + t * num_kv_heads * packed_dh + kv_h * packed_dh;
                    float v_scale = v_scales[t * num_kv_heads + kv_h];
                    llaisys::core::dequantize_int4_to_fp32(v_buf.data(), v_vec, v_scale, head_dim);

                    for (int d = 0; d < head_dim; ++d)
                        acc[d] = correction * acc[d] + p * v_buf[d];
                    m = m_new;
                }
            }

            float *out_vec = output + b * num_heads * head_dim + h * head_dim;
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            for (int d = 0; d < head_dim; ++d)
                out_vec[d] = acc[d] * inv_l;
        }
    }
}

// ── Dispatch ──────────────────────────────────────────────────────

void paged_attention(
    float *output, const float *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDeviceType_t device_type,
    KVQuantMode kv_quant)
{
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA && kv_quant == KVQuantMode::FP32) {
        if (nvidia::flashinfer_available()) {
            nvidia::flashinfer_paged_attention(output, query, k_pool, v_pool,
                                               block_tables, seq_lens,
                                               batch_size, num_heads, num_kv_heads, head_dim,
                                               block_size, max_blocks_per_seq,
                                               pool_block_stride, pool_layer_stride,
                                               layer_idx, scale);
            return;
        }
        nvidia::paged_attention(output, query, k_pool, v_pool,
                                block_tables, seq_lens,
                                batch_size, num_heads, num_kv_heads, head_dim,
                                block_size, max_blocks_per_seq,
                                pool_block_stride, pool_layer_stride,
                                layer_idx, scale);
        return;
    }
#endif

    switch (kv_quant) {
        case KVQuantMode::INT8:
            paged_attention_cpu_int8(output, query, k_pool, v_pool,
                                     block_tables, seq_lens,
                                     batch_size, num_heads, num_kv_heads, head_dim,
                                     block_size, max_blocks_per_seq,
                                     pool_block_stride, pool_layer_stride,
                                     layer_idx, scale);
            break;
        case KVQuantMode::INT4:
            paged_attention_cpu_int4(output, query, k_pool, v_pool,
                                     block_tables, seq_lens,
                                     batch_size, num_heads, num_kv_heads, head_dim,
                                     block_size, max_blocks_per_seq,
                                     pool_block_stride, pool_layer_stride,
                                     layer_idx, scale);
            break;
        default:
            paged_attention_cpu_fp32(output, query, k_pool, v_pool,
                                     block_tables, seq_lens,
                                     batch_size, num_heads, num_kv_heads, head_dim,
                                     block_size, max_blocks_per_seq,
                                     pool_block_stride, pool_layer_stride,
                                     layer_idx, scale);
            break;
    }
}

} // namespace llaisys::ops

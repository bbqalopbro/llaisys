#pragma once

#include "llaisys.h"

namespace llaisys::ops {

void deepseek_v4_sparse_attention_reference(
    float *output, const float *query, const float *latent,
    const float *attn_sink, const int *indices,
    int batch_size, int query_len, int num_heads, int head_dim,
    int latent_len, int topk, float scale,
    llaisysDeviceType_t device_type);

void deepseek_v4_sparse_attention_reference_typed(
    void *output, const void *query, const void *latent,
    const float *attn_sink, const int *indices,
    int batch_size, int query_len, int num_heads, int head_dim,
    int latent_len, int topk, float scale, llaisysDataType_t dtype,
    llaisysDeviceType_t device_type);

void deepseek_v4_router_reference(
    float *weights, int *indices, const float *logits,
    const float *selection_bias, int num_tokens, int num_experts,
    int topk, float route_scale, llaisysDeviceType_t device_type);

} // namespace llaisys::ops

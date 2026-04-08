#pragma once

#include <cstddef>

namespace llaisys::ops::nvidia {

void paged_attention(
    float *output, const float *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale);

} // namespace llaisys::ops::nvidia

#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

// dtype 参数控制 output/query/KV pool 的数据类型 (FP32/FP16/BF16)
// 中间计算 (softmax/累加) 始终使用 FP32
void paged_attention(
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDataType_t dtype = LLAISYS_DTYPE_F32);

} // namespace llaisys::ops::nvidia

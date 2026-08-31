#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

// reshape_and_cache: 将计算出的 K/V 写入 paged block pool
// 根据 positions (device) 和 block_tables (device) 确定目标位置
// 支持 FP16/FP32, 模板化实现
void reshape_and_cache(
    const void *k_src,             // [batch_size, nkvh, dh]
    const void *v_src,             // [batch_size, nkvh, dh]
    void *k_pool,                  // block pool K base
    void *v_pool,                  // block pool V base
    const int *block_tables_dev,   // [batch_size, max_blocks_per_seq] on device
    const int64_t *positions_dev,  // [batch_size] on device
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    size_t pool_block_stride,
    size_t pool_layer_stride,
    int layer_idx,
    llaisysDataType_t dtype = LLAISYS_DTYPE_F32);

void gather_paged_cache(
    void *k_dst,
    void *v_dst,
    const void *k_pool,
    const void *v_pool,
    const int *block_table_dev,
    int total_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    size_t pool_block_stride,
    size_t pool_layer_stride,
    int layer_idx,
    llaisysDataType_t dtype = LLAISYS_DTYPE_F32);

void copy_next_token_to_input_ids(
    const int32_t *next_token_dev,
    int64_t *input_ids_dev);

} // namespace llaisys::ops::nvidia

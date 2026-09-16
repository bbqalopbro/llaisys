#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops {

// reshape_and_cache: 将计算出的 K/V 写入 paged block pool (GPU kernel)
// 根据 positions_dev 和 block_tables_dev (均在 device 上) 确定目标位置
// CUDA Graph 兼容 — 所有参数为 device 指针或常量
void reshape_and_cache(
    const void *k_src,             // [batch_size, nkvh, dh] device
    const void *v_src,             // [batch_size, nkvh, dh] device
    void *k_pool,                  // block pool K base
    void *v_pool,                  // block pool V base
    const int *block_tables_dev,   // [batch_size, max_blocks_per_seq] device
    const int64_t *positions_dev,  // [batch_size] device
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    size_t pool_block_stride,
    size_t pool_layer_stride,
    int layer_idx,
    llaisysDeviceType_t device_type,
    llaisysDataType_t dtype = LLAISYS_DTYPE_F32);

} // namespace llaisys::ops

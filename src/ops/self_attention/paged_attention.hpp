#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops {

// KV-Cache quantization type for paged attention
enum class KVQuantMode : int {
    FP32 = 0,
    INT8 = 1,
    INT4 = 2,
};

// Paged Attention: compute Q·K^T→softmax→·V where K/V live in a block pool
// addressed via per-sequence page tables. Supports GQA.
//
// Layout of block pool (FP32): flat [num_blocks, nlayer, block_size, nkvh, dh]
// Layout of block pool (INT8): flat [num_blocks, nlayer, {block_size*nkvh*dh INT8 + block_size*nkvh FP32 scales}]
void paged_attention(
    void *output,               // [batch_size, num_heads, head_dim] (dtype 决定类型)
    const void *query,          // [batch_size, num_heads, head_dim] (dtype 决定类型)
    const void *k_pool,         // Block pool K base pointer
    const void *v_pool,         // Block pool V base pointer
    const int *block_tables,    // [batch_size, max_blocks_per_seq] CPU-side
    const int *seq_lens,        // [batch_size] CPU-side
    int batch_size,
    int num_heads,              // query heads (nh)
    int num_kv_heads,           // KV heads (nkvh, local for TP)
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    size_t pool_block_stride,   // bytes per block across all layers
    size_t pool_layer_stride,   // bytes per (block_size, nkvh, dh) region
    int layer_idx,
    float scale,
    llaisysDeviceType_t device_type,
    KVQuantMode kv_quant = KVQuantMode::FP32,
    llaisysDataType_t dtype = LLAISYS_DTYPE_F32
);

} // namespace llaisys::ops

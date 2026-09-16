#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

// Build-time flag: set by xmake when FlashInfer headers are found
#ifdef ENABLE_FLASHINFER

bool flashinfer_available();

void flashinfer_paged_attention(
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDataType_t dtype);

#else

inline bool flashinfer_available() { return false; }

inline void flashinfer_paged_attention(
    void *, const void *, const void *, const void *,
    const int *, const int *, int, int, int, int, int, int,
    size_t, size_t, int, float, llaisysDataType_t) {}

#endif

} // namespace llaisys::ops::nvidia

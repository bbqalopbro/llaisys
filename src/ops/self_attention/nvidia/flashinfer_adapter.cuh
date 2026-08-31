#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

// Build-time flag: set by xmake when FlashInfer headers are found
#ifdef ENABLE_FLASHINFER

bool flashinfer_available();

// Per-runtime workspace for FlashInfer paged prefill. The workspace owns the
// plan buffers and device-side CSR page metadata, so prepare() is called once
// per engine step and every transformer layer reuses the same plan.
void *flashinfer_paged_prefill_workspace_create();
void flashinfer_paged_prefill_workspace_destroy(void *workspace);

bool flashinfer_paged_prefill_supported(
    int num_heads, int num_kv_heads, int head_dim,
    llaisysDataType_t dtype);

void flashinfer_paged_prefill_prepare(
    void *workspace,
    const int *block_tables, const int *seq_lens, const int *query_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    llaisysDataType_t dtype);

void flashinfer_paged_prefill_run(
    void *workspace,
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale, llaisysDataType_t dtype);

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

inline void *flashinfer_paged_prefill_workspace_create() { return nullptr; }
inline void flashinfer_paged_prefill_workspace_destroy(void *) {}

inline bool flashinfer_paged_prefill_supported(
    int, int, int, llaisysDataType_t) { return false; }

inline void flashinfer_paged_prefill_prepare(
    void *, const int *, const int *, const int *,
    int, int, int, int, int, int, llaisysDataType_t) {}

inline void flashinfer_paged_prefill_run(
    void *, void *, const void *, const void *, const void *,
    size_t, size_t, int, float, llaisysDataType_t) {}

inline void flashinfer_paged_attention(
    void *, const void *, const void *, const void *,
    const int *, const int *, int, int, int, int, int, int,
    size_t, size_t, int, float, llaisysDataType_t) {}

#endif

} // namespace llaisys::ops::nvidia

#include "cache_ops.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/cache_kernels.cuh"
#endif

#include <stdexcept>

namespace llaisys::ops {

void reshape_and_cache(
    const void *k_src, const void *v_src,
    void *k_pool, void *v_pool,
    const int *block_tables_dev, const int64_t *positions_dev,
    int batch_size, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx,
    llaisysDeviceType_t device_type,
    llaisysDataType_t dtype)
{
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        nvidia::reshape_and_cache(k_src, v_src, k_pool, v_pool,
                                  block_tables_dev, positions_dev,
                                  batch_size, num_kv_heads, head_dim,
                                  block_size, max_blocks_per_seq,
                                  pool_block_stride, pool_layer_stride,
                                  layer_idx, dtype);
        return;
    }
#endif
    throw std::runtime_error("reshape_and_cache: only supported on NVIDIA GPU");
}

void gather_paged_cache(
    void *k_dst, void *v_dst,
    const void *k_pool, const void *v_pool,
    const int *block_table_dev,
    int total_tokens, int num_kv_heads, int head_dim, int block_size,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, llaisysDeviceType_t device_type,
    llaisysDataType_t dtype)
{
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        nvidia::gather_paged_cache(
            k_dst, v_dst, k_pool, v_pool, block_table_dev,
            total_tokens, num_kv_heads, head_dim, block_size,
            pool_block_stride, pool_layer_stride, layer_idx, dtype);
        return;
    }
#endif
    throw std::runtime_error("gather_paged_cache: only supported on NVIDIA GPU");
}

void copy_next_token_to_input_ids(
    const int32_t *next_token_dev,
    int64_t *input_ids_dev,
    llaisysDeviceType_t device_type)
{
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        nvidia::copy_next_token_to_input_ids(next_token_dev, input_ids_dev);
        return;
    }
#endif
    throw std::runtime_error("copy_next_token_to_input_ids: only supported on NVIDIA GPU");
}

} // namespace llaisys::ops

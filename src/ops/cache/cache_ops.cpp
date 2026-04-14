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

} // namespace llaisys::ops

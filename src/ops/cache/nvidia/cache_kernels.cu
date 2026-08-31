#include "cache_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>
#include <cstdint>

// ── reshape_and_cache kernel ──────────────────────────────────────
// 将 K/V [B, nkvh, dh] 根据 positions 和 block_tables 写入 paged block pool
// 每个 thread block 处理一个 (batch, element) 对
template<typename T>
__global__ void reshape_and_cache_kernel(
    const T *__restrict__ k_src,       // [B, nkvh * dh]
    const T *__restrict__ v_src,       // [B, nkvh * dh]
    char *__restrict__ k_pool,
    char *__restrict__ v_pool,
    const int *__restrict__ block_tables,   // [B, max_blocks_per_seq] (device)
    const int64_t *__restrict__ positions,  // [B] (device)
    int kv_dim,            // nkvh * dh
    int block_size,
    int max_blocks_per_seq,
    size_t pool_block_stride,
    size_t pool_layer_stride,
    int layer_idx)
{
    int b = blockIdx.x;
    int elem = blockIdx.y * blockDim.x + threadIdx.x;
    if (elem >= kv_dim) return;

    int64_t pos = positions[b];
    int block_idx = (int)(pos / block_size);
    int offset    = (int)(pos % block_size);
    int block_id  = block_tables[b * max_blocks_per_seq + block_idx];

    // 目标地址: pool + block_id * block_stride + layer * layer_stride + offset * kv_dim
    T *k_dst = reinterpret_cast<T*>(
        k_pool + (size_t)block_id * pool_block_stride
               + (size_t)layer_idx * pool_layer_stride)
        + offset * kv_dim;
    T *v_dst = reinterpret_cast<T*>(
        v_pool + (size_t)block_id * pool_block_stride
               + (size_t)layer_idx * pool_layer_stride)
        + offset * kv_dim;

    k_dst[elem] = k_src[b * kv_dim + elem];
    v_dst[elem] = v_src[b * kv_dim + elem];
}

__global__ void copy_next_token_to_input_ids_kernel(
    const int32_t *__restrict__ next_token,
    int64_t *__restrict__ input_ids)
{
    input_ids[0] = static_cast<int64_t>(next_token[0]);
}

template<typename T>
__global__ void gather_paged_cache_kernel(
    T *__restrict__ k_dst,
    T *__restrict__ v_dst,
    const char *__restrict__ k_pool,
    const char *__restrict__ v_pool,
    const int *__restrict__ block_table,
    int total_tokens,
    int kv_dim,
    int block_size,
    size_t pool_block_stride,
    size_t pool_layer_stride,
    int layer_idx)
{
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(total_tokens) * kv_dim;
    if (linear >= count) return;

    int token = static_cast<int>(linear / kv_dim);
    int elem = static_cast<int>(linear % kv_dim);
    int logical_block = token / block_size;
    int block_offset = token % block_size;
    int block_id = block_table[logical_block];
    const T *k_src = reinterpret_cast<const T *>(
        k_pool + static_cast<size_t>(block_id) * pool_block_stride
               + static_cast<size_t>(layer_idx) * pool_layer_stride)
        + static_cast<size_t>(block_offset) * kv_dim;
    const T *v_src = reinterpret_cast<const T *>(
        v_pool + static_cast<size_t>(block_id) * pool_block_stride
               + static_cast<size_t>(layer_idx) * pool_layer_stride)
        + static_cast<size_t>(block_offset) * kv_dim;
    k_dst[linear] = k_src[elem];
    v_dst[linear] = v_src[elem];
}

// ── reshape_and_cache 启动器 (模板化) ─────────────────────────────
template<typename T>
static void reshape_and_cache_typed(
    const T *k_src, const T *v_src,
    void *k_pool, void *v_pool,
    const int *block_tables_dev, const int64_t *positions_dev,
    int batch_size, int kv_dim, int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride, int layer_idx)
{
    int threads = 256;
    int elems_per_block = (kv_dim + threads - 1) / threads;
    dim3 grid(batch_size, elems_per_block);

    reshape_and_cache_kernel<T><<<grid, threads>>>(
        k_src, v_src,
        static_cast<char*>(k_pool), static_cast<char*>(v_pool),
        block_tables_dev, positions_dev,
        kv_dim, block_size, max_blocks_per_seq,
        pool_block_stride, pool_layer_stride, layer_idx);
}

namespace llaisys::ops::nvidia {

template<typename T>
static void gather_paged_cache_typed(
    T *k_dst, T *v_dst,
    const void *k_pool, const void *v_pool,
    const int *block_table_dev,
    int total_tokens, int kv_dim, int block_size,
    size_t pool_block_stride, size_t pool_layer_stride, int layer_idx)
{
    size_t count = static_cast<size_t>(total_tokens) * kv_dim;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    gather_paged_cache_kernel<T><<<blocks, threads>>>(
        k_dst, v_dst,
        static_cast<const char *>(k_pool), static_cast<const char *>(v_pool),
        block_table_dev, total_tokens, kv_dim, block_size,
        pool_block_stride, pool_layer_stride, layer_idx);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void gather_paged_cache(
    void *k_dst, void *v_dst,
    const void *k_pool, const void *v_pool,
    const int *block_table_dev,
    int total_tokens, int num_kv_heads, int head_dim, int block_size,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, llaisysDataType_t dtype)
{
    int kv_dim = num_kv_heads * head_dim;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        gather_paged_cache_typed<float>(
            static_cast<float *>(k_dst), static_cast<float *>(v_dst),
            k_pool, v_pool, block_table_dev, total_tokens, kv_dim, block_size,
            pool_block_stride, pool_layer_stride, layer_idx);
        break;
    case LLAISYS_DTYPE_F16:
        gather_paged_cache_typed<__half>(
            static_cast<__half *>(k_dst), static_cast<__half *>(v_dst),
            k_pool, v_pool, block_table_dev, total_tokens, kv_dim, block_size,
            pool_block_stride, pool_layer_stride, layer_idx);
        break;
    case LLAISYS_DTYPE_BF16:
        gather_paged_cache_typed<__nv_bfloat16>(
            static_cast<__nv_bfloat16 *>(k_dst),
            static_cast<__nv_bfloat16 *>(v_dst),
            k_pool, v_pool, block_table_dev, total_tokens, kv_dim, block_size,
            pool_block_stride, pool_layer_stride, layer_idx);
        break;
    default:
        throw std::runtime_error("[gather_paged_cache] unsupported dtype");
    }
}

void reshape_and_cache(
    const void *k_src, const void *v_src,
    void *k_pool, void *v_pool,
    const int *block_tables_dev, const int64_t *positions_dev,
    int batch_size, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, llaisysDataType_t dtype)
{
    int kv_dim = num_kv_heads * head_dim;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        reshape_and_cache_typed<float>(
            (const float*)k_src, (const float*)v_src,
            k_pool, v_pool, block_tables_dev, positions_dev,
            batch_size, kv_dim, block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride, layer_idx);
        break;
    case LLAISYS_DTYPE_F16:
        reshape_and_cache_typed<__half>(
            (const __half*)k_src, (const __half*)v_src,
            k_pool, v_pool, block_tables_dev, positions_dev,
            batch_size, kv_dim, block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride, layer_idx);
        break;
    default:
        throw std::runtime_error("[reshape_and_cache] unsupported dtype");
    }
}

void copy_next_token_to_input_ids(
    const int32_t *next_token_dev,
    int64_t *input_ids_dev)
{
    copy_next_token_to_input_ids_kernel<<<1, 1>>>(next_token_dev, input_ids_dev);
}

} // namespace llaisys::ops::nvidia

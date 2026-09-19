#include "paged_attention_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",              \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);       \
            throw std::runtime_error(cudaGetErrorString(err));                    \
        }                                                                         \
    } while (0)

#include "../../llmops.hpp"
#include <llmops/ops/paged.hpp>
#include <type_traits>
namespace llaisys::ops::nvidia {
template<typename T>
static void paged_attention_typed(
    T *output, const T *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables_host, const int *seq_lens_host,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    size_t bt_bytes = (size_t)batch_size * max_blocks_per_seq * sizeof(int);
    size_t sl_bytes = (size_t)batch_size * sizeof(int);

    int *d_block_tables = nullptr;
    int *d_seq_lens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_tables, bt_bytes));
    CUDA_CHECK(cudaMalloc(&d_seq_lens, sl_bytes));
    CUDA_CHECK(cudaMemcpy(d_block_tables, block_tables_host, bt_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens, seq_lens_host, sl_bytes, cudaMemcpyHostToDevice));

    // Use WARP_SIZE threads per block (sufficient for head_dim=128)
    // For head_dim > WARP_SIZE, use multiple warps
    llmops::cuda::paged_attention_device(output, query, k_pool, v_pool, d_block_tables, d_seq_lens,
        batch_size, num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq,
        pool_block_stride, pool_layer_stride, layer_idx, scale,
        std::is_same_v<T,float> ? LLMOPS_F32 : std::is_same_v<T,__half> ? LLMOPS_F16 : LLMOPS_BF16,
        cudaStreamPerThread);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_block_tables);
    cudaFree(d_seq_lens);
}

// ── Device-pointer variant (CUDA Graph 兼容, 无 malloc/free) ──────
void paged_attention(
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables_host, const int *seq_lens_host,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDataType_t dtype)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        paged_attention_typed<float>(
            (float*)output, (const float*)query,
            k_pool, v_pool, block_tables_host, seq_lens_host,
            batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride,
            layer_idx, scale);
        break;
    case LLAISYS_DTYPE_F16:
        paged_attention_typed<__half>(
            (__half*)output, (const __half*)query,
            k_pool, v_pool, block_tables_host, seq_lens_host,
            batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride,
            layer_idx, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        paged_attention_typed<__nv_bfloat16>(
            (__nv_bfloat16*)output, (const __nv_bfloat16*)query,
            k_pool, v_pool, block_tables_host, seq_lens_host,
            batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride,
            layer_idx, scale);
        break;
    default:
        throw std::runtime_error("paged_attention: unsupported dtype");
    }
}

void paged_attention_device(
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables_dev, const int *seq_lens_dev,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDataType_t dtype, void *workspace, size_t workspace_bytes) {
    llmops::cuda::paged_attention_device_with_workspace(output, query, k_pool, v_pool, block_tables_dev, seq_lens_dev, batch_size, num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq, pool_block_stride, pool_layer_stride, layer_idx, scale, kernel_dtype(dtype), cudaStreamPerThread, workspace, workspace_bytes);
}
}

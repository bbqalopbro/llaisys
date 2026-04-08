#include "paged_attention_nvidia.cuh"

#include <cuda_runtime.h>
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

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// Paged Attention decode kernel: one (batch, head) pair per thread block.
// Threads cooperate on head_dim for QK dot products and output accumulation.
// Uses online softmax across KV blocks referenced by the page table.
__global__ void paged_attention_kernel(
    float *__restrict__ output,
    const float *__restrict__ query,
    const char *__restrict__ k_pool,
    const char *__restrict__ v_pool,
    const int *__restrict__ block_tables,
    const int *__restrict__ seq_lens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    size_t pool_block_stride,
    size_t pool_layer_stride,
    int layer_idx,
    float scale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int group_size = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / group_size;

    int seq_len = seq_lens[batch_idx];
    int num_blocks = (seq_len + block_size - 1) / block_size;

    const float *q_vec = query + batch_idx * num_heads * head_dim + head_idx * head_dim;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Shared memory for scores and cross-warp reduction
    extern __shared__ float smem[];
    float *s_scores = smem;
    float *s_reduce = smem + block_size;  // [2 * num_warps] for max and sum

    int num_warps = nthreads / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    float m = -1e30f;
    float l = 0.0f;

    // Thread-local accumulator: each thread handles dims [tid, tid+nthreads, ...]
    float acc[8];
    int dims_per_thread = (head_dim + nthreads - 1) / nthreads;
    for (int i = 0; i < dims_per_thread; ++i) acc[i] = 0.0f;

    for (int bi = 0; bi < num_blocks; ++bi) {
        int block_id = block_tables[batch_idx * max_blocks_per_seq + bi];
        int tokens_in_block = min(block_size, seq_len - bi * block_size);

        const float *k_base = reinterpret_cast<const float *>(
            k_pool + (size_t)block_id * pool_block_stride +
            (size_t)layer_idx * pool_layer_stride);
        const float *v_base = reinterpret_cast<const float *>(
            v_pool + (size_t)block_id * pool_block_stride +
            (size_t)layer_idx * pool_layer_stride);

        // Compute QK^T scores: threads partition over head_dim, reduce to score
        for (int t = 0; t < tokens_in_block; ++t) {
            const float *k_vec = k_base + t * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float partial = 0.0f;
            for (int d = tid; d < head_dim; d += nthreads)
                partial += q_vec[d] * k_vec[d];

            // Warp-level reduction
            partial = warp_reduce_sum(partial);

            // Cross-warp reduction via shared memory
            if (lane_id == 0) s_reduce[warp_id] = partial;
            __syncthreads();

            float score;
            if (tid == 0) {
                score = 0.0f;
                for (int w = 0; w < num_warps; ++w) score += s_reduce[w];
                s_scores[t] = score * scale;
            }
            __syncthreads();
        }

        // Online softmax: all threads read the same scores,
        // but each accumulates its own subset of V dimensions
        for (int t = 0; t < tokens_in_block; ++t) {
            float score = s_scores[t];
            float m_new = fmaxf(m, score);
            float p = expf(score - m_new);
            float correction = expf(m - m_new);
            l = correction * l + p;

            const float *v_vec = v_base + t * num_kv_heads * head_dim + kv_head_idx * head_dim;
            for (int i = 0; i < dims_per_thread; ++i) {
                int d = tid + i * nthreads;
                if (d < head_dim)
                    acc[i] = correction * acc[i] + p * v_vec[d];
            }
            m = m_new;
        }
        __syncthreads();
    }

    // Write output: each thread writes its own dimensions
    float *out_vec = output + batch_idx * num_heads * head_dim + head_idx * head_dim;
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * nthreads;
        if (d < head_dim)
            out_vec[d] = acc[i] * inv_l;
    }
}

namespace llaisys::ops::nvidia {

void paged_attention(
    float *output, const float *query,
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

    dim3 grid(batch_size, num_heads);
    // Use WARP_SIZE threads per block (sufficient for head_dim=128)
    // For head_dim > WARP_SIZE, use multiple warps
    int threads = min(128, max(WARP_SIZE, ((head_dim + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE));
    int num_warps = threads / WARP_SIZE;
    size_t smem = block_size * sizeof(float) + 2 * num_warps * sizeof(float);

    paged_attention_kernel<<<grid, threads, smem>>>(
        output, query,
        static_cast<const char *>(k_pool),
        static_cast<const char *>(v_pool),
        d_block_tables, d_seq_lens,
        num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq,
        pool_block_stride, pool_layer_stride,
        layer_idx, scale);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_block_tables);
    cudaFree(d_seq_lens);
}

} // namespace llaisys::ops::nvidia

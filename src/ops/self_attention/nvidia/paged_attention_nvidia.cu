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

// ── FP32/FP16/BF16 混合精度转换辅助 ────────────────────────────────
// 所有中间计算（QK点积、softmax、V累加）保持 FP32，仅 I/O 使用模板类型 T
template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

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

// Paged Attention decode kernel (模板化版本):
// 一个 (batch, head) 对应一个 thread block。
// 线程协作完成 head_dim 维度的 QK 点积和输出累加。
// 使用 online softmax 跨越 page table 引用的 KV blocks。
// T = float / __half / __nv_bfloat16 (I/O 类型)
// 中间计算始终使用 FP32 保证数值稳定性。
template<typename T>
__global__ void paged_attention_kernel(
    T *__restrict__ output,
    const T *__restrict__ query,
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
    int group_size = num_heads / num_kv_heads; //GQA适配
    int kv_head_idx = head_idx / group_size;

    int seq_len = seq_lens[batch_idx];
    int num_blocks = (seq_len + block_size - 1) / block_size;

    const T *q_vec = query + batch_idx * num_heads * head_dim + head_idx * head_dim;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Shared memory for scores and cross-warp reduction (始终 FP32)
    extern __shared__ float smem[];
    float *s_scores = smem;
    float *s_reduce = smem + block_size;  // [2 * num_warps] for max and sum

    int num_warps = nthreads / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    float m = -1e30f;
    float l = 0.0f;

    // Thread-local accumulator: each thread handles dims [tid, tid+nthreads, ...]
    // 累加器始终 FP32，最终写出时转回 T
    float acc[8];
    int dims_per_thread = (head_dim + nthreads - 1) / nthreads;
    for (int i = 0; i < dims_per_thread; ++i) acc[i] = 0.0f;

    for (int bi = 0; bi < num_blocks; ++bi) {
        int block_id = block_tables[batch_idx * max_blocks_per_seq + bi];
        int tokens_in_block = min(block_size, seq_len - bi * block_size); //最后一个 block 可能不满，所以要算真实 token 数。

        // KV pool 按字节寻址，reinterpret_cast 到实际 I/O 类型 T
        const T *k_base = reinterpret_cast<const T *>( //k_base = K pool 里第 block_id 个物理 block第 layer_idx 层的起始地址
            k_pool + (size_t)block_id * pool_block_stride +
            (size_t)layer_idx * pool_layer_stride);
        const T *v_base = reinterpret_cast<const T *>(
            v_pool + (size_t)block_id * pool_block_stride +
            (size_t)layer_idx * pool_layer_stride);

        // Compute QK^T scores: threads partition over head_dim, reduce to score
        for (int t = 0; t < tokens_in_block; ++t) {
            const T *k_vec = k_base + t * num_kv_heads * head_dim + kv_head_idx * head_dim;

            // 读入 T → 转 float 计算点积
            float partial = 0.0f;
            for (int d = tid; d < head_dim; d += nthreads)
                partial += to_float(q_vec[d]) * to_float(k_vec[d]);

            // Warp-level reduction (FP32)
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

        // Online softmax: all threads read the same scores (FP32),
        // but each accumulates its own subset of V dimensions
        for (int t = 0; t < tokens_in_block; ++t) {
            float score = s_scores[t];
            float m_new = fmaxf(m, score);
            float p = expf(score - m_new);
            float correction = expf(m - m_new);
            l = correction * l + p;

            const T *v_vec = v_base + t * num_kv_heads * head_dim + kv_head_idx * head_dim;
            for (int i = 0; i < dims_per_thread; ++i) {
                int d = tid + i * nthreads;
                if (d < head_dim)
                    acc[i] = correction * acc[i] + p * to_float(v_vec[d]);
            }
            m = m_new;
        }
        __syncthreads();
    }

    // Write output: FP32 累加结果 → from_float 转回 T 类型写出
    T *out_vec = output + batch_idx * num_heads * head_dim + head_idx * head_dim;
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * nthreads;
        if (d < head_dim)
            out_vec[d] = from_float<T>(acc[i] * inv_l);
    }
}

namespace llaisys::ops::nvidia {

// ── 模板化 GPU paged attention 启动器 ──────────────────────────────
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

    dim3 grid(batch_size, num_heads); //当前 CUDA block 负责：第 batch_idx 个 sequence、 第 head_idx 个 Q head
    // Use WARP_SIZE threads per block (sufficient for head_dim=128)
    // For head_dim > WARP_SIZE, use multiple warps
    int threads = min(128, max(WARP_SIZE, ((head_dim + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE));
    int num_warps = threads / WARP_SIZE;
    size_t smem = block_size * sizeof(float) + 2 * num_warps * sizeof(float);

    paged_attention_kernel<T><<<grid, threads, smem>>>(
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

// ── Device-pointer variant (CUDA Graph 兼容, 无 malloc/free) ──────
template<typename T>
static void paged_attention_device_typed(
    T *output, const T *query,
    const void *k_pool, const void *v_pool,
    const int *d_block_tables, const int *d_seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    dim3 grid(batch_size, num_heads);
    int threads = min(128, max(WARP_SIZE, ((head_dim + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE));
    int num_warps = threads / WARP_SIZE;
    size_t smem = block_size * sizeof(float) + 2 * num_warps * sizeof(float);

    paged_attention_kernel<T><<<grid, threads, smem>>>(
        output, query,
        static_cast<const char *>(k_pool),
        static_cast<const char *>(v_pool),
        d_block_tables, d_seq_lens,
        num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq,
        pool_block_stride, pool_layer_stride,
        layer_idx, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ── dtype 分发入口 ────────────────────────────────────────────────
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
    llaisysDataType_t dtype)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        paged_attention_device_typed<float>(
            (float*)output, (const float*)query,
            k_pool, v_pool, block_tables_dev, seq_lens_dev,
            batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride,
            layer_idx, scale);
        break;
    case LLAISYS_DTYPE_F16:
        paged_attention_device_typed<__half>(
            (__half*)output, (const __half*)query,
            k_pool, v_pool, block_tables_dev, seq_lens_dev,
            batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride,
            layer_idx, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        paged_attention_device_typed<__nv_bfloat16>(
            (__nv_bfloat16*)output, (const __nv_bfloat16*)query,
            k_pool, v_pool, block_tables_dev, seq_lens_dev,
            batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride,
            layer_idx, scale);
        break;
    default:
        throw std::runtime_error("paged_attention_device: unsupported dtype");
    }
}

} // namespace llaisys::ops::nvidia

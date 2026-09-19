#include <llmops/ops/blas_support.hpp>
#include "../../nvidia/cublas.hpp"
#include "../../dispatch.hpp"
#include "self_attention_nvidia.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",              \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);       \
            throw std::runtime_error(cudaGetErrorString(err));                    \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[cuBLAS ERROR] code %d at %s:%d\n",                  \
                    (int)status, __FILE__, __LINE__);                             \
            throw std::runtime_error("cuBLAS call failed");                       \
        }                                                                         \
    } while (0)

// ---- FP32/FP16/BF16 conversion helpers ----
// ----------------------------------------------------------------
// Lazy-initialized thread-local cuBLAS handle
// ----------------------------------------------------------------
// ----------------------------------------------------------------
// Batched self-attention: all heads computed in parallel via
// cublasSgemmStridedBatched. Eliminates per-head loop.
// ----------------------------------------------------------------
namespace llaisys::ops::nvidia {

template<typename T>
static void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int64_t seq_len   = q->shape()[0];
    int64_t n_head    = q->shape()[1];
    int64_t head_dim  = q->shape()[2];
    int64_t total_len = k->shape()[0];
    int64_t n_kv_head = k->shape()[1];
    int64_t v_dim     = v->shape()[2];
    int64_t group_size = n_head / n_kv_head;

    const T *q_ptr = reinterpret_cast<const T *>(q->data());
    const T *k_ptr = reinterpret_cast<const T *>(k->data());
    const T *v_ptr = reinterpret_cast<const T *>(v->data());
    T *o_ptr = reinterpret_cast<T *>(attn_val->data());

    ptrdiff_t q_s0 = q->strides()[0], q_s1 = q->strides()[1], q_s2 = q->strides()[2];
    ptrdiff_t k_s0 = k->strides()[0], k_s1 = k->strides()[1], k_s2 = k->strides()[2];
    ptrdiff_t v_s0 = v->strides()[0], v_s1 = v->strides()[1], v_s2 = v->strides()[2];
    ptrdiff_t o_s0 = attn_val->strides()[0], o_s1 = attn_val->strides()[1], o_s2 = attn_val->strides()[2];

    cublasHandle_t handle = get_cublas_handle();

    // Ensure cached temp buffers: [n_head, ...] for all heads
    size_t q_need = (size_t)(n_head * seq_len * head_dim) * sizeof(float);
    size_t k_need = (size_t)(n_head * total_len * head_dim) * sizeof(float);
    size_t v_need = (size_t)(n_head * total_len * v_dim) * sizeof(float);
    size_t s_need = (size_t)(n_head * seq_len * total_len) * sizeof(float);
    size_t o_need = (size_t)(n_head * seq_len * v_dim) * sizeof(float);

    auto *s_q_all=vendor_buffer<float>(3,q_need/sizeof(float));
    auto *s_k_all=vendor_buffer<float>(4,k_need/sizeof(float));
    auto *s_v_all=vendor_buffer<float>(5,v_need/sizeof(float));
    auto *s_scores=vendor_buffer<float>(6,s_need/sizeof(float));
    auto *s_o_all=vendor_buffer<float>(7,o_need/sizeof(float));


    // 1. Gather ALL Q heads: T -> float [n_head, seq_len, head_dim]
    {


        llmops::cuda::support::gather_all_q_kernel<T>(
            s_q_all, q_ptr, seq_len, n_head, head_dim, q_s0, q_s1, q_s2, cudaStreamPerThread);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. Gather + expand KV: T -> float [n_head, total_len, dim]
    {


        llmops::cuda::support::gather_expand_kv_kernel<T>(
            s_k_all, k_ptr, total_len, n_head, n_kv_head, head_dim, group_size,
            k_s0, k_s1, k_s2, cudaStreamPerThread);
        CUDA_CHECK(cudaGetLastError());
    }
    {


        llmops::cuda::support::gather_expand_kv_kernel<T>(
            s_v_all, v_ptr, total_len, n_head, n_kv_head, v_dim, group_size,
            v_s0, v_s1, v_s2, cudaStreamPerThread);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. Batched GEMM: scores[h] = Q[h] * K[h]^T for all heads
    //    Q[h]: [seq_len, head_dim], K[h]: [total_len, head_dim]
    //    scores[h]: [seq_len, total_len]
    //    Col-major: scores^T = K * Q^T
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * head_dim);
        long long strideB = (long long)(seq_len * head_dim);
        long long strideC = (long long)(seq_len * total_len);
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)total_len, (int)seq_len, (int)head_dim,
            &alpha,
            s_k_all, (int)head_dim, strideA,
            s_q_all, (int)head_dim, strideB,
            &beta,
            s_scores, (int)total_len, strideC,
            (int)n_head));
    }

    // 4. Scale + causal mask (single kernel for all heads)
    {


        llmops::cuda::support::scale_kernel(s_scores, scale, n_head * seq_len * total_len, cudaStreamPerThread);
        CUDA_CHECK(cudaGetLastError());
        llmops::cuda::support::causal_mask_batched_kernel(s_scores, n_head, seq_len, total_len, cudaStreamPerThread);
        CUDA_CHECK(cudaGetLastError());
    }

    // 5. Softmax per row (n_head * seq_len rows)
    {
        int64_t num_rows = n_head * seq_len;



        llmops::cuda::support::softmax_row_kernel(s_scores, num_rows, total_len, cudaStreamPerThread);
        CUDA_CHECK(cudaGetLastError());
    }

    // 6. Batched GEMM: O[h] = probs[h] * V[h] for all heads
    //    probs[h]: [seq_len, total_len], V[h]: [total_len, v_dim]
    //    O[h]: [seq_len, v_dim]
    //    Col-major: O^T = V^T * probs^T  =>  CUBLAS_OP_N, CUBLAS_OP_N
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * v_dim);
        long long strideB = (long long)(seq_len * total_len);
        long long strideC = (long long)(seq_len * v_dim);
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)v_dim, (int)seq_len, (int)total_len,
            &alpha,
            s_v_all, (int)v_dim, strideA,
            s_scores, (int)total_len, strideB,
            &beta,
            s_o_all, (int)v_dim, strideC,
            (int)n_head));
    }

    // 7. Scatter ALL heads: float -> T [seq_len, n_head, v_dim]
    {


        llmops::cuda::support::scatter_all_heads_kernel<T>(
            o_ptr, s_o_all, seq_len, n_head, v_dim, o_s0, o_s1, o_s2, cudaStreamPerThread);
        CUDA_CHECK(cudaGetLastError());
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    record_dispatch("self_attention", "framework.cublas");
    auto dtype = q->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl<float>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_impl<__half>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_impl<__nv_bfloat16>(attn_val, q, k, v, scale);
        break;
    default:
        throw std::runtime_error("NVIDIA self_attention: unsupported dtype");
    }
}

} // namespace llaisys::ops::nvidia

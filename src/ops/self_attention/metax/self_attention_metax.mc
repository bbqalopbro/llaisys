#include <llmops/ops/blas_support.hpp>
#include <cstdlib>
#include <string>
// =============================================================
// MetaX (沐曦) C500 — Self-Attention kernel
// =============================================================
#include "self_attention_metax.hpp"

#include <mcblas.h>
#include <mc_runtime_api.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>

#define GPU_CHECK(call)                                                           \
    do {                                                                          \
        auto err = (call);                                                        \
        if (err != 0) {                                                           \
            fprintf(stderr, "[MetaX GPU ERROR] code %d at %s:%d\n",              \
                    (int)err, __FILE__, __LINE__);                                \
            throw std::runtime_error("MetaX GPU call failed");                    \
        }                                                                         \
    } while (0)

#define BLAS_CHECK(call)                                                          \
    do {                                                                          \
        auto status = (call);                                                     \
        if (status != MCBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[MetaX BLAS ERROR] code %d at %s:%d\n",             \
                    (int)status, __FILE__, __LINE__);                             \
            throw std::runtime_error("MetaX BLAS call failed");                   \
        }                                                                         \
    } while (0)

// ---- FP32/FP16/BF16 conversion helpers ----
static mcblasHandle_t get_blas_handle() {
    static thread_local mcblasHandle_t handle = nullptr;
    if (!handle) {
        BLAS_CHECK(mcblasCreate(&handle));
    }
    return handle;
}

// ----------------------------------------------------------------
// Cached GPU temp buffers — grow-only, never freed during inference.
// ----------------------------------------------------------------
static float *s_q_all = nullptr;
static float *s_k_all = nullptr;
static float *s_v_all = nullptr;
static float *s_scores = nullptr;
static float *s_o_all = nullptr;
static size_t s_q_all_sz = 0;
static size_t s_k_all_sz = 0;
static size_t s_v_all_sz = 0;
static size_t s_scores_sz = 0;
static size_t s_o_all_sz = 0;

static void ensure_buf(float *&ptr, size_t &cur, size_t need) {
    if (need <= cur) return;
    if (ptr) mcFree(ptr);
    GPU_CHECK(mcMalloc(&ptr, need));
    cur = need;
}

// ----------------------------------------------------------------
// Batched self-attention: all heads computed in parallel via
// mcblasSgemmStridedBatched (mxBLAS on MetaX).
// ----------------------------------------------------------------
namespace llaisys::ops::metax {

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

    const char *mode=std::getenv("LLAISYS_METAX_BLAS");
    if (!mode || std::string(mode)!="mcblas") throw std::runtime_error("MetaX native self_attention unavailable; explicitly select LLAISYS_METAX_BLAS=mcblas");
    mcblasHandle_t handle = get_blas_handle();

    // Ensure cached temp buffers
    size_t q_need = (size_t)(n_head * seq_len * head_dim) * sizeof(float);
    size_t k_need = (size_t)(n_head * total_len * head_dim) * sizeof(float);
    size_t v_need = (size_t)(n_head * total_len * v_dim) * sizeof(float);
    size_t s_need = (size_t)(n_head * seq_len * total_len) * sizeof(float);
    size_t o_need = (size_t)(n_head * seq_len * v_dim) * sizeof(float);

    ensure_buf(s_q_all, s_q_all_sz, q_need);
    ensure_buf(s_k_all, s_k_all_sz, k_need);
    ensure_buf(s_v_all, s_v_all_sz, v_need);
    ensure_buf(s_scores, s_scores_sz, s_need);
    ensure_buf(s_o_all, s_o_all_sz, o_need);


    // 1. Gather ALL Q heads: T -> float [n_head, seq_len, head_dim]
    {


        llmops::metax::support::gather_all_q_kernel<T>(
            s_q_all, q_ptr, seq_len, n_head, head_dim, q_s0, q_s1, q_s2, nullptr);
        GPU_CHECK(mcGetLastError());
    }

    // 2. Gather + expand KV: T -> float [n_head, total_len, dim]
    {


        llmops::metax::support::gather_expand_kv_kernel<T>(
            s_k_all, k_ptr, total_len, n_head, n_kv_head, head_dim, group_size,
            k_s0, k_s1, k_s2, nullptr);
        GPU_CHECK(mcGetLastError());
    }
    {


        llmops::metax::support::gather_expand_kv_kernel<T>(
            s_v_all, v_ptr, total_len, n_head, n_kv_head, v_dim, group_size,
            v_s0, v_s1, v_s2, nullptr);
        GPU_CHECK(mcGetLastError());
    }

    // 3. Batched GEMM: scores[h] = Q[h] * K[h]^T for all heads
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * head_dim);
        long long strideB = (long long)(seq_len * head_dim);
        long long strideC = (long long)(seq_len * total_len);
        BLAS_CHECK(mcblasSgemmStridedBatched(handle,
            MCBLAS_OP_T, MCBLAS_OP_N,
            (int)total_len, (int)seq_len, (int)head_dim,
            &alpha,
            s_k_all, (int)head_dim, strideA,
            s_q_all, (int)head_dim, strideB,
            &beta,
            s_scores, (int)total_len, strideC,
            (int)n_head));
    }

    // 4. Scale + causal mask
    {


        llmops::metax::support::scale_kernel(s_scores, scale, n_head * seq_len * total_len, nullptr);
        GPU_CHECK(mcGetLastError());
        llmops::metax::support::causal_mask_batched_kernel(s_scores, n_head, seq_len, total_len, nullptr);
        GPU_CHECK(mcGetLastError());
    }

    // 5. Softmax per row (n_head * seq_len rows)
    {
        int64_t num_rows = n_head * seq_len;



        llmops::metax::support::softmax_row_kernel(s_scores, num_rows, total_len, nullptr);
        GPU_CHECK(mcGetLastError());
    }

    // 6. Batched GEMM: O[h] = probs[h] * V[h] for all heads
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * v_dim);
        long long strideB = (long long)(seq_len * total_len);
        long long strideC = (long long)(seq_len * v_dim);
        BLAS_CHECK(mcblasSgemmStridedBatched(handle,
            MCBLAS_OP_N, MCBLAS_OP_N,
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


        llmops::metax::support::scatter_all_heads_kernel<T>(
            o_ptr, s_o_all, seq_len, n_head, v_dim, o_s0, o_s1, o_s2, nullptr);
        GPU_CHECK(mcGetLastError());
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    auto dtype = q->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl<float>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_impl<__half>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_impl<__maca_bfloat16>(attn_val, q, k, v, scale);
        break;
    default:
        throw std::runtime_error("MetaX self_attention: unsupported dtype");
    }
}

} // namespace llaisys::ops::metax

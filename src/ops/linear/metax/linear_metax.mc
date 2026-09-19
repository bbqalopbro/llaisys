#include <llmops/ops/blas_support.hpp>
#include <cstdlib>
#include <string>
// =============================================================
// MetaX (沐曦) C500 — Linear kernel (Y = X * W^T + bias)
// 使用 mxBLAS (沐曦 BLAS 库) 进行矩阵乘法
//
// 注：当 MXMACA SDK 不可用时（当前开发环境），
// 使用 CUDA cuBLAS 头文件编译，到沐曦平台上替换为 mxBLAS。
// mxBLAS API 与 cuBLAS 保持高度兼容。
// =============================================================
#include "linear_metax.hpp"

#include "../../../utils.hpp"

// BLAS 库头文件
// 在沐曦平台上替换为: #include <mxblas.h>
#include <mcblas.h>
#include <mc_runtime_api.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>
#include <cstdio>
#include <stdexcept>

#define BLAS_CHECK(call)                                                          \
    do {                                                                          \
        mcblasStatus_t status = (call);                                           \
        if (status != MCBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[MetaX BLAS ERROR] code %d at %s:%d\n",             \
                    (int)status, __FILE__, __LINE__);                             \
            throw std::runtime_error("MetaX BLAS call failed");                   \
        }                                                                         \
    } while (0)

#define GPU_CHECK(call)                                                           \
    do {                                                                          \
        auto err = (call);                                                        \
        if (err != 0) {                                                           \
            fprintf(stderr, "[MetaX GPU ERROR] code %d at %s:%d\n",              \
                    (int)err, __FILE__, __LINE__);                                \
            throw std::runtime_error("MetaX GPU call failed");                    \
        }                                                                         \
    } while (0)

// Lazy-initialized thread-local BLAS handle
// 注：在沐曦平台上 mcblasHandle_t → mxblasHandle_t, mcblasCreate → mxblasCreate
static mcblasHandle_t get_blas_handle() {
    static thread_local mcblasHandle_t handle = nullptr;
    if (!handle) {
        BLAS_CHECK(mcblasCreate(&handle));
    }
    return handle;
}

namespace llaisys::ops::metax {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto w_dtype = weight->dtype();
    auto in_dtype = in->dtype();
    auto out_dtype = out->dtype();

    int64_t M = in->shape()[0];
    int64_t K = in->shape()[1];
    int64_t N = weight->shape()[0];

    const char *mode=std::getenv("LLAISYS_METAX_BLAS");
    if (!mode || std::string(mode)!="mcblas") throw std::runtime_error("MetaX native linear unavailable; explicitly select LLAISYS_METAX_BLAS=mcblas");
    mcblasHandle_t handle = get_blas_handle();

    float alpha = 1.0f;
    float beta  = 0.0f;

    // ---- Mixed precision path: FP16 weight + FP32 input → FP32 output ----
    if (w_dtype == LLAISYS_DTYPE_F16 && in_dtype == LLAISYS_DTYPE_F32 && out_dtype == LLAISYS_DTYPE_F32) {
        static thread_local __half *in_f16_buf = nullptr;
        static thread_local int64_t in_f16_cap = 0;

        int64_t in_elems = M * K;
        if (in_elems > in_f16_cap) {
            if (in_f16_buf) mcFree(in_f16_buf);
            mcMalloc(&in_f16_buf, in_elems * sizeof(__half));
            in_f16_cap = in_elems;
        }

        llmops::metax::support::convert_f32_to_f16_kernel(in_f16_buf, (const float*)in->data(), in_elems, nullptr);

        BLAS_CHECK(mcblasGemmEx(handle,
                                MCBLAS_OP_T, MCBLAS_OP_N,
                                (int)N, (int)M, (int)K,
                                &alpha,
                                weight->data(), MACA_R_16F, (int)K,
                                in_f16_buf,     MACA_R_16F, (int)K,
                                &beta,
                                out->data(),    MACA_R_32F, (int)N,
                                MCBLAS_COMPUTE_32F,
                                MCBLAS_GEMM_DEFAULT));

        if (bias && bias->data()) {


            if (bias->dtype() == LLAISYS_DTYPE_F16) {
                llmops::metax::support::add_bias_f16_to_f32_kernel(
                    (float*)out->data(), (const __half*)bias->data(), M, N, nullptr);
            } else {
                llmops::metax::support::add_bias_kernel<float>(
                    (float*)out->data(), (const float*)bias->data(), M, N, nullptr);
            }
            GPU_CHECK(mcGetLastError());
        }
        return;
    }

    // ---- Standard path ----
    macaDataType_t maca_dtype;
    switch (w_dtype) {
    case LLAISYS_DTYPE_F32:  maca_dtype = MACA_R_32F;  break;
    case LLAISYS_DTYPE_F16:  maca_dtype = MACA_R_16F;  break;
    case LLAISYS_DTYPE_BF16: maca_dtype = MACA_R_16BF; break;
    default:
        throw std::runtime_error("MetaX linear: unsupported dtype");
    }

    BLAS_CHECK(mcblasGemmEx(handle,
                            MCBLAS_OP_T, MCBLAS_OP_N,
                            (int)N, (int)M, (int)K,
                            &alpha,
                            weight->data(), maca_dtype, (int)K,
                            in->data(),     maca_dtype, (int)K,
                            &beta,
                            out->data(),    maca_dtype, (int)N,
                            MCBLAS_COMPUTE_32F,
                            MCBLAS_GEMM_DEFAULT));

    if (bias && bias->data()) {


        switch (w_dtype) {
        case LLAISYS_DTYPE_F32:
            llmops::metax::support::add_bias_kernel<float>(
                (float*)out->data(), (const float*)bias->data(), M, N, nullptr);
            break;
        case LLAISYS_DTYPE_F16:
            llmops::metax::support::add_bias_kernel<__half>(
                (__half*)out->data(), (const __half*)bias->data(), M, N, nullptr);
            break;
        case LLAISYS_DTYPE_BF16:
            llmops::metax::support::add_bias_kernel<__maca_bfloat16>(
                (__maca_bfloat16*)out->data(), (const __maca_bfloat16*)bias->data(), M, N, nullptr);
            break;
        default: break;
        }
        GPU_CHECK(mcGetLastError());
    }
}

} // namespace llaisys::ops::metax

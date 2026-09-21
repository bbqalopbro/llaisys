#include <llmops/ops/blas_support.hpp>
#include "../../nvidia/cublas.hpp"
#include "linear_nvidia.cuh"
#include "../../llmops.hpp"
#include <map>
#include <memory>

#ifdef LLAISYS_USE_LLMOPS
#include "adapter.hpp"
#endif

#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t status = (call);                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "[cuBLAS ERROR] code %d at %s:%d\n",                      \
                    (int)status, __FILE__, __LINE__);                                 \
            throw std::runtime_error("cuBLAS call failed");                           \
        }                                                                             \
    } while (0)

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",                  \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);           \
            throw std::runtime_error(cudaGetErrorString(err));                        \
        }                                                                             \
    } while (0)

// Lazy-initialized thread-local cuBLAS handle with pre-allocated workspace
// for CUDA Graph compatibility (cuBLAS must not call cudaMalloc during capture)
namespace llaisys::ops::nvidia {

// Y = X * W^T + bias
// Uses cublasGemmEx to support F32/F16/BF16 with F32 compute.
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
#ifdef LLAISYS_USE_LLMOPS
    if (llmops_integration::try_linear(out, in, weight, bias)) return;
#endif
    if(out->ndim()!=2 || in->ndim()!=2 || weight->ndim()!=2 || out->shape()[0]!=in->shape()[0] || out->shape()[1]!=weight->shape()[0] || in->shape()[1]!=weight->shape()[1])
        throw std::runtime_error("cuBLAS linear shape mismatch");
    for(auto t:{out,in,weight,bias}) if(t && !t->isContiguous()) throw std::runtime_error("framework cuBLAS linear currently requires contiguous tensors");
    bool same=out->dtype()==in->dtype() && out->dtype()==weight->dtype();
    bool mixed1=weight->dtype()==LLAISYS_DTYPE_F16 && out->dtype()==LLAISYS_DTYPE_F32 && (in->dtype()==LLAISYS_DTYPE_F16 || in->dtype()==LLAISYS_DTYPE_F32);
    bool mixed2=weight->dtype()==LLAISYS_DTYPE_F32 && in->dtype()==LLAISYS_DTYPE_F16 && (out->dtype()==LLAISYS_DTYPE_F16 || out->dtype()==LLAISYS_DTYPE_F32);
    if(!same && !mixed1 && !mixed2) throw std::runtime_error("cuBLAS linear dtype combination unsupported");
    if(bias && (bias->ndim()!=1 || bias->shape()[0]!=out->shape()[1] ||
        (same ? bias->dtype()!=out->dtype() : (bias->dtype()!=LLAISYS_DTYPE_F16 && bias->dtype()!=LLAISYS_DTYPE_F32))))
        throw std::runtime_error("cuBLAS linear bias mismatch");
    if(!out->numel()) return;
    record_dispatch("linear", "framework.cublas");
    auto w_dtype = weight->dtype();
    auto in_dtype = in->dtype();
    auto out_dtype = out->dtype();

    int64_t M = in->shape()[0];
    int64_t K = in->shape()[1];
    int64_t N = weight->shape()[0];

    cublasHandle_t handle = get_cublas_handle();

    float alpha = 1.0f;
    float beta  = 0.0f;

    // ---- Mixed precision path: FP16 weight → FP32 output ----
    // 情况1: F16w × F32in → F32out (BF16 模型 + FP32 激活)
    // 情况2: F16w × F16in → F32out (LM Head: FP16 权重/激活, FP32 logits)
    if (w_dtype == LLAISYS_DTYPE_F16 && out_dtype == LLAISYS_DTYPE_F32) {
        const void *in_gemm_ptr = in->data();

        // If input is F32, convert to F16 first (cuBLAS requires A/B same type)
        if (in_dtype == LLAISYS_DTYPE_F32) {
            int64_t in_elems = M * K;
            auto *in_f16_buf=vendor_buffer<__half>(0,in_elems);

            llmops::cuda::support::convert_f32_to_f16_kernel(in_f16_buf, (const float*)in->data(), in_elems, cudaStreamPerThread);
            in_gemm_ptr = in_f16_buf;
        }
        // else: input is already F16, use directly

        // cuBLAS: A=F16 (weight), B=F16 (input), C=F32 (output)
        CUBLAS_CHECK(cublasGemmEx(handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  (int)N, (int)M, (int)K,
                                  &alpha,
                                  weight->data(), CUDA_R_16F, (int)K,
                                  in_gemm_ptr,    CUDA_R_16F, (int)K,
                                  &beta,
                                  out->data(),    CUDA_R_32F, (int)N,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));

        // Add FP16 bias to FP32 output
        if (bias && bias->data()) {


            if (bias->dtype() == LLAISYS_DTYPE_F16) {
                llmops::cuda::support::add_bias_f16_to_f32_kernel(
                    (float*)out->data(), (const __half*)bias->data(), M, N, cudaStreamPerThread);
            } else {
                llmops::cuda::support::add_bias_kernel<float>(
                    (float*)out->data(), (const float*)bias->data(), M, N, cudaStreamPerThread);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // ---- Mixed precision path: F32 weight + FP16 input ----
    // 场景: 量化权重 dequant 到 FP32 后, 与 FP16 激活做 GEMM
    // 策略: FP16 input → 转 F32 → cublasGemmEx(F32, F32, F32) → 转 FP16 output (或直接 F32)
    if (w_dtype == LLAISYS_DTYPE_F32 && in_dtype == LLAISYS_DTYPE_F16) {
        // 1. Convert FP16 input → FP32 (input is small: [B, K])
        int64_t in_elems = M * K;
        auto *in_f32_buf=vendor_buffer<float>(1,in_elems);

        llmops::cuda::support::convert_f16_to_f32_kernel(in_f32_buf, (const __half*)in->data(), in_elems, cudaStreamPerThread);

        if (out_dtype == LLAISYS_DTYPE_F32) {
            // F32 weight × F16 input → F32 output (lm_head 场景)
            CUBLAS_CHECK(cublasGemmEx(handle,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      (int)N, (int)M, (int)K,
                                      &alpha,
                                      weight->data(), CUDA_R_32F, (int)K,
                                      in_f32_buf,     CUDA_R_32F, (int)K,
                                      &beta,
                                      out->data(),    CUDA_R_32F, (int)N,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));
            // Bias
            if (bias && bias->data()) {


                if(bias->dtype()==LLAISYS_DTYPE_F16)
                    llmops::cuda::support::add_bias_f16_to_f32_kernel((float*)out->data(),bias->data(),M,N,cudaStreamPerThread);
                else llmops::cuda::support::add_bias_kernel<float>(
                    (float*)out->data(), (const float*)bias->data(), M, N, cudaStreamPerThread);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // F32 weight × F16 input → FP16 output (标准激活场景)
            // 2. Allocate FP32 output buffer
            int64_t out_elems = M * N;
            auto *out_f32_buf=vendor_buffer<float>(2,out_elems);

            // 3. cublasGemmEx: F32 × F32 → F32
            CUBLAS_CHECK(cublasGemmEx(handle,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      (int)N, (int)M, (int)K,
                                      &alpha,
                                      weight->data(), CUDA_R_32F, (int)K,
                                      in_f32_buf,     CUDA_R_32F, (int)K,
                                      &beta,
                                      out_f32_buf,    CUDA_R_32F, (int)N,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));

            // 4. Add bias to FP32 buffer BEFORE converting to FP16
            if (bias && bias->data()) {


                if (bias->dtype() == LLAISYS_DTYPE_F32) {
                    llmops::cuda::support::add_bias_kernel<float>(
                        out_f32_buf, (const float*)bias->data(), M, N, cudaStreamPerThread);
                } else {
                    // FP16 bias → FP32 buffer
                    llmops::cuda::support::add_bias_f16_to_f32_kernel(
                        out_f32_buf, (const __half*)bias->data(), M, N, cudaStreamPerThread);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            // 5. Convert F32 output (with bias) → FP16

            llmops::cuda::support::convert_f32_to_f16_kernel((__half*)out->data(), out_f32_buf, out_elems, cudaStreamPerThread);
        }
        return;
    }

    // ---- Standard path: all tensors have the same dtype ----
    cudaDataType_t cuda_dtype;
    switch (w_dtype) {
    case LLAISYS_DTYPE_F32:  cuda_dtype = CUDA_R_32F;  break;
    case LLAISYS_DTYPE_F16:  cuda_dtype = CUDA_R_16F;  break;
    case LLAISYS_DTYPE_BF16: cuda_dtype = CUDA_R_16BF; break;
    default:
        throw std::runtime_error("NVIDIA linear: unsupported dtype");
    }

    // row-major Y = X * W^T  <=>  col-major Y^T = W * X^T
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_T,    // W stored row-major, viewed col-major => transpose
                              CUBLAS_OP_N,    // X stored row-major, X^T in col-major => no-transpose
                              (int)N, (int)M, (int)K,
                              &alpha,
                              weight->data(), cuda_dtype, (int)K,
                              in->data(),     cuda_dtype, (int)K,
                              &beta,
                              out->data(),    cuda_dtype, (int)N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));

    // Add bias if present
    if (bias && bias->data()) {


        switch (w_dtype) {
        case LLAISYS_DTYPE_F32:
            llmops::cuda::support::add_bias_kernel<float>(
                (float*)out->data(), (const float*)bias->data(), M, N, cudaStreamPerThread);
            break;
        case LLAISYS_DTYPE_F16:
            llmops::cuda::support::add_bias_kernel<__half>(
                (__half*)out->data(), (const __half*)bias->data(), M, N, cudaStreamPerThread);
            break;
        case LLAISYS_DTYPE_BF16:
            llmops::cuda::support::add_bias_kernel<__nv_bfloat16>(
                (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N, cudaStreamPerThread);
            break;
        default: break;
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

// Y = X * W^T + bias + residual  (fused GEMV+Add for M=1 FP16 decode)
// Falls back to linear() + separate add for non-M=1 or non-FP16 cases
void linear_add(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
                tensor_t residual) {
#ifdef LLAISYS_USE_LLMOPS
    if (llmops_integration::try_linear(out, in, weight, bias, residual)) return;
#endif

    if(residual && (residual->dtype()!=out->dtype() || residual->shape()!=out->shape() || !residual->isContiguous()))
        throw std::runtime_error("cuBLAS linear_add residual layout mismatch");
    linear(out, in, weight, bias);
    if (residual) llmops::cuda::add(out->data(),out->data(),residual->data(),kernel_dtype(out->dtype()),out->numel(),cudaStreamPerThread);
}

// ---- Fused W4A16 Linear: INT4 权重 × FP16 输入 ----
//
// 调用路径: qwen2.cpp linear_maybe_dequant() → ops::linear_int4() → 本函数
//
// M=1 (decode): 走 fused GEMV, 一个 kernel 完成 dequant + matmul + bias + residual
//   - 无中间缓冲区, 无 cudaMalloc, 对 CUDA Graph 友好
//   - 支持 FP16 输出 (层间) 和 FP32 输出 (lm_head)
//
// M>1 (prefill): 由调用者回退到 dequant_int4 + cuBLAS GEMM
//   - GEMM 是 compute-bound, cuBLAS 用 Tensor Core 更快
//   - prefill 只执行一次, 不影响吞吐
//
void linear_int4(tensor_t out, tensor_t in, tensor_t weight, tensor_t scale,
                 tensor_t bias, int group_size, tensor_t residual) {
    llmops::cuda::linear_int4(view(out),view(in),view(weight),view(scale),view(bias),group_size,view(residual),cudaStreamPerThread);
    record_dispatch("linear_int4", "llmops.cuda.w4a16");
}
} // namespace llaisys::ops::nvidia

namespace llaisys::ops::nvidia {
void linear_fp8(tensor_t out, tensor_t in, tensor_t sx, tensor_t w,
                tensor_t sw, tensor_t bias, tensor_t residual) {
    llmops_integration::fp8_linear(out,in,sx,w,sw,bias,residual);
}
void quantize_fp8(tensor_t q, tensor_t scales, tensor_t in, tensor_t weight,
                  tensor_t up, tensor_t floating, float eps) {
    llmops_integration::fp8_quantize(q,scales,in,weight,up,floating,eps);
}
}

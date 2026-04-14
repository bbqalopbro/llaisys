#include "linear_nvidia.cuh"

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

// ---- GEMV kernel for M=1 decode (FP16 weights+input, FP32 accumulation) ----
// y[row] = dot(W[row, :], x[:]) + bias[row]
// W is row-major [N, K], x is [K], y is [N]
// Uses half2 vectorized loads for 2x bandwidth
template<int WARPS_PER_ROW>
__global__ void gemv_f16_kernel(const __half *__restrict__ W,
                                const __half *__restrict__ x,
                                __half *__restrict__ y,
                                const __half *__restrict__ bias,
                                int N, int K) {
    const int WARP_SIZE = 32;
    const int THREADS_PER_ROW = WARPS_PER_ROW * WARP_SIZE;

    int local_thread = threadIdx.x;
    int row_in_block = local_thread / THREADS_PER_ROW;
    int thread_in_row = local_thread % THREADS_PER_ROW;
    int warp_in_row = thread_in_row / WARP_SIZE;
    int lane_id = thread_in_row % WARP_SIZE;

    int rows_per_block = blockDim.x / THREADS_PER_ROW;
    int row = blockIdx.x * rows_per_block + row_in_block;
    if (row >= N) return;

    const __half *row_ptr = W + (int64_t)row * K;
    float sum = 0.0f;

    // half2 vectorized load (2 halfs = 4 bytes per load)
    int k2 = K / 2;
    int global_lane = warp_in_row * WARP_SIZE + lane_id;
    for (int i = global_lane; i < k2; i += THREADS_PER_ROW) {
        half2 w2 = reinterpret_cast<const half2 *>(row_ptr)[i];
        half2 x2 = reinterpret_cast<const half2 *>(x)[i];
        sum += __half2float(w2.x) * __half2float(x2.x)
             + __half2float(w2.y) * __half2float(x2.y);
    }
    // Handle odd K
    if (global_lane == 0 && (K & 1)) {
        sum += __half2float(row_ptr[K - 1]) * __half2float(x[K - 1]);
    }

    // Warp shuffle reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduce via shared memory
    extern __shared__ float smem[];
    if (lane_id == 0) {
        smem[row_in_block * WARPS_PER_ROW + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0 && lane_id == 0) {
        float total = 0.0f;
        for (int w = 0; w < WARPS_PER_ROW; w++) {
            total += smem[row_in_block * WARPS_PER_ROW + w];
        }
        if (bias) {
            total += __half2float(bias[row]);
        }
        y[row] = __float2half(total);
    }
}

// GEMV launcher: WARPS_PER_ROW tuned by K dimension
static void gemv_f16_launch(const __half *W, const __half *x, __half *y,
                            const __half *bias, int N, int K,
                            cudaStream_t stream = 0) {
    if (K <= 512) {
        // 1 warp/row, 8 rows/block = 256 threads
        constexpr int WPR = 1;
        int rpb = 256 / (WPR * 32);  // 8
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_f16_kernel<WPR><<<grid, 256, smem, stream>>>(W, x, y, bias, N, K);
    } else if (K <= 2048) {
        // 2 warps/row, 4 rows/block
        constexpr int WPR = 2;
        int rpb = 256 / (WPR * 32);  // 4
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_f16_kernel<WPR><<<grid, 256, smem, stream>>>(W, x, y, bias, N, K);
    } else {
        // 4 warps/row, 2 rows/block
        constexpr int WPR = 4;
        int rpb = 256 / (WPR * 32);  // 2
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_f16_kernel<WPR><<<grid, 256, smem, stream>>>(W, x, y, bias, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---- Bias add kernel (replaces the ones-vector GEMM approach) ----
template<typename T>
__global__ void add_bias_kernel(T *Y, const T *bias, int64_t M, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (tid >= total) return;
    int64_t j = tid % N;
    float y_val = to_float(Y[tid]);
    float b_val = to_float(bias[j]);
    Y[tid] = from_float<T>(y_val + b_val);
}

// ---- FP32→FP16 conversion kernel (for mixed-precision linear) ----
__global__ void convert_f32_to_f16_kernel(__half *out, const float *in, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = __float2half(in[tid]);
}

// ---- FP16→FP32 conversion kernel (for dequant path with FP16 activations) ----
__global__ void convert_f16_to_f32_kernel(float *out, const __half *in, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = __half2float(in[tid]);
}

// ---- FP16 bias add to FP32 output ----
__global__ void add_bias_f16_to_f32_kernel(float *Y, const __half *bias, int64_t M, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (tid >= total) return;
    int64_t j = tid % N;
    Y[tid] += __half2float(bias[j]);
}

// Lazy-initialized thread-local cuBLAS handle
static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    return handle;
}

namespace llaisys::ops::nvidia {

// Y = X * W^T + bias
// Uses cublasGemmEx to support F32/F16/BF16 with F32 compute.
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
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
        static thread_local __half *in_f16_buf = nullptr;
        static thread_local int64_t in_f16_cap = 0;
        if (in_dtype == LLAISYS_DTYPE_F32) {
            int64_t in_elems = M * K;
            if (in_elems > in_f16_cap) {
                if (in_f16_buf) cudaFree(in_f16_buf);
                cudaMalloc(&in_f16_buf, in_elems * sizeof(__half));
                in_f16_cap = in_elems;
            }
            int thr = 256, blk = ((int)in_elems + thr - 1) / thr;
            convert_f32_to_f16_kernel<<<blk, thr>>>(in_f16_buf, (const float*)in->data(), in_elems);
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
            int64_t total = M * N;
            int thr = 256, blk = ((int)total + thr - 1) / thr;
            if (bias->dtype() == LLAISYS_DTYPE_F16) {
                add_bias_f16_to_f32_kernel<<<blk, thr>>>(
                    (float*)out->data(), (const __half*)bias->data(), M, N);
            } else {
                add_bias_kernel<float><<<blk, thr>>>(
                    (float*)out->data(), (const float*)bias->data(), M, N);
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
        static thread_local float *in_f32_buf = nullptr;
        static thread_local int64_t in_f32_cap = 0;
        int64_t in_elems = M * K;
        if (in_elems > in_f32_cap) {
            if (in_f32_buf) cudaFree(in_f32_buf);
            cudaMalloc(&in_f32_buf, in_elems * sizeof(float));
            in_f32_cap = in_elems;
        }
        int thr = 256, blk = ((int)in_elems + thr - 1) / thr;
        convert_f16_to_f32_kernel<<<blk, thr>>>(in_f32_buf, (const __half*)in->data(), in_elems);

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
                int64_t total = M * N;
                thr = 256; blk = ((int)total + thr - 1) / thr;
                add_bias_kernel<float><<<blk, thr>>>(
                    (float*)out->data(), (const float*)bias->data(), M, N);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // F32 weight × F16 input → FP16 output (标准激活场景)
            // 2. Allocate FP32 output buffer
            static thread_local float *out_f32_buf = nullptr;
            static thread_local int64_t out_f32_cap = 0;
            int64_t out_elems = M * N;
            if (out_elems > out_f32_cap) {
                if (out_f32_buf) cudaFree(out_f32_buf);
                cudaMalloc(&out_f32_buf, out_elems * sizeof(float));
                out_f32_cap = out_elems;
            }

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
                int64_t total = M * N;
                thr = 256; blk = ((int)total + thr - 1) / thr;
                if (bias->dtype() == LLAISYS_DTYPE_F32) {
                    add_bias_kernel<float><<<blk, thr>>>(
                        out_f32_buf, (const float*)bias->data(), M, N);
                } else {
                    // FP16 bias → FP32 buffer
                    add_bias_f16_to_f32_kernel<<<blk, thr>>>(
                        out_f32_buf, (const __half*)bias->data(), M, N);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            // 5. Convert F32 output (with bias) → FP16
            blk = ((int)out_elems + thr - 1) / thr;
            convert_f32_to_f16_kernel<<<blk, thr>>>((__half*)out->data(), out_f32_buf, out_elems);
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

    // Note: Custom gemv_f16_kernel available for M=1 FP16, but cuBLAS is
    // faster (~4%) due to Tensor Core HMMA usage. Kept for CUDA Graph
    // compatibility if needed in the future (cuBLAS requires workspace
    // pre-allocation for graph capture via cublasSetWorkspace).

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
        int64_t total = M * N;
        int thr = 256, blk = ((int)total + thr - 1) / thr;
        switch (w_dtype) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel<float><<<blk, thr>>>(
                (float*)out->data(), (const float*)bias->data(), M, N);
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel<__half><<<blk, thr>>>(
                (__half*)out->data(), (const __half*)bias->data(), M, N);
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel<__nv_bfloat16><<<blk, thr>>>(
                (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
            break;
        default: break;
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace llaisys::ops::nvidia

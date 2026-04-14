#include "rms_norm_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",              \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);       \
            throw std::runtime_error(cudaGetErrorString(err));                    \
        }                                                                         \
    } while (0)

template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// RMSNorm: Y[i,j] = X[i,j] * W[j] / sqrt(mean(X[i,:]^2) + eps)
// Always accumulate in float for numerical stability.
template<typename T>
__global__ void rms_norm_kernel(
    T *Y, const T *X, const T *W,
    int64_t rows, int64_t cols, float eps,
    int64_t y_row_stride, int64_t y_col_stride,
    int64_t x_row_stride, int64_t x_col_stride,
    int64_t w_stride
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    // Step 1: sum of squares (float accumulation)
    float local_sum = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = to_float(X[row * x_row_stride + j * x_col_stride]);
        local_sum += val * val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)cols + eps);

    // Step 2: normalize and scale
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float x_val = to_float(X[row * x_row_stride + j * x_col_stride]);
        float w_val = to_float(W[j * w_stride]);
        Y[row * y_row_stride + j * y_col_stride] = from_float<T>(x_val * w_val * inv_rms);
    }
}

// Mixed precision RMS Norm: Tio for input/output, Tw for weight
// 支持 FP16 激活 + FP32 权重的场景
template<typename Tio, typename Tw>
__global__ void rms_norm_mixed_kernel(
    Tio *Y, const Tio *X, const Tw *W,
    int64_t rows, int64_t cols, float eps,
    int64_t y_row_stride, int64_t y_col_stride,
    int64_t x_row_stride, int64_t x_col_stride,
    int64_t w_stride
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = to_float(X[row * x_row_stride + j * x_col_stride]);
        local_sum += val * val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)cols + eps);

    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float x_val = to_float(X[row * x_row_stride + j * x_col_stride]);
        float w_val = to_float(W[j * w_stride]);
        Y[row * y_row_stride + j * y_col_stride] = from_float<Tio>(x_val * w_val * inv_rms);
    }
}

namespace llaisys::ops::nvidia {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    auto in_dtype = in->dtype();
    auto w_dtype  = weight->dtype();

    int64_t rows = in->shape()[0];
    int64_t cols = in->shape()[1];

    int threads = 1;
    while (threads < cols && threads < 1024) threads <<= 1;
    size_t smem_size = threads * sizeof(float);

    auto ys0 = out->strides()[0], ys1 = out->strides()[1];
    auto xs0 = in->strides()[0], xs1 = in->strides()[1];
    auto ws0 = weight->strides()[0];

    // Mixed precision: FP16 input/output + FP32 weight
    if (in_dtype == LLAISYS_DTYPE_F16 && w_dtype == LLAISYS_DTYPE_F32) {
        rms_norm_mixed_kernel<__half, float><<<(int)rows, threads, smem_size>>>(
            (__half*)out->data(), (const __half*)in->data(), (const float*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (in_dtype == LLAISYS_DTYPE_BF16 && w_dtype == LLAISYS_DTYPE_F32) {
        rms_norm_mixed_kernel<__nv_bfloat16, float><<<(int)rows, threads, smem_size>>>(
            (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const float*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Same-type dispatch
    switch (in_dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<(int)rows, threads, smem_size>>>(
            (float*)out->data(), (const float*)in->data(), (const float*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<(int)rows, threads, smem_size>>>(
            (__half*)out->data(), (const __half*)in->data(), (const __half*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__nv_bfloat16><<<(int)rows, threads, smem_size>>>(
            (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const __nv_bfloat16*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        break;
    default:
        throw std::runtime_error("NVIDIA rms_norm: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia

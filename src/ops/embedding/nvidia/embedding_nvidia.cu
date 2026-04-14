#include "embedding_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
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

// Embedding is a table lookup (pure copy), no float conversion needed.
template<typename T>
__global__ void embedding_kernel(
    T *out,
    const int64_t *index,
    const T *weight,
    int64_t num_rows,
    int64_t embed_dim,
    int64_t w_row_stride, int64_t w_col_stride,
    int64_t o_row_stride, int64_t o_col_stride,
    int64_t i_stride
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_rows * embed_dim;
    if (tid >= total) return;

    int64_t i = tid / embed_dim;
    int64_t j = tid % embed_dim;

    int64_t token_id = index[i * i_stride];
    out[i * o_row_stride + j * o_col_stride] = weight[token_id * w_row_stride + j * w_col_stride];
}

// Mixed precision: FP16 weight → FP32 output
__global__ void embedding_f16_to_f32_kernel(
    float *out,
    const int64_t *index,
    const __half *weight,
    int64_t num_rows,
    int64_t embed_dim,
    int64_t w_row_stride, int64_t w_col_stride,
    int64_t o_row_stride, int64_t o_col_stride,
    int64_t i_stride
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_rows * embed_dim;
    if (tid >= total) return;

    int64_t i = tid / embed_dim;
    int64_t j = tid % embed_dim;

    int64_t token_id = index[i * i_stride];
    out[i * o_row_stride + j * o_col_stride] = __half2float(weight[token_id * w_row_stride + j * w_col_stride]);
}

// Mixed precision: BF16 weight → FP32 output
__global__ void embedding_bf16_to_f32_kernel(
    float *out,
    const int64_t *index,
    const __nv_bfloat16 *weight,
    int64_t num_rows,
    int64_t embed_dim,
    int64_t w_row_stride, int64_t w_col_stride,
    int64_t o_row_stride, int64_t o_col_stride,
    int64_t i_stride
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_rows * embed_dim;
    if (tid >= total) return;

    int64_t i = tid / embed_dim;
    int64_t j = tid % embed_dim;

    int64_t token_id = index[i * i_stride];
    out[i * o_row_stride + j * o_col_stride] = __bfloat162float(weight[token_id * w_row_stride + j * w_col_stride]);
}

// Mixed precision: FP32 weight → FP16 output
__global__ void embedding_f32_to_f16_kernel(
    __half *out,
    const int64_t *index,
    const float *weight,
    int64_t num_rows,
    int64_t embed_dim,
    int64_t w_row_stride, int64_t w_col_stride,
    int64_t o_row_stride, int64_t o_col_stride,
    int64_t i_stride
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_rows * embed_dim;
    if (tid >= total) return;

    int64_t i = tid / embed_dim;
    int64_t j = tid % embed_dim;

    int64_t token_id = index[i * i_stride];
    out[i * o_row_stride + j * o_col_stride] = __float2half(weight[token_id * w_row_stride + j * w_col_stride]);
}

namespace llaisys::ops::nvidia {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    auto dtype = weight->dtype();

    int64_t num_rows  = index->shape()[0];
    int64_t embed_dim = weight->shape()[1];

    int64_t w_row_stride = weight->strides()[0];
    int64_t w_col_stride = weight->strides()[1];
    int64_t o_row_stride = out->strides()[0];
    int64_t o_col_stride = out->strides()[1];
    int64_t i_stride     = index->strides()[0];

    int64_t total = num_rows * embed_dim;
    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        if (out->dtype() == LLAISYS_DTYPE_F16) {
            // Mixed precision: FP32 weight → FP16 output
            embedding_f32_to_f16_kernel<<<blocks, threads>>>(
                (__half *)out->data(), (const int64_t *)index->data(), (const float *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        } else {
            embedding_kernel<float><<<blocks, threads>>>(
                (float *)out->data(), (const int64_t *)index->data(), (const float *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        }
        break;
    case LLAISYS_DTYPE_F16:
        if (out->dtype() == LLAISYS_DTYPE_F32) {
            // Mixed precision: FP16 weight → FP32 output
            embedding_f16_to_f32_kernel<<<blocks, threads>>>(
                (float *)out->data(), (const int64_t *)index->data(), (const __half *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        } else {
            embedding_kernel<__half><<<blocks, threads>>>(
                (__half *)out->data(), (const int64_t *)index->data(), (const __half *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        }
        break;
    case LLAISYS_DTYPE_BF16:
        if (out->dtype() == LLAISYS_DTYPE_F32) {
            // Mixed precision: BF16 weight → FP32 output
            embedding_bf16_to_f32_kernel<<<blocks, threads>>>(
                (float *)out->data(), (const int64_t *)index->data(), (const __nv_bfloat16 *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        } else {
            embedding_kernel<__nv_bfloat16><<<blocks, threads>>>(
                (__nv_bfloat16 *)out->data(), (const int64_t *)index->data(), (const __nv_bfloat16 *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        }
        break;
    default:
        throw std::runtime_error("NVIDIA embedding: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia

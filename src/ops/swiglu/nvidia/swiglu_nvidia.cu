#include "swiglu_nvidia.cuh"

#ifdef LLAISYS_USE_LLMOPS
#include "adapter.hpp"
#endif

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

template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ void swiglu_kernel(
    T *out, const T *gate, const T *up,
    int64_t seq_len, int64_t dim,
    int64_t out_s0, int64_t out_s1,
    int64_t gate_s0, int64_t gate_s1,
    int64_t up_s0, int64_t up_s1
) {
    int64_t total = seq_len * dim;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t i = tid / dim;
    int64_t j = tid % dim;

    float val_gate = to_float(gate[i * gate_s0 + j * gate_s1]);
    float val_up   = to_float(up[i * up_s0 + j * up_s1]);

    float silu_gate = val_gate / (1.0f + expf(-val_gate));
    out[i * out_s0 + j * out_s1] = from_float<T>(val_up * silu_gate);
}

namespace llaisys::ops::nvidia {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
#ifdef LLAISYS_USE_LLMOPS
    if (llmops_integration::try_swiglu(out, gate, up)) return;
#endif
    auto dtype = gate->dtype();

    int64_t seq_len = gate->shape()[0];
    int64_t dim     = gate->shape()[1];
    int64_t total = seq_len * dim;
    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;

    auto os0 = out->strides()[0], os1 = out->strides()[1];
    auto gs0 = gate->strides()[0], gs1 = gate->strides()[1];
    auto us0 = up->strides()[0], us1 = up->strides()[1];

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<blocks, threads>>>(
            (float *)out->data(), (const float *)gate->data(), (const float *)up->data(),
            seq_len, dim, os0, os1, gs0, gs1, us0, us1);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<blocks, threads>>>(
            (__half *)out->data(), (const __half *)gate->data(), (const __half *)up->data(),
            seq_len, dim, os0, os1, gs0, gs1, us0, us1);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<blocks, threads>>>(
            (__nv_bfloat16 *)out->data(), (const __nv_bfloat16 *)gate->data(), (const __nv_bfloat16 *)up->data(),
            seq_len, dim, os0, os1, gs0, gs1, us0, us1);
        break;
    default:
        throw std::runtime_error("NVIDIA swiglu: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia

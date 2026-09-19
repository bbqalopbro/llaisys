#pragma once
#include "../tensor/tensor.hpp"
#include "dispatch.hpp"
#include <llmops/ops/extended.hpp>

#if defined(ENABLE_NVIDIA_API) && !defined(__CUDACC__)
extern "C" int cudaGetDevice(int *);
#endif
namespace llaisys::ops {
inline llmopsDtype kernel_dtype(llaisysDataType_t d) {
    switch(d) {
    case LLAISYS_DTYPE_F32: return LLMOPS_F32;
    case LLAISYS_DTYPE_F16: return LLMOPS_F16;
    case LLAISYS_DTYPE_BF16: return LLMOPS_BF16;
    case LLAISYS_DTYPE_I8: return LLMOPS_I8;
    case LLAISYS_DTYPE_U8: return LLMOPS_U8;
    case LLAISYS_DTYPE_I32: return LLMOPS_I32;
    case LLAISYS_DTYPE_I64: return LLMOPS_I64;
    default: throw std::runtime_error("llmops: unsupported storage dtype");
    }
}
inline llmops::TensorView view(tensor_t t) {
    if (!t) return nullptr;
    return {t->data(), kernel_dtype(t->dtype()), t->shape(), t->strides()};
}
inline void check_devices(std::initializer_list<tensor_t> tensors) {
    tensor_t first;
    for(auto t:tensors) if(t) {
        if(!first) first=t;
        if(t->deviceType()!=first->deviceType() || t->deviceId()!=first->deviceId())
            throw std::runtime_error("operator tensors must share device and device id");
    }
#ifdef ENABLE_NVIDIA_API
    if(first && first->deviceType()==LLAISYS_DEVICE_NVIDIA) {
        int current=-1;
        if(cudaGetDevice(&current)!=0 || current!=first->deviceId()) throw std::runtime_error("operator device must be current");
    }
#endif
}
// CUDA per-thread default stream, matching framework runtime and Graph runner.
inline void *operator_stream() { return reinterpret_cast<void *>(2); }
}

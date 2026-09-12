#include "quantized_linear.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/quantized_linear.cuh"
#endif

#include <stdexcept>

namespace llaisys::ops {

void deepseek_v4_quantized_linear_reference(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode,
    llaisysDataType_t dtype, llaisysDeviceType_t device_type) {
    if (!output || !input || !weight || !scale || rows <= 0 ||
        out_features <= 0 || in_features <= 0 ||
        (quant_mode != 0 && quant_mode != 1)) {
        throw std::invalid_argument("invalid DeepSeek-V4 quantized linear arguments");
    }
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA && dtype == LLAISYS_DTYPE_BF16) {
        return nvidia::deepseek_v4_quantized_linear_reference(
            output, input, weight, scale, rows, out_features, in_features,
            quant_mode);
    }
#endif
    throw std::invalid_argument(
        "DeepSeek-V4 quantized linear supports NVIDIA BF16 only");
}

void deepseek_v4_quantized_linear_cublas(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode,
    llaisysDataType_t dtype, llaisysDeviceType_t device_type) {
    if (!output || !input || !weight || !scale || rows <= 0 ||
        out_features <= 0 || in_features <= 0 ||
        (quant_mode != 0 && quant_mode != 1)) {
        throw std::invalid_argument("invalid DeepSeek-V4 cuBLAS linear arguments");
    }
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA && dtype == LLAISYS_DTYPE_BF16) {
        return nvidia::deepseek_v4_quantized_linear_cublas(
            output, input, weight, scale, rows, out_features, in_features,
            quant_mode);
    }
#endif
    throw std::invalid_argument(
        "DeepSeek-V4 cuBLAS linear supports NVIDIA BF16 only");
}

} // namespace llaisys::ops

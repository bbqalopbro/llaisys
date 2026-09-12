#pragma once

#include "llaisys.h"

namespace llaisys::ops {

void deepseek_v4_quantized_linear_reference(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode,
    llaisysDataType_t dtype, llaisysDeviceType_t device_type);

void deepseek_v4_quantized_linear_cublas(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode,
    llaisysDataType_t dtype, llaisysDeviceType_t device_type);

} // namespace llaisys::ops

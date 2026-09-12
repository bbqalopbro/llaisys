#pragma once

namespace llaisys::ops::nvidia {

void deepseek_v4_quantized_linear_reference(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode);

void deepseek_v4_quantized_linear_cublas(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode);

} // namespace llaisys::ops::nvidia

#pragma once

#include "../../tensor/tensor.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
void linear_add(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
               tensor_t residual);
void linear_int4(tensor_t out, tensor_t in, tensor_t weight, tensor_t scale,
                tensor_t bias, int group_size, tensor_t residual = nullptr);
void linear_fp8(tensor_t out, tensor_t in, tensor_t input_scale, tensor_t weight,
                tensor_t weight_scale, tensor_t bias, tensor_t residual);
void quantize_fp8(tensor_t quantized, tensor_t scales, tensor_t in, tensor_t weight,
                  tensor_t up, tensor_t floating_out, float eps);
}

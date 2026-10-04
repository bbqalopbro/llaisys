#pragma once

#include "../../tensor/tensor.hpp"
#include <cmath>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale, bool stable);
}

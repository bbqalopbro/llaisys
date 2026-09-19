#pragma once

#include "../../tensor/tensor.hpp"
#include <limits>

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
void argmax_rows(tensor_t indices, tensor_t maxima, tensor_t values);
}

#include "dequantize_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void dequantize(tensor_t out, tensor_t weight, tensor_t scale) { llmops::metax::dequantize(view(out), view(weight), view(scale), nullptr); }
void dequantize_int4(tensor_t out, tensor_t weight, tensor_t scale, int group_size) { llmops::metax::dequantize_int4(view(out), view(weight), view(scale), group_size, nullptr); }
}

#include "argmax_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) { llmops::metax::argmax(view(max_idx), view(max_val), view(vals), nullptr); }
}

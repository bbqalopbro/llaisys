#include "rope_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) { llmops::metax::rope(view(out), view(in), view(pos_ids), theta, nullptr); }
}

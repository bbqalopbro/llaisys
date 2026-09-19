#include "swiglu_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) { llmops::metax::swiglu(view(out), view(gate), view(up), nullptr); }
}

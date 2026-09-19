#include "rms_norm_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) { llmops::metax::rms_norm(view(out), view(in), view(weight), eps, nullptr); }
}

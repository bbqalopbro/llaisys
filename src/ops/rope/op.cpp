#include "op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_METAX_API
#include "metax/rope_metax.hpp"
#endif
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    check_devices({out, in, pos_ids});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::rope(view(out), view(in), view(pos_ids), theta); record_dispatch("rope", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::rope(view(out), view(in), view(pos_ids), theta, operator_stream()); record_dispatch("rope", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) return metax::rope(out, in, pos_ids, theta);
#endif
    throw std::runtime_error("rope: device backend not compiled or unsupported");
}
}

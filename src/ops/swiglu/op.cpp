#include "op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_METAX_API
#include "metax/swiglu_metax.hpp"
#endif
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    check_devices({out, gate, up});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::swiglu(view(out), view(gate), view(up)); record_dispatch("swiglu", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::swiglu(view(out), view(gate), view(up), operator_stream()); record_dispatch("swiglu", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) return metax::swiglu(out, gate, up);
#endif
    throw std::runtime_error("swiglu: device backend not compiled or unsupported");
}
}

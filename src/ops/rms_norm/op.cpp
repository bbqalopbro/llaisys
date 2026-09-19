#include "op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_METAX_API
#include "metax/rms_norm_metax.hpp"
#endif
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    check_devices({out, in, weight});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::rms_norm(view(out), view(in), view(weight), eps); record_dispatch("rms_norm", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::rms_norm(view(out), view(in), view(weight), eps, operator_stream()); record_dispatch("rms_norm", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) return metax::rms_norm(out, in, weight, eps);
#endif
    throw std::runtime_error("rms_norm: device backend not compiled or unsupported");
}
}

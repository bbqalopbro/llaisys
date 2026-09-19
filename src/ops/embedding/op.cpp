#include "op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_METAX_API
#include "metax/embedding_metax.hpp"
#endif
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    check_devices({out, index, weight});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::embedding(view(out), view(index), view(weight)); record_dispatch("embedding", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::embedding(view(out), view(index), view(weight), operator_stream()); record_dispatch("embedding", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) return metax::embedding(out, index, weight);
#endif
    throw std::runtime_error("embedding: device backend not compiled or unsupported");
}
}

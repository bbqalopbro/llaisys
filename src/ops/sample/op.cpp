#include "op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_METAX_API
#include "metax/sample_metax.hpp"
#endif
namespace llaisys::ops {
void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k,
            float top_p, uint64_t seed) {
    check_devices({out_idx, logits});
    if (out_idx->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::sample(view(out_idx), view(logits), temperature, top_k, top_p, seed); record_dispatch("sample", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out_idx->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::sample(view(out_idx), view(logits), temperature, top_k, top_p, seed, operator_stream()); record_dispatch("sample", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (out_idx->deviceType() == LLAISYS_DEVICE_METAX) return metax::sample(out_idx, logits, temperature, top_k, top_p, seed);
#endif
    throw std::runtime_error("sample: device backend not compiled or unsupported");
}
}

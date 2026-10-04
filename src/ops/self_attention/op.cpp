#include "op.hpp"
#include "../llmops.hpp"
#include <cstdlib>
#include <string>
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/self_attention_metax.hpp"
#endif
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    self_attention(attn_val,q,k,v,scale,false);
}
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale, bool stable) {
    check_devices({attn_val, q, k, v});
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::self_attention(view(attn_val), view(q), view(k), view(v), scale); record_dispatch("self_attention", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        const char *mode=std::getenv("LLAISYS_ATTENTION");
        if (!mode || std::string(mode)=="native")
            { llmops::cuda::self_attention(view(attn_val),view(q),view(k),view(v),scale,operator_stream(),stable); record_dispatch("self_attention", "llmops.cuda"); return; }
        if (std::string(mode)=="cublas") return nvidia::self_attention(attn_val,q,k,v,scale);
        throw std::runtime_error("contiguous Attention supports native (default) or explicit cublas");
    }
#endif
#ifdef ENABLE_METAX_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_METAX) return metax::self_attention(attn_val, q, k, v, scale);
#endif
    throw std::runtime_error("self_attention: device backend not compiled or unsupported");
}
}

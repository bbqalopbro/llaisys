#pragma once

#include "llaisys/runtime.h"

namespace llaisys::backends {

struct BackendCapabilities {
    bool available = false;
    bool paged_attention = false;
    bool mla_attention = false;
    bool cuda_graph = false;
    bool collective_communication = false;
};

inline BackendCapabilities capabilities(llaisysDeviceType_t device) {
    BackendCapabilities result;
    if (device == LLAISYS_DEVICE_CPU) {
        result.available = true;
        result.paged_attention = true;
        return result;
    }
#ifdef ENABLE_NVIDIA_API
    if (device == LLAISYS_DEVICE_NVIDIA) {
        result.available = true;
        result.paged_attention = true;
        result.cuda_graph = true;
#ifdef ENABLE_DIST_NCCL
        result.collective_communication = true;
#endif
        return result;
    }
#endif
#ifdef ENABLE_METAX_API
    if (device == LLAISYS_DEVICE_METAX) {
        result.available = true;
        result.paged_attention = true;
        return result;
    }
#endif
    return result;
}

} // namespace llaisys::backends

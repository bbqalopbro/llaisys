#pragma once

#include "llaisys/runtime.h"

#include <cstddef>

namespace llaisys::models {

enum class ModelFamily { Qwen2, DeepSeek };
enum class AttentionFamily { StandardKV, MLA };

struct ModelCapabilities {
    ModelFamily model_family;
    AttentionFamily attention_family;
    bool supports_chunked_prefill = false;
    bool supports_paged_cache = false;
    bool supports_tensor_parallel = false;
    bool supports_expert_parallel = false;
};

// Minimal runtime-facing model contract. Scheduling remains in Python; model
// implementations expose capabilities and placement without queue policy.
class ModelRunner {
public:
    virtual ~ModelRunner() = default;
    virtual const char *name() const = 0;
    virtual llaisysDeviceType_t deviceType() const = 0;
    virtual int deviceId() const = 0;
    virtual ModelCapabilities capabilities() const = 0;
};

} // namespace llaisys::models

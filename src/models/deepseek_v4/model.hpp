#pragma once
#include "checkpoint.hpp"

namespace llaisys::models::deepseek_v4 {
using KernelBindings = std::map<std::string, std::shared_ptr<backends::native::Kernel>>;
struct ModelBackends {
    // Shape-specialized quant/GEMM bindings and per-layer routing options are
    // explicitly prepared before model loading. Backends may differ per op.
    std::vector<KernelBindings> layers;
    KernelBindings global;
};

class Model;
class ModelState {
public:
    ModelState(core::Runtime &runtime, const ModelConfig &config);
    std::vector<std::unique_ptr<BlockState>> layers;
    int64_t position() const { return position_; }
    bool failed() const { return failed_; }
private:
    friend class Model;
    const Model *owner_ = nullptr;
    int64_t position_ = 0;
    bool failed_ = true;
};

class ModelWorkspace {
public:
    using Tensor = backends::native::Tensor;
    ModelWorkspace(core::Runtime &runtime, const ModelConfig &config, int64_t position, int64_t tokens, bool emit_logits = true);
    const int64_t position, tokens;
    const bool emit_logits;
    Tensor hidden, normalized, head_input, logits, next_ids;
    std::vector<std::unique_ptr<BlockWorkspace>> layers;
    std::unique_ptr<HCWorkspace> head;
    bool failed() const { return failed_; }
private:
    friend class Model;
    bool failed_ = false;
};
struct ModelOutput {
    // Borrowed workspace results. Intermediate prefill explicitly returns null.
    backends::native::Tensor *logits, *next_ids;
};

// Full MP1 base-model execution, not a scheduler. No Python callbacks occur in
// load/reset/forward; logits and greedy IDs remain on device until caller reads.
// Current request storage is the explicit continuous/ring reference backend.
class Model {
public:
    using Tensor = backends::native::Tensor;
    Model(core::Runtime &runtime, ModelConfig config, Checkpoint &checkpoint, ModelBackends backends);
    const ModelConfig &config() const { return config_; }
    void reset(ModelState &state);
    ModelOutput forward(Tensor &token_ids, ModelState &state, ModelWorkspace &workspace);
    const Tensor &frequencies(bool compressed) const { return *frequencies_.at(compressed); }
    uint64_t calls() const { return calls_; }
    uint64_t failures() const { return failures_; }
private:
    core::Runtime &runtime_;
    const ModelConfig config_;
    ModelBackends backends_;
    std::shared_ptr<Tensor> embedding_;
    Tensor norm_, head_weight_;
    std::map<bool, std::shared_ptr<Tensor>> frequencies_;
    std::vector<std::unique_ptr<Block>> layers_;
    std::unique_ptr<HyperConnection> head_;
    uint64_t calls_ = 0, failures_ = 0;
};
} // namespace llaisys::models::deepseek_v4

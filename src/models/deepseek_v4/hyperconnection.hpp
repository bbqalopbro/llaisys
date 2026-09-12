#pragma once

#include "src/backends/native/kernel.hpp"

namespace llaisys::models::deepseek_v4 {

struct HCConfig { int64_t hidden, copies; bool head; };
struct HCWeights { std::shared_ptr<backends::native::Tensor> projection, scale, base; };
struct HCBackends {
    using Kernel = backends::native::Kernel;
    // Scalar semantics (norm epsilon, Sinkhorn iterations/epsilon, head epsilon)
    // are explicitly bound by the caller, not inferred from weight names.
    std::shared_ptr<Kernel> cast, dense, inverse_rms, row_multiply, pre_reduce;
    std::shared_ptr<Kernel> split, post_mix, head_weights;
};

class HyperConnection;
class HCWorkspace {
public:
    using Tensor = backends::native::Tensor;
    HCWorkspace(core::Runtime &runtime, HCConfig config, int64_t tokens);
    const HCConfig config;
    const int64_t tokens;
    Tensor input_fp32, inverse_rms, mixes, pre_weights, post_weights, combination, reduced, expanded;
    bool failed() const { return failed_; }
    bool awaitingPost() const { return ready_; }
private:
    friend class HyperConnection;
    // Retain the original BF16 residual through asynchronous post execution.
    // Caller must not mutate it between pre() and post(). This is not a copy.
    std::shared_ptr<Tensor> residual_;
    const HyperConnection *owner_ = nullptr;
    bool failed_ = false, ready_ = false;
};

// Native model-owned HC composition. FP32 projection/normalization/mixing,
// BF16 hidden boundaries; not an FP8 Linear and not a simple residual add.
// Outputs are borrowed from workspace and valid until its next execution.
class HyperConnection {
public:
    using Tensor = backends::native::Tensor;
    HyperConnection(HCConfig config, HCWeights weights, HCBackends backends);
    Tensor &pre(Tensor &hidden, HCWorkspace &workspace);
    Tensor &post(Tensor &branch, HCWorkspace &workspace);
    Tensor &head(Tensor &hidden, HCWorkspace &workspace);
    uint64_t calls() const { return calls_; }
    uint64_t failures() const { return failures_; }
private:
    void validate(Tensor &input, HCWorkspace &workspace, bool post) const;
    void project(Tensor &input, HCWorkspace &workspace);
    HCConfig config_;
    HCWeights weights_;
    HCBackends backends_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::models::deepseek_v4

#pragma once

#include "linear.hpp"
#include <map>

namespace llaisys::models::deepseek_v4 {

struct ExpertWorkspace {
    ExpertWorkspace(core::Runtime &runtime, int64_t rows, int64_t hidden, int64_t intermediate);
    LinearWorkspace input_quant, intermediate_quant;
    backends::native::Tensor gate, up, activation;
};

// Model composition only: each projection and activation is independently
// replaceable. Routed weights multiply FP32 SwiGLU BEFORE BF16/down projection.
class Expert {
public:
    using Tensor = backends::native::Tensor;
    using Kernel = backends::native::Kernel;
    Expert(std::shared_ptr<QuantizedLinear> gate, std::shared_ptr<QuantizedLinear> up,
           std::shared_ptr<QuantizedLinear> down, std::shared_ptr<Kernel> activation);
    void forward(Tensor &input, Tensor &output, ExpertWorkspace &workspace, Tensor *routing_weights = nullptr);
    int64_t hiddenDim() const { return gate_->inputDim(); }
    int64_t intermediateDim() const { return gate_->outputDim(); }
    core::Runtime &runtime() const { return activation_->runtime(); }
private:
    std::shared_ptr<QuantizedLinear> gate_, up_, down_;
    std::shared_ptr<Kernel> activation_;
};

struct MoeConfig {
    int64_t hidden, intermediate, experts, top_k, vocabulary;
    bool hash_routing;
};

struct MoeBackends {
    using Kernel = backends::native::Kernel;
    // All backends explicit; router score function/route scale and activation
    // clamp are bound when constructing their kernel, not guessed here.
    std::shared_ptr<Kernel> cast, dense, gather, router, dispatch, combine, finalize;
};

// Per-execution/request storage, not serving policy. Expert scratch is pooled
// by row count and survives enqueue; it can be reused by later calls of the
// same shape. Reference execution downloads offsets, then loops over experts.
class MoeWorkspace {
public:
    using Tensor = backends::native::Tensor;
    MoeWorkspace(core::Runtime &runtime, MoeConfig config, int64_t rows);
    const MoeConfig config;
    const int64_t rows;
    Tensor input_fp32, logits, hash_ids, route_weights, route_ids;
    Tensor packed_hidden, packed_weights, packed_rows, offsets, packed_output;
    Tensor routed_output, shared_output;
    bool failed() const { return failed_; }
    size_t scratchShapes() const { return scratch_.size(); }
    uint64_t offsetDownloads() const { return offset_downloads_; }
    uint64_t expertExecutions() const { return expert_executions_; }
private:
    friend class MoE;
    ExpertWorkspace &scratch(int64_t count);
    std::map<int64_t, std::unique_ptr<ExpertWorkspace>> scratch_;
    std::vector<int64_t> host_offsets_;
    bool failed_ = false;
    uint64_t offset_downloads_ = 0, expert_executions_ = 0;
};

// Single-rank COMPLETE MoE layer. No Python, ATen or TileLang types occur in
// this model interface/implementation. Not an EP dispatcher or fused GEMM.
class MoE {
public:
    using Tensor = backends::native::Tensor;
    MoE(MoeConfig config, std::shared_ptr<Tensor> gate_weight, std::shared_ptr<Tensor> selector,
        std::vector<std::shared_ptr<Expert>> experts, std::shared_ptr<Expert> shared, MoeBackends backends);
    void forward(Tensor &input, Tensor &token_ids, Tensor &output, MoeWorkspace &workspace);
    uint64_t calls() const { return calls_; }
    uint64_t failures() const { return failures_; }
    const MoeConfig &config() const { return config_; }
    core::Runtime &runtime() const { return gate_fp32_.runtime(); }
private:
    const MoeConfig config_;
    std::shared_ptr<Tensor> gate_weight_, selector_;
    Tensor gate_fp32_;
    std::vector<std::shared_ptr<Expert>> experts_;
    std::shared_ptr<Expert> shared_;
    MoeBackends backends_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::models::deepseek_v4

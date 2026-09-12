#pragma once

#include "src/backends/native/kernel.hpp"
#include <memory>
#include <thread>

namespace llaisys::backends::aten {

// Model scalar semantics are bound explicitly. Shape-dependent values (e.g.
// top-k <= current compressed length) come from the validated output shape.
struct StructuralOptions {
    double epsilon = 1e-6;
    double activation_limit = 0;
    double route_scale = 1;
    int64_t reduction_axis = 2;
    bool inverse_rotary = false;
    bool stable_indexer_ties = false;
    bool hash_routing = false;
    std::string score_function = "sqrtsoftplus";
};

// Explicit ATen C++ correctness backend, not a silent fallback or a fused
// kernel claim. Public interfaces expose no ATen/Torch/Python types. Temporary
// ATen allocations and reference MoE host metadata reads remain observable.
class StructuralKernel final : public native::Kernel {
public:
    StructuralKernel(core::Runtime &runtime, native::KernelIdentity identity,
                     StructuralOptions options = {});
    ~StructuralKernel() override;
    void call(const std::vector<native::Tensor *> &arguments) override;
    void callWithScalars(const std::vector<native::Tensor *> &arguments, const native::CallScalars &scalars) override;
    void close() override;
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return failures_; }
    static const char *compiledVersion();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::thread::id thread_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::backends::aten

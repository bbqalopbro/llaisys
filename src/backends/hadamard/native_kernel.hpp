#pragma once
#include "src/backends/native/kernel.hpp"
#include <thread>

namespace llaisys::backends::hadamard {

// Calls the unchanged upstream CUDA object, NOT its Python extension wrapper.
// Model use is a contiguous power-of-two BF16/FP32 transform with explicit scale.
class NativeKernel final : public native::Kernel {
public:
    NativeKernel(core::Runtime &runtime, native::KernelIdentity identity, float scale);
    ~NativeKernel() override;
    void call(const std::vector<native::Tensor *> &arguments) override;
    void close() override;
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return failures_; }
private:
    std::thread::id thread_;
    float scale_;
    bool closed_ = false;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::backends::hadamard

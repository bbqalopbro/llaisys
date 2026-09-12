#pragma once

#include "src/backends/native/kernel.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace llaisys::backends::tilelang {

using NativeTensor = native::Tensor;

// Pure C++ invocation of an exported TileLang/TVM-FFI module. Python may compile
// and export a bundle OFF the execution path; no Python callback or Python ABI
// is referenced here. This is a kernel backend, not a model executor/scheduler.
class NativeKernel final : public native::Kernel {
public:
    NativeKernel(core::Runtime &runtime, const std::string &library,
                 native::KernelIdentity identity,
                 const std::string &entry = "main");
    ~NativeKernel() override;
    NativeKernel(const NativeKernel &) = delete;
    NativeKernel &operator=(const NativeKernel &) = delete;
    void call(const std::vector<NativeTensor *> &arguments) override;
    void close() override;
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return failures_; }
    static uint64_t destructionErrors();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::thread::id thread_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::backends::tilelang

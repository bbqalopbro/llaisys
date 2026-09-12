#pragma once

#include "src/backends/native/kernel.hpp"
#include <memory>

namespace llaisys::models::deepseek_v4 {

// Explicit per-execution workspace. Allocate before a batch, retain until its
// stream completes, reuse for the same shape. forward() never allocates, copies
// to host or synchronizes between quantization and GEMM.
struct LinearWorkspace {
    LinearWorkspace(core::Runtime &runtime, int64_t rows, int64_t input_dim);
    backends::native::Tensor activation;
    backends::native::Tensor scales;
};

// Native model component, not a second scheduler or a generic model graph.
// Retains actual packed weights/scales and separately selected quant/GEMM
// implementations. The final C++ model will own these plus request workspaces.
// Weights must not be mutated while this layer is used.
class QuantizedLinear {
public:
    using Tensor = backends::native::Tensor;
    using Kernel = backends::native::Kernel;
    QuantizedLinear(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> scales,
                    std::shared_ptr<Kernel> quant, std::shared_ptr<Kernel> gemm);
    void forward(Tensor &input, Tensor &output, LinearWorkspace &workspace);
    int64_t inputDim() const { return input_dim_; }
    int64_t outputDim() const { return output_dim_; }
    bool packedFp4() const { return fp4_; }
    core::Runtime &runtime() const { return weight_->runtime(); }
    const backends::native::KernelIdentity &quantBackend() const { return quant_->identity(); }
    const backends::native::KernelIdentity &gemmBackend() const { return gemm_->identity(); }

private:
    std::shared_ptr<Tensor> weight_, scales_;
    std::shared_ptr<Kernel> quant_, gemm_;
    int64_t input_dim_, output_dim_;
    bool fp4_;
};

} // namespace llaisys::models::deepseek_v4

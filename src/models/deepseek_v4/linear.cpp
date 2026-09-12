#include "linear.hpp"

#include <stdexcept>

namespace llaisys::models::deepseek_v4 {
namespace {
bool dtypeIs(DLDataType value, uint8_t code, uint8_t bits) {
    return value.code == code && value.bits == bits && value.lanes == 1;
}
bool fp8(DLDataType dtype) {
    return dtypeIs(dtype, kDLFloat8_e4m3fn, 8) || dtypeIs(dtype, kDLFloat8_e4m3, 8);
}
void check(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
int64_t checkedDim(int64_t value) {
    check(value > 0 && value % 128 == 0, "V4 quantized input dimension must be positive and divisible by 128");
    return value;
}
}

LinearWorkspace::LinearWorkspace(core::Runtime &runtime, int64_t rows, int64_t input_dim)
    : activation(runtime, {rows, checkedDim(input_dim)}, {kDLFloat8_e4m3fn, 8, 1}),
      scales(runtime, {rows, input_dim / 128}, {kDLFloat8_e8m0fnu, 8, 1}) {}

QuantizedLinear::QuantizedLinear(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> scales,
                               std::shared_ptr<Kernel> quant, std::shared_ptr<Kernel> gemm)
    : weight_(std::move(weight)), scales_(std::move(scales)), quant_(std::move(quant)), gemm_(std::move(gemm)) {
    check(weight_ && scales_ && quant_ && gemm_, "native Linear requires weights, scales and both explicit backends");
    check(weight_->shape().size() == 2, "native Linear weight must have logical [N,K] shape");
    output_dim_ = weight_->shape()[0];
    input_dim_ = checkedDim(weight_->shape()[1]);
    fp4_ = dtypeIs(weight_->dtype(), kDLFloat4_e2m1fn, 4);
    check(fp4_ || fp8(weight_->dtype()), "native quantized Linear requires E2M1 FP4 or E4M3 FP8 weights");
    const std::vector<int64_t> expected_scales = fp4_
        ? std::vector<int64_t>{output_dim_, input_dim_ / 32}
        : std::vector<int64_t>{(output_dim_ + 127) / 128, input_dim_ / 128};
    check(scales_->shape() == expected_scales && dtypeIs(scales_->dtype(), kDLFloat8_e8m0fnu, 8),
          "native Linear weight scale layout/dtype mismatch");
    auto *runtime = &weight_->runtime();
    check(&scales_->runtime() == runtime && &quant_->runtime() == runtime && &gemm_->runtime() == runtime,
          "native Linear cannot mix runtimes/devices");
    check(quant_->identity().operation == "act_quant" && quant_->identity().contract_revision == 1
          && gemm_->identity().operation == (fp4_ ? "fp4_gemm" : "fp8_gemm")
          && gemm_->identity().contract_revision == 1,
          "native Linear backend does not implement the selected V4 contract");
    // Validate owning thread/device before accepting lifetime-bearing weights.
    (void)weight_->view();
    (void)scales_->view();
}

void QuantizedLinear::forward(Tensor &input, Tensor &output, LinearWorkspace &workspace) {
    check(input.shape().size() == 2 && input.shape()[1] == input_dim_
          && dtypeIs(input.dtype(), kDLBfloat, 16), "native Linear input must be BF16 [M,K]");
    const int64_t rows = input.shape()[0];
    check(output.shape() == std::vector<int64_t>{rows, output_dim_}
          && dtypeIs(output.dtype(), kDLBfloat, 16), "native Linear output must be BF16 [M,N]");
    check(workspace.activation.shape() == input.shape() && fp8(workspace.activation.dtype())
          && workspace.scales.shape() == std::vector<int64_t>{rows, input_dim_ / 128}
          && dtypeIs(workspace.scales.dtype(), kDLFloat8_e8m0fnu, 8), "native Linear workspace shape/dtype mismatch");
    check(!input.overlaps(output), "native Linear output cannot alias its input");
    // Validate every argument before enqueueing the first operation. A later
    // backend execution error still invalidates the caller's request state.
    for (Tensor *tensor : {&input, &output, &workspace.activation, &workspace.scales, weight_.get(), scales_.get()}) {
        check(&tensor->runtime() == &weight_->runtime(), "native Linear argument belongs to another runtime");
        check(tensor->isContiguous(), "native quantized Linear requires contiguous arguments");
        (void)tensor->view();
    }
    if (!rows) return;
    quant_->call({&input, &workspace.activation, &workspace.scales});
    gemm_->call({&workspace.activation, weight_.get(), &output, &workspace.scales, scales_.get()});
}

} // namespace llaisys::models::deepseek_v4

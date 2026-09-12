// Vendor headers precede the legacy __C macro from llaisys.h.
#include <cuda_runtime_api.h>
#include <c10/util/BFloat16.h>
#include <cstdint>
#include <fast_hadamard_transform.h>

#include "native_kernel.hpp"
#include <cmath>
#include <cstdio>

// Symbol instantiated in upstream fast_hadamard_transform_cuda.cu (Tri Dao,
// BSD-3-Clause). The external source/object and license remain unmodified.
template <typename T>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream);

namespace llaisys::backends::hadamard {
namespace {
void check(bool value, const char *message) {
    if (!value) throw std::invalid_argument(message);
}
}

NativeKernel::NativeKernel(core::Runtime &runtime, native::KernelIdentity identity, float scale)
    : native::Kernel(runtime, std::move(identity)), thread_(std::this_thread::get_id()), scale_(scale) {
    check(runtime.deviceType() == LLAISYS_DEVICE_NVIDIA && runtime.isActive(), "native Hadamard requires active NVIDIA runtime");
    check(identity_.backend == "cuda-hadamard" && identity_.operation == "hadamard" && identity_.contract_revision == 1,
          "Hadamard backend/operation/contract identity mismatch");
    check(std::isfinite(scale_), "Hadamard scale must be finite");
}

void NativeKernel::call(const std::vector<native::Tensor *> &arguments) {
    if (closed_ || std::this_thread::get_id() != thread_ || !runtime_.isActive())
        throw std::runtime_error("Hadamard kernel is closed or outside its owning runtime/thread");
    ++calls_;
    try {
        check(arguments.size() == 2 && arguments[0] && arguments[1], "Hadamard requires input and output");
        auto &input = *arguments[0]; auto &output = *arguments[1];
        check(&input.runtime() == &runtime_ && &output.runtime() == &runtime_, "Hadamard received a foreign tensor");
        const auto dtype = input.dtype(), out_dtype = output.dtype();
        const bool bf16 = dtype.code == kDLBfloat && dtype.bits == 16;
        check(dtype.lanes == 1 && (bf16 || (dtype.code == kDLFloat && dtype.bits == 32))
              && dtype.code == out_dtype.code && dtype.bits == out_dtype.bits && dtype.lanes == out_dtype.lanes,
              "Hadamard requires matching BF16/FP32 tensors");
        check(input.shape() == output.shape() && input.isContiguous() && output.isContiguous()
              && !input.overlaps(output), "Hadamard shape/layout/alias mismatch");
        const int64_t dim = input.shape().back();
        check(dim >= 8 && dim <= 32768 && !(dim & (dim - 1)), "native Hadamard supports power-of-two dimensions 8..32768");
        const uint64_t rows = input.bytes() / (dtype.bits / 8) / dim;
        check(rows <= INT32_MAX, "Hadamard batch exceeds upstream integer range");
        auto x = input.view(), y = output.view();
        if (!rows) return;
        HadamardParamsBase params{};
        params.batch = static_cast<int>(rows); params.dim = static_cast<int>(dim);
        for (int64_t n = dim; n > 1; n >>= 1) ++params.log_N;
        params.x_batch_stride = params.out_batch_stride = dim;
        params.x_ptr = static_cast<char *>(x.data) + x.byte_offset;
        params.out_ptr = static_cast<char *>(y.data) + y.byte_offset;
        params.scale = scale_;
        auto stream = reinterpret_cast<cudaStream_t>(runtime_.stream());
        if (bf16) fast_hadamard_transform_cuda<c10::BFloat16>(params, stream);
        else fast_hadamard_transform_cuda<float>(params, stream);
    } catch (...) { ++failures_; throw; }
}

void NativeKernel::close() {
    if (closed_) return;
    if (std::this_thread::get_id() != thread_ || !runtime_.isActive())
        throw std::runtime_error("close Hadamard on its owning runtime/thread");
    runtime_.synchronize(); closed_ = true;
}
NativeKernel::~NativeKernel() {
    try { close(); }
    catch (const std::exception &error) { std::fprintf(stderr, "llaisys Hadamard teardown failed: %s\n", error.what()); }
}
} // namespace llaisys::backends::hadamard

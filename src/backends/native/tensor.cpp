#include <cuda_runtime_api.h>
#include "tensor.hpp"
#include "src/core/context/context.hpp"

#include <stdexcept>

namespace llaisys::backends::native {

size_t Tensor::checkedBytes(const std::vector<int64_t> &shape, DLDataType dtype) {
    if (shape.empty() || shape.size() > 16 || !dtype.bits || !dtype.lanes
        || (dtype.code == kDLFloat4_e2m1fn && dtype.bits != 4))
        throw std::invalid_argument("invalid native tensor shape or low-precision dtype");
    uint64_t elements = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        const int64_t dim = shape[i];
        if (dim < 0 || (dim && elements > static_cast<uint64_t>(INT64_MAX) / static_cast<uint64_t>(dim)))
            throw std::invalid_argument("native tensor dimensions are negative or overflowing");
        elements *= static_cast<uint64_t>(dim);
    }
    const uint64_t width = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
    if (elements > static_cast<uint64_t>(INT64_MAX) / width)
        throw std::invalid_argument("native tensor size is overflowing");
    const uint64_t bits = elements * width;
    return static_cast<size_t>((bits + 7) / 8);
}

Tensor::Tensor(core::Runtime &runtime, std::vector<int64_t> shape, DLDataType dtype)
    : runtime_(runtime), thread_(std::this_thread::get_id()), shape_(std::move(shape)), dtype_(dtype),
      bytes_(checkedBytes(shape_, dtype)) {
    if (runtime.deviceType() != LLAISYS_DEVICE_NVIDIA || !runtime.isActive()
        || &core::context().runtime() != &runtime)
        throw std::invalid_argument("native tensors require an active NVIDIA runtime");
    checkThread();
    strides_.resize(shape_.size());
    int64_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
    if (bytes_) {
        auto storage = runtime.allocateDeviceStorage(bytes_);
        data_ = storage->memory(); capacity_ = storage->size(); owner_ = std::move(storage);
    }
    span_bytes_ = bytes_;
}

Tensor::Tensor(core::Runtime &runtime, std::shared_ptr<void> owner, std::byte *data, size_t capacity,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides, DLDataType dtype, uint64_t byte_offset)
    : runtime_(runtime), thread_(std::this_thread::get_id()), shape_(std::move(shape)),
      strides_(std::move(strides)), dtype_(dtype), bytes_(checkedBytes(shape_, dtype)),
      byte_offset_(byte_offset), owner_(std::move(owner)), data_(data), capacity_(capacity) {
    if (capacity_ && (!owner_ || !data_)) throw std::invalid_argument("native view has no storage owner");
    if (shape_.size() != strides_.size()) throw std::invalid_argument("native view rank mismatch");
    uint64_t last = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (strides_[i] < 0) throw std::invalid_argument("native negative strides are unsupported");
        const uint64_t count = shape_[i] ? static_cast<uint64_t>(shape_[i] - 1) : 0;
        if (count && static_cast<uint64_t>(strides_[i]) > (static_cast<uint64_t>(INT64_MAX) - last) / count)
            throw std::invalid_argument("native view span is overflowing");
        last += count * strides_[i];
    }
    const uint64_t width = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
    if (last >= static_cast<uint64_t>(INT64_MAX) / width)
        throw std::invalid_argument("native view bit span is overflowing");
    span_bytes_ = bytes_ ? ((last + 1) * width + 7) / 8 : 0;
    if (byte_offset_ > capacity_ || span_bytes_ > capacity_ - byte_offset_)
        throw std::invalid_argument("native view extends beyond its storage");
}

std::shared_ptr<Tensor> Tensor::asStrided(std::vector<int64_t> shape, std::vector<int64_t> strides,
                                        int64_t element_offset) const {
    checkThread();
    const uint64_t width = static_cast<uint64_t>(dtype_.bits) * dtype_.lanes;
    if (element_offset < 0 || static_cast<uint64_t>(element_offset) > static_cast<uint64_t>(INT64_MAX) / width)
        throw std::invalid_argument("invalid native view offset");
    const uint64_t bit_offset = static_cast<uint64_t>(element_offset) * width;
    if (bit_offset % 8) throw std::invalid_argument("native sub-byte view offset must be byte aligned");
    return std::shared_ptr<Tensor>(new Tensor(runtime_, owner_, data_, capacity_, std::move(shape), std::move(strides),
                                             dtype_, byte_offset_ + bit_offset / 8));
}

bool Tensor::isContiguous() const {
    if (!bytes_) return true;
    int64_t expected = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        if (shape_[i] > 1 && strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

bool Tensor::overlaps(const Tensor &other) const {
    // Independently exported views of the same paged component have different
    // export tokens, but still alias the same authoritative allocation.
    return data_ && data_ == other.data_ && span_bytes_ && other.span_bytes_
        && byte_offset_ < other.byte_offset_ + other.span_bytes_
        && other.byte_offset_ < byte_offset_ + span_bytes_;
}

void Tensor::checkThread() const {
    if (std::this_thread::get_id() != thread_ || !runtime_.isActive()
        || &core::context().runtime() != &runtime_)
        throw std::runtime_error("native tensor used outside its owning active runtime/thread");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != runtime_.deviceId())
        throw std::runtime_error("native tensor CUDA device does not match its runtime");
}

DLTensor Tensor::view() const {
    checkThread();
    return DLTensor{data_, {kDLCUDA, runtime_.deviceId()}, static_cast<int32_t>(shape_.size()),
                    dtype_, const_cast<int64_t *>(shape_.data()), const_cast<int64_t *>(strides_.data()), byte_offset_};
}

void Tensor::upload(const void *host, size_t bytes) {
    checkThread();
    if (bytes != bytes_ || (bytes && !host) || !isContiguous()) throw std::invalid_argument("native upload size/layout mismatch");
    if (!bytes) return;
    runtime_.api()->memcpy_async(data_ + byte_offset_, host, bytes, LLAISYS_MEMCPY_H2D, runtime_.stream());
    // Transfers are correctness/setup operations. Kernel call() does not sync.
    runtime_.synchronize();
}

void Tensor::download(void *host, size_t bytes) const {
    checkThread();
    if (bytes != bytes_ || (bytes && !host) || !isContiguous()) throw std::invalid_argument("native download size/layout mismatch");
    if (!bytes) return;
    runtime_.api()->memcpy_async(host, data_ + byte_offset_, bytes, LLAISYS_MEMCPY_D2H, runtime_.stream());
    runtime_.synchronize();
}

} // namespace llaisys::backends::native

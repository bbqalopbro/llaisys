#pragma once

#include "src/core/runtime/runtime.hpp"
#include "src/core/storage/storage.hpp"
#include <dlpack/dlpack.h>

#include <cstdint>
#include <thread>
#include <vector>

namespace llaisys::backends::native {

class PagedStorage;

// Backend-independent owner over the EXISTING Runtime/Storage allocator.
// Shape/strides count logical dtype elements, including packed FP4. Backends
// with a different ABI must adapt their descriptors, not reinterpret storage.
// Runtime and its owning thread must outlive the tensor; queued calls require
// the tensor to stay alive. No new allocator or Python ownership is introduced.
class Tensor {
public:
    Tensor(core::Runtime &runtime, std::vector<int64_t> shape, DLDataType dtype);
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    DLTensor view() const;
    // A checked view retaining the same native storage. Offsets are relative
    // to this view and count logical elements; sub-byte offsets must be byte aligned.
    std::shared_ptr<Tensor> asStrided(std::vector<int64_t> shape, std::vector<int64_t> strides,
                                    int64_t element_offset = 0) const;
    bool isContiguous() const;
    bool overlaps(const Tensor &other) const;
    const std::vector<int64_t> &shape() const { return shape_; }
    DLDataType dtype() const { return dtype_; }
    size_t bytes() const { return bytes_; }
    void upload(const void *host, size_t bytes);
    void download(void *host, size_t bytes) const;
    core::Runtime &runtime() const { return runtime_; }
    static size_t checkedBytes(const std::vector<int64_t> &shape, DLDataType dtype);

private:
    friend class PagedStorage;
    // Only native owners may supply storage. No public naked-pointer adoption.
    Tensor(core::Runtime &runtime, std::shared_ptr<void> owner, std::byte *data, size_t capacity,
           std::vector<int64_t> shape,
           std::vector<int64_t> strides, DLDataType dtype, uint64_t byte_offset);
    void checkThread() const;
    core::Runtime &runtime_;
    std::thread::id thread_;
    std::vector<int64_t> shape_, strides_;
    DLDataType dtype_;
    size_t bytes_;
    uint64_t byte_offset_ = 0, span_bytes_ = 0;
    std::shared_ptr<void> owner_;
    std::byte *data_ = nullptr;
    size_t capacity_ = 0;
};

} // namespace llaisys::backends::native

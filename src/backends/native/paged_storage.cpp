// Vendor headers precede the legacy __C macro in llaisys.h.
#include <cuda_runtime_api.h>
#include "paged_storage.hpp"
#include "src/core/context/context.hpp"

#include <atomic>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace llaisys::backends::native {
namespace {
std::atomic<uint64_t> live_bytes{0}, destruction_errors{0};
}

struct PagedStorage::Export {
    std::shared_ptr<PagedStorage> storage;
    explicit Export(std::shared_ptr<PagedStorage> owner) : storage(std::move(owner)) { ++storage->exports_; }
    ~Export() { --storage->exports_; }
};

std::shared_ptr<PagedStorage> PagedStorage::create(core::Runtime &runtime, size_t blocks,
                                                  core::cache_layout_t layout) {
    return std::shared_ptr<PagedStorage>(new PagedStorage(runtime, blocks, std::move(layout)));
}

PagedStorage::PagedStorage(core::Runtime &runtime, size_t blocks, core::cache_layout_t layout)
    : runtime_(runtime), thread_(std::this_thread::get_id()), blocks_(blocks), layout_(std::move(layout)) {
    checkThread();
    if (!layout_ || !blocks_ || !layout_->numComponents() || !layout_->numLayers())
        throw std::invalid_argument("native paged storage requires a non-empty layout and block pool");
    // Validate every component before the first allocation. No silent integer
    // wrapping, partial undersized pool, or dtype-specific allocator policy.
    constexpr size_t maximum = static_cast<size_t>(INT64_MAX);
    for (size_t c = 0; c < layout_->numComponents(); ++c) {
        size_t stride = 0;
        for (size_t layer = 0; layer < layout_->numLayers(); ++layer) {
            const size_t count = layout_->layerBytes(c, layer);
            if (count > maximum - stride) throw std::overflow_error("native paged layer bytes overflow");
            stride += count;
        }
        if (!stride || blocks_ > maximum / stride) throw std::overflow_error("native paged component bytes overflow");
        const size_t count = blocks_ * stride;
        if (count > maximum - bytes_) throw std::overflow_error("native paged total bytes overflow");
        bytes_ += count;
    }
    storage_ = std::make_unique<core::PagedCacheStorage>(blocks_, layout_, runtime_.api());
    live_bytes += bytes_;
}

void PagedStorage::checkThread() const {
    if (std::this_thread::get_id() != thread_ || !runtime_.isActive()
        || runtime_.deviceType() != LLAISYS_DEVICE_NVIDIA || &core::context().runtime() != &runtime_)
        throw std::runtime_error("native paged storage requires its owning active NVIDIA runtime/thread");
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != runtime_.deviceId())
        throw std::runtime_error("native paged storage CUDA device does not match its runtime");
}

void PagedStorage::checkOpen() const {
    checkThread();
    if (!storage_) throw std::runtime_error("native paged storage is closed");
}

size_t PagedStorage::componentBytes(size_t component) const {
    checkOpen();
    return blocks_ * storage_->componentBlockStride(component);
}
size_t PagedStorage::componentBlockStride(size_t component) const {
    checkOpen(); return storage_->componentBlockStride(component);
}
size_t PagedStorage::componentLayerOffset(size_t component, size_t layer) const {
    checkOpen();
    if (layer >= layout_->numLayers()) throw std::out_of_range("native paged layer out of range");
    return storage_->componentLayerOffset(component, layer);
}
size_t PagedStorage::activeExports() const { checkThread(); return exports_; }

std::shared_ptr<Tensor> PagedStorage::componentTensor(size_t component, std::vector<int64_t> shape, DLDataType dtype) {
    const size_t capacity = componentBytes(component);
    if (Tensor::checkedBytes(shape, dtype) != capacity)
        throw std::invalid_argument("native paged component tensor must describe the complete component");
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) { strides[i] = stride; stride *= shape[i]; }
    auto token = std::make_shared<Export>(shared_from_this());
    return std::shared_ptr<Tensor>(new Tensor(runtime_, std::move(token),
        static_cast<std::byte *>(storage_->componentPool(component)), capacity, std::move(shape),
        std::move(strides), dtype, 0));
}

void PagedStorage::close() {
    checkThread();
    if (!storage_) return;
    if (exports_) throw std::runtime_error("cannot close native paged storage with live tensor exports");
    runtime_.synchronize();
    storage_->release();
    storage_.reset();
    live_bytes -= bytes_;
}

PagedStorage::~PagedStorage() {
    try { close(); }
    catch (const std::exception &error) {
        ++destruction_errors;
        std::fprintf(stderr, "native paged storage teardown failed: %s\n", error.what());
        // A missing runtime/thread/stream must not free queued payload or claim
        // that bytes were recovered. This exceptional leak is observable.
        (void)storage_.release();
    } catch (...) {
        ++destruction_errors;
        std::fputs("native paged storage teardown failed with unknown error\n", stderr);
        (void)storage_.release();
    }
}

uint64_t PagedStorage::liveBytes() { return live_bytes.load(); }
uint64_t PagedStorage::destructionErrors() { return destruction_errors.load(); }
} // namespace llaisys::backends::native

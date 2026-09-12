#pragma once

#include "tensor.hpp"
#include "src/core/cache/paged_cache_storage.hpp"
#include <atomic>

namespace llaisys::backends::native {

// Runtime/thread owner for the EXISTING layout-driven paged allocator. It does
// not allocate logical blocks or interpret model-specific slot mappings. The
// core storage is private: callers cannot release a live exported component.
// All calls, and the final owner/view destruction, belong to the Runtime's
// owning thread. Runtime must outlive this owner and every exported Tensor.
class PagedStorage final : public std::enable_shared_from_this<PagedStorage> {
public:
    static std::shared_ptr<PagedStorage> create(core::Runtime &runtime, size_t blocks,
                                               core::cache_layout_t layout);
    ~PagedStorage();
    PagedStorage(const PagedStorage &) = delete;
    PagedStorage &operator=(const PagedStorage &) = delete;

    // Full-component contiguous view; dtype/shape interpretation remains the
    // model/backend's responsibility. asStrided() creates checked subviews.
    std::shared_ptr<Tensor> componentTensor(size_t component, std::vector<int64_t> shape, DLDataType dtype);
    size_t componentBytes(size_t component) const;
    size_t componentBlockStride(size_t component) const;
    size_t componentLayerOffset(size_t component, size_t layer) const;
    size_t numBlocks() const { return blocks_; }
    core::cache_layout_t layoutHandle() const { return layout_; }
    size_t activeExports() const;
    bool closed() const { return !storage_; }
    // Refuse close while a Tensor/view retains a component; otherwise wait for
    // queued work before releasing. Repeated close is harmless on owner thread.
    void close();
    static uint64_t liveBytes();
    static uint64_t destructionErrors();

private:
    struct Export;
    PagedStorage(core::Runtime &runtime, size_t blocks, core::cache_layout_t layout);
    void checkThread() const;
    void checkOpen() const;
    core::Runtime &runtime_;
    std::thread::id thread_;
    size_t blocks_, bytes_ = 0;
    std::atomic<size_t> exports_{0};
    core::cache_layout_t layout_;
    std::unique_ptr<core::PagedCacheStorage> storage_;
};

} // namespace llaisys::backends::native

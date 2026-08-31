#include "paged_cache_storage.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace llaisys::core {

PagedCacheStorage::PagedCacheStorage(size_t num_blocks, cache_layout_t layout,
                                     const LlaisysRuntimeAPI *api)
    : num_blocks_(num_blocks), layout_(std::move(layout)), api_(api) {
    if (num_blocks_ == 0) throw std::invalid_argument("PagedCacheStorage: num_blocks must be non-zero");
    if (!layout_) throw std::invalid_argument("PagedCacheStorage: layout is null");
    if (!api_) throw std::invalid_argument("PagedCacheStorage: RuntimeAPI is null");

    pools_.resize(layout_->numComponents());
    try {
        for (size_t c = 0; c < pools_.size(); ++c) {
            auto &pool = pools_[c];
            pool.layer_offsets.resize(layout_->numLayers() + 1, 0);
            for (size_t layer = 0; layer < layout_->numLayers(); ++layer) {
                pool.layer_offsets[layer + 1] =
                    pool.layer_offsets[layer] + layout_->layerBytes(c, layer);
            }
            pool.block_stride = pool.layer_offsets.back();
            if (pool.block_stride == 0) {
                throw std::invalid_argument("PagedCacheStorage: empty component");
            }
            pool.data = api_->malloc_device(num_blocks_ * pool.block_stride);
            if (!pool.data) throw std::bad_alloc();
        }
    } catch (...) {
        for (auto &pool : pools_) {
            if (pool.data) api_->free_device(pool.data);
            pool.data = nullptr;
        }
        throw;
    }
}

PagedCacheStorage::~PagedCacheStorage() {
    for (auto &pool : pools_) {
        if (pool.data) api_->free_device(pool.data);
        pool.data = nullptr;
    }
}

void PagedCacheStorage::validate(size_t component, int block_id, size_t layer) const {
    if (component >= pools_.size()) throw std::out_of_range("cache component out of range");
    if (block_id < 0 || static_cast<size_t>(block_id) >= num_blocks_)
        throw std::out_of_range("cache block id out of range");
    if (layer >= layout_->numLayers()) throw std::out_of_range("cache layer out of range");
}

void *PagedCacheStorage::componentPtr(size_t component, int block_id, size_t layer) {
    validate(component, block_id, layer);
    auto &pool = pools_[component];
    return static_cast<std::byte *>(pool.data) +
           static_cast<size_t>(block_id) * pool.block_stride +
           pool.layer_offsets[layer];
}

const void *PagedCacheStorage::componentPtr(size_t component, int block_id,
                                            size_t layer) const {
    validate(component, block_id, layer);
    const auto &pool = pools_[component];
    return static_cast<const std::byte *>(pool.data) +
           static_cast<size_t>(block_id) * pool.block_stride +
           pool.layer_offsets[layer];
}

void *PagedCacheStorage::componentPool(size_t component) const {
    return pools_.at(component).data;
}

int PagedCacheStorage::componentIndex(const std::string &name) const {
    for (size_t i = 0; i < layout_->numComponents(); ++i) {
        if (layout_->componentName(i) == name) return static_cast<int>(i);
    }
    return -1;
}

size_t PagedCacheStorage::componentBlockStride(size_t component) const {
    return pools_.at(component).block_stride;
}

size_t PagedCacheStorage::componentLayerOffset(size_t component, size_t layer) const {
    return pools_.at(component).layer_offsets.at(layer);
}

size_t PagedCacheStorage::componentLayerStride(size_t component, size_t layer) const {
    validate(component, 0, layer);
    const auto &offsets = pools_[component].layer_offsets;
    return offsets[layer + 1] - offsets[layer];
}

} // namespace llaisys::core

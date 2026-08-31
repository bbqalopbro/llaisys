#pragma once

#include "cache_layout.hpp"
#include "llaisys/runtime.h"

#include <cstddef>
#include <string>
#include <vector>

namespace llaisys::core {

// Owns device memory for a cache layout. It has no allocation policy and no
// request knowledge: callers address storage with a physical block id.
class PagedCacheStorage {
public:
    PagedCacheStorage(size_t num_blocks, cache_layout_t layout,
                      const LlaisysRuntimeAPI *api);
    ~PagedCacheStorage();

    PagedCacheStorage(const PagedCacheStorage &) = delete;
    PagedCacheStorage &operator=(const PagedCacheStorage &) = delete;

    void *componentPtr(size_t component, int block_id, size_t layer);
    const void *componentPtr(size_t component, int block_id, size_t layer) const;
    void *componentPool(size_t component) const;
    int componentIndex(const std::string &name) const;

    size_t componentBlockStride(size_t component) const;
    size_t componentLayerOffset(size_t component, size_t layer) const;
    size_t componentLayerStride(size_t component, size_t layer) const;
    size_t numBlocks() const { return num_blocks_; }
    const CacheLayout &layout() const { return *layout_; }
    cache_layout_t layoutHandle() const { return layout_; }

private:
    struct ComponentPool {
        void *data = nullptr;
        size_t block_stride = 0;
        std::vector<size_t> layer_offsets;
    };

    void validate(size_t component, int block_id, size_t layer) const;

    size_t num_blocks_;
    cache_layout_t layout_;
    const LlaisysRuntimeAPI *api_;
    std::vector<ComponentPool> pools_;
};

} // namespace llaisys::core

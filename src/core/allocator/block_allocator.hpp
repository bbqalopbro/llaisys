#pragma once

#include "../cache/block_manager.hpp"
#include "../cache/paged_cache_storage.hpp"
#include "llaisys/runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace llaisys::core {

struct BlockAllocatorConfig {
    size_t num_blocks;      //总共有多少个物理块
    size_t block_size;    // 每块能装多少个 token
    size_t nlayer;
    size_t nkvh;          // KV head 数
    size_t dh;            // 每个 head 的维度
    size_t elem_size;     // 单元素字节数。比如 FP32 是 4，FP16 是 2
};

// Pool-based allocator for paged KV-Cache blocks.
//内存不是按“每个请求一整段连续 cache”分配，而是扁平化成：: [num_blocks, nlayer, block_size, nkvh, dh]
// Each (block_id, layer) pair maps to a contiguous [block_size, nkvh, dh] region.
class BlockAllocator {
private:
    BlockAllocatorConfig _config;
    std::unique_ptr<BlockManager> _blocks;
    std::unique_ptr<PagedCacheStorage> _storage;

public:
    BlockAllocator(const BlockAllocatorConfig &config, const LlaisysRuntimeAPI *api);
    ~BlockAllocator();

    BlockAllocator(const BlockAllocator &) = delete;
    BlockAllocator &operator=(const BlockAllocator &) = delete;
    BlockAllocator(BlockAllocator &&) = delete;
    BlockAllocator &operator=(BlockAllocator &&) = delete;

    int alloc();
    void free(int block_id);

    void *get_k_ptr(int block_id, int layer);
    void *get_v_ptr(int block_id, int layer);

    size_t num_free() const;
    size_t num_total() const;

    size_t block_size() const { return _config.block_size; }
    size_t nlayer() const { return _config.nlayer; }
    size_t nkvh() const { return _config.nkvh; }
    size_t dh() const { return _config.dh; }
    size_t elem_size() const { return _config.elem_size; }
    size_t block_stride() const { return _storage->componentBlockStride(0); }
    size_t layer_stride() const { return _storage->componentLayerStride(0, 0); }

    void *pool_k_raw() const { return _storage->componentPool(0); }
    void *pool_v_raw() const { return _storage->componentPool(1); }

    // Generic interfaces used by future attention layouts (for example MLA).
    BlockManager &block_manager() { return *_blocks; }
    const BlockManager &block_manager() const { return *_blocks; }
    PagedCacheStorage &storage() { return *_storage; }
    const PagedCacheStorage &storage() const { return *_storage; }
    const CacheLayout &layout() const { return _storage->layout(); }
    void *component_ptr(size_t component, int block_id, size_t layer) {
        return _storage->componentPtr(component, block_id, layer);
    }
};

} // namespace llaisys::core

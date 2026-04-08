#pragma once

#include "llaisys/runtime.h"

#include <deque>
#include <cstddef>
#include <cstdint>

namespace llaisys::core {

struct BlockAllocatorConfig {
    size_t num_blocks;
    size_t block_size;    // tokens per block (e.g. 16)
    size_t nlayer;
    size_t nkvh;          // local KV heads (already divided by tp_size)
    size_t dh;            // head dimension
    size_t elem_size;     // bytes per element (e.g. 4 for FP32)
};

// Pool-based allocator for paged KV-Cache blocks.
// Memory layout (flat): [num_blocks, nlayer, block_size, nkvh, dh]
// Each (block_id, layer) pair maps to a contiguous [block_size, nkvh, dh] region.
class BlockAllocator {
private:
    void *_pool_K;
    void *_pool_V;
    const LlaisysRuntimeAPI *_api;
    BlockAllocatorConfig _config;
    size_t _block_stride;   // bytes per block across all layers
    size_t _layer_stride;   // bytes per single (block_size, nkvh, dh) region
    std::deque<int> _free_list;

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
    size_t block_stride() const { return _block_stride; }
    size_t layer_stride() const { return _layer_stride; }

    void *pool_k_raw() const { return _pool_K; }
    void *pool_v_raw() const { return _pool_V; }
};

} // namespace llaisys::core

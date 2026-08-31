#include "block_allocator.hpp"

#include <memory>
#include <stdexcept>

namespace llaisys::core {

BlockAllocator::BlockAllocator(const BlockAllocatorConfig &config, const LlaisysRuntimeAPI *api)
    : _config(config) {
    if (!api)
        throw std::runtime_error("BlockAllocator: RuntimeAPI is null");
    if (_config.num_blocks == 0)
        throw std::runtime_error("BlockAllocator: num_blocks must be > 0");

    auto layout = std::make_shared<StandardKVCacheLayout>(
        _config.block_size, _config.nlayer, _config.nkvh,
        _config.dh, _config.elem_size);
    _storage = std::make_unique<PagedCacheStorage>(_config.num_blocks,
                                                   std::move(layout), api);
    _blocks = std::make_unique<BlockManager>(_config.num_blocks);
}

BlockAllocator::~BlockAllocator() = default;

int BlockAllocator::alloc() {
    return _blocks->allocate();
}

void BlockAllocator::free(int block_id) {
    _blocks->release(block_id);
}

void *BlockAllocator::get_k_ptr(int block_id, int layer) {
    return _storage->componentPtr(0, block_id, static_cast<size_t>(layer));
    /*先从 K pool 起点开始
    跳过 block_id 个完整 block
    再跳到这个 block 内的第 layer 层
    得到该 (block_id, layer) 的 K 区域首地址
    */
}

void *BlockAllocator::get_v_ptr(int block_id, int layer) {
    return _storage->componentPtr(1, block_id, static_cast<size_t>(layer));
}

size_t BlockAllocator::num_free() const {
    return _blocks->numFree();
}

size_t BlockAllocator::num_total() const {
    return _blocks->numTotal();
}

} // namespace llaisys::core

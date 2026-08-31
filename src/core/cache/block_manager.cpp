#include "block_manager.hpp"

#include <limits>
#include <stdexcept>

namespace llaisys::core {

BlockManager::BlockManager(size_t num_blocks) : blocks_(num_blocks) {
    if (num_blocks == 0) throw std::invalid_argument("BlockManager: num_blocks must be non-zero");
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_[i].block_id = static_cast<int>(i);
        free_list_.push_back(static_cast<int>(i));
    }
}

uint64_t BlockManager::tick() { return ++clock_; }

BlockMetadata *BlockManager::mutableMetadata(int block_id) {
    if (block_id < 0 || static_cast<size_t>(block_id) >= blocks_.size()) return nullptr;
    return &blocks_[block_id];
}

const BlockMetadata &BlockManager::metadata(int block_id) const {
    if (block_id < 0 || static_cast<size_t>(block_id) >= blocks_.size())
        throw std::out_of_range("BlockManager: block id out of range");
    return blocks_[block_id];
}

int BlockManager::allocate() {
    if (free_list_.empty() && !evictOneCached()) return -1;
    const int block_id = free_list_.front();
    free_list_.pop_front();
    auto &block = blocks_[block_id];
    block.ref_count = 1;
    block.num_tokens = 0;
    block.allocated = true;
    block.computed = false;
    block.cached = false;
    block.block_hash = 0;
    block.last_access = tick();
    return block_id;
}

bool BlockManager::retain(int block_id) {
    auto *block = mutableMetadata(block_id);
    if (!block || !block->allocated) return false;
    ++block->ref_count;
    block->last_access = tick();
    return true;
}

void BlockManager::recycle(BlockMetadata &block) {
    if (block.cached && block.block_hash != 0) prefix_index_.erase(block.block_hash);
    const int block_id = block.block_id;
    block = BlockMetadata{};
    block.block_id = block_id;
    free_list_.push_back(block_id);
}

bool BlockManager::release(int block_id) {
    auto *block = mutableMetadata(block_id);
    if (!block || !block->allocated || block->ref_count == 0) return false;
    --block->ref_count;
    block->last_access = tick();
    if (block->ref_count == 0 && !block->cached) recycle(*block);
    return true;
}

bool BlockManager::markComputed(int block_id, uint32_t num_tokens) {
    auto *block = mutableMetadata(block_id);
    if (!block || !block->allocated) return false;
    block->computed = true;
    block->num_tokens = num_tokens;
    block->last_access = tick();
    return true;
}

bool BlockManager::cache(int block_id, uint64_t block_hash) {
    auto *block = mutableMetadata(block_id);
    if (!block || !block->allocated || !block->computed || block_hash == 0) return false;
    auto existing = prefix_index_.find(block_hash);
    if (existing != prefix_index_.end() && existing->second != block_id) return false;
    block->cached = true;
    block->block_hash = block_hash;
    block->last_access = tick();
    prefix_index_[block_hash] = block_id;
    return true;
}

int BlockManager::findCached(uint64_t block_hash) {
    auto it = prefix_index_.find(block_hash);
    if (it == prefix_index_.end()) return -1;
    auto *block = mutableMetadata(it->second);
    if (!block || !block->allocated || !block->cached) {
        prefix_index_.erase(it);
        return -1;
    }
    retain(block->block_id);
    return block->block_id;
}

bool BlockManager::uncache(int block_id) {
    auto *block = mutableMetadata(block_id);
    if (!block || !block->allocated || !block->cached) return false;
    if (block->block_hash != 0) prefix_index_.erase(block->block_hash);
    block->cached = false;
    block->block_hash = 0;
    if (block->ref_count == 0) recycle(*block);
    return true;
}

bool BlockManager::evictOneCached() {
    BlockMetadata *victim = nullptr;
    for (auto &block : blocks_) {
        if (block.allocated && block.cached && block.ref_count == 0 &&
            (!victim || block.last_access < victim->last_access)) {
            victim = &block;
        }
    }
    if (!victim) return false;
    recycle(*victim);
    return true;
}

size_t BlockManager::numCached() const {
    size_t count = 0;
    for (const auto &block : blocks_) if (block.allocated && block.cached) ++count;
    return count;
}

size_t BlockManager::numFree() const {
    size_t available = free_list_.size();
    for (const auto &block : blocks_) {
        if (block.allocated && block.cached && block.ref_count == 0) ++available;
    }
    return available;
}

} // namespace llaisys::core

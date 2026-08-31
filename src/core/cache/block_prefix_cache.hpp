#pragma once

#include "block_manager.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::core {

struct PrefixMatch {
    std::vector<int> block_ids;
    size_t matched_tokens = 0;
    uint64_t terminal_hash = 0;
};

// Block-granular prefix index. Only complete, computed blocks are inserted;
// partial blocks remain request-owned to avoid ambiguous cache validity.
class BlockPrefixCache {
public:
    BlockPrefixCache(BlockManager &blocks, size_t block_size,
                     uint64_t cache_salt = 0);

    PrefixMatch match(const int64_t *tokens, size_t num_tokens);
    bool insert(const int64_t *tokens, size_t num_tokens,
                const std::vector<int> &block_ids);
    void release(const PrefixMatch &match);

    static uint64_t hashBlock(uint64_t parent_hash, const int64_t *tokens,
                              size_t count, uint64_t cache_salt = 0);

private:
    BlockManager &blocks_;
    size_t block_size_;
    uint64_t cache_salt_;
};

} // namespace llaisys::core

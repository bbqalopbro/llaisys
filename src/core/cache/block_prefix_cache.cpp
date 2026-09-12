#include "block_prefix_cache.hpp"

#include <stdexcept>

namespace llaisys::core {
namespace {

constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;

void hashBytes(uint64_t &hash, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        hash ^= static_cast<uint8_t>(value & 0xffU);
        hash *= FNV_PRIME;
        value >>= 8U;
    }
}

} // namespace

BlockPrefixCache::BlockPrefixCache(BlockManager &blocks, size_t block_size,
                                   uint64_t cache_salt)
    : blocks_(blocks), block_size_(block_size), cache_salt_(cache_salt) {
    if (block_size_ == 0) {
        throw std::invalid_argument("BlockPrefixCache: block_size must be non-zero");
    }
}

uint64_t BlockPrefixCache::hashBlock(uint64_t parent_hash,
                                     const int64_t *tokens, size_t count,
                                     uint64_t cache_salt) {
    uint64_t hash = FNV_OFFSET;
    hashBytes(hash, cache_salt);
    hashBytes(hash, parent_hash);
    hashBytes(hash, static_cast<uint64_t>(count));
    for (size_t i = 0; i < count; ++i) {
        hashBytes(hash, static_cast<uint64_t>(tokens[i]));
    }
    // BlockManager reserves zero as the absence of a hash.
    return hash == 0 ? 1 : hash;
}

PrefixMatch BlockPrefixCache::match(const int64_t *tokens, size_t num_tokens) {
    if (!tokens && num_tokens != 0) {
        throw std::invalid_argument("BlockPrefixCache: tokens is null");
    }
    PrefixMatch result;
    uint64_t parent_hash = 0;
    const size_t full_blocks = num_tokens / block_size_;
    // Reserve before findCached retains anything. Otherwise a growing vector
    // can throw after a retain and strand references outside the result.
    result.block_ids.reserve(full_blocks);
    try {
        for (size_t block = 0; block < full_blocks; ++block) {
            const uint64_t hash = hashBlock(parent_hash,
                                            tokens + block * block_size_,
                                            block_size_, cache_salt_);
            const int block_id = blocks_.findCached(hash);
            if (block_id < 0) break;
            result.block_ids.push_back(block_id);
            result.matched_tokens += block_size_;
            result.terminal_hash = hash;
            parent_hash = hash;
        }
    } catch (...) {
        release(result);
        throw;
    }
    return result;
}

bool BlockPrefixCache::insert(const int64_t *tokens, size_t num_tokens,
                              const std::vector<int> &block_ids) {
    if (!tokens && num_tokens != 0) return false;
    const size_t full_blocks = num_tokens / block_size_;
    if (block_ids.size() < full_blocks) return false;

    uint64_t parent_hash = 0;
    for (size_t block = 0; block < full_blocks; ++block) {
        const uint64_t hash = hashBlock(parent_hash,
                                        tokens + block * block_size_,
                                        block_size_, cache_salt_);
        if (!blocks_.markComputed(block_ids[block],
                                  static_cast<uint32_t>(block_size_)) ||
            !blocks_.cache(block_ids[block], hash)) {
            return false;
        }
        parent_hash = hash;
    }
    return true;
}

void BlockPrefixCache::release(const PrefixMatch &match) {
    for (int block_id : match.block_ids) blocks_.release(block_id);
}

} // namespace llaisys::core

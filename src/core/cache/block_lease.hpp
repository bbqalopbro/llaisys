#pragma once

#include "block_manager.hpp"
#include "block_prefix_cache.hpp"

#include <memory>
#include <vector>

namespace llaisys::core {

class CacheBlockPool;
class CachePrefixIndex;

// Owns block references, never payload or scheduling policy. All operations
// require external serialization (the GIL or one native Session worker).
// Only pool/index factories may initially adopt references from BlockManager.
class CacheBlockLease {
public:
    ~CacheBlockLease();
    CacheBlockLease(const CacheBlockLease &) = delete;
    CacheBlockLease &operator=(const CacheBlockLease &) = delete;
    void close() noexcept;
    std::shared_ptr<CacheBlockLease> share();
    std::shared_ptr<CacheBlockLease> prefix(size_t count);
    void replace(size_t index, CacheBlockLease &other);
    void append(CacheBlockLease &other);
    void markComputed(uint32_t tokens);
    void markComputedCounts(const std::vector<uint32_t> &tokens);
    bool closed() const { return closed_; }
    const std::vector<int> &ids() const { return ids_; }
    size_t matchedTokens() const { return matched_tokens_; }
    uint64_t terminalHash() const { return terminal_hash_; }
    const std::shared_ptr<CacheBlockPool> &pool() const { return pool_; }
    void requireOpen() const;

private:
    friend class CacheBlockPool;
    friend class CachePrefixIndex;
    explicit CacheBlockLease(std::shared_ptr<CacheBlockPool> pool);
    std::shared_ptr<CacheBlockPool> pool_;
    std::vector<int> ids_;
    size_t matched_tokens_ = 0;
    uint64_t terminal_hash_ = 0;
    bool closed_ = false;
};

// Metadata facade over the existing BlockManager; no second allocator/index.
// A pool must be shared-owned before allocate() creates an owning lease.
class CacheBlockPool : public std::enable_shared_from_this<CacheBlockPool> {
public:
    explicit CacheBlockPool(size_t count);
    CacheBlockPool(const CacheBlockPool &) = delete;
    CacheBlockPool &operator=(const CacheBlockPool &) = delete;
    std::shared_ptr<CacheBlockLease> allocate(size_t count = 1);
    BlockMetadata metadata(int id) const;
    size_t free() const;
    size_t total() const;
    size_t cached() const;
    bool uncache(int id);

private:
    friend class CacheBlockLease;
    friend class CachePrefixIndex;
    BlockManager blocks_;
};

class CachePrefixIndex {
public:
    CachePrefixIndex(std::shared_ptr<CacheBlockPool> pool, size_t block_size, uint64_t salt = 0);
    bool publish(const std::vector<int64_t> &tokens, const CacheBlockLease &lease);
    std::shared_ptr<CacheBlockLease> lookup(const std::vector<int64_t> &tokens);

private:
    std::shared_ptr<CacheBlockPool> pool_;
    size_t block_size_;
    uint64_t salt_;
    BlockPrefixCache index_;
};

} // namespace llaisys::core

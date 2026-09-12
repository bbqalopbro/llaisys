#include "block_lease.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaisys::core {
namespace {
size_t checkedCount(size_t count) {
    if (count == 0 || count > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("cache block count must fit a positive int32 ID range");
    return count;
}

size_t checkedSize(size_t size) {
    if (size == 0 || size > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("prefix block size must fit a positive uint32 token count");
    return size;
}

std::shared_ptr<CacheBlockPool> checkedPool(std::shared_ptr<CacheBlockPool> pool) {
    if (!pool) throw std::invalid_argument("prefix index requires a block pool");
    return pool;
}
} // namespace

CacheBlockPool::CacheBlockPool(size_t count) : blocks_(checkedCount(count)) {}

std::shared_ptr<CacheBlockLease> CacheBlockPool::allocate(size_t count) {
    if (count > blocks_.numFree()) throw std::runtime_error("insufficient free cache blocks");
    // Prepare the holder/control block/vector before allocating any IDs. A
    // later failure destroys the lease and releases every already-adopted ID.
    auto lease = std::shared_ptr<CacheBlockLease>(new CacheBlockLease(shared_from_this()));
    lease->ids_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        int id = blocks_.allocate();
        if (id < 0) throw std::runtime_error("cache block allocation failed");
        lease->ids_.push_back(id); // Reserved int storage cannot allocate here.
    }
    return lease;
}

BlockMetadata CacheBlockPool::metadata(int id) const { return blocks_.metadata(id); }
size_t CacheBlockPool::free() const { return blocks_.numFree(); }
size_t CacheBlockPool::total() const { return blocks_.numTotal(); }
size_t CacheBlockPool::cached() const { return blocks_.numCached(); }
bool CacheBlockPool::uncache(int id) { return blocks_.uncache(id); }

CacheBlockLease::CacheBlockLease(std::shared_ptr<CacheBlockPool> pool) : pool_(std::move(pool)) {}
CacheBlockLease::~CacheBlockLease() { close(); }

void CacheBlockLease::requireOpen() const {
    if (closed_) throw std::runtime_error("cache block lease is closed");
}

void CacheBlockLease::close() noexcept {
    if (closed_) return;
    for (int id : ids_) pool_->blocks_.release(id);
    ids_.clear();
    closed_ = true;
}

std::shared_ptr<CacheBlockLease> CacheBlockLease::share() { return prefix(ids_.size()); }

std::shared_ptr<CacheBlockLease> CacheBlockLease::prefix(size_t count) {
    requireOpen();
    if (count > ids_.size()) throw std::invalid_argument("prefix exceeds lease length");
    auto result = std::shared_ptr<CacheBlockLease>(new CacheBlockLease(pool_));
    result->matched_tokens_ = ids_.empty() ? 0 : matched_tokens_ / ids_.size() * count;
    result->terminal_hash_ = count ? pool_->metadata(ids_[count - 1]).block_hash : 0;
    result->ids_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        int id = ids_[i];
        if (!pool_->blocks_.retain(id)) throw std::runtime_error("cannot retain cache block");
        result->ids_.push_back(id);
    }
    return result;
}

void CacheBlockLease::replace(size_t index, CacheBlockLease &other) {
    requireOpen();
    other.requireOpen();
    if (this == &other || pool_.get() != other.pool_.get() || other.ids_.size() != 1 || index >= ids_.size())
        throw std::invalid_argument("replace requires one owned block from the same pool and a valid index");
    int id = other.ids_[0];
    if (std::find(ids_.begin(), ids_.end(), id) != ids_.end())
        throw std::invalid_argument("replace cannot introduce duplicate physical blocks");
    // Payload copying and device metadata preparation happen before this
    // ownership commit. No allocation remains in this lease operation.
    pool_->blocks_.release(ids_[index]);
    ids_[index] = id;
    other.ids_.clear();
    other.closed_ = true;
    matched_tokens_ = 0;
    terminal_hash_ = 0;
}

void CacheBlockLease::markComputed(uint32_t tokens) {
    markComputedCounts(std::vector<uint32_t>(ids_.size(), tokens));
}

void CacheBlockLease::append(CacheBlockLease &other) {
    requireOpen();
    other.requireOpen();
    if (this == &other || pool_.get() != other.pool_.get())
        throw std::invalid_argument("append requires distinct leases from the same block pool");
    for (int id : other.ids_) {
        if (std::find(ids_.begin(), ids_.end(), id) != ids_.end())
            throw std::invalid_argument("cannot append duplicate physical cache blocks");
    }
    // Reserve before transferring ownership; failure preserves both leases.
    ids_.reserve(ids_.size() + other.ids_.size());
    ids_.insert(ids_.end(), other.ids_.begin(), other.ids_.end());
    other.ids_.clear();
    other.closed_ = true;
    matched_tokens_ = 0;
    terminal_hash_ = 0;
}

void CacheBlockLease::markComputedCounts(const std::vector<uint32_t> &tokens) {
    requireOpen();
    if (tokens.size() != ids_.size()) throw std::invalid_argument("one validity count per block is required");
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (tokens[i] == 0) throw std::invalid_argument("computed token count must be positive");
        const auto metadata = pool_->metadata(ids_[i]);
        if (metadata.cached && metadata.num_tokens != tokens[i])
            throw std::invalid_argument("cannot change the validity of a published cache block");
    }
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (!pool_->blocks_.markComputed(ids_[i], tokens[i])) throw std::runtime_error("cannot mark cache block computed");
    }
}

CachePrefixIndex::CachePrefixIndex(std::shared_ptr<CacheBlockPool> pool, size_t block_size, uint64_t salt)
    : pool_(checkedPool(std::move(pool))), block_size_(checkedSize(block_size)), salt_(salt),
      index_(pool_->blocks_, block_size_, salt_) {}

bool CachePrefixIndex::publish(const std::vector<int64_t> &tokens, const CacheBlockLease &lease) {
    lease.requireOpen();
    if (lease.pool().get() != pool_.get()) throw std::invalid_argument("lease belongs to another block pool");
    if (tokens.empty() || tokens.size() % block_size_ != 0 || tokens.size() / block_size_ > lease.ids().size())
        throw std::invalid_argument("publish requires complete token blocks backed by the lease");
    uint64_t parent = 0;
    for (size_t i = 0; i < tokens.size() / block_size_; ++i) {
        const auto metadata = pool_->metadata(lease.ids()[i]);
        const uint64_t hash = BlockPrefixCache::hashBlock(parent, tokens.data() + i * block_size_, block_size_, salt_);
        if (!metadata.computed || metadata.num_tokens != block_size_)
            throw std::invalid_argument("only complete computed blocks may be published");
        if (metadata.cached && metadata.block_hash != hash)
            throw std::invalid_argument("cannot republish cached payload under a different prefix");
        const int existing = pool_->blocks_.findCached(hash);
        if (existing >= 0) {
            pool_->blocks_.release(existing);
            if (existing != lease.ids()[i]) return false;
        }
        parent = hash;
    }
    return index_.insert(tokens.data(), tokens.size(), lease.ids());
}

std::shared_ptr<CacheBlockLease> CachePrefixIndex::lookup(const std::vector<int64_t> &tokens) {
    // The empty holder is allocated before matching retains any references.
    // BlockPrefixCache::match rolls back on failure; swap cannot allocate.
    auto result = std::shared_ptr<CacheBlockLease>(new CacheBlockLease(pool_));
    auto match = index_.match(tokens.data(), tokens.size());
    result->ids_.swap(match.block_ids);
    result->matched_tokens_ = match.matched_tokens;
    result->terminal_hash_ = match.terminal_hash;
    return result;
}

} // namespace llaisys::core

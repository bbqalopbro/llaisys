// CPU-only ownership and failure contracts. No CUDA, payload or Python runtime.
#include "src/core/cache/block_lease.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace allocation_fault {
// Fail exactly one allocation, then permit exception cleanup to allocate.
// This executable is single-threaded and never injects into production code.
long remaining = -1;
class Scope {
public:
    explicit Scope(long count) { remaining = count; }
    ~Scope() { remaining = -1; }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
};
} // namespace allocation_fault

void *operator new(std::size_t bytes) {
    if (allocation_fault::remaining >= 0 && allocation_fault::remaining-- == 0) {
        allocation_fault::remaining = -1;
        throw std::bad_alloc();
    }
    if (void *memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
// Keep replacement deallocators out of STL callers: GCC otherwise diagnoses
// its own inlined new -> replacement delete -> free as mismatched allocation.
[[gnu::noinline]] void operator delete(void *memory) noexcept { std::free(memory); }
[[gnu::noinline]] void operator delete[](void *memory) noexcept { std::free(memory); }
[[gnu::noinline]] void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
[[gnu::noinline]] void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

namespace {
using namespace llaisys::core;
size_t checks = 0;

void require(bool condition, const char *message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class Function>
void rejects(Function function, const char *message) {
    bool caught = false;
    try { function(); } catch (const Exception &) { caught = true; }
    require(caught, message);
}

static_assert(!std::is_default_constructible_v<CacheBlockLease>);
static_assert(!std::is_constructible_v<CacheBlockLease, std::shared_ptr<CacheBlockPool>>);
static_assert(!std::is_constructible_v<CacheBlockLease, std::shared_ptr<CacheBlockPool>, std::vector<int>>);
static_assert(!std::is_copy_constructible_v<CacheBlockLease>);
static_assert(!std::is_copy_constructible_v<CacheBlockPool>);

void allocationAndClose() {
    auto pool = std::make_shared<CacheBlockPool>(4);
    require(pool->total() == 4 && pool->free() == 4, "initial capacity");
    auto empty = pool->allocate(0);
    require(empty->ids().empty() && !empty->closed(), "empty lease is open");
    auto source = pool->allocate(2);
    const auto ids = source->ids();
    auto shared = source->share();
    const auto snapshot = pool->metadata(ids[0]);
    require(snapshot.ref_count == 2, "share retains refs");
    source->close(); source->close();
    require(source->closed() && source->ids().empty(), "close is idempotent");
    require(pool->metadata(ids[0]).ref_count == 1 && snapshot.ref_count == 2, "metadata is a snapshot");
    shared.reset();
    require(pool->free() == 4 && !pool->metadata(ids[0]).allocated, "destruction releases refs");
    rejects<std::runtime_error>([&] { source->share(); }, "share closed lease");
    rejects<std::runtime_error>([&] { source->prefix(0); }, "prefix closed lease");
    rejects<std::runtime_error>([&] { source->markComputed(4); }, "mark closed lease");
    rejects<std::runtime_error>([&] { source->append(*empty); }, "append to closed lease");
    rejects<std::runtime_error>([&] { empty->append(*source); }, "append closed donor");
    empty->close();
}

void poolLifetimeAndCapacity() {
    auto pool = std::make_shared<CacheBlockPool>(2);
    std::weak_ptr<CacheBlockPool> weak = pool;
    auto owner = pool->allocate(2);
    rejects<std::runtime_error>([&] { pool->allocate(1); }, "allocation must preflight capacity");
    require(pool->free() == 0 && pool->metadata(owner->ids()[0]).ref_count == 1, "failed allocation preserves ownership");
    pool.reset();
    require(!weak.expired(), "lease keeps pool alive");
    owner.reset();
    require(weak.expired(), "last owner releases pool");
    CacheBlockPool stack_pool(2);
    rejects<std::bad_weak_ptr>([&] { stack_pool.allocate(); }, "unshared pool must reject owning lease");
    require(stack_pool.free() == 2, "shared_from_this failure leaves all blocks free");
    rejects<std::invalid_argument>([] { CacheBlockPool invalid(0); }, "zero pool count");
    rejects<std::invalid_argument>([] {
        CacheBlockPool invalid(static_cast<size_t>(std::numeric_limits<int>::max()) + 1);
    }, "pool count above int32");
    auto valid = std::make_shared<CacheBlockPool>(1);
    rejects<std::out_of_range>([&] { valid->metadata(-1); }, "negative block id");
    rejects<std::out_of_range>([&] { valid->metadata(1); }, "block id beyond pool");
    rejects<std::invalid_argument>([] { CachePrefixIndex invalid(nullptr, 4); }, "null prefix pool");
    rejects<std::invalid_argument>([&] { CachePrefixIndex invalid(valid, 0); }, "zero prefix block size");
    rejects<std::invalid_argument>([&] {
        CachePrefixIndex invalid(valid, static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1);
    }, "prefix block size above uint32");
}

void appendAndReplace() {
    auto pool = std::make_shared<CacheBlockPool>(6);
    auto first = pool->allocate(2), second = pool->allocate(2);
    auto expected = first->ids();
    expected.insert(expected.end(), second->ids().begin(), second->ids().end());
    first->append(*second);
    require(first->ids() == expected && second->closed() && second->ids().empty(), "append transfers donor refs");
    for (int id : expected) require(pool->metadata(id).ref_count == 1, "append must not retain or release");
    auto alias = first->prefix(1);
    auto other = std::make_shared<CacheBlockPool>(1)->allocate();
    rejects<std::invalid_argument>([&] { first->append(*alias); }, "append duplicate ids");
    rejects<std::invalid_argument>([&] { first->append(*first); }, "append self");
    rejects<std::invalid_argument>([&] { first->append(*other); }, "append foreign pool");
    rejects<std::invalid_argument>([&] { first->replace(0, *alias); }, "replace duplicate ids");
    rejects<std::invalid_argument>([&] { first->replace(0, *other); }, "replace foreign pool");
    rejects<std::invalid_argument>([&] { first->replace(0, *first); }, "replace self");
    rejects<std::invalid_argument>([&] { first->prefix(5); }, "prefix beyond lease size");
    require(first->ids() == expected && !alias->closed(), "invalid transfers preserve both leases");
    auto replacement = pool->allocate();
    rejects<std::invalid_argument>([&] { first->replace(4, *replacement); }, "replace invalid index");
    const int old_id = first->ids()[0], new_id = replacement->ids()[0];
    first->replace(0, *replacement);
    require(replacement->closed() && first->ids()[0] == new_id, "replace transfers exactly one ref");
    require(pool->metadata(old_id).ref_count == 1 && pool->metadata(new_id).ref_count == 1, "replace preserves other reader");
    rejects<std::runtime_error>([&] { first->replace(0, *replacement); }, "replace closed donor");
    alias->close(); first->close();
    require(pool->free() == 6, "all transferred refs recovered");
}

void validityAndPublish() {
    auto pool = std::make_shared<CacheBlockPool>(4);
    auto lease = pool->allocate(2);
    const auto ids = lease->ids();
    CachePrefixIndex index(pool, 4);
    const std::vector<int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8};
    rejects<std::invalid_argument>([&] { lease->markComputedCounts({4}); }, "validity count size");
    rejects<std::invalid_argument>([&] { lease->markComputedCounts({4, 0}); }, "validity count zero");
    require(!pool->metadata(ids[0]).computed && !pool->metadata(ids[1]).computed, "prevalidate all validity counts");
    rejects<std::invalid_argument>([&] { index.publish(tokens, *lease); }, "uncomputed publish");
    lease->markComputedCounts({4, 1});
    require(pool->metadata(ids[1]).num_tokens == 1, "partial token validity");
    rejects<std::invalid_argument>([&] { index.publish(tokens, *lease); }, "partial block publish");
    require(pool->cached() == 0, "publish validates every block before mutation");
    lease->markComputed(4);
    for (const auto &invalid : std::vector<std::vector<int64_t>>{{}, {1, 2, 3}, std::vector<int64_t>(12)})
        rejects<std::invalid_argument>([&] { index.publish(invalid, *lease); }, "publish shape/budget");
    auto foreign = std::make_shared<CacheBlockPool>(1)->allocate();
    foreign->markComputed(4);
    rejects<std::invalid_argument>([&] { index.publish({1, 2, 3, 4}, *foreign); }, "publish foreign lease");
    require(index.publish(tokens, *lease), "publish complete blocks");
    rejects<std::invalid_argument>([&] { lease->markComputedCounts({4, 3}); }, "published validity immutable");
    require(pool->metadata(ids[1]).num_tokens == 4, "failed mark preserves published counts");
    lease->close();
    rejects<std::runtime_error>([&] { index.publish(tokens, *lease); }, "publish closed lease");
}

void prefixesAndLru() {
    auto pool = std::make_shared<CacheBlockPool>(4);
    CachePrefixIndex index(pool, 4, 123);
    auto source = pool->allocate(2);
    const auto ids = source->ids();
    source->markComputed(4);
    const std::vector<int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8};
    require(index.publish(tokens, *source), "prefix publish");
    auto longer = tokens; longer.insert(longer.end(), {9, 10, 11});
    auto match = index.lookup(longer);
    require(match->ids() == ids && match->matchedTokens() == 8 && match->terminalHash() != 0, "match only complete token blocks");
    auto prefix = match->prefix(1);
    require(prefix->matchedTokens() == 4 && prefix->terminalHash() == pool->metadata(ids[0]).block_hash, "prefix metadata matches its boundary");
    auto empty = match->prefix(0);
    require(empty->matchedTokens() == 0 && empty->terminalHash() == 0 && empty->ids().empty(), "empty matched prefix");
    prefix->close(); source->close();
    require(pool->free() == 2, "cached active readers are not free");
    rejects<std::runtime_error>([&] { pool->allocate(3); }, "LRU cannot evict active readers");
    match->close();
    require(pool->free() == 4 && pool->cached() == 2, "unreferenced cached blocks are reclaimable");
    auto replacement = pool->allocate(4);
    require(pool->cached() == 0 && index.lookup(tokens)->matchedTokens() == 0, "recycled prefix no longer matches");
    replacement->close();
    require(pool->free() == 4, "LRU replacement refs released");

    auto lru_pool = std::make_shared<CacheBlockPool>(2);
    CachePrefixIndex lru(lru_pool, 4);
    auto a = lru_pool->allocate(), b = lru_pool->allocate();
    const int a_id = a->ids()[0], b_id = b->ids()[0];
    a->markComputed(4); b->markComputed(4);
    require(lru.publish({1, 2, 3, 4}, *a) && lru.publish({5, 6, 7, 8}, *b), "independent cached prefixes");
    a->close(); b->close();
    lru.lookup({1, 2, 3, 4})->close();
    auto reused = lru_pool->allocate();
    require(reused->ids()[0] == b_id && lru_pool->metadata(a_id).cached, "LRU evicts least recently touched cached block");
}

void saltDuplicateAndUncache() {
    auto pool = std::make_shared<CacheBlockPool>(3);
    CachePrefixIndex index(pool, 4, 1), other_salt(pool, 4, 2);
    auto a = pool->allocate(), b = pool->allocate();
    a->markComputed(4); b->markComputed(4);
    const int id = a->ids()[0];
    require(index.publish({1, 2, 3, 4}, *a), "original prefix");
    require(!index.publish({1, 2, 3, 4}, *b), "duplicate different payload is explicit no-op");
    require(pool->cached() == 1 && pool->metadata(id).ref_count == 1, "duplicate check does not leak retain");
    rejects<std::invalid_argument>([&] { index.publish({5, 6, 7, 8}, *a); }, "cannot rebind published payload");
    require(other_salt.lookup({1, 2, 3, 4})->matchedTokens() == 0, "salt isolates prefix identity");
    require(index.lookup({1, 2, 3, 9})->matchedTokens() == 0, "different tokens miss");
    auto reader = index.lookup({1, 2, 3, 4});
    require(pool->uncache(id), "uncache published block");
    require(!pool->metadata(id).cached && pool->metadata(id).ref_count == 2, "uncache preserves live readers");
    require(index.lookup({1, 2, 3, 4})->matchedTokens() == 0, "uncached prefix misses");
    a->close(); reader->close(); b->close();
    require(pool->free() == 3 && !pool->uncache(id), "uncache and close fully recover refs");
}

void holderAllocationFailures() {
    // Cover lease object, shared_ptr control block, and vector allocation.
    for (long point = 0; point < 3; ++point) {
        auto pool = std::make_shared<CacheBlockPool>(3);
        rejects<std::bad_alloc>([&] { allocation_fault::Scope fault(point); pool->allocate(2); }, "allocate injection must throw");
        require(pool->free() == 3, "failed holder allocation cannot consume blocks");
        for (int id = 0; id < 3; ++id) require(!pool->metadata(id).allocated, "no hidden adopted refs after allocation failure");
        auto owner = pool->allocate(2);
        rejects<std::bad_alloc>([&] { allocation_fault::Scope fault(point); owner->share(); }, "share injection must throw");
        for (int id : owner->ids()) require(pool->metadata(id).ref_count == 1, "failed share cannot leak retain");
    }
    auto pool = std::make_shared<CacheBlockPool>(3);
    auto a = pool->allocate(), b = pool->allocate();
    const auto a_ids = a->ids(), b_ids = b->ids();
    rejects<std::bad_alloc>([&] { allocation_fault::Scope fault(0); a->append(*b); }, "append reserve injection");
    require(a->ids() == a_ids && b->ids() == b_ids && !b->closed(), "failed append preserves both leases");
    for (int id : {a_ids[0], b_ids[0]}) require(pool->metadata(id).ref_count == 1, "failed append refs unchanged");
}

void prefixAllocationFailures() {
    auto pool = std::make_shared<CacheBlockPool>(3);
    auto owner = pool->allocate(2);
    owner->markComputed(4);
    const std::vector<int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8};
    CachePrefixIndex index(pool, 4);
    require(index.publish(tokens, *owner), "injection prefix setup");
    for (long point = 0; point < 3; ++point) {
        rejects<std::bad_alloc>([&] { allocation_fault::Scope fault(point); index.lookup(tokens); }, "lookup holder/vector injection");
        for (int id : owner->ids()) require(pool->metadata(id).ref_count == 1, "failed lookup cannot retain hidden refs");
    }
    auto recovered = index.lookup(tokens);
    require(recovered->matchedTokens() == 8, "lookup recovers after bad_alloc");
    recovered->close();

    // Also exercise BlockPrefixCache directly, so wrapper changes cannot mask
    // a retain-before-vector-growth regression in the shared core matcher.
    BlockManager blocks(3);
    const std::vector<int> ids{blocks.allocate(), blocks.allocate()};
    BlockPrefixCache core_index(blocks, 4);
    require(core_index.insert(tokens.data(), tokens.size(), ids), "core prefix setup");
    rejects<std::invalid_argument>([&] { core_index.match(nullptr, 1); }, "null nonempty lookup");
    require(core_index.match(nullptr, 0).block_ids.empty(), "empty core lookup");
    rejects<std::bad_alloc>([&] {
        allocation_fault::Scope fault(0); core_index.match(tokens.data(), tokens.size());
    }, "core match reserve injection");
    for (int id : ids) require(blocks.metadata(id).ref_count == 1, "core failed match does not leak retained refs");
    PrefixMatch match;
    {
        allocation_fault::Scope fault(1);
        match = core_index.match(tokens.data(), tokens.size());
    }
    require(match.block_ids == ids && match.matched_tokens == 8, "core match never grows after its first retain");
    core_index.release(match);
    for (int id : ids) {
        require(blocks.metadata(id).ref_count == 1, "core release restores refs");
        require(blocks.release(id), "core source release");
    }
}
} // namespace

int main() {
    try {
        allocationAndClose();
        poolLifetimeAndCapacity();
        appendAndReplace();
        validityAndPublish();
        prefixesAndLru();
        saltDuplicateAndUncache();
        holderAllocationFailures();
        prefixAllocationFailures();
        std::cout << "{\"all_passed\":true,\"scope\":\"CPU block lease contracts\",\"checks\":" << checks << "}\n";
    } catch (const std::exception &error) {
        std::cerr << "block lease contract failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

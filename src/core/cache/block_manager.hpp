#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace llaisys::core {

struct BlockMetadata {
    int block_id = -1;
    uint32_t ref_count = 0;
    uint32_t num_tokens = 0;
    bool allocated = false;
    bool computed = false;
    bool cached = false;
    uint64_t block_hash = 0;
    uint64_t last_access = 0;
};

// Physical block-id lifecycle. It is independent of the bytes stored in a
// block, so the same manager can back standard KV and MLA latent caches.
class BlockManager {
public:
    explicit BlockManager(size_t num_blocks);

    int allocate();
    bool retain(int block_id);
    bool release(int block_id);

    bool markComputed(int block_id, uint32_t num_tokens);
    bool cache(int block_id, uint64_t block_hash);
    int findCached(uint64_t block_hash);
    bool uncache(int block_id);
    bool evictOneCached();

    // Blocks immediately allocatable, including unreferenced cached blocks
    // that can be reclaimed by the LRU policy.
    size_t numFree() const;
    size_t numTotal() const { return blocks_.size(); }
    size_t numCached() const;
    const BlockMetadata &metadata(int block_id) const;

private:
    BlockMetadata *mutableMetadata(int block_id);
    void recycle(BlockMetadata &block);
    uint64_t tick();

    std::vector<BlockMetadata> blocks_;
    std::deque<int> free_list_;
    std::unordered_map<uint64_t, int> prefix_index_;
    uint64_t clock_ = 0;
};

} // namespace llaisys::core

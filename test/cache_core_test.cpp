#include "src/core/cache/block_manager.hpp"
#include "src/core/cache/block_prefix_cache.hpp"
#include "src/core/cache/cache_layout.hpp"
#include "src/engine/schedule_plan.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using namespace llaisys::core;

    StandardKVCacheLayout kv(16, 2, 4, 128, 2);
    assert(kv.numComponents() == 2);
    assert(kv.layerBytes(0, 0) == 16U * 4U * 128U * 2U);

    MLACacheLayout mla(16, 2, 512, 64, 2, {1, 2});
    assert(mla.numComponents() == 2);
    assert(mla.layerBytes(0, 0) == 16U * 512U * 2U);
    assert(mla.layerBytes(0, 1) == 8U * 512U * 2U);
    assert(mla.layerBytes(1, 1) == 8U * 64U * 2U);

    BlockManager blocks(4);
    std::vector<int> block_ids;
    for (int i = 0; i < 2; ++i) block_ids.push_back(blocks.allocate());
    assert(blocks.numFree() == 2);

    const std::vector<int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9};
    BlockPrefixCache prefixes(blocks, 4, 1234);
    assert(prefixes.insert(tokens.data(), tokens.size(), block_ids));
    assert(blocks.release(block_ids[0]));
    assert(blocks.release(block_ids[1]));
    assert(blocks.numCached() == 2);

    auto match = prefixes.match(tokens.data(), tokens.size());
    assert(match.matched_tokens == 8);
    assert(match.block_ids == block_ids);
    prefixes.release(match);

    llaisys::engine::SchedulePlan plan;
    plan.step_id = 42;
    plan.prefills.push_back({7, 0, {1, 2}, 0, true, {0.0F, 1, 1.0F}});
    assert(plan.prefills.front().sampling.top_k == 1);

    std::cout << "cache core tests passed\n";
    return 0;
}

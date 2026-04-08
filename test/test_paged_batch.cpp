#include "src/core/allocator/block_allocator.hpp"
#include "src/core/page_table.hpp"
#include "src/ops/self_attention/paged_attention.hpp"
#include "llaisys/runtime.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>
#include <cassert>

#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << std::endl; \
    std::exit(1); \
} } while(0)

using namespace llaisys::core;

static const LlaisysRuntimeAPI *cpu_api() {
    return llaisysGetRuntimeAPI(LLAISYS_DEVICE_CPU);
}

// Simulates the full batch decode flow: KV write + paged attention
static void test_batch_decode_simulation() {
    int block_size = 4;
    int nlayer = 2;
    int num_kv_heads = 2;
    int num_heads = 4;
    int head_dim = 8;
    float scale = 1.0f / std::sqrt((float)head_dim);

    BlockAllocatorConfig cfg = {128, (size_t)block_size, (size_t)nlayer,
                                 (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)};
    BlockAllocator alloc(cfg, cpu_api());

    int batch_size = 3;
    std::vector<PageTable> pts(batch_size, PageTable(block_size));
    std::vector<int> seq_positions(batch_size, 0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Simulate 10 decode steps
    for (int step = 0; step < 10; ++step) {
        // Allocate blocks if needed (before layer loop, like batch_decode_impl)
        for (int b = 0; b < batch_size; ++b) {
            if (pts[b].needs_new_block()) {
                int bid = alloc.alloc();
                CHECK(bid >= 0);
                pts[b].append_block(bid);
            }
        }

        // Build block tables and seq_lens
        int max_blocks = 0;
        for (int b = 0; b < batch_size; ++b)
            max_blocks = std::max(max_blocks, pts[b].num_blocks());

        std::vector<int> block_tables(batch_size * max_blocks, 0);
        std::vector<int> seq_lens(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            seq_lens[b] = seq_positions[b] + 1;
            for (int j = 0; j < pts[b].num_blocks(); ++j)
                block_tables[b * max_blocks + j] = pts[b].block_ids()[j];
        }

        // Generate Q/K/V for this step
        std::vector<float> query(batch_size * num_heads * head_dim);
        std::vector<float> key(batch_size * num_kv_heads * head_dim);
        std::vector<float> value(batch_size * num_kv_heads * head_dim);
        for (auto &v : query) v = dist(rng);
        for (auto &v : key) v = dist(rng);
        for (auto &v : value) v = dist(rng);

        for (int layer = 0; layer < nlayer; ++layer) {
            // Write KV to block pool
            for (int b = 0; b < batch_size; ++b) {
                int pos = seq_positions[b];
                int bid = pts[b].get_block_for_token(pos);
                int off = pts[b].get_offset_in_block(pos);

                float *k_dst = (float *)alloc.get_k_ptr(bid, layer)
                               + off * num_kv_heads * head_dim;
                float *v_dst = (float *)alloc.get_v_ptr(bid, layer)
                               + off * num_kv_heads * head_dim;
                float *k_src = key.data() + b * num_kv_heads * head_dim;
                float *v_src = value.data() + b * num_kv_heads * head_dim;
                std::memcpy(k_dst, k_src, num_kv_heads * head_dim * sizeof(float));
                std::memcpy(v_dst, v_src, num_kv_heads * head_dim * sizeof(float));
            }

            // Paged attention
            std::vector<float> output(batch_size * num_heads * head_dim, 0.0f);
            llaisys::ops::paged_attention(
                output.data(), query.data(),
                alloc.pool_k_raw(), alloc.pool_v_raw(),
                block_tables.data(), seq_lens.data(),
                batch_size, num_heads, num_kv_heads, head_dim,
                block_size, max_blocks,
                alloc.block_stride(), alloc.layer_stride(),
                layer, scale, LLAISYS_DEVICE_CPU);

            // Verify output is finite
            for (size_t i = 0; i < output.size(); ++i) {
                CHECK(std::isfinite(output[i]));
            }
        }

        // Update positions
        for (int b = 0; b < batch_size; ++b) {
            pts[b].inc_num_tokens();
            seq_positions[b]++;
        }
    }

    // Verify block allocator accounting
    size_t total_blocks_used = 0;
    for (int b = 0; b < batch_size; ++b)
        total_blocks_used += pts[b].num_blocks();
    CHECK(alloc.num_free() == alloc.num_total() - total_blocks_used);

    // Release all
    for (int b = 0; b < batch_size; ++b)
        pts[b].release_all(alloc);
    CHECK(alloc.num_free() == alloc.num_total());

    std::cout << "  test_batch_decode_simulation PASSED" << std::endl;
}

// Test slot save/restore simulation
static void test_slot_save_restore() {
    int block_size = 4;
    int nlayer = 2;
    int num_kv_heads = 1;
    int head_dim = 4;
    int seq_len = 6;

    BlockAllocatorConfig cfg = {32, (size_t)block_size, (size_t)nlayer,
                                 (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)};
    BlockAllocator alloc(cfg, cpu_api());
    PageTable pt(block_size);

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t kv_row_bytes = num_kv_heads * head_dim * sizeof(float);

    // Fill KV data
    for (int t = 0; t < seq_len; ++t) {
        if (pt.needs_new_block()) {
            int bid = alloc.alloc();
            CHECK(bid >= 0);
            pt.append_block(bid);
        }
        int bid = pt.get_block_for_token(t);
        int off = pt.get_offset_in_block(t);
        for (int layer = 0; layer < nlayer; ++layer) {
            float *k_ptr = (float *)alloc.get_k_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            float *v_ptr = (float *)alloc.get_v_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            for (int j = 0; j < num_kv_heads * head_dim; ++j) {
                k_ptr[j] = dist(rng);
                v_ptr[j] = dist(rng);
            }
        }
        pt.inc_num_tokens();
    }

    // Save: linearize paged data to contiguous
    std::vector<std::vector<uint8_t>> saved_buffers(nlayer * 2);
    size_t total_bytes = seq_len * kv_row_bytes;
    for (auto &b : saved_buffers) b.resize(total_bytes);

    for (int t = 0; t < seq_len; ++t) {
        int bid = pt.get_block_for_token(t);
        int off = pt.get_offset_in_block(t);
        for (int layer = 0; layer < nlayer; ++layer) {
            float *k_src = (float *)alloc.get_k_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            float *v_src = (float *)alloc.get_v_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            std::memcpy(saved_buffers[layer * 2].data() + t * kv_row_bytes, k_src, kv_row_bytes);
            std::memcpy(saved_buffers[layer * 2 + 1].data() + t * kv_row_bytes, v_src, kv_row_bytes);
        }
    }

    // Release original
    int original_blocks = pt.num_blocks();
    pt.release_all(alloc);
    CHECK(pt.num_tokens() == 0);
    CHECK(pt.num_blocks() == 0);

    // Restore to a new page table
    PageTable pt2(block_size);
    for (int t = 0; t < seq_len; ++t) {
        if (pt2.needs_new_block()) {
            int bid = alloc.alloc();
            CHECK(bid >= 0);
            pt2.append_block(bid);
        }
        int bid = pt2.get_block_for_token(t);
        int off = pt2.get_offset_in_block(t);
        for (int layer = 0; layer < nlayer; ++layer) {
            float *k_dst = (float *)alloc.get_k_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            float *v_dst = (float *)alloc.get_v_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            std::memcpy(k_dst, saved_buffers[layer * 2].data() + t * kv_row_bytes, kv_row_bytes);
            std::memcpy(v_dst, saved_buffers[layer * 2 + 1].data() + t * kv_row_bytes, kv_row_bytes);
        }
        pt2.inc_num_tokens();
    }

    CHECK(pt2.num_tokens() == seq_len);
    CHECK(pt2.num_blocks() == original_blocks);

    // Verify data matches original
    for (int t = 0; t < seq_len; ++t) {
        int bid = pt2.get_block_for_token(t);
        int off = pt2.get_offset_in_block(t);
        for (int layer = 0; layer < nlayer; ++layer) {
            float *k_ptr = (float *)alloc.get_k_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            float *v_ptr = (float *)alloc.get_v_ptr(bid, layer)
                           + off * num_kv_heads * head_dim;
            const float *k_ref = (const float *)(saved_buffers[layer * 2].data() + t * kv_row_bytes);
            const float *v_ref = (const float *)(saved_buffers[layer * 2 + 1].data() + t * kv_row_bytes);
            for (int j = 0; j < num_kv_heads * head_dim; ++j) {
                CHECK(k_ptr[j] == k_ref[j]);
                CHECK(v_ptr[j] == v_ref[j]);
            }
        }
    }

    pt2.release_all(alloc);
    CHECK(alloc.num_free() == alloc.num_total());

    std::cout << "  test_slot_save_restore PASSED" << std::endl;
}

// Stress test: many sequences with varying lengths
static void test_stress_many_sequences() {
    int block_size = 16;
    int nlayer = 1;
    int num_kv_heads = 1;
    int head_dim = 4;
    int num_heads = 1;
    int num_sequences = 32;
    int max_steps = 50;
    float scale = 1.0f / std::sqrt((float)head_dim);

    size_t total_blocks_needed = num_sequences * ((max_steps + block_size - 1) / block_size + 1);
    BlockAllocatorConfig cfg = {total_blocks_needed, (size_t)block_size, (size_t)nlayer,
                                 (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)};
    BlockAllocator alloc(cfg, cpu_api());

    std::vector<PageTable> pts(num_sequences, PageTable(block_size));
    std::vector<int> seq_positions(num_sequences, 0);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::uniform_int_distribution<int> step_dist(5, max_steps);

    // Each sequence gets a random number of steps
    std::vector<int> total_steps(num_sequences);
    for (int i = 0; i < num_sequences; ++i)
        total_steps[i] = step_dist(rng);

    for (int step = 0; step < max_steps; ++step) {
        // Find active sequences for this step
        std::vector<int> active;
        for (int b = 0; b < num_sequences; ++b) {
            if (seq_positions[b] < total_steps[b])
                active.push_back(b);
        }
        if (active.empty()) break;

        int B = active.size();

        // Allocate blocks
        for (int idx = 0; idx < B; ++idx) {
            int b = active[idx];
            if (pts[b].needs_new_block()) {
                int bid = alloc.alloc();
                CHECK(bid >= 0);
                pts[b].append_block(bid);
            }
        }

        // Build batch structures
        int max_blocks = 0;
        for (int idx = 0; idx < B; ++idx)
            max_blocks = std::max(max_blocks, pts[active[idx]].num_blocks());

        std::vector<int> block_tables(B * max_blocks, 0);
        std::vector<int> seq_lens(B);
        for (int idx = 0; idx < B; ++idx) {
            int b = active[idx];
            seq_lens[idx] = seq_positions[b] + 1;
            for (int j = 0; j < pts[b].num_blocks(); ++j)
                block_tables[idx * max_blocks + j] = pts[b].block_ids()[j];
        }

        // Generate data and write KV
        std::vector<float> query(B * num_heads * head_dim);
        for (auto &v : query) v = dist(rng);

        for (int idx = 0; idx < B; ++idx) {
            int b = active[idx];
            int pos = seq_positions[b];
            int bid = pts[b].get_block_for_token(pos);
            int off = pts[b].get_offset_in_block(pos);
            float *k_dst = (float *)alloc.get_k_ptr(bid, 0)
                           + off * num_kv_heads * head_dim;
            float *v_dst = (float *)alloc.get_v_ptr(bid, 0)
                           + off * num_kv_heads * head_dim;
            for (int j = 0; j < num_kv_heads * head_dim; ++j) {
                k_dst[j] = dist(rng);
                v_dst[j] = dist(rng);
            }
        }

        // Paged attention
        std::vector<float> output(B * num_heads * head_dim, 0.0f);
        llaisys::ops::paged_attention(
            output.data(), query.data(),
            alloc.pool_k_raw(), alloc.pool_v_raw(),
            block_tables.data(), seq_lens.data(),
            B, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks,
            alloc.block_stride(), alloc.layer_stride(),
            0, scale, LLAISYS_DEVICE_CPU);

        for (size_t i = 0; i < output.size(); ++i)
            CHECK(std::isfinite(output[i]));

        // Update positions
        for (int idx = 0; idx < B; ++idx) {
            int b = active[idx];
            pts[b].inc_num_tokens();
            seq_positions[b]++;
        }
    }

    // Release all
    for (int b = 0; b < num_sequences; ++b)
        pts[b].release_all(alloc);
    CHECK(alloc.num_free() == alloc.num_total());

    std::cout << "  test_stress_many_sequences PASSED (32 seqs, up to 50 steps)" << std::endl;
}

int main() {
    std::cout << "=== Phase 3: Paged Batch Decode Tests ===" << std::endl;
    test_batch_decode_simulation();
    test_slot_save_restore();
    test_stress_many_sequences();
    std::cout << "\n✓ All Phase 3 tests passed!" << std::endl;
    return 0;
}

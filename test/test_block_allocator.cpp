#include "src/core/allocator/block_allocator.hpp"
#include "src/core/page_table.hpp"
#include "llaisys/runtime.h"

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << std::endl; \
    std::exit(1); \
} } while(0)

using namespace llaisys::core;

static const LlaisysRuntimeAPI *cpu_api() {
    return llaisysGetRuntimeAPI(LLAISYS_DEVICE_CPU);
}

static BlockAllocatorConfig make_config(size_t num_blocks, size_t block_size,
                                         size_t nlayer, size_t nkvh, size_t dh) {
    return {num_blocks, block_size, nlayer, nkvh, dh, sizeof(float)};
}

// --- BlockAllocator tests ---

static void test_basic_alloc_free() {
    auto cfg = make_config(10, 16, 2, 2, 64);
    BlockAllocator alloc(cfg, cpu_api());

    CHECK(alloc.num_free() == 10);
    CHECK(alloc.num_total() == 10);

    int b0 = alloc.alloc();
    CHECK(b0 >= 0 && b0 < 10);
    CHECK(alloc.num_free() == 9);

    int b1 = alloc.alloc();
    CHECK(b1 >= 0 && b1 < 10);
    CHECK(b0 != b1);
    CHECK(alloc.num_free() == 8);

    alloc.free(b0);
    CHECK(alloc.num_free() == 9);

    alloc.free(b1);
    CHECK(alloc.num_free() == 10);

    std::cout << "  test_basic_alloc_free PASSED" << std::endl;
}

static void test_exhaust_pool() {
    auto cfg = make_config(5, 16, 2, 2, 64);
    BlockAllocator alloc(cfg, cpu_api());

    std::vector<int> blocks;
    for (int i = 0; i < 5; ++i) {
        int b = alloc.alloc();
        CHECK(b >= 0);
        blocks.push_back(b);
    }
    CHECK(alloc.num_free() == 0);

    int overflow = alloc.alloc();
    CHECK(overflow == -1);

    alloc.free(blocks[2]);
    CHECK(alloc.num_free() == 1);

    int reused = alloc.alloc();
    CHECK(reused == blocks[2]);
    CHECK(alloc.num_free() == 0);

    for (int b : blocks) alloc.free(b);
    alloc.free(reused);

    std::cout << "  test_exhaust_pool PASSED" << std::endl;
}

static void test_unique_ids() {
    auto cfg = make_config(100, 16, 2, 2, 64);
    BlockAllocator alloc(cfg, cpu_api());

    std::set<int> ids;
    for (int i = 0; i < 100; ++i) {
        int b = alloc.alloc();
        CHECK(b >= 0);
        CHECK(ids.find(b) == ids.end());
        ids.insert(b);
    }
    CHECK(alloc.num_free() == 0);

    for (int b : ids) alloc.free(b);
    CHECK(alloc.num_free() == 100);

    std::cout << "  test_unique_ids PASSED" << std::endl;
}

static void test_pointer_arithmetic() {
    size_t block_size = 16, nlayer = 3, nkvh = 2, dh = 64;
    auto cfg = make_config(4, block_size, nlayer, nkvh, dh);
    BlockAllocator alloc(cfg, cpu_api());

    size_t layer_stride = block_size * nkvh * dh * sizeof(float);
    size_t block_stride = nlayer * layer_stride;

    int b0 = alloc.alloc();
    int b1 = alloc.alloc();

    auto *base_k = static_cast<std::byte *>(alloc.pool_k_raw());
    auto *base_v = static_cast<std::byte *>(alloc.pool_v_raw());

    for (int layer = 0; layer < (int)nlayer; ++layer) {
        auto *k_ptr = static_cast<std::byte *>(alloc.get_k_ptr(b0, layer));
        auto *v_ptr = static_cast<std::byte *>(alloc.get_v_ptr(b0, layer));
        CHECK(k_ptr == base_k + b0 * block_stride + layer * layer_stride);
        CHECK(v_ptr == base_v + b0 * block_stride + layer * layer_stride);
    }

    auto *k1_l0 = static_cast<std::byte *>(alloc.get_k_ptr(b1, 0));
    CHECK(k1_l0 == base_k + b1 * block_stride);

    alloc.free(b0);
    alloc.free(b1);

    std::cout << "  test_pointer_arithmetic PASSED" << std::endl;
}

static void test_write_read_memory() {
    size_t block_size = 4, nlayer = 1, nkvh = 1, dh = 2;
    auto cfg = make_config(2, block_size, nlayer, nkvh, dh);
    BlockAllocator alloc(cfg, cpu_api());

    int b = alloc.alloc();
    float *k_ptr = static_cast<float *>(alloc.get_k_ptr(b, 0));
    float *v_ptr = static_cast<float *>(alloc.get_v_ptr(b, 0));

    size_t n = block_size * nkvh * dh;
    for (size_t i = 0; i < n; ++i) {
        k_ptr[i] = static_cast<float>(i + 1);
        v_ptr[i] = static_cast<float>(i + 100);
    }

    for (size_t i = 0; i < n; ++i) {
        CHECK(k_ptr[i] == static_cast<float>(i + 1));
        CHECK(v_ptr[i] == static_cast<float>(i + 100));
    }

    alloc.free(b);
    std::cout << "  test_write_read_memory PASSED" << std::endl;
}

// --- PageTable tests ---

static void test_page_table_basic() {
    PageTable pt(16);
    CHECK(pt.num_tokens() == 0);
    CHECK(pt.num_blocks() == 0);
    CHECK(pt.needs_new_block());

    pt.append_block(5);
    CHECK(pt.num_blocks() == 1);

    CHECK(pt.get_block_for_token(0) == 5);
    CHECK(pt.get_block_for_token(15) == 5);
    CHECK(pt.get_offset_in_block(0) == 0);
    CHECK(pt.get_offset_in_block(15) == 15);
    CHECK(pt.get_block_for_token(16) == -1);

    pt.append_block(3);
    CHECK(pt.get_block_for_token(16) == 3);
    CHECK(pt.get_block_for_token(31) == 3);
    CHECK(pt.get_offset_in_block(16) == 0);
    CHECK(pt.get_offset_in_block(31) == 15);

    std::cout << "  test_page_table_basic PASSED" << std::endl;
}

static void test_page_table_needs_new_block() {
    PageTable pt(4);

    CHECK(pt.needs_new_block());
    pt.append_block(0);

    for (int i = 0; i < 4; ++i) {
        pt.set_num_tokens(i);
        if (i == 0) {
            CHECK(pt.needs_new_block());
        } else {
            CHECK(!pt.needs_new_block());
        }
    }
    pt.set_num_tokens(4);
    CHECK(pt.needs_new_block());

    std::cout << "  test_page_table_needs_new_block PASSED" << std::endl;
}

static void test_page_table_release_all() {
    auto cfg = make_config(10, 16, 2, 2, 64);
    BlockAllocator alloc(cfg, cpu_api());

    PageTable pt(16);
    int b0 = alloc.alloc();
    int b1 = alloc.alloc();
    int b2 = alloc.alloc();
    pt.append_block(b0);
    pt.append_block(b1);
    pt.append_block(b2);
    pt.set_num_tokens(40);

    CHECK(alloc.num_free() == 7);
    CHECK(pt.num_blocks() == 3);

    pt.release_all(alloc);

    CHECK(alloc.num_free() == 10);
    CHECK(pt.num_blocks() == 0);
    CHECK(pt.num_tokens() == 0);

    std::cout << "  test_page_table_release_all PASSED" << std::endl;
}

static void test_tp_local_nkvh() {
    size_t nkvh_full = 8, tp_size = 4;
    size_t local_nkvh = nkvh_full / tp_size;  // 2

    auto cfg = make_config(4, 16, 2, local_nkvh, 64);
    BlockAllocator alloc(cfg, cpu_api());

    size_t expected_layer_bytes = 16 * local_nkvh * 64 * sizeof(float);
    CHECK(alloc.layer_stride() == expected_layer_bytes);

    int b = alloc.alloc();
    auto *p0 = static_cast<std::byte *>(alloc.get_k_ptr(b, 0));
    auto *p1 = static_cast<std::byte *>(alloc.get_k_ptr(b, 1));
    CHECK(static_cast<size_t>(p1 - p0) == expected_layer_bytes);

    alloc.free(b);
    std::cout << "  test_tp_local_nkvh PASSED" << std::endl;
}

// --- Integrated scenario ---

static void test_integrated_scenario() {
    size_t block_size = 4, nlayer = 2, nkvh = 2, dh = 4;
    auto cfg = make_config(8, block_size, nlayer, nkvh, dh);
    BlockAllocator alloc(cfg, cpu_api());

    // Simulate two sequences
    PageTable pt_a(block_size), pt_b(block_size);

    // Sequence A: 10 tokens → needs ceil(10/4) = 3 blocks
    for (int t = 0; t < 10; ++t) {
        if (pt_a.needs_new_block()) {
            int b = alloc.alloc();
            CHECK(b >= 0);
            pt_a.append_block(b);
        }
        // Write token KV for each layer
        int bid = pt_a.get_block_for_token(t);
        int off = pt_a.get_offset_in_block(t);
        for (int l = 0; l < (int)nlayer; ++l) {
            float *kp = static_cast<float *>(alloc.get_k_ptr(bid, l));
            float *vp = static_cast<float *>(alloc.get_v_ptr(bid, l));
            size_t base = off * nkvh * dh;
            for (size_t j = 0; j < nkvh * dh; ++j) {
                kp[base + j] = static_cast<float>(t * 100 + l * 10 + j);
                vp[base + j] = static_cast<float>(t * 100 + l * 10 + j + 1000);
            }
        }
        pt_a.inc_num_tokens();
    }
    CHECK(pt_a.num_blocks() == 3);
    CHECK(pt_a.num_tokens() == 10);
    CHECK(alloc.num_free() == 5);

    // Sequence B: 5 tokens → 2 blocks
    for (int t = 0; t < 5; ++t) {
        if (pt_b.needs_new_block()) {
            int b = alloc.alloc();
            CHECK(b >= 0);
            pt_b.append_block(b);
        }
        pt_b.inc_num_tokens();
    }
    CHECK(pt_b.num_blocks() == 2);
    CHECK(alloc.num_free() == 3);

    // Verify sequence A data integrity
    for (int t = 0; t < 10; ++t) {
        int bid = pt_a.get_block_for_token(t);
        int off = pt_a.get_offset_in_block(t);
        for (int l = 0; l < (int)nlayer; ++l) {
            float *kp = static_cast<float *>(alloc.get_k_ptr(bid, l));
            size_t base = off * nkvh * dh;
            for (size_t j = 0; j < nkvh * dh; ++j) {
                float expected = static_cast<float>(t * 100 + l * 10 + j);
                CHECK(kp[base + j] == expected);
            }
        }
    }

    // Release sequence A
    pt_a.release_all(alloc);
    CHECK(alloc.num_free() == 6);

    // Release sequence B
    pt_b.release_all(alloc);
    CHECK(alloc.num_free() == 8);

    std::cout << "  test_integrated_scenario PASSED" << std::endl;
}

int main() {
    std::cout << "=== BlockAllocator Tests ===" << std::endl;
    test_basic_alloc_free();
    test_exhaust_pool();
    test_unique_ids();
    test_pointer_arithmetic();
    test_write_read_memory();

    std::cout << "\n=== PageTable Tests ===" << std::endl;
    test_page_table_basic();
    test_page_table_needs_new_block();
    test_page_table_release_all();

    std::cout << "\n=== TP Compatibility Tests ===" << std::endl;
    test_tp_local_nkvh();

    std::cout << "\n=== Integrated Scenario ===" << std::endl;
    test_integrated_scenario();

    std::cout << "\n✓ All Phase 1 tests passed!" << std::endl;
    return 0;
}

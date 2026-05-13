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

#ifdef ENABLE_NVIDIA_API
#include <cuda_fp16.h>
#endif

#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << std::endl; \
    std::exit(1); \
} } while(0)

using namespace llaisys::core;

static const LlaisysRuntimeAPI *cpu_api() {
    return llaisysGetRuntimeAPI(LLAISYS_DEVICE_CPU);
}

static bool gpu_available() {
#ifdef ENABLE_NVIDIA_API
    auto *api = llaisysGetRuntimeAPI(LLAISYS_DEVICE_NVIDIA);
    if (api && api->get_device_count && api->get_device_count() > 0) return true;
#endif
    return false;
}

// Reference: standard contiguous-KV attention with online softmax
// Q: [batch, nh, dh], K: [total_len, nkvh, dh], V: [total_len, nkvh, dh]
static void reference_attention(
    float *output,
    const float *query,
    const float *k_data,
    const float *v_data,
    int batch_idx, int seq_len,
    int num_heads, int num_kv_heads, int head_dim,
    float scale)
{
    int group_size = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_size;
        float m = -1e30f;
        float l = 0.0f;
        std::vector<float> acc(head_dim, 0.0f);

        for (int t = 0; t < seq_len; ++t) {
            const float *q_vec = query + batch_idx * num_heads * head_dim + h * head_dim;
            const float *k_vec = k_data + t * num_kv_heads * head_dim + kv_h * head_dim;
            const float *v_vec = v_data + t * num_kv_heads * head_dim + kv_h * head_dim;

            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d)
                score += q_vec[d] * k_vec[d];
            score *= scale;

            float m_new = std::max(m, score);
            float p = std::exp(score - m_new);
            float correction = std::exp(m - m_new);
            l = correction * l + p;
            for (int d = 0; d < head_dim; ++d)
                acc[d] = correction * acc[d] + p * v_vec[d];
            m = m_new;
        }

        float *out_vec = output + batch_idx * num_heads * head_dim + h * head_dim;
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int d = 0; d < head_dim; ++d)
            out_vec[d] = acc[d] * inv_l;
    }
}

static float max_abs_error(const float *a, const float *b, size_t n) {
    float err = 0.0f;
    for (size_t i = 0; i < n; ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

// Fill KV data into block pool via page table, return contiguous copy for reference
static void fill_kv_data(
    BlockAllocator &alloc, PageTable &pt,
    std::vector<float> &contiguous_k, std::vector<float> &contiguous_v,
    int seq_len, int num_kv_heads, int head_dim, int block_size,
    int layer_idx, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    contiguous_k.resize(seq_len * num_kv_heads * head_dim);
    contiguous_v.resize(seq_len * num_kv_heads * head_dim);

    for (int t = 0; t < seq_len; ++t) {
        if (pt.needs_new_block()) {
            int bid = alloc.alloc();
            CHECK(bid >= 0);
            pt.append_block(bid);
        }

        int bid = pt.get_block_for_token(t);
        int off = pt.get_offset_in_block(t);

        float *k_ptr = static_cast<float *>(alloc.get_k_ptr(bid, layer_idx));
        float *v_ptr = static_cast<float *>(alloc.get_v_ptr(bid, layer_idx));

        size_t base = off * num_kv_heads * head_dim;
        for (int j = 0; j < num_kv_heads * head_dim; ++j) {
            float kv = dist(rng);
            float vv = dist(rng);
            k_ptr[base + j] = kv;
            v_ptr[base + j] = vv;
            contiguous_k[t * num_kv_heads * head_dim + j] = kv;
            contiguous_v[t * num_kv_heads * head_dim + j] = vv;
        }
        pt.inc_num_tokens();
    }
}

static void test_single_sequence() {
    int block_size = 4, nlayer = 1, num_kv_heads = 2, head_dim = 8;
    int num_heads = 4;  // GQA: group_size = 2
    int seq_len = 10;
    int layer_idx = 0;
    float scale = 1.0f / std::sqrt((float)head_dim);

    BlockAllocatorConfig cfg = {32, (size_t)block_size, (size_t)nlayer,
                                 (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)};
    BlockAllocator alloc(cfg, cpu_api());
    PageTable pt(block_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fill KV data
    std::vector<float> contig_k, contig_v;
    fill_kv_data(alloc, pt, contig_k, contig_v,
                 seq_len, num_kv_heads, head_dim, block_size, layer_idx, rng);

    // Generate query [1, nh, dh]
    int batch_size = 1;
    std::vector<float> query(batch_size * num_heads * head_dim);
    for (auto &v : query) v = dist(rng);

    // Build block table [1, max_blocks]
    int max_blocks = pt.num_blocks();
    std::vector<int> block_tables(batch_size * max_blocks);
    for (int i = 0; i < max_blocks; ++i)
        block_tables[i] = pt.block_ids()[i];
    std::vector<int> seq_lens = {seq_len};

    // Paged attention output
    std::vector<float> paged_out(batch_size * num_heads * head_dim, 0.0f);
    llaisys::ops::paged_attention(
        paged_out.data(), query.data(),
        alloc.pool_k_raw(), alloc.pool_v_raw(),
        block_tables.data(), seq_lens.data(),
        batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks,
        cfg.nlayer * cfg.block_size * cfg.nkvh * cfg.dh * cfg.elem_size,
        cfg.block_size * cfg.nkvh * cfg.dh * cfg.elem_size,
        layer_idx, scale, LLAISYS_DEVICE_CPU);

    // Reference attention output
    std::vector<float> ref_out(batch_size * num_heads * head_dim, 0.0f);
    reference_attention(ref_out.data(), query.data(),
                        contig_k.data(), contig_v.data(),
                        0, seq_len, num_heads, num_kv_heads, head_dim, scale);

    float err = max_abs_error(paged_out.data(), ref_out.data(),
                              batch_size * num_heads * head_dim);
    CHECK(err < 1e-5f);

    pt.release_all(alloc);
    std::cout << "  test_single_sequence PASSED (max_err=" << err << ")" << std::endl;
}

static void test_batch_sequences() {
    int block_size = 8, nlayer = 2, num_kv_heads = 2, head_dim = 16;
    int num_heads = 4;
    int layer_idx = 1;
    float scale = 1.0f / std::sqrt((float)head_dim);
    int batch_size = 3;
    int seq_lens_arr[] = {5, 12, 20};

    BlockAllocatorConfig cfg = {64, (size_t)block_size, (size_t)nlayer,
                                 (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)};
    BlockAllocator alloc(cfg, cpu_api());

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Per-sequence page tables and contiguous KV for reference
    std::vector<PageTable> pts;
    std::vector<std::vector<float>> all_contig_k(batch_size), all_contig_v(batch_size);

    int max_blocks = 0;
    for (int b = 0; b < batch_size; ++b) {
        pts.emplace_back(block_size);
        fill_kv_data(alloc, pts[b], all_contig_k[b], all_contig_v[b],
                     seq_lens_arr[b], num_kv_heads, head_dim, block_size, layer_idx, rng);
        max_blocks = std::max(max_blocks, pts[b].num_blocks());
    }

    // Build batch block table [batch_size, max_blocks]
    std::vector<int> block_tables(batch_size * max_blocks, 0);
    std::vector<int> seq_lens_vec(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        seq_lens_vec[b] = seq_lens_arr[b];
        for (int i = 0; i < pts[b].num_blocks(); ++i)
            block_tables[b * max_blocks + i] = pts[b].block_ids()[i];
    }

    // Generate query [B, nh, dh]
    std::vector<float> query(batch_size * num_heads * head_dim);
    for (auto &v : query) v = dist(rng);

    // Paged attention
    size_t pool_block_stride = nlayer * block_size * num_kv_heads * head_dim * sizeof(float);
    size_t pool_layer_stride = block_size * num_kv_heads * head_dim * sizeof(float);

    std::vector<float> paged_out(batch_size * num_heads * head_dim, 0.0f);
    llaisys::ops::paged_attention(
        paged_out.data(), query.data(),
        alloc.pool_k_raw(), alloc.pool_v_raw(),
        block_tables.data(), seq_lens_vec.data(),
        batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks,
        pool_block_stride, pool_layer_stride,
        layer_idx, scale, LLAISYS_DEVICE_CPU);

    // Reference per-sequence
    std::vector<float> ref_out(batch_size * num_heads * head_dim, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        reference_attention(ref_out.data(), query.data(),
                            all_contig_k[b].data(), all_contig_v[b].data(),
                            b, seq_lens_arr[b], num_heads, num_kv_heads, head_dim, scale);
    }

    float err = max_abs_error(paged_out.data(), ref_out.data(),
                              batch_size * num_heads * head_dim);
    CHECK(err < 1e-5f);

    for (auto &pt : pts) pt.release_all(alloc);
    std::cout << "  test_batch_sequences PASSED (max_err=" << err << ")" << std::endl;
}

static void test_non_aligned_seq_len() {
    int block_size = 16, nlayer = 1, num_kv_heads = 1, head_dim = 4;
    int num_heads = 1;
    int seq_len = 7;  // Not a multiple of block_size
    int layer_idx = 0;
    float scale = 1.0f / std::sqrt((float)head_dim);

    BlockAllocatorConfig cfg = {16, (size_t)block_size, (size_t)nlayer,
                                 (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)};
    BlockAllocator alloc(cfg, cpu_api());
    PageTable pt(block_size);

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> contig_k, contig_v;
    fill_kv_data(alloc, pt, contig_k, contig_v,
                 seq_len, num_kv_heads, head_dim, block_size, layer_idx, rng);

    CHECK(pt.num_blocks() == 1);

    std::vector<float> query(num_heads * head_dim);
    for (auto &v : query) v = dist(rng);

    int max_blocks = pt.num_blocks();
    std::vector<int> block_tables(max_blocks);
    for (int i = 0; i < max_blocks; ++i)
        block_tables[i] = pt.block_ids()[i];
    std::vector<int> seq_lens = {seq_len};

    size_t pool_block_stride = nlayer * block_size * num_kv_heads * head_dim * sizeof(float);
    size_t pool_layer_stride = block_size * num_kv_heads * head_dim * sizeof(float);

    std::vector<float> paged_out(num_heads * head_dim, 0.0f);
    llaisys::ops::paged_attention(
        paged_out.data(), query.data(),
        alloc.pool_k_raw(), alloc.pool_v_raw(),
        block_tables.data(), seq_lens.data(),
        1, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks,
        pool_block_stride, pool_layer_stride,
        layer_idx, scale, LLAISYS_DEVICE_CPU);

    std::vector<float> ref_out(num_heads * head_dim, 0.0f);
    reference_attention(ref_out.data(), query.data(),
                        contig_k.data(), contig_v.data(),
                        0, seq_len, num_heads, num_kv_heads, head_dim, scale);

    float err = max_abs_error(paged_out.data(), ref_out.data(), num_heads * head_dim);
    CHECK(err < 1e-5f);

    pt.release_all(alloc);
    std::cout << "  test_non_aligned_seq_len PASSED (max_err=" << err << ")" << std::endl;
}

static void test_larger_dims() {
    int block_size = 16, nlayer = 4, num_kv_heads = 2, head_dim = 128;
    int num_heads = 12;
    int seq_len = 100;
    int layer_idx = 2;
    float scale = 1.0f / std::sqrt((float)head_dim);

    BlockAllocatorConfig cfg = {128, (size_t)block_size, (size_t)nlayer,
                                 (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)};
    BlockAllocator alloc(cfg, cpu_api());
    PageTable pt(block_size);

    std::mt19937 rng(7777);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> contig_k, contig_v;
    fill_kv_data(alloc, pt, contig_k, contig_v,
                 seq_len, num_kv_heads, head_dim, block_size, layer_idx, rng);

    std::vector<float> query(num_heads * head_dim);
    for (auto &v : query) v = dist(rng);

    int max_blocks = pt.num_blocks();
    std::vector<int> block_tables(max_blocks);
    for (int i = 0; i < max_blocks; ++i)
        block_tables[i] = pt.block_ids()[i];
    std::vector<int> seq_lens = {seq_len};

    size_t pool_block_stride = nlayer * block_size * num_kv_heads * head_dim * sizeof(float);
    size_t pool_layer_stride = block_size * num_kv_heads * head_dim * sizeof(float);

    std::vector<float> paged_out(num_heads * head_dim, 0.0f);
    llaisys::ops::paged_attention(
        paged_out.data(), query.data(),
        alloc.pool_k_raw(), alloc.pool_v_raw(),
        block_tables.data(), seq_lens.data(),
        1, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks,
        pool_block_stride, pool_layer_stride,
        layer_idx, scale, LLAISYS_DEVICE_CPU);

    std::vector<float> ref_out(num_heads * head_dim, 0.0f);
    reference_attention(ref_out.data(), query.data(),
                        contig_k.data(), contig_v.data(),
                        0, seq_len, num_heads, num_kv_heads, head_dim, scale);

    float err = max_abs_error(paged_out.data(), ref_out.data(), num_heads * head_dim);
    CHECK(err < 1e-4f);

    pt.release_all(alloc);
    std::cout << "  test_larger_dims PASSED (max_err=" << err << ")" << std::endl;
}

static void test_gpu_f16_paged_attention() {
#ifdef ENABLE_NVIDIA_API
    if (!gpu_available()) {
        std::cout << "  test_gpu_f16_paged_attention SKIPPED (no NVIDIA GPU)" << std::endl;
        return;
    }

    auto *api = llaisysGetRuntimeAPI(LLAISYS_DEVICE_NVIDIA);
    int block_size = 16, nlayer = 1, num_kv_heads = 2, head_dim = 128;
    int num_heads = 8, batch_size = 2, layer_idx = 0;
    int seq_lens_arr[] = {31, 53};
    size_t kv_row_elems = (size_t)num_kv_heads * head_dim;
    size_t kv_row_bytes = kv_row_elems * sizeof(__half);
    float scale = 1.0f / std::sqrt((float)head_dim);

    BlockAllocatorConfig cfg = {
        64, (size_t)block_size, (size_t)nlayer,
        (size_t)num_kv_heads, (size_t)head_dim, sizeof(__half)
    };
    BlockAllocator alloc(cfg, api);

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<PageTable> pts;
    std::vector<std::vector<float>> all_contig_k(batch_size), all_contig_v(batch_size);
    int max_blocks = 0;
    std::vector<__half> host_k_half(kv_row_elems), host_v_half(kv_row_elems);

    for (int b = 0; b < batch_size; ++b) {
        pts.emplace_back(block_size);
        int seq_len = seq_lens_arr[b];
        all_contig_k[b].resize((size_t)seq_len * kv_row_elems);
        all_contig_v[b].resize((size_t)seq_len * kv_row_elems);

        for (int t = 0; t < seq_len; ++t) {
            if (pts[b].needs_new_block()) {
                int bid = alloc.alloc();
                CHECK(bid >= 0);
                pts[b].append_block(bid);
            }

            int bid = pts[b].get_block_for_token(t);
            int off = pts[b].get_offset_in_block(t);
            void *k_dst = static_cast<char *>(alloc.get_k_ptr(bid, layer_idx)) + (size_t)off * kv_row_bytes;
            void *v_dst = static_cast<char *>(alloc.get_v_ptr(bid, layer_idx)) + (size_t)off * kv_row_bytes;

            for (size_t j = 0; j < kv_row_elems; ++j) {
                float kv = dist(rng);
                float vv = dist(rng);
                all_contig_k[b][(size_t)t * kv_row_elems + j] = kv;
                all_contig_v[b][(size_t)t * kv_row_elems + j] = vv;
                host_k_half[j] = __float2half(kv);
                host_v_half[j] = __float2half(vv);
            }

            api->memcpy_sync(k_dst, host_k_half.data(), kv_row_bytes, LLAISYS_MEMCPY_H2D);
            api->memcpy_sync(v_dst, host_v_half.data(), kv_row_bytes, LLAISYS_MEMCPY_H2D);
            pts[b].inc_num_tokens();
        }
        max_blocks = std::max(max_blocks, pts[b].num_blocks());
    }

    std::vector<int> block_tables(batch_size * max_blocks, 0);
    std::vector<int> seq_lens(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        seq_lens[b] = seq_lens_arr[b];
        for (int j = 0; j < pts[b].num_blocks(); ++j)
            block_tables[b * max_blocks + j] = pts[b].block_ids()[j];
    }

    size_t q_elems = (size_t)batch_size * num_heads * head_dim;
    std::vector<float> query_f32(q_elems);
    std::vector<__half> query_f16(q_elems), out_f16(q_elems);
    std::vector<__half> zero_f16(q_elems, __float2half(0.0f));
    for (size_t i = 0; i < q_elems; ++i) {
        query_f32[i] = dist(rng);
        query_f16[i] = __float2half(query_f32[i]);
    }

    __half *d_query = static_cast<__half *>(api->malloc_device(q_elems * sizeof(__half)));
    __half *d_output = static_cast<__half *>(api->malloc_device(q_elems * sizeof(__half)));
    CHECK(d_query != nullptr);
    CHECK(d_output != nullptr);

    api->memcpy_sync(d_query, query_f16.data(), q_elems * sizeof(__half), LLAISYS_MEMCPY_H2D);
    api->memcpy_sync(d_output, zero_f16.data(), q_elems * sizeof(__half), LLAISYS_MEMCPY_H2D);

    llaisys::ops::paged_attention(
        d_output, d_query,
        alloc.pool_k_raw(), alloc.pool_v_raw(),
        block_tables.data(), seq_lens.data(),
        batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks,
        alloc.block_stride(), alloc.layer_stride(),
        layer_idx, scale, LLAISYS_DEVICE_NVIDIA,
        llaisys::ops::KVQuantMode::FP32, LLAISYS_DTYPE_F16);
    api->device_synchronize();

    api->memcpy_sync(out_f16.data(), d_output, q_elems * sizeof(__half), LLAISYS_MEMCPY_D2H);

    std::vector<float> out_f32(q_elems), ref_out(q_elems, 0.0f);
    for (size_t i = 0; i < q_elems; ++i)
        out_f32[i] = __half2float(out_f16[i]);

    for (int b = 0; b < batch_size; ++b) {
        reference_attention(ref_out.data(), query_f32.data(),
                            all_contig_k[b].data(), all_contig_v[b].data(),
                            b, seq_lens[b], num_heads, num_kv_heads, head_dim, scale);
    }

    float err = max_abs_error(out_f32.data(), ref_out.data(), q_elems);
    CHECK(err < 5e-2f);

    api->free_device(d_query);
    api->free_device(d_output);
    for (auto &pt : pts) pt.release_all(alloc);
    std::cout << "  test_gpu_f16_paged_attention PASSED (max_err=" << err << ")" << std::endl;
#else
    std::cout << "  test_gpu_f16_paged_attention SKIPPED (CUDA disabled)" << std::endl;
#endif
}

int main() {
    std::cout << "=== Paged Attention Correctness Tests ===" << std::endl;
    test_single_sequence();
    test_batch_sequences();
    test_non_aligned_seq_len();
    test_larger_dims();
    test_gpu_f16_paged_attention();
    std::cout << "\n✓ All Phase 2 tests passed!" << std::endl;
    return 0;
}

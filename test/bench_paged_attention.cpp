/**
 * Performance Benchmark: Paged Attention vs Contiguous Attention
 *
 * Measures:
 *   1. Block Allocator alloc/free throughput
 *   2. Paged Attention kernel latency (varying batch size, seq_len, head_dim)
 *   3. Contiguous self_attention kernel latency (same configs, as baseline)
 *   4. Memory utilization comparison (paged vs contiguous)
 */

#include "src/core/allocator/block_allocator.hpp"
#include "src/core/page_table.hpp"
#include "src/ops/self_attention/paged_attention.hpp"
#include "src/ops/self_attention/op.hpp"
#include "src/tensor/tensor.hpp"
#include "llaisys/runtime.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

#ifdef ENABLE_NVIDIA_API
#include <cuda_fp16.h>
#endif

using namespace llaisys::core;
using Clock = std::chrono::high_resolution_clock;

static const LlaisysRuntimeAPI *cpu_api() {
    return llaisysGetRuntimeAPI(LLAISYS_DEVICE_CPU);
}

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Benchmark 1: Block Allocator Throughput ──────────────────────

static void bench_allocator(int num_blocks, int rounds) {
    BlockAllocatorConfig cfg = {
        (size_t)num_blocks, 16, 1, 2, 128, sizeof(float)
    };
    BlockAllocator alloc(cfg, cpu_api());

    // Alloc all
    auto t0 = Clock::now();
    std::vector<int> ids(num_blocks);
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < num_blocks; ++i)
            ids[i] = alloc.alloc();
        for (int i = 0; i < num_blocks; ++i)
            alloc.free(ids[i]);
    }
    auto t1 = Clock::now();

    double total_ops = 2.0 * num_blocks * rounds;
    double ms = elapsed_ms(t0, t1);
    double ops_per_sec = total_ops / (ms / 1000.0);

    std::cout << "  Allocator: " << num_blocks << " blocks × " << rounds
              << " rounds = " << std::fixed << std::setprecision(1)
              << ops_per_sec / 1e6 << " M ops/s (" << ms << " ms)" << std::endl;
}

// ── Benchmark 2: Paged Attention Latency ─────────────────────────

struct BenchConfig {
    int batch_size;
    int seq_len;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int block_size;
    int nlayer;
    const char* label;
};

static void fill_random(float* data, size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(rng);
}

#ifdef ENABLE_NVIDIA_API
static void float_to_half(const float* src, __half* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = __float2half(src[i]);
}
#endif

static double bench_paged_attention(const BenchConfig& cfg, int warmup, int iters) {
    int block_size = cfg.block_size;
    size_t num_blocks_needed = 0;
    std::vector<int> seq_lens(cfg.batch_size);

    for (int b = 0; b < cfg.batch_size; ++b) {
        seq_lens[b] = cfg.seq_len;
        num_blocks_needed += (cfg.seq_len + block_size - 1) / block_size;
    }

    BlockAllocatorConfig alloc_cfg = {
        num_blocks_needed + 16, (size_t)block_size, (size_t)cfg.nlayer,
        (size_t)cfg.num_kv_heads, (size_t)cfg.head_dim, sizeof(float)
    };
    BlockAllocator alloc(alloc_cfg, cpu_api());

    std::mt19937 rng(42);

    // Build page tables and fill KV data
    std::vector<PageTable> pts;
    int max_blocks_per_seq = (cfg.seq_len + block_size - 1) / block_size;

    for (int b = 0; b < cfg.batch_size; ++b) {
        pts.emplace_back(block_size);
        for (int t = 0; t < cfg.seq_len; ++t) {
            if (pts[b].needs_new_block()) {
                int bid = alloc.alloc();
                pts[b].append_block(bid);
            }
            int bid = pts[b].get_block_for_token(t);
            int off = pts[b].get_offset_in_block(t);
            for (int layer = 0; layer < cfg.nlayer; ++layer) {
                float* k_ptr = (float*)alloc.get_k_ptr(bid, layer)
                               + off * cfg.num_kv_heads * cfg.head_dim;
                float* v_ptr = (float*)alloc.get_v_ptr(bid, layer)
                               + off * cfg.num_kv_heads * cfg.head_dim;
                fill_random(k_ptr, cfg.num_kv_heads * cfg.head_dim, rng);
                fill_random(v_ptr, cfg.num_kv_heads * cfg.head_dim, rng);
            }
            pts[b].inc_num_tokens();
        }
    }

    // Build block tables
    std::vector<int> block_tables(cfg.batch_size * max_blocks_per_seq, 0);
    for (int b = 0; b < cfg.batch_size; ++b)
        for (int j = 0; j < pts[b].num_blocks(); ++j)
            block_tables[b * max_blocks_per_seq + j] = pts[b].block_ids()[j];

    // Query
    std::vector<float> query(cfg.batch_size * cfg.num_heads * cfg.head_dim);
    fill_random(query.data(), query.size(), rng);

    std::vector<float> output(cfg.batch_size * cfg.num_heads * cfg.head_dim, 0.0f);
    float scale = 1.0f / std::sqrt((float)cfg.head_dim);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        llaisys::ops::paged_attention(
            output.data(), query.data(),
            alloc.pool_k_raw(), alloc.pool_v_raw(),
            block_tables.data(), seq_lens.data(),
            cfg.batch_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
            block_size, max_blocks_per_seq,
            alloc.block_stride(), alloc.layer_stride(),
            0, scale, LLAISYS_DEVICE_CPU);
    }

    // Benchmark
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        llaisys::ops::paged_attention(
            output.data(), query.data(),
            alloc.pool_k_raw(), alloc.pool_v_raw(),
            block_tables.data(), seq_lens.data(),
            cfg.batch_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
            block_size, max_blocks_per_seq,
            alloc.block_stride(), alloc.layer_stride(),
            0, scale, LLAISYS_DEVICE_CPU);
    }
    auto t1 = Clock::now();

    double avg_ms = elapsed_ms(t0, t1) / iters;

    for (auto& pt : pts) pt.release_all(alloc);
    return avg_ms;
}

// ── Benchmark 3: Contiguous self_attention Latency ───────────────

static double bench_contiguous_attention(const BenchConfig& cfg, int warmup, int iters) {
    using llaisys::Tensor;

    std::mt19937 rng(42);

    float scale = 1.0f / std::sqrt((float)cfg.head_dim);

    // Per-batch serial attention (simulating old batch decode path)
    // Each batch element gets its own Q [1, nh, dh], K [seq, nkvh, dh], V [seq, nkvh, dh]
    std::vector<llaisys::tensor_t> qs, ks, vs, outs;
    for (int b = 0; b < cfg.batch_size; ++b) {
        auto q = Tensor::create({1, (size_t)cfg.num_heads, (size_t)cfg.head_dim},
                                LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
        auto k = Tensor::create({(size_t)cfg.seq_len, (size_t)cfg.num_kv_heads, (size_t)cfg.head_dim},
                                LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
        auto v = Tensor::create({(size_t)cfg.seq_len, (size_t)cfg.num_kv_heads, (size_t)cfg.head_dim},
                                LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
        auto o = Tensor::create({1, (size_t)cfg.num_heads, (size_t)cfg.head_dim},
                                LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
        fill_random((float*)q->data(), cfg.num_heads * cfg.head_dim, rng);
        fill_random((float*)k->data(), cfg.seq_len * cfg.num_kv_heads * cfg.head_dim, rng);
        fill_random((float*)v->data(), cfg.seq_len * cfg.num_kv_heads * cfg.head_dim, rng);
        qs.push_back(q); ks.push_back(k); vs.push_back(v); outs.push_back(o);
    }

    auto run_serial = [&]() {
        for (int b = 0; b < cfg.batch_size; ++b)
            llaisys::ops::self_attention(outs[b], qs[b], ks[b], vs[b], scale);
    };

    for (int i = 0; i < warmup; ++i) run_serial();

    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) run_serial();
    auto t1 = Clock::now();

    return elapsed_ms(t0, t1) / iters;
}

// ── Benchmark 4: Memory Utilization ──────────────────────────────

static void bench_memory_comparison(int num_slots, int max_seq, int nlayer,
                                     int nkvh, int dh, int block_size) {
    // Contiguous: each slot pre-allocates [maxseq, nkvh, dh] × nlayer × 2 (K+V)
    size_t contiguous_per_slot = (size_t)max_seq * nkvh * dh * sizeof(float) * nlayer * 2;
    size_t contiguous_total = contiguous_per_slot * num_slots;

    // Paged: only used tokens consume blocks
    // Simulate varying utilization
    std::cout << "\n  Memory comparison: " << num_slots << " slots, maxseq=" << max_seq
              << ", nlayer=" << nlayer << ", nkvh=" << nkvh << ", dh=" << dh << std::endl;
    std::cout << "  Contiguous total: "
              << std::fixed << std::setprecision(1)
              << contiguous_total / (1024.0 * 1024.0) << " MB" << std::endl;

    size_t block_data_bytes = (size_t)block_size * nkvh * dh * sizeof(float) * nlayer * 2;

    int utilizations[] = {10, 25, 50, 75, 100};
    for (int pct : utilizations) {
        size_t avg_tokens = max_seq * pct / 100;
        size_t total_tokens = avg_tokens * num_slots;
        size_t total_blocks = (total_tokens + block_size - 1) / block_size;
        size_t paged_bytes = total_blocks * block_data_bytes;
        double savings = 100.0 * (1.0 - (double)paged_bytes / contiguous_total);

        std::cout << "  Paged @" << std::setw(3) << pct << "% util: "
                  << std::setw(8) << std::fixed << std::setprecision(1)
                  << paged_bytes / (1024.0 * 1024.0) << " MB  (saving "
                  << std::setprecision(0) << savings << "%)" << std::endl;
    }
}

// ── Benchmark 5: Full Decode Step Simulation ─────────────────────

static void bench_decode_step(int batch_size, int seq_len,
                               int num_heads, int num_kv_heads, int head_dim,
                               int block_size, int nlayer, int iters) {
    size_t max_tokens = seq_len + iters;
    size_t blocks_per_seq = (max_tokens + block_size - 1) / block_size;
    BlockAllocatorConfig cfg = {
        blocks_per_seq * batch_size + 16,
        (size_t)block_size, (size_t)nlayer,
        (size_t)num_kv_heads, (size_t)head_dim, sizeof(float)
    };
    BlockAllocator alloc(cfg, cpu_api());

    std::mt19937 rng(42);
    std::vector<PageTable> pts(batch_size, PageTable(block_size));
    std::vector<int> positions(batch_size, 0);
    float scale = 1.0f / std::sqrt((float)head_dim);

    // Prefill: fill KV up to seq_len
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            if (pts[b].needs_new_block()) {
                pts[b].append_block(alloc.alloc());
            }
            int bid = pts[b].get_block_for_token(t);
            int off = pts[b].get_offset_in_block(t);
            for (int layer = 0; layer < nlayer; ++layer) {
                float* kp = (float*)alloc.get_k_ptr(bid, layer)
                            + off * num_kv_heads * head_dim;
                float* vp = (float*)alloc.get_v_ptr(bid, layer)
                            + off * num_kv_heads * head_dim;
                fill_random(kp, num_kv_heads * head_dim, rng);
                fill_random(vp, num_kv_heads * head_dim, rng);
            }
            pts[b].inc_num_tokens();
            positions[b]++;
        }
    }

    size_t kv_bytes = num_kv_heads * head_dim * sizeof(float);

    // Benchmark decode steps
    std::vector<double> step_times;
    for (int step = 0; step < iters; ++step) {
        auto t0 = Clock::now();

        // Allocate new blocks if needed
        for (int b = 0; b < batch_size; ++b) {
            if (pts[b].needs_new_block())
                pts[b].append_block(alloc.alloc());
        }

        // Build block tables
        int max_blocks = 0;
        for (int b = 0; b < batch_size; ++b)
            max_blocks = std::max(max_blocks, pts[b].num_blocks());

        std::vector<int> block_tables(batch_size * max_blocks, 0);
        std::vector<int> sl(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            sl[b] = positions[b] + 1;
            for (int j = 0; j < pts[b].num_blocks(); ++j)
                block_tables[b * max_blocks + j] = pts[b].block_ids()[j];
        }

        std::vector<float> query(batch_size * num_heads * head_dim);
        std::vector<float> new_k(batch_size * num_kv_heads * head_dim);
        std::vector<float> new_v(batch_size * num_kv_heads * head_dim);
        fill_random(query.data(), query.size(), rng);
        fill_random(new_k.data(), new_k.size(), rng);
        fill_random(new_v.data(), new_v.size(), rng);

        for (int layer = 0; layer < nlayer; ++layer) {
            // Write KV for current token
            for (int b = 0; b < batch_size; ++b) {
                int pos = positions[b];
                int bid = pts[b].get_block_for_token(pos);
                int off = pts[b].get_offset_in_block(pos);
                float* kd = (float*)alloc.get_k_ptr(bid, layer)
                            + off * num_kv_heads * head_dim;
                float* vd = (float*)alloc.get_v_ptr(bid, layer)
                            + off * num_kv_heads * head_dim;
                std::memcpy(kd, new_k.data() + b * num_kv_heads * head_dim, kv_bytes);
                std::memcpy(vd, new_v.data() + b * num_kv_heads * head_dim, kv_bytes);
            }

            // Paged attention
            std::vector<float> output(batch_size * num_heads * head_dim, 0.0f);
            llaisys::ops::paged_attention(
                output.data(), query.data(),
                alloc.pool_k_raw(), alloc.pool_v_raw(),
                block_tables.data(), sl.data(),
                batch_size, num_heads, num_kv_heads, head_dim,
                block_size, max_blocks,
                alloc.block_stride(), alloc.layer_stride(),
                layer, scale, LLAISYS_DEVICE_CPU);
        }

        // Update positions
        for (int b = 0; b < batch_size; ++b) {
            pts[b].inc_num_tokens();
            positions[b]++;
        }

        auto t1 = Clock::now();
        step_times.push_back(elapsed_ms(t0, t1));
    }

    double avg = std::accumulate(step_times.begin(), step_times.end(), 0.0) / step_times.size();
    std::sort(step_times.begin(), step_times.end());
    double p50 = step_times[step_times.size() / 2];
    double p99 = step_times[(size_t)(step_times.size() * 0.99)];
    double tps = batch_size / (avg / 1000.0);

    std::cout << "  B=" << batch_size << " seq=" << seq_len << "+"
              << iters << " layers=" << nlayer
              << " nh=" << num_heads << " nkvh=" << num_kv_heads
              << " dh=" << head_dim
              << "  avg=" << std::fixed << std::setprecision(2) << avg << "ms"
              << "  p50=" << p50 << "ms"
              << "  p99=" << p99 << "ms"
              << "  throughput=" << std::setprecision(1) << tps << " tok/s"
              << std::endl;

    for (auto& pt : pts) pt.release_all(alloc);
}

// ── GPU Benchmark Helpers ─────────────────────────────────────────

static bool gpu_available() {
#ifdef ENABLE_NVIDIA_API
    auto *api = llaisysGetRuntimeAPI(LLAISYS_DEVICE_NVIDIA);
    if (api && api->get_device_count && api->get_device_count() > 0) return true;
#endif
    return false;
}

static void bench_paged_attention_gpu(const BenchConfig& cfg, int warmup, int iters) {
#ifdef ENABLE_NVIDIA_API
    if (!gpu_available()) return;

    auto *api = llaisysGetRuntimeAPI(LLAISYS_DEVICE_NVIDIA);
    int block_size = cfg.block_size;
    std::mt19937 rng(42);

    size_t num_blocks_needed = 0;
    std::vector<int> seq_lens(cfg.batch_size);
    for (int b = 0; b < cfg.batch_size; ++b) {
        seq_lens[b] = cfg.seq_len;
        num_blocks_needed += (cfg.seq_len + block_size - 1) / block_size;
    }

    // Allocate on GPU via runtime API
    BlockAllocatorConfig alloc_cfg = {
        num_blocks_needed + 16, (size_t)block_size, (size_t)cfg.nlayer,
        (size_t)cfg.num_kv_heads, (size_t)cfg.head_dim, sizeof(__half)
    };
    BlockAllocator alloc(alloc_cfg, api);

    // Build page tables
    std::vector<PageTable> pts;
    int max_blocks_per_seq = (cfg.seq_len + block_size - 1) / block_size;

    for (int b = 0; b < cfg.batch_size; ++b) {
        pts.emplace_back(block_size);
        for (int t = 0; t < cfg.seq_len; ++t) {
            if (pts[b].needs_new_block())
                pts[b].append_block(alloc.alloc());
            pts[b].inc_num_tokens();
        }
    }

    // Fill KV on GPU (random data)
    size_t kv_row_elements = cfg.num_kv_heads * cfg.head_dim;
    std::vector<float> host_kv_f32(kv_row_elements);
    std::vector<__half> host_kv_f16(kv_row_elements);
    for (int b = 0; b < cfg.batch_size; ++b) {
        for (int t = 0; t < cfg.seq_len; ++t) {
            int bid = pts[b].get_block_for_token(t);
            int off = pts[b].get_offset_in_block(t);
            fill_random(host_kv_f32.data(), kv_row_elements, rng);
            float_to_half(host_kv_f32.data(), host_kv_f16.data(), kv_row_elements);
            void* k_ptr = (char*)alloc.get_k_ptr(bid, 0) + (size_t)off * kv_row_elements * sizeof(__half);
            void* v_ptr = (char*)alloc.get_v_ptr(bid, 0) + (size_t)off * kv_row_elements * sizeof(__half);
            api->memcpy_sync(k_ptr, host_kv_f16.data(), kv_row_elements * sizeof(__half), LLAISYS_MEMCPY_H2D);
            fill_random(host_kv_f32.data(), kv_row_elements, rng);
            float_to_half(host_kv_f32.data(), host_kv_f16.data(), kv_row_elements);
            api->memcpy_sync(v_ptr, host_kv_f16.data(), kv_row_elements * sizeof(__half), LLAISYS_MEMCPY_H2D);
        }
    }

    // Block tables
    std::vector<int> block_tables(cfg.batch_size * max_blocks_per_seq, 0);
    for (int b = 0; b < cfg.batch_size; ++b)
        for (int j = 0; j < pts[b].num_blocks(); ++j)
            block_tables[b * max_blocks_per_seq + j] = pts[b].block_ids()[j];

    // Query on GPU
    size_t q_elements = cfg.batch_size * cfg.num_heads * cfg.head_dim;
    std::vector<float> host_query_f32(q_elements);
    std::vector<__half> host_query_f16(q_elements);
    fill_random(host_query_f32.data(), q_elements, rng);
    float_to_half(host_query_f32.data(), host_query_f16.data(), q_elements);
    __half* d_query = (__half*)api->malloc_device(q_elements * sizeof(__half));
    __half* d_output = (__half*)api->malloc_device(q_elements * sizeof(__half));
    api->memcpy_sync(d_query, host_query_f16.data(), q_elements * sizeof(__half), LLAISYS_MEMCPY_H2D);

    float scale = 1.0f / std::sqrt((float)cfg.head_dim);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        llaisys::ops::paged_attention(
            d_output, d_query,
            alloc.pool_k_raw(), alloc.pool_v_raw(),
            block_tables.data(), seq_lens.data(),
            cfg.batch_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
            block_size, max_blocks_per_seq,
            alloc.block_stride(), alloc.layer_stride(),
            0, scale, LLAISYS_DEVICE_NVIDIA,
            llaisys::ops::KVQuantMode::FP32, LLAISYS_DTYPE_F16);
    }
    api->device_synchronize();

    // Benchmark
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        llaisys::ops::paged_attention(
            d_output, d_query,
            alloc.pool_k_raw(), alloc.pool_v_raw(),
            block_tables.data(), seq_lens.data(),
            cfg.batch_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
            block_size, max_blocks_per_seq,
            alloc.block_stride(), alloc.layer_stride(),
            0, scale, LLAISYS_DEVICE_NVIDIA,
            llaisys::ops::KVQuantMode::FP32, LLAISYS_DTYPE_F16);
    }
    api->device_synchronize();
    auto t1 = Clock::now();

    double avg_ms = elapsed_ms(t0, t1) / iters;
    std::cout << "  GPU-F16 " << std::left << std::setw(12) << cfg.label
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << avg_ms << " ms" << std::endl;

    api->free_device(d_query);
    api->free_device(d_output);
    for (auto& pt : pts) pt.release_all(alloc);
#else
    (void)cfg; (void)warmup; (void)iters;
#endif
}

// ── Main ─────────────────────────────────────────────────────────

int main() {
    bool has_gpu = gpu_available();
    std::string device_str = has_gpu ? "CPU + GPU" : "CPU";

    std::cout << "================================================================" << std::endl;
    std::cout << "  LLAISYS Paged Attention Performance Benchmark (" << device_str << ")" << std::endl;
    std::cout << "================================================================\n" << std::endl;

    // ── 1. Block Allocator ──
    std::cout << "--- 1. Block Allocator Throughput ---" << std::endl;
    bench_allocator(1024, 100);
    bench_allocator(4096, 100);
    bench_allocator(16384, 50);

    // ── 2. Paged Attention vs Contiguous Attention ──
    std::cout << "\n--- 2. Paged vs Contiguous Attention Latency ---" << std::endl;

    BenchConfig configs[] = {
        {1,   64,  12, 2, 128, 16, 1, "B=1  seq=64"},
        {1,  256,  12, 2, 128, 16, 1, "B=1  seq=256"},
        {1, 1024,  12, 2, 128, 16, 1, "B=1  seq=1024"},
        {4,   64,  12, 2, 128, 16, 1, "B=4  seq=64"},
        {4,  256,  12, 2, 128, 16, 1, "B=4  seq=256"},
        {8,   64,  12, 2, 128, 16, 1, "B=8  seq=64"},
        {8,  256,  12, 2, 128, 16, 1, "B=8  seq=256"},
    };

    BenchConfig gpu_configs[] = {
        {1,   64,   8, 2, 128, 16, 1, "B=1 seq=64"},
        {1,  256,   8, 2, 128, 16, 1, "B=1 seq=256"},
        {4,   64,   8, 2, 128, 16, 1, "B=4 seq=64"},
        {4,  256,   8, 2, 128, 16, 1, "B=4 seq=256"},
        {8,   64,   8, 2, 128, 16, 1, "B=8 seq=64"},
        {8,  256,   8, 2, 128, 16, 1, "B=8 seq=256"},
    };

    std::cout << std::left << std::setw(18) << "  Config"
              << std::right << std::setw(12) << "Paged(ms)"
              << std::setw(12) << "Contig(ms)"
              << std::setw(10) << "Ratio" << std::endl;
    std::cout << "  " << std::string(50, '-') << std::endl;

    for (auto& c : configs) {
        double paged_ms = bench_paged_attention(c, 3, 20);
        double contig_ms = bench_contiguous_attention(c, 3, 20);
        double ratio = paged_ms / contig_ms;

        std::cout << "  " << std::left << std::setw(16) << c.label
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3)
                  << paged_ms << "  "
                  << std::setw(10) << contig_ms << "  "
                  << std::setw(6) << std::setprecision(2) << ratio << "x"
                  << std::endl;
    }

    // ── 2b. GPU Paged Attention (if available) ──
    if (has_gpu) {
        std::cout << "\n--- 2b. GPU Paged Attention Latency ---" << std::endl;
        for (auto& c : gpu_configs) bench_paged_attention_gpu(c, 5, 100);
    }

    // ── 3. Memory Utilization ──
    std::cout << "\n--- 3. Memory Utilization Comparison ---" << std::endl;
    // DeepSeek-R1-Distill-Qwen-1.5B config
    bench_memory_comparison(8, 2048, 28, 2, 128, 16);
    // Qwen2-7B-like config
    bench_memory_comparison(8, 2048, 28, 4, 128, 16);

    // ── 4. Full Decode Step (all layers) ──
    std::cout << "\n--- 4. Full Decode Step (KV write + Attention × all layers) ---"
              << std::endl;
    // Small model config: 2 layers
    bench_decode_step(1, 64,   12, 2, 128, 16, 2, 50);
    bench_decode_step(4, 64,   12, 2, 128, 16, 2, 50);
    bench_decode_step(8, 64,   12, 2, 128, 16, 2, 30);
    bench_decode_step(1, 256,  12, 2, 128, 16, 2, 30);
    bench_decode_step(4, 256,  12, 2, 128, 16, 2, 20);
    bench_decode_step(8, 256,  12, 2, 128, 16, 2, 20);
    // Larger seq
    bench_decode_step(1, 1024, 12, 2, 128, 16, 2, 10);
    bench_decode_step(4, 1024, 12, 2, 128, 16, 2, 10);

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Benchmark complete." << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}

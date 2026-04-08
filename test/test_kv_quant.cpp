/**
 * Test: INT8/INT4 KV-Cache Quantization + Paged Attention
 *
 * Validates:
 *   1. Quantize/Dequantize roundtrip accuracy (INT8, INT4)
 *   2. Paged attention with INT8 KV matches FP32 within tolerance
 *   3. Paged attention with INT4 KV matches FP32 within tolerance
 *   4. Memory savings from quantized KV
 */

#include "src/core/kv_quant.hpp"
#include "src/ops/self_attention/paged_attention.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>

using namespace llaisys::core;
using namespace llaisys::ops;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << (msg) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        ++g_fail; \
    } else { \
        std::cout << "  PASS: " << (msg) << std::endl; \
        ++g_pass; \
    } \
} while(0)

static void fill_random(float* data, size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(rng);
}

static float max_abs_error(const float* a, const float* b, size_t n) {
    float err = 0.0f;
    for (size_t i = 0; i < n; ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

// ── Test 1: INT8 roundtrip ──

static void test_int8_roundtrip() {
    std::cout << "\n--- Test 1: INT8 Quantize/Dequantize Roundtrip ---" << std::endl;
    std::mt19937 rng(42);

    const int dh = 128;
    std::vector<float> src(dh), dst(dh);
    std::vector<int8_t> q_data(dh);

    fill_random(src.data(), dh, rng);
    float scale = quantize_fp32_to_int8(q_data.data(), src.data(), dh);
    dequantize_int8_to_fp32(dst.data(), q_data.data(), scale, dh);

    float err = max_abs_error(src.data(), dst.data(), dh);
    CHECK(err < 0.01f, "INT8 roundtrip error < 0.01 (got " + std::to_string(err) + ")");
}

// ── Test 2: INT4 roundtrip ──

static void test_int4_roundtrip() {
    std::cout << "\n--- Test 2: INT4 Quantize/Dequantize Roundtrip ---" << std::endl;
    std::mt19937 rng(42);

    const int dh = 128;
    std::vector<float> src(dh), dst(dh);
    std::vector<uint8_t> q_data((dh + 1) / 2);

    fill_random(src.data(), dh, rng);
    float scale = quantize_fp32_to_int4(q_data.data(), src.data(), dh);
    dequantize_int4_to_fp32(dst.data(), q_data.data(), scale, dh);

    float err = max_abs_error(src.data(), dst.data(), dh);
    CHECK(err < 0.15f, "INT4 roundtrip error < 0.15 (got " + std::to_string(err) + ")");
}

// ── Test 3: Paged Attention INT8 vs FP32 ──

static void test_paged_attention_int8() {
    std::cout << "\n--- Test 3: Paged Attention INT8 vs FP32 ---" << std::endl;
    std::mt19937 rng(42);

    const int batch = 2, nh = 4, nkvh = 2, dh = 64, seq = 32, bs = 16;
    const int nlayer = 1;
    float scale = 1.0f / std::sqrt((float)dh);
    int max_blocks = (seq + bs - 1) / bs;

    // FP32 pool
    size_t fp32_layer_stride = bs * nkvh * dh * sizeof(float);
    size_t fp32_block_stride = nlayer * fp32_layer_stride;
    size_t fp32_pool_bytes = max_blocks * batch * fp32_block_stride;
    std::vector<float> k_pool_fp32(fp32_pool_bytes / sizeof(float));
    std::vector<float> v_pool_fp32(fp32_pool_bytes / sizeof(float));

    // INT8 pool
    size_t int8_data_per_layer = bs * nkvh * dh * sizeof(int8_t);
    size_t int8_scales_per_layer = bs * nkvh * sizeof(float);
    size_t int8_layer_stride = int8_data_per_layer + int8_scales_per_layer;
    size_t int8_block_stride = nlayer * int8_layer_stride;
    size_t int8_pool_bytes = max_blocks * batch * int8_block_stride;
    std::vector<char> k_pool_int8(int8_pool_bytes);
    std::vector<char> v_pool_int8(int8_pool_bytes);

    // Fill KV data: FP32 first, then quantize to INT8
    std::vector<float> kv_row(nkvh * dh);
    int block_id = 0;
    std::vector<int> block_tables(batch * max_blocks);
    std::vector<int> seq_lens(batch, seq);

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            int bi = t / bs;
            int off = t % bs;
            if (off == 0) {
                block_tables[b * max_blocks + bi] = block_id;
                block_id++;
            }
            int bid = block_tables[b * max_blocks + bi];

            fill_random(kv_row.data(), nkvh * dh, rng);

            // FP32
            float* k_fp32_dst = k_pool_fp32.data() + bid * (fp32_block_stride / sizeof(float))
                                + off * nkvh * dh;
            std::memcpy(k_fp32_dst, kv_row.data(), nkvh * dh * sizeof(float));

            // INT8
            int8_t* k_int8_dst = reinterpret_cast<int8_t*>(k_pool_int8.data() + bid * int8_block_stride)
                                 + off * nkvh * dh;
            float* k_int8_scales = reinterpret_cast<float*>(
                k_pool_int8.data() + bid * int8_block_stride + int8_data_per_layer)
                + off * nkvh;
            quantize_kv_row_int8(k_int8_dst, k_int8_scales, kv_row.data(), nkvh, dh);

            // V
            fill_random(kv_row.data(), nkvh * dh, rng);

            float* v_fp32_dst = v_pool_fp32.data() + bid * (fp32_block_stride / sizeof(float))
                                + off * nkvh * dh;
            std::memcpy(v_fp32_dst, kv_row.data(), nkvh * dh * sizeof(float));

            int8_t* v_int8_dst = reinterpret_cast<int8_t*>(v_pool_int8.data() + bid * int8_block_stride)
                                 + off * nkvh * dh;
            float* v_int8_scales = reinterpret_cast<float*>(
                v_pool_int8.data() + bid * int8_block_stride + int8_data_per_layer)
                + off * nkvh;
            quantize_kv_row_int8(v_int8_dst, v_int8_scales, kv_row.data(), nkvh, dh);
        }
    }

    // Query
    std::vector<float> query(batch * nh * dh);
    fill_random(query.data(), query.size(), rng);

    std::vector<float> out_fp32(batch * nh * dh, 0.0f);
    std::vector<float> out_int8(batch * nh * dh, 0.0f);

    paged_attention(out_fp32.data(), query.data(),
                    k_pool_fp32.data(), v_pool_fp32.data(),
                    block_tables.data(), seq_lens.data(),
                    batch, nh, nkvh, dh, bs, max_blocks,
                    fp32_block_stride, fp32_layer_stride,
                    0, scale, LLAISYS_DEVICE_CPU, KVQuantMode::FP32);

    paged_attention(out_int8.data(), query.data(),
                    k_pool_int8.data(), v_pool_int8.data(),
                    block_tables.data(), seq_lens.data(),
                    batch, nh, nkvh, dh, bs, max_blocks,
                    int8_block_stride, int8_layer_stride,
                    0, scale, LLAISYS_DEVICE_CPU, KVQuantMode::INT8);

    float err = max_abs_error(out_fp32.data(), out_int8.data(), batch * nh * dh);
    CHECK(err < 0.05f, "INT8 paged attention max error < 0.05 (got " + std::to_string(err) + ")");

    // Memory savings
    double fp32_mem = fp32_pool_bytes * 2.0;
    double int8_mem = int8_pool_bytes * 2.0;
    double savings = 100.0 * (1.0 - int8_mem / fp32_mem);
    std::cout << "  Memory: FP32=" << fp32_mem / 1024 << "KB INT8=" << int8_mem / 1024
              << "KB savings=" << savings << "%" << std::endl;
    CHECK(savings > 50.0, "INT8 memory savings > 50%");
}

// ── Test 4: Paged Attention INT4 vs FP32 ──

static void test_paged_attention_int4() {
    std::cout << "\n--- Test 4: Paged Attention INT4 vs FP32 ---" << std::endl;
    std::mt19937 rng(42);

    const int batch = 1, nh = 4, nkvh = 2, dh = 64, seq = 16, bs = 16;
    const int nlayer = 1;
    float scale = 1.0f / std::sqrt((float)dh);
    int max_blocks = (seq + bs - 1) / bs;
    size_t packed_dh = (dh + 1) / 2;

    // FP32 pool
    size_t fp32_layer_stride = bs * nkvh * dh * sizeof(float);
    size_t fp32_block_stride = nlayer * fp32_layer_stride;
    std::vector<float> k_pool_fp32(max_blocks * fp32_block_stride / sizeof(float));
    std::vector<float> v_pool_fp32(max_blocks * fp32_block_stride / sizeof(float));

    // INT4 pool
    size_t int4_data_per_layer = bs * nkvh * packed_dh * sizeof(uint8_t);
    size_t int4_scales_per_layer = bs * nkvh * sizeof(float);
    size_t int4_layer_stride = int4_data_per_layer + int4_scales_per_layer;
    size_t int4_block_stride = nlayer * int4_layer_stride;
    std::vector<char> k_pool_int4(max_blocks * int4_block_stride);
    std::vector<char> v_pool_int4(max_blocks * int4_block_stride);

    std::vector<float> kv_row(nkvh * dh);
    std::vector<int> block_tables(max_blocks);
    std::vector<int> seq_lens = {seq};

    for (int t = 0; t < seq; ++t) {
        int bi = t / bs, off = t % bs;
        if (off == 0) block_tables[bi] = bi;

        fill_random(kv_row.data(), nkvh * dh, rng);
        float* k_fp32 = k_pool_fp32.data() + bi * (fp32_block_stride / sizeof(float)) + off * nkvh * dh;
        std::memcpy(k_fp32, kv_row.data(), nkvh * dh * sizeof(float));

        uint8_t* k_int4 = reinterpret_cast<uint8_t*>(k_pool_int4.data() + bi * int4_block_stride) + off * nkvh * packed_dh;
        float* k_scales = reinterpret_cast<float*>(k_pool_int4.data() + bi * int4_block_stride + int4_data_per_layer) + off * nkvh;
        quantize_kv_row_int4(k_int4, k_scales, kv_row.data(), nkvh, dh);

        fill_random(kv_row.data(), nkvh * dh, rng);
        float* v_fp32 = v_pool_fp32.data() + bi * (fp32_block_stride / sizeof(float)) + off * nkvh * dh;
        std::memcpy(v_fp32, kv_row.data(), nkvh * dh * sizeof(float));

        uint8_t* v_int4 = reinterpret_cast<uint8_t*>(v_pool_int4.data() + bi * int4_block_stride) + off * nkvh * packed_dh;
        float* v_scales = reinterpret_cast<float*>(v_pool_int4.data() + bi * int4_block_stride + int4_data_per_layer) + off * nkvh;
        quantize_kv_row_int4(v_int4, v_scales, kv_row.data(), nkvh, dh);
    }

    std::vector<float> query(batch * nh * dh);
    fill_random(query.data(), query.size(), rng);

    std::vector<float> out_fp32(batch * nh * dh, 0.0f);
    std::vector<float> out_int4(batch * nh * dh, 0.0f);

    paged_attention(out_fp32.data(), query.data(),
                    k_pool_fp32.data(), v_pool_fp32.data(),
                    block_tables.data(), seq_lens.data(),
                    batch, nh, nkvh, dh, bs, max_blocks,
                    fp32_block_stride, fp32_layer_stride,
                    0, scale, LLAISYS_DEVICE_CPU, KVQuantMode::FP32);

    paged_attention(out_int4.data(), query.data(),
                    k_pool_int4.data(), v_pool_int4.data(),
                    block_tables.data(), seq_lens.data(),
                    batch, nh, nkvh, dh, bs, max_blocks,
                    int4_block_stride, int4_layer_stride,
                    0, scale, LLAISYS_DEVICE_CPU, KVQuantMode::INT4);

    float err = max_abs_error(out_fp32.data(), out_int4.data(), batch * nh * dh);
    CHECK(err < 0.2f, "INT4 paged attention max error < 0.2 (got " + std::to_string(err) + ")");

    double fp32_mem = max_blocks * fp32_block_stride * 2.0;
    double int4_mem = max_blocks * int4_block_stride * 2.0;
    double savings = 100.0 * (1.0 - int4_mem / fp32_mem);
    std::cout << "  Memory: FP32=" << fp32_mem / 1024 << "KB INT4=" << int4_mem / 1024
              << "KB savings=" << savings << "%" << std::endl;
    CHECK(savings > 70.0, "INT4 memory savings > 70%");
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  INT8/INT4 KV-Cache Quantization Tests" << std::endl;
    std::cout << "================================================================" << std::endl;

    test_int8_roundtrip();
    test_int4_roundtrip();
    test_paged_attention_int8();
    test_paged_attention_int4();

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Results: " << g_pass << " PASSED, " << g_fail << " FAILED" << std::endl;
    std::cout << "================================================================" << std::endl;

    return g_fail > 0 ? 1 : 0;
}

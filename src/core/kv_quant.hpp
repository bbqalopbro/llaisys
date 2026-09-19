#include <llmops/ops/kv_quant.hpp>
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>

namespace llaisys::core {

enum class KVQuantType : int {
    NONE = 0,  // FP32, 4 bytes
    INT8 = 1,  // per-head symmetric INT8, 1 byte + scale
    INT4 = 2,  // per-head symmetric INT4 packed, 0.5 byte + scale
};

inline size_t kv_quant_elem_size(KVQuantType qt) {
    switch (qt) {
        case KVQuantType::INT8: return 1;
        case KVQuantType::INT4: return 1; // packed 2-per-byte; block allocator treats as half
        default: return 4;
    }
}

// Per-head scale stored alongside quantized data.
// Layout: [block_size, nkvh, dh] quantized data + [block_size, nkvh] FP32 scales
inline size_t kv_quant_block_bytes(KVQuantType qt, size_t block_size,
                                    size_t nkvh, size_t dh) {
    switch (qt) {
        case KVQuantType::INT8:
            return block_size * nkvh * dh * sizeof(int8_t) +
                   block_size * nkvh * sizeof(float);
        case KVQuantType::INT4:
            return block_size * nkvh * ((dh + 1) / 2) * sizeof(uint8_t) +
                   block_size * nkvh * sizeof(float);
        default:
            return block_size * nkvh * dh * sizeof(float);
    }
}

using llmops::cpu::quantize_fp32_to_int8;
using llmops::cpu::dequantize_int8_to_fp32;
using llmops::cpu::quantize_fp32_to_int4;
using llmops::cpu::dequantize_int4_to_fp32;
using llmops::cpu::quantize_kv_row_int8;
using llmops::cpu::dequantize_kv_row_int8;
using llmops::cpu::quantize_kv_row_int4;
using llmops::cpu::dequantize_kv_row_int4;
}

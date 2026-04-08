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

// Quantize a single KV head vector [dh] from FP32 to INT8.
// Returns the scale factor.
inline float quantize_fp32_to_int8(int8_t *dst, const float *src, size_t dh) {
    float amax = 0.0f;
    for (size_t d = 0; d < dh; ++d)
        amax = std::max(amax, std::abs(src[d]));

    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (127.0f / amax) : 0.0f;

    for (size_t d = 0; d < dh; ++d) {
        float v = src[d] * inv_scale;
        v = std::max(-127.0f, std::min(127.0f, std::round(v)));
        dst[d] = static_cast<int8_t>(v);
    }
    return scale;
}

// Dequantize INT8 head vector back to FP32.
inline void dequantize_int8_to_fp32(float *dst, const int8_t *src,
                                     float scale, size_t dh) {
    for (size_t d = 0; d < dh; ++d)
        dst[d] = static_cast<float>(src[d]) * scale;
}

// Quantize FP32 to INT4 packed (2 values per byte, symmetric).
// Returns the scale factor.
inline float quantize_fp32_to_int4(uint8_t *dst, const float *src, size_t dh) {
    float amax = 0.0f;
    for (size_t d = 0; d < dh; ++d)
        amax = std::max(amax, std::abs(src[d]));

    float scale = amax / 7.0f;
    float inv_scale = (scale > 0.0f) ? (7.0f / amax) : 0.0f;

    for (size_t d = 0; d < dh; d += 2) {
        float v0 = src[d] * inv_scale;
        v0 = std::max(-7.0f, std::min(7.0f, std::round(v0)));
        int8_t q0 = static_cast<int8_t>(v0);

        int8_t q1 = 0;
        if (d + 1 < dh) {
            float v1 = src[d + 1] * inv_scale;
            v1 = std::max(-7.0f, std::min(7.0f, std::round(v1)));
            q1 = static_cast<int8_t>(v1);
        }

        // Pack: low nibble = q0 + 8, high nibble = q1 + 8
        dst[d / 2] = static_cast<uint8_t>(((q1 + 8) << 4) | ((q0 + 8) & 0x0F));
    }
    return scale;
}

// Dequantize INT4 packed back to FP32.
inline void dequantize_int4_to_fp32(float *dst, const uint8_t *src,
                                     float scale, size_t dh) {
    for (size_t d = 0; d < dh; d += 2) {
        uint8_t packed = src[d / 2];
        int8_t q0 = static_cast<int8_t>((packed & 0x0F)) - 8;
        int8_t q1 = static_cast<int8_t>((packed >> 4)) - 8;
        dst[d] = static_cast<float>(q0) * scale;
        if (d + 1 < dh)
            dst[d + 1] = static_cast<float>(q1) * scale;
    }
}

// Quantize a full KV row [nkvh, dh] with per-head scales.
inline void quantize_kv_row_int8(int8_t *q_data, float *scales,
                                  const float *src, size_t nkvh, size_t dh) {
    for (size_t h = 0; h < nkvh; ++h)
        scales[h] = quantize_fp32_to_int8(q_data + h * dh, src + h * dh, dh);
}

inline void dequantize_kv_row_int8(float *dst, const int8_t *q_data,
                                    const float *scales, size_t nkvh, size_t dh) {
    for (size_t h = 0; h < nkvh; ++h)
        dequantize_int8_to_fp32(dst + h * dh, q_data + h * dh, scales[h], dh);
}

inline void quantize_kv_row_int4(uint8_t *q_data, float *scales,
                                  const float *src, size_t nkvh, size_t dh) {
    size_t packed_dh = (dh + 1) / 2;
    for (size_t h = 0; h < nkvh; ++h)
        scales[h] = quantize_fp32_to_int4(q_data + h * packed_dh, src + h * dh, dh);
}

inline void dequantize_kv_row_int4(float *dst, const uint8_t *q_data,
                                    const float *scales, size_t nkvh, size_t dh) {
    size_t packed_dh = (dh + 1) / 2;
    for (size_t h = 0; h < nkvh; ++h)
        dequantize_int4_to_fp32(dst + h * dh, q_data + h * packed_dh, scales[h], dh);
}

} // namespace llaisys::core

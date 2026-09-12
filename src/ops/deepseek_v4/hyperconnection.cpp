#include "hyperconnection.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/hyperconnection.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace llaisys::ops {

static float sigmoid(float value) {
    return value >= 0.0F ? 1.0F / (1.0F + std::exp(-value))
                         : std::exp(value) / (1.0F + std::exp(value));
}

static void cpu_reference(
    float *pre, float *post, float *combination,
    const float *mixes, const float *scale, const float *base,
    int rows, int hc, int iterations, float eps) {
    const int width = (2 + hc) * hc;
    for (int row = 0; row < rows; ++row) {
        const float *input = mixes + row * width;
        float *row_pre = pre + row * hc;
        float *row_post = post + row * hc;
        float *matrix = combination + row * hc * hc;
        for (int j = 0; j < hc; ++j) {
            row_pre[j] = sigmoid(input[j] * scale[0] + base[j]) + eps;
            row_post[j] = 2.0F * sigmoid(
                input[hc + j] * scale[1] + base[hc + j]);
        }
        for (int j = 0; j < hc; ++j) {
            float maximum = -INFINITY;
            for (int k = 0; k < hc; ++k) {
                const int index = 2 * hc + j * hc + k;
                matrix[j * hc + k] = input[index] * scale[2] + base[index];
                maximum = std::max(maximum, matrix[j * hc + k]);
            }
            float sum = 0.0F;
            for (int k = 0; k < hc; ++k) {
                matrix[j * hc + k] = std::exp(matrix[j * hc + k] - maximum);
                sum += matrix[j * hc + k];
            }
            for (int k = 0; k < hc; ++k) matrix[j * hc + k] = matrix[j * hc + k] / sum + eps;
        }
        std::vector<float> sums(hc);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const bool normalize_rows = iteration > 0;
            if (normalize_rows) {
                for (int j = 0; j < hc; ++j) {
                    float sum = 0.0F;
                    for (int k = 0; k < hc; ++k) sum += matrix[j * hc + k];
                    for (int k = 0; k < hc; ++k) matrix[j * hc + k] /= sum + eps;
                }
            }
            std::fill(sums.begin(), sums.end(), 0.0F);
            for (int j = 0; j < hc; ++j)
                for (int k = 0; k < hc; ++k) sums[k] += matrix[j * hc + k];
            for (int j = 0; j < hc; ++j)
                for (int k = 0; k < hc; ++k) matrix[j * hc + k] /= sums[k] + eps;
        }
    }
}

void deepseek_v4_hyperconnection_split_reference(
    float *pre, float *post, float *combination,
    const float *mixes, const float *scale, const float *base,
    int rows, int hc_mult, int iterations, float eps,
    llaisysDeviceType_t device_type) {
    if (!pre || !post || !combination || !mixes || !scale || !base ||
        rows <= 0 || hc_mult <= 0 || hc_mult > 8 || iterations <= 0) {
        throw std::invalid_argument("invalid DeepSeek-V4 hyperconnection arguments");
    }
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::deepseek_v4_hyperconnection_split_reference(
            pre, post, combination, mixes, scale, base, rows, hc_mult,
            iterations, eps);
    }
#endif
    if (device_type != LLAISYS_DEVICE_CPU) {
        throw std::invalid_argument("DeepSeek-V4 hyperconnection device is unavailable");
    }
    cpu_reference(pre, post, combination, mixes, scale, base, rows, hc_mult,
                  iterations, eps);
}

} // namespace llaisys::ops

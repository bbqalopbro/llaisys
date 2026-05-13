#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/sample_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/sample_metax.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace llaisys::ops {

static void sample_cpu_kernel(tensor_t out_idx, tensor_t logits,
                              float temperature, int top_k, float top_p,
                              uint64_t seed) {
    const float *src = reinterpret_cast<const float *>(logits->data());
    int *dst = reinterpret_cast<int *>(out_idx->data());

    // Determine vocab size – support [N] and [1, N]
    size_t vocab = 0;
    if (logits->ndim() == 1) {
        vocab = logits->shape()[0];
    } else if (logits->ndim() == 2) {
        if (logits->shape()[0] != 1)
            throw std::runtime_error("sample: 2D input must have shape [1, N]");
        vocab = logits->shape()[1];
    } else {
        throw std::runtime_error("sample: only supports 1D [N] or 2D [1, N]");
    }

    // Copy logits for manipulation
    std::vector<float> logit_vec(vocab);
    if (logits->isContiguous()) {
        std::memcpy(logit_vec.data(), src, vocab * sizeof(float));
    } else {
        ptrdiff_t stride = (logits->ndim() == 1) ? logits->strides()[0]
                                                  : logits->strides()[1];
        for (size_t i = 0; i < vocab; ++i) {
            logit_vec[i] = src[i * stride];
        }
    }

    // --- 1.temperature 缩放 ：logits[i] /= temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_t = 1.0f / temperature;
        for (size_t i = 0; i < vocab; ++i) {
            logit_vec[i] *= inv_t;
        }
    }

    // Build index array sorted by logit descending
    std::vector<int> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(),
                      indices.begin() + std::min((size_t)top_k, vocab),
                      indices.end(),
                      [&](int a, int b) { return logit_vec[a] > logit_vec[b]; });

    // --- 2. Top-K：只保留概率最高的 k 个候选词
    size_t k = (top_k > 0 && (size_t)top_k < vocab) ? (size_t)top_k : vocab;

    // --- 3. Softmax：对 top-k 个候选做 softmax → 概率分布
    float max_logit = logit_vec[indices[0]];
    std::vector<float> probs(k);
    float sum = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        probs[i] = std::exp(logit_vec[indices[i]] - max_logit);
        sum += probs[i];
    }
    for (size_t i = 0; i < k; ++i) {
        probs[i] /= sum;
    }

    // --- 4. Top-P ：从概率最高的开始累加，直到累积概率 ≥ p 时截止
    // 例：p=0.9 → 保留概率累积到 90% 的最少候选集
    // 然后重新归一化
    size_t cutoff = k;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    // Re-normalize after top-p truncation
    if (cutoff < k) {
        float new_sum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            new_sum += probs[i];
        }
        for (size_t i = 0; i < cutoff; ++i) {
            probs[i] /= new_sum;
        }
    }

    // --- 5. Random sampling ---
    std::mt19937 rng(seed);
    std::discrete_distribution<int> dist(probs.begin(),
                                          probs.begin() + cutoff);
    int sampled = dist(rng);
    dst[0] = indices[sampled];
}

void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k,
            float top_p, uint64_t seed) {
    auto dtype = logits->dtype();

#ifdef ENABLE_NVIDIA_API
    if (logits->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::sample(out_idx, logits, temperature, top_k, top_p, seed);
    }
#endif

#ifdef ENABLE_METAX_API
    if (logits->deviceType() == LLAISYS_DEVICE_METAX) {
        return metax::sample(out_idx, logits, temperature, top_k, top_p, seed);
    }
#endif

    if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        sample_cpu_kernel(out_idx, logits, temperature, top_k, top_p, seed);
    } else {
        throw std::runtime_error(
            "sample: only F32 logits are supported currently");
    }
}

} // namespace llaisys::ops

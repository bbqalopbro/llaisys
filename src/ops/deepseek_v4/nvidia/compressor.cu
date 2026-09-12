#include "compressor.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace llaisys::ops::nvidia {

// Each thread owns one feature in both state halves. No cross-thread state
// dependency: a sequence can cross any number of compression boundaries.
__global__ static void compress_projected_kernel(
    float *output, float *kv_state, float *score_state,
    const float *kv, const float *score, const float *ape,
    int sequence, int dimension, int ratio, int start_pos) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= dimension) return;
    const int batch = blockIdx.y;
    const bool overlap = ratio == 4;
    const int coefficient = overlap ? 2 : 1;
    const int width = coefficient * dimension;
    const int state_rows = coefficient * ratio;
    const int current_offset = overlap ? ratio : 0;
    const int output_groups = (start_pos + sequence) / ratio - start_pos / ratio;
    float *state_kv = kv_state + batch * state_rows * width;
    float *state_score = score_state + batch * state_rows * width;
    if (start_pos == 0) {
        for (int row = 0; row < state_rows; ++row) {
            for (int half = 0; half < coefficient; ++half) {
                const int index = row * width + half * dimension + feature;
                state_kv[index] = 0.0F;
                state_score[index] = -INFINITY;
            }
        }
    }
    int emitted = 0;
    for (int token = 0; token < sequence; ++token) {
        const int position = start_pos + token;
        const int slot = position % ratio;
        for (int half = 0; half < coefficient; ++half) {
            const int column = half * dimension + feature;
            const int source = (batch * sequence + token) * width + column;
            const int destination = (current_offset + slot) * width + column;
            state_kv[destination] = kv[source];
            state_score[destination] = score[source] + ape[slot * width + column];
        }
        if ((position + 1) % ratio != 0) continue;
        float maximum = -INFINITY;
        for (int row = 0; row < state_rows; ++row) {
            const int column = feature + (overlap && row >= ratio ? dimension : 0);
            maximum = fmaxf(maximum, state_score[row * width + column]);
        }
        float denominator = 0.0F;
        for (int row = 0; row < state_rows; ++row) {
            const int column = feature + (overlap && row >= ratio ? dimension : 0);
            denominator += expf(state_score[row * width + column] - maximum);
        }
        // Match the published PyTorch FP32 sum's four independent
        // accumulators. At ratio 128 it also splits rows across four lanes
        // before a pairwise lane reduction. A serial sum can cross a BF16
        // rounding midpoint and perturb routing several layers later.
        const int lanes = overlap ? 1 : 4;
        float partial[4][4] = {};
        for (int row = 0; row < state_rows; ++row) {
            const int column = feature + (overlap && row >= ratio ? dimension : 0);
            const int index = row * width + column;
            const float probability = expf(state_score[index] - maximum) / denominator;
            // Keep the separate multiply and add of the published FP32 path.
            const int lane = row % lanes;
            const int accumulator = (row / lanes) % 4;
            partial[lane][accumulator] = __fadd_rn(
                partial[lane][accumulator], __fmul_rn(state_kv[index], probability));
        }
        float lane_sum[4];
        for (int lane = 0; lane < lanes; ++lane) {
            lane_sum[lane] = partial[lane][0];
            for (int accumulator = 1; accumulator < 4; ++accumulator)
                lane_sum[lane] = __fadd_rn(lane_sum[lane], partial[lane][accumulator]);
        }
        const float pooled = overlap ? lane_sum[0] : __fadd_rn(
            __fadd_rn(lane_sum[0], lane_sum[2]),
            __fadd_rn(lane_sum[1], lane_sum[3]));
        output[(batch * output_groups + emitted) * dimension + feature] = pooled;
        ++emitted;
        if (overlap) {
            for (int row = 0; row < ratio; ++row) {
                for (int half = 0; half < coefficient; ++half) {
                    const int index = row * width + half * dimension + feature;
                    state_kv[index] = state_kv[index + ratio * width];
                    state_score[index] = state_score[index + ratio * width];
                }
            }
        }
    }
}

void deepseek_v4_compress_projected_reference(
    float *output, float *kv_state, float *score_state,
    const float *kv, const float *score, const float *ape,
    int batch, int sequence, int dimension, int ratio, int start_pos,
    void *stream) {
    constexpr int threads = 128;
    compress_projected_kernel<<<dim3((dimension + threads - 1) / threads, batch),
                               threads, 0, static_cast<cudaStream_t>(stream)>>>(
        output, kv_state, score_state, kv, score, ape,
        sequence, dimension, ratio, start_pos);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}

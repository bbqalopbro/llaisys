#include "hyperconnection.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

__device__ static float sigmoid(float value) {
    return value >= 0.0F ? 1.0F / (1.0F + expf(-value))
                         : expf(value) / (1.0F + expf(value));
}

__global__ static void hyperconnection_kernel(
    float *pre, float *post, float *combination,
    const float *mixes, const float *scale, const float *base,
    int rows, int hc, int iterations, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int width = (2 + hc) * hc;
    const float *input = mixes + row * width;
    float matrix[64];
    for (int j = 0; j < hc; ++j) {
        pre[row * hc + j] = sigmoid(input[j] * scale[0] + base[j]) + eps;
        post[row * hc + j] = 2.0F * sigmoid(
            input[hc + j] * scale[1] + base[hc + j]);
    }
    for (int j = 0; j < hc; ++j) {
        float maximum = -INFINITY;
        for (int k = 0; k < hc; ++k) {
            const int index = 2 * hc + j * hc + k;
            matrix[j * hc + k] = input[index] * scale[2] + base[index];
            maximum = fmaxf(maximum, matrix[j * hc + k]);
        }
        float sum = 0.0F;
        for (int k = 0; k < hc; ++k) {
            matrix[j * hc + k] = expf(matrix[j * hc + k] - maximum);
            sum += matrix[j * hc + k];
        }
        for (int k = 0; k < hc; ++k) matrix[j * hc + k] = matrix[j * hc + k] / sum + eps;
    }
    float sums[8];
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (iteration > 0) {
            for (int j = 0; j < hc; ++j) {
                float sum = 0.0F;
                for (int k = 0; k < hc; ++k) sum += matrix[j * hc + k];
                for (int k = 0; k < hc; ++k) matrix[j * hc + k] /= sum + eps;
            }
        }
        for (int k = 0; k < hc; ++k) sums[k] = 0.0F;
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) sums[k] += matrix[j * hc + k];
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) matrix[j * hc + k] /= sums[k] + eps;
    }
    for (int i = 0; i < hc * hc; ++i) combination[row * hc * hc + i] = matrix[i];
}

void deepseek_v4_hyperconnection_split_reference(
    float *pre, float *post, float *combination,
    const float *mixes, const float *scale, const float *base,
    int rows, int hc_mult, int iterations, float eps) {
    constexpr int threads = 128;
    hyperconnection_kernel<<<(rows + threads - 1) / threads, threads>>>(
        pre, post, combination, mixes, scale, base, rows, hc_mult, iterations,
        eps);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia

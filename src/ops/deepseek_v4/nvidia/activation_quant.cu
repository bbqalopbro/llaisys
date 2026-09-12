#include "activation_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

__device__ __constant__ float fp4_e2m1_table[16] = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
};

__global__ static void activation_quant_kernel(
    __nv_bfloat16 *data, int columns, int block_size,
    int quant_mode, bool power_of_two_scale) {
    const int group = blockIdx.x;
    const int groups_per_row = columns / block_size;
    const int row = group / groups_per_row;
    const int column = (group % groups_per_row) * block_size + threadIdx.x;
    __shared__ float maximum[128];
    float value = 0.0F;
    if (threadIdx.x < block_size) {
        value = __bfloat162float(data[row * columns + column]);
    }
    maximum[threadIdx.x] = threadIdx.x < block_size ? fabsf(value) : 0.0F;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(
                maximum[threadIdx.x], maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x >= block_size) return;
    const float quant_max = quant_mode == 0 ? 448.0F : 6.0F;
    const float minimum = quant_mode == 0 ? 1.0e-4F : 6.0F * exp2f(-126.0F);
    float scale = fmaxf(maximum[0], minimum) / quant_max;
    if (power_of_two_scale) scale = exp2f(ceilf(log2f(scale)));
    const float normalized = fminf(quant_max, fmaxf(-quant_max, value / scale));
    float quantized;
    if (quant_mode == 0) {
        quantized = static_cast<float>(__nv_fp8_e4m3(normalized));
    } else {
        // The published PyTorch QAT oracle uses argmin over this ordered
        // table. At exact midpoints that differs from the hardware FP4
        // conversion's ties-to-even rule, so preserve the checkpoint rule.
        int nearest = 0;
        float distance = fabsf(normalized - fp4_e2m1_table[0]);
        for (int index = 1; index < 16; ++index) {
            const float candidate = fabsf(normalized - fp4_e2m1_table[index]);
            if (candidate < distance) {
                distance = candidate;
                nearest = index;
            }
        }
        quantized = fp4_e2m1_table[nearest];
    }
    data[row * columns + column] = __float2bfloat16(quantized * scale);
}

void deepseek_v4_activation_quant_reference(
    void *data, int rows, int columns, int block_size,
    int quant_mode, int power_of_two_scale) {
    if (block_size > 128) {
        throw std::invalid_argument("activation quant block_size exceeds 128");
    }
    const int groups = rows * columns / block_size;
    activation_quant_kernel<<<groups, 128>>>(
        static_cast<__nv_bfloat16 *>(data), columns, block_size,
        quant_mode, power_of_two_scale != 0);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia

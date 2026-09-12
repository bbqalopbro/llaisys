#include "quantized_linear.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

__device__ __constant__ float fp4_weight_table[16] = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
};

__device__ static float e8m0_scale(unsigned char value) {
    return value == 255 ? NAN : ldexpf(1.0F, static_cast<int>(value) - 127);
}

__device__ static float fp8_weight_value(
    const unsigned char *weight, const unsigned char *scale,
    int output, int input, int out_features, int in_features) {
    __nv_fp8_e4m3 value;
    value.__x = weight[output * in_features + input];
    const int scale_columns = (in_features + 127) / 128;
    const int scale_index = (output / 128) * scale_columns + input / 128;
    return __bfloat162float(__float2bfloat16(
        static_cast<float>(value) * e8m0_scale(scale[scale_index])));
}

__device__ static float fp4_weight_value(
    const unsigned char *weight, const unsigned char *scale,
    int output, int input, int in_features) {
    const unsigned char packed = weight[output * (in_features / 2) + input / 2];
    const int nibble = input & 1 ? packed >> 4 : packed & 0x0F;
    return __bfloat162float(__float2bfloat16(
        fp4_weight_table[nibble] *
        e8m0_scale(scale[output * (in_features / 32) + input / 32])));
}

__global__ static void quantized_linear_kernel(
    __nv_bfloat16 *output, const __nv_bfloat16 *input,
    const unsigned char *weight, const unsigned char *scale,
    int out_features, int in_features, int quant_mode) {
    const int output_feature = blockIdx.x;
    const int row = blockIdx.y;
    float partial = 0.0F;
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
        const float activation = __bfloat162float(input[row * in_features + k]);
        const float value = quant_mode == 0
            ? fp8_weight_value(weight, scale, output_feature, k,
                               out_features, in_features)
            : fp4_weight_value(weight, scale, output_feature, k, in_features);
        partial += activation * value;
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row * out_features + output_feature] =
            __float2bfloat16(reduction[0]);
    }
}

void deepseek_v4_quantized_linear_reference(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode) {
    if ((quant_mode == 0 && (out_features % 128 || in_features % 128)) ||
        (quant_mode == 1 && in_features % 32)) {
        throw std::invalid_argument("quantized linear dimensions are not block aligned");
    }
    quantized_linear_kernel<<<dim3(out_features, rows), 256>>>(
        static_cast<__nv_bfloat16 *>(output),
        static_cast<const __nv_bfloat16 *>(input),
        static_cast<const unsigned char *>(weight),
        static_cast<const unsigned char *>(scale),
        out_features, in_features, quant_mode);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ static void decode_quantized_weight_kernel(
    float *decoded, const unsigned char *weight, const unsigned char *scale,
    int elements, int out_features, int in_features, int quant_mode) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x) {
        const int output = index / in_features;
        const int input = index - output * in_features;
        if (quant_mode == 0) {
            __nv_fp8_e4m3 value;
            value.__x = weight[index];
            const int scale_columns = in_features / 128;
            const int scale_index = (output / 128) * scale_columns + input / 128;
            decoded[index] = static_cast<float>(value) * e8m0_scale(scale[scale_index]);
        } else {
            const unsigned char packed =
                weight[output * (in_features / 2) + input / 2];
            const int nibble = input & 1 ? packed >> 4 : packed & 0x0F;
            decoded[index] = fp4_weight_table[nibble] *
                e8m0_scale(scale[output * (in_features / 32) + input / 32]);
        }
    }
}

__global__ static void bf16_to_float_kernel(
    float *output, const __nv_bfloat16 *input, int elements) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x) {
        output[index] = __bfloat162float(input[index]);
    }
}

__global__ static void float_to_bf16_kernel(
    __nv_bfloat16 *output, const float *input, int elements) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x) {
        output[index] = __float2bfloat16(input[index]);
    }
}

struct CublasLinearWorkspace {
    float *weight = nullptr;
    float *input = nullptr;
    float *output = nullptr;
    size_t weight_capacity = 0;
    size_t input_capacity = 0;
    size_t output_capacity = 0;
    cublasHandle_t handle = nullptr;
    int device = -1;
};

static CublasLinearWorkspace &cublas_workspace() {
    // The current bring-up path is deliberately single-rank/single-thread.
    // Keeping the largest buffers avoids tens of thousands of cudaMalloc calls
    // during expert execution. Device allocations live until process exit.
    static thread_local CublasLinearWorkspace workspace;
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (workspace.device != -1 && workspace.device != device) {
        throw std::runtime_error("cuBLAS correctness workspace cannot change CUDA device");
    }
    workspace.device = device;
    if (!workspace.handle) {
        cublasStatus_t status = cublasCreate(&workspace.handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("failed to create cuBLAS handle");
        }
        // Precision validation uses full FP32 accumulation, not TF32.
        status = cublasSetMathMode(workspace.handle, CUBLAS_PEDANTIC_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("failed to select pedantic cuBLAS math mode");
        }
    }
    return workspace;
}

static void reserve_float_buffer(float **buffer, size_t *capacity, size_t elements) {
    if (*capacity >= elements) return;
    if (*buffer) {
        cudaError_t error = cudaFree(*buffer);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(buffer), elements * sizeof(float));
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    *capacity = elements;
}

void deepseek_v4_quantized_linear_cublas(
    void *output, const void *input, const void *weight, const void *scale,
    int rows, int out_features, int in_features, int quant_mode) {
    if ((quant_mode == 0 && (out_features % 128 || in_features % 128)) ||
        (quant_mode == 1 && in_features % 32)) {
        throw std::invalid_argument("cuBLAS linear dimensions are not block aligned");
    }
    CublasLinearWorkspace &workspace = cublas_workspace();
    const size_t weight_elements =
        static_cast<size_t>(out_features) * static_cast<size_t>(in_features);
    const size_t input_elements =
        static_cast<size_t>(rows) * static_cast<size_t>(in_features);
    const size_t output_elements =
        static_cast<size_t>(rows) * static_cast<size_t>(out_features);
    reserve_float_buffer(&workspace.weight, &workspace.weight_capacity, weight_elements);
    reserve_float_buffer(&workspace.input, &workspace.input_capacity, input_elements);
    reserve_float_buffer(&workspace.output, &workspace.output_capacity, output_elements);

    constexpr int threads = 256;
    const int weight_blocks = static_cast<int>((weight_elements + threads - 1) / threads);
    const int input_blocks = static_cast<int>((input_elements + threads - 1) / threads);
    decode_quantized_weight_kernel<<<weight_blocks, threads>>>(
        workspace.weight, static_cast<const unsigned char *>(weight),
        static_cast<const unsigned char *>(scale), static_cast<int>(weight_elements),
        out_features, in_features, quant_mode);
    bf16_to_float_kernel<<<input_blocks, threads>>>(
        workspace.input, static_cast<const __nv_bfloat16 *>(input),
        static_cast<int>(input_elements));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));

    // Row-major Y[rows,N] = X[rows,K] W[N,K]^T is the column-major
    // equivalent Y^T[N,rows] = W[N,K] X^T[K,rows].
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublasStatus_t status = cublasSgemm(
        workspace.handle, CUBLAS_OP_T, CUBLAS_OP_N,
        out_features, rows, in_features, &alpha,
        workspace.weight, in_features, workspace.input, in_features,
        &beta, workspace.output, out_features);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("DeepSeek-V4 cuBLAS SGEMM failed");
    }
    const int output_blocks = static_cast<int>((output_elements + threads - 1) / threads);
    float_to_bf16_kernel<<<output_blocks, threads>>>(
        static_cast<__nv_bfloat16 *>(output), workspace.output,
        static_cast<int>(output_elements));
    error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia

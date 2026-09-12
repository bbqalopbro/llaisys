#include "indexer.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace llaisys::ops::nvidia {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Indexer cuBLAS operation failed");
}

struct BlasHandle {
    cublasHandle_t handle = nullptr;
    int device = 0;
    BlasHandle() {
        check(cudaGetDevice(&device));
        check(cublasCreate(&handle));
    }
    ~BlasHandle() {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cublasDestroy(handle);
        cudaSetDevice(previous);
    }
};
cublasHandle_t blas_handle(cudaStream_t stream) {
    // No buffers are shared between streams; cuBLAS handles are per host
    // thread/device and rebound to the caller's stream before each launch.
    static thread_local std::unordered_map<int, std::unique_ptr<BlasHandle>> handles;
    int device = 0;
    check(cudaGetDevice(&device));
    auto &context = handles[device];
    if (!context) context = std::make_unique<BlasHandle>();
    check(cublasSetStream(context->handle, stream));
    check(cublasSetMathMode(context->handle, CUBLAS_DEFAULT_MATH));
    return context->handle;
}

int floor_pow2(int value) {
    int result = 1;
    while (result <= value / 2) result *= 2;
    return result;
}
int reduction_lanes(int rows, int candidates) {
    if (candidates == 1) return 64;
    // Published BF16 sum over H=64, using PyTorch's four accumulators and
    // its output-vectorization / cross-warp reduction shape convention.
    const int vec = candidates % 4 == 0 ? 4 : candidates % 2 == 0 ? 2 : 1;
    const int maximum = 512 / vec;
    const int dim0 = std::min(floor_pow2(rows * candidates / vec), maximum);
    const int width = std::min(dim0, 32);
    const int height = std::min(64, maximum / width);
    return 64 >= std::min(height * 16, 256) ? height : 1;
}

__global__ void reduce_heads(
    float *scores, const __nv_bfloat16 *dots, const __nv_bfloat16 *weights,
    int rows, int candidates, int lanes) {
    for (int item = blockIdx.x * blockDim.x + threadIdx.x;
         item < rows * candidates; item += blockDim.x * gridDim.x) {
        const int row = item / candidates;
        const int candidate = item % candidates;
        float products[64];
        for (int head = 0; head < 64; ++head) {
            const float dot = __bfloat162float(dots[(row * 64 + head) * candidates + candidate]);
            const float weight = __bfloat162float(weights[row * 64 + head]);
            // The published multiplication materializes BF16 before sum(H).
            products[head] = __bfloat162float(__float2bfloat16_rn(fmaxf(dot, 0.0F) * weight));
        }
        float result;
        if (lanes == 64) {
            for (int stride = 32; stride > 0; stride /= 2)
                for (int h = 0; h < stride; ++h)
                    products[h] = __fadd_rn(products[h], products[h + stride]);
            result = products[0];
        } else {
            float partial[4][4] = {};
            for (int h = 0; h < 64; ++h) {
                const int lane = h % lanes;
                const int accumulator = h / lanes % 4;
                partial[lane][accumulator] = __fadd_rn(partial[lane][accumulator], products[h]);
            }
            float sums[4];
            for (int lane = 0; lane < lanes; ++lane) {
                sums[lane] = partial[lane][0];
                for (int j = 1; j < 4; ++j) sums[lane] = __fadd_rn(sums[lane], partial[lane][j]);
            }
            for (int stride = lanes / 2; stride > 0; stride /= 2)
                for (int lane = 0; lane < stride; ++lane)
                    sums[lane] = __fadd_rn(sums[lane], sums[lane + stride]);
            result = sums[0];
        }
        scores[item] = __bfloat162float(__float2bfloat16_rn(result));
    }
}

size_t aligned(size_t bytes) { return (bytes + 255) / 256 * 256; }
size_t sort_storage(int rows, int candidates) {
    size_t bytes = 0;
    check(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, bytes, static_cast<const float *>(nullptr), static_cast<float *>(nullptr),
        static_cast<const int *>(nullptr), static_cast<int *>(nullptr),
        rows * candidates, rows, static_cast<const int *>(nullptr), static_cast<const int *>(nullptr)));
    return bytes;
}

__global__ void prepare_sort(
    float *masked, int *ids, int *offsets, const float *scores,
    int rows, int sequence, int candidates, int start_pos, int ratio) {
    const int count = rows * candidates;
    for (int item = blockIdx.x * blockDim.x + threadIdx.x;
         item < count; item += blockDim.x * gridDim.x) {
        const int row = item / candidates;
        const int candidate = item % candidates;
        const int valid = (start_pos + row % sequence + 1) / ratio;
        masked[item] = candidate < valid ? scores[item] : -INFINITY;
        ids[item] = candidate;
        if (candidate == 0) offsets[row] = row * candidates;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) offsets[rows] = count;
}

__global__ void gather_topk(
    int *output, const int *sorted_ids, int rows, int sequence, int candidates,
    int topk, int start_pos, int ratio, int index_offset) {
    for (int item = blockIdx.x * blockDim.x + threadIdx.x;
         item < rows * topk; item += blockDim.x * gridDim.x) {
        const int row = item / topk;
        const int candidate = sorted_ids[row * candidates + item % topk];
        const int valid = (start_pos + row % sequence + 1) / ratio;
        output[item] = candidate < valid ? candidate + index_offset : -1;
    }
}
} // namespace

void deepseek_v4_indexer_scores_cublas(
    float *scores, void *dots, const void *query, const void *latent,
    const void *weights, int batch, int sequence, int candidates, void *raw_stream) {
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const auto handle = blas_handle(stream);
    const float alpha = 1.0F, beta = 0.0F;
    // Q[B,S*64,128] @ K[B,T,128]^T -> BF16 dots[B,S*64,T].
    check(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, candidates, sequence * 64, 128,
        &alpha, latent, CUDA_R_16BF, 128, static_cast<long long>(candidates) * 128,
        query, CUDA_R_16BF, 128, static_cast<long long>(sequence) * 64 * 128,
        &beta, dots, CUDA_R_16BF, candidates, static_cast<long long>(sequence) * 64 * candidates,
        batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    const int rows = batch * sequence;
    const int blocks = std::min((rows * candidates + 255) / 256, 65535);
    reduce_heads<<<blocks, 256, 0, stream>>>(
        scores, static_cast<const __nv_bfloat16 *>(dots),
        static_cast<const __nv_bfloat16 *>(weights), rows, candidates,
        reduction_lanes(rows, candidates));
    check(cudaGetLastError());
}

size_t deepseek_v4_indexer_topk_workspace(int rows, int candidates) {
    return 4 * aligned(static_cast<size_t>(rows) * candidates * 4)
         + aligned(static_cast<size_t>(rows + 1) * 4) + sort_storage(rows, candidates);
}

void deepseek_v4_indexer_topk_cub(
    int *indices, const float *scores, void *workspace, size_t workspace_bytes,
    int batch, int sequence, int candidates, int topk, int start_pos,
    int ratio, int index_offset, void *raw_stream) {
    const int rows = batch * sequence;
    const size_t array_bytes = aligned(static_cast<size_t>(rows) * candidates * 4);
    const size_t offset_bytes = aligned(static_cast<size_t>(rows + 1) * 4);
    const size_t fixed_bytes = 4 * array_bytes + offset_bytes;
    const size_t required = fixed_bytes + sort_storage(rows, candidates);
    if (workspace_bytes < required) throw std::invalid_argument("Indexer sort workspace is too small");
    auto *base = static_cast<unsigned char *>(workspace);
    auto *masked = reinterpret_cast<float *>(base);
    auto *sorted = reinterpret_cast<float *>(base + array_bytes);
    auto *ids = reinterpret_cast<int *>(base + 2 * array_bytes);
    auto *sorted_ids = reinterpret_cast<int *>(base + 3 * array_bytes);
    auto *offsets = reinterpret_cast<int *>(base + 4 * array_bytes);
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const int blocks = std::min((rows * candidates + 255) / 256, 65535);
    prepare_sort<<<blocks, 256, 0, stream>>>(
        masked, ids, offsets, scores, rows, sequence, candidates, start_pos, ratio);
    size_t temporary_bytes = workspace_bytes - fixed_bytes;
    // CUB's stable sort preserves ascending candidate ID within score ties.
    check(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        base + fixed_bytes, temporary_bytes, masked, sorted, ids, sorted_ids,
        rows * candidates, rows, offsets, offsets + 1, 0, 32, stream));
    gather_topk<<<std::min((rows * topk + 255) / 256, 65535), 256, 0, stream>>>(
        indices, sorted_ids, rows, sequence, candidates, topk, start_pos, ratio, index_offset);
    check(cudaGetLastError());
}
}

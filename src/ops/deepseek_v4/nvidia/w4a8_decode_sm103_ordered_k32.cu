// The normal project's CUDA glob also sees this translation unit. The
// independent CUDA 13 / SM103a backend is explicitly enabled by its build.
#ifdef LLAISYS_B300_STANDALONE
#include "w4a8_decode_sm103.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <limits>

namespace {

__device__ __forceinline__ float ue8m0(uint8_t value) {
    // UE8M0 byte zero represents 2^-127, not zero; 255 is NaN.
    return value == 0 ? __uint_as_float(0x00400000u)
                     : __uint_as_float(value == 255 ? 0x7fc00000u : uint32_t(value) << 23);
}

template <int I>
__device__ __forceinline__ uint32_t word(const uint4 &v) {
    static_assert(I >= 0 && I < 4);
    if constexpr (I == 0) return v.x;
    if constexpr (I == 1) return v.y;
    if constexpr (I == 2) return v.z;
    return v.w;
}

template <int Pair>
__device__ __forceinline__ float2 product_pair(
    const uint4 &a_lo, const uint4 &a_hi, const uint4 &b) {
    constexpr int a_word = (Pair % 8) / 2;
    const uint32_t packed_a = Pair < 8 ? word<a_word>(a_lo) : word<a_word>(a_hi);
    const uint16_t av = static_cast<uint16_t>(packed_a >> ((Pair % 2) * 16));
    const uint8_t bv = static_cast<uint8_t>(word<Pair / 4>(b) >> ((Pair % 4) * 8));
    const __half2 ah = static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(av, __NV_E4M3));
    const __half2 bh = static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(bv, __NV_E2M1));
    // Each finite E4M3 * E2M1 product is exactly representable in FP16:
    // magnitudes are <=448*6 and the smallest nonzero product is 2^-10;
    // product significands require <=6 bits. Only the multiplication uses
    // packed half arithmetic; all reductions and scale arithmetic are FP32.
    return __half22float2(__hmul2(ah, bh));
}

template <int Pair = 0>
__device__ __forceinline__ void accumulate_pairs(
    const uint4 &a_lo, const uint4 &a_hi, const uint4 &b, float &lo, float &hi) {
    const float2 products = product_pair<Pair>(a_lo, a_hi, b);
    lo += products.x;
    hi += products.y;
    if constexpr (Pair < 15) accumulate_pairs<Pair + 1>(a_lo, a_hi, b, lo, hi);
}

__device__ __forceinline__ float dot32(const uint8_t *a, const uint4 &b) {
    const uint4 a_lo = *reinterpret_cast<const uint4 *>(a);
    const uint4 a_hi = *reinterpret_cast<const uint4 *>(a + 16);
    float lo = 0.0f, hi = 0.0f;
    accumulate_pairs(a_lo, a_hi, b, lo, hi);
    return lo + hi;
}

template <int M, int Warps>
__global__ __launch_bounds__(Warps * 32) void decode(
    const uint8_t *__restrict__ a, const uint8_t *__restrict__ sa,
    const uint8_t *__restrict__ b, const uint8_t *__restrict__ sb,
    __nv_bfloat16 *__restrict__ c, int n, int k, int64_t a_stride, int64_t sa_stride) {
    const int column = blockIdx.x * Warps + threadIdx.x / 32;
    if (column >= n) return; // Entire warp exits together.
    const int lane = threadIdx.x & 31;
    const int groups = k / 32;
    float sum[M] = {};

    // EXPERIMENT ONLY: parallel K32 dot products, then ascending K32
    // accumulation in lane zero to follow the published TileLang outer loop.
    // This does not establish equivalence to the tensor-core dot32 internals.
    for (int group_base = 0; group_base < groups; group_base += 32) {
        const int group = group_base + lane;
        float scaled[M] = {};
        if (group < groups) {
            const uint4 weights = *reinterpret_cast<const uint4 *>(
                b + int64_t(column) * (k / 2) + group * 16);
            const float weight_scale = ue8m0(sb[int64_t(column) * groups + group]);
            #pragma unroll
            for (int row = 0; row < M; ++row) {
                const float dot = dot32(a + int64_t(row) * a_stride + group * 32, weights);
                const float activation_scale = ue8m0(sa[int64_t(row) * sa_stride + group / 4]);
                scaled[row] = __fmul_rn(__fmul_rn(dot, activation_scale), weight_scale);
            }
        }
        // Every lane participates in each shuffle; only lane zero owns the
        // sequential accumulator. Skip nonexistent groups, including +0 adds.
        #pragma unroll
        for (int source_lane = 0; source_lane < 32; ++source_lane) {
            #pragma unroll
            for (int row = 0; row < M; ++row) {
                const float value = __shfl_sync(0xffffffffu, scaled[row], source_lane);
                if (lane == 0 && group_base + source_lane < groups)
                    sum[row] = __fadd_rn(sum[row], value);
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < M; ++row) {
        if (lane == 0) c[int64_t(row) * n + column] = __float2bfloat16_rn(sum[row]);
    }
}

template <int Warps>
void launch(const uint8_t *a, const uint8_t *sa, const uint8_t *b, const uint8_t *sb,
            __nv_bfloat16 *c, int m, int n, int k, int64_t a_stride,
            int64_t sa_stride, cudaStream_t stream) {
    const unsigned blocks = (n + Warps - 1) / Warps;
    #define LAUNCH_CASE(M) case M: decode<M, Warps><<<blocks, Warps * 32, 0, stream>>>( \
        a, sa, b, sb, c, n, k, a_stride, sa_stride); break
    switch (m) {
        LAUNCH_CASE(1); LAUNCH_CASE(2); LAUNCH_CASE(3); LAUNCH_CASE(4);
        LAUNCH_CASE(5); LAUNCH_CASE(6); LAUNCH_CASE(7); LAUNCH_CASE(8);
    }
    #undef LAUNCH_CASE
}

template <int M, int Warps>
__global__ __launch_bounds__(Warps * 32) void decode_split_k(
    const uint8_t *__restrict__ a, const uint8_t *__restrict__ sa,
    const uint8_t *__restrict__ b, const uint8_t *__restrict__ sb,
    __nv_bfloat16 *__restrict__ c, int n, int k, int64_t a_stride, int64_t sa_stride) {
    const int column = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int groups = k / 32;
    float sum[M] = {};
    __shared__ float partial[M][Warps];

    // More CTAs and independent K groups expose latency-hiding work for M4/8.
    // All reductions stay inside this CTA: no global workspace or atomics.
    for (int group = threadIdx.x; group < groups; group += Warps * 32) {
        const uint4 weights = *reinterpret_cast<const uint4 *>(
            b + int64_t(column) * (k / 2) + group * 16);
        const float weight_scale = ue8m0(sb[int64_t(column) * groups + group]);
        #pragma unroll
        for (int row = 0; row < M; ++row) {
            const float dot = dot32(a + int64_t(row) * a_stride + group * 32, weights);
            const float activation_scale = ue8m0(sa[int64_t(row) * sa_stride + group / 4]);
            const float scaled = __fmul_rn(__fmul_rn(dot, activation_scale), weight_scale);
            sum[row] = __fadd_rn(sum[row], scaled);
        }
    }
    #pragma unroll
    for (int row = 0; row < M; ++row) {
        #pragma unroll
        for (int delta = 16; delta; delta /= 2)
            sum[row] += __shfl_down_sync(0xffffffffu, sum[row], delta);
        if (lane == 0) partial[row][warp] = sum[row];
    }
    __syncthreads();
    if (threadIdx.x < M) {
        float result = partial[threadIdx.x][0];
        #pragma unroll
        for (int w = 1; w < Warps; ++w) result = __fadd_rn(result, partial[threadIdx.x][w]);
        c[int64_t(threadIdx.x) * n + column] = __float2bfloat16_rn(result);
    }
}

void launch_split_k(const uint8_t *a, const uint8_t *sa, const uint8_t *b, const uint8_t *sb,
                    __nv_bfloat16 *c, int m, int n, int k, int64_t a_stride,
                    int64_t sa_stride, cudaStream_t stream) {
    #define LAUNCH_CASE(M) case M: decode_split_k<M, 4><<<n, 128, 0, stream>>>( \
        a, sa, b, sb, c, n, k, a_stride, sa_stride); break
    switch (m) {
        LAUNCH_CASE(1); LAUNCH_CASE(2); LAUNCH_CASE(3); LAUNCH_CASE(4);
        LAUNCH_CASE(5); LAUNCH_CASE(6); LAUNCH_CASE(7); LAUNCH_CASE(8);
    }
    #undef LAUNCH_CASE
}

bool address_extent(const void *p, uint64_t bytes, uintptr_t &begin, uintptr_t &end) {
    begin = reinterpret_cast<uintptr_t>(p);
    if (bytes > std::numeric_limits<uintptr_t>::max() - begin) return false;
    end = begin + bytes;
    return true;
}

} // namespace

extern "C" const char *llaisys_w4a8_decode_sm103_version(void) {
    return "llaisys-w4a8-sm103-experimental-v2";
}

extern "C" int llaisys_w4a8_decode_sm103(
    const void *a, const uint8_t *sa, const uint8_t *b, const uint8_t *sb,
    void *c, int m, int n, int k, int64_t a_row_stride,
    int64_t sa_row_stride, cudaStream_t stream, int variant) {
    if (!a || !sa || !b || !sb || !c || m < 1 || m > 8 || n < 1 || n > 65536
        || k < 128 || k > 32768 || k % 128 || a_row_stride < k
        || sa_row_stride < k / 128 || a_row_stride % 16
        || (reinterpret_cast<uintptr_t>(a) & 15u)
        || (reinterpret_cast<uintptr_t>(b) & 15u)
        || (reinterpret_cast<uintptr_t>(c) & 1u)
        || variant < 0 || variant > 3)
        return cudaErrorInvalidValue;
    const uint64_t max_stride = (uint64_t(std::numeric_limits<int64_t>::max()) - uint64_t(k)) / 8;
    if (uint64_t(a_row_stride) > max_stride || uint64_t(sa_row_stride) > max_stride)
        return cudaErrorInvalidValue;
    uintptr_t c_begin, c_end;
    if (!address_extent(c, uint64_t(m) * n * 2, c_begin, c_end)) return cudaErrorInvalidValue;
    const void *inputs[] = {a, sa, b, sb};
    const uint64_t extents[] = {uint64_t(m - 1) * a_row_stride + k,
        uint64_t(m - 1) * sa_row_stride + k / 128, uint64_t(n) * k / 2, uint64_t(n) * k / 32};
    for (int i = 0; i < 4; ++i) {
        uintptr_t begin, end;
        if (!address_extent(inputs[i], extents[i], begin, end)
            || (begin < c_end && c_begin < end)) return cudaErrorInvalidValue;
    }
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    static thread_local int validated_device = -1;
    if (device != validated_device) {
        cudaDeviceProp prop{};
        error = cudaGetDeviceProperties(&prop, device);
        if (error != cudaSuccess) return error;
        if (prop.major != 10 || prop.minor != 3) return cudaErrorNotSupported;
        validated_device = device;
    }
    if (variant == 3)
        launch_split_k(static_cast<const uint8_t *>(a), sa, b, sb, static_cast<__nv_bfloat16 *>(c),
                       m, n, k, a_row_stride, sa_row_stride, stream);
    else if (variant == 2)
        launch<8>(static_cast<const uint8_t *>(a), sa, b, sb, static_cast<__nv_bfloat16 *>(c),
                  m, n, k, a_row_stride, sa_row_stride, stream);
    else
        launch<4>(static_cast<const uint8_t *>(a), sa, b, sb, static_cast<__nv_bfloat16 *>(c),
                  m, n, k, a_row_stride, sa_row_stride, stream);
    return cudaGetLastError();
}
#endif // LLAISYS_B300_STANDALONE

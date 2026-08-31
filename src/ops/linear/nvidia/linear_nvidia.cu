#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>
#include <cstdlib>

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t status = (call);                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "[cuBLAS ERROR] code %d at %s:%d\n",                      \
                    (int)status, __FILE__, __LINE__);                                 \
            throw std::runtime_error("cuBLAS call failed");                           \
        }                                                                             \
    } while (0)

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",                  \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);           \
            throw std::runtime_error(cudaGetErrorString(err));                        \
        }                                                                             \
    } while (0)

static bool linear_alloc_verbose() {
    const char *v = std::getenv("LLAISYS_LINEAR_ALLOC_VERBOSE");
    return v && v[0] == '1';
}

// ---- GEMV kernel: decode 阶段核心 (FP16 权重+输入, FP32 累加) ----
//
// 计算: y[row] = dot(W[row, :], x[:]) + bias[row] + residual[row]
//
// 矩阵布局:
//   W: [N, K] (row-major, 每行是一个输出神经元的权重)
//   x: [K]   (输入向量, 通常是 hidden_states)
//   y: [N]   (输出向量)
//
// 为什么 decode 是 GEMV:
//   decode 时 batch_size=1, 输入只有 1 个 token 的 hidden_states [1, K]
//   所以 matmul([1,K], [K,N]) 退化为向量×矩阵 = GEMV
//
// 性能特点:
//   GEMV 是 memory bandwidth bound (每个权重元素只被读 1 次, 用 1 次)
//   RTX 4060: 256 GB/s 带宽, 3GB FP16 权重 → 理论极限 ~85 tok/s
//
// 优化手段:
//   1. half2 向量化加载 (每次读 4 字节 = 2 个 FP16, 提高带宽利用率)
//   2. FP32 累加 (避免 FP16 精度损失)
//   3. 两级归约: warp shuffle (fastest) + shared memory (cross-warp)
//   4. 多行并行: 每个 block 处理多行, 提高 SM 占用率
//   5. residual + bias 融合: 避免额外的 kernel 启动和显存读写
//
// 线程组织:
//   每行分配 WARPS_PER_ROW 个 warp (32 × WARPS_PER_ROW 个线程)
//   每个 block 256 个线程, 可处理 256/(32*WPR) 行
//   WPR=1 适用于 K≤512, WPR=2 适用于 K≤2048, WPR=4 适用于更大 K
//
template<int WARPS_PER_ROW>
__global__ void gemv_f16_kernel(const __half *__restrict__ W,
                                const __half *__restrict__ x,
                                __half *__restrict__ y,
                                const __half *__restrict__ bias,
                                const __half *__restrict__ residual,
                                int N, int K) {
    const int WARP_SIZE = 32;
    const int THREADS_PER_ROW = WARPS_PER_ROW * WARP_SIZE;

    // ── 线程定位 ──
    int local_thread = threadIdx.x;                      // block 内线程 ID (0~255)
    int row_in_block = local_thread / THREADS_PER_ROW;   // 当前线程处理第几行 (block 内)
    int thread_in_row = local_thread % THREADS_PER_ROW;  // 当前线程在行内的位置
    int warp_in_row = thread_in_row / WARP_SIZE;         // 行内第几个 warp
    int lane_id = thread_in_row % WARP_SIZE;             // warp 内 lane ID (0~31)

    int rows_per_block = blockDim.x / THREADS_PER_ROW;   // 每个 block 处理几行
    int row = blockIdx.x * rows_per_block + row_in_block; // 全局行号 (输出维度索引)
    if (row >= N) return;

    const __half *row_ptr = W + (int64_t)row * K;  // 指向当前行的权重起始地址
    float sum = 0.0f;

    // ── 向量化加载: 每次读 half2 (4 字节 = 2 个 FP16) ──
    // 比逐个读 half (2 字节) 带宽利用率提升 2x
    int k2 = K / 2;
    int global_lane = warp_in_row * WARP_SIZE + lane_id;
    for (int i = global_lane; i < k2; i += THREADS_PER_ROW) {
        half2 w2 = reinterpret_cast<const half2 *>(row_ptr)[i];  // 读 2 个 FP16 权重
        half2 x2 = reinterpret_cast<const half2 *>(x)[i];        // 读 2 个 FP16 输入
        // FP32 累加: 先转 float 再乘, 避免 FP16 乘法溢出
        sum += __half2float(w2.x) * __half2float(x2.x)
             + __half2float(w2.y) * __half2float(x2.y);
    }
    // 处理 K 为奇数时的最后一个元素
    if (global_lane == 0 && (K & 1)) {
        sum += __half2float(row_ptr[K - 1]) * __half2float(x[K - 1]);
    }

    // ── 第一级归约: warp 内 shuffle (无需 shared memory, 最快) ──
    // 32 个线程的 sum 通过蝶式交换归约为 1 个值 (lane 0)
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // ── 第二级归约: 跨 warp 通过 shared memory ──
    // 当 WARPS_PER_ROW > 1 时, 多个 warp 的部分和需要再归约
    extern __shared__ float smem[];
    if (lane_id == 0) {
        smem[row_in_block * WARPS_PER_ROW + warp_in_row] = sum;  // 每个 warp 的 lane 0 写结果
    }
    __syncthreads();  // 等所有 warp 写完

    // ── 最终输出: 融合 bias + residual (减少额外 kernel 调用) ──
    if (warp_in_row == 0 && lane_id == 0) {
        float total = 0.0f;
        for (int w = 0; w < WARPS_PER_ROW; w++) {
            total += smem[row_in_block * WARPS_PER_ROW + w];
        }
        if (bias) {
            total += __half2float(bias[row]);      // 融合 bias add
        }
        if (residual) {
            total += __half2float(residual[row]);   // 融合 residual add
        }
        y[row] = __float2half(total);  // FP32 → FP16 写回
    }
}

// GEMV 启动器: 根据 K 维度选择最优的 WARPS_PER_ROW
// K 越大需要越多 warp 来并行处理一行的点积
// residual 可为 nullptr (不做 add 融合)
static void gemv_f16_launch(const __half *W, const __half *x, __half *y,
                            const __half *bias, const __half *residual,
                            int N, int K, cudaStream_t stream = 0) {
    if (K <= 512) {
        constexpr int WPR = 1;                       // K≤512: 1 warp (32 线程) 够用
        int rpb = 256 / (WPR * 32);                  // 每 block 处理 8 行
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_f16_kernel<WPR><<<grid, 256, smem, stream>>>(W, x, y, bias, residual, N, K);
    } else if (K <= 2048) {
        constexpr int WPR = 2;                       // K≤2048: 2 warps (64 线程) 处理一行
        int rpb = 256 / (WPR * 32);                  // 每 block 处理 4 行
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_f16_kernel<WPR><<<grid, 256, smem, stream>>>(W, x, y, bias, residual, N, K);
    } else {
        constexpr int WPR = 4;                       // K>2048: 4 warps (128 线程) 处理一行
        int rpb = 256 / (WPR * 32);                  // 每 block 处理 2 行
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_f16_kernel<WPR><<<grid, 256, smem, stream>>>(W, x, y, bias, residual, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---- W4A16 Fused Dequant-GEMV: INT4 权重 × FP16 输入, 无中间缓冲区 ----
//
// 核心思想:
//   传统路径: INT4→FP32(显存) →cuBLAS GEMM → 读写 ~5GB/token, 效率极低
//   本 kernel: INT4→FP32(寄存器) × FP16(输入) → FP32 累加 → 只读 ~0.94GB/token
//
// 数据流 (每个线程):
//   ┌─────────────┐   ┌──────────┐   ┌──────────┐
//   │ W_packed[N,K/2]│   │ x[K] FP16│   │scale[N,G]│
//   │   4B=8个INT4  │   │ 4B=2个FP16│   │  FP32    │
//   └──────┬────────┘   └────┬─────┘   └────┬─────┘
//          │uint32 load      │half2 load     │cache hit
//          ▼                 ▼               ▼
//   ┌──────────────────────────────────────────┐
//   │ 寄存器内: (int4_val - 8) × scale × x_fp16 │
//   │         FP32 累加 sum                      │
//   └──────────────────┬───────────────────────┘
//                      │
//        ┌─────────────┴──────────────┐
//        ▼                            ▼
//   warp shuffle 归约            shared memory 归约
//   (32线程→1值, 无需同步)       (跨warp, 需__syncthreads)
//        └─────────────┬──────────────┘
//                      ▼
//            y[row] = total + bias + residual
//            (OutT=__half层间 / float=lm_head)
//
// 计算: y[row] = sum_k( dequant(W_int4[row,k]) * x[k] ) + bias[row] + residual[row]
// 其中: dequant(v) = (v - 8) * scale[row][k / group_size]
//
// 与 gemv_f16_kernel 的区别:
//   - 权重从 FP16 (2 bytes/elem) 变为 INT4 packed (0.5 bytes/elem) → 带宽降 4×
//   - 每个 uint8 字节包含 2 个 INT4 值: low nibble 和 high nibble
//   - 需要额外读取 per-group scale (FP32, 很小, L1 cache 命中)
//
// 带宽分析 (Qwen2-1.5B, 28 层 + lm_head):
//   FP16 路径: 每 token 读 ~3.0 GB 权重 → 256 GB/s → ~85 tok/s 上限
//   INT4 路径: 每 token 读 ~0.94 GB (0.82GB 层权重 + 0.12GB lm_head)
//              → 256 GB/s → ~272 tok/s 理论上限
//   实测: ~98 tok/s (带宽利用率 ~36%, 受 kernel launch + attention + norm 等开销影响)
//
// 权重打包格式 (per-group symmetric INT4, group_size=128):
//   W_packed: [N, K/2] uint8, 每字节存 2 个 4-bit 整数
//     低 4 位 (byte & 0x0F) - 8 → 偶数列 (col_even), 范围 [-8, 7]
//     高 4 位 (byte >> 4)   - 8 → 奇数列 (col_odd),  范围 [-8, 7]
//   scale:    [N, num_groups] float32, num_groups = K / group_size
//
// 向量化加载策略:
//   权重: uint32 (4 bytes) 一次读 8 个 INT4
//   输入: half2  (4 bytes) 一次读 2 个 FP16 × 4 = 8 值, 与权重对齐
//   注: 实测 uint4 (16B/128-bit) 反而因寄存器压力大、占用率下降而慢 14%
//        warp 内 32 线程 × uint32 (4B) = 128B 事务, GPU 已自动合并为最优宽度
//
// 线程组织 (与 gemv_f16_kernel 相同):
//   每个 block 256 线程, 每行分配 WPR 个 warp (WPR × 32 线程)
//   WPR=1 → 每 block 8 行, WPR=2 → 4 行, WPR=4 → 2 行
//   K 越大需要越多 warp 来分摊点积循环
//
// OutT 模板参数:
//   __half → 层间激活 (q/k/v/o_proj, gate/up/down_proj)
//   float  → lm_head 输出 logits (vocab_size=151936 维)
//
template<int WARPS_PER_ROW, typename OutT>
__global__ void gemv_w4a16_kernel(
        const uint8_t *__restrict__ W_packed,  // [N, K/2] packed INT4 权重
        const __half  *__restrict__ x,         // [K] FP16 输入向量
        OutT          *__restrict__ y,         // [N] 输出向量 (FP16 或 FP32)
        const float   *__restrict__ scale,     // [N, num_groups] per-group scale
        const __half  *__restrict__ bias,      // [N] 可选 bias (nullptr 跳过)
        const __half  *__restrict__ residual,  // [N] 可选 residual (nullptr 跳过)
        int N, int K, int num_groups, int group_size) {

    const int WARP_SIZE = 32;
    const int THREADS_PER_ROW = WARPS_PER_ROW * WARP_SIZE;

    // ── 线程定位 (与 gemv_f16_kernel 完全相同) ──
    int local_thread = threadIdx.x;
    int row_in_block = local_thread / THREADS_PER_ROW;
    int thread_in_row = local_thread % THREADS_PER_ROW;
    int warp_in_row = thread_in_row / WARP_SIZE;
    int lane_id = thread_in_row % WARP_SIZE;

    int rows_per_block = blockDim.x / THREADS_PER_ROW;
    int row = blockIdx.x * rows_per_block + row_in_block;
    if (row >= N) return;

    int packed_cols = K / 2;  // 每行的 packed 字节数
    const uint8_t *row_ptr = W_packed + (int64_t)row * packed_cols;
    const float *scale_row = scale + (int64_t)row * num_groups;
    float sum = 0.0f;

    // ── 主循环: uint32 向量化加载 (4 bytes = 8 INT4 values) ──
    // 注: 尝试过 uint4 (128-bit), 反而因寄存器压力降低占用率导致变慢 14%
    // uint32 是最优: warp 内 32 线程 × 4B = 128B 事务, GPU 自动合并
    int packed4 = packed_cols / 4;  // 每行的 uint32 数量
    int global_lane = warp_in_row * WARP_SIZE + lane_id;

    for (int i = global_lane; i < packed4; i += THREADS_PER_ROW) {
        // 一次读 4 字节 = 8 个 INT4 权重值
        uint32_t pack = reinterpret_cast<const uint32_t *>(row_ptr)[i];
        int col_base = i * 8;  // 对应原始矩阵的起始列号

        // 预加载对应的 8 个 FP16 输入值 (4 个 half2 加载)
        half2 x01 = reinterpret_cast<const half2 *>(x)[col_base / 2];
        half2 x23 = reinterpret_cast<const half2 *>(x)[col_base / 2 + 1];
        half2 x45 = reinterpret_cast<const half2 *>(x)[col_base / 2 + 2];
        half2 x67 = reinterpret_cast<const half2 *>(x)[col_base / 2 + 3];

        // 转为 FP32 数组, 方便 unroll 循环访问
        float xf[8] = {
            __half2float(x01.x), __half2float(x01.y),
            __half2float(x23.x), __half2float(x23.y),
            __half2float(x45.x), __half2float(x45.y),
            __half2float(x67.x), __half2float(x67.y)
        };

        // 解包 4 个字节, 每字节 2 个 INT4 → 共 8 个值
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint8_t byte_val = (pack >> (b * 8)) & 0xFF;
            int val_lo = (int)(byte_val & 0x0F) - 8;  // 低 4 位 → 有符号 [-8, 7]
            int val_hi = (int)(byte_val >> 4) - 8;     // 高 4 位 → 有符号 [-8, 7]

            // scale 查找: 同一字节的两个值必在同一 group (相邻两列)
            float s = scale_row[(col_base + b * 2) / group_size];

            // dequant + multiply + accumulate
            sum += __int2float_rn(val_lo) * s * xf[b * 2]
                 + __int2float_rn(val_hi) * s * xf[b * 2 + 1];
        }
    }

    // ── 第一级归约: warp shuffle ──
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // ── 第二级归约: 跨 warp shared memory ──
    extern __shared__ float smem[];
    if (lane_id == 0) {
        smem[row_in_block * WARPS_PER_ROW + warp_in_row] = sum;
    }
    __syncthreads();

    // ── 最终输出: 融合 bias + residual ──
    if (warp_in_row == 0 && lane_id == 0) {
        float total = 0.0f;
        for (int w = 0; w < WARPS_PER_ROW; w++) {
            total += smem[row_in_block * WARPS_PER_ROW + w];
        }
        if (bias) total += __half2float(bias[row]);
        if (residual) total += __half2float(residual[row]);
        // OutT = __half 时转 FP16, OutT = float 时直接写
        y[row] = from_float<OutT>(total);
    }
}

// W4A16 GEMV 启动器: 根据 K 维度自动选择最优的 WARPS_PER_ROW
//
// Qwen2-1.5B 各层的 K 值和对应配置:
//   K=1536 (q/k/v/o_proj, gate/up, lm_head): WPR=2, 每 block 4 行
//   K=8960 (down_proj):                      WPR=4, 每 block 2 行
//
// OutT = __half (层间激活) 或 float (lm_head logits)
template<typename OutT>
static void gemv_w4a16_launch(
        const uint8_t *W_packed, const __half *x, OutT *y,
        const float *scale, const __half *bias, const __half *residual,
        int N, int K, int num_groups, int group_size,
        cudaStream_t stream = 0) {
    if (K <= 512) {
        constexpr int WPR = 1;
        int rpb = 256 / (WPR * 32);
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_w4a16_kernel<WPR, OutT><<<grid, 256, smem, stream>>>(
            W_packed, x, y, scale, bias, residual, N, K, num_groups, group_size);
    } else if (K <= 2048) {
        constexpr int WPR = 2;
        int rpb = 256 / (WPR * 32);
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_w4a16_kernel<WPR, OutT><<<grid, 256, smem, stream>>>(
            W_packed, x, y, scale, bias, residual, N, K, num_groups, group_size);
    } else {
        constexpr int WPR = 4;
        int rpb = 256 / (WPR * 32);
        int grid = (N + rpb - 1) / rpb;
        int smem = rpb * WPR * (int)sizeof(float);
        gemv_w4a16_kernel<WPR, OutT><<<grid, 256, smem, stream>>>(
            W_packed, x, y, scale, bias, residual, N, K, num_groups, group_size);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- Bias add kernel (replaces the ones-vector GEMM approach) ----
template<typename T>
__global__ void add_bias_kernel(T *Y, const T *bias, int64_t M, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (tid >= total) return;
    int64_t j = tid % N;
    float y_val = to_float(Y[tid]);
    float b_val = to_float(bias[j]);
    Y[tid] = from_float<T>(y_val + b_val);
}

// ---- FP32→FP16 conversion kernel (for mixed-precision linear) ----
__global__ void convert_f32_to_f16_kernel(__half *out, const float *in, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = __float2half(in[tid]);
}

// ---- FP16→FP32 conversion kernel (for dequant path with FP16 activations) ----
__global__ void convert_f16_to_f32_kernel(float *out, const __half *in, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = __half2float(in[tid]);
}

// ---- FP16 bias add to FP32 output ----
__global__ void add_bias_f16_to_f32_kernel(float *Y, const __half *bias, int64_t M, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (tid >= total) return;
    int64_t j = tid % N;
    Y[tid] += __half2float(bias[j]);
}

// Lazy-initialized thread-local cuBLAS handle with pre-allocated workspace
// for CUDA Graph compatibility (cuBLAS must not call cudaMalloc during capture)
static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    static thread_local void *workspace = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
        // Pre-allocate 4 MB workspace for cuBLAS (avoids internal malloc during graph capture)
        constexpr size_t CUBLAS_WORKSPACE_SIZE = 4 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&workspace, CUBLAS_WORKSPACE_SIZE));
        CUBLAS_CHECK(cublasSetWorkspace(handle, workspace, CUBLAS_WORKSPACE_SIZE));
        // Use per-thread default stream for CUDA Graph capture compatibility
        CUBLAS_CHECK(cublasSetStream(handle, cudaStreamPerThread));
    }
    return handle;
}

namespace llaisys::ops::nvidia {

// Y = X * W^T + bias
// Uses cublasGemmEx to support F32/F16/BF16 with F32 compute.
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto w_dtype = weight->dtype();
    auto in_dtype = in->dtype();
    auto out_dtype = out->dtype();

    int64_t M = in->shape()[0];
    int64_t K = in->shape()[1];
    int64_t N = weight->shape()[0];

    cublasHandle_t handle = get_cublas_handle();

    float alpha = 1.0f;
    float beta  = 0.0f;

    // ---- Mixed precision path: FP16 weight → FP32 output ----
    // 情况1: F16w × F32in → F32out (BF16 模型 + FP32 激活)
    // 情况2: F16w × F16in → F32out (LM Head: FP16 权重/激活, FP32 logits)
    if (w_dtype == LLAISYS_DTYPE_F16 && out_dtype == LLAISYS_DTYPE_F32) {
        const void *in_gemm_ptr = in->data();

        // If input is F32, convert to F16 first (cuBLAS requires A/B same type)
        static thread_local __half *in_f16_buf = nullptr;
        static thread_local int64_t in_f16_cap = 0;
        if (in_dtype == LLAISYS_DTYPE_F32) {
            int64_t in_elems = M * K;
            if (in_elems > in_f16_cap) {
                if (linear_alloc_verbose()) {
                    fprintf(stderr, "[linear] grow in_f16_buf: elems=%ld old_cap=%ld\n",
                            (long)in_elems, (long)in_f16_cap);
                }
                if (in_f16_buf) cudaFree(in_f16_buf);
                cudaMalloc(&in_f16_buf, in_elems * sizeof(__half));
                in_f16_cap = in_elems;
            }
            int thr = 256, blk = ((int)in_elems + thr - 1) / thr;
            convert_f32_to_f16_kernel<<<blk, thr>>>(in_f16_buf, (const float*)in->data(), in_elems);
            in_gemm_ptr = in_f16_buf;
        }
        // else: input is already F16, use directly

        // cuBLAS: A=F16 (weight), B=F16 (input), C=F32 (output)
        CUBLAS_CHECK(cublasGemmEx(handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  (int)N, (int)M, (int)K,
                                  &alpha,
                                  weight->data(), CUDA_R_16F, (int)K,
                                  in_gemm_ptr,    CUDA_R_16F, (int)K,
                                  &beta,
                                  out->data(),    CUDA_R_32F, (int)N,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));

        // Add FP16 bias to FP32 output
        if (bias && bias->data()) {
            int64_t total = M * N;
            int thr = 256, blk = ((int)total + thr - 1) / thr;
            if (bias->dtype() == LLAISYS_DTYPE_F16) {
                add_bias_f16_to_f32_kernel<<<blk, thr>>>(
                    (float*)out->data(), (const __half*)bias->data(), M, N);
            } else {
                add_bias_kernel<float><<<blk, thr>>>(
                    (float*)out->data(), (const float*)bias->data(), M, N);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // ---- Mixed precision path: F32 weight + FP16 input ----
    // 场景: 量化权重 dequant 到 FP32 后, 与 FP16 激活做 GEMM
    // 策略: FP16 input → 转 F32 → cublasGemmEx(F32, F32, F32) → 转 FP16 output (或直接 F32)
    if (w_dtype == LLAISYS_DTYPE_F32 && in_dtype == LLAISYS_DTYPE_F16) {
        // 1. Convert FP16 input → FP32 (input is small: [B, K])
        static thread_local float *in_f32_buf = nullptr;
        static thread_local int64_t in_f32_cap = 0;
        int64_t in_elems = M * K;
        if (in_elems > in_f32_cap) {
            if (linear_alloc_verbose()) {
                fprintf(stderr, "[linear] grow in_f32_buf: elems=%ld old_cap=%ld\n",
                        (long)in_elems, (long)in_f32_cap);
            }
            if (in_f32_buf) cudaFree(in_f32_buf);
            cudaMalloc(&in_f32_buf, in_elems * sizeof(float));
            in_f32_cap = in_elems;
        }
        int thr = 256, blk = ((int)in_elems + thr - 1) / thr;
        convert_f16_to_f32_kernel<<<blk, thr>>>(in_f32_buf, (const __half*)in->data(), in_elems);

        if (out_dtype == LLAISYS_DTYPE_F32) {
            // F32 weight × F16 input → F32 output (lm_head 场景)
            CUBLAS_CHECK(cublasGemmEx(handle,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      (int)N, (int)M, (int)K,
                                      &alpha,
                                      weight->data(), CUDA_R_32F, (int)K,
                                      in_f32_buf,     CUDA_R_32F, (int)K,
                                      &beta,
                                      out->data(),    CUDA_R_32F, (int)N,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));
            // Bias
            if (bias && bias->data()) {
                int64_t total = M * N;
                thr = 256; blk = ((int)total + thr - 1) / thr;
                add_bias_kernel<float><<<blk, thr>>>(
                    (float*)out->data(), (const float*)bias->data(), M, N);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // F32 weight × F16 input → FP16 output (标准激活场景)
            // 2. Allocate FP32 output buffer
            static thread_local float *out_f32_buf = nullptr;
            static thread_local int64_t out_f32_cap = 0;
            int64_t out_elems = M * N;
            if (out_elems > out_f32_cap) {
                if (linear_alloc_verbose()) {
                    fprintf(stderr, "[linear] grow out_f32_buf: elems=%ld old_cap=%ld\n",
                            (long)out_elems, (long)out_f32_cap);
                }
                if (out_f32_buf) cudaFree(out_f32_buf);
                cudaMalloc(&out_f32_buf, out_elems * sizeof(float));
                out_f32_cap = out_elems;
            }

            // 3. cublasGemmEx: F32 × F32 → F32
            CUBLAS_CHECK(cublasGemmEx(handle,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      (int)N, (int)M, (int)K,
                                      &alpha,
                                      weight->data(), CUDA_R_32F, (int)K,
                                      in_f32_buf,     CUDA_R_32F, (int)K,
                                      &beta,
                                      out_f32_buf,    CUDA_R_32F, (int)N,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));

            // 4. Add bias to FP32 buffer BEFORE converting to FP16
            if (bias && bias->data()) {
                int64_t total = M * N;
                thr = 256; blk = ((int)total + thr - 1) / thr;
                if (bias->dtype() == LLAISYS_DTYPE_F32) {
                    add_bias_kernel<float><<<blk, thr>>>(
                        out_f32_buf, (const float*)bias->data(), M, N);
                } else {
                    // FP16 bias → FP32 buffer
                    add_bias_f16_to_f32_kernel<<<blk, thr>>>(
                        out_f32_buf, (const __half*)bias->data(), M, N);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            // 5. Convert F32 output (with bias) → FP16
            blk = ((int)out_elems + thr - 1) / thr;
            convert_f32_to_f16_kernel<<<blk, thr>>>((__half*)out->data(), out_f32_buf, out_elems);
        }
        return;
    }

    // ---- Standard path: all tensors have the same dtype ----
    cudaDataType_t cuda_dtype;
    switch (w_dtype) {
    case LLAISYS_DTYPE_F32:  cuda_dtype = CUDA_R_32F;  break;
    case LLAISYS_DTYPE_F16:  cuda_dtype = CUDA_R_16F;  break;
    case LLAISYS_DTYPE_BF16: cuda_dtype = CUDA_R_16BF; break;
    default:
        throw std::runtime_error("NVIDIA linear: unsupported dtype");
    }

    // Note: Custom gemv_f16_kernel available for M=1 FP16, but cuBLAS is
    // faster (~4%) due to Tensor Core HMMA usage. Kept for CUDA Graph
    // compatibility if needed in the future (cuBLAS requires workspace
    // pre-allocation for graph capture via cublasSetWorkspace).

    // row-major Y = X * W^T  <=>  col-major Y^T = W * X^T
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_T,    // W stored row-major, viewed col-major => transpose
                              CUBLAS_OP_N,    // X stored row-major, X^T in col-major => no-transpose
                              (int)N, (int)M, (int)K,
                              &alpha,
                              weight->data(), cuda_dtype, (int)K,
                              in->data(),     cuda_dtype, (int)K,
                              &beta,
                              out->data(),    cuda_dtype, (int)N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));

    // Add bias if present
    if (bias && bias->data()) {
        int64_t total = M * N;
        int thr = 256, blk = ((int)total + thr - 1) / thr;
        switch (w_dtype) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel<float><<<blk, thr>>>(
                (float*)out->data(), (const float*)bias->data(), M, N);
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel<__half><<<blk, thr>>>(
                (__half*)out->data(), (const __half*)bias->data(), M, N);
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel<__nv_bfloat16><<<blk, thr>>>(
                (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
            break;
        default: break;
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

// Y = X * W^T + bias + residual  (fused GEMV+Add for M=1 FP16 decode)
// Falls back to linear() + separate add for non-M=1 or non-FP16 cases
void linear_add(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
                tensor_t residual) {
    int64_t M = in->shape()[0];
    auto w_dtype = weight->dtype();

    // Fused path: M=1, FP16, no quantization mixing
    if (M == 1 && w_dtype == LLAISYS_DTYPE_F16
        && in->dtype() == LLAISYS_DTYPE_F16
        && out->dtype() == LLAISYS_DTYPE_F16) {
        int64_t K = in->shape()[1];
        int64_t N = weight->shape()[0];
        const __half *bias_ptr = (bias && bias->data())
            ? (const __half *)bias->data() : nullptr;
        const __half *res_ptr = (residual && residual->data())
            ? (const __half *)residual->data() : nullptr;
        gemv_f16_launch((const __half *)weight->data(),
                        (const __half *)in->data(),
                        (__half *)out->data(),
                        bias_ptr, res_ptr, (int)N, (int)K);
        return;
    }

    // Fallback: separate linear + add
    linear(out, in, weight, bias);
    if (residual && residual->data()) {
        // element-wise add: out[i] += residual[i]
        int64_t total = out->numel();
        int thr = 256, blk = ((int)total + thr - 1) / thr;
        if (w_dtype == LLAISYS_DTYPE_F16) {
            add_bias_kernel<__half><<<blk, thr>>>(
                (__half *)out->data(), (const __half *)residual->data(), M, out->shape()[1]);
        } else {
            add_bias_kernel<float><<<blk, thr>>>(
                (float *)out->data(), (const float *)residual->data(), M, out->shape()[1]);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

// ---- Fused W4A16 Linear: INT4 权重 × FP16 输入 ----
//
// 调用路径: qwen2.cpp linear_maybe_dequant() → ops::linear_int4() → 本函数
//
// M=1 (decode): 走 fused GEMV, 一个 kernel 完成 dequant + matmul + bias + residual
//   - 无中间缓冲区, 无 cudaMalloc, 对 CUDA Graph 友好
//   - 支持 FP16 输出 (层间) 和 FP32 输出 (lm_head)
//
// M>1 (prefill): 由调用者回退到 dequant_int4 + cuBLAS GEMM
//   - GEMM 是 compute-bound, cuBLAS 用 Tensor Core 更快
//   - prefill 只执行一次, 不影响吞吐
//
void linear_int4(tensor_t out, tensor_t in, tensor_t weight, tensor_t scale,
                 tensor_t bias, int group_size, tensor_t residual) {
    int64_t M = in->shape()[0];
    int64_t K = in->shape()[1];
    int64_t N = out->shape()[1];
    int64_t num_groups = scale->shape()[1];

    // Fused W4A16 GEMV: M=1, FP16 输入
    if (M == 1 && in->dtype() == LLAISYS_DTYPE_F16) {
        const __half *bias_ptr = (bias && bias->data())
            ? (const __half *)bias->data() : nullptr;
        const __half *res_ptr = (residual && residual->data())
            ? (const __half *)residual->data() : nullptr;

        if (out->dtype() == LLAISYS_DTYPE_F16) {
            // 层间激活: FP16 输出
            gemv_w4a16_launch<__half>(
                (const uint8_t *)weight->data(),
                (const __half *)in->data(),
                (__half *)out->data(),
                (const float *)scale->data(),
                bias_ptr, res_ptr,
                (int)N, (int)K, (int)num_groups, group_size);
        } else {
            // lm_head: FP32 输出 (logits)
            gemv_w4a16_launch<float>(
                (const uint8_t *)weight->data(),
                (const __half *)in->data(),
                (float *)out->data(),
                (const float *)scale->data(),
                bias_ptr, res_ptr,
                (int)N, (int)K, (int)num_groups, group_size);
        }
        return;
    }

    // Fallback: dequantize_int4 → FP32 buffer → linear (调用者处理)
    fprintf(stderr, "[WARN] linear_int4 fallback: M=%ld, use dequant+cuBLAS path\n", (long)M);
}

} // namespace llaisys::ops::nvidia

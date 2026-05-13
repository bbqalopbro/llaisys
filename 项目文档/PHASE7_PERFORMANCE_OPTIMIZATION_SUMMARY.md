# llaisys 推理性能优化总结

## 性能里程碑

| 阶段 | 配置 | 吞吐量 | TPOT | 相对基线 |
|------|------|--------|------|---------|
| 基线 | FP16 + cuBLAS (eager) | ~55 tok/s | 18.2 ms | 1.0× |
| +FP16 GEMV 融合 | FP16 + fused GEMV+Add | ~58 tok/s | 17.2 ms | 1.05× |
| +CUDA Graph | FP16 + CUDA Graph | ~60 tok/s | 16.7 ms | 1.09× |
| **+W4A16 Fused GEMV** | **INT4 + fused dequant-GEMV** | **~98 tok/s** | **10.2 ms** | **1.78×** |

> 测试环境: NVIDIA RTX 4060 Laptop (8GB, 256 GB/s), Qwen2-1.5B, greedy decode  
> TPOT = Time Per Output Token, 吞吐量 = 生成 token 数 / 总时间

---

## 优化 1: FP16 GEMV + Bias/Residual 融合

### 文件
- [src/ops/linear/nvidia/linear_nvidia.cu](../src/ops/linear/nvidia/linear_nvidia.cu) — `gemv_f16_kernel`, `linear_add()`

### 原理
Decode 阶段 batch_size=1, matmul 退化为 GEMV (向量×矩阵)。cuBLAS GEMM 在 M=1 时无法充分利用 Tensor Core (需要 M≥16), 自定义 GEMV kernel 可以:
1. 使用 `half2` 向量化加载 (4 bytes/load)
2. 融合 bias add + residual add 到同一 kernel, 减少 2 次额外的 kernel launch 和显存读写
3. FP32 累加避免 FP16 精度溢出

### 关键设计
```cuda
// 线程组织: 每行 WARPS_PER_ROW 个 warp, 256 线程/block
// WPR=1 (K≤512), WPR=2 (K≤2048), WPR=4 (K>2048)
// 两级归约: warp shuffle (最快) → shared memory (跨 warp)
template<int WARPS_PER_ROW>
__global__ void gemv_f16_kernel(W, x, y, bias, residual, N, K)
```

### 效果
- 纯 linear: cuBLAS 比自定义 GEMV 快 ~4% (Tensor Core HMMA)
- linear + add 融合: 自定义 GEMV 省一次 kernel launch + 一次显存读写 → 快 ~5%
- 只在 `linear_add()` (o_proj + residual, down_proj + residual) 中使用

---

## 优化 2: CUDA Graph

### 文件
- [src/core/cuda_graph.cpp](../src/core/cuda_graph.cpp) — `CUDAGraphRunner::launch()`
- [src/llaisys/models/qwen2.cpp](../src/llaisys/models/qwen2.cpp) — `decode_graph.launch(decode_fn)`
- [xmake/nvidia.lua](../xmake/nvidia.lua), [xmake.lua](../xmake.lua) — `--default-stream=per-thread`

### 原理
Decode 阶段每步执行 ~200 个 CUDA kernel, 每个 kernel launch 有 ~2-3μs CPU 开销。CUDA Graph 将首次执行的所有 kernel 录制下来, 后续直接重放 (replay), 消除 CPU→GPU 的 launch 延迟。

### 关键挑战与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `cudaStreamBeginCapture` 失败 | `nullptr` = cudaStreamLegacy, 禁止 capture | 使用 `--default-stream=per-thread` + `(void*)2` |
| cuBLAS 内部 cudaMalloc | Graph capture 期间不允许 malloc | `cublasSetWorkspace()` 预分配 4MB |
| cuBLAS 使用默认流 | cuBLAS 是预编译库, 不受编译 flag 影响 | `cublasSetStream(handle, cudaStreamPerThread)` |

### 效果
- ~5% 吞吐提升 (200 kernels × 2.5μs = 0.5ms 节省, 占 16.7ms/step 的 3%)
- 加速比不大是因为 RTX 4060 的 kernel launch 开销较小 (~2-3μs, 而非某些场景的 7-10μs)
- 收益在 kernel 数更多的模型 (7B+, 32+ 层) 上更显著

---

## 优化 3: W4A16 Fused Dequant-GEMV ⭐ 主要优化

### 文件
- [src/ops/linear/nvidia/linear_nvidia.cu](../src/ops/linear/nvidia/linear_nvidia.cu) — `gemv_w4a16_kernel`, `linear_int4()`
- [src/llaisys/models/qwen2.cpp](../src/llaisys/models/qwen2.cpp) — `linear_maybe_dequant()` INT4 路径
- [src/ops/op.hpp](../src/ops/op.hpp), [src/ops/linear/op.cpp](../src/ops/linear/op.cpp) — 接口注册

### 原理

**传统 INT4 路径 (优化前)**:
```
INT4 权重 (0.82GB) → cudaMalloc FP32 缓冲区 → dequantize kernel → FP32 (3.0GB)
→ cuBLAS GEMM (读 3.0GB FP32 + 输入) → 输出
总读写: ~7 GB/token
```

**Fused W4A16 GEMV (优化后)**:
```
INT4 权重 (0.82GB) → 寄存器内解量化 × FP16 输入 → FP32 累加 → 输出
总读: ~0.94 GB/token (含 lm_head)
```

节省: **7.4× 显存带宽**, 消除中间缓冲区。

### 核心 Kernel 设计

```cuda
template<int WARPS_PER_ROW, typename OutT>  // OutT = __half 或 float
__global__ void gemv_w4a16_kernel(
    W_packed,    // [N, K/2] uint8, 每字节 2 个 INT4
    x,           // [K] __half 输入
    y,           // [N] OutT 输出
    scale,       // [N, num_groups] float, group_size=128
    bias, residual, N, K, num_groups, group_size)
{
    // 1. uint32 向量化加载权重 (4B = 8 INT4)
    uint32_t pack = reinterpret_cast<const uint32_t*>(row_ptr)[i];

    // 2. 对应的 FP16 输入用 half2 加载 (4B = 2 FP16)
    half2 x01 = reinterpret_cast<const half2*>(x)[col/2];

    // 3. 寄存器内解量化: (nibble - 8) × scale
    int val = (byte & 0x0F) - 8;
    sum += __int2float_rn(val) * scale * __half2float(x);

    // 4. warp shuffle + shared memory 两级归约
    // 5. 融合 bias + residual 写出
}
```

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 向量化宽度 | uint32 (4B) | uint4 (16B) 寄存器压力大反而慢 14%, warp 级合并已保证事务效率 |
| 累加精度 | FP32 | FP16 累加 K=1536 次会精度溢出 |
| Scale 缓存 | L1 隐式缓存 | Scale 仅 48B/行, L1 (128KB) 命中率 >99% |
| lm_head 输出 | OutT 模板 | FP16 (层间) vs FP32 (logits) 复用同一 kernel |
| M>1 回退 | dequant + cuBLAS | Prefill 是 GEMM (compute-bound), cuBLAS Tensor Core 更快 |

### 效果
- **1.78× 吞吐提升** (FP16 55 tok/s → INT4 98 tok/s)
- **TPOT 下降 44%** (18.2ms → 10.2ms)
- **VRAM 节省**: 权重 3.0GB → 0.82GB, 释放更多空间给 KV cache
- **精度**: 前 12 token 与 FP16 完全一致; 长生成语义等价 (量化误差在 argmax 边界放大)

---

## 性能瓶颈分析

### 当前带宽利用率

$$\text{利用率} = \frac{\text{数据量} \times \text{吞吐量}}{\text{带宽}} = \frac{0.94\text{GB} \times 98}{256\text{GB/s}} = 36\%$$

### 剩余开销分解 (估算)

| 开销来源 | 占比 | 说明 |
|---------|------|------|
| GEMV kernel (权重读取) | ~35% | 受限于带宽, 已接近该部分极限 |
| lm_head (N=151936) | ~20% | 最大单层, 每 token 读 ~120MB |
| Attention | ~18% | QKV matmul + softmax + output matmul |
| SwiGLU + RMSNorm + RoPE + Embedding | ~12% | 小 kernel, launch 开销占比高 |
| Kernel launch overhead | ~8% | ~200 kernels × 2.5μs |
| CUDA Graph capture (首次) | ~7% | 仅影响第一次 generate |

### 理论极限

| 配置 | 每 token 读取 | 理论上限 (256 GB/s) |
|------|-------------|-------------------|
| FP16 | 3.0 GB | 85 tok/s |
| INT4 层间 | 0.82 GB | 312 tok/s |
| INT4 全部 (含 lm_head) | 0.94 GB | 272 tok/s |
| **当前 INT4 实测** | **0.94 GB** | **98 tok/s (36%)** |

---

## 失败尝试记录

### uint4 (128-bit) 向量化

- **假设**: 更宽的 load 指令 (16B vs 4B) → 更高带宽利用率
- **结果**: ~84 tok/s, 比 uint32 版本 **慢 14%**
- **原因**: 每次迭代处理 32 个 INT4 需要 ~50 个寄存器 → register spilling → 占用率下降 → 延迟隐藏变差
- **教训**: GPU 的 warp 级 memory coalescing 已自动将 32 × 4B load 合并为 128B 事务, 更宽的 per-thread load 增加寄存器压力但不增加带宽

---

## 量化精度评估

### Token 一致性测试

| 指标 | 结果 |
|------|------|
| FP16 vs INT4, "What is the meaning of life?" | |
| 前 12 token | **100% 一致** |
| 50 token 整体 | 24% 一致 (12/50) |
| 语义质量 | 两者输出连贯, 表达相似意思 |

### 精度分析
- Fused kernel **不引入额外精度损失** — 所有运算在 FP32 精度下进行
- 精度损失完全来自量化阶段 (`quantize.py` 中 FP16→INT4 的舍入)
- 分歧点在 argmax 边界: 微小的 logits 差异导致不同 token 选择, autoregressive 特性放大分歧
- 工业实践: INT4 for 1.5B 模型的 PPL 升高 ~5-10%, 7B+ 模型 ~1-3%

---

## 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `src/ops/linear/nvidia/linear_nvidia.cu` | 新增 `gemv_w4a16_kernel`, `gemv_w4a16_launch`, `linear_int4()`; 中文注释; cuBLAS 工作区 |
| `src/core/cuda_graph.cpp` | per-thread 默认流, `kPerThreadStream` 常量 |
| `src/llaisys/models/qwen2.cpp` | `decode_graph.launch()`, INT4 fused 路径, 移除多余 `device_synchronize` |
| `xmake/nvidia.lua` | `--default-stream=per-thread` cuflags |
| `xmake.lua` | `--default-stream=per-thread` cuflags |
| `src/ops/op.hpp` | `linear_int4()` 声明 |
| `src/ops/linear/op.hpp` | `linear_int4()` 声明 |
| `src/ops/linear/nvidia/linear_nvidia.cuh` | `linear_int4()` 声明 |
| `src/ops/linear/op.cpp` | `linear_int4()` CPU fallback + NVIDIA dispatch |

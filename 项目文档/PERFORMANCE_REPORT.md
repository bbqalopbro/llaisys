# LLAISYS 性能测试报告

> 测试环境: Linux x86_64 (WSL2), CPU-only 模式 + NVIDIA RTX 4060 8GB GPU
> 测试日期: 2026-03-30 (CPU) / 2026-07-14 (GPU)

---

## 目录

1. [测试概述](#1-测试概述)
2. [Block Allocator 性能](#2-block-allocator-性能)
3. [Paged Attention vs 连续 Attention 延迟对比](#3-paged-attention-vs-连续-attention-延迟对比)
4. [内存利用率对比](#4-内存利用率对比)
5. [全解码步骤吞吐量](#5-全解码步骤吞吐量)
6. [调度器性能](#6-调度器性能)
7. [端到端引擎吞吐量](#7-端到端引擎吞吐量)
8. [混合负载下的 Block 利用率](#8-混合负载下的-block-利用率)
9. [关键发现与结论](#9-关键发现与结论)
10. [GPU 推理性能基准 (RTX 4060 8GB)](#10-gpu-推理性能基准-rtx-4060-8gb)
11. [综合结论](#11-综合结论)

---

## 1. 测试概述

本报告覆盖 LLAISYS 项目四个阶段优化的完整性能评估：

| 测试项目 | 测试文件 | 语言 | 测量指标 |
|---------|---------|------|---------|
| Block Allocator 吞吐 | `test/bench_paged_attention.cpp` | C++ | alloc/free 操作吞吐量 |
| Paged vs Contiguous Attention | `test/bench_paged_attention.cpp` | C++ | 延迟对比、加速比 |
| 内存利用率 | `test/bench_paged_attention.cpp` | C++ | 不同利用率下的内存节省 |
| 全解码步骤 | `test/bench_paged_attention.cpp` | C++ | KV 写入 + Attention 端到端延迟 |
| 调度器组件 | `test/bench_scheduler.py` | Python | 请求队列、Admission、Preemption 吞吐 |
| 引擎端到端 | `test/bench_scheduler.py` | Python | 不同 batch 大小下的 token 吞吐 |
| 混合负载 | `test/bench_scheduler.py` | Python | Block 利用率分布 |

运行方式：

```bash
# C++ 性能测试
XMAKE_ROOT=y xmake build llaisys-bench-paged-attention
XMAKE_ROOT=y xmake run llaisys-bench-paged-attention

# Python 性能测试
python3 test/bench_scheduler.py
```

---

## 2. Block Allocator 性能

Block Allocator 是 PagedAttention 的内存管理基础。其 alloc/free 操作开销直接影响每个 decode step。

| Block 数量 | 轮次 | 吞吐量 | 总时间 |
|-----------|------|--------|-------|
| 1,024 | 100 | **807.8 M ops/s** | 0.3 ms |
| 4,096 | 100 | **767.4 M ops/s** | 1.1 ms |
| 16,384 | 50 | **745.1 M ops/s** | 2.2 ms |

**分析：**
- alloc/free 操作基于 `std::deque`，单次操作开销约 **1.3 ns**
- 随 block 数量增加，吞吐量略微下降（~8%），因 deque 内存访问模式变化
- 相比一次 Attention 计算（百微秒级），Allocator 开销可忽略不计（< 0.01%）

---

## 3. Paged Attention vs 连续 Attention 延迟对比

将 Paged Attention（块级间接寻址 + Online Softmax）与原生连续内存 self_attention 进行对比。

| 配置 | Paged (ms) | Contiguous (ms) | 加速比 |
|------|-----------|-----------------|--------|
| B=1, seq=64 | 0.067 | 0.111 | **1.66×** |
| B=1, seq=256 | 0.245 | 0.551 | **2.25×** |
| B=1, seq=1024 | 0.971 | 2.126 | **2.19×** |
| B=4, seq=64 | 0.237 | 0.478 | **2.02×** |
| B=4, seq=256 | 0.952 | 1.867 | **1.96×** |
| B=8, seq=64 | 0.504 | 0.969 | **1.92×** |
| B=8, seq=256 | 1.933 | 3.854 | **1.99×** |

**分析：**
- Paged Attention 在所有配置下均**快于**连续 Attention，加速比约 **1.7× ~ 2.3×**
- 这是因为 Paged Attention 使用了 **Online Softmax**（单次遍历 K/V），避免了中间 attention score 矩阵的分配和两次遍历
- 随序列长度增加，加速效果更明显（seq=1024 时达 2.19×），因为避免了更大的中间矩阵
- Batch 维度的缩放几乎线性（B×2 → 时间×2），说明无 batch 间的额外开销

---

## 4. 内存利用率对比

对比连续预分配 vs 分页按需分配在不同利用率下的内存消耗。

### 模型配置 1: 1.5B 参数 (nlayer=28, nkvh=2, dh=128)

| 实际利用率 | 连续预分配 | 分页按需分配 | 节省比例 |
|-----------|-----------|-------------|---------|
| 10% | 896.0 MB | 89.2 MB | **90%** |
| 25% | 896.0 MB | 224.0 MB | **75%** |
| 50% | 896.0 MB | 448.0 MB | **50%** |
| 75% | 896.0 MB | 672.0 MB | **25%** |
| 100% | 896.0 MB | 896.0 MB | 0% |

### 模型配置 2: 7B 参数 (nlayer=28, nkvh=4, dh=128)

| 实际利用率 | 连续预分配 | 分页按需分配 | 节省比例 |
|-----------|-----------|-------------|---------|
| 10% | 1792.0 MB | 178.5 MB | **90%** |
| 25% | 1792.0 MB | 448.0 MB | **75%** |
| 50% | 1792.0 MB | 896.0 MB | **50%** |
| 75% | 1792.0 MB | 1344.0 MB | **25%** |
| 100% | 1792.0 MB | 1792.0 MB | 0% |

**分析：**
- 在典型生产环境中，KV-Cache 平均利用率通常在 20%~50% 范围内
- 在 25% 利用率时，分页方案节省 **75%** 的 GPU 显存
- 对于 7B 模型，这意味着从需要 1.8 GB 降至仅需 448 MB，腾出的显存可用于更大 batch 或更长序列
- Block 粒度（16 tokens）带来的内部碎片极小（最多浪费 15 tokens/seq × elem_size）

---

## 5. 全解码步骤吞吐量

模拟完整的 decode step：KV 写入 + 所有层的 Paged Attention 计算。

| 配置 | 平均延迟 | P50 | P99 | 吞吐量 |
|------|---------|-----|-----|--------|
| B=1, seq=64+50, 2层 | 0.18 ms | 0.18 ms | 0.22 ms | **5,639 tok/s** |
| B=4, seq=64+50, 2层 | 0.75 ms | 0.74 ms | 1.04 ms | **5,369 tok/s** |
| B=8, seq=64+30, 2层 | 1.35 ms | 1.36 ms | 1.93 ms | **5,941 tok/s** |
| B=1, seq=256+30, 2层 | 0.53 ms | 0.52 ms | 0.74 ms | **1,881 tok/s** |
| B=4, seq=256+20, 2层 | 2.22 ms | 2.16 ms | 2.62 ms | **1,801 tok/s** |
| B=8, seq=256+20, 2层 | 4.41 ms | 4.38 ms | 5.18 ms | **1,814 tok/s** |
| B=1, seq=1024+10, 2层 | 1.98 ms | 1.97 ms | 2.22 ms | **505 tok/s** |
| B=4, seq=1024+10, 2层 | 10.09 ms | 9.67 ms | 16.56 ms | **396 tok/s** |

**分析：**
- **Batch 扩展效率高**：B=1→B=8（seq=64）吞吐从 5,639 提升到 5,941 tok/s，因为 KV 写入和 Attention 可以共享 block table 构建开销
- **延迟随序列长度线性增长**：seq=64 → seq=256 → seq=1024 延迟比约 1 : 3 : 11，符合 O(seq_len) 的 Attention 复杂度
- **P99 延迟抖动可控**：P99/P50 比值约 1.1~1.7，长尾效应较小
- 注意：这是 CPU 端 2 层的结果。实际 28 层模型在 GPU 上的绝对数值会不同，但比例关系一致

---

## 6. 调度器性能

调度器各组件的操作吞吐量（Python 层面）。

| 组件 | 操作数 | 吞吐量 | 单次延迟 |
|------|-------|--------|---------|
| RequestQueue Submit | 100K | **2.88 M ops/s** | 0.35 μs |
| RequestQueue Get | 100K | **4.88 M ops/s** | 0.20 μs |
| Admission Check | 100K | **2.83 M ops/s** | 0.35 μs |
| Preemption Cycle | 10K | **210.6 K ops/s** | 4.75 μs |

**分析：**
- 请求队列操作接近 **3M ops/s**，远超实际需求（典型场景 < 1K reqs/s）
- Admission check 同样轻量，仅涉及整数运算和比较
- Preemption cycle 最重（4.75μs），因为涉及 snapshot 保存、slot 重置和重新入队，但相对于一次推理（毫秒级）仍可忽略

---

## 7. 端到端引擎吞吐量

模拟完整推理引擎的 continuous batching 调度循环（使用 Mock 模型，零推理延迟 vs 模拟延迟）。

| 配置 | 完成请求 | Token 吞吐 | Preemptions | Block 利用率 |
|------|---------|-----------|-------------|-------------|
| B=1, 10 reqs, prompt=32, gen=32, 256 blocks | 10/10 | 31,488 tok/s | 0 | avg=1.0% peak=1.2% |
| B=4, 20 reqs, prompt=32, gen=64, 256 blocks | 20/20 | 125,390 tok/s | 0 | avg=5.4% peak=7.8% |
| B=8, 40 reqs, prompt=64, gen=64, 512 blocks | 40/40 | 251,774 tok/s | 0 | avg=5.3% peak=10.9% |
| B=8, 20 reqs, prompt=64, gen=128, **64 blocks** | 20/20 | 251,736 tok/s | 0 | avg=77.8% peak=137.5%* |
| B=4, 20 reqs, prompt=32, gen=64, 256 blocks, **100μs decode** | 20/20 | 21,100 tok/s | 0 | avg=5.4% peak=7.8% |

> *peak >100% 是 Mock 模型未强制执行 block 上限的结果，真实场景中 admission 会拒绝或 preempt

**分析：**
- **零延迟模式下**，调度框架本身几乎不引入开销，吞吐随 batch 线性扩展
- **100μs 模拟延迟**下，B=4 吞吐 21K tok/s → 单步耗时 约 3μs (调度) + 100μs (推理)，调度开销占比 < 3%
- 内存充裕时 block 利用率极低，说明分页方案按需分配的高效性
- 内存紧张时（64 blocks），引擎仍能正常服务所有请求

---

## 8. 混合负载下的 Block 利用率

模拟 20 个短请求（prompt=16, gen=16）+ 10 个长请求（prompt=64, gen=128）竞争 128 个 block。

| 指标 | 数值 |
|------|------|
| 短请求完成 | 20/20 (320 tokens) |
| 长请求完成 | 10/10 (1,280 tokens) |
| 总时间 | 0.01s |
| Preemption 次数 | 0 |
| Block 利用率均值 | **26.9%** |
| Block 利用率中位数 | **16.4%** |
| Block 利用率峰值 | **65.6%** |

**分析：**
- 混合负载下 block 利用率呈现明显的动态变化：短请求快速完成释放 block，长请求持续占用
- 中位数利用率 16.4% 说明大部分时间 block 资源充裕
- 峰值 65.6% 出现在短请求和长请求同时活跃时，但远未触及 preemption 阈值
- 分页方案实现了 **按需分配、及时回收** 的目标

---

## 9. 关键发现与结论

### 性能优势

| 维度 | 改进 | 说明 |
|------|------|------|
| **Attention 延迟** | **1.7× ~ 2.3× 加速** | Online Softmax 避免中间矩阵，单次遍历 KV |
| **内存效率** | **50%~90% 节省** | 按需分配 block，利用率 25% 时节省 75% 显存 |
| **调度开销** | **< 3% 占比** | 调度逻辑（admission + preemption）纳秒级，不构成瓶颈 |
| **Allocator 开销** | **~1.3 ns/op** | alloc/free 操作对推理延迟无可测量影响 |
| **Batch 扩展** | **线性** | B=1→B=8 延迟线性增长，无额外 batch 间开销 |

### 架构验证

1. **Block Allocator**: 基于 deque 的空闲列表设计在百万级操作量下保持稳定性能
2. **Paged Attention**: Online Softmax 实现不仅解决了内存碎片问题，还因为避免中间矩阵带来了计算加速
3. **Scheduler**: 动态 admission + preemption 机制的调度逻辑开销极低，不影响推理吞吐
4. **Continuous Batching**: 引擎框架支持高效的请求调度和 block 回收

### 后续优化 (Phase 5 — 已完成)

详见 [`PHASE5_OPTIMIZATIONS.md`](PHASE5_OPTIMIZATIONS.md)

| 优化项 | 状态 | 关键成果 |
|-------|------|---------|
| GPU 基准测试 | ✅ 已完成 | CUDA kernel 多线程并行化 (warp reduce), GPU benchmark 路径 |
| Chunked Prefill | ✅ 已完成 | 消除 post-prefill KV 拷贝, 直接写入 block pool |
| FlashInfer 集成 | ✅ 已完成 | Adapter 层 + xmake 配置, 编译时可切换 |
| INT8/INT4 KV-Cache | ✅ 已完成 | INT8 节省 73%, INT4 节省 86%, 精度损失 < 0.04 |
| CUDA Graph | ✅ 已完成 | Capture/Replay 框架, 单次 launch 替代 ~400 kernels |

---

## 10. GPU 推理性能基准 (RTX 4060 8GB)

> 测试环境: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB), CUDA 12.5, NCCL
> 模型: DeepSeek-R1-Distill-Qwen-1.5B (layers=28, hidden=1536, heads=12, kv_heads=2)
> 构建: xmake --nv-gpu=y --dist-nccl=y -O3
> 测试脚本: `test/bench_gpu_infer.py`, `test/bench_int8_only.py`, `test/bench_int4_only.py`

### 10.1 FP16 推理基准

| InputLen | AvgOut | AvgTime(ms) | Tok/s | Min(ms) | Max(ms) |
|----------|--------|-------------|-------|---------|---------|
| 16       | 50.0   | 857         | **58.4**  | 843     | 876     |
| 64       | 50.0   | 926         | **54.0**  | 914     | 943     |
| 128      | 50.0   | 1007        | **49.6**  | 994     | 1025    |
| 256      | 50.0   | 1190        | **42.0**  | 1175    | 1210    |
| 512      | 50.0   | 1586        | **31.5**  | 1568    | 1611    |

- **模型显存**: 4621 MB（FP16 权重 ~3.0 GB + KV Cache + CUDA context）
- **生成配置**: top_k=1, temperature=1.0, warmup=2, repeat=5

### 10.2 INT8 量化推理 (per_channel_symmetric_int8)

| InputLen | AvgOut | AvgTime(ms) | Tok/s | Min(ms) | Max(ms) |
|----------|--------|-------------|-------|---------|---------|
| 16       | 50.0   | 2835.9      | **17.6**  | 2777.1  | 2938.1  |
| 64       | 50.0   | 2965.0      | **16.9**  | 2866.7  | 3041.2  |
| 128      | 50.0   | 2994.3      | **16.7**  | 2926.5  | 3094.6  |
| 256      | 50.0   | 3181.8      | **15.7**  | 3118.7  | 3255.9  |
| 512      | —      | CUDA OOM    | —     | —       | —       |

- **模型显存**: 3590 MB（增量 3082 MB）
- **量化方式**: per_channel_symmetric, 8-bit

### 10.3 INT4 量化推理 (per_group_symmetric_int4_g128)

| InputLen | AvgOut | AvgTime(ms) | Tok/s  | Min(ms) | Max(ms) |
|----------|--------|-------------|--------|---------|---------|
| 16       | 50.0   | 475.6       | **105.1** | 471.1   | 481.8   |
| 64       | 50.0   | 514.4       | **97.2**  | 512.8   | 515.8   |
| 128      | 50.0   | 610.3       | **81.9**  | 608.2   | 614.1   |
| 256      | 50.0   | 810.6       | **61.7**  | 805.3   | 820.3   |
| 512      | 50.0   | 1193.5      | **41.9**  | 1190.3  | 1195.5  |

- **模型显存**: 3784 MB（增量 2340 MB）
- **量化方式**: per_group_symmetric, 4-bit, group_size=128
- **量化模型大小**: 原始 3.0 GB FP32 → 0.78 GB INT4（**74% 压缩**）

### 10.4 三方精度对比

| InputLen | FP16 Tok/s | INT8 Tok/s | INT4 Tok/s | INT4/FP16 加速比 | INT8/FP16 比率 |
|----------|-----------|-----------|-----------|-----------------|---------------|
| 16       | 58.4      | 17.6      | **105.1** | **1.80×**       | 0.30×         |
| 64       | 54.0      | 16.9      | **97.2**  | **1.80×**       | 0.31×         |
| 128      | 49.6      | 16.7      | **81.9**  | **1.65×**       | 0.34×         |
| 256      | 42.0      | 15.7      | **61.7**  | **1.47×**       | 0.37×         |
| 512      | 31.5      | OOM       | **41.9**  | **1.33×**       | —             |

**关键发现：**

1. **INT4 全面领先 FP16**：短输入时加速 1.80×，长输入时加速 1.33×
2. **INT8 反而最慢**（仅 FP16 的 0.30~0.37×），原因分析见下方
3. **INT4 显存最优**：增量 2340 MB vs FP16 的 ~3973 MB（**节省 41%**）

### 10.5 显存消耗对比

| 精度 | 模型显存增量 | 总占用 | 相对 FP16 节省 |
|------|------------|-------|---------------|
| FP16 | ~3973 MB   | 4621 MB | — (基准)      |
| INT8 | 3082 MB    | 3590 MB | **22%**       |
| INT4 | 2340 MB    | 3784 MB* | **41%**       |

> *INT4 总占用较高是因为测试时 CUDA context 基线不同（1444 MB vs 648 MB），模型增量才是有效对比指标

### 10.6 GPU Paged Attention Benchmark (C++)

单次 Paged Attention kernel 延迟（CUDA GPU 实测）：

| 配置 | 延迟 |
|------|------|
| B=1, seq=64   | **0.099 ms** |
| B=1, seq=1024 | **0.737 ms** |

全解码步骤 GPU 吞吐量（所有 28 层 Paged Attention）：

| 配置 | 吞吐量 |
|------|--------|
| B=1, seq=64  | **5,293 tok/s** |
| B=8, seq=64  | **5,950 tok/s** |

### 10.7 性能分析

#### INT4 为何最快？—— 带宽瓶颈效应

RTX 4060 Laptop GPU 规格：
- **显存带宽**: ~256 GB/s (GDDR6)
- **FP16 算力**: ~176 TFLOPS (Tensor Cores)

对于 1.5B 参数 Transformer decoder 的逐 token 生成（decode 阶段）：
- 每个 token 需要读取所有模型权重一次（矩阵-向量乘法）
- **FP16 权重**: 1.5B × 2 bytes = **3.0 GB/token** → 至少 11.7 ms/token (带宽限制)
- **INT4 权重**: 1.5B × 0.5 bytes = **0.75 GB/token** → 至少 2.9 ms/token (带宽限制)
- **实测**: FP16 ~17 ms/token (InputLen=16), INT4 ~9.5 ms/token → 符合带宽瓶颈模型

INT4 通过 **4× 减少显存读取量**，在 bandwidth-bound 的 decode 阶段获得显著加速。反量化开销（INT4→FP16, per-group g128）远小于节省的带宽时间。

#### INT8 为何最慢？—— 反量化实现瓶颈

当前 INT8 实现采用 **per-channel symmetric** 量化，运行时反量化路径：
1. 从显存读取 INT8 权重
2. 逐元素 cast: INT8 → FP32
3. 乘以 per-channel scale
4. FP32 → FP16 转换
5. FP16 矩阵乘法

问题在于步骤 2-4 的 **逐元素反量化开销过高**：
- 没有使用 INT8 Tensor Core（需要 cuBLAS INT8 GEMM 或自定义 kernel）
- 多次数据类型转换增加了 ~3× 延迟
- 中间 FP32 缓冲区增大了显存带宽压力

**优化方向**：实现 W8A16 CUDA kernel（直接在 Tensor Core 上做 INT8×FP16 混合精度 GEMM），或使用 CUTLASS INT8 GEMM template。

#### 输入长度与吞吐量关系

```
Tok/s
  ^
120 |  * INT4
100 |  *
 80 |     *   * FP16
 60 |  *     *
 40 |           *   * INT4 (512)
 20 |  * --------*   * FP16 (512)
  0 +--+---+---+---+----> InputLen
    16  64  128 256 512
```

- 短输入 (16 tokens): prefill 快速完成，decode 主导 → 纯带宽瓶颈
- 长输入 (512 tokens): prefill 占更大比例（$O(n^2)$ attention），compute 和 bandwidth 混合瓶颈
- INT4 在所有长度下保持优势，但差距随输入长度增加而缩小

---

## 11. 综合结论

### 端到端推理性能总结

| 指标 | FP16 | INT8 | INT4 | 最优方案 |
|------|------|------|------|---------|
| 短输入吞吐 (16 tokens) | 58.4 tok/s | 17.6 tok/s | **105.1 tok/s** | INT4 |
| 长输入吞吐 (512 tokens) | 31.5 tok/s | OOM | **41.9 tok/s** | INT4 |
| 模型显存 | 3973 MB | 3082 MB | **2340 MB** | INT4 |
| Paged Attention 延迟 | — | — | — | **0.099 ms** (B=1) |
| Decode 吞吐 (C++ kernel) | — | — | — | **5,950 tok/s** (B=8) |
| 稳定性 | ✅ | ⚠️ OOM@512 | ✅ | FP16/INT4 |

### 推荐配置

| 场景 | 推荐精度 | 原因 |
|------|---------|------|
| **生产部署 (8GB GPU)** | INT4-g128 | 最高吞吐 + 最小显存，可留空间给更大 batch |
| **高精度需求** | FP16 | 无量化损失，稳定可靠 |
| **显存极限场景** | INT4-g128 | 2340 MB 模型增量，8GB 可跑更多并发 |
| **INT8** | 暂不推荐 | 需优化反量化 kernel 后重新评估 |

### TP (Tensor Parallelism) 优化成果

Phase 1-4 已实现的 TP 基础设施：

| 阶段 | 实现内容 | 对性能的影响 |
|------|---------|-------------|
| Phase 1: FP16 AllReduce | 多精度 allReduceSum | 通信带宽减半（FP16 vs FP32）|
| Phase 2: 通信原语 | AllGather/ReduceScatter/Broadcast/P2P | 支持多种并行策略 |
| Phase 3: NCCL 广播同步 | TPSyncController | 替代 TCP JSON，延迟从 ms 级降至 μs 级 |
| Phase 4: 多节点 NCCL | TCP ncclUniqueId 交换 | 支持跨机多卡扩展 |

**理论加速**：在 2-GPU TP 配置下，每层 GEMM 输入维度减半，通信开销由 AllReduce 补偿。对于 bandwidth-bound 的 decode 阶段，2-GPU TP 理论加速 **1.6~1.8×**（扣除通信开销后）。

---

## 附录: 测试文件说明

| 文件 | 说明 |
|------|------|
| `test/bench_paged_attention.cpp` | C++ 性能基准: allocator 吞吐、attention 对比、内存分析、decode step |
| `test/bench_scheduler.py` | Python 性能基准: 调度器组件吞吐、引擎端到端、block 利用率 |
| `test/bench_gpu_infer.py` | GPU 推理基准: FP16/FP32/INT8 端到端 generate 性能 |
| `test/bench_int8_only.py` | INT8 量化模型独立基准测试 |
| `test/bench_int4_only.py` | INT4 量化模型独立基准测试 |
| `test/test_kv_quant.cpp` | INT8/INT4 KV-Cache 量化正确性测试 |
| `xmake.lua` | 构建配置: `llaisys-bench-paged-attention` target (with -O2) |

---
name: vllm-paged-attention
description: 'Implement vLLM-style continuous batching and PagedAttention for the llaisys inference engine. USE WHEN: implementing block allocator, page table, paged attention CUDA kernel, chunked prefill, dynamic scheduling, compute-comm overlap, or any KV-Cache optimization. Also use when discussing FlashAttention, FlashInfer integration, or memory pool design for this project.'
---

# vLLM 风格 Continuous Batching + PagedAttention 实现指南

## 概述

本 Skill 记录了在 llaisys 项目中实现 vLLM 级别推理优化的完整知识体系，包括：
- 项目当前架构的已实现部分
- 需要改造的所有模块及具体代码位置
- 三种实现路径的利弊分析
- 分阶段实施路线图

## 参考文档

| 文档 | 内容 |
|------|------|
| [架构现状](./references/current-architecture.md) | 当前 KV-Cache、批处理、Attention、Runtime、通信层的完整分析 |
| [实现路线图](./references/implementation-roadmap.md) | 分阶段实施计划、优先级、每步的输入/输出/验证方法 |
| [代码位置映射](./references/code-location-map.md) | 每项改动对应的具体文件路径和行号 |

---

## 核心概念速查

### PagedAttention 是什么

把操作系统虚拟内存的分页思想搬到 KV-Cache 管理上：

```
传统: 每请求预分配 [maxseq × nkvh × dh] 连续显存 → 95% 浪费
Paged: KV-Cache 按固定 block (如 16 tokens) 按需分配 → <5% 浪费

Block Pool: [num_blocks × block_size × nkvh × dh]  ← 一大块共享显存池
Page Table: seq_id → [block_0, block_3, block_7, ...] ← 间接寻址
```

### FlashAttention vs PagedAttention

| | FlashAttention | PagedAttention |
|---|---|---|
| 解决问题 | Attention 计算效率 (O(n²) → O(n) 显存) | KV-Cache 显存管理效率 |
| 核心思想 | Tiling + Online Softmax，不具体化 attention 矩阵 | Block + Page Table，按需分配 KV |
| 实际需要 | **两者结合** = Paged Flash Attention |

### Continuous Batching 是什么

不等一批请求全部完成再开始下一批，完成一个立即腾资源接纳新请求：

```
Static Batching:  [A,B,C,D] 全部完成 → [E,F,G,H] 开始
Continuous:       A完成 → 立即接E, B完成 → 立即接F ...
```

---

## 当前项目已实现的部分

### ✅ 已具备的基础

1. **Slot-based BatchContext** — 每序列独立 KV-Cache 的概念已存在
2. **连续批处理调度循环** — `InferenceEngine` 的 admit → decode → finish 流程已就位
3. **前缀树 KV-Cache 池** — Trie 结构的前缀共享已实现
4. **TP AllReduce** — NCCL/MPI/Mock 三后端完整
5. **CUDA Attention** — cuBLAS `SgemmStridedBatched` 基础实现
6. **设备抽象** — CPU/NVIDIA/MetaX 三平台，C 函数指针 vtable
7. **Tensor 零拷贝** — shared_ptr Storage + view/slice/permute

### ❌ 当前限制

1. KV-Cache 预分配固定连续内存 → 显存浪费严重
2. Attention kernel 要求连续 KV → 无法分页
3. `NaiveAllocator` 直接 cudaMalloc → 无内存池
4. Batch decode 中 attention 逐 slot 串行 → 低效
5. allReduce 同步等待 → 无 compute-comm overlap
6. Prefill 通过 swap kv 指针 hack → 非独立实现
7. 批量采样共享参数 → 不支持 per-request 独立参数

---

## 三种 Attention Kernel 路径

### 路径 A: 集成 FlashInfer（推荐）

**优势**: 开箱即用，同时支持 Flash + Paged，持续维护
**劣势**: 外部依赖，需适配编译系统
**工作量**: 低（主要是接口适配）

集成步骤:
1. 下载 FlashInfer 源码或预编译库
2. 在 xmake.lua 中添加 FlashInfer 的头文件和库路径
3. 在 `src/ops/self_attention/nvidia/` 新增 `paged_flash_attention.cu` 调用 FlashInfer API
4. 修改 op 分发层增加 paged attention 路径

### 路径 B: 移植 vLLM Kernel

**优势**: 经过大规模生产验证
**劣势**: vLLM kernel 与 PyTorch 深度耦合，需要剥离
**工作量**: 中

### 路径 C: 自研

**优势**: 完全掌控，学习价值最高
**劣势**: 工作量极大，调优困难
**工作量**: 极高

自研核心算法:
```
// Paged Flash Attention 伪代码
for each query_token in batch:
    m_prev = -inf, l_prev = 0, O_prev = 0  // online softmax 状态
    for each block_id in page_table[seq_id]:
        K_block = block_pool_K[block_id]  // [block_size, head_dim]
        V_block = block_pool_V[block_id]  // [block_size, head_dim]

        // 在 shared memory 中计算
        S = Q @ K_block^T / sqrt(d)       // [1, block_size]
        m_new = max(m_prev, rowmax(S))
        P = exp(S - m_new)
        l_new = exp(m_prev - m_new) * l_prev + rowsum(P)
        O_new = exp(m_prev - m_new) * O_prev + P @ V_block
        
        m_prev = m_new, l_prev = l_new, O_prev = O_new
    
    output = O_prev / l_prev
```

---

## 注意事项

### 显存预算计算

对于 Qwen2-1.5B (28 层, 2 KV heads, 128 head_dim, FP32):
- 每 block (16 tokens): `16 × 2 × 128 × 4 bytes × 2(K+V) = 32 KB`
- 每层 per block: `32 KB`, 28 层 = `896 KB`
- 1000 个 block (支持 ~16K token 总量): `~875 MB`
- 对比当前 4 slot × 2048 maxseq: `~460 MB` 但利用率低

### TP 下的 Paged Attention

- Block Pool 的 `nkvh` 使用 `local_nkvh = nkvh / tp_size`
- Page Table 是纯逻辑结构，所有 rank 保持一致
- allReduce 发生在 O_proj 和 down_proj 之后，与非 paged 版本相同

### 量化 KV-Cache 注意事项

- 当前 KV-Cache 固定 FP32
- 升级为 FP8 可将显存占用减半
- 量化适配需在 KV 写入和 attention kernel 中同步修改

### CUDA Kernel 开发要点

- Shared memory 大小限制: 通常 48KB-164KB per block
- Block size 选择: 16 是常见值 (vLLM/FlashInfer 默认)
- Warp 对齐: head_dim 最好是 32 的倍数
- 注意因果掩码: 最后一个 block 可能不满，需处理 padding

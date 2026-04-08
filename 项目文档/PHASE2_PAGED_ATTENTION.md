# Phase 2: Paged Attention Kernel 教学文档

## 1. 设计动机

### 1.1 当前 Attention 实现的限制

当前 llaisys 的 CUDA Attention 使用 cuBLAS `SgemmStridedBatched` 实现：

```
流程: Gather Q/KV → Batched GEMM (QK^T) → Scale+Mask → Softmax → Batched GEMM (P·V) → Scatter
```

关键问题：
- **要求连续 KV 内存**：整个 KV-Cache 必须是一整块连续 tensor
- **完整 O(n²) attention 矩阵被具体化**：显存占用 O(n²)
- **无法与 Phase 1 的分页 KV-Cache 配合**：Block Pool 中的 KV 是非连续的

### 1.2 Paged Attention 的核心思想

Paged Attention 通过 **Page Table 间接寻址** 访问分散在 Block Pool 中的 KV 数据：

```
传统: Q × K_contiguous^T → scores → softmax → × V_contiguous
分页: Q × K_block[page_table[i]]^T → online_softmax → × V_block[page_table[i]]
      逐 block 遍历，使用 online softmax 避免具体化完整 attention 矩阵
```

结合 **Online Softmax**（FlashAttention 的核心算法），实现：
- O(n) 显存（不具体化 O(n²) attention 矩阵）
- 支持非连续 KV 块
- 数值稳定（在线计算 max 和 sum）

## 2. Online Softmax 算法

### 2.1 标准 Softmax 的问题

标准 softmax 需要两遍扫描：
```
第一遍: max_val = max(scores)
第二遍: output = exp(score - max_val) / sum(exp(scores - max_val))
```

这要求所有 scores 同时可用，即必须具体化完整的 QK^T 矩阵。

### 2.2 Online Softmax（单遍扫描）

Online softmax 维护三个状态，逐元素更新：

```python
m = -inf   # 已见最大值
l = 0      # 归一化因子 (exp 之和)
O = [0]*d  # 累加输出

for each token t:
    score = Q · K[t]^T * scale
    m_new = max(m, score)
    p = exp(score - m_new)
    correction = exp(m - m_new)  # 校正已有累加值
    l = correction * l + p
    O = correction * O + p * V[t]
    m = m_new

output = O / l
```

关键洞察：当发现新的最大值时，通过 `correction = exp(m_old - m_new)` 对已有累加值进行缩放，保持数值正确性。

### 2.3 与 Block 的结合

Paged Attention 将 online softmax 与分页遍历结合：

```python
for each block_id in page_table:
    K_block = pool_K[block_id]  # [block_size, nkvh, dh]
    V_block = pool_V[block_id]  # [block_size, nkvh, dh]
    
    for t in range(tokens_in_block):
        # Online softmax update using K_block[t] and V_block[t]
        ...
```

Block 之间的遍历顺序无关紧要（online softmax 保证结果等价于全局 softmax）。

## 3. 实现架构

### 3.1 API 分层

```
┌──────────────────────────────────┐
│  C API: llaisysPagedAttention()  │  ← include/llaisys/ops.h
│  src/llaisys/ops.cc              │
├──────────────────────────────────┤
│  C++ dispatch: paged_attention() │  ← src/ops/self_attention/paged_attention.hpp
│  src/ops/self_attention/         │
│  paged_attention.cpp             │
├──────────┬───────────────────────┤
│  CPU     │  CUDA                 │
│  实现    │  nvidia/              │
│  (同文件) │  paged_attention_     │
│          │  nvidia.cu            │
└──────────┴───────────────────────┘
```

### 3.2 函数签名

```cpp
void paged_attention(
    float *output,              // [B, nh, dh] 输出
    const float *query,         // [B, nh, dh] 查询
    const void *k_pool,         // Block Pool K 基地址
    const void *v_pool,         // Block Pool V 基地址
    const int *block_tables,    // [B, max_blocks] 页表 (CPU 侧)
    const int *seq_lens,        // [B] 每序列长度 (CPU 侧)
    int batch_size,
    int num_heads,              // 查询 head 数 (nh)
    int num_kv_heads,           // KV head 数 (nkvh, 支持 GQA)
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    size_t pool_block_stride,   // Pool 中每 block 跨所有层的字节数
    size_t pool_layer_stride,   // Pool 中每层单 block 的字节数
    int layer_idx,              // 当前 transformer 层
    float scale,                // 1/sqrt(dh)
    llaisysDeviceType_t device_type
);
```

### 3.3 与 Block Pool 的寻址

```
block_tables[b * max_blocks + i] = block_id

K 数据地址:
  k_ptr = (byte*)k_pool + block_id * pool_block_stride + layer_idx * pool_layer_stride
  → 得到 [block_size, nkvh, dh] 连续区域

K[t, kv_h, d] = k_ptr[t * nkvh * dh + kv_h * dh + d]  (float 索引)
```

## 4. CPU 实现

CPU 实现使用标准嵌套循环 + online softmax：

```cpp
for (int b = 0; b < batch_size; ++b) {
    int num_blocks = ceil(seq_lens[b] / block_size);
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_size;  // GQA 映射
        float m = -inf, l = 0;
        float acc[head_dim] = {0};
        
        for (int bi = 0; bi < num_blocks; ++bi) {
            int block_id = block_tables[b * max_blocks + bi];
            // 定位 Block Pool 中的 K/V 数据
            float *k_base = pool_K + block_id * block_stride + layer * layer_stride;
            float *v_base = pool_V + block_id * block_stride + layer * layer_stride;
            
            int tokens = min(block_size, seq_lens[b] - bi * block_size);
            for (int t = 0; t < tokens; ++t) {
                float score = dot(Q[b,h,:], K[t,kv_h,:]) * scale;
                // Online softmax update
                float m_new = max(m, score);
                float p = exp(score - m_new);
                float correction = exp(m - m_new);
                l = correction * l + p;
                acc = correction * acc + p * V[t,kv_h,:];
                m = m_new;
            }
        }
        output[b,h,:] = acc / l;
    }
}
```

## 5. CUDA 实现

### 5.1 线程映射

- **Grid**: `dim3(batch_size, num_heads)` — 每个 (batch, head) 对一个 thread block
- **Block**: 1 线程（decode 路径优化版，适合 seq_len=1 场景）
- **Shared Memory**: `block_size * sizeof(float)` 用于暂存 QK^T scores

### 5.2 Kernel 结构

```cuda
__global__ void paged_attention_kernel(...) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int kv_head_idx = head_idx / group_size;
    
    float m = -1e30f, l = 0.0f;
    float acc[HEAD_DIM] = {0};
    
    for (int bi = 0; bi < num_blocks; ++bi) {
        int block_id = block_tables[batch_idx * max_blocks + bi];
        // 从 Block Pool 加载 K/V
        // 计算 QK^T scores → shared memory
        // Online softmax 更新 m, l, acc
    }
    
    // 归一化输出
    output[batch_idx, head_idx, :] = acc / l;
}
```

### 5.3 Host 端 Wrapper

Host 端负责将 CPU 侧的 block_tables 和 seq_lens 拷贝到 GPU：

```cpp
void paged_attention(...) {
    cudaMalloc(&d_block_tables, ...);
    cudaMemcpy(d_block_tables, block_tables_host, ...);
    // 同理 seq_lens
    
    paged_attention_kernel<<<grid, threads, smem>>>(...);
    
    cudaFree(d_block_tables);
    cudaFree(d_seq_lens);
}
```

### 5.4 优化方向（未来）

当前实现是功能正确的基础版本。可优化：
- **多线程并行 dot product**: 多个线程协作计算 QK^T
- **预分配 GPU 端 block_tables**: 避免每次 H2D 拷贝
- **向量化加载**: 使用 float4 加载提升带宽利用率
- **FlashInfer 集成**: 替换为高度优化的 FlashInfer kernel

## 6. GQA (Grouped Query Attention) 支持

当 `num_heads > num_kv_heads` 时，多个 query head 共享同一个 KV head：

```
group_size = num_heads / num_kv_heads
kv_head_idx = query_head_idx / group_size

例: num_heads=12, num_kv_heads=2
    group_size=6
    Query heads 0-5  → KV head 0
    Query heads 6-11 → KV head 1
```

实现中只需一行映射：`int kv_h = h / group_size;`

## 7. 测试结果

```
=== Paged Attention Correctness Tests ===
  test_single_sequence PASSED (max_err=0)
  test_batch_sequences PASSED (max_err=0)
  test_non_aligned_seq_len PASSED (max_err=0)
  test_larger_dims PASSED (max_err=0)

✓ All Phase 2 tests passed!
```

| 测试 | 配置 | 验证内容 |
|------|------|---------|
| single_sequence | 1序列, 10 tokens, GQA(4:2), dh=8 | 基本正确性 |
| batch_sequences | 3序列 (5,12,20 tokens), layer_idx=1 | 批量 + 多层 |
| non_aligned_seq_len | 7 tokens, block_size=16 | 非对齐边界 |
| larger_dims | 100 tokens, 12 heads, dh=128 | 实际规模 |

所有测试通过参考实现（标准连续 KV attention）对比验证，最大绝对误差为 0。

## 8. 与 vLLM 的对比

| 方面 | vLLM | llaisys (本实现) |
|------|------|-----------------|
| Kernel 来源 | 自研 + FlashInfer | 自研 CPU + CUDA |
| Decode 优化 | 高度优化 (warp-level, vectorized) | 基础版本 (单线程/block) |
| Prefill | FlashAttention (无分页) | 暂用现有 attention |
| GQA | 完整支持 | 完整支持 |
| Block Table 位置 | GPU tensor | CPU→GPU 按需拷贝 |
| FP16/BF16 | 原生支持 | FP32 (可扩展) |

## 9. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/ops/self_attention/paged_attention.hpp` | 新建 | C++ 函数声明 |
| `src/ops/self_attention/paged_attention.cpp` | 新建 | CPU 实现 + 设备分发 |
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cuh` | 新建 | CUDA 头文件 |
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cu` | 新建 | CUDA kernel |
| `include/llaisys/ops.h` | 修改 | 新增 `llaisysPagedAttention` C API |
| `src/llaisys/ops.cc` | 修改 | C API wrapper 实现 |
| `test/test_paged_attention.cpp` | 新建 | 数值正确性测试 (4 个用例) |
| `xmake.lua` | 修改 | 新增测试目标 |

## 10. 下一步

Phase 3 将在模型层接入 Paged Attention：
- 改造 `BatchSlot` 使用 `PageTable` 替代独立 KV-Cache
- 改造 KV 写入路径通过 Page Table 寻址
- 替换 `batchDecode` 中的串行 attention 为单次 `paged_attention` 调用
- 改造 `batchPrefill` 为独立 chunked prefill

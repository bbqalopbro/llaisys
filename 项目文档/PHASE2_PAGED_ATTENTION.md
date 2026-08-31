# Phase 2: Paged Attention Kernel 教学文档

> 更新（2026-08-27）：本文前半保留早期 native decode kernel 的教学背景。当前生产路径已同时接入 FlashInfer paged decode 和 direct paged prefill；chunked prefill 不再依赖连续历史 K/V。

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
- **Block**: 根据 `head_dim` 选择 32~128 个线程，多个线程协作完成一个 head 的 QK 点积和 V 累加
- **Shared Memory**: `block_size * sizeof(float) + 2 * num_warps * sizeof(float)`，前半部分暂存当前 block 内 QK scores，后半部分做跨 warp reduction

### 5.2 Kernel 结构

```cuda
__global__ void paged_attention_kernel(...) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int kv_head_idx = head_idx / group_size;

    float m = -1e30f, l = 0.0f;
    float acc[dims_per_thread] = {0};  // 每个线程负责若干 head_dim 维度

    for (int bi = 0; bi < num_blocks; ++bi) {
        int block_id = block_tables[batch_idx * max_blocks + bi];
        // 从 Block Pool 定位当前 physical block + 当前 layer
        // 多线程协作计算当前 block 内每个 token 的 QK score -> shared memory
        // Online softmax 更新 m, l, acc，并累加 V
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

## 7. FlashInfer Adapter

### 7.1 FlashInfer 路径什么时候启用

`paged_attention.cpp` 是通用 paged attention 分发入口。Decode 的 FlashInfer 路径满足：

```cpp
device_type == LLAISYS_DEVICE_NVIDIA
kv_quant == KVQuantMode::FP32
dtype == LLAISYS_DTYPE_F16
head_dim in {64, 128, 256}
group_size in {1, 2, 4, 8}
flashinfer_available()
```

否则会 fallback 到本项目自己的 CUDA kernel。

注意这里的 `KVQuantMode::FP32` 表示 KV cache 没有走 INT8/INT4 量化路径；`dtype == F16` 表示实际 K/V/Q/O 的 I/O 类型是 FP16。

Prefill 使用独立的 `paged_prefill_prepare/run` 接口，当前支持 NVIDIA、FP16/BF16、`head_dim in {64, 128, 256}` 和任意整数 GQA group size。例如 Qwen2-1.5B 的 `12/2=6` 可以直接使用；上面的 `{1,2,4,8}` 是 decode 模板限制，不适用于 prefill。

### 7.2 Dense block_tables 到 CSR 页表

LLAISYS 自己的页表格式是 dense 二维表：

```cpp
block_tables[b * max_blocks_per_seq + bi]
```

逻辑含义：

```cpp
第 b 个 sequence 的第 bi 个逻辑 block -> 物理 block_id
```

FlashInfer 更喜欢 CSR 风格的分页表：

```cpp
indices       // 所有有效物理 block_id 连续拼接
indptr        // 每个 sequence 在 indices 里的起止位置
last_page_len // 每个 sequence 最后一个 page 的有效 token 数
```

假设：

```cpp
block_size = 16
max_blocks_per_seq = 4
seq_lens = [37, 9, 32]
```

每个 sequence 的物理 block 映射是：

```cpp
seq0: [5, 8, 2]
seq1: [7]
seq2: [4, 9]
```

那么 dense `block_tables` 会被 padding 成：

```cpp
block_tables = [
  [5, 8, 2, 0],
  [7, 0, 0, 0],
  [4, 9, 0, 0]
]
```

实际一维内存是：

```cpp
[5, 8, 2, 0, 7, 0, 0, 0, 4, 9, 0, 0]
```

FlashInfer adapter 会只收集有效 block：

```cpp
indices = [5, 8, 2, 7, 4, 9]
```

再计算每个 sequence 的 block 数：

```cpp
seq0 num_blocks = ceil(37 / 16) = 3
seq1 num_blocks = ceil(9  / 16) = 1
seq2 num_blocks = ceil(32 / 16) = 2
```

于是：

```cpp
indptr[0] = 0
indptr[1] = 0 + 3 = 3
indptr[2] = 3 + 1 = 4
indptr[3] = 4 + 2 = 6

indptr = [0, 3, 4, 6]
```

含义：

```cpp
seq0 的 blocks 在 indices[0:3] = [5, 8, 2]
seq1 的 blocks 在 indices[3:4] = [7]
seq2 的 blocks 在 indices[4:6] = [4, 9]
```

最后计算 `last_page_len`：

```cpp
last_page_len[b] = (seq_len > 0) ? ((seq_len - 1) % block_size + 1) : 0;
```

手算：

```cpp
seq0: 37 = 16 + 16 + 5  -> last_page_len = 5
seq1: 9                 -> last_page_len = 9
seq2: 32 = 16 + 16      -> last_page_len = 16

last_page_len = [5, 9, 16]
```

这里 `seq2=32` 刚好整除 `block_size=16`，最后一页是满的，所以最后一页长度是 16，而不是 0。公式写成 `(seq_len - 1) % block_size + 1` 就是为了处理这个边界。

FlashInfer 内部使用这三张表时，可以理解成：

```cpp
start = indptr[b];
end   = indptr[b + 1];

for p in [start, end):
    physical_block_id = indices[p];
    tokens = (p 是最后一页) ? last_page_len[b] : block_size;
    读取 physical_block_id 里的 tokens 个 KV
```

对应到 naive kernel：

```cpp
// naive dense 表
block_id = block_tables[b * max_blocks_per_seq + bi];

// FlashInfer CSR 表
block_id = indices[indptr[b] + bi];
```

所以 adapter 的核心职责就是：

```cpp
LLAISYS dense page table
        ↓
FlashInfer CSR page table
```

### 7.3 FlashInfer 如何理解 KV Pool

LLAISYS 的 KV pool layout 是：

```cpp
[num_blocks, nlayer, block_size, num_kv_heads, head_dim]
```

FlashInfer 每次 attention 只处理一个 layer，所以 adapter 先把 base pointer 偏移到当前 layer：

```cpp
k_layer = (char*)k_pool + layer_idx * pool_layer_stride;
v_layer = (char*)v_pool + layer_idx * pool_layer_stride;
```

此时 FlashInfer 看到的 layout 可以理解为：

```cpp
[num_blocks, block_size, num_kv_heads, head_dim]
```

然后通过 `kv_strides` 描述寻址方式：

```cpp
kv_strides[0] = pool_block_stride / sizeof(T); // page/block stride，单位是元素
kv_strides[1] = num_kv_heads * HEAD_DIM;       // token stride
kv_strides[2] = HEAD_DIM;                      // kv head stride
kv_strides[3] = 1;                             // dim stride
```

FlashInfer 访问：

```cpp
K[page_id][token_offset][kv_head][dim]
```

等价地址是：

```cpp
k_layer
+ page_id      * pool_block_stride
+ token_offset * num_kv_heads * HEAD_DIM * sizeof(T)
+ kv_head      * HEAD_DIM * sizeof(T)
+ dim          * sizeof(T)
```

这和 naive kernel 的地址公式一致，只是 FlashInfer 把查表和计算封装进库内部。

### 7.4 paged_kv_t 的含义

`flashinfer::paged_kv_t<T, int32_t>` 可以理解为 FlashInfer 眼里的 paged KV cache 描述符，里面包含：

- 当前 layer 的 K/V base pointer
- `num_kv_heads`
- `block_size`
- `HEAD_DIM`
- `kv_strides`
- `indices`
- `indptr`
- `last_page_len`

所以 FlashInfer 没有替代 `BlockAllocator` 或 `PageTable`。它只是接管了 paged attention kernel 的计算部分；本项目仍然负责维护 block pool 和页表。

## 8. 测试结果

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

## 9. 与 vLLM 的对比

| 方面 | vLLM | llaisys (本实现) |
|------|------|-----------------|
| Kernel 来源 | 自研 + FlashInfer | 自研 CPU + CUDA + FlashInfer adapter |
| Decode 优化 | 高度优化 (warp-level, vectorized) | naive CUDA 路径较基础；满足条件时可走 FlashInfer |
| Prefill | FlashAttention/FlashInfer 等后端 | FlashInfer direct paged prefill，gather+GEMM fallback |
| GQA | 完整支持 | 完整支持 |
| Block Table 位置 | GPU tensor | 普通路径 CPU→GPU 按需拷贝；device variant 可直接用 GPU 表 |
| FP16/BF16 | 原生支持 | native kernel 支持 FP32/FP16/BF16；FlashInfer decode 当前 FP16，prefill 支持 FP16/BF16 |

## 10. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/ops/self_attention/paged_attention.hpp` | 新建 | C++ 函数声明 |
| `src/ops/self_attention/paged_attention.cpp` | 新建 | CPU 实现 + 设备分发 |
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cuh` | 新建 | CUDA 头文件 |
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cu` | 新建 | CUDA kernel |
| `src/ops/self_attention/nvidia/flashinfer_adapter.cuh` | 新建 | FlashInfer adapter 声明 |
| `src/ops/self_attention/nvidia/flashinfer_adapter.cu` | 新建 | FlashInfer paged KV prefill/decode 适配 |
| `include/llaisys/ops.h` | 修改 | 新增 `llaisysPagedAttention` C API |
| `src/llaisys/ops.cc` | 修改 | C API wrapper 实现 |
| `test/test_paged_attention.cpp` | 新建 | 数值正确性测试 (4 个用例) |
| `xmake.lua` | 修改 | 新增测试目标 |

## 11. 下一步

Phase 3 将在模型层接入 Paged Attention：
- 改造 `BatchSlot` 使用 `PageTable` 替代独立 KV-Cache
- 改造 KV 写入路径通过 Page Table 寻址
- 替换 `batchDecode` 中的串行 attention 为单次 `paged_attention` 调用
- 改造 `batchPrefill` 为独立 chunked prefill

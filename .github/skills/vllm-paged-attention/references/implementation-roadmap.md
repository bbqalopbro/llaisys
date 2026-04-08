# 实现路线图

## 总览

```
Phase 1: Block Allocator + Page Table         ← 基础设施
Phase 2: Paged Attention Kernel               ← 核心计算
Phase 3: 模型层适配                            ← 前向推理改造
Phase 4: Continuous Batching Scheduler         ← 调度优化
Phase 5: 可选优化                              ← 锦上添花
```

---

## Phase 1: Block Allocator + Page Table

### 目标

替换固定预分配的 KV-Cache，引入块级内存池和页表。

### 1.1 Block Allocator

**输入**: GPU 显存总预算、block_size (默认 16 tokens)、nkvh、dh、nlayer
**输出**: 可分配/回收 block 的内存池

数据结构:
```cpp
class BlockAllocator {
    void* pool_K;              // cudaMalloc 的一大块: [num_blocks × block_size × nkvh × dh × sizeof(float)]
    void* pool_V;              // 同上
    size_t num_blocks;
    size_t block_size;         // tokens per block (e.g. 16)
    size_t block_bytes;        // block_size × nkvh × dh × sizeof(float) × nlayer × 2(K+V)
    std::deque<int> free_list; // 可用 block_id

    int alloc();               // 从 free_list 取一个 block_id，返回 -1 表示满
    void free(int block_id);   // 归还到 free_list
    void* get_k_ptr(int block_id, int layer);  // 定位到具体 block 的 K 内存
    void* get_v_ptr(int block_id, int layer);  // 定位到具体 block 的 V 内存
    size_t num_free() const;   // 可用 block 数
};
```

显存布局选择 (二选一):

**方案 A — 扁平布局** (推荐，连续性好):
```
pool_K: [num_blocks, nlayer, block_size, nkvh, dh]
pool_V: [num_blocks, nlayer, block_size, nkvh, dh]
```

**方案 B — 分层布局** (每层独立 pool):
```
pool_K[layer]: [num_blocks, block_size, nkvh, dh]   ← nlayer 个独立池
pool_V[layer]: [num_blocks, block_size, nkvh, dh]
```

方案 B 更灵活但管理更复杂，方案 A 定位简单: `ptr = pool_K + block_id * nlayer * block_size * nkvh * dh + layer * block_size * nkvh * dh`

### 1.2 Page Table

```cpp
class PageTable {
    std::vector<int> block_ids;     // 有序 block_id 列表
    int num_tokens;                 // 当前已写入的总 token 数

    void append_block(int block_id);
    int get_block_for_token(int token_pos);      // token_pos / block_size → 索引
    int get_offset_in_block(int token_pos);      // token_pos % block_size
    void release_all(BlockAllocator& allocator);  // 归还所有 block
};
```

### 1.3 验证方法

- 单元测试: alloc/free 100 次，验证 free_list 正确
- 显存测试: 分配全部 block → free_list 为空 → 释放一半 → free_list 恢复
- TP 测试: 验证 `local_nkvh` 正确传入

### 文件清单

| 操作 | 文件 |
|------|------|
| 新建 | `src/core/allocator/block_allocator.hpp` |
| 新建 | `src/core/allocator/block_allocator.cpp` |
| 新建 | `src/core/page_table.hpp` |
| 新建 | `test/test_block_allocator.cpp` |

---

## Phase 2: Paged Attention Kernel

### 目标

替换 cuBLAS `SgemmStridedBatched` attention，支持非连续 KV block 间接寻址。

### 路径选择

#### 路径 A: 集成 FlashInfer (推荐)

步骤:
1. 下载 FlashInfer: `pip install flashinfer` 或从源码编译 (`github.com/flashinfer-ai/flashinfer`)
2. FlashInfer 提供 C++ header-only API，可直接 `#include`
3. 在 xmake.lua 添加 FlashInfer 头文件路径
4. 新建 `src/ops/self_attention/nvidia/paged_flash_attention.cu`:
   ```cpp
   #include <flashinfer/attention.cuh>
   // 调用 flashinfer::BatchDecodeWithPagedKVCache(...)
   ```
5. 修改 op 分发层，根据是否传入 page_table 选择 paged 路径

FlashInfer 关键 API:
```cpp
// Prefill (连续 KV, Flash 算法)
flashinfer::SinglePrefillWithKVCache(q, k, v, output, causal=true);

// Decode (分页 KV, Paged Flash 算法)
flashinfer::BatchDecodeWithPagedKVCache(
    q,                    // [batch, n_heads, head_dim]
    paged_kv_cache,       // {data_ptr, indices, indptr, last_page_len}
    output                // [batch, n_heads, head_dim]
);
```

#### 路径 B: 移植 vLLM Kernel

vLLM 的 `csrc/attention/attention_kernels.cu` 提供 `paged_attention_v1/v2`。
需要剥离 PyTorch 依赖 (替换 `at::Tensor` 为裸指针)。

#### 路径 C: 自研

核心算法 (Paged Flash Attention decode):
```cuda
__global__ void paged_attention_kernel(
    float* output,           // [batch, n_heads, head_dim]
    const float* query,      // [batch, n_heads, head_dim]
    const float* k_pool,     // Block pool K
    const float* v_pool,     // Block pool V
    const int* block_tables, // [batch, max_blocks]
    const int* seq_lens,     // [batch]
    int block_size, int head_dim, int num_kv_heads
) {
    // 每个 thread block 处理一个 (batch, head) 对
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    // Online softmax 状态
    float m = -INFINITY, l = 0.0f;
    float acc[HEAD_DIM] = {0};  // 累加器 (register)
    
    // 遍历 page table 中的所有 block
    int num_blocks = (seq_lens[batch_idx] + block_size - 1) / block_size;
    for (int b = 0; b < num_blocks; b++) {
        int block_id = block_tables[batch_idx * max_blocks + b];
        
        // 从 block pool 加载 K, V 到 shared memory
        __shared__ float s_key[BLOCK_SIZE][HEAD_DIM];
        __shared__ float s_val[BLOCK_SIZE][HEAD_DIM];
        // ... load from k_pool[block_id * block_size * head_dim + ...]
        
        // 计算 QK^T
        for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
            float score = 0;
            for (int d = 0; d < head_dim; d++)
                score += query[...] * s_key[t][d];
            score /= sqrtf(head_dim);
            
            // Online softmax update
            float m_new = fmaxf(m, score);
            float p = expf(score - m_new);
            float scale = expf(m - m_new);
            l = scale * l + p;
            for (int d = 0; d < head_dim; d++)
                acc[d] = scale * acc[d] + p * s_val[t][d];
            m = m_new;
        }
    }
    
    // 最终归一化
    for (int d = 0; d < head_dim; d++)
        output[...] = acc[d] / l;
}
```

### 验证方法

- 数值一致性: 对比 cuBLAS 版本的输出 (相对误差 < 1e-3)
- 性能: 比较 latency (应优于 cuBLAS，尤其长序列)
- 边界: 测试 seq_len 不是 block_size 整数倍的情况

### 文件清单

| 操作 | 文件 |
|------|------|
| 新建 | `src/ops/self_attention/nvidia/paged_flash_attention.cu` (或集成 FlashInfer) |
| 修改 | `src/ops/self_attention/op.hpp` — 新增 paged attention 接口声明 |
| 修改 | `src/ops/self_attention/op.cpp` — 增加分发路径 |
| 修改 | `include/llaisys/ops.h` — 导出 C API |

---

## Phase 3: 模型层适配

### 目标

改造 qwen2.cpp 的 BatchSlot 和前向推理，使用 Block Allocator + Paged Attention。

### 3.1 BatchSlot 改造

```cpp
// 之前
struct BatchSlot {
    std::vector<std::vector<tensor_t>> kv_caches; // 完整预分配
    int64_t current_pos;
};

// 之后
struct BatchSlot {
    PageTable page_table;     // block_id 列表
    int64_t current_pos;
    bool active;
};
```

### 3.2 KV 写入路径改造

之前:
```cpp
memcpy(kv_cache_k->data() + pos * pos_bytes, k_data, pos_bytes);
```

之后:
```cpp
int block_id = page_table.get_block_for_token(pos);
int offset = page_table.get_offset_in_block(pos);
if (offset == 0 && block_id < 0) {
    block_id = allocator.alloc();  // 新 block
    page_table.append_block(block_id);
}
void* dst = allocator.get_k_ptr(block_id, layer) + offset * nkvh * dh * sizeof(float);
memcpy(dst, k_data, nkvh * dh * sizeof(float));
```

### 3.3 Batch Decode 改造

之前 (逐 slot 串行 attention):
```cpp
for (size_t i = 0; i < B; ++i) {
    // 单独 attention for slot i
}
```

之后 (单 kernel 并行):
```cpp
// 构建 batch page table 和 seq_lens
paged_attention(batch_Q, k_pool, v_pool, batch_page_tables, seq_lens, batch_output);
// allReduceIfTP(...)
```

### 3.4 Prefill 改造

之前: swap kv 指针 hack
之后: 独立 chunked prefill
```cpp
// 将长 prompt 分成 chunk_size（如 256）的段
for (int chunk_start = 0; chunk_start < prompt_len; chunk_start += chunk_size) {
    int chunk_len = min(chunk_size, prompt_len - chunk_start);
    // 分配新 block (如果需要)
    // Flash attention prefill for this chunk
    // 写入 KV-Cache block
}
```

### 验证方法

- 回归测试: 单序列 (tp=1) 推理输出应与改造前一致
- TP 测试: tp=2 mock comm 不崩溃
- 压力测试: 100 个短请求连续处理，验证 block 正确分配和回收

### 文件清单

| 操作 | 文件 |
|------|------|
| 重构 | `src/llaisys/models/qwen2.cpp` — BatchSlot, batchPrefill, batchDecode |
| 修改 | `include/llaisys/models/qwen2.h` — 新增 API |
| 修改 | `python/llaisys/models/qwen2.py` — Python 层适配 |

---

## Phase 4: Continuous Batching Scheduler

### 目标

将固定 FIFO 调度升级为基于 block 可用量的动态调度。

### 4.1 动态 Admission

```python
# 之前
if len(active_slots) < max_batch_size:
    admit(request)

# 之后
needed_blocks = estimate_blocks(request.prompt_len + request.max_gen)
if block_allocator.num_free() >= needed_blocks:
    admit(request)
```

### 4.2 Preemption

当 free blocks 不足时:
1. 找到最低优先级的活跃序列
2. 保存其 page table 和 block 内容到 CPU (swap)
3. 释放其所有 block
4. 接纳新请求
5. 后续恢复时重新分配 block 并加载

### 4.3 Chunked Prefill 调度

```python
# 之前: prefill 阻塞整个 step
def step():
    while pending_requests:
        admit_and_full_prefill(req)  # 可能很慢
    batch_decode(active_slots)

# 之后: prefill 与 decode 交错
def step():
    # 预算: 每 step 最多处理 budget 个 prefill tokens
    budget = max_prefill_tokens_per_step  # e.g. 512
    for req in pending_with_remaining_prefill:
        chunk = min(budget, req.remaining_prefill)
        chunked_prefill(req, chunk)
        budget -= chunk
        if budget <= 0: break
    batch_decode(active_slots)
```

### 4.4 Per-request 采样参数

```cpp
// 之前
batch_sample(logits, temperature, top_k, top_p);  // 共享参数

// 之后
for (int i = 0; i < B; i++) {
    sample(logits[i], slots[i].temperature, slots[i].top_k, slots[i].top_p);
}
// 或: batch_sample 接受参数数组
```

### 文件清单

| 操作 | 文件 |
|------|------|
| 重构 | `python/server/engine.py` — 调度逻辑 |
| 修改 | `src/llaisys/models/qwen2.cpp` — batch sample 参数化 |
| 修改 | `scripts/tp_worker.py` — TP 协调适配 |

---

## Phase 5: 可选优化

### 5.1 Compute-Comm Overlap

改造 NCCL allReduce 为异步:
```cpp
// 之前
ncclAllReduce(..., _stream);
cudaStreamSynchronize(_stream);

// 之后
ncclAllReduce(..., comm_stream);
cudaEventRecord(comm_done, comm_stream);
// 在下一步需要结果前:
cudaStreamWaitEvent(compute_stream, comm_done);
```

需要修改 `nccl_comm.cu` 和 `qwen2.cpp` 前向推理流程。

### 5.2 FP8/INT8 KV-Cache

将 KV-Cache 从 FP32 压缩为 FP8:
- Block Pool 显存减半
- 写入时量化: FP32 → FP8
- 读取时反量化: FP8 → FP32 (在 attention kernel 中)
- 需要修改 Block Allocator + Attention Kernel

### 5.3 Prefix Caching (Copy-on-Write Block)

复用已有前缀树概念，但在 block 级别:
- 共享前缀的序列指向相同 block (引用计数)
- 当前缀分叉时，拷贝 block (copy-on-write)
- 需要为 block 添加 `ref_count` 字段

### 5.4 扩展 Comm Ops

```cpp
class Comm {
    virtual void allReduceSum(float* data, size_t count) = 0;
    virtual void allGather(float* send, float* recv, size_t count) = 0;  // 新增
    virtual void reduceScatter(float* data, size_t count) = 0;           // 新增
    virtual void barrier() = 0;
};
```

---

## 优先级总结

```
必须做（核心功能）:
  ① Phase 1: Block Allocator + Page Table
  ② Phase 2: Paged Attention Kernel (推荐路径 A: FlashInfer)
  ③ Phase 3: 模型层适配
  ④ Phase 4.1: 动态 Admission

推荐做:
  ⑤ Phase 4.2: Preemption
  ⑥ Phase 4.3: Chunked Prefill
  ⑦ Phase 4.4: Per-request 采样参数

优化项:
  ⑧ Phase 5.1: Compute-Comm Overlap
  ⑨ Phase 5.2: FP8 KV-Cache
  ⑩ Phase 5.3: Prefix Caching (CoW Block)
  ⑪ Phase 5.4: 扩展 Comm Ops
```

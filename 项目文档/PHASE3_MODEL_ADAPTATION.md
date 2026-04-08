# Phase 3: 模型层 Paged KV-Cache 适配 教学文档

## 1. 概述

Phase 3 是整个 PagedAttention 实现中改动最大的一步。我们将 Phase 1 的 BlockAllocator/PageTable 和 Phase 2 的 Paged Attention Kernel 真正接入到 Qwen2 模型的批量推理路径中，替换原来的 per-slot 连续 KV-Cache 和逐序列串行 attention。

### 核心改变

| 方面 | 改造前 | 改造后 |
|------|--------|--------|
| KV-Cache 所有权 | 每个 BatchSlot 独立持有 | 共享 BlockAllocator |
| KV 内存布局 | `[maxseq, nkvh, dh]` 连续 tensor | Block Pool 分页存储 |
| Decode Attention | per-slot 串行 self_attention | 单次 paged_attention 批量调用 |
| 内存效率 | 按 maxseq 预分配（浪费严重） | 按需分配 block（无碎片） |
| Slot 管理 | `kv_caches[nlayer][2]` tensor 数组 | `PageTable` + 共享 block pool |

## 2. BatchSlot 改造

### 2.1 改造前

```cpp
struct BatchSlot {
    std::vector<std::vector<tensor_t>> kv_caches; // [nlayer][2]
    int64_t current_pos = 0;
    bool active = false;

    void init(size_t nlayer, size_t maxseq, size_t nkvh, size_t dh,
              llaisysDeviceType_t dev, int dev_id) {
        for (size_t i = 0; i < nlayer; ++i) {
            // 每个 slot 预分配完整 [maxseq, nkvh, dh] KV tensor
            auto k = Tensor::create({maxseq, nkvh, dh}, ...);
            auto v = Tensor::create({maxseq, nkvh, dh}, ...);
            kv_caches.push_back({k, v});
        }
    }
    void reset() { current_pos = 0; active = false; }
};
```

**问题**：每个 slot 独立预分配 `maxseq * nkvh * dh * sizeof(float) * nlayer * 2` 字节。8 个 slot × maxseq=2048 → 巨大内存浪费。

### 2.2 改造后

```cpp
struct BatchSlot {
    llaisys::core::PageTable page_table;
    int64_t current_pos = 0;
    bool active = false;

    void init(int block_size) {
        page_table = PageTable(block_size);
        current_pos = 0;
        active = false;
    }

    void reset(BlockAllocator &alloc) {
        page_table.release_all(alloc);  // 归还所有 block
        current_pos = 0;
        active = false;
    }
};
```

**优势**：slot 本身不持有内存，只有一个轻量的 `PageTable`（几个 int 的 vector）。实际 KV 数据存储在共享 Block Pool 中，按需分配。

## 3. LlaisysQwen2BatchContext 改造

### 3.1 共享 BlockAllocator

```cpp
struct LlaisysQwen2BatchContext {
    std::unique_ptr<BlockAllocator> block_allocator;
    int block_size;  // tokens per block (e.g. 16)
    ...
};
```

构造时创建统一的 Block Pool：

```cpp
LlaisysQwen2BatchContext(model, max_bs, max_seq) {
    // 计算所需 block 总数
    size_t max_total_tokens = max_bs * max_seq_per_slot;
    size_t num_blocks = ceil(max_total_tokens / block_size) + max_bs;

    BlockAllocatorConfig cfg = {
        num_blocks, block_size, nlayer, local_nkvh, dh, sizeof(float)
    };
    block_allocator = make_unique<BlockAllocator>(cfg, runtime_api);

    // Slot 只需知道 block_size
    for (auto& s : slots) s.init(block_size);
}
```

### 3.2 内存节省分析

假设配置：`max_bs=8, maxseq=2048, nlayer=28, nkvh=2, dh=128`

| 方案 | 公式 | 内存 |
|------|------|------|
| 改造前 | 8 × 2048 × 2 × 128 × 28 × 2 × 4B | **29.4 GB** |
| 改造后 | 只分配实际使用的 block | 按需增长，通常 < 1 GB |

## 4. batch_decode_impl 改造

这是性能提升最关键的改变：将 per-slot 串行 attention 替换为单次 paged_attention 调用。

### 4.1 改造前的 per-slot 循环

```cpp
for (size_t layer = 0; layer < nlayer; ++layer) {
    // ... QKV + RoPE (batch) ...

    // 逐 slot 串行: O(B) 次 attention 调用
    for (size_t i = 0; i < B; ++i) {
        // 拷贝 K/V 到 slot 的连续 KV-Cache
        memcpy(slot.kv_caches[layer][0] + pos * kv_bytes, k_src, kv_bytes);
        memcpy(slot.kv_caches[layer][1] + pos * kv_bytes, v_src, kv_bytes);

        // 提取单个 Q → single_q
        memcpy(single_q, q_src, q_bytes);

        // 单序列 attention
        ops::self_attention(single_attn_out, single_q, k_slice, v_slice, scale);

        // 写回 batch buffer
        memcpy(batch_attn_out + i * attn_bytes, single_attn_out, attn_bytes);
    }
}
```

**问题**：
- 每个 slot 独立调用 `self_attention`，无法批量化
- 需要 `single_q` / `single_attn_out` 中间缓冲区
- 大量小粒度 memcpy

### 4.2 改造后的 paged_attention 批量调用

```cpp
// Step 0: 分配 block (一次性，在 layer 循环前)
for (size_t i = 0; i < B; ++i) {
    if (slot.page_table.needs_new_block()) {
        int bid = alloc.alloc();
        slot.page_table.append_block(bid);
    }
}

// Step 1: 构建批量 block tables 和 seq_lens
int max_blocks_per_seq = max(all slots' page_table.num_blocks());
vector<int> block_tables(B * max_blocks_per_seq);
vector<int> seq_lens(B);
for (size_t i = 0; i < B; ++i) {
    seq_lens[i] = slot.current_pos + 1;
    for (int j = 0; j < slot.page_table.num_blocks(); ++j)
        block_tables[i * max_blocks_per_seq + j] = slot.page_table.block_ids()[j];
}

for (size_t layer = 0; layer < nlayer; ++layer) {
    // ... QKV + RoPE (batch) ...

    // Step 2: 写 KV 到 Block Pool (per slot)
    for (size_t i = 0; i < B; ++i) {
        int bid = slot.page_table.get_block_for_token(pos);
        int off = slot.page_table.get_offset_in_block(pos);
        float* k_dst = alloc.get_k_ptr(bid, layer) + off * nkvh * dh;
        float* v_dst = alloc.get_v_ptr(bid, layer) + off * nkvh * dh;
        memcpy(k_dst, k_src, kv_bytes);
        memcpy(v_dst, v_src, kv_bytes);
    }

    // Step 3: 单次 paged_attention 处理所有序列
    paged_attention(
        batch_attn_out,     // [B, nh, dh]
        batch_q,            // [B, nh, dh]
        alloc.pool_k_raw(), alloc.pool_v_raw(),
        block_tables, seq_lens,
        B, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq,
        alloc.block_stride(), alloc.layer_stride(),
        layer, scale, device_type
    );
}
```

**优势**：
- 单次函数调用处理所有序列（GPU 上可充分并行）
- 无需 `single_q` / `single_attn_out` 中间缓冲区
- KV 写入直接到 Block Pool，无冗余拷贝

## 5. batch_prefill_impl 改造

Prefill 采用 **"先用连续推理，后拷贝到 Block Pool"** 的策略：

```
1. 保存 model 的 KV-Cache 状态
2. 使用 model 的单序列推理路径执行 prefill（复用已有高效实现）
3. 将 model 的连续 KV-Cache 逐 token 拷贝到 Block Pool
4. 恢复 model 状态
```

```cpp
static int64_t batch_prefill_impl(ctx, slot_id, tokens, ntoken, ...) {
    // Use model's single-sequence inference
    model->current_pos = 0;
    output_token = llaisysQwen2ModelInfer(model, tokens, ntoken);

    int64_t final_pos = model->current_pos;

    // Copy KV from model's contiguous cache into block pool
    for (int64_t t = 0; t < final_pos; ++t) {
        if (slot.page_table.needs_new_block()) {
            int bid = alloc.alloc();
            slot.page_table.append_block(bid);
        }
        // Copy from model->kv_caches[layer][kv]->data() + t * kv_row_bytes
        // to alloc.get_k_ptr(bid, layer) + off * nkvh * dh
        ...
        slot.page_table.inc_num_tokens();
    }

    slot.current_pos = final_pos;
    // Restore model state
}
```

**设计权衡**：
- 优点：复用已有单序列推理路径（已测试、已优化），无需重写 prefill 逻辑
- 缺点：额外一次 KV 拷贝（从连续 → 分页）
- 权衡：prefill 是计算密集型（O(n²) attention），拷贝开销相比 GEMM 可忽略

## 6. KV-Cache 快照 (Save/Restore) 改造

### 6.1 Save：分页 → 连续

```cpp
// 遍历 page table 中的每个 token，从 block pool 拷贝到连续 buffer
for (int64_t t = 0; t < pos; ++t) {
    int bid = page_table.get_block_for_token(t);
    int off = page_table.get_offset_in_block(t);
    for (size_t layer = 0; layer < nlayer; ++layer) {
        // K/V 从 block pool 拷贝到 snapshot.buffers[layer*2+kv]
        memcpyD2H(snap_k + t * row_bytes, alloc.get_k_ptr(bid, layer) + off * ..., row_bytes);
        memcpyD2H(snap_v + t * row_bytes, alloc.get_v_ptr(bid, layer) + off * ..., row_bytes);
    }
}
```

### 6.2 Restore：连续 → 分页

```cpp
// 释放旧 blocks，重新按 token 分配并写入
slot.page_table.release_all(alloc);
for (int64_t t = 0; t < snapshot->pos; ++t) {
    if (page_table.needs_new_block()) {
        int bid = alloc.alloc();
        page_table.append_block(bid);
    }
    // 从 snapshot.buffers 拷贝回 block pool
    memcpyH2D(alloc.get_k_ptr(bid, layer) + off * ..., snap_k + t * row_bytes, row_bytes);
    ...
    page_table.inc_num_tokens();
}
```

## 7. C API 新增

```c
// 查询 block allocator 状态（用于调度决策）
size_t llaisysQwen2BatchGetFreeBlocks(ctx);   // 可用 block 数
size_t llaisysQwen2BatchGetTotalBlocks(ctx);  // 总 block 数
int    llaisysQwen2BatchGetBlockSize(ctx);    // 每 block 的 token 数
```

Python 层新增对应方法：

```python
class BatchContext:
    def get_free_blocks(self) -> int: ...
    def get_total_blocks(self) -> int: ...
    def get_block_size(self) -> int: ...
    def get_block_usage(self) -> dict:
        """返回 {total, free, used, utilization}"""
```

## 8. 数据流对比

### 改造前

```
token_ids → Embedding → [Layer × nlayer]:
  → Norm → QKV → RoPE
  → for each slot:           ← 串行瓶颈
      → memcpy K/V to slot KV tensor
      → self_attention(single_q, k_slice, v_slice)
      → memcpy attn_out back
  → O proj → MLP
→ LM Head → Sample
```

### 改造后

```
token_ids → Embedding → [Layer × nlayer]:
  → Norm → QKV → RoPE
  → for each slot: write K/V to block pool  ← 小粒度写入
  → paged_attention(all slots at once)      ← 单次批量调用
  → O proj → MLP
→ LM Head → Sample
```

## 9. 测试结果

```
=== Phase 3: Paged Batch Decode Tests ===
  test_batch_decode_simulation PASSED
  test_slot_save_restore PASSED
  test_stress_many_sequences PASSED (32 seqs, up to 50 steps)

✓ All Phase 3 tests passed!
```

| 测试 | 验证内容 |
|------|---------|
| batch_decode_simulation | 3 序列 × 10 步 × 2 层，完整 decode 流程 |
| slot_save_restore | 分页数据的 save → release → restore 往返一致性 |
| stress_many_sequences | 32 序列 × 不同长度 (5~50 步)，block 分配回收正确性 |

## 10. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/llaisys/models/qwen2.cpp` | 修改 | BatchSlot/BatchContext/decode/prefill/save/restore 全面改造 |
| `include/llaisys/models/qwen2.h` | 修改 | 新增 3 个 block allocator 查询 API |
| `src/core/allocator/block_allocator.hpp` | 修改 | 新增 `block_stride()` accessor |
| `python/llaisys/libllaisys/qwen2.py` | 修改 | 新增 ctypes 绑定 |
| `python/llaisys/models/qwen2.py` | 修改 | BatchContext 新增 block 查询方法 |
| `test/test_paged_batch.cpp` | 新建 | Phase 3 集成测试 |
| `xmake.lua` | 修改 | 新增测试目标 |

## 11. 与 vLLM 的对比

| 方面 | vLLM | llaisys (本实现) |
|------|------|-----------------|
| Prefill | FlashAttention (无分页) | 复用单序列推理 + KV 拷贝 |
| Decode | PagedAttention v2 (高度优化) | 自研 paged_attention (功能正确) |
| Block 分配时机 | Scheduler 统一管理 | decode 开始前按需分配 |
| Block 释放 | Scheduler + eviction | slot reset 时释放 |
| Preemption | 支持 swap/recompute | Phase 4 实现 |
| 数据格式 | FP16/BF16 原生 | FP32 (可扩展) |

## 12. 下一步 (Phase 4)

Phase 4 将在调度层面进一步优化：
1. **动态 Admission**: 根据 `get_free_blocks()` 判断是否接受新请求
2. **Preemption**: 当 block 耗尽时，evict 优先级最低的序列
3. **Chunked Prefill**: 将 prefill 分块与 decode 交替执行
4. **Per-request 采样参数**: 每个请求独立的 temperature/top_k/top_p

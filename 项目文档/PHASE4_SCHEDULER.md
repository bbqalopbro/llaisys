# Phase 4: 调度器升级 教学文档

## 1. 概述

Phase 4 在 Phase 1-3 建立的 PagedAttention 基础设施之上，升级推理引擎的调度策略。主要实现三个核心功能：

| 功能 | 说明 | 解决的问题 |
|------|------|-----------|
| 动态 Admission | 根据 block 可用量决定是否接受新请求 | 防止 OOM / block 耗尽 |
| Preemption | block 不足时驱逐低优先级请求 | 避免死锁，提高系统弹性 |
| Per-request 采样 | 每个请求独立 temperature/top_k/top_p | 支持多用户不同采样需求 |

## 2. 动态 Admission Control

### 2.1 问题

改造前的调度器只检查 slot 是否空闲：

```python
while free_slots and queue.waiting_count > 0:
    slot_id = free_slots.pop(0)
    req = queue.get_pending(1)[0]
    batch_ctx.prefill(slot_id, req.input_ids, ...)
```

这会导致：如果多个长序列同时 prefill，block pool 可能被耗尽，后续 decode 无法分配新 block → 所有请求卡死。

### 2.2 解决方案：Block Budget Check

每次 admit 新请求前，估算所需 block 数，检查是否有足够余量：

```python
def _estimate_blocks_needed(self, prompt_len, max_tokens, block_size):
    total_tokens = prompt_len + max_tokens
    return ceil(total_tokens / block_size)

def _can_admit(self, batch_ctx, prompt_len, max_tokens):
    needed = self._estimate_blocks_needed(prompt_len, max_tokens, block_size)
    free = batch_ctx.get_free_blocks()
    total = batch_ctx.get_total_blocks()
    watermark = int(total * self.block_watermark)  # 保留 10% 安全余量
    return free >= needed + watermark
```

### 2.3 Watermark 机制

Block watermark（默认 10%）预留一部分 block 不用于 admission，确保：
- 已运行请求的 decode 有足够 block 继续生成
- 突发的长序列生成不会立即耗尽所有 block

```
Total blocks: 1000
Watermark: 10% = 100 blocks reserved
Available for admission: 900 blocks

已用 700 blocks → free = 300 → 可用 200 (300 - 100 watermark) → OK
已用 850 blocks → free = 150 → 可用  50 (150 - 100 watermark) → 新请求需 80 blocks → REJECT
```

### 2.4 与 vLLM 的对比

| 方面 | vLLM | llaisys |
|------|------|---------|
| 准入检查 | Scheduler 中基于 block 计数 | `_can_admit()` + watermark |
| 粒度 | 精确到每个 sequence 的 block 需求 | 估算 (prompt + max_tokens) / block_size |
| 策略 | 配置化 (等待/抢占/丢弃) | watermark + preemption fallback |

## 3. Preemption (请求驱逐)

### 3.1 触发条件

当 `_can_admit()` 返回 False 时，调度器尝试 preempt（驱逐）正在运行的请求来释放 block：

```python
if not self._can_admit(batch_ctx, prompt_len, max_tokens):
    if not self._try_preempt(batch_ctx, running, free_slots, needed):
        # 驱逐也无法满足 → 暂缓 admit
        break
```

### 3.2 驱逐策略

采用 **最长生成序列优先驱逐**（Longest Output First）：

```python
def _try_preempt(self, batch_ctx, running, free_slots, needed_blocks):
    # 选择已生成最多 token 的请求作为 victim
    victim_slot = max(running.keys(),
                      key=lambda s: len(running[s].generated_tokens))
    victim_req = running[victim_slot]
    
    # 1. 保存 KV-Cache 快照 (到 CPU 内存)
    snapshot = batch_ctx.slot_save(victim_slot)
    victim_req.kv_cache_snapshot = snapshot
    
    # 2. 释放 slot (归还所有 block)
    batch_ctx.slot_reset(victim_slot)
    
    # 3. 将请求放回等待队列
    victim_req.status = RequestStatus.WAITING
    self.queue.submit(victim_req)
```

### 3.3 恢复机制

被驱逐的请求保留了 KV-Cache 快照。重新 admit 时直接恢复：

```python
if req.kv_cache_snapshot is not None:
    batch_ctx.slot_restore(slot_id, req.kv_cache_snapshot)
    req.kv_cache_snapshot = None
    req.status = RequestStatus.DECODING
    running[slot_id] = req
    # 无需重新 prefill!
```

### 3.4 设计权衡

| 策略 | 优点 | 缺点 |
|------|------|------|
| Swap to CPU (本实现) | 恢复快，无需重新计算 | 额外 CPU 内存，D2H/H2D 延迟 |
| Recompute | 不需要额外内存 | 恢复慢（重跑 prefill + decode） |
| Drop | 最简单 | 用户体验差 |

vLLM 支持 swap 和 recompute 两种策略。我们实现了 swap 策略，因为它在延迟和资源间有最好的平衡。

### 3.5 Preemption 流程图

```
新请求到达 → can_admit? ──YES──→ Prefill + 加入 running
                │
               NO
                │
                ▼
           try_preempt?
                │
    ┌───────────┴───────────┐
   YES                     NO
    │                       │
    ▼                       ▼
 Save victim snapshot    暂缓 admit
 Reset victim slot       (等下一轮)
 Re-queue victim
    │
    ▼
 Retry can_admit ──YES──→ Prefill 新请求
                   NO──→ 暂缓 admit
```

## 4. Per-request 采样参数

### 4.1 改造前

批量 decode 使用统一的采样参数（来自第一个请求）：

```python
# 所有请求被迫使用相同的 temperature/top_k/top_p
first_req = running[active_slots[0]]
batch_temp = first_req.params.temperature
batch_top_k = first_req.params.top_k
batch_top_p = first_req.params.top_p

next_tokens = batch_ctx.decode(
    active_slots, current_tokens,
    temperature=batch_temp, top_k=batch_top_k, top_p=batch_top_p
)
```

### 4.2 改造后

新增 C++ API `llaisysQwen2BatchDecodePerRequest`，接受采样参数数组：

```c
void llaisysQwen2BatchDecodePerRequest(
    ctx, active_slots, num_active, current_tokens,
    float *temperatures,  // [num_active]
    int   *top_ks,        // [num_active]
    float *top_ps,        // [num_active]
    output_tokens
);
```

Python 端调用：

```python
temperatures = [running[s].params.temperature for s in active_slots]
top_ks = [running[s].params.top_k for s in active_slots]
top_ps = [running[s].params.top_p for s in active_slots]

next_tokens = batch_ctx.decode_per_request(
    active_slots=active_slots,
    current_tokens=current_tokens,
    temperatures=temperatures,
    top_ks=top_ks,
    top_ps=top_ps,
)
```

### 4.3 C++ 实现

Kernel 层面，sampling 仍然是 per-slot 串行的（因为采样本身很轻量），但每个 slot 使用自己的参数：

```cpp
for (size_t i = 0; i < B; ++i) {
    float temp = temperatures[i];
    int topk = top_ks[i];
    float topp = top_ps[i];
    bool greedy = (topk == 1) || (temp <= 0.0f);

    if (greedy) {
        ops::argmax(next_token, logits);
    } else {
        ops::sample(next_token, logits, temp, topk, topp, seed++);
    }
}
```

## 5. 调度循环对比

### 改造前

```
while running:
    1. 有空 slot → admit（无 block 检查）
    2. batch_decode（统一采样参数）
    3. 完成 → 释放
```

### 改造后

```
while running:
    1. 有空 slot + block 预算充足 → admit
       block 不足 → try preempt → 重试 admit
    2. batch_decode_per_request（独立采样参数）
    3. 完成 → 释放
    4. 被驱逐的请求在 re-admit 时直接 restore snapshot
```

## 6. 测试结果

```
test_can_admit_check ... ok
test_estimate_blocks ... ok
test_different_params ... ok
test_preempt_longest ... ok
test_max_size ... ok
test_submit_and_get ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.000s
OK
```

| 测试 | 验证内容 |
|------|---------|
| test_can_admit_check | Block 充足/不足时的准入判断 |
| test_estimate_blocks | Block 需求估算正确性 |
| test_preempt_longest | 驱逐最长序列、正确回收 slot |
| test_different_params | Per-request 采样参数传递 |
| test_submit_and_get | 请求队列基本功能 |
| test_max_size | 队列容量限制 |

## 7. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/llaisys/models/qwen2.cpp` | 修改 | 新增 `batch_decode_per_request_impl` |
| `include/llaisys/models/qwen2.h` | 修改 | 新增 `llaisysQwen2BatchDecodePerRequest` |
| `python/llaisys/libllaisys/qwen2.py` | 修改 | 新增 ctypes 绑定 |
| `python/llaisys/models/qwen2.py` | 修改 | 新增 `decode_per_request`, `get_block_usage` |
| `python/server/engine.py` | 修改 | 动态 admission + preemption + per-request 采样 |
| `test/test_scheduler.py` | 新建 | 调度器逻辑单元测试 |

## 8. 与 vLLM Scheduler 的对比

| 方面 | vLLM | llaisys (本实现) |
|------|------|-----------------|
| Admission 策略 | Block 计数 + 配置策略 | Block 预算 + watermark |
| Preemption 触发 | 调度器主动检测 | Admission 失败时触发 |
| Preemption 策略 | FCFS / Priority | 最长输出优先驱逐 |
| Preemption 恢复 | Swap to CPU / Recompute | Swap to CPU (KV snapshot) |
| 采样参数 | Per-request | Per-request |
| Chunked Prefill | 原生支持 | 通过 KV 拷贝实现 |
| Sequence Groups | 支持 beam search 组 | 不支持 |

## 9. 整体架构回顾

经过四个 Phase 的实现，llaisys 的推理引擎已具备 vLLM 的核心特性：

```
┌─────────────────────────────────────────────────┐
│ Phase 4: InferenceEngine (engine.py)            │
│  - Dynamic Admission (block budget check)       │
│  - Preemption (swap KV to CPU)                  │
│  - Per-request Sampling                         │
├─────────────────────────────────────────────────┤
│ Phase 3: Model Adaptation (qwen2.cpp)           │
│  - BatchSlot → PageTable                        │
│  - batch_decode → paged_attention              │
│  - batch_prefill → contiguous → block pool     │
├─────────────────────────────────────────────────┤
│ Phase 2: Paged Attention Kernel                 │
│  - CPU: online softmax + block traversal        │
│  - CUDA: per-(batch, head) thread block         │
│  - C API: llaisysPagedAttention                 │
├─────────────────────────────────────────────────┤
│ Phase 1: Block Allocator + Page Table           │
│  - BlockAllocator: pool-based alloc/free        │
│  - PageTable: logical → physical mapping        │
│  - Flat layout: [num_blocks, nlayer, BS, nkvh, dh] │
└─────────────────────────────────────────────────┘
```

每一层构建在前一层之上，形成完整的 PagedAttention 推理系统。

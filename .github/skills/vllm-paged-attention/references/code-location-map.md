# 代码位置映射

每项改动对应的具体文件、行号范围、改动类型。

---

## 1. 需要新建的文件

| 文件路径 | 用途 |
|---------|------|
| `src/core/allocator/block_allocator.hpp` | Block Allocator 类定义 |
| `src/core/allocator/block_allocator.cpp` | Block Allocator GPU 实现 |
| `src/core/page_table.hpp` | Page Table 数据结构 |
| `src/ops/self_attention/nvidia/paged_flash_attention.cu` | Paged Attention CUDA Kernel (或 FlashInfer 适配) |
| `test/test_block_allocator.cpp` | Block Allocator 单元测试 |
| `test/test_paged_attention.cpp` | Paged Attention 数值正确性测试 |

---

## 2. 需要修改的 C++ 文件

### `src/llaisys/models/qwen2.cpp` — 改动最大的文件

| 行号范围 (参考) | 内容 | 改动 |
|----------------|------|------|
| ~L31-L43 | `LlaisysQwen2CacheSnapshot` 结构 | 适配 page table 快照 |
| ~L57-L69 | `TrieNode` 前缀树结构 | 适配 block 级别前缀共享 |
| ~L110-L120 | 模型成员变量 `kv_caches`, `current_pos` | 替换为 BlockAllocator + PageTable 成员 |
| ~L148-L153 | `allReduceIfTP()` | 无需改动 (allReduce 逻辑不变) |
| ~L231-L238 | `init_cache()` | 替换为 BlockAllocator 初始化 |
| ~L240-L264 | `init_buffers()` | 缓冲区维度可能需调整 |
| ~L337-L397 | `classifyWeight()` + `slice2D/1D` | 无需改动 (权重分片不变) |
| ~L478-L553 | 单序列前向推理 (KV 写入) | KV 写入改为通过 page table 寻址 |
| ~L510-L515 | KV-Cache 写入点 | 替换为 block 定位 + 写入 |
| ~L516 | `allReduceIfTP(hidden, hs)` — 注意力后 | 不变 |
| ~L530 | `allReduceIfTP(hidden, hs)` — MLP 后 | 不变 |
| ~L800-L840 | `SaveCache / RestoreCache` | 适配 page table 保存/恢复 |
| ~L895-L916 | 前缀树查找 | 适配 block 级别前缀匹配 |
| ~L936-L955 | `BatchSlot` 结构 | 移除独立 KV-Cache，改为 PageTable |
| ~L957-L1030 | `BatchContext` 初始化 | 使用共享 BlockAllocator |
| ~L1050-L1070 | `batchPrefill` | 独立 chunked prefill |
| ~L1075-L1216 | `batchDecode` | 替换串行 attention 为 paged attention kernel |
| ~L1171  | batch attention 后 allReduce | 适配新 attention 输出 |
| ~L1205 | batch MLP 后 allReduce | 不变 |

### `include/llaisys/models/qwen2.h`

| 改动 | 内容 |
|------|------|
| 新增 | `llaisysQwen2CreateWithPaged(...)` — 支持 paged 配置的模型创建 |
| 新增 | Block/Page 相关的 C API 导出 |

### `include/llaisys/ops.h`

| 改动 | 内容 |
|------|------|
| 新增 | `llaisysPagedAttention(...)` — paged attention C API 声明 |

### `src/ops/self_attention/op.hpp` / `op.cpp`

| 改动 | 内容 |
|------|------|
| 新增 | `paged_attention()` 函数声明和分发逻辑 |
| 新增 | 接受 `block_pool_K/V`, `page_tables`, `seq_lens` 参数 |

### `src/distributed/nccl_comm.cu` (Phase 5 可选)

| 行号 | 改动 |
|------|------|
| ~L141-L155 | `allReduceSum` 移除 `cudaStreamSynchronize`，改为 event-based |
| 构造函数 | 新增 `comm_stream` (独立于 compute stream) |

### `src/distributed/comm.hpp` (Phase 5 可选)

| 改动 | 内容 |
|------|------|
| 新增 | `allGather()`, `reduceScatter()` 虚函数 |
| 新增 | `allReduceSumAsync()` + `waitComm()` 异步接口 |

### `src/core/allocator/naive_allocator.cpp`

| 改动 | 内容 |
|------|------|
| 无需修改 | Block Allocator 是独立的新类，NaiveAllocator 继续用于非 KV-Cache 的分配 |

---

## 3. 需要修改的 Python 文件

### `python/server/engine.py`

| 改动 | 内容 |
|------|------|
| admit 逻辑 | 从固定 slot 数检查 → 基于 free block 数检查 |
| decode 循环 | 适配 chunked prefill 交错 |
| 完成处理 | 释放 block 回 free list |
| 新增 | preemption 策略 (swap/recompute) |

### `python/llaisys/models/qwen2.py`

| 改动 | 内容 |
|------|------|
| 模型构造 | 传入 block_allocator 配置参数 |
| BatchContext | 适配新的 C API |

### `python/llaisys/libllaisys/` (ctypes 绑定)

| 改动 | 内容 |
|------|------|
| 新增 | paged attention 和 block allocator 的 ctypes 绑定 |

### `scripts/tp_worker.py`

| 改动 | 内容 |
|------|------|
| TPBatchContext | 适配新的 prefill/decode 接口 |
| 协调命令 | 可能需新增 block 管理相关命令 |

---

## 4. 构建系统

### `xmake.lua`

| 改动 | 内容 |
|------|------|
| 新增源文件 | `block_allocator.cpp`, `paged_flash_attention.cu` |
| 新增依赖 | FlashInfer 头文件路径 (如果走路径 A) |
| 新增编译选项 | `ENABLE_PAGED_ATTENTION` 宏 (可选，做渐进式迁移) |

---

## 5. 测试文件

### 已有测试 (需适配)

| 文件 | 改动 |
|------|------|
| `test/tp_fwd_smoke.cpp` | 添加 paged 模式 forward 测试 |
| `test/tp_cache_smoke.cpp` | 适配 page table 快照测试 |
| `test/tp_shard_smoke.cpp` | 无需改动 (权重分片不变) |
| `test/test_distributed.py` | 无需改动 (comm 接口不变) |
| `test/test_nccl_real.py` | 无需改动 |
| `test/test_tp_e2e.py` | 添加 paged 模式端到端测试 |

### 新增测试

| 文件 | 内容 |
|------|------|
| `test/test_block_allocator.cpp` | alloc/free/fragmentation/TP 测试 |
| `test/test_paged_attention.cpp` | 数值正确性 (对比 cuBLAS 版本) |
| `test/test_paged_batch.py` | Python 层 paged batch 端到端测试 |

---

## 6. 依赖关系图

```
Phase 1: Block Allocator + Page Table
    │
    ├──→ Phase 2: Paged Attention Kernel (依赖: block pool 指针格式)
    │        │
    │        └──→ Phase 3: 模型层适配 (依赖: 新 attention API)
    │                 │
    │                 └──→ Phase 4: Scheduler (依赖: block 分配/释放 API)
    │
    └──→ Phase 5.1: Compute-Comm Overlap (独立于 Phase 2-4)

Phase 5.2: FP8 KV-Cache (依赖 Phase 1 + 2)
Phase 5.3: Prefix CoW (依赖 Phase 1 + 3)
Phase 5.4: Comm Ops 扩展 (独立)
```

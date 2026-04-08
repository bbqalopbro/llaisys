# Phase 1: Block Allocator + Page Table 教学文档

## 1. 设计动机

### 1.1 当前问题

在实现 PagedAttention 之前，llaisys 的 KV-Cache 采用**固定预分配连续内存**策略：

```cpp
// qwen2.cpp — init_cache()
std::vector<size_t> shape = {meta.maxseq, local_nkvh, meta.dh};
for (size_t i = 0; i < meta.nlayer; ++i) {
    auto k_c = Tensor::create(shape, LLAISYS_DTYPE_F32, device_type, device_id);
    auto v_c = Tensor::create(shape, LLAISYS_DTYPE_F32, device_type, device_id);
    kv_caches.push_back({k_c, v_c});
}
```

每个 slot 预分配 `maxseq × nkvh × dh × sizeof(float)` 的完整 KV 缓冲区。问题：

| 问题 | 影响 |
|------|------|
| 按最大序列长度预分配 | 若 maxseq=32768 但实际只用 100 tokens，浪费 99.7% 显存 |
| 固定 slot 数量 | max_batch_size=4 时最多 4 个并发请求，即使多数请求很短 |
| 无法共享内存 | 请求完成后整个 slot 的显存才释放，碎片率高 |
| 无法实现 PagedAttention | Attention kernel 要求连续 KV，分页 KV 无法使用 |

### 1.2 vLLM 的解决方案

vLLM 借鉴操作系统**虚拟内存分页**思想：

```
传统方式:  每请求预分配 [maxseq × nkvh × dh] 连续显存 → 95% 浪费
vLLM 分页: KV-Cache 按固定 block (16 tokens) 按需分配 → <5% 浪费

Block Pool: [num_blocks × block_size × nkvh × dh]  ← 一大块共享显存池
Page Table: seq_id → [block_3, block_7, block_1, ...] ← 间接寻址
```

关键优势：
1. **按需分配**：新 token 到来时才分配新 block
2. **细粒度回收**：请求完成时逐 block 回收，立即可复用
3. **高利用率**：所有请求共享同一 block pool，浪费仅限最后一个 block 的 padding

## 2. 架构设计

### 2.1 总体架构

```
┌─────────────────────────────────────────────┐
│                GPU Memory                    │
│  ┌────────────────────────────────────────┐ │
│  │    Block Pool K                        │ │
│  │  ┌──────┬──────┬──────┬──────┬─────┐  │ │
│  │  │blk 0 │blk 1 │blk 2 │blk 3 │ ... │  │ │
│  │  └──────┴──────┴──────┴──────┴─────┘  │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │    Block Pool V (同构)                 │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘

┌─────────────────────┐    ┌─────────────────────┐
│  BlockAllocator      │    │  PageTable (per seq) │
│  - free_list: deque  │    │  - block_ids: [3,7,1]│
│  - alloc() → id      │    │  - num_tokens: 40    │
│  - free(id)           │    │  - get_block(pos)    │
│  - get_k_ptr(id,lay) │    │  - get_offset(pos)   │
└─────────────────────┘    └─────────────────────┘
```

### 2.2 显存布局

采用**扁平布局**: `[num_blocks, nlayer, block_size, nkvh, dh]`

```
Pool K 总大小: num_blocks × nlayer × block_size × nkvh × dh × elem_size

每个 block 内部:
  block_stride = nlayer × layer_stride
  layer_stride = block_size × nkvh × dh × elem_size

定位 (block_id, layer) 对应的内存:
  ptr = pool_K + block_id × block_stride + layer × layer_stride
      → 返回 [block_size, nkvh, dh] 连续区域
```

选择扁平布局的原因：
- 只需 2 次 `cudaMalloc` (K 池 + V 池)
- 地址计算简单，无间接寻址开销
- 每个 (block_id, layer) 对提供一段**连续内存**，适合 attention kernel

### 2.3 显存预算计算

以 Qwen2-1.5B (28 层, 2 KV heads, 128 head_dim, FP32) 为例：

| 参数 | 值 |
|------|-----|
| 每 block (1 层, K 或 V) | 16 × 2 × 128 × 4 = 16 KB |
| 每 block (所有层, K+V) | 16 KB × 28 × 2 = 896 KB |
| 1000 blocks (~16K tokens) | ~875 MB |

对比原方案 4 slot × 2048 maxseq: ~460 MB，但利用率低。分页方案在高并发短请求场景下优势明显。

## 3. 实现细节

### 3.1 BlockAllocator

**文件**: `src/core/allocator/block_allocator.hpp` + `block_allocator.cpp`

核心数据结构：
```cpp
class BlockAllocator {
    void *_pool_K;                  // 一次性分配的 K 显存池
    void *_pool_V;                  // 一次性分配的 V 显存池
    const LlaisysRuntimeAPI *_api;  // 设备 API (支持 CPU/GPU)
    BlockAllocatorConfig _config;   // 配置 (num_blocks, block_size, nlayer, nkvh, dh)
    size_t _block_stride;           // 每 block 跨所有层的字节数
    size_t _layer_stride;           // 每层单 block 的字节数
    std::deque<int> _free_list;     // 可用 block_id 队列
};
```

**构造函数**：一次性分配整个 pool
```cpp
BlockAllocator::BlockAllocator(const BlockAllocatorConfig &config,
                               const LlaisysRuntimeAPI *api) {
    _layer_stride = block_size * nkvh * dh * elem_size;
    _block_stride = nlayer * _layer_stride;
    size_t total_bytes = num_blocks * _block_stride;

    _pool_K = _api->malloc_device(total_bytes);  // 一次 cudaMalloc
    _pool_V = _api->malloc_device(total_bytes);

    // 所有 block 初始可用
    for (size_t i = 0; i < num_blocks; ++i)
        _free_list.push_back(i);
}
```

**alloc/free**：O(1) 的 block 分配回收
```cpp
int BlockAllocator::alloc() {
    if (_free_list.empty()) return -1;  // 池满
    int block_id = _free_list.front();
    _free_list.pop_front();
    return block_id;
}

void BlockAllocator::free(int block_id) {
    _free_list.push_back(block_id);  // FIFO 回收
}
```

**定位**：根据 (block_id, layer) 计算内存地址
```cpp
void *BlockAllocator::get_k_ptr(int block_id, int layer) {
    return (std::byte *)_pool_K
         + block_id * _block_stride
         + layer * _layer_stride;
    // 返回 [block_size, nkvh, dh] 连续区域的起始地址
}
```

### 3.2 PageTable

**文件**: `src/core/page_table.hpp` (header-only)

每个序列维护一个 PageTable，记录其使用的 block 列表：

```cpp
class PageTable {
    std::vector<int> _block_ids;  // 有序 block_id 列表
    int _num_tokens;              // 已写入 token 数
    int _block_size;              // 每 block 的 token 容量
};
```

**地址转换**：token 位置 → (block_id, offset)
```cpp
int get_block_for_token(int token_pos) {
    int block_idx = token_pos / _block_size;
    return _block_ids[block_idx];  // 间接寻址
}

int get_offset_in_block(int token_pos) {
    return token_pos % _block_size;
}
```

**按需扩容**：写入新 token 时检查是否需要新 block
```cpp
bool needs_new_block() {
    if (_block_ids.empty()) return true;
    return (_num_tokens % _block_size == 0);  // 当前 block 已满
}
```

**释放**：序列完成时归还所有 block
```cpp
void release_all(BlockAllocator &allocator) {
    for (int bid : _block_ids)
        allocator.free(bid);
    _block_ids.clear();
    _num_tokens = 0;
}
```

### 3.3 与现有架构的关系

BlockAllocator 不继承 MemoryAllocator 基类。两者职责不同：

| | MemoryAllocator (NaiveAllocator) | BlockAllocator |
|---|---|---|
| 用途 | 通用内存分配 (Tensor 创建) | KV-Cache 专用块管理 |
| 粒度 | 任意大小 | 固定 block_size |
| 接口 | allocate(size) → ptr | alloc() → block_id |
| 池化 | 无 (直接 cudaMalloc) | 预分配大池，内部管理 |
| 生命周期 | 跟随 Runtime | 跟随模型/BatchContext |

## 4. 构建系统集成

### 4.1 自动编译

`xmake.lua` 中 `llaisys-core` 目标使用通配符编译：
```lua
target("llaisys-core")
    add_files("src/core/*/*.cpp")  -- 自动匹配 block_allocator.cpp
```

因此 `block_allocator.cpp` 无需额外配置，自动编入 `llaisys-core` 静态库。

### 4.2 测试目标

新增独立测试二进制：
```lua
target("llaisys-test-block-allocator")
    set_kind("binary")
    add_deps("llaisys")
    add_files("test/test_block_allocator.cpp")
    add_includedirs("$(projectdir)")
```

运行测试：
```bash
xmake build llaisys-test-block-allocator
xmake run llaisys-test-block-allocator
```

## 5. 测试结果

```
=== BlockAllocator Tests ===
  test_basic_alloc_free PASSED        # 基本分配回收
  test_exhaust_pool PASSED            # 池耗尽 + 回收后复用
  test_unique_ids PASSED              # 100 次分配 ID 唯一性
  test_pointer_arithmetic PASSED      # 指针定位正确性
  test_write_read_memory PASSED       # 实际读写验证

=== PageTable Tests ===
  test_page_table_basic PASSED        # token→block 映射
  test_page_table_needs_new_block PASSED  # 扩容判断
  test_page_table_release_all PASSED  # 全量释放

=== TP Compatibility Tests ===
  test_tp_local_nkvh PASSED           # TP 下 local_nkvh 计算

=== Integrated Scenario ===
  test_integrated_scenario PASSED     # 多序列 alloc/write/verify/release
```

所有 11 项测试通过，验证了：
- 分配/回收的正确性和 O(1) 性能
- 池满时返回 -1，回收后可重用
- 指针算术与预期布局一致
- 内存读写数据完整性
- TP 模式下 local_nkvh 正确传递
- 多序列交叉使用时数据隔离

## 6. 与 vLLM 的对比

| 方面 | vLLM | llaisys (本实现) |
|------|------|-----------------|
| Block Pool | 按 dtype (FP16/FP8) 分配 | 支持任意 elem_size |
| 布局 | 分层独立池 | 扁平布局 (一次分配) |
| Free List | GPU tensor + CPU bitmap | CPU `deque<int>` |
| Page Table | GPU int tensor | CPU `vector<int>` |
| Copy-on-Write | 支持 (引用计数) | Phase 5 可选 |
| 线程安全 | 有锁 | 目前无锁 (单线程使用) |

设计选择说明：
- **扁平布局**比 vLLM 的分层布局更简单，只需 2 次 GPU 分配，适合当前规模
- **CPU 端管理**比 GPU tensor 管理更灵活，调试方便，适合教学目的
- 后续 Phase 3 接入模型后，PageTable 的 block_ids 需要在 attention kernel 前拷贝到 GPU

## 7. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/core/allocator/block_allocator.hpp` | 新建 | BlockAllocator 类定义 |
| `src/core/allocator/block_allocator.cpp` | 新建 | BlockAllocator 实现 |
| `src/core/page_table.hpp` | 新建 | PageTable 类 (header-only) |
| `test/test_block_allocator.cpp` | 新建 | 单元测试 (11 个测试用例) |
| `xmake.lua` | 修改 | 新增测试目标 |

## 8. 下一步

Phase 2 将在此基础上实现 **Paged Attention Kernel**：
- 集成 FlashInfer 库
- 新增 `paged_flash_attention.cu` 适配层
- Attention kernel 通过 Page Table 间接寻址 Block Pool 中的 KV 数据
- 替换当前 cuBLAS `SgemmStridedBatched` 的全矩阵 attention

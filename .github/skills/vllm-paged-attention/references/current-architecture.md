# 当前架构分析

## 1. KV-Cache 实现

### 1.1 单序列 KV-Cache

**文件**: `src/llaisys/models/qwen2.cpp`

分配方式 (`init_cache()`):
```cpp
// 形状: [maxseq, local_nkvh, dh] — 每层一对 K/V tensor
std::vector<size_t> shape = {meta.maxseq, local_nkvh, meta.dh};
for (size_t i = 0; i < meta.nlayer; ++i) {
    auto k_c = Tensor::create(shape, LLAISYS_DTYPE_F32, device_type, device_id);
    auto v_c = Tensor::create(shape, LLAISYS_DTYPE_F32, device_type, device_id);
    kv_caches.push_back({k_c, v_c});
}
```

关键特征:
- **预分配固定大小连续内存** — 无动态增长/缩减
- **FP32 固定** — 所有 KV-Cache 强制 FP32
- **TP 感知** — 使用 `local_nkvh = meta.nkvh / tp_size`
- **cache_pos**: 单一整数 `current_pos` 追踪写入位置
- **写入**: `data + current_pos * bytes` 偏移直接写
- **读取**: `slice(0, 0, current_pos + 1)` 截取有效范围

### 1.2 Cache 快照

```cpp
struct LlaisysQwen2CacheSnapshot {
    int64_t pos;
    size_t nlayer, pos_bytes;
    int tp_size, tp_rank;
    std::vector<std::vector<uint8_t>> buffers;  // CPU 侧完整拷贝
};
```

- 保存: 逐层 GPU → CPU memcpy
- 恢复: CPU → GPU memcpy + TP 兼容性校验
- 截断: 只修改 `current_pos`，不清理数据

### 1.3 前缀树 KV-Cache 池

```cpp
struct TrieNode {
    std::unordered_map<int64_t, std::unique_ptr<TrieNode>> children;
    LlaisysQwen2CacheSnapshot* snapshot = nullptr;
};
```

以 token 序列为键，最长前缀匹配挂载完整 CacheSnapshot。

---

## 2. 批处理实现

### 2.1 BatchSlot

```cpp
struct BatchSlot {
    std::vector<std::vector<tensor_t>> kv_caches; // [nlayer][2]
    int64_t current_pos = 0;
    bool active = false;
};
```

- 每 slot 拥有**独立完整的 KV-Cache**
- 默认 `max_seq_per_slot = 2048`
- 总共预分配 `max_batch_size` 个 slot (默认 4)

### 2.2 Prefill

通过 swap KV-Cache 指针复用单序列推理代码（hack 方式）:
```cpp
auto saved_caches = model->kv_caches;
auto saved_pos = model->current_pos;
model->kv_caches = slot.kv_caches;
model->current_pos = slot.current_pos;
// ... 调用单序列 ModelInfer ...
model->kv_caches = saved_caches;
model->current_pos = saved_pos;
```

### 2.3 Batch Decode

- **Linear/RMSNorm/SwiGLU**: 真正批量 `[B, dim]`
- **Attention**: 逐 slot 串行循环
- **采样**: 逐 slot 串行，共享第一个请求的参数

---

## 3. Attention 实现

### 3.1 CPU Attention (`src/ops/self_attention/op.cpp`)

标准三重循环 QK^T → softmax → V，支持 GQA，因果 mask。

### 3.2 CUDA Attention (`src/ops/self_attention/nvidia/self_attention_nvidia.cu`)

- cuBLAS `SgemmStridedBatched` 实现所有 head 并行
- 流程: Gather Q/KV → Batched GEMM (QK^T) → Scale+Mask → Softmax → Batched GEMM (P·V) → Scatter
- **完整 O(n²) attention 矩阵被具体化** — 非 FlashAttention
- **全局 grow-only 缓冲区** — static cudaMalloc 只增不减
- **单序列接口** — 无 batch 维度

---

## 4. Runtime 层

### 4.1 设备抽象

C 函数指针 vtable (`LlaisysRuntimeAPI`):
- 12 个函数指针: `malloc_device`, `free_device`, `memcpy_sync`, `memcpy_async`, `create_stream`, `stream_synchronize` 等
- 三套实现: CPU (`std::malloc`), NVIDIA (`cudaMalloc`), MetaX (MXMACA)

### 4.2 Context (线程局部)

```cpp
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}
```

每线程独立设备上下文，包含 Runtime + stream + allocator。

### 4.3 内存分配

```
Tensor::create() → Runtime::allocateDeviceStorage() → NaiveAllocator::allocate() → cudaMalloc()
```

**无内存池** — 每次直接系统调用。

### 4.4 Storage (RAII + 引用计数)

`shared_ptr<Storage>` 管理，view/slice/permute 共享同一 storage。

---

## 5. 通信层

### 5.1 双层架构

**第一层 (TCP)**: Python tp_worker.py 中 rank 0 广播命令 → followers 执行同一操作
**第二层 (NCCL/MPI)**: C++ op 内部自动集合通信

### 5.2 NCCL 后端

- 专用 CUDA stream
- ncclUniqueId 通过文件交换
- `allReduceSum` + 立即 `cudaStreamSynchronize` ← **同步，无重叠**
- `barrier` 用单元素 allReduce 模拟

### 5.3 MPI 后端

GPU 数据需 D2H → MPI_Allreduce → H2D 中转。

### 5.4 Comm 接口

```cpp
class Comm {
    virtual void allReduceSum(float* data, size_t count) = 0;
    virtual void barrier() = 0;
};
```

仅支持 `allReduceSum` 和 `barrier`。

---

## 6. Ops 层

已实现 12 个算子: add, argmax, embedding, linear, rearrange, rms_norm, rope, self_attention, swiglu, sample, dequantize, dequantize_int4。

分发模式: 检查 tensor 设备 → setDevice → 调用对应命名空间实现。

GPU ops 异步 launch 但推理本身同步等待 (`runtime.synchronize()`)。

---

## 7. 模型前向推理中的 TP

Megatron-LM 风格 Column/Row Parallel，每层 2 次 AllReduce:

```
hidden [1, hs] (所有 rank 相同)
  ├─ Q/K/V (Column Parallel → local heads)
  ├─ Self-Attention (local heads)
  ├─ O_proj (Row Parallel) → partial sum
  ├─ ★ AllReduce Sum ← 第1次
  ├─ + residual
  ├─ gate+up (Column Parallel → local_di)
  ├─ SwiGLU
  ├─ down_proj (Row Parallel) → partial sum
  ├─ ★ AllReduce Sum ← 第2次
  └─ + residual
```

### 权重分片策略

| 策略 | 权重 | 切分方式 |
|------|------|---------|
| ColDim0 | q/k/v_proj, gate/up_proj | 按 dim 0 (output features) 切分 |
| RowDim1 | o_proj, down_proj | 按 dim 1 (input features) 切分 |
| None | embed_tokens, norms, lm_head | 不切分 |

AWQ 特殊处理: qweight layout `[in_features, out_packed]` 与标准转置，ColDim0 ↔ RowDim1 交换。

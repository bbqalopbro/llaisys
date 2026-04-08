# 面试深挖问答记录：核心引擎与性能优化

> 简历片段：核心引擎与性能优化：实现支持零拷贝（View/Slice）的 Tensor 抽象及 DeepSeek-R1 全套算子；利用 Nsight Systems 深度 Profiling，通过 cuBLAS 句柄全局复用（消除 81K 次隐式同步）、自建显存缓冲池（分配开销直降 99%）、重写 Batched Attention 融合算子与全链路异步化，将 GPU 单次推理耗时从 118s 优化至 2.73s（43倍加速）。

---

## 一、零拷贝 Tensor 抽象

### Q1: Tensor 类的核心设计——存储与元数据如何分离？`_offset` 字段的作用？

**参考答案**：

Tensor 类采用 **"存储 + 元数据 + 偏移量"三元组** 设计：

```cpp
class Tensor {
    TensorMeta _meta;           // 元数据：dtype, shape, strides
    core::storage_t _storage;   // 存储：shared_ptr<Storage>，指向真实的显存/内存块
    size_t _offset;             // 字节偏移量：当前 tensor 数据在 storage 中的起始位置
};
```

| 组件 | 职责 | 为什么分离 |
|------|------|-----------|
| `_meta` | 描述"形状是什么"（shape/strides/dtype） | view 操作仅需修改 meta，不触碰数据 |
| `_storage` | 持有"数据在哪里"（一块连续显存的 `shared_ptr`） | 多个 tensor 可共享同一块显存，引用计数管理生命周期 |
| `_offset` | 描述"数据从哪里开始读"（字节偏移） | slice 操作仅需调整 offset，不做任何数据拷贝 |

**`_offset` 的作用**：使得 `data()` 方法返回 `_storage->data() + _offset`，即 slice 后的 tensor 指向原始 storage 中的某个中间位置。这就是零拷贝 slice 的核心——不搬数据，只移指针。

**为什么三个字段缺一不可**：
- **没有 `_offset`**：slice 必须 `cudaMemcpy` 拷贝数据到新显存 → 性能大幅下降
- **没有 `shared_ptr`**：原始 tensor 析构后 slice 变成悬空指针 → 程序崩溃（use-after-free）
- **没有 `_meta`**：无法知道数据的形状和解读方式

---

### Q2: view() 和 slice() 的实现细节及前置约束？

**参考答案**：

**view()**: 仅修改 shape 和 strides，不触碰底层数据
- 前置约束：张量必须是 **连续的（contiguous）**，否则抛异常
- 原因：view 基于"元素在内存中是连续存放"的假设来重新计算 strides。非连续张量的内存布局和新 shape 对不上

**slice()**: 修改目标维度的 shape，并计算新的 `_offset`
- `added_offset = start × strides[dim] × elementSize()`（字节）
- 新 tensor 的 `_offset = 旧_offset + added_offset`
- 不要求张量连续（stride 信息已足够定位数据）

**isContiguous() 判断逻辑**：从最后一维开始检查 `strides[i] == expected_stride`，若 shape[i]≠1 但 stride 不连续，则返回 false。

---

### Q3: 为什么推理中需要频繁 view/slice？不做零拷贝会带来什么问题？

**参考答案**：

**场景 1：view 用于 reshape**
- Linear 输出是 `[seq_len, hidden_size]`（如 `[1, 1536]`），Self-Attention 需要 `[seq_len, n_head, head_dim]`（如 `[1, 12, 128]`）
- 没有零拷贝 view → 每个 token × 28 层 × 3 次（Q/K/V）= 84 次无意义的 memcpy

**场景 2：slice 用于 KV Cache 截取**
- KV Cache 预分配为 `[max_seq_len, n_kv_head, head_dim]`
- 推理到第 pos 个 token，需要 `cache[0:pos+1, :, :]`
- 没有零拷贝 slice → 每个 token 都要 `cudaMemcpy` 拷贝不断增长的 KV 子矩阵，累积开销随序列长度线性增长

**核心洞察**：不做零拷贝，推理中光 view/slice 就会产生数万次 cudaMemcpy，而优化前 118s 的瓶颈恰好就是这类不必要的 CPU-GPU 同步。

---

### Q4: shared_ptr 引用计数在什么场景下至关重要？

**参考答案**：

核心场景：原始 tensor 被析构，但它的 slice 还在被使用。

- A 和 B 共享同一个 `shared_ptr<Storage>`
- A 析构时引用计数从 2 → 1，显存 **不释放**
- B 析构时引用计数从 1 → 0，此时才 `cudaFree`

没有 shared_ptr，A 析构直接 cudaFree，B 变成悬空指针 → GPU 段错误。

---

### Q5: 非连续张量的拷贝（copy_strided_cpu 递归函数）

非连续张量的数据在内存中不连续（有"跳格"），不能直接 `memcpy`（会读到错误的数据）。递归函数按 stride 逐维度跳跃访问源数据，连续写入目标缓冲区。GPU 端非连续拷贝可以用 CUDA kernel 实现。

---

## 二、cuBLAS 句柄全局复用

### Q1: 怎么发现 cuBLAS handle 是性能瓶颈的？

**参考答案**：

使用 **Nsight Systems (nsys)** profiling：

```bash
nsys profile --stats=true --trace=cuda python3 test/test_infer.py --device nvidia
```

通过 `cuda_api_sum` 报告发现：

| CUDA API | 耗时 | 占比 | 调用次数 |
|---|---|---|---|
| `cudaDeviceSynchronize` | **71.9s** | **59%** | **81,000 次** |
| `cudaFree` | 27.0s | 22% | 94,011 次 |
| `cudaMalloc` | 13.4s | 11% | 73,812 次 |

95% 时间在 Host 端 CUDA Runtime 调用，GPU 计算不是瓶颈。

81K 次 `cudaDeviceSynchronize` 不是显式调用的 → 追踪到 `cublasCreate()` 内部隐式触发。每次 `linear` op 都 create + destroy handle：240 次/token × 88 token ≈ 21K 次 create，量级与 81K 匹配。

---

### Q2: cublasCreate() 为什么导致隐式同步？

**参考答案**：

`cublasCreate()` 内部需要：
1. 查询 GPU 设备属性
2. 分配内部工作空间
3. 初始化 kernel 调度表

这些操作需要 GPU 处于空闲状态，因此 CUDA Runtime 在内部插入 `cudaDeviceSynchronize()`——等待所有已入队的 GPU 操作全部完成，才开始初始化。

**本质**：`cudaDeviceSynchronize` 是全局屏障（global barrier），打断 CPU-GPU 异步流水线，形成"流水线气泡"。

---

### Q3: 为什么用 `static thread_local`？

**参考答案**：

```cpp
static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    return handle;
}
```

| 修饰符 | 作用 | 缺少的后果 |
|--------|------|-----------|
| `static` | 只初始化一次，函数多次调用间保持值 | 每次调用都重新创建 handle |
| `thread_local` | 每个线程独立一份副本 | 多线程共享同一 handle → 数据竞争 |

组合效果 = **线程安全的单例模式**：每个线程第一次调用时 create，之后永远复用。

这是 TensorRT、vLLM、llama.cpp 等生产级推理框架的标准做法。

---

### Q4: handle 永不销毁，有资源泄漏问题吗？

推理服务通常在进程退出时统一释放所有 GPU 资源。cuBLAS handle 占用极小（KB 级），进程结束时 OS 会回收所有资源，因此实践中不构成问题。

在以下场景需要显式销毁：
- GPU 热迁移（rare）
- 单进程内多次创建/销毁推理引擎实例（如测试框架）

---

## 三、显存缓冲池（Grow-Only Buffer）

### Q1: 优化前 self_attention 的显存分配模式

**参考答案**：

优化前，`self_attention` 每次调用分配 **5 个临时 GPU 缓冲区**：

| 缓冲区 | 用途 | 大小示例（pos=100） |
|--------|------|-------------------|
| `s_q_all` | gather 后的 Q | 14KB |
| `s_k_all` | expand 后的 K | 1.4MB |
| `s_v_all` | expand 后的 V | 1.4MB |
| `s_scores` | 注意力分数矩阵 | 11KB |
| `s_o_all` | 输出缓冲区 | 14KB |

每次 op 完毕后 `cudaFree`，下次再 `cudaMalloc`。

28 层 × 88 token × 5 = **12,320 次 malloc + 12,320 次 free**。

nsys 数据：`cudaMalloc` 73,812 次 = 13.4s，`cudaFree` 94,011 次 = 27.0s，合计 **40.4s（33%）**。

`cudaMalloc/cudaFree` 慢的原因：
- 涉及 GPU 设备端内存管理器的锁和查找
- 是同步的 Host API，CPU 等待完成才能继续

---

### Q2: Grow-Only 策略是什么？为什么适合自回归解码？

**参考答案**：

```cpp
static void ensure_buf(float *&ptr, size_t &cur, size_t need) {
    if (need <= cur) return;        // 够用 → 零开销复用
    if (ptr) cudaFree(ptr);         // 不够 → 释放旧的
    cudaMalloc(&ptr, need);         // 分配更大的
    cur = need;
}
```

**三个字概括：只增不缩。**

自回归解码中 `total_len` 单调递增 → 缓冲区大小只会增长 → Grow-Only 完美匹配。

| 维度 | 传统 Memory Pool | Grow-Only |
|------|-----------------|-----------|
| 复杂度 | 空闲链表、碎片合并 | 5 个静态指针 + 15 行代码 |
| 适用场景 | 大小多变的通用场景 | 大小单调递增的特殊场景 |
| 碎片 | 需处理 | 无碎片 |

结果：`cudaMalloc` 73,812 → 740 次，`cudaFree` 94,011 → 680 次，减少 **99%+**。

---

### Q3: Grow-Only 在什么场景下失效？

- Prefill 阶段处理长 prompt（如 2048 tokens），分配大缓冲区
- Decode 阶段 `seq_len=1`，但缓冲区容量已被 prefill 撑大，造成显存浪费
- 多请求并发时，不同请求的 `total_len` 差异大，static 缓冲区被最大的请求撑大

改进方向：加入 shrink 策略（空闲时缩小）或使用 per-request 缓冲区。

---

## 四、Batched Attention 融合算子

### Q1: 优化前逐 head 循环的具体问题？

**参考答案**：

优化前用 `for (h = 0; h < n_head; h++)` 逐 head 串行计算。每个 head 8 次 kernel launch：
gather → GEMM(Q×K^T) → scale → mask → softmax → GEMM(P×V) → scatter。

12 heads × 8 = 96 次/call，28 层 × 88 token × 96 ≈ **~236K 次 kernel launch**。

问题不在于单个 kernel 慢，而在于：
- 每次 `cudaLaunchKernel` 约 5-10μs 开销，370K 次 = 5.4s
- 每个 head 的 GEMM 矩阵太小（`[1,128]×[128,101]`），GPU SM 大量空闲

核心矛盾：小矩阵 + 高频 launch = GPU 利用率极低。

---

### Q2: 怎么变成批量计算的？用了什么 API？

**参考答案**：

核心：把所有 head 数据 gather 成 `[n_head, seq_len, dim]` 连续布局 →用 `cublasSgemmStridedBatched` 一次完成所有 head 的矩阵乘法。

7 步流程：
```
Step 1: gather Q → [12, 1, 128]           1 kernel
Step 2: gather+expand K,V → [12, pos+1, 128]  2 kernels
Step 3: Batched GEMM: scores = Q×K^T       1 cublasSgemmStridedBatched
Step 4: Scale + Causal Mask                2 kernels
Step 5: Softmax                            1 kernel
Step 6: Batched GEMM: output = P×V         1 cublasSgemmStridedBatched
Step 7: scatter → [1, 12, 128]             1 kernel
合计: 8 次（vs 原来 96 次）
```

`cublasSgemmStridedBatched` 把 12 个独立小 GEMM 打包成 1 次调用：
- 减少 launch 次数（12→1）
- 让 GPU 所有 SM 同时计算不同 head

---

### Q3: GQA 中 KV heads 少于 Q heads 怎么处理？

**参考答案**：

Q 12 个 head，KV 只有 2 个 head，`group_size = 12/2 = 6`。

在 Step 2（gather_expand_kv_kernel）中完成扩展：
```cpp
int64_t kv_h = h / group_size;  // h=0~5→kv_h=0, h=6~11→kv_h=1
```

把 K/V 从 `[pos+1, 2, 128]` 扩展为 `[12, pos+1, 128]`。

扩展后 batch 维度统一为 12，后续 batched GEMM 无需特殊处理。

**权衡**：用额外显存（2→12 份 KV）换取计算的统一性。对 decode 阶段（seq_len=1）额外拷贝极小，收益远大于代价。

---

### Q4: Softmax 数值稳定性？

**参考答案**：

不能直接 `exp(score)`：若 score=100，`exp(100)≈2.69×10^43` → float32 溢出 → NaN。

解决方案：先减最大值 $\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$

CUDA kernel 三步：
1. Tree reduction 求行最大值
2. `exp(x - max)` 并求和（指数输入 ≤ 0，结果 ≤ 1，不溢出）
3. 归一化

这是所有深度学习框架的标准做法。

---

## 五、异步流水线优化

### Q1: 优化前每个 token 有几个同步点？为什么同步点多导致性能差？

**参考答案**：

优化前每个 token 有 **4 个同步点**：

| 同步点 | 操作 | 方向 |
|--------|------|------|
| ① | memcpy_sync(token_id) | H2D |
| ② | memcpy_sync(position_id) | H2D |
| ③ | memcpy_sync(KV cache update) | D2D（×28层×2=56次） |
| ④ | memcpy_sync(argmax result) | D2H |

每个同步点 = 一个"流水线气泡"：CPU 阻塞等 GPU，GPU 完成后等 CPU 提交新工作。4 个气泡/token × 88 token = 352 个气泡。

CUDA 执行模型：CPU 应不停给 GPU 提交工作，`memcpy_sync` 打断了这个流水线。

---

### Q2: memcpy_async 为什么不阻塞 CPU？

**参考答案**：

CUDA Stream 是一个任务队列。CPU 往队列提交任务后**立即返回**，GPU 按顺序执行。

| 特性 | cudaMemcpy (sync) | cudaMemcpyAsync (async) |
|------|-------------------|------------------------|
| CPU | 阻塞等待完成 | 立即返回 |
| 顺序 | — | 同一 stream 内保证顺序 |

因为 H2D 和后续 kernel 在同一 stream（default stream），CUDA 保证先完成 memcpy 再执行 kernel，正确性不受影响。

---

### Q3: 保留的唯一同步点在哪里？

**参考答案**：

**D2H 的 argmax 结果读回**。

CPU 必须拿到 token_id 才能：1) 判断 EOS 终止条件；2) 作为下一步输入。

自回归解码的因果链：第 t+1 步输入依赖第 t 步输出 → 每个 token 至少 1 个不可避免的同步点。

优化目标：从 4 个同步点减到 1 个。

---

### Q4: std::swap 替代 D2D memcpy（Ping-Pong Buffer）

**参考答案**：

```
交换前：
  residual      → [显存块 A]
  hidden_states → [显存块 B]（当前数据）

std::swap 后：
  residual      → [显存块 B]（作为残差保存）
  hidden_states → [显存块 A]（空闲，供后续计算写入）
```

两块显存交替扮演"输入"和"输出"角色，O(1) 指针交换，零数据搬运。

效果：消除 28层×2次×88token = 4,928 次 D2D memcpy。

## 六、性能归因分析

### 完整优化链路 118s → 2.73s

| 优化项 | 加速贡献 | 核心指标变化 |
|---|---|---|
| cuBLAS Handle 复用 | ~1.7× | `cudaDeviceSynchronize` 81K→0 次 |
| Grow-Only 缓冲区 | ~1.2× | `cudaMalloc+cudaFree` 168K→1.4K 次 |
| Batched Self-Attention | ~1.03× | `cudaLaunchKernel` 370K→164K 次 |
| Residual 指针交换 | ~1.01× | D2D memcpy 消除 4,928 次 |
| 异步 memcpy | 与 6 合并 | 每 token 同步点 4→1 |
| GPU 显存池释放 | **~21×** | 消除 VRAM 争用换页 |

| 阶段 | 总时间 | 吞吐量 | 加速比 |
|---|---|---|---|
| 优化前 | 118.0s | 0.75 tok/s | 1× |
| 优化 1+2 | 59.0s | 1.49 tok/s | 2.0× |
| 优化 3+4 | 57.4s | 1.53 tok/s | 2.1× |
| 优化 5+6（最终） | **2.73s** | **27.0 tok/s** | **43.2×** |
| HuggingFace 参考 | 2.89s | 26.2 tok/s | — |

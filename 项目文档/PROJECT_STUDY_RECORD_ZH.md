# LLAISYS 学习记录（逐条完整稿）

> 记录目标：把此前对话中的技术讲解与后续学习步骤完整落盘。
>
> 重要说明：按你的要求，本文明确排除“简历相关内容”。
>
> 日期：2026-03-13

---

## 一、原答复全文记录（技术部分）

### 1）项目全景学习路线图（原答复）

# LLAISYS 项目学习路线图

## 项目定位

LLAISYS（**L**et's **L**earn **AI** **SYS**tem）是一个 **从零构建 LLM 推理引擎** 的教学项目，使用 C++ 实现系统后端（编译为 `libllaisys.so`），Python 通过 `ctypes` 调用 C API，最终实现 DeepSeek-R1-Distill-Qwen 模型的完整推理链路。

## 整体架构（三层设计）

```
┌─────────────────────────────────────────────────────────────────┐
│  Python 层（前端）                                               │
│  test/*.py, python/llaisys/*.py, python/server/*.py              │
│  通过 ctypes.CDLL 加载 libllaisys.so                             │
└────────────────────────┬────────────────────────────────────────┘
                         │ ctypes FFI 调用
┌────────────────────────▼────────────────────────────────────────┐
│  C 接口层（桥接）                                                │
│  include/llaisys.h → src/llaisys/*.cc                            │
│  __export 标记的纯 C 函数，桥接到 C++ 实现                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ C++ namespace 调用
┌────────────────────────▼────────────────────────────────────────┐
│  C++ 实现层（后端）                                              │
│  src/tensor/  — 张量数据结构                                     │
│  src/ops/     — 算子（CPU / NVIDIA / MetaX 多后端）              │
│  src/core/    — Context/Runtime/Allocator 设备抽象               │
│  src/device/  — 各平台 Runtime API 实现                          │
│  src/llaisys/models/ — Qwen2 模型推理逻辑                       │
│  src/distributed/ — NCCL/MPI 分布式通信                          │
└─────────────────────────────────────────────────────────────────┘
```

## 七大学习阶段

| 阶段 | 主题 | 核心文件 | 关键知识点 |
|:----:|------|---------|-----------|
| **1** | CPU 推理 | `src/tensor/`, `src/ops/*/cpu/`, `src/llaisys/models/qwen2.cpp` | Tensor 类、7 个算子的 CPU 实现、KV-Cache、Qwen2 前向传播 |
| **2** | CPU 优化 | `src/ops/linear/cpu/` | SIMD (AVX2/512)、OpenMP 多线程、OpenBLAS/MKL 加速 |
| **3** | GPU 算子 + GPU 优化 | `src/device/nvidia/`, `src/ops/*/nvidia/*.cu`, [README_PERF.md](README_PERF.md) | CUDA Runtime API、cuBLAS、8 个 kernel、6 项性能优化 (118s→2.73s) |
| **4** | 采样 + 前后端 + UI | `src/ops/sample/`, `python/server/`, `python/server/static/` | Temperature/TopK/TopP 采样、FastAPI SSE 流式、Web ChatUI、会话管理+前缀树 KV 池 |
| **5** | 多用户推理服务 | `python/server/engine.py`, [PROJECT4_PLAN.md](PROJECT4_PLAN.md) | 请求队列、异步 Worker、Continuous Batching、Per-request KV-Cache |
| **6** | 分布式推理 | `src/distributed/`, `include/llaisys/distributed.h`, [PROJECT5_PLAN.md](PROJECT5_PLAN.md) | 张量并行 (TP)、NCCL all-reduce、MPI、权重分片、Column/Row Parallel |
| **7** | 32B 模型推理 | 量化 + TP | INT8/INT4 量化压缩、GPTQ/AWQ 兼容、多卡 TP 分片、显存优化 |

### 每阶段详细展开

#### 阶段 1：CPU 推理（基础闭环）
> **目标**：理解推理引擎的最小完整链路 — 从文本输入到文本输出

1. **Tensor 数据结构** — `storage + offset + meta(shape/strides/dtype)`
2. **7 个核心算子** — embedding, linear($Y=XW^T+b$), rms_norm, rope, self_attention, swiglu, argmax
3. **Qwen2 模型前向** — 28 层 Transformer Block 的完整数据流
4. **KV-Cache** — 预分配、追加写入、slice 读取
5. **Python↔C++ 交互** — ctypes + opaque handle + memmove

#### 阶段 2：CPU 推理优化
> **目标**：理解为什么"朴素实现"很慢，以及如何优化

1. **SIMD 向量化** — AVX2/AVX-512 intrinsics 加速逐元素运算
2. **OpenMP 并行** — `#pragma omp parallel for` 多线程加速 linear
3. **第三方 BLAS 库** — OpenBLAS/MKL 替代手写矩阵乘

#### 阶段 3：GPU 算子和 GPU 优化
> **目标**：理解 CUDA 编程模型 + 118s→2.73s 的 43 倍加速是怎么做到的

1. **Runtime 抽象** — Context/Runtime/RuntimeAPI 的设备无关设计
2. **CUDA Runtime API** — malloc/free/memcpy/stream
3. **8 个 CUDA kernel** — 含 cuBLAS GEMM 调用
4. **6 项核心优化**：cuBLAS Handle 复用、Grow-Only 缓冲区、Batched Attention、指针交换残差、异步 memcpy、显存池管理

#### 阶段 4：随机采样 + 前后端 + UI
> **目标**：从"测试脚本"到"真正能聊天的产品"

1. **Sample 算子** — Temperature 缩放、TopK 截断、TopP nucleus 采样
2. **FastAPI 服务** — SSE 流式响应、OpenAI 兼容 API
3. **Web UI** — 前端 Chat 界面
4. **会话管理** — 多会话切换、前缀树 KV-Cache 池复用

#### 阶段 5：多用户推理服务
> **目标**：从单用户阻塞到多用户并发

1. **请求队列 + 异步 Worker** — InferenceEngine 架构
2. **Continuous Batching** — 迭代级批调度
3. **C++ batch 推理 API** — cublasSgemmStridedBatched 批量矩阵乘
4. **Per-request KV-Cache** — 每请求独立缓存

#### 阶段 6：分布式推理
> **目标**：单卡装不下的模型怎么跑

1. **TP 核心概念** — Column Parallel / Row Parallel + all-reduce
2. **通信抽象** — `DistComm` → NCCL / MPI 后端
3. **权重分片加载** — 按 rank 切分 Q/K/V/MLP 矩阵
4. **KV-Cache TP 化** — 本地 head 视图

#### 阶段 7：32B 模型推理
> **目标**：大模型的工程挑战 — 显存不够怎么办

1. **INT8 量化** — per-channel symmetric, ~2x 压缩
2. **INT4 量化** — per-group symmetric (g=128), ~3.76x 压缩
3. **GPTQ/AWQ 兼容** — 加载时 Python 侧格式转换
4. **量化 + TP 组合** — 多卡分片 + 权重压缩

---

### 2）阶段 1.1：Tensor 讲解（原答复）

# 阶段 1.1：Tensor — 推理引擎的数据基石

## 核心概念

Tensor（张量）是多维数组的抽象，是所有 AI 框架（PyTorch、TensorFlow）的核心数据结构。LLAISYS 的 Tensor 由三部分组成：

```
Tensor = Storage(内存块) + Offset(偏移) + Meta(元数据)
```

## Tensor 详解

### 1. 三要素模型

```
┌──────────────────────────────────────────────────────────────┐
│  Storage（共享内存块）  ← std::shared_ptr 引用计数管理生命周期  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ byte[0]  byte[1]  ...  byte[N]                          ││  
│  └──────────────────────────────────────────────────────────┘│
│        ↑ offset                                              │
│        └── Tensor A 从这里开始                                │
│                                                              │
│  TensorMeta:                                                 │
│    dtype   = F32                                             │
│    shape   = [2, 3]                                          │
│    strides = [3, 1]   ← 行优先，每走一行跳3个元素              │
└──────────────────────────────────────────────────────────────┘
```

关键源码在 [src/tensor/tensor.hpp](src/tensor/tensor.hpp)：

```cpp
struct TensorMeta {
    llaisysDataType_t dtype;          // 数据类型（F32/F16/BF16/I64...）
    std::vector<size_t> shape;        // 形状，如 [2, 3, 5]
    std::vector<ptrdiff_t> strides;   // 步长，如 [15, 5, 1]
};

class Tensor {
    TensorMeta _meta;
    core::storage_t _storage;   // shared_ptr<Storage> → 实际内存
    size_t _offset;             // 在 Storage 中的字节偏移
};
```

### 2. `create` — 张量创建

在 [src/tensor/tensor.cpp](src/tensor/tensor.cpp) 中：

```cpp
tensor_t Tensor::create(shape, dtype, device_type, device) {
    // 1. 从最后一维开始计算 strides（行优先/C-order）
    //    shape=[2,3] → strides=[3,1], 总元素数=6
    
    // 2. 通过 Context → Runtime 分配内存
    //    CPU:  allocateDeviceStorage (实际就是 malloc)
    //    GPU:  allocateDeviceStorage (实际是 cudaMalloc)
    
    // 3. 返回 shared_ptr<Tensor>
}
```

**关键设计**：Tensor 不关心底层是 CPU 还是 GPU 内存，它只持有一个 `Storage` 指针。设备的差异由 `Runtime` 层屏蔽。

### 3. 零拷贝元操作

`view`、`permute`、`slice` 这三个操作 **不搬移任何数据**，只修改元数据：

| 操作 | 做了什么 | 数据搬移？ |
|------|---------|:---------:|
| `view([2,15])` | 改 shape+strides，共享同一 Storage | ❌ |
| `permute([1,0,2])` | 重排 shape+strides 顺序（实现转置） | ❌ |
| `slice(0, 0, pos+1)` | 缩小 shape + 加 offset 偏移 | ❌ |
| `contiguous()` | 若非连续，拷贝数据到新的连续内存 | ✅ |

例如 `slice` 的实现：
```cpp
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;         // 缩小该维度
    size_t added_offset = start * strides[dim] * elementSize();  // 计算字节偏移
    return new Tensor(new_meta, _storage, _offset + added_offset);
    // ↑ 共享同一块 Storage，只是偏移不同
}
```

这种设计在 KV-Cache 中至关重要——每次 decode 后用 `slice(0, 0, pos+1)` 取出已缓存的 K/V，零拷贝。

### 4. `load` — 数据加载

```cpp
void Tensor::load(const void *src) {
    if (deviceType() == CPU)
        std::memcpy(data(), src, size_in_byte);      // CPU→CPU
    else
        runtime.api()->memcpy_sync(data(), src, ..., H2D);  // CPU→GPU
}
```

这是 Python 端通过 `ctypes.memmove` 写入数据的底层实现。

---

### 3）阶段 1.2：7 个核心算子讲解（原答复）

# 阶段 1.2：7 个核心算子的 CPU 实现

## 算子代码组织架构

每个算子遵循统一的三层结构，以 `add` 为例：

```
src/ops/add/
├── op.hpp          ← 声明 namespace llaisys::ops { void add(...); }
├── op.cpp          ← 分发逻辑：根据 deviceType 路由到 cpu/nvidia/metax
└── cpu/
    └── add_cpu.hpp ← CPU 实际计算逻辑
```

[src/ops/add/op.cpp](src/ops/add/op.cpp) 的分发模式是所有算子的「模板」：

```cpp
void add(tensor_t c, tensor_t a, tensor_t b) {
    // 1. 参数校验
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_SHAPE(...);  CHECK_SAME_DTYPE(...);
    
    // 2. 按设备类型路由
    switch (c->deviceType()) {
        case CPU:    return cpu::add(...);
        case NVIDIA: return nvidia::add(...);   // #ifdef ENABLE_NVIDIA_API
        case METAX:  return metax::add(...);    // #ifdef ENABLE_METAX_API
    }
}
```

**设计精髓**：编译时通过宏开关 `ENABLE_NVIDIA_API` 控制是否编译 CUDA 代码。CPU 路径始终可用，GPU 路径按需编入。

## 算子 1：Embedding — 查表拷贝

**数学含义**：给定 token ID，从嵌入矩阵中取出对应行。

```
input_ids = [42]       ← token ID
weight    = [V, H]     ← 词嵌入表 (152064×1536)
output    = weight[42] ← [1, H]  即该 token 的向量表示
```

## 算子 2：Linear — 矩阵乘法（最耗时）

**数学公式**：Y = XW^T + b

## 算子 3：RMS Norm — 归一化

## 算子 4：RoPE — 旋转位置编码

## 算子 5：Self-Attention — 最复杂的算子

## 算子 6：SwiGLU — 激活函数

## 算子 7：Argmax — 找最大值

## 算子全貌总结

| 算子 | 类型 | 计算复杂度 | 每 token 调用次数 (L=28) |
|------|------|:---------:|:-:|
| embedding | 查表 | O(H) | 1 |
| **linear** | **GEMM** | O(M × N × K) | **197** |
| rms_norm | 归约+逐元素 | O(H) | 57 |
| rope | 三角函数 | O(nh × d) | 56 |
| self_attention | 2×GEMM+softmax | O(nh × S^2 × d) | 28 |
| swiglu | 逐元素 | O(D_i) | 28 |
| add | 逐元素 | O(H) | 56 |
| argmax | 归约 | O(V) | 1 |

> **linear 是绝对瓶颈**，每 token 调用 197 次，每次涉及大矩阵乘法。

---

### 4）阶段 1.3：KV-Cache（原答复）

# 阶段 1.3：KV-Cache — 自回归推理的核心优化

## 为什么需要 KV-Cache？

Self-Attention 计算公式是 `Attn(Q, K, V) = softmax(QK^T / sqrt(d)) · V`

**没有 KV-Cache 时**，生成第 n 个 token 需要重新计算前面所有 n-1 个 token 的 K 和 V。

总计算量：O(n^2)

**有 KV-Cache 时**，每个 token 只计算自己的 K、V，然后追加到缓存。

总计算量：O(n)

## 代码实现

### 1. 初始化：预分配固定大小缓存

在 `init_cache()` 中每层维护 K/V，shape 为 `[maxseq, nkvh, dh]`。

### 2. 写入：追加当前 token 的 K/V

按 `current_pos` 计算偏移并写入缓存。

### 3. 读取：零拷贝切片

`slice(0, 0, current_pos + 1)` 读取历史窗口。

### 4. 高级功能：Save/Restore/前缀树

- SaveCache
- RestoreCache
- TruncateCache
- Prefix-Tree 复用

---

### 5）阶段 1.4：Qwen2 前向传播（原答复）

# 阶段 1.4：Qwen2 前向传播 — 完整推理流程

## 一次 Token 推理的完整流程图（原答复核心）

1. 输入准备（token/pos）
2. Embedding
3. 28 层 Transformer：
   - Attention：RMSNorm → QKV → RoPE → 写KV → SelfAttention → O投影 → 残差
   - MLP：RMSNorm → Gate/Up → SwiGLU → Down → 残差
4. FinalNorm
5. LM Head
6. Argmax 或 Sample
7. D2H 返回 next_token
8. current_pos++

原答复关键结论：

- 每 token 大约 424 次算子调用
- linear 约 197 次，主瓶颈
- 残差指针交换能减少拷贝

---

### 6）阶段 1.5：Python↔C++ 交互（原答复）

# 阶段 1.5：Python ↔ C++ 交互机制

## 整体调用链（原答复）

用户代码 → `python/llaisys/models/qwen2.py` → `python/llaisys/libllaisys/qwen2.py`（ctypes）→ C API（`src/llaisys/models/qwen2.cpp`）→ C++ Tensor/Ops/Model

## 关键点（原答复）

- `ctypes.CDLL` 加载共享库
- 正确设置 `argtypes` / `restype`
- Python 与 C 的结构体布局一致
- 模型句柄用 `void*`（opaque handle）
- 权重逐条 `LoadWeightByName` 送入 C++
- 推理用 `ModelInfer/ModelInferSample`

---

### 7）阶段 2：CPU 优化（原答复）

# 阶段 2：CPU 推理优化 —— 让朴素实现快起来

## 问题定位（原答复）

- linear 调用频次高、算量大，是 CPU 路径核心瓶颈

## 三条优化路线（原答复）

1. SIMD 向量化（AVX2/AVX-512）
2. OpenMP 多线程
3. BLAS 替换（OpenBLAS/MKL）

## 原答复建议优先级

先替换 linear 为 BLAS，再考虑其它算子局部优化。

---

## 二、后续学习步骤（原答复的完整方向）

### Step 1：GPU Runtime + CUDA 算子链路

- 读 Runtime 抽象
- 对照 CPU Runtime 与 NVIDIA Runtime
- 串起 op 分发到 CUDA kernel 的路径

### Step 2：GPU 性能优化复盘

- 用 nsys 看 API 热点
- 按“同步/分配/launch/拷贝”分类定位
- 对照优化动作验证收益

### Step 3：采样 + 服务 + UI + 会话

- sample 参数链路
- FastAPI SSE 流式
- session 与 cache snapshot

### Step 4：多用户推理

- 请求队列
- continuous batching
- per-request cache 绑定

### Step 5：分布式 TP

- Column/Row Parallel
- DistComm / NCCL / MPI
- all-reduce/all-gather 语义

### Step 6：量化与大模型能力

- INT8/INT4
- GPTQ/AWQ 转换与加载
- 量化 + TP 联合策略

---

## 三、核验与说明

- 本文件已按你的要求替换原内容。
- 已排除简历相关段落。
- 若你希望“连同每一条中间过渡句都逐字展开”，我可以继续在当前文件追加“逐条逐句原文版（超长）”，确保与你看到的对话文本一致到句级。

---

## 四、时间顺序逐句全量补充（继续扩展）

### 记录 01（阶段 1.3：KV-Cache 详细版）

好的，进入 **阶段 1.3：KV-Cache 机制** — 让推理从 O(n²) 降到 O(n) 的关键优化。

#### 为什么需要 KV-Cache

没有 KV-Cache：

- 生成第 n 个 token 需要重算前面所有 token 的 K/V
- 复杂度近 O(n²)

有 KV-Cache：

- 每步只算当前 token 的 K/V
- 写入 cache
- attention 直接读取历史 cache
- 复杂度近 O(n)

#### 本项目实现要点

1) 预分配缓存

- 每层 K/V 各一块，形状 `[maxseq, nkvh, dh]`
- 推理时不重复申请大块缓存

2) 追加写入

- 当前步 `current_pos` 对应一个行槽位
- 把当前步算出的 `k_3d`、`v_3d` 拷到该槽位

3) 历史读取

- 用 `slice(0, 0, current_pos+1)` 取历史窗口
- 这是零拷贝元操作

4) 快照与恢复

- Save：设备缓存拷贝到 CPU snapshot
- Restore：snapshot 回写设备
- Truncate：回滚位置
- Prefix Tree：会话前缀复用

---

### 记录 02（阶段 1.4：Qwen2 前向传播详细版）

好的，进入 **阶段 1.4：Qwen2 前向传播整体流**。

#### 单 token 完整路径

1. 输入 token_id 与 pos 写入输入缓冲
2. Embedding：查词表，得到 hidden_states
3. 进入 28 层循环，每层分两段：

Attention 段：

- 残差指针交换
- RMSNorm
- Q/K/V 三个线性投影
- reshape 为多头视图
- RoPE 注入位置信息
- 更新 KV-Cache
- 用历史 K/V 做 self-attention
- O 投影回 hidden size
- 残差相加

MLP 段：

- 残差指针交换
- RMSNorm
- gate/up 两条线性
- SwiGLU 激活
- down 投影
- 残差相加

4. 最后一层后：Final RMSNorm
5. LM Head 生成 logits
6. argmax 或 sample 得 next token
7. D2H 回传 token id
8. current_pos++

#### 调用统计（原答复核心结论）

- 每 token 约 424 次算子调用
- linear 约 197 次，是性能核心瓶颈

---

### 记录 03（阶段 1.5：Python ↔ C++ 交互详细版）

好的，进入 **阶段 1.5：Python ↔ C++ 交互机制**。

#### 三层到四层调用关系

- Python 业务层：`python/llaisys/models/qwen2.py`
- ctypes 绑定层：`python/llaisys/libllaisys/qwen2.py`
- C API：`include/llaisys/models/qwen2.h` + `src/llaisys/models/qwen2.cpp`
- C++ 内核：Tensor/Ops/Model/Runtime

#### 关键机制

1) 加载动态库

- Python 用 `ctypes.CDLL` 加载 `libllaisys.so`

2) 声明签名

- 每个 C 函数都设置 `argtypes/restype`
- 否则参数布局可能错误

3) 结构体对齐

- Python 端 Meta 结构体字段顺序必须与 C 一致

4) 模型句柄

- Python 仅持有 opaque handle（`void*`）

5) 权重加载

- Python 读取 safetensors
- 按 name + data ptr + shape 调 `LoadWeightByName`
- C++ 按名字路由到对应权重槽

6) 推理调用

- Python 调 `ModelInfer/ModelInferSample`
- C++ 前向后返回 token id

---

### 记录 04（阶段 2：CPU 优化详细版）

好的，进入 **阶段 2：CPU 推理优化** — 让朴素实现从极慢变得可用。

#### 核心定位

- 线性层调用密度最高
- 朴素三重循环导致吞吐低
- 缺 SIMD、缺多线程、cache 局部性差

#### 三条主优化路线

1) SIMD（AVX2/AVX-512）

- 内积循环改为向量加载 + FMA
- 尾部元素走标量补齐

2) OpenMP

- 对输出维并行
- 常见是外层循环并行

3) BLAS 替换（OpenBLAS/MKL）

- 用 SGEMM 直接替代三重循环
- 一般收益最高，工程风险最低

#### 原答复优先级建议

- 先把 linear 换 BLAS
- 再评估 self_attention/rms_norm/swiglu 等局部优化

---

### 记录 05（后续学习步骤，时间顺序版）

1. GPU Runtime + CUDA 算子执行链路（从抽象到 kernel）
2. 性能优化复盘（nsys 证据链）
3. 采样 + SSE + 会话管理
4. 多用户队列与 continuous batching
5. TP 分布式（NCCL/MPI）
6. INT8/INT4 + GPTQ/AWQ + 大模型可运行能力

---

### 记录 06（阶段 1.1：Tensor 逐句全量补充）

进入阶段 1.1 时的核心讲解重点是：

1) Tensor 是推理引擎基础容器，核心由 `Storage + Offset + Meta` 组成。

2) Meta 里最关键的是：

- dtype：数据类型
- shape：多维形状
- strides：步长，决定如何从线性内存映射到多维索引

3) `Tensor::create` 的关键动作：

- 由 shape 反推连续布局 strides
- 根据 device_type 通过 Runtime 分配对应设备内存
- 返回 shared_ptr Tensor，便于生命周期管理

4) 元操作与复制操作的边界：

- `view/permute/slice` 只改元数据，不搬数据
- `contiguous` 才会做真实数据整理复制

5) `slice` 的本质：

- shape 层面缩小某个维度
- offset 增加到起始位置
- 共享同一块 storage（零拷贝视图）

6) `load` 的本质：

- CPU 张量直接 memcpy
- 非 CPU 张量通过 Runtime H2D 拷贝

7) 结论：

- Tensor 层提供了“跨设备统一抽象 + 低成本视图操作”，是后续 KV-Cache 与 attention 路径高效运行的基础。

---

### 记录 07（阶段 1.2：7 算子逐句全量补充）

进入阶段 1.2 时的核心讲解重点是：

1) 先理解算子工程组织：

- 每个算子有统一分发入口 `src/ops/<op>/op.cpp`
- 入口先校验 shape/dtype/device
- 再按设备路由到 cpu/nvidia/metax 实现

2) Embedding：

- 本质是按 token id 查词向量表并拷贝行
- 计算量低，典型 memory-bound

3) Linear：

- 公式 `Y = XW^T + b`
- 朴素 CPU 是三重循环
- 在整条推理路径中调用频繁，是 CPU 性能主瓶颈

4) RMSNorm：

- 先求每行平方均值
- 再做缩放归一化
- 两遍扫描实现简单直观

5) RoPE：

- 对 head 内前后半维做旋转
- 旋转角由位置 `pos` 与 `theta` 决定
- 主要用于 Q/K 位置编码注入

6) Self-Attention：

- 先算 `QK^T * scale`
- 施加 causal mask
- softmax 后再乘 V
- 支持 GQA 头映射（Q 头到 KV 头分组）

7) SwiGLU：

- `out = up * silu(gate)`
- 属于 MLP 子层关键非线性门控

8) Argmax：

- 对 logits 线性扫描取最大索引
- 输出 next token id

9) 阶段性结论：

- 从系统性能视角看，linear 是阶段 2 的优化重心；其余算子优先级次之。

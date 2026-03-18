# Llaisys 推理引擎深度分析报告

> 基于 Qwen2 模型实现，逐层剖析推理过程中的数据流向、维度变换与底层算子调用。

---

## 一、项目内存管理架构

本项目**没有实现传统意义上的内存缓存池**（memory pool with reuse/caching），但有三层内存管理机制：

### 1.1 分配器抽象层

```
MemoryAllocator（抽象基类）
    └── NaiveAllocator（唯一实现）── 直接调用 cudaMalloc / cudaFree
```

- `allocator.hpp` 定义接口：`allocate(size)` / `release(memory)`
- `NaiveAllocator` 每次分配/释放都**直接转发到设备 API**，无池化逻辑

### 1.2 Storage + Runtime 生命周期管理

- `Storage` 持有裸内存指针，析构时通过 `Runtime::freeStorage()` 自动释放
- `Runtime` 使用 `std::shared_ptr<Storage>`（即 `storage_t`）实现引用计数

### 1.3 GPU 临时缓冲区的 grow-only 缓存

项目中**最接近"缓存池"**的实现，位于 `self_attention_nvidia.cu`：

```cpp
static float *s_q_all = nullptr;
static size_t s_q_all_sz = 0;

static void ensure_buf(float *&ptr, size_t &cur, size_t need) {
    if (need <= cur) return;       // 够用就复用
    if (ptr) cudaFree(ptr);        // 不够就释放旧的
    cudaMalloc(&ptr, need);        // 分配更大的
    cur = need;
}
```

5 个全局静态 GPU buffer（Q/K/V/Scores/Output），**只增长不缩小**，推理期间永不释放。

### 1.4 汇总

| 组件 | 策略 | 是否池化 |
|------|------|:--------:|
| NaiveAllocator | 直接 malloc/free | ❌ |
| Storage + shared_ptr | 引用计数自动释放 | ❌ |
| self-attention temp buffers | grow-only 静态缓存 | ✅ 简单复用 |
| KV Cache | 预分配固定大小 | ❌（预分配） |

> 项目预留了 `MemoryAllocator` 抽象接口，可方便替换为 caching allocator，但当前只有 `NaiveAllocator`。

---

## 二、算子概览与复杂度分析

### 2.1 算子列表

| 算子 | 每 token 调用次数 | 计算类型 | 瓶颈类型 |
|------|:-:|------|------|
| `embedding` | 1 | 查表拷贝 | memory |
| `rms_norm` | 2L+1 | 归约 + 逐元素 | memory |
| `linear` | **7L+1** | GEMM/GEMV | **compute** |
| `rope` | 2L | 三角函数 | compute |
| `self_attention` | L | 2×GEMM + softmax | **compute + memory** |
| `swiglu` | L | 逐元素 | memory |
| `add` | 2L | 逐元素 | memory |
| `argmax` | 1 | 归约 | memory |

> 以 Qwen2-7B（L=28）为例：每 token 共调用 **424 次算子**，其中 `linear` 占 197 次。

### 2.2 各算子实现要点

#### Argmax — 树形归约

- **CPU**：简单线性扫描，O(N)
- **GPU**：单 block shared memory 归约，使用 `ValIdx {float val; int idx}` 结构体
- **优化价值**：极低。每 token 只调用 1 次，对 152K 个 float 做比较，计算量比总推理量小 4~5 个数量级

#### Linear — 委托 cuBLAS

```cpp
// Y = X · W^T，核心就一行：
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...);
```

- GEMM 完全交给 `cublasGemmEx`，支持 F32/F16/BF16，统一用 `CUBLAS_COMPUTE_32F`
- Bias 加法用自写的 `add_bias_kernel`（因为 cuBLAS 不支持广播语义的 bias）
- Decoding 阶段 `M=1`，退化为 **GEMV**，cuBLAS 内部自动选择最优路径

#### Embedding — 纯查表

```cpp
out[i, j] = weight[token_id, j];  // 每线程拷贝一个元素
```

零浮点运算，纯 memory-bound，优化空间极小。

#### Self Attention — 最复杂的算子

7 步流程（详见第四章），含 GQA 展开、两次 batched GEMM、causal mask、softmax。

---

## 三、关键概念解析

### 3.1 Bias（偏置）与 Epsilon 的区别

| 概念 | 公式 | 作用 | 来源 |
|------|------|------|------|
| **Bias** | $Y = XW^T + b$ | 线性层的可学习参数 | 训练得到，从模型文件加载 |
| **Epsilon** | $\text{rsqrt}(\frac{\sum x^2}{H} + \epsilon)$ | RMSNorm 中防除零的极小值 | 超参数（如 `1e-6`） |

并非所有 linear 层都有 bias，模型中 MLP 的 gate/up/down 投影均为 `nullptr`。

### 3.2 模板全特化语法

```cpp
template<typename T> __device__ inline T from_float(float v);          // 主模板
template<> __device__ inline float from_float<float>(float v);          // float 特化
template<> __device__ inline __half from_float<__half>(float v);        // FP16 特化
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v); // BF16 特化
```

**为什么不能用普通函数重载？** 因为 `T` 出现在**返回值类型**而非参数类型中，C++ 不允许仅凭返回值类型不同来重载函数。必须通过 `from_float<T>(v)` 显式指定类型，这要求它是模板。

### 3.3 Meta 字段含义

| 字段 | 全称 | 含义 | Qwen2-7B |
|------|------|------|:--------:|
| `nlayer` | number of layers | Transformer 层数 | 28 |
| `hs` | hidden size | 隐藏维度（每 token 向量长度） | 3584 |
| `nh` | number of heads | Q 注意力头数 | 28 |
| `nkvh` | number of KV heads | KV 头数（GQA） | 4 |
| `dh` | dimension per head | 每头维度，`hs = nh × dh` | 128 |
| `di` | dimension intermediate | FFN 中间维度 | 18944 |
| `maxseq` | max sequence length | KV Cache 最大长度 | 32768 |
| `voc` | vocabulary size | 词表大小 | 152064 |
| `epsilon` | — | RMSNorm 防除零值 | 1e-6 |
| `theta` | — | RoPE 基频参数 | 1e6 |

---

## 四、推理完整数据流

### 4.1 第一阶段：从文本到矩阵

#### Tokenizer → Token IDs

```
"你好世界" → Tokenizer（Python 端）→ [token_0, token_1, ..., token_{n-1}]  (int64)
```

#### Embedding Lookup

```
算子：ops::embedding(hidden_states, input_ids_buf, weights.in_embed)
```

| 张量 | 维度 | 说明 |
|------|------|------|
| `input_ids_buf` | `[1]` int64 | 单个 token ID |
| `in_embed`（权重） | `[V, H]` = `[152064, 3584]` | 嵌入表 |
| **`hidden_states`（输出）** | **`[1, H]`** = `[1, 3584]` | 该 token 的向量表示 |

> **为什么输出是 `[1, H]` 而不是 `[S, H]`？** 本项目逐 token 推理，即使 prefill 阶段也是一个一个 token 送入（`S` 始终为 1）。这是一个可优化点——prefill 阶段可以批量处理。

每个 token_id 被转换为一个 **H=3584 维**的浮点向量，代表该 token 在语义空间中的表示。

---

### 4.2 第二阶段：Transformer Block（×L 层）

整个 Transformer 就是对这个 **H 维向量**反复变换——融入位置信息、上下文信息和语义变换。每层输入输出都是 `[1, H]`。

#### 总体流程图

```
residual ← hidden_states（指针 swap，零开销）
    │
    ▼
┌─ RMSNorm ──────────────────────────► norm_out [1, H]
│
├─ Q/K/V Linear ─────────────────────► q [1, h·d], k [1, h_kv·d], v [1, h_kv·d]
│
├─ Reshape（零拷贝）─────────────────► q_3d [1, h, d], k_3d [1, h_kv, d]
│
├─ RoPE（in-place）──────────────────► q_3d, k_3d（加入位置信息）
│
├─ KV Cache 写入 ────────────────────► cache[pos] ← k_3d, v_3d
│
├─ KV Cache 读取 ────────────────────► k_slice [pos+1, h_kv, d]
│
├─ Self Attention ───────────────────► attn_out [1, h, d]
│
├─ Output Linear ────────────────────► hidden_states [1, H]
│
├─ Residual Add 1 ───────────────────► hidden_states += residual
│
│   residual ← hidden_states（指针 swap）
│
├─ RMSNorm ──────────────────────────► norm_out [1, H]
│
├─ Gate Linear + Up Linear ──────────► gate [1, D_i], up [1, D_i]
│
├─ SwiGLU ───────────────────────────► mlp_act [1, D_i]
│
├─ Down Linear ──────────────────────► hidden_states [1, H]
│
└─ Residual Add 2 ───────────────────► hidden_states += residual
```

---

#### 2.1 RMSNorm

```
ops::rms_norm(norm_out, residual, attn_norm_w, epsilon)
```

$$Y_j = \frac{X_j \cdot W_j}{\sqrt{\frac{1}{H}\sum_{i=0}^{H-1} X_i^2 + \epsilon}}$$

| 输入/输出 | 维度 | 说明 |
|-----------|------|------|
| `residual` → `norm_out` | `[1, H] → [1, H]` | 维度不变 |
| `attn_norm_w` | `[H]` | 可学习权重 |

**GPU 实现**：每行一个 block，shared memory 树形归约求平方和，`rsqrtf` 取倒数平方根，逐元素归一化并乘以权重。

---

#### 2.2 Q/K/V 线性投影

```
ops::linear(q, norm_out, attn_q_w, attn_q_b)
ops::linear(k, norm_out, attn_k_w, attn_k_b)
ops::linear(v, norm_out, attn_v_w, attn_v_b)
```

| 张量 | 维度 | 示例 |
|------|------|------|
| `norm_out`（输入） | `[1, H]` | `[1, 3584]` |
| `attn_q_w` | `[h·d, H]` | `[3584, 3584]` |
| **`q`（输出）** | **`[1, h·d]`** | `[1, 3584]` |
| `attn_k_w` | `[h_kv·d, H]` | `[512, 3584]` |
| **`k`（输出）** | **`[1, h_kv·d]`** | `[1, 512]` |

> 因为 `M=1`（逐 token 推理），矩阵乘退化为 **GEMV**，cuBLAS 内部自动优化。

---

#### 2.3 Reshape → 多头视图

```cpp
q_3d = q->reshape({1, nh, dh});     // [1, 3584] → [1, 28, 128]
k_3d = k->reshape({1, nkvh, dh});   // [1, 512]  → [1, 4, 128]
v_3d = v->reshape({1, nkvh, dh});   // [1, 512]  → [1, 4, 128]
```

纯元数据操作，**零拷贝**，只改变 shape/stride 描述。

---

#### 2.4 RoPE（旋转位置编码）

```
ops::rope(q_3d, q_3d, pos_ids_buf, theta)  // in-place
ops::rope(k_3d, k_3d, pos_ids_buf, theta)  // in-place
```

对每个头的维度两两配对 `(l, l+d/2)`，应用旋转：

$$\begin{cases} \text{out}_{...,l} &= x_{...,l} \cdot \cos\theta_l - x_{...,l+d/2} \cdot \sin\theta_l \\ \text{out}_{...,l+d/2} &= x_{...,l+d/2} \cdot \cos\theta_l + x_{...,l} \cdot \sin\theta_l \end{cases}$$

其中 $\theta_l = \text{pos} / \theta^{2l/d}$。维度不变 `[1, h, d] → [1, h, d]`，compute-bound（含 `powf`/`cosf`/`sinf`）。

---

#### 2.5 KV Cache 交互

**写入**（D2D 拷贝）：

```cpp
cache_k[layer][pos, :, :] ← k_3d[0, :, :]   // h_kv × d 个元素
cache_v[layer][pos, :, :] ← v_3d[0, :, :]
```

| KV Cache | 完整维度 | 每次写入量 |
|----------|----------|:---------:|
| K cache | `[maxseq, h_kv, d]` = `[32768, 4, 128]` | 2 KB |
| V cache | 同上 | 2 KB |

**读取**（零拷贝 slice）：

```cpp
k_slice = kv_caches[i][0]->slice(0, 0, pos + 1);  // [pos+1, h_kv, d]
v_slice = kv_caches[i][1]->slice(0, 0, pos + 1);  // [pos+1, h_kv, d]
```

这是最简单的**连续 KV Cache** 方案——预分配、追加写入、slice 读取。无 PagedAttention、无动态扩容。

---

#### 2.6 Self Attention（最复杂的算子）

设当前总 token 数 `P = pos + 1`，完整 7 步流程：

| 步骤 | 操作 | 输入维度 | 输出维度 | 底层调用 |
|:----:|------|----------|----------|----------|
| ① | Gather Q | `[1, h, d]` strided T | `[h, 1, d]` contiguous float | `gather_all_q_kernel` |
| ② | Gather+Expand KV (GQA) | `[P, h_kv, d]` strided T | `[h, P, d]` contiguous float | `gather_expand_kv_kernel` |
| ③ | **Scores = Q·Kᵀ** | `[h,1,d]` × `[h,P,d]ᵀ` | `[h, 1, P]` | `cublasSgemmStridedBatched` |
| ④ | Scale + Causal Mask | `[h, 1, P]` | `[h, 1, P]` | `scale_kernel` + `causal_mask_kernel` |
| ⑤ | Softmax | `[h, 1, P]` 每行 | `[h, 1, P]` 概率 | `softmax_row_kernel` |
| ⑥ | **Output = Probs·V** | `[h,1,P]` × `[h,P,d]` | `[h, 1, d]` | `cublasSgemmStridedBatched` |
| ⑦ | Scatter | `[h, 1, d]` float | `[1, h, d]` strided T | `scatter_all_heads_kernel` |

**GQA 展开**：`h=28, h_kv=4, group_size=7`，步骤②中每个 KV 头被复制 7 次以匹配 Q 头数。

**Causal Mask**：对 `j > current_pos` 的位置设为 `-1e30`，softmax 后趋近于 0，防止看到未来 token。

---

#### 2.7 Output 投影 + 残差连接

```
reshape:  [1, h, d] → [1, H]           // 零拷贝
linear:   [1, H] × [H, H]ᵀ → [1, H]   // cublasGemmEx
add:      [1, H] + [1, H] → [1, H]     // add_kernel 逐元素
```

---

#### 2.8 FFN（前馈网络）

| 步骤 | 算子 | 维度变换 | 说明 |
|------|------|----------|------|
| RMSNorm | `rms_norm` | `[1, H] → [1, H]` | 归一化 |
| Gate 投影 | `linear` | `[1, H] → [1, D_i]` 即 `[1, 18944]` | 膨胀 ×5.3 |
| Up 投影 | `linear` | `[1, H] → [1, D_i]` | 膨胀 ×5.3 |
| **SwiGLU** | `swiglu` | `gate + up → [1, D_i]` | 激活函数 |
| Down 投影 | `linear` | `[1, D_i] → [1, H]` | 压缩回 H |
| 残差加 | `add` | `[1, H] + [1, H] → [1, H]` | 逐元素 |

**SwiGLU 公式**：

$$\text{out} = \text{up} \odot \left(\text{gate} \cdot \sigma(\text{gate})\right)$$

其中 $\sigma$ 是 sigmoid 函数。Gate 和 Up 两个 linear 当前**分别调用** cuBLAS，可优化为合并成一次 GEMM。

---

### 4.3 第三阶段：生成输出

#### Final Norm + LM Head

```
ops::rms_norm(hidden_states, hidden_states, out_norm_w, epsilon)   // [1, H] → [1, H]
ops::linear(logits, hidden_states, out_embed, nullptr)              // [1, H] × [V, H]ᵀ → [1, V]
```

LM Head 是**最大的一次 GEMM**，权重矩阵 `[152064, 3584]`。

#### Argmax

```
ops::argmax(next_token, max_val, logits_2d)   // [1, V] → token_id (int32)
```

单 block shared memory 归约，152K 次比较。

> ⚠️ 本项目只实现了 **greedy decoding（argmax）**，没有 Top-K / Top-P 采样。

#### D2H 回传

```cpp
model->memcpyD2H(&host_token, model->next_token, sizeof(int32_t));
model->current_pos++;
```

Token ID 返回 Python 端，由 Tokenizer 解码为文字，完成一次生成。

---

## 五、优化方向分析

### 5.1 FlashAttention

当前 self_attention 使用标准的 GEMM + 显式 score 矩阵 `[h, S, P]`，显存 O(N²)。FlashAttention 通过分块（tiling）在 SRAM 中完成，显存降至 O(N)，速度提升 2-4x，**是当前最值得做的优化**。

### 5.2 PagedAttention

| 场景 | 是否需要 |
|------|:--------:|
| 单请求推理 | ❌ 连续预分配已最优 |
| 张量并行（TP）分布式推理 | ❌ |
| 多用户并发服务（serving） | ✅ |

### 5.3 分布式推理 vs 分布式服务

| | 分布式推理 | 分布式服务 |
|--|----------|----------|
| **目标** | 单请求跑得动/跑得快 | 同时服务大量并发用户 |
| **典型方案** | 张量并行（TP）、流水线并行（PP） | + 请求调度、Continuous Batching、PagedAttention |
| **关注指标** | 单请求延迟 | 吞吐量（tokens/s） |
| **当前项目下一步** | ✅ TP + FlashAttention | 更远的目标 |

### 5.4 其他可优化点

| 优化点 | 说明 |
|--------|------|
| Prefill 批量化 | 当前逐 token prefill，可改为一次性处理 `[S, H]`，用 GEMM 而非 GEMV |
| Gate/Up 合并 | 两次 `cublasGemmEx` 可合并为一次 `[1, H] × [2·D_i, H]ᵀ` |
| 算子融合 | Linear + Bias + Activation 可通过 cuBLAS LT epilogue 融合 |
| CPU F16/BF16 | 当前用 `uint16_t` 做 `>` 比较，对负数结果错误 |

# Phase 6: FP16 半精度推理

## 概述

在 Phase 1-5（PagedAttention + 调度器 + 优化）的基础上，本阶段为推理引擎添加 **FP16 半精度激活**支持。

核心思路：**Mixed Precision（混合精度）**——激活张量和 KV-Cache 使用 FP16 存储与传输，所有数学运算（QK 点积、softmax、V 累加、归约等）保持 FP32，兼顾速度与数值稳定性。

### 收益

| 指标 | FP32 | FP16 |
|------|------|------|
| 激活缓冲区大小 | 4 bytes/elem | **2 bytes/elem** |
| KV-Cache 显存 | 4 bytes/elem | **2 bytes/elem** |
| 显存带宽利用 | 基准 | **约 2× 带宽节省** |
| 数值精度 | 完整 | 计算精度保持 FP32 |

---

## 修改清单

| 文件 | 修改内容 |
|------|----------|
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cu` | 内核模板化 + dtype 分发 |
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cuh` | 声明改 `void*` + dtype 参数 |
| `src/ops/self_attention/paged_attention.hpp` | 平台无关接口 `void*` + dtype |
| `src/ops/self_attention/paged_attention.cpp` | 转发 dtype 到 GPU 后端 |
| `src/ops/self_attention/nvidia/flashinfer_adapter.cuh` | FlashInfer 签名 `void*` |
| `src/ops/linear/nvidia/linear_nvidia.cu` | 新增混合精度路径（F32 权重 × F16 激活） |
| `src/llaisys/models/qwen2.cpp` | `act_dtype` 字段 + 全流程 FP16 集成 |

---

## 1. PagedAttention 内核模板化

**文件**: `src/ops/self_attention/nvidia/paged_attention_nvidia.cu`

### 设计

已有的 5/7 个 CUDA 内核（RMSNorm、RoPE、Embedding、SwiGLU、Add）已经通过模板 `to_float<T>()` / `from_float<T>()` 支持 FP16，唯独 PagedAttention 硬编码 `float*`。本次将其模板化。

### 关键实现

**1) 精度转换辅助函数**

```cpp
template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v)           { return v; }
template<> __device__ inline float to_float<__half>(__half v)         { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v)                { return v; }
template<> __device__ inline __half from_float<__half>(float v)              { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v){ return __float2bfloat16(v); }
```

**2) 模板化内核签名**

```cpp
template<typename T>
__global__ void paged_attention_kernel(
    T *__restrict__ output,         // I/O 类型 = T
    const T *__restrict__ query,
    const char *__restrict__ k_pool, // KV pool 按字节寻址
    const char *__restrict__ v_pool,
    ...);
```

- KV pool 使用 `const char*`（字节指针），通过 `reinterpret_cast<const T*>` 转换
- 中间变量 `partial`、`acc[]`、`m`、`l` 全部 FP32
- 最终输出 `from_float<T>(acc[i] * inv_l)` 转回 T 类型

**3) dtype 运行时分发**

```cpp
void paged_attention(..., llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  paged_attention_typed<float>(...);            break;
    case LLAISYS_DTYPE_F16:  paged_attention_typed<__half>(...);           break;
    case LLAISYS_DTYPE_BF16: paged_attention_typed<__nv_bfloat16>(...);   break;
    default: throw std::runtime_error("unsupported dtype");
    }
}
```

### 接口变更

```
// 旧签名
void paged_attention(float *output, const float *query, ...);

// 新签名 — 所有 4 层 (cuh / hpp / cpp / cu)
void paged_attention(void *output, const void *query, ..., llaisysDataType_t dtype = LLAISYS_DTYPE_F32);
```

默认值 `LLAISYS_DTYPE_F32` 保证已有调用站点（测试、benchmark）无需修改。

---

## 2. Linear 算子混合精度扩展

**文件**: `src/ops/linear/nvidia/linear_nvidia.cu`

### 问题

原有 linear 有一条 mixed-precision 路径：`FP16 权重 × FP32 输入`。但 FP16 推理中出现的场景是反过来的：

- INT8/INT4/AWQ 量化模型 → dequantize 输出 FP32 权重
- 激活是 FP16

因此需要新增 `FP32 权重 × FP16 输入 → FP16 输出` 路径。

### 实现

```
新增路径: w_dtype == F32 && in_dtype == F16
  1. FP16 输入 → convert_f16_to_f32_kernel → FP32
  2. cublasGemmEx F32 × F32 → F32
  3. F32 输出 → convert_f32_to_f16_kernel → FP16
```

使用 thread-local 缓存缓冲区（`in_f32_buf` / `out_f32_buf`），避免每次调用 cudaMalloc。

---

## 3. 模型层 FP16 集成

**文件**: `src/llaisys/models/qwen2.cpp`

### 3.1 `act_dtype` 字段

```cpp
// 模型结构体新增字段
llaisysDataType_t act_dtype;

// 构造函数中根据设备决定
if (dev != LLAISYS_DEVICE_CPU && tp_sz <= 1) {
    act_dtype = LLAISYS_DTYPE_F16;   // GPU 单卡 → FP16
} else {
    act_dtype = LLAISYS_DTYPE_F32;   // CPU 或 TP 多卡 → FP32
}
```

**为什么 TP 模式强制 FP32？** 当前 `allReduceSum` 实现只支持 `float*`（NCCL FP32）。未来可扩展为 FP16 allReduce。

### 3.2 KV-Cache

```cpp
void init_cache() {
    // 旧: Tensor::create(shape, LLAISYS_DTYPE_F32, ...)
    // 新: 使用 act_dtype, FP16 时 KV-Cache 显存减半
    auto k_c = Tensor::create(shape, act_dtype, device_type, device_id);
    auto v_c = Tensor::create(shape, act_dtype, device_type, device_id);
}
```

### 3.3 激活缓冲区

`init_buffers()` 中所有激活张量统一使用 `act_dtype`：

| 缓冲区 | dtype |
|--------|-------|
| hidden_states, residual, norm_out | `act_dtype` |
| q, k, v, attn_out | `act_dtype` |
| gate, up, mlp_act | `act_dtype` |
| **logits** | **FP32（固定）** |
| **max_val** | **FP32（固定）** |

logits 保持 FP32 是因为采样（softmax + argmax）对精度敏感。

### 3.4 BlockAllocator 适配

```cpp
// 旧: elem_size = sizeof(float)  →  4 bytes
// 新: elem_size = dsize(act_dtype) → FP16 时 = 2 bytes
BlockAllocatorConfig cfg;
cfg.elem_size = llaisys::utils::dsize(model->act_dtype);
```

### 3.5 KV-Cache 指针算术

原来使用 `float*` 指针偏移，只对 FP32 正确。改为 `char*` 字节寻址：

```cpp
// 旧: float* k_ptr = (float*)pool->data() + block_id * stride + ...;
// 新: dtype 无关的字节寻址
size_t kv_bytes = kv_dim * dsize(act_dtype);   // 原来: kv_dim * sizeof(float)
char* pool_base = (char*)pool->data();
char* k_dst = pool_base + block_id * pool_block_stride + layer * pool_layer_stride
              + t * nkvh * dh * dsize(act_dtype) + head * dh * dsize(act_dtype);
cudaMemcpyAsync(k_dst, k_src_bytes, kv_bytes, ...);
```

### 3.6 PagedAttention 调用

```cpp
// 旧: paged_attention((float*)attn_out->data(), (const float*)q->data(), ...)
// 新: 直接传 void*, 加 dtype 参数
paged_attention(attn_out->data(), q->data(), ..., model->act_dtype);
```

---

## 4. 已有 Kernel 兼容性

以下 5 个 kernel 已经通过模板支持 FP16，本次**无需修改**：

| Kernel | 文件 | FP16 支持方式 |
|--------|------|---------------|
| RMSNorm | `rms_norm_nvidia.cu` | `to_float<T>` / `from_float<T>` 模板 |
| RoPE | `rope_nvidia.cu` | 同上 |
| Embedding | `embedding_nvidia.cu` | 同上 |
| SwiGLU | `swiglu_nvidia.cu` | 同上 |
| Add | `add_nvidia.cu` | 同上 |

---

## 5. 设计决策总结

| 决策 | 原因 |
|------|------|
| 中间计算保持 FP32 | 避免 FP16 精度不足导致 softmax/归约数值不稳定 |
| logits 保持 FP32 | 采样对精度敏感,vocab 维度大时 FP16 容易溢出 |
| TP 模式强制 FP32 | 现有 allReduce 只支持 float*,待后续优化 |
| KV pool 用 `char*` 字节寻址 | dtype-agnostic 指针算术,避免硬编码 `float*` 偏移 |
| 默认参数 `dtype=F32` | 向后兼容已有测试和 benchmark,无需修改调用站点 |
| `to_float`/`from_float` 模板 | 与项目已有 5 个 kernel 的风格一致 |

---

## 6. 混合精度 Bug 修复记录

初始实现编译通过但推理输出垃圾。原因：多个 kernel 的 dtype 分发逻辑不兼容 FP16 激活 + FP32 权重场景。

### Bug 1: Embedding — 缺少 F32 权重 → F16 输出路径

**问题**: `embedding_nvidia.cu` 按 `weight->dtype()` (F32) 分发，将 FP16 output buffer cast 为 `float*`，导致 2× stride 错位。

**修复**: 在 `LLAISYS_DTYPE_F32` case 中检查 `out->dtype() == F16`，调用新增的 `embedding_f32_to_f16_kernel`。

### Bug 2: RMSNorm — 按 weight dtype 而非 input dtype 分发

**问题**: `rms_norm_nvidia.cu` 按 `weight->dtype()` (F32) 分发，将 FP16 input/output cast 为 `float*`，同样导致 stride 错位。这是每层最先执行的操作，错误立即级联。

**修复**: 
- 新增 `rms_norm_mixed_kernel<Tio, Tw>` 双类型模板
- 先检查 `in_dtype vs w_dtype` 是否需要混合路径，再 fallback 到同类型 switch

### Bug 3: Linear — FP32 bias + FP16 output 处理错误

**问题**: F32w × F16in → F16out 路径中，bias 在 F32→F16 转换**之后**添加到已废弃的 `out_f32_buf`，且将 FP32 bias 错误 cast 为 `__half*`。

**修复**: 将 bias 加法移到 F32→F16 转换**之前**，在 F32 空间完成 bias 加法。

### Bug 4: Linear — 缺少 F32w × F16in → F32out 路径

**问题**: LM Head 需要 F32 weight × F16 input → F32 logits，但该组合不存在。

**修复**: 在混合精度分支中按 `out_dtype` 分流，F32 输出直接写入 `out->data()`。

---

## 7. FP16 权重加载

### 发现

原始模型权重在 safetensors 中以 **BF16** 存储（非 FP32），但我们的加载代码将其转为 FP32：

```python
# 旧: BF16 → FP32 (不必要的精度提升, 显存翻倍)
tensor = tensor.to(torch.float32)
dtype_enum = 13  # LLAISYS_DTYPE_F32
```

### 修改

**文件**: `python/llaisys/models/qwen2.py`

```python
# 新: GPU 模式直接用 FP16, 跳过 FP32 中间步骤
if use_fp16_weights:
    tensor = tensor.to(torch.float16)
    dtype_enum = 12  # LLAISYS_DTYPE_F16
```

**文件**: `src/ops/linear/nvidia/linear_nvidia.cu`

- 合并 F16w × F32in → F32out 和 F16w × F16in → F32out 为统一路径
- cuBLAS 同时支持两种情况: `cublasGemmEx(..., CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, ...)`

### 收益

权重为 FP16 后，所有 Linear 层走原生 F16 GEMM（Tensor Core 加速），无需 F16↔F32 转换开销。

---

## 8. 性能基准测试

**模型**: DeepSeek-R1-Distill-Qwen-1.5B (原始 BF16)  
**设备**: NVIDIA GPU | **生成**: 128 tokens | **采样**: greedy (top_k=1) | **3 runs avg**

| 指标 | FP32 基准 | FP16 激活 (FP32 权重) | FP16 全面 (FP16 权重) |
|------|-----------|---------------------|--------------------|
| **吞吐量** | 30.1 tok/s | 30.3 tok/s (+0.7%) | **53.9 tok/s (+79%)** |
| **显存 (加载后)** | 7252 MB | 7196 MB (-0.8%) | **3840 MB (-47%)** |
| **显存 (推理后)** | 7252 MB | 7216 MB | **3866 MB (-47%)** |
| KV-Cache 理论大小 | 224 MB | 112 MB (-50%) | **112 MB (-50%)** |
| 正确性 | 基准 | token 一致 ✅ | **token 一致 ✅** |

### 分析

**Phase 1 — 仅 FP16 激活 (FP32 权重):**
- 吞吐量几乎不变: 权重仍 FP32, GEMM 通过 cublasGemmEx 在 FP32 精度执行, F16↔F32 转换开销抵消带宽节省
- 显存节省有限: 权重占 ~6.8GB, KV-Cache 节省 ~112MB 相对占比小

**Phase 2 — FP16 权重 + FP16 激活:**
- **吞吐量提升 79%**: 所有 GEMM 走 Tensor Core FP16 路径, 无转换开销
- **显存减半 47%**: 权重 3.4GB → 1.7GB, KV-Cache 224MB → 112MB, 激活缓冲区减半
- 三次运行结果完全一致, token-for-token 匹配 HuggingFace BF16 参考

### 环境变量

```bash
# 强制 FP32 模式 (用于 benchmark 对比)
LLAISYS_FORCE_FP32=1 python test/test_infer.py --device nvidia ...
```

---

## 9. 编译验证

```bash
$ xmake build
[100%]: build ok
```

所有目标编译通过，包括：
- `libllaisys.so`（主库）
- `llaisys-test-paged-attention`（PA 单元测试）
- `llaisys-bench-paged-attention`（PA 基准测试）
- `llaisys-test-block-allocator`、`llaisys-test-paged-batch` 等

---

## 10. Batch Prefill 优化

### 10.1 问题背景

原始推理路径中，prefill 阶段采用**逐 token 串行**方式：

```cpp
for (size_t t = 0; t < ntoken; ++t) {
    // 每次只处理 1 个 token，循环 ntoken 次
}
```

这导致 prefill 阶段需要多次 kernel launch，每次仅处理 1 个 token 的矩阵运算，GPU 利用率极低。

### 10.2 实现方案

新增 `prefill_batch()` 静态函数，将所有输入 token **一次性**送入各层计算：

```
输入: [S] token_ids  →  Embedding → [S, hidden]
      [S] pos_ids     →  每层:  RMSNorm → QKV → RoPE → Attention → FFN
                       →  一次性写入 S 个 KV 到 cache
                       →  取最后一个 token → LM Head → Sample
```

**关键实现细节：**

1. **临时缓冲区**: 所有中间张量形状从 `[1, ...]` 变为 `[S, ...]`
2. **位置编码**: `pos_ids = [0, 1, 2, ..., S-1]`，一次性计算所有 RoPE
3. **Self-Attention**: prefill 阶段仍使用标准 `ops::self_attention`（非 PagedAttention），可并行处理 S 个 query
4. **KV-Cache 写入**: 单次 `cudaMemcpy` 将 S 个 KV 向量写入 block 0 起始的连续缓存
5. **输出提取**: 仅取最后一个 token 的 hidden state 送入 LM Head

**修改文件：** `src/llaisys/models/qwen2.cpp`

### 10.2.1 核心代码

#### 入口路由

```cpp
// llaisysQwen2ModelInferSample 入口
if (ntoken > 1) {
    // 首次推理 (prefill): 所有 token 一次性通过 transformer
    return prefill_batch(model, token_ids, ntoken, temperature, top_k, top_p);
}
// ntoken == 1: 逐 token decode 路径 (使用 PagedAttention)
```

#### 临时缓冲区分配

```cpp
// 所有中间张量从 [1, ...] 扩展为 [S, ...]
auto hs_buf   = Tensor::create({S, hs}, dt, dev, did);       // [S, 1536]
auto q_buf    = Tensor::create({S, q_dim}, dt, dev, did);     // [S, 1536] (12*128)
auto k_buf    = Tensor::create({S, kv_dim}, dt, dev, did);    // [S, 256]  (2*128)
auto v_buf    = Tensor::create({S, kv_dim}, dt, dev, did);    // [S, 256]
auto attn_buf = Tensor::create({S, nh_local, dh}, dt, dev, did); // [S, 12, 128]
auto gate_buf = Tensor::create({S, di_local}, dt, dev, did);  // [S, 8960]
// ... 更多缓冲区
```

#### KV-Cache 批量写入

**单模型路径** — 写入连续 KV-Cache:

```cpp
// 将 S 个 token 的 KV 向量一次性写入连续缓存
size_t kv_row_bytes = nkvh_local * dh * dsize(dt);  // 每个 token 的 KV 大小
char* k_dst = (char*)model->kv_caches[layer][0]->data();  // 从 pos 0 开始
char* v_dst = (char*)model->kv_caches[layer][1]->data();
model->memcpyOnDevice(k_dst, k_3d->data(), S * kv_row_bytes);  // 单次拷贝
model->memcpyOnDevice(v_dst, v_3d->data(), S * kv_row_bytes);
```

**批量上下文路径** — Scatter 到 Block Pool (PagedAttention):

```cpp
// 按 block 粒度拷贝: 每个 block 一次 memcpy
for (size_t bi = 0; bi < blocks_needed; ++bi) {
    int block_id = slot.page_table.block_ids()[bi];
    size_t tok_start = bi * bs;
    size_t ntok = std::min((size_t)bs, S - tok_start);

    char* k_src = (char*)k_3d->data() + tok_start * kv_bytes;
    char* k_dst = (char*)alloc.get_k_ptr(block_id, (int)layer);
    model->memcpyOnDevice(k_dst, k_src, ntok * kv_bytes);  // 类似 v
}
```

#### 最后一个 token 提取

```cpp
// hs_buf 形状 [S, hs]，取最后一行 [S-1, :] → model->hidden_states [1, hs]
char* last_hs_src = (char*)hs_buf->data() + (S - 1) * hs * elem_sz;
model->memcpyOnDevice(model->hidden_states->data(), last_hs_src, hs * elem_sz);
// 之后 Final Norm → LM Head → Sample，与 decode 路径完全相同
```

#### 原始逐 token 路径 vs Batch Prefill 对比

| 操作 | 逐 token (原始) | Batch Prefill |
|------|:--:|:--:|
| Kernel launch 次数 | ~10 × S × L | ~10 × L |
| GEMM 矩阵尺寸 | [1, hidden] × [hidden, dim] | [S, hidden] × [hidden, dim] |
| cudaMemcpy (KV写入) | S × L 次 (每次1行) | L 次 (每次S行) |
| GPU 利用率 | 低 (大量 launch overhead) | 高 (矩阵运算并行) |

> L = 28 (Qwen2-1.5B 层数), S = prefill token 数

### 10.3 正确性验证

```
$ python test/test_infer.py --device nvidia \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B \
    --prompt "What is 2+3?" --max_steps 32 --test
Test passed!
```

输出与 HuggingFace BF16 参考完全一致。

### 10.4 性能对比

| 配置 | 路径 | 总吞吐 (tok/s) | 显存 (MB) | 相对基线 |
|------|------|---------------|-----------|---------|
| FP32 权重 + 逐 token prefill | 单模型 | 30.1 | 7252 | 基线 |
| FP16 权重 + 逐 token prefill | 单模型 | 53.9 | 3840 | +79% |
| FP16 权重 + Batch Prefill | 单模型 | 55.8 | 3840 | +85% |
| FP16 权重 + Batch Prefill | **BatchContext** | **59.1** | 3840 | **+96%** |

> - 测试条件: 128 decode tokens, greedy sampling, 3 次取平均, prompt 17 tokens
> - BatchContext 路径 decode 使用 PagedAttention kernel（专为单 query 优化），比标准 self_attention 更快
> - Batch Prefill 相对逐 token: 减少 ~(S-1) × L 次 kernel launch，prompt 越长提升越显著

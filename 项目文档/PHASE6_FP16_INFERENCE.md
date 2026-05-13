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
| `src/ops/self_attention/nvidia/flashinfer_adapter.cu` | 适配新版 FlashInfer public batch decode API（F16 paged decode） |
| `src/ops/linear/nvidia/linear_nvidia.cu` | 新增混合精度路径（F32 权重 × F16 激活） |
| `src/llaisys/models/qwen2.cpp` | `act_dtype` 字段 + 全流程 FP16 集成 |
| `xmake.lua`, `xmake/nvidia.lua` | FlashInfer 编译链、CUDA devlink、测试/benchmark CUDA include |
| `third_party/flashinfer/flashinfer/page.cuh` | 本地 overlay，修复 FlashInfer 头文件在 `-Werror` 下的 member-init-order 编译问题 |

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

### 8.1 FlashInfer F16 Decode 补充

本次补充了 `flashinfer_adapter.cu`，将 LLAISYS 的 paged KV block pool 对接到 FlashInfer 当前公开的 batch decode API，并将 F16 decode 路径接入 `ops::paged_attention(...)` 的 NVIDIA 分发。

#### 启用条件

当前 FlashInfer 路径只在以下条件同时满足时启用：

- `dtype == F16`
- `head_dim ∈ {64, 128, 256}`
- `group_size = num_heads / num_kv_heads ∈ {1, 2, 4, 8}`
- `kv_quant == FP32`
- 编译时开启 `--flashinfer=y`

其余情况统一自动回退到项目自带的 `paged_attention_nvidia.cu`，不会影响原有功能正确性。

#### 与当前项目模型的关系

这点需要特别说明：

- 当前项目默认 Qwen2 单卡本地 head 配置通常是 `local_nh=12, local_nkvh=2`
- 因此 `group_size = 12 / 2 = 6`
- **6 不在 FlashInfer 当前 decode kernel 支持集合中**

所以对当前默认 Qwen2 配置，系统会**自动回退到自研 paged attention kernel**。  
换句话说：这次接入让 FlashInfer 路径“可用且受保护”，但**默认 Qwen2-1.5B 配置本身不会直接吃到 FlashInfer 加速**。

#### 正确性验证

新增 GPU F16 correctness case，配置为支持 FlashInfer 的：

- `batch_size=2`
- `num_heads=8`
- `num_kv_heads=2`
- `head_dim=128`
- `block_size=16`

执行：

```bash
xmake run llaisys-test-paged-attention
```

结果：

- CPU 全部已有测试通过
- GPU F16 paged attention 通过
- 最大绝对误差：`7.33137e-05`

#### 性能对比（RTX 4060 Laptop, GPU F16 decode synthetic bench）

说明：

- 基线：`xmake f -c --nv-gpu=y`
- FlashInfer：`xmake f -c --nv-gpu=y --flashinfer=y --flashinfer-include=/home/bbq/.local/lib/python3.10/site-packages/flashinfer/data/include`
- benchmark 稳定使用 FlashInfer 支持的配置：`num_heads=8, num_kv_heads=2, head_dim=128`

| 配置 | 原始 F16 内建 kernel | FlashInfer F16 | 加速比 | 延迟下降 |
|------|----------------------|----------------|--------|----------|
| `B=1, seq=64`  | 0.136 ms | **0.110 ms** | **1.24x** | 19.1% |
| `B=1, seq=256` | 0.282 ms | **0.113 ms** | **2.50x** | 59.9% |
| `B=4, seq=64`  | 0.117 ms | **0.103 ms** | **1.14x** | 12.0% |
| `B=4, seq=256` | 0.269 ms | **0.099 ms** | **2.72x** | 63.2% |
| `B=8, seq=64`  | 0.137 ms | **0.093 ms** | **1.47x** | 32.1% |
| `B=8, seq=256` | 0.239 ms | **0.112 ms** | **2.13x** | 53.1% |

结论：

- 对 **support matrix 内** 的 F16 decode 场景，FlashInfer 在中长上下文上带来明显收益
- `seq=256` 时收益最明显，约 **2.1x ~ 2.7x**
- 短上下文 `seq=64` 也有收益，但提升更温和

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

## 10. Batch Prefill 优化（非 Chunked Prefill）

> 纠正说明：当前代码实现的是 **Batch Prefill**，不是严格意义上的
> **Chunked Prefill**。它会把整个 prompt 的 `S` 个 token 一次性送入模型计算，
> 并把生成的 K/V 按 block 粒度写入 Paged KV block pool；但 prefill 阶段的
> attention 仍然调用标准 `ops::self_attention`，没有使用 `paged_attention(...)`。
>
> 因此，当前实现没有做到“将长 prompt 切成多个 chunk，并在 chunk 之间与
> decode 请求交错调度”。它主要解决的是逐 token prefill 的低 GPU 利用率和
> 多次 kernel launch 问题。

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
4. **KV-Cache 写入**:
   - SingleModel 路径写入连续 KV-Cache
   - BatchContext 路径按 block 粒度写入 Paged KV block pool
5. **输出提取**: 仅取最后一个 token 的 hidden state 送入 LM Head

### 10.2.1 与真正 Chunked Prefill 的区别

真正的 Chunked Prefill 通常指：

```
长 prompt S tokens
→ 切成 chunk_0, chunk_1, ...
→ 每次只处理一个 chunk
→ chunk 之间允许插入 decode step
```

它的主要目标是避免长 prompt 长时间独占 GPU，提高在线服务中的 TTFT/TPOT
平衡和调度公平性。

当前实现没有 chunk 级调度，也没有在 prefill 阶段通过 `paged_attention(...)`
读取历史 paged KV。当前实现更准确地说是：

```
Batch Prefill:
  一次性处理完整 prompt
  attention 使用连续临时 K/V + self_attention
  K/V 结果同步写入连续 cache 或 Paged KV block pool
```

**修改文件：** `src/llaisys/models/qwen2.cpp`

### 10.2.2 核心代码

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

**批量上下文路径** — Scatter 到 Paged KV Block Pool（注意：不是 PagedAttention）:

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
> 当前 BatchContext 路径按 block 粒度拷贝 K/V，因此拷贝次数约为
> `L × ceil(S / block_size)`，而不是严格的 `L` 次；表格中的 `L 次`
> 更适用于 SingleModel 连续 KV 写入路径。

### 10.3 正确性验证

```
$ python test/test_infer.py --device nvidia \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B \
    --prompt "What is 2+3?" --max_steps 32 --test
Test passed!
```

输出与 HuggingFace BF16 参考完全一致。

### 10.4 性能对比

| 配置 | 路径 | TTFT (ms) | Decode (tok/s) | Total (tok/s) | 峰值显存 (MB) |
|------|------|-----------|---------------|---------------|-------------|
| FP32 + 逐token | SingleModel | 640 | 22.7 | 25.3 | 6974 |
| FP32 + Batch Prefill | BatchCtx | 147 | 24.4 | 27.0 | 6974 |
| **FP16 + Batch Prefill** | **SingleModel** | **20.4** | **50.8** | **57.6** | **3865** |
| **FP16 + Batch Prefill** | **BatchCtx** | **20.3** | **49.5** | **56.0** | **3865** |

> - 测试条件: prompt 17 tokens, decode 128 tokens, greedy, 3 次取平均
> - TTFT = Time To First Token (首 token 延迟, 即 prefill 耗时)
> - FP16 vs FP32 TTFT: 20.4ms vs 640ms → **31× 加速** (batch prefill + FP16 GEMM)
> - FP16 vs FP32 Decode: 50.8 vs 22.7 → **2.2× 加速**
> - 显存: 3865 vs 6974 → **-45%**

### 10.5 Bug 修复: `llaisysQwen2ModelInfer` 未使用 Batch Prefill

**问题**: `llaisysQwen2ModelInfer`（greedy argmax 路径）仍保留逐 token for 循环，
未调用 `prefill_batch()`，导致 `model.generate(top_k=1)` 的 TTFT 高达 285ms。

**修复**: 将 `llaisysQwen2ModelInfer` 改为委托调用 `llaisysQwen2ModelInferSample`：

```cpp
__export int64_t llaisysQwen2ModelInfer(...) {
    // 委托给 InferSample, 使用 greedy 参数 (top_k=1)
    return llaisysQwen2ModelInferSample(model, token_ids, ntoken,
                                        0.0f, 1, 1.0f);
}
```

修复效果: TTFT 285ms → **20.4ms** (14× 提升)。

---

## 11. SingleModel PagedAttention 迁移 (Decode +30%)

### 11.1 问题分析

SingleModel（非 batch 路径）的 decode 阶段使用 `ops::self_attention`，该函数内部每次调用都执行：

1. **56 次 `cudaMalloc`** — 为 `block_tables` (int*) 和 `seq_lens` (int*) 各分配 28 层 × 1 次
2. **56 次 `cudaMemcpy` (H2D)** — 拷贝上述数据到 GPU
3. **56 次 `cudaFree`** — 释放临时 GPU 内存

每个 decode token 都重复此开销。在 RTX 4060 上，`cudaMalloc/cudaFree` 单次约 5-10μs，56 次累积 ~0.5ms/step，对短序列推理瓶颈显著。

### 11.2 解决方案

#### (a) `paged_attention_device` — 零 malloc 变体

新增 `paged_attention_device()` 函数，接受**已在 GPU 上的** `block_tables` 和 `seq_lens` 指针，跳过所有 `cudaMalloc/cudaMemcpy/cudaFree`：

```cpp
// src/ops/self_attention/nvidia/paged_attention_nvidia.cu
template<typename T>
void paged_attention_device_typed(
    void* out, void* q, void* k_pool, void* v_pool,
    void* block_tables_dev, void* seq_lens_dev,  // 已在 GPU 上
    int batch, int nhead, int nkvh, int dh, int block_size,
    int max_blocks_per_seq, int max_ctx_len, cudaStream_t stream)
{
    // 直接 reinterpret_cast，不分配/释放任何 GPU 内存
    // 与 paged_attention_typed<T> 使用相同 kernel launch
}
```

#### (b) `reshape_and_cache` — GPU KV 写入 kernel

新增 CUDA kernel 将 K/V 数据写入 paged block pool：

```cpp
// src/ops/cache/nvidia/cache_kernels.cu
template<typename T>
__global__ void reshape_and_cache_kernel(
    const T* __restrict__ key,      // [B, kv_dim]
    const T* __restrict__ value,    // [B, kv_dim]
    T* __restrict__ k_pool,         // block pool
    T* __restrict__ v_pool,
    const int64_t* __restrict__ positions,      // [B]
    const int* __restrict__ block_tables,       // [B, max_blocks]
    int kv_dim, int block_size, int max_blocks_per_seq)
```

Grid: `(batch_size, ceil(kv_dim/256))`，每个 thread 写一个 K+V 元素。利用 `positions` 计算目标 block ID 和 block-offset。

#### (c) SingleModel 集成

`qwen2.cpp` 修改：

| 组件 | 变更 |
|------|------|
| `BlockAllocator` | 在 `init_cache()` 创建，pool shape=`[num_blocks, nlayer, block_size, nkvh, dh]` |
| `PageTable` | block_size=16，预分配所有 blocks（单序列不需要动态分配） |
| `d_block_tables` | GPU 上的 block table，init 时一次上传 |
| `d_seq_lens` | GPU 上的 seq_lens buffer，decode 前更新 |
| decode 路径 | `reshape_and_cache` → `paged_attention_device` (GPU) |
| prefill 路径 | self_attention 不变 → KV 按 block 写入 pool (GPU) |
| SaveCache | 从 block pool 按 block 拷贝到 CPU |
| RestoreCache | 从 CPU 按 block 写回 block pool |

### 11.3 CUDA Graph 尝试与放弃

初始目标是用 CUDA Graph 捕获 decode 路径，消除 kernel launch 开销。但发现三个障碍：

1. **`ops::add` 调用 `cudaSetDevice()`** — 不可在 stream capture 中执行（已修复：移除 setDevice 调用）
2. **`linear_nvidia.cu` 中的 lazy `cudaMalloc`** — 混合精度路径首次执行时分配临时 buffer
3. **`seq_lens`/`positions` 每步变化** — Graph 捕获的是固定参数，需要 `cudaGraphExecKernelNodeSetParams` 逐节点更新，实现复杂度高

**决策**：暂不使用 CUDA Graph。仅通过 `paged_attention_device` 消除 malloc 开销已获得显著收益。

### 11.4 性能对比

**测试环境**: RTX 4060 Laptop 8GB, Qwen2-1.5B FP16, max_tokens=128

| 指标 | Before (self_attention) | After (paged_attention) | 变化 |
|------|------------------------|------------------------|------|
| TTFT | 20.4 ms | **18.2 ms** | -11% |
| Decode | 50.8 tok/s | **65.8 tok/s** | **+30%** |
| Total | 57.6 tok/s | **65.7 tok/s** | +14% |
| vs HuggingFace | 1.6× | **1.8×** | — |

### 11.5 修改文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/ops/cache/nvidia/cache_kernels.cu` | **新增** | `reshape_and_cache` CUDA kernel |
| `src/ops/cache/nvidia/cache_kernels.cuh` | **新增** | kernel 声明 |
| `src/ops/cache/cache_ops.hpp` | **新增** | 高层 `reshape_and_cache` 声明 |
| `src/ops/cache/cache_ops.cpp` | **新增** | 设备类型分发 |
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cu` | 修改 | `paged_attention_device_typed` |
| `src/ops/self_attention/nvidia/paged_attention_nvidia.cuh` | 修改 | `paged_attention_device` 声明 |
| `src/ops/self_attention/paged_attention.hpp` | 修改 | 高层 `paged_attention_device` 声明 |
| `src/ops/self_attention/paged_attention.cpp` | 修改 | 分发到 nvidia 后端 |
| `src/ops/add/op.cpp` | 修改 | 移除 `setDevice` (CUDA Graph 兼容) |
| `src/llaisys/models/qwen2.cpp` | 修改 | 集成 BlockAllocator、PageTable、paged_attention |
| `xmake.lua` | 修改 | cache_ops.cpp 编译到 llaisys target |

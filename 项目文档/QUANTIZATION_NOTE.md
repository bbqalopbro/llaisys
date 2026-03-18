# LLAISYS 量化实现笔记（INT8 / INT4 / GPTQ/AWQ）

> 日期：2026-03  
> 分支：`feat/quantization`

---

## 1. 目标与范围

本次量化迭代目标：

1. 支持 **INT8 W8A16**（权重 INT8、激活 FP16/FP32）
2. 支持 **INT4 W4A16**（权重 INT4 打包、激活 FP16/FP32）
3. 在不改动对外推理 API 的前提下，兼容 **GPTQ/AWQ** 权重加载
4. 保持可回退：原 FP32 路径继续可用

---

## 2. 最终效果（结论）

- INT8（per-channel symmetric）：约 **2x** 压缩
- INT4（per-group symmetric, g=128）：约 **3.76x** 压缩
- GPTQ/AWQ：通过 Python 侧加载时转换，复用同一 INT4 推理链路
- INT4 kernel：完成一轮低风险优化（每线程处理 1 packed-byte，写 2 个输出）

---

## 3. 实现架构

### 3.1 INT8 路径（已稳定）

- 量化公式（按行）：
  - `scale = max(abs(W_row)) / 127`
  - `W_int8 = round(W / scale).clamp(-128, 127)`
- 推理时：
  - `W_fp32 = W_int8 * scale`（dequantize）
  - 再走已有 `linear`

关键点：

- C++ 侧在 `linear_maybe_dequant` 中按 `dtype` 自动分流
- `I8` 走 INT8 dequantize

### 3.2 INT4 路径（已稳定）

- 量化粒度：per-group（默认 `group_size=128`）
- 值域：`[-8, 7]`
- 打包：2 个 INT4 存入 1 个 `uint8`
  - `byte = ((val1 + 8) << 4) | ((val0 + 8) & 0x0F)`
- 推理时：
  - 从 `U8` 解包成 2 个 INT4
  - 按 group 查 scale，反量化到 FP32
  - 再走已有 `linear`

关键点：

- `U8` 作为 INT4 packed 存储类型
- group_size 由 `scale` 形状自动推导，避免新增对外接口

### 3.3 GPTQ/AWQ 兼容路径（加载时转换）

加载阶段识别到 GPTQ/AWQ 后，将其 `qweight/qzeros/scales` 转为内部 INT4 格式：

1. 解包 `int32` 中的 4-bit 权重
2. 反量化回 FP32
3. 重新量化为内部 symmetric INT4
4. 打包为 `uint8` + 生成内部 `scale`

注意：

- 增加了 **zero-point +1 修正**（兼容常见 AutoGPTQ 存储约定）
- 这样 C++ 推理侧无需新增 GPTQ/AWQ 专用 op

---

## 4. 关键代码位置

- 量化脚本：`scripts/quantize.py`
- 模型加载与 GPTQ/AWQ 转换：`python/llaisys/models/qwen2.py`
- 推理服务 tokenizer 路径修复：`python/server/app.py`
- 量化推理路由：`src/llaisys/models/qwen2.cpp`
- INT8/INT4 dequantize 入口：`src/ops/dequantize/op.cpp`
- INT4 CUDA kernel：`src/ops/dequantize/nvidia/dequantize_nvidia.cu`

---

## 5. 踩坑与修复

### 5.1 dequant cache 键冲突导致非法访存

问题：早期把 dequant 缓冲区按 `numel` 做 key，不同 shape 但同元素数的矩阵会复用同一 buffer，导致 CUDA illegal memory access。  
修复：改为按 `(rows, cols)` 作为 key。

### 5.2 GPTQ 转换后输出乱码

问题：未做 zero-point +1 修正，反量化偏移导致语义明显劣化。  
修复：`z = z + 1` 后再执行反量化，输出恢复可读。

### 5.3 tokenizer 在线依赖导致离线不稳

问题：服务端曾硬编码从远端 repo 拉 tokenizer，网络波动会阻塞启动。  
修复：改为从 `resolved model path` 加载 tokenizer，支持本地快照与离线运行。

---

## 6. 复现命令

### 6.1 构建

```bash
cd /home/bbq/llaisys
xmake f --nv-gpu=y -y
xmake build -y
xmake install -y
```

### 6.2 INT8 量化

```bash
python3 scripts/quantize.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./quantized_model \
  --bits 8
```

### 6.3 INT4 量化

```bash
python3 scripts/quantize.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./quantized_model_int4 \
  --bits 4 --group-size 128
```

### 6.4 启动服务（INT4）

```bash
cd python
python3 -m server.app \
  --model /home/bbq/llaisys/quantized_model_int4 \
  --device nvidia --port 8000
```

### 6.5 启动服务（GPTQ/AWQ）

```bash
cd python
python3 -m server.app \
  --model <hf_repo_or_local_snapshot> \
  --device nvidia --port 8000
```

---

## 7. 当前已知边界

- GPTQ/AWQ 目前是“加载时转换”，首启有额外 CPU 时间
- 尚未做“转换结果落盘缓存”（可作为下一步优化）
- INT4 kernel 目前完成第一轮优化，后续可继续做向量化读取与 block 参数调优

---

## 8. AWQ 大模型适配记录（2026-03-06）

### 8.1 配置驱动改造

原代码中 `Qwen2.__init__` 将模型架构参数（层数、隐藏维度、注意力头数等）硬编码为 DeepSeek-R1-Distill-Qwen-1.5B 的值。已改为**从 `config.json` 自动读取**，保留硬编码值作为回退默认值。

改动文件：`python/llaisys/models/qwen2.py`

映射关系：

| HF config.json 字段 | Meta 字段 | 1.5B 默认值 |
|---|---|---|
| `num_hidden_layers` | `nlayer` | 28 |
| `hidden_size` | `hs` | 1536 |
| `num_attention_heads` | `nh` | 12 |
| `num_key_value_heads` | `nkvh` | 2 |
| `hidden_size / num_attention_heads` | `dh` | 128（自动计算） |
| `intermediate_size` | `di` | 8960 |
| `vocab_size` | `voc` | 151936 |
| `rms_norm_eps` | `epsilon` | 1e-6 |
| `rope_theta` | `theta` | 10000.0 |
| `eos_token_id` | `end_token` | 151643（支持 int 或 list 格式） |

`maxseq` 取值优先级：用户显式传入 `max_seq_len` > `sliding_window` > `max_position_embeddings`（上限截断到 32768）。

### 8.2 AWQ GEMM 格式（区别于 GPTQ）

实测 `Qwen/Qwen2-7B-Instruct-AWQ`（AutoAWQ 量化）时发现，AWQ GEMM 的 packing 轴与 GPTQ **不同**：

| | GPTQ | AWQ GEMM |
|---|---|---|
| `qweight` | `[in_features // 8, out_features]` 行打包 | `[in_features, out_features // 8]` **列打包** |
| `qzeros` | `[groups, out // 8]` | `[groups, out // 8]`（相同） |
| zero-point 修正 | 需要 +1（AutoGPTQ 约定） | **不需要**（AWQ 直接存储） |

已新增 `_convert_awq_layer()` 方法，并在加载时根据 `quant_method` 字段自动选择转换器。

### 8.3 Qwen2-7B-Instruct-AWQ 显存分析

| 组件 | 大小 (GB) | 说明 |
|---|---|---|
| Embedding | 2.18 | `[152064, 3584]` FP32 未量化 |
| LM Head | 2.18 | `[152064, 3584]` FP32 未量化 |
| INT4 权重（28 层） | 3.47 | packed U8 + scale F32 |
| KV-Cache（seq=512） | 0.06 | 2 × 28 × 512 × 4 × 128 × 4B |
| Dequant 缓存 | 0.27 | 最大层 `down_proj [3584, 18944]` |
| CUDA + buffers | 0.35 | |
| **总计** | **8.51** | **超出 8GB 显存约 0.5GB** |

**根因**：Embedding 和 lm_head **未被 AWQ 量化**，各占 2.18GB FP32。

### 8.4 待解决：OOM 优化方案

以下方案可减少显存使用：

1. **FP16 存储 Embedding/LM Head**：AWQ 模型的原始 dtype 为 FP16，当前代码强制转 FP32 造成浪费。改为 FP16 存储可节省 **2.18 GB**，但需要 C++ 侧支持 FP16 embedding lookup 和 lm_head matmul。
2. **量化 LM Head 为 INT8**：对 lm_head.weight 做 per-channel INT8 量化，节省 ~1.6GB。
3. **分层 dequantize**：避免缓存所有层的 dequant 结果，改为用完即释放。

---

## 9. AWQ 7B 推理调试全记录（2026-03-07）

本节记录从 "OOM → 加载成功但输出乱码 → 根因分析 → 方案选择" 的完整过程。

### 9.1 FP16 混合精度内核实现

为解决 embedding/lm_head 占用过多显存的问题，在 C++ 层实现了 FP16 混合精度支持：

**修改文件清单：**

| 文件 | 修改内容 |
|---|---|
| `src/ops/embedding/nvidia/embedding_nvidia.cu` | 新增 `embedding_f16_to_f32_kernel` / `embedding_bf16_to_f32_kernel`，FP16 权重 → FP32 输出 |
| `src/ops/embedding/op.cpp` | 新增 `embedding_mixed_cpu_kernel<SrcT>` CPU 回退 |
| `src/ops/linear/nvidia/linear_nvidia.cu` | 新增 FP16 权重 + FP32 输入 → FP32 输出路径：F32→F16 输入转换 + cuBLAS F16×F16→F32 GEMM + FP16 bias→F32 输出 |
| `src/ops/linear/op.cpp` | 新增 CPU 混合精度 linear 路径 |
| `src/llaisys/models/qwen2.cpp` | `linear_maybe_dequant` else 分支支持 FP16/BF16 权重透传至 `ops::linear` |

**效果**：embedding/lm_head 从 FP32 (各 2.18 GB) → FP16 (各 1.04 GB)，节省 2.28 GB。

cuBLAS 混合精度要求 A/B 矩阵类型相同，因此 linear 内核使用 **线程局部缓存** 的 FP16 buffer 将输入 F32→F16 转换，避免每次 `cudaMalloc/Free`。

### 9.2 maxseq 配置 Bug 修复

**问题**：`config.json` 中 `sliding_window: 131072`、`use_sliding_window: false`。代码未检查 `use_sliding_window` 字段，直接使用 131072 作为 maxseq → KV-cache 虚拟分配 14.3 GB → 推理极慢（50 tokens 用时 21 分钟）。

**修复**：在 `qwen2.py` 中增加 `use_sliding_window` 检查：
```python
use_sw = cfg.get("use_sliding_window", True)
sw = cfg.get("sliding_window")
if use_sw and sw is not None and isinstance(sw, int) and sw > 0:
    maxseq = sw
else:
    maxseq = min(cfg.get("max_position_embeddings", ...), 32768)
```

### 9.3 双重量化问题（根因分析）

**现象**：FP16 embed/lm_head 加载成功、maxseq 修复后，推理速度正常 (2.8 tok/s)，但输出完全乱码：
```
掂.Annotation|min踪挞.LoggerFactory interrupt...
```

**根因**：AWQ→FP32→INT4 的"双重量化"导致累积精度损失。

AWQ 使用 **非对称** 量化 (unsigned INT4 + zero_point)，而我们的内部格式是 **对称** INT4 (signed, 无 zero_point)。转换路径：

```
AWQ INT4 (非对称, unsigned 0-15, 带 zero_point)
  → FP32 反量化
    → 对称 INT4 (signed -8..7, 无 zero_point) ← 精度损失发生处
```

**数值分析**：
- 单层权重相对误差 (MAE / weight_std)：~9.4%
- 28 层累积：$(1 - 0.094)^{28} \approx 0.063$ → 仅保留 **6.3%** 信号
- 结论：**~94% 的原始权重信息被双重量化破坏**

误差来源：
1. 非对称 → 对称格式转换丢失 zero_point 信息
2. FP32 → INT4 再量化引入新的舍入误差
3. 两次 4-bit 量化的舍入误差叠加，非互相抵消

### 9.4 AWQ→FP16 直接转换方案

为消除双重量化，修改代码使 AWQ 权重在加载时直接反量化为 FP16（不再重新量化为 INT4）：

```python
# _convert_awq_layer 返回 FP16 而非 INT4
w_float = w_float.T.contiguous()  # [out, in]
return w_float.to(torch.float16)  # 直接转 FP16，无精度损失
```

**但**：FP16 权重的显存需求：

| 组件 | 大小 |
|---|---|
| 196 层 linear (FP16) | 12.15 GB |
| Embedding + LM Head (FP16) | 2.03 GB |
| KV-Cache (maxseq=2048) | 0.22 GB |
| CUDA + buffers | 0.40 GB |
| **总计** | **~14.8 GB** |

本地 RTX 4060 (8 GB) 无法运行。需要 16+ GB 显存的 GPU。

### 9.5 替代方案对比

| 方案 | 显存 | 精度 | 工作量 | 状态 |
|---|---|---|---|---|
| **双重量化 (AWQ→INT4)** | ~6.5 GB | ❌ 乱码 | 已完成 | 已弃用 |
| **AWQ→FP16 直接转换** | ~14.8 GB | ✅ 无损 | 已完成 | 代码就绪，需 16+ GB GPU |
| **原生 AWQ INT4 内核** | ~6.2 GB | ✅ 无损 | 需新增 CUDA kernel + 权重结构 | 未实现 |
| **自有 INT4 模型 (quantize.py)** | ~3.5 GB (1.5B) | ✅ 正确 | 已完成 | ✅ 正常运行 |

### 9.6 结论

1. **自有对称 INT4 格式**（通过 `quantize.py` 量化）是当前唯一可靠的量化推理路径。1.5B 模型已验证正确。
2. **AWQ/GPTQ 兼容**受限于"双重量化"问题，需要实现原生非对称 INT4 解量化内核才能在低显存 GPU 上正确推理。
3. 代码中保留了 FP16 混合精度内核和 AWQ→FP16 转换逻辑，供未来在大显存 GPU 上使用。

---

## 10. 当前支持的推理路径总结

| 模型 | 格式 | 显存 | 状态 |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B (自有 INT4) | 对称 INT4 g128 | ~3.5 GB | ✅ 正常 |
| DeepSeek-R1-Distill-Qwen-1.5B (自有 INT8) | 对称 INT8 per-channel | ~1.5 GB | ✅ 正常 |
| 任意 Qwen2 FP32/FP16 | 原始精度 | 按模型大小 | ✅ 正常 |
| Qwen2-7B-Instruct-AWQ | AWQ→FP16 | ~14.8 GB | ⚠️ 需 16+ GB GPU |
| 任意 GPTQ/AWQ 模型 | 双重量化→INT4 | 低 | ❌ 精度不可接受 |

---

## 11. 推荐下一步

1. **原生 AWQ INT4 内核**：实现 CUDA dequantize 内核直接从 AWQ 原始格式 (qweight/qzeros/scales) 在飞行中反量化，无精度损失，显存与 INT4 相当
2. 增加 GPTQ/AWQ 转换缓存文件（首次转换后落盘，后续直接加载）
3. 输出统一性能表：FP32 / INT8 / INT4（tokens/s、首 token 延迟、显存）
4. 继续优化 INT4 kernel（vectorized load、访存合并）

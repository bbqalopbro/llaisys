# Phase 7 量化精度评估与建议

## 结论

当前 `quantized_model_int4` 的 W4A16 fused GEMV **性能收益明确**，但精度评估结果不支持“量化精度好”的结论。

- INT4 平均吞吐：约 **107 tok/s**
- FP16 平均吞吐：约 **60.8 tok/s**
- INT4 加速比：约 **1.76x**
- 平均前缀一致：**10.75 token**
- 平均逐位置 token 一致率：**24.5%**
- 平均 edit similarity：**36.5%**
- 8 条 prompt 中完全一致：**0/8**

因此更准确的评价是：

> 当前 INT4 是一个有价值的性能优化原型，但量化方案偏粗糙，精度不稳定，不应直接作为高质量推理默认方案。

## 评估命令

```bash
python scripts/eval_quant_accuracy.py \
  --device nvidia \
  --max-new-tokens 50 \
  --json-out quant_eval_int4_50.json
```

## 评估口径

使用 FP16 LLAISYS 模型作为 baseline，与 INT4 LLAISYS 模型进行 greedy decode 对比。

默认 prompt 覆盖：

- 英文问答
- 技术解释
- 代码生成
- 中文解释
- 翻译
- 简单数学推理

指标包括：

- `prefix_match`：从第一个生成 token 开始连续一致的长度
- `positional_match_rate`：同一位置 token 相同的比例
- `edit_similarity`：基于 token 编辑距离的相似度
- `exact_match`：完整生成是否完全一致
- `tok/s`：端到端生成吞吐

注意：自回归生成中 token match 是严格指标，一旦早期 token 分歧，后续上下文会改变。因此 token match 低不能单独证明语义完全坏，但足以说明当前量化不能通过 correctness 对齐。

## INT4 结果

| 指标 | 结果 |
|---|---:|
| prompt 数 | 8 |
| 完全一致 | 0/8 |
| 平均前缀一致 | 10.75 tokens |
| 平均逐位置一致率 | 24.5% |
| 平均 edit similarity | 36.5% |
| FP16 平均吞吐 | 60.76 tok/s |
| INT4 平均吞吐 | 107.01 tok/s |
| 加速比 | 1.76x |

观察：

- 英文代码/技术类 prompt 有时能保持 20+ token 前缀一致。
- 中文 prompt 分歧很早，部分样例前缀一致只有 2-3 token。
- 原文档中 “What is the meaning of life?” 的 12/50 一致不是孤例，整体平均也在约 25% token match。

## INT4 Kernel 正确性验证

新增脚本：

```bash
python scripts/validate_int4_kernel.py
```

该脚本比较：

```text
linear_int4 fused W4A16
vs
dequantize_int4 -> linear
```

验证结果：

| Case | max abs diff | 结果 |
|---|---:|---|
| N=128, K=512, g=128, F16 | 5.96e-08 | PASS |
| N=384, K=1536, g=128, F16+bias | 2.44e-04 | PASS |
| N=384, K=1536, g=64, F16+residual | 0 | PASS |
| N=1024, K=8960, g=128, F16 | 3.05e-05 | PASS |
| N=4096, K=1536, g=128, F32 logits | 2.62e-06 | PASS |

结论：

> fused W4A16 kernel 与显式反量化参考路径数值一致。当前生成质量下降主要来自 naive symmetric INT4 量化方案，而不是 fused kernel 额外引入明显误差。

## 方案 C 实施验证：AWQ-like INT4

新增脚本：

```bash
python scripts/quantize_awq_like.py \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --output quantized_model_int4_awqlike \
  --group-size 128 \
  --max-length 192 \
  --device cuda:0 \
  --search-steps 11 \
  --row-chunk 128
```

该方案不是完整 AWQ/GPTQ，而是一个可落地的 AWQ-like baseline：

- 用校准 prompt 收集每个 linear 输入通道的 activation importance。
- 对每个 weight group 搜索 INT4 scale，使 activation-weighted reconstruction error 最小。
- 保持原有 LLAISYS packed INT4 格式和 W4A16 fused decode kernel。
- 默认保留 `lm_head.weight` 为 FP16，避免最终 logits 排序被 INT4 直接破坏。

生成模型：

```text
quantized_model_int4_awqlike
```

量化配置：

| 项目 | 结果 |
|---|---:|
| quantized tensors | 196 |
| kept tensors | 143 |
| calibration prompts | 8 |
| group size | 128 |
| search steps | 11 |
| 压缩比 | 2.18x |

评估命令：

```bash
python scripts/eval_quant_accuracy.py \
  --device nvidia \
  --quant-model quantized_model_int4_awqlike \
  --max-new-tokens 50 \
  --json-out quant_eval_int4_awqlike_50.json
```

结果对比：

| 指标 | naive INT4 | AWQ-like INT4 |
|---|---:|---:|
| 完全一致 | 0/8 | 1/8 |
| 平均前缀一致 | 10.75 tokens | 18.88 tokens |
| 平均逐位置一致率 | 24.5% | 38.5% |
| 平均 edit similarity | 36.5% | 55.2% |
| FP16 平均吞吐 | 60.76 tok/s | 62.27 tok/s |
| INT4 平均吞吐 | 107.01 tok/s | 97.33 tok/s |
| 加速比 | 1.76x | 1.56x |

结论：

> 方案 C 明显优于 naive INT4，说明校准信息和保留 `lm_head` FP16 是有效方向。但当前实现仍不是高质量工业 INT4：平均逐位置一致率只有 38.5%，还不能作为默认高质量推理方案。

下一步若继续优化 INT4，应优先做：

- 更大的校准集，至少覆盖中英文、代码、数学、长上下文。
- 尝试 group size 64。
- 实现真正 AWQ channel scaling，或接入 GPTQ/AWQ 现成量化参数。
- 增加 logits 级评估，而不是只看生成 token 一致率。

## INT8 结果

尝试运行：

```bash
python scripts/eval_quant_accuracy.py \
  --device nvidia \
  --quant-model quantized_model \
  --max-new-tokens 50 \
  --json-out quant_eval_int8_50.json
```

结果：运行中触发 CUDA illegal memory access。

当前 INT8 路径不建议作为可用方案评估，至少需要先修复稳定性问题。

## 当前 INT4 精度差的主要原因

1. 量化方法是普通 symmetric per-group INT4，没有校准数据，也不是 activation-aware。
2. group size 为 128，对 1.5B 小模型来说误差相对明显。
3. `lm_head.weight` 也被 INT4 量化，会直接影响最终 logits 排序。
4. token 一致率对 early divergence 很敏感，但当前前缀一致长度也不高，说明误差不是只发生在长尾。

## 推荐量化方案

### 方案 A：默认高质量方案

**FP16 权重 + FP16 KV cache**

适合：

- 正确性优先
- 对输出质量敏感
- 用作 baseline

这是当前项目最稳妥的默认推理精度。

### 方案 B：显存受限的实用方案

**W4A16，但保留部分敏感层为 FP16**

建议：

- `lm_head.weight` 保持 FP16
- `embed_tokens.weight` 保持 FP16
- norm/bias 保持 FP16/FP32
- attention/MLP linear 层使用 INT4
- 重新评估 token match、edit similarity、人工质量

这通常比“全 linear 包括 lm_head 都 INT4”更稳，显存稍高但 logits 排序更可靠。

### 方案 C：更合理的 INT4 方案

**AWQ/GPTQ 校准量化 + W4A16 fused kernel**

建议目标：

- 用校准集做 activation-aware scale
- 支持 per-group INT4，但避免简单 max-abs symmetric 作为最终方案
- 优先复用 AWQ/GPTQ 的 scale/zero-point，而不是反量化后再二次量化
- 对 decode 保留 fused W4A16 路径

这是更接近工业实践的路线。

### 方案 D：INT8 暂不推荐

当前 INT8 有两个问题：

- 性能上已有报告显示慢于 FP16
- 本次评估触发 CUDA illegal memory access

除非实现 W8A16 fused GEMV 或 cuBLASLt/CUTLASS INT8 路径，否则 INT8 不是优先方向。

## 建议验收标准

下一阶段不要再用单 prompt token 一致率作为结论，建议至少满足：

- prompt 数 >= 100
- greedy decode
- 平均前缀一致长度
- 平均 token position match
- edit similarity
- 中英文分别统计
- 代码/数学/解释类任务分别统计
- 对代表性样例做人类阅读检查

更进一步应增加 logits 级评估：

- top-1 一致率
- top-5 overlap
- KL divergence
- max/mean absolute logit error
- 小型语料 PPL

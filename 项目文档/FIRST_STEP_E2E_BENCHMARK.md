# 第一阶段端到端推理与 TTFT/TPOT 基线

## 测试环境

- 日期：2026-08-27
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 显存：8188 MiB
- Compute Capability：8.9
- Driver：610.88
- 模型：`models/DeepSeek-R1-Distill-Qwen-1.5B`
- 模型结构：28 layers、hidden size 1536、12 query heads、2 KV heads
- 执行精度：llaisys NVIDIA FP16 路径
- Python/C++ 边界：pybind11 `Qwen2BatchRuntime`
- sampling：greedy，`temperature=0`、`top_k=1`、`top_p=1`
- TPOT repeat：每种 prompt 运行 3 次，每次 decode 20 tokens

本报告是当前 4060 单卡的功能和性能基线，不代表 B300、多卡、MLA 或 DeepSeek V4 Flash 的最终性能。

## 正确性

### Hugging Face BF16 对齐

命令：

```bash
PYTHONPATH=python python3 test/test_infer.py \
  --device nvidia \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompt "What is 2+3?" \
  --max_steps 8 \
  --test
```

Hugging Face BF16 与 llaisys single-model 的完整 token 序列完全一致，测试输出 `Test passed!`。

### 新 BatchContext 和 InferenceEngine 对齐

真实 chat-template prompt 共 13 tokens。新 BatchContext/pybind 路径生成：

```text
[40, 1184, 311, 11047, 279, 2629, 315, 220]
```

解码文本：

```text
I need to calculate the sum of
```

该序列与已经通过 Hugging Face BF16 对齐的 single-model greedy 结果完全一致。完整 `InferenceEngine` worker/stream 路径也生成同一序列。

## 模型加载与冷启动

| 指标 | 结果 |
|---|---:|
| 模型加载时间 | 1.907 s |
| 模型加载后首次 BatchContext TTFT，13-token prompt | 476.710 ms |

首次 TTFT 包含 CUDA/cuBLAS lazy initialization 和首次 kernel 路径启动。不同独立进程中测得过约 247 ms，因此该数值波动较大，不应当作为稳态服务 SLO。

## Runtime warm TTFT 与 TPOT

| Prompt | Warm TTFT mean | Warm TTFT min | TPOT mean | TPOT P95 | Steady TPOT mean | Steady TPOT P95 | Steady decode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 tokens | 35.306 ms | 35.248 ms | 23.505 ms | 24.391 ms | 23.461 ms | 24.203 ms | 42.62 tok/s |
| 256 tokens | 120.264 ms | 119.205 ms | 33.735 ms | 35.010 ms | 33.647 ms | 34.327 ms | 29.72 tok/s |

`Steady TPOT` 对每轮 decode 排除了前两个 step。随着 KV sequence length 从 32 增长到 256，Paged Attention 读取的历史 KV 增多，TPOT 从约 23.46 ms 增长到约 33.65 ms。

## 完整 InferenceEngine 端到端结果

测试范围包含：

```text
request submit
  -> worker admission
  -> BatchContext prefill
  -> pybind11
  -> C++ Qwen2 runtime
  -> per-request decode
  -> asyncio token stream
  -> request future completion
```

13-token真实 prompt、8 个输出 token：

| 指标 | 结果 |
|---|---:|
| Scheduler E2E TTFT | 30.911 ms |
| Scheduler E2E TPOT mean | 22.359 ms |
| Scheduler E2E TPOT P95 | 23.504 ms |
| Scheduler E2E total latency | 187.424 ms |
| 输出与 greedy reference 一致 | 是 |

该测试在模型和 CUDA kernel 已经 warm 的进程后半段运行，所以它表示 warm serving latency，不是进程冷启动延迟。

## Block Prefix Cache

使用 257-token prompt，先完整 prefill 并发布 cache，然后在新 slot 查询相同 prompt：

| 指标 | 结果 |
|---|---:|
| Prompt tokens | 257 |
| 命中 tokens | 256 |
| 完整 prefill TTFT | 155.025 ms |
| Prefix hit TTFT | 35.667 ms |
| TTFT speedup | 4.35x |
| 首 token 与完整 prefill 一致 | 是 |

结果验证了：

- Prefix Cache 命中真实共享了 16-token 粒度的物理 blocks；
- 最后一个 prompt token 留在执行路径中以重新得到 logits；
- Prefix Cache 没有改变 greedy 输出；
- 对接近完整 block 边界的长共享前缀，TTFT 有明显收益。

## Chunked prefill 优化

### 原始基线

原 correctness-first 实现对 chunk 中每个 prompt token 执行完整 decode，包括 28 层前向、vocab lm_head、argmax 和 D2H。256-token prompt 实测：

| 指标 | 原始结果 |
|---|---:|
| 完整并行 prefill TTFT | 119.690 ms |
| 逐 token chunked prefill TTFT | 7010.737 ms |
| Slowdown | 58.57x |

### 生产路径：direct paged prefill

当前 NVIDIA 默认路径已改为 FlashInfer `BatchPrefillWithPagedKVCache`：

- 每个 chunk 批量执行 embedding/QKV/RoPE/MLP；
- 当前 chunk K/V 写入 block pool 后，attention kernel 通过 CSR page metadata 直接读取 paged KV；
- plan 和 `indices/kv_indptr/qo_indptr/last_page_len` 每个 chunk 准备一次，28 层复用；
- 不再为历史 prefix 分配并 gather 连续 K/V workspace；
- gather+GEMM 保留为编译时或运行时 fallback。

512-token prompt、256+256 chunk 的对照测试使用同一份权重和 benchmark：

| 路径 | Chunk 对照前的 full prefill | Chunked prefill | Chunk/full | 首 token |
|---|---:|---:|---:|---:|
| FlashInfer direct paged | 79.678 ms | 84.115 ms | 1.056x | 198 |
| gather+GEMM fallback | 94.807 ms | 108.910 ms | 1.149x | 198 |

direct paged 相对 fallback，这组紧邻 full prefill 延迟降低约 16.0%，chunked prefill 延迟降低约 22.8%。3 次 full prefill 中排除首次后的 warm mean 分别为 77.370 ms 和 94.131 ms，降低约 17.8%。这一次测试通过 `LLAISYS_REQUIRE_PAGED_PREFILL=1` 强制 direct backend；如果 shape/dtype 不受支持会立即失败，不会静默把 fallback 结果记为 direct。

非 block 对齐的 300-token、100+100+100 场景也已以强制 direct 路径验证：

| 指标 | 结果 |
|---|---:|
| Full prefill | 52.838 ms |
| Chunked prefill | 76.408 ms |
| 首 token | 两者均为 275 |
| Prefix Cache | 301 tokens 中命中 288 tokens |
| Prefix hit TTFT | 18.699 ms，首 token 一致 |

完整 Python `InferenceEngine` 对 512-token prompt 实际走 256+256 direct paged chunk：

| 指标 | 结果 |
|---|---:|
| Scheduler long-prompt TTFT | 98.556 ms |
| TPOT mean | 27.066 ms |
| 3-token E2E latency | 152.689 ms |
| 生成 tokens | `[198, 220, 16]` |
| 与完整 prefill + decode 序列一致 | 是 |

注：RTX 4060 Laptop 的独立进程测量会受 GPU 频率和首次 kernel 初始化影响，该结果用于当前功能/性能回归，不外推 B300 性能。

## 复现命令

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=python \
python3 test/bench_e2e_serving.py \
  --device nvidia \
  --lengths 32,256 \
  --decode-tokens 20 \
  --repeat 3 \
  --chunk-size 256 \
  --prefix-length 257 \
  --correctness-tokens 8 \
  --paged-prefill require \
  --json-output /tmp/llaisys_e2e_final.json
```

benchmark 脚本会自动检查：

- BatchContext 与 single-model greedy token 一致；
- InferenceEngine 与 greedy reference 一致；
- full prefill 与 chunked prefill 的首 token 一致；
- full prefill 与 Prefix Cache hit 的首 token 一致；
- pybind11 runtime 实际启用。

复现跨 chunk 长 prompt 与完整 scheduler 路径：

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=python \
python3 test/bench_e2e_serving.py \
  --device nvidia \
  --lengths 512 \
  --decode-tokens 8 \
  --repeat 2 \
  --chunk-size 256 \
  --prefix-length 513 \
  --correctness-tokens 4 \
  --paged-prefill require \
  --scheduler-long
```

`--paged-prefill auto` 是默认生产选择；`require` 用于 CI/回归测试；`fallback` 显式禁用 FlashInfer prefill，用于 A/B 对照。

## 下一步性能优先级

1. 让 scheduler/cache manager 直接维护 device-side CSR metadata，取消每个 chunk 的 host dense-to-CSR 转换和 H2D。
2. 把多请求 prefill/decode 整合成真正 mixed batch，而不是每次 C++ prefill 仅处理一个 slot。
3. 将 Python worker 每轮操作完全收敛成一个 `SchedulePlan`，减少 Python/C++ 边界调用。
4. benchmark chunked prefill 与 decode 混合 workload，重点看 P95/P99 TPOT、KV utilization 和 prefix hit rate。
5. 到 B300 环境后接入 Blackwell/FlashMLA、FP8、TP/EP、NVLink/NVSwitch，并重测多卡 tail latency。

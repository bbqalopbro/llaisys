# LLAISYS 第一阶段工程改造完整记录

## 1. 文档用途

本文记录 LLAISYS 从 Qwen2 单模型原型向可扩展 serving runtime 改造的完整过程，重点包括：

- Python 调度层和 C++ runtime 的责任边界；
- ctypes 主链路向 pybind11 batch runtime 迁移；
- model、backend、block manager 和 attention payload 解耦；
- chunked prefill 从 correctness-first 原型到 direct paged FlashInfer 的三次迭代；
- block-level Prefix Cache 与 chunked prefill 的调度协作；
- 冗余代码清理、构建矩阵和端到端验证；
- 第一阶段和 B300/DeepSeek 适配之间的真实边界。

这份文档可以作为项目答辩、技术分享、面试介绍和后续开发交接材料。

## 2. 第一阶段的出发点

### 2.1 最终目标

项目的长期目标是在 B300 多卡服务器上适配 DeepSeek 模型，并进一步支持：

```text
MLA attention
+ MoE
+ TP / EP
+ FP8 / Blackwell kernels
+ continuous batching
+ chunked prefill
+ Prefix Cache
```

但是在直接开始 DeepSeek 权重映射之前，旧代码存在三个结构性问题：

1. Python 和 C++ 都包含部分调度逻辑，责任边界不够清晰。
2. Qwen2 的 Standard K/V 格式、block 分配和 attention kernel 耦合，不适合直接扩展 MLA latent cache。
3. 旧 prefill 和 Prefix Cache 更像单请求功能原型，没有成为 serving scheduler 可使用的基础能力。

因此，第一阶段不是立即实现 DeepSeek，而是把不应该由 DeepSeek 重新实现的基础能力先收敛稳定。

### 2.2 阶段内与阶段外

| 第一阶段完成 | 暂不在第一阶段完成 |
|---|---|
| Python/C++ 调度边界 | DeepSeek 权重映射 |
| pybind11 batch runtime | MLA/FlashMLA kernel |
| BlockManager 与 cache layout 解耦 | MoE router/expert kernel |
| chunked prefill | TP+EP 通信闭环 |
| block Prefix Cache | B300 实机调优 |
| direct paged prefill | FP8/FP4 生产精度路径 |
| TTFT/TPOT/E2E benchmark | 多机容错与生产 API |

## 3. 架构决策：Python 管策略，C++ 管执行

### 3.1 最终责任边界

```text
HTTP / Python API
        |
        v
Python InferenceEngine
  - request queue
  - WAITING/PREFILLING/DECODING state
  - admission / token budget / KV budget
  - decode priority
  - chunk boundary
  - preemption policy
        |
        | SchedulePlan / pybind11 runtime methods
        v
C++ Runtime
  - model forward
  - tensor/device memory
  - block lifecycle
  - paged cache storage
  - attention backend dispatch
  - CUDA kernels / collectives
```

这个边界的原因是：

- 请求队列、公平性、优先级和 SLO 策略迭代快，适合 Python；
- tensor、device pointer、block pool 和 kernel launch 需要稳定 ABI 和较小开销，适合 C++；
- DeepSeek 接入时可以换 ModelRunner 和 attention backend，不需要重写 Python 服务层。

### 3.2 为什么不继续用 ctypes 做 batch 主链路

原 ctypes 路径需要手工组装：

- token pointer；
- slot id 数组；
- sampling parameter 数组；
- snapshot 裸句柄；
- C 函数的 argtypes/restype。

这会带来所有权不清晰、异常丢失和复杂数组易配错等问题。改造后：

- batch runtime 优先使用 pybind11；
- C++ 对 BatchContext 使用 RAII；
- 计算时释放 GIL；
- C++ 异常可以变成 Python 异常；
- `SchedulePlan/StepResult` 可以作为稳定数据合同。

当前模型创建和权重装载仍保留部分 ctypes C API，因为它们不在 per-token 热路径上，且一次性改写权重量化/TP 装载会放大回归范围。这是有意的渐进迁移，不是同时维护两套调度主链路。

## 4. Block 管理与 attention payload 解耦

### 4.1 旧问题

旧 `BlockAllocator` 同时承担：

```text
physical id allocation
+ K/V pool allocation
+ Standard KV stride assumption
+ block free list
```

这意味着如果 DeepSeek MLA 只缓存 latent 和 rope component，就必须修改 block allocator，从而影响 Qwen 和 Prefix Cache。

### 4.2 改造后的组合

```text
BlockManager
  - allocate / retain / release
  - refcount
  - computed/cached state
  - prefix hash
  - LRU eviction

PagedCacheStorage
  - allocate payload components from CacheLayout

CacheLayout
  - StandardKV: key + value
  - MLA: latent + optional rope
```

可以复用给 MLA 的部分：

- block id；
- PageTable；
- refcount；
- hash 链；
- cached/computed 状态；
- LRU 淘汰。

不可以直接复用的部分：

- Qwen K/V 双 pool 的 stride；
- StandardKV attention kernel；
- Qwen head/GQA 假设；
- FlashInfer Standard KV descriptor。

所以 DeepSeek 接入时应增加 `MLACacheLayout + MLA attention backend`，而不是修改 `BlockManager`。

## 5. Chunked prefill 三次迭代

### 5.1 语义先定义

对 prompt `[0, S)` 和 chunk size `C`：

```text
chunk 0: query [0, C),     KV [0, C)
chunk 1: query [C, 2C),    KV [0, 2C)
chunk 2: query [2C, 3C),   KV [0, 3C)
...
```

每个 query token 需要：

- 看到所有历史 prefix K/V；
- 看到当前 chunk 中自己之前的 token；
- 不能看到当前 chunk 中未来 token；
- 将当前 chunk K/V 追加到同一份 paged cache。

`start_pos` 必须等于 slot 已计算 token 数，从而防止 chunk 重复、跳过或覆盖 cache。

### 5.2 迭代 A：逐 token decode 模拟 prefill

最早的 correctness-first 实现将 chunk 中每个 prompt token 当作 decode token：

```text
for token in chunk:
    run 28 transformer layers
    run final norm
    run vocab lm_head
    run argmax/sample
    copy result D2H
```

它的优点是容易证明 KV 语义正确，但是重复执行了不必要的 lm_head、sampling 和 host synchronization，也完全没有利用 prefill 的 token 并行性。

256-token 实测：

| 路径 | TTFT |
|---|---:|
| 完整并行 prefill | 119.690 ms |
| 逐 token chunked prefill | 7010.737 ms |
| slowdown | 58.57x |

结论：这只能作为短期 correctness oracle，不能作为 serving 路径。

### 5.3 迭代 B：multi-token + gather/GEMM

第二版将每个 chunk 改为 `[chunk_tokens, hidden]` 批量前向：

- embedding/QKV/RoPE/MLP 批量执行；
- 中间 chunk 跳过 final norm/lm_head/sample/D2H；
- 当前 K/V scatter 到 block pool；
- 将 prefix+chunk K/V gather 成连续视图；
- 复用已有 batched-GEMM causal self-attention。

这一版将逐 token 路径的数十倍慢收敛到接近 full prefill，并建立了正确的 incremental chunk API。

但它仍然存在工程问题：

- 历史 K/V 在 block pool 中已经存在，又被复制到临时连续 buffer；
- prefix 越长，gather 显存带宽开销越大；
- workspace 随序列长度增长；
- 与 vLLM/SGLang 主流的 direct paged prefill 实现不一致。

因此将它保留为无 FlashInfer 时的 fallback，但不再作为 NVIDIA 默认路径。

### 5.4 迭代 C：FlashInfer direct paged prefill

最终 NVIDIA 默认路径为：

```text
Python scheduler chooses chunk
        |
        v
C++ computes Q/K/V for the chunk
        |
        +-- scatter K/V into authoritative block pool
        |
        +-- dense PageTable -> CSR page metadata
        |
        +-- FlashInfer PrefillPlan once per chunk
        |
        `-- BatchPrefillWithPagedKVCache for every layer
```

FlashInfer 元数据包括：

```text
indices        valid physical block ids
kv_indptr      page ranges of each sequence
qo_indptr      query ranges of each sequence
last_page_len  valid tokens in the final page
```

每个 chunk 只进行一次 dense-to-CSR 转换和 plan，所有 transformer layer 复用 workspace/metadata。Attention kernel 直接从 block pool 读取 K/V，不再构造连续历史 K/V。

### 5.5 关键踩坑：GQA group-size=6 静默 fallback

Qwen2-1.5B 的 attention shape 是：

```text
query heads = 12
KV heads    = 2
group size  = 6
head dim    = 128
```

初版 support check 误用了 decode kernel 的特化集合：

```text
group size in {1, 2, 4, 8}
```

因此所谓的 direct benchmark 其实静默走了 gather fallback。经过检查 FlashInfer prefill 参数后确认，prefill 的 group size 是运行时 fast-divisor，不受 decode 特化集合限制。

最终修改：

- prefill 支持任意可整除的 GQA group size；
- decode 仍保留自己的 specialization 限制；
- 增加 `LLAISYS_REQUIRE_PAGED_PREFILL=1`，没有真正进入 direct backend 时直接失败；
- 增加 `LLAISYS_DISABLE_FLASHINFER_PREFILL=1`，用于显式 fallback A/B。

这个问题的工程启示是：存在 fallback 时，“输出正确”不能证明“优化路径被执行”，必须有可强制、可观测的 backend 选择。

## 6. Prefix Cache 与 chunked prefill 的结合

### 6.1 为什么不使用 token Trie + CPU KV snapshot

旧方案按 token Trie 匹配，每个前缀节点持有一份 CPU KV 深拷贝。它的问题是：

- KV 在 GPU→CPU→GPU 间复制；
- 同一前缀可能重复占用大量 host memory；
- 不能和 block refcount/LRU 统一；
- 多卡下快照所有权和 rank placement 更复杂；
- 旧快照不理解 incremental chunk position，scheduler 不得不回退 full prefill。

最终旧 Trie prefix pool 的 C++ 结构、C API、ctypes 包装、Python 方法和 scheduler 兼容分支全部删除。

CacheSnapshot 本身没有删除，因为它仍用于：

- request preemption 恢复；
- session 状态保存。

即“删除 snapshot prefix cache”不等于“删除所有 snapshot”。

### 6.2 Block Prefix Cache 语义

```text
block_hash = hash(parent_hash, block_tokens, cache_salt)
```

只发布完整且已计算的 block：

- 命中时 retain 物理 block；
- 新 slot 直接 attach block id；
- 请求释放时 decrement refcount；
- `refcount == 0` 的 cached block 可以 LRU 淘汰；
- partial block 不发布，避免不完整 payload 被共享。

查找时不会把 prompt 最后一个 token 完全吃掉，因为 KV cache 只保存 attention K/V，不保存该 token 对应的输出 logits。至少保留末尾 token 经过 prefill，才能产生正确首个 output token。

### 6.3 Scheduler 组合

```text
request admitted
    |
    +-- prefix_lookup -> matched complete blocks
    |
    +-- start_pos = matched_tokens
    |
    +-- chunked prefill only for unmatched suffix
    |
    `-- prefix_publish after successful prefill
```

## 7. Attention backend 隔离

Qwen2 runner 只调用通用接口：

```cpp
paged_prefill_workspace_create(...)
paged_prefill_supported(...)
paged_prefill_prepare(...)
paged_prefill_run(...)
```

它不直接 include FlashInfer/NVIDIA adapter。当前 dispatcher 映射为：

```text
NVIDIA + supported FP16/BF16 shape
    -> FlashInfer direct paged prefill

NVIDIA without FlashInfer / disabled / unsupported
    -> contiguous first chunk or gather+GEMM suffix fallback

CPU
    -> native correctness fallback
```

未来接入 FlashMLA 时，应由 DeepSeek runner 选择 MLA backend contract，不应让 StandardKV `paged_prefill_run` 强行解析 latent cache。

## 8. 构建系统改造

### 8.1 FlashInfer 单独 target

vendored FlashInfer 头文件在项目全局 `-Werror` 下会触发第三方模板 warning。因此 adapter 使用独立静态 target：

```text
llaisys-flashinfer-nvidia
        |
        v
llaisys-ops-nvidia
        |
        v
llaisys
```

这样可以：

- 仅对第三方 adapter 隔离 warning policy；
- 保持项目自己的 CUDA 代码 `warnings as errors`；
- 避免 adapter 被 glob 重复编译；
- `--flashinfer=n` 时完整移除该 target。

### 8.2 CUDA architecture 可配置

原 NVIDIA target 写死 `-arch=sm_80`。现在改为：

```bash
--cuda-arch=<target architecture>
```

当前 4060 回归配置保持 `sm_80`。到 B300 服务器后应根据当地 driver/CUDA toolkit/GPU 查询结果设置对应架构，而不要在文档中猜测并写死。

## 9. 冗余代码清理

### 9.1 已删除

- `TrieNode` 和 `LlaisysKVCachePool`；
- `llaisysKVCachePoolCreate/Destroy/Insert/Lookup/Clear` C API；
- 对应 ctypes 函数签名和导出；
- Qwen2 Python 层 `create_cache_pool/cache_pool_*`；
- `enable_legacy_prefix_cache` 和 scheduler 的旧 prefix snapshot 分支；
- scheduler 中已无用的 `remaining_ids` 临时变量；
- test/benchmark mock 中已无用的 `create_cache_pool`；
- 根目录误提交的空文件 `float`；
- 根目录误命名且包含本地 Git 配置的文件。

### 9.2 经审计后保留

| 代码 | 保留原因 |
|---|---|
| CacheSnapshot | 抢占恢复和 session 仍在使用 |
| gather+GEMM | 无 FlashInfer/不支持 shape 的正确性 fallback |
| native paged attention | CPU、教学、fallback 和对照测试 |
| INT8/INT4 | 存在独立量化权重、kernel 和测试，不是死代码 |
| ctypes model loading | 非热路径，仍承担权重装载/TP 兼容 |
| C Batch API | pybind11 底层仍需要稳定 C/C++ 入口 |

删除冗余代码的标准不是“不在默认路径”，而是“已被替代且没有独立兼容/降级价值”。

## 10. 性能与正确性结果

### 10.1 测试环境

- NVIDIA GeForce RTX 4060 Laptop GPU；
- DeepSeek-R1-Distill-Qwen-1.5B；
- 28 layers；
- hidden size 1536；
- 12 query heads / 2 KV heads / head dim 128；
- FP16 runtime；
- pybind11 BatchContext；
- greedy sampling。

这些数据是当前单卡功能/性能回归基线，不能外推 B300 性能。

### 10.2 Direct paged 与 gather fallback

512-token prompt，256+256 chunk：

| 路径 | 紧邻 full prefill | chunked prefill | chunk/full | 首 token |
|---|---:|---:|---:|---:|
| FlashInfer direct paged | 79.678 ms | 84.115 ms | 1.056x | 198 |
| gather+GEMM fallback | 94.807 ms | 108.910 ms | 1.149x | 198 |

结果：

- direct paged chunk 相对 gather fallback 降低约 22.8% 延迟；
- direct paged chunk 相对 direct full prefill 增加约 5.6% 开销；
- direct/fallback 输出一致；
- 3 次 full prefill 排除首轮的 warm mean 为 77.370 ms 对 94.131 ms。

### 10.3 非 block 对齐测试

300-token prompt，100+100+100 chunk，block size 16：

| 指标 | 第一次 direct 验证 | 冗余清理后回归 |
|---|---:|---:|
| Full prefill | 52.838 ms | 57.099 ms |
| Chunked prefill | 76.408 ms | 74.808 ms |
| Full/chunk 首 token | 275 | 275 |
| Prefix matched | 288/301 | 288/301 |
| Prefix hit TTFT | 18.699 ms | 16.578 ms |
| Scheduler output | `[275, 264]` | `[275, 264]` |

这个 case 覆盖：

- chunk 从 partial block 中间开始/结束；
- `last_page_len`；
- Prefix Cache 只命中完整 block；
- 命中后 unmatched suffix 继续 chunked prefill；
- scheduler 端到端 token 一致。

三个 100-token chunk 相对单次 full prefill 仍有明显固定开销。Chunked prefill 的主要目标是让长 prompt 可被调度器切分，在 chunk 之间保护其他请求的 TPOT/tail latency，而不是保证单请求的任意小 chunk 都比 full prefill 快。

### 10.4 Prefix Cache

513-token prompt：

| 指标 | 结果 |
|---|---:|
| matched tokens | 512 |
| full prefill | 85.997 ms |
| prefix hit TTFT | 15.668 ms |
| speedup | 5.49x |
| 首 token | 一致 |

### 10.5 Scheduler 长 prompt

512-token prompt，256+256 direct paged chunk：

| 指标 | 结果 |
|---|---:|
| TTFT | 98.556 ms |
| TPOT mean | 27.066 ms |
| 3-token E2E | 152.689 ms |
| output | `[198, 220, 16]` |
| matches reference | 是 |

TPOT 样本数较少，这里用作功能回归，不作为最终 serving capacity 结论。

## 11. 验证矩阵

| 验证项 | 结果 |
|---|---|
| NVIDIA + FlashInfer + pybind11 build | 通过 |
| NVIDIA + `flashinfer=n` fallback build | 通过 |
| CPU-only build | 通过 |
| cache core test | `cache core tests passed` |
| Python scheduler tests | 7 passed |
| Python py_compile | 通过 |
| `git diff --check` | 通过 |
| single-model vs BatchContext greedy | 一致 |
| full vs chunked first token | 一致 |
| direct vs fallback first token | 一致 |
| Prefix hit vs full first token | 一致 |
| non-aligned partial block | 通过 |
| Python InferenceEngine E2E | 通过 |
| `LLAISYS_REQUIRE_PAGED_PREFILL=1` | 确认 direct backend 被执行 |

## 12. 复现命令

### 12.1 NVIDIA + FlashInfer + pybind11

```bash
xmake f \
  --nv-gpu=y \
  --cuda-arch=sm_80 \
  --flashinfer=y \
  --python-bindings=y \
  --python-include=/usr/include/python3.10 \
  --pybind11-include=/path/to/pybind11/include

xmake build llaisys-python
```

### 12.2 强制 direct paged 端到端测试

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

### 12.3 Fallback A/B

```bash
PYTHONPATH=python python3 test/bench_e2e_serving.py \
  --device nvidia \
  --lengths 512 \
  --chunk-size 256 \
  --paged-prefill fallback
```

### 12.4 Core/scheduler tests

```bash
xmake build llaisys-cache-core-test
xmake run llaisys-cache-core-test
PYTHONPATH=python pytest -q test/test_scheduler.py
```

## 13. 当前边界：为什么能上 B300 开始 DeepSeek，但还不能直接跑

第一阶段完成后，可以直接在 B300 服务器开始 DeepSeek 适配，因为以下基础能力已经不需要重写：

- Python request scheduler；
- chunk boundary/state progression；
- BlockManager/PageTable/refcount/LRU；
- Prefix Cache hash 语义；
- pybind11 runtime 边界；
- backend dispatch 框架；
- TTFT/TPOT/correctness benchmark 框架。

但是 B300 上仍必须实现：

1. DeepSeek config 解析和权重映射；
2. DeepSeek ModelRunner；
3. MLA latent/rope CacheLayout；
4. FlashMLA 或等价 Blackwell MLA prefill/decode backend；
5. MoE router、top-k expert selection、expert GEMM；
6. TP/EP rank topology 和 expert placement；
7. NCCL all-reduce/all-to-all 执行闭环；
8. FP8/FP4/BF16 精度策略；
9. B300 目标架构编译与 kernel profiling；
10. 多卡 correctness、TTFT、TPOT、throughput 和 P95/P99 验证。

## 14. 下一阶段建议顺序

### Step 1：B300 环境基线

- 记录 GPU/driver/CUDA/NCCL/topology；
- 确定 `--cuda-arch`；
- 先让现有 Qwen correctness benchmark 在 B300 单卡通过；
- 再进行多卡 NCCL smoke test。

### Step 2：DeepSeek dense skeleton

- config/weight loader；
- ModelRunner 骨架；
- RMSNorm/embedding/lm head；
- 先不启用 MoE 优化，建立可对齐的前向骨架。

### Step 3：MLA

- `MLACacheLayout`；
- latent/rope scatter；
- MLA prefill/decode backend；
- chunked prefill + Prefix Cache correctness。

### Step 4：MoE 与 EP

- 先单卡 router/expert correctness；
- 再进行 expert placement；
- 接 all-to-all；
- 检查 load balance、capacity 和 tail latency。

### Step 5：mixed batch 与 metadata 优化

- 将多请求 prefill/decode 合并为 GPU mixed batch；
- scheduler/cache manager 直接维护 device-side CSR metadata；
- 移除每 chunk host dense-to-CSR/H2D；
- workspace 从 per-context 改成 backend/runtime 共享池。

## 15. 项目介绍口径

### 15.1 30 秒版

> 我把一个 Qwen2 单模型推理原型改造成了可扩展 serving core。Python 负责 continuous batching 和 chunk 策略，C++ 负责 runtime、block cache 和 kernel，热路径从 ctypes 迁到 pybind11。Cache 管理与 Standard KV/MLA 数据格式解耦，实现了 block-level Prefix Cache 和 chunked prefill。Chunked prefill 从逐 token 的 58.57 倍慢原型，迭代到 multi-token fallback，最后接入 FlashInfer direct paged prefill，512-token 场景比 gather fallback 降低约 22.8% 延迟。这套边界现在可以直接承接 B300 上的 DeepSeek MLA/MoE/EP 适配。

### 15.2 2 分钟版的主线

1. 先讲问题：调度与 runtime 混在一起，Qwen KV 格式侵入 block allocator，prefill 不能支持真正在线调度。
2. 再讲边界：Python 管策略，C++ 管资源和执行，pybind11 作为热路径合同。
3. 讲 cache 解耦：BlockManager 不理解 K/V 或 MLA，CacheLayout 描述 payload。
4. 讲 chunk 迭代：逐 token 确保正确性→multi-token gather 收敛性能→FlashInfer direct paged 对齐主流方案。
5. 讲一个踩坑：Qwen GQA group-size=6 被 support matrix 静默 fallback，所以增加 require-mode 保证 benchmark 真正执行目标 backend。
6. 最后讲边界：现在可以开始 DeepSeek，但 MLA/MoE/EP 还需要在 B300 上实现和验证。

### 15.3 容易被追问的问题

#### Q：为什么 chunked prefill 单请求不一定比 full prefill 快？

因为 chunk 会增加多次 kernel launch、plan 和 QKV/MLP 调度固定开销。它的 serving 价值是打断长 prompt，允许 decode 在 chunk 之间获得调度，改善混合负载的 TPOT 和 P95/P99，而不是单请求极限 TTFT。

#### Q：为什么 gather+GEMM 还没删？

它是无 FlashInfer、CPU 或不支持 shape/dtype 的 correctness fallback。主路径已经 direct paged，但删除 fallback 会让 backend 兼容性和 A/B 能力变差。

#### Q：Paged Attention 的 block 调度能不能复用给 MLA？

能复用 block id、PageTable、refcount、hash 和 LRU；不能复用 Standard K/V payload layout 和 attention kernel。MLA 需要自己的 CacheLayout 和 backend adapter。

#### Q：MoE 是否应该使用 EP？

对超大 MoE 模型，EP 通常是 B300 多卡的重要维度，因为它可以将 expert 权重和 token 计算分散到不同 GPU。但 EP 会引入 all-to-all、load imbalance、capacity 和 tail latency 问题，所以应在单卡 router/expert correctness 后接入，并与 TP 组合而不是一开始就把所有逻辑写进 scheduler/BlockManager。

## 16. 最终结论

第一阶段的核心产出不是“Qwen 又多了几个功能”，而是建立了一条可继续演进的 serving 主链路：

```text
Python scheduler
    -> pybind11 runtime
    -> ModelRunner
    -> format-agnostic BlockManager
    -> layout-driven cache storage
    -> backend-dispatched paged prefill/decode
```

Qwen2 用来证明这条链路的正确性和性能，而不再是整个框架的固定数据模型。因此，下一阶段可以直接在 B300 上从 DeepSeek ModelRunner、MLACacheLayout 和 MLA backend 开始，不需要重新设计调度、block 生命周期和 Prefix Cache。

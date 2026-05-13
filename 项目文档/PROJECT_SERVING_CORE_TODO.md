# Serving Core 工业级优化 TODO

## 目标

本项目后续不再追求“补完所有功能”，而是聚焦一条工业推理框架主链路：

```text
Paged Attention
+ KV cache manager
+ continuous batching scheduler
+ chunked prefill
+ prefix cache
+ benchmark / trace / 压测
```

目标不是复刻完整 vLLM/SGLang，而是把 LLAISYS 从原型级 batch decode 升级为可解释、可评估、可对标的 serving core。

## 当前问题

- `BlockAllocator` 只负责物理 block 分配，缺少 block 元数据、引用计数、cache 状态、LRU 淘汰。
- `PageTable` 只记录 request 到 block 的映射，缺少 computed/cached token 状态。
- `batch_decode` 能批量 decode，但 scheduler 不是 token budget / KV budget 驱动。
- prefill 仍是普通 prefill，不是真正 chunked prefill。
- 长 prompt 会阻塞 decode，无法体现在线 serving 的低延迟调度能力。
- prefix cache 有快照/Trie 思路，但不是 block-level KV cache manager 级复用。
- benchmark 偏局部，缺少 TTFT、TPOT、P95/P99、cache hit rate、KV utilization 等 serving 指标。

## 阶段 1：KV Cache Manager 重构

### 目标

把当前 `BlockAllocator + PageTable` 升级成显式 KV cache manager。

### 数据结构

```cpp
struct KVBlock {
    int block_id;
    int ref_count;
    int num_tokens;
    bool computed;
    bool cached;
    uint64_t block_hash;
    uint64_t last_access_time;
};

struct SequenceState {
    uint64_t request_id;
    std::vector<int> block_ids;
    int prompt_len;
    int num_computed_tokens;
    int num_cached_tokens;
    int num_generated_tokens;
};
```

### TODO

- [ ] 新增 `KVCacheManager`，统一管理 block 分配、释放、引用计数。
- [ ] `KVBlock` 预分配，避免运行时频繁创建元数据对象。
- [ ] block 支持 `ref_count`，为 prefix cache / fork / share 做准备。
- [ ] request 释放时不立即清空可缓存 block，而是进入 cached/free 状态。
- [ ] 增加 LRU free queue，只淘汰 `ref_count == 0` 的 cached block。
- [ ] 将 `BatchSlot.page_table` 改为由 `KVCacheManager` 生成 block table。
- [ ] 增加 debug dump：active blocks、cached blocks、free blocks、ref count。

### 验收

- [ ] 单请求输出与当前实现一致。
- [ ] 多请求并发时 block 不泄漏、不重复释放。
- [ ] 随机 request abort / finish 压测后 free block 数恢复正确。
- [ ] 能输出 KV block utilization。

## 阶段 2：Continuous Batching Scheduler

### 目标

从“固定 active slots 批量 decode”升级为 token budget / KV budget 驱动的动态 scheduler。

### 核心参数

```text
max_num_seqs
max_num_batched_tokens
max_num_prefill_tokens_per_step
max_num_decode_tokens_per_step
max_model_len
block_size
watermark_blocks
```

### 调度策略

```text
decode 优先，保护 TPOT
prefill 使用剩余 token budget，保护 TTFT
长 prompt 拆 chunk，避免独占一次 iteration
KV block 不足时触发排队、抢占或拒绝
```

### TODO

- [ ] 新增 request 状态机：WAITING / PREFILLING / DECODING / FINISHED / ABORTED。
- [ ] scheduler 每轮生成 `SchedulerBatch`。
- [ ] batch 内区分 decode tokens 和 prefill chunks。
- [ ] 实现 token budget 检查。
- [ ] 实现 KV block budget 检查。
- [ ] decode request 优先进入 batch。
- [ ] prefill request 按剩余预算切 chunk。
- [ ] 支持 request abort 后即时释放 block 引用。
- [ ] 记录每个 request 的 queue time、prefill time、decode time。

### 验收

- [ ] 多请求流式输出稳定。
- [ ] 短请求不会被长 prompt 长时间阻塞。
- [ ] 在相同 workload 下，比旧 scheduler 有更低 P95/P99 TPOT。
- [ ] 在长短混合 workload 下，TTFT 明显改善。

## 阶段 3：Chunked Prefill

### 目标

实现真正 chunk 级 prefill，而不是一次性处理完整 prompt。

### 正确语义

```text
prompt = [0, ..., S-1]
chunk_0: compute tokens [0, C)
chunk_1: compute tokens [C, 2C), attend to previous KV + current chunk
...
prefill chunk 之间允许插入其他 request 的 decode
```

### TODO

- [ ] `SequenceState.num_computed_tokens` 作为 prefill 进度。
- [ ] 每轮只处理 `min(remaining_prompt, chunk_budget)`。
- [ ] 为 chunk 分配 KV blocks。
- [ ] 当前 chunk 的 K/V 写入 paged KV pool。
- [ ] chunk attention 支持访问历史 paged KV。
- [ ] 首版可使用 correctness 优先路径，后续再接 FlashInfer prefill。
- [ ] 支持 chunk 结束后 request 从 PREFILLING 转入 DECODING。

### 验收

- [ ] chunk size = prompt length 时，与普通 prefill 输出一致。
- [ ] 不同 chunk size 下 greedy 输出一致或 logits 误差可解释。
- [ ] 一个 8k/16k 长 prompt 不再阻塞其他短 decode 请求。

## 阶段 4：Paged Attention Backend 工业化

### 目标

让 scheduler 只生成 attention metadata，attention backend 只负责执行。

### Metadata

```cpp
struct AttentionMetadata {
    int* block_tables;
    int* seq_lens;
    int* query_start_offsets;
    int* kv_start_offsets;
    int num_decode_tokens;
    int num_prefill_tokens;
    int max_seq_len;
};
```

### TODO

- [ ] 将 block table / seq lens 构建从模型 forward 中抽离。
- [ ] decode attention 主路径使用 FlashInfer paged decode。
- [ ] prefill attention 增加 FlashInfer/FlashAttention 路径调研和接入点。
- [ ] 保留手写 paged attention 作为 fallback 和学习实现。
- [ ] 删除或隔离未接入的 `KVQuantMode::INT8/INT4` 原型路径，避免误导文档。

### 验收

- [ ] decode attention 与当前结果一致。
- [ ] metadata 构建开销可统计。
- [ ] attention backend 可以通过配置切换：native / FlashInfer。

## 阶段 5：Prefix Cache

### 目标

实现 block-level automatic prefix caching，而不是整段 KV snapshot 复用。

### 核心思想

```text
block_hash = hash(parent_hash, block_tokens, extra_keys)
```

只缓存完整 block。新请求进入时按 block 查 hash，命中的 block 增加 `ref_count`，跳过对应 prefill。

### TODO

- [ ] 为完整 block 计算 hash。
- [ ] 建立 `hash -> block_id` 映射。
- [ ] 新请求 admit 时查找最长完整 block prefix。
- [ ] 命中 block 后增加 `ref_count`。
- [ ] 未命中部分继续 chunked prefill。
- [ ] LRU 淘汰 cached block。
- [ ] hash key 纳入 tokenizer/model/lora/cache_salt 等 extra keys。

### 验收

- [ ] 共享 system prompt 的请求 TTFT 明显下降。
- [ ] prefix cache hit rate 可观测。
- [ ] 禁用 prefix cache 时输出一致。
- [ ] cache eviction 后不会复用错误 KV。

## 阶段 6：Benchmark 与对标

### 指标

```text
throughput: output tokens/s
TTFT: time to first token
TPOT: time per output token
ITL: inter-token latency
P50 / P95 / P99
KV block utilization
prefix cache hit rate
scheduler queue time
GPU memory usage
```

### Workload

- [ ] 短请求高并发：prompt 64，output 128，concurrency 16/32/64。
- [ ] 长短混合：prompt 64/512/4096/8192 混合。
- [ ] 长 prompt 干扰：1 个 16k prompt + 多个短 decode。
- [ ] prefix cache：共享 system prompt / 共享 RAG 文档。
- [ ] KV 压力：接近 block pool 上限。

### 对照

- [ ] LLAISYS old scheduler。
- [ ] LLAISYS new scheduler。
- [ ] vLLM 同模型、同 prompt、同输出长度、同并发。
- [ ] SGLang 同模型、同 prompt、同输出长度、同并发。

### 控制变量

- [ ] 同一 GPU。
- [ ] 同一模型和 tokenizer。
- [ ] greedy decode。
- [ ] 固定 prompt/output 长度分布。
- [ ] 固定并发请求到达模式。
- [ ] warmup 后统计。
- [ ] 分开报告 prefill-heavy、decode-heavy、mixed workload。

## 推荐里程碑

| 里程碑 | 内容 | 预期时间 |
|---|---|---:|
| M1 | KVCacheManager + block ref count + block utilization | 1-2 周 |
| M2 | token budget scheduler + decode/prefill batch 状态机 | 1-2 周 |
| M3 | chunked prefill correctness | 1-2 周 |
| M4 | FlashInfer backend metadata 整理 | 3-7 天 |
| M5 | block-level prefix cache | 1-2 周 |
| M6 | benchmark harness + vLLM/SGLang 对标 | 1 周 |

## 最终展示口径

```text
我围绕 serving core 做了系统优化：
KV cache manager 管理 block 生命周期，
continuous batching 按 token/KV budget 调度，
chunked prefill 避免长 prompt 阻塞 decode，
prefix cache 复用共享上下文，
paged attention backend 读取分页 KV。
最后用 TTFT/TPOT/P99/KV utilization/cache hit rate 做对照评估。
```

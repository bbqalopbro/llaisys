# DeepSeek-V4 多算子后端接口

## 用途与当前范围

面向实际 `DeepSeek-V4-Flash-0731` 的独立模型执行器。模型负责配置、权重、层结构和
请求状态；算子后端负责 tensor 计算；Python serving scheduler 不进入这里。

当前有 19 类版本化算子契约。默认是六类上游 TileLang 核心算子加十三类 PyTorch
模型算子，Hadamard 仍通过显式函数参数接入上游 CUDA 扩展。接口位于：

- `python/llaisys/models/deepseek_v4_backends.py`：契约、注册、选择和调用统计。
- `python/llaisys/models/deepseek_v4_model_ops.py`：结构算子参考实现及 `ExpertDispatch`。
- `python/llaisys/models/deepseek_v4_model/`：接口的真实模型调用方。

本节的 19 类 Python 契约描述独立 Python 正确性执行器；另有已实现的 C++/pybind
整模型与原生 serving 接口，见文末。两种执行器均已连接既有 InferenceEngine，
未连接 HTTP 入口，也不是真正 fused batch。
另有显式 paged latent 路径和四类 cache 算子契约，见文末。把 Python 中的专家执行
循环提取为接口，也不等于完成 grouped GEMM
或高性能 GPU dispatch。

## 对话输入与独立生成接口

### 既有 scheduler 的显式参考后端

`deepseek_v4_model.batch.DeepSeekV4ServingModel` 让已经加载的独立模型消费现有
`server.engine.InferenceEngine` 的 slot/batch 协议，不新增调度队列或分块策略。

```python
from llaisys.models.deepseek_v4_model.batch import DeepSeekV4ServingModel
from server.engine import InferenceEngine, SamplingParams

serving_model = DeepSeekV4ServingModel(model, eos_id=codec.eos_id)
engine = InferenceEngine(
    serving_model, max_batch_size=2, max_seq_per_slot=model.config.max_seq_len,
    prefill_chunk_size=model.config.max_seq_len,  # 原始策略先保持完整 prefill
    enable_block_prefix_cache=False, block_watermark=0,
)
# 在运行的 asyncio event loop 中使用：
engine.start()
request = engine.submit(input_ids, SamplingParams(temperature=0, top_k=1, max_tokens=16), "session")
try:
    generated_ids = await request.future
finally:
    await asyncio.to_thread(engine.stop)
```

`asyncio` 由调用方正常导入。可以设置 `stream=True` 后消费 `request.stream_tokens()`；
取消使用 `engine.cancel(request.request_id)`，调用方取消 Future 也会通知 worker。
Facade 必须在完成模型权重写入的 CUDA stream 上创建，记录 event 后由 worker 的
专属 stream 等待。创建后不得并发修改权重。一个 facade 同时只能拥有一个 context，
所有 context 方法只允许所属 worker 调用。close 后才可创建下一个 context。

默认禁止形状敏感的增量 prefill；实验必须显式设置
`allow_experimental_chunking=True`，需要 Prefix 再设置 `enable_prefix_cache=True`。
这些开关不修改已绑定算子或数值策略，不是精度通过声明。实际 scope 写入 report。

主机抢占快照保存 latent/Indexer、压缩器、prefix 边界及采样 RNG，释放 slot 后不
持有 GPU blocks；恢复分配新 blocks，复用四类 cache 算子接口。它是进程内对象，
不是磁盘 checkpoint、跨进程通信协议或多卡 rank 状态。CPU 搬运明确属于当前参考
抢占路径，后续 C++ runtime 应保留同样语义并优化搬运/workspace。

当前多个 slots 在一个 stream 上串行执行，不是 fused/mixed GPU batch。模型仍
在 Python 发起 19 类算子调用；只接通既有 InferenceEngine，不代表 HTTP app、C++
model execution 或 TP/EP 已完成。观察 logits 的正确性 runner 不作为性能 benchmark。

`deepseek_v4_model.chat.ChatCodec` 显式加载所选本地权重目录中的
`encoding/encoding_dsv4.py`。复用官方格式，不使用缺失的 Hugging Face chat template，
不下载或执行远程代码，不导入上游 `model.py`。调用方须确认本地编码源码可信。
报告绑定实际执行的编码源码 SHA256、tokenizer backend SHA256 和 BOS/EOS ID。
编码器已包含 BOS，因此 tokenizer 使用 `add_special_tokens=False`，避免重复添加。

支持 chat/thinking、reasoning effort、多轮消息、工具调用历史与工具结果排序。
这些是编码/解析能力，不会执行工具或调用外部服务。非文本 content、未知角色及
预处理后的 content_blocks 显式拒绝，不静默转换为占位文本。

`deepseek_v4_model.generation.greedy_generate` 是单请求 Python 生成驱动，消费独立
模型和可选 C++ paged pool，不依赖 golden token，也不导入上游模型。它负责：

- 显式 greedy 选择与实际生成 token 的反馈；暂不实现随机采样或 MTP。
- 可选 chunked prefill，中间 chunk 不执行输出 head；不创建第二套 serving 队列。
- EOS 停止、输出长度/上下文容量限制，以及 chunk/token 边界的取消回调。
- 正常、取消、算子异常、输出回调异常均在 finally 中关闭请求，归还 block lease；
  共享 pool 仍由调用方持有，单 CUDA stream 约束不变。

返回 token IDs 包含模型真正生成的 EOS。达到长度上限时不人为补 EOS，解析器保留
原始截断文本并标记 `complete=false`；有 EOS 但格式错误时返回显式 parse_error，
不会把错误内容修补后宣称格式正确。可选 token 回调不等于已接入流式 HTTP API。

独立入口 `tools/deepseek_v4_reference/run_chat.py` 接受 JSON 对话用例，无需生成或
提供 golden 文件。默认六类 TileLang + 十三类 Torch、CUDA Hadamard、C++ paged
storage；支持显式切换已有 GEMM/数值策略。它仍不是最终 C++ batch runtime。
结果中的单次未预热 wall time 不代表正式 TTFT/TPOT，也不是统计任务精度。

参考 runner 的 messages 用例现在按相同模型版本的官方编码生成并在真实 EOS 停止；
原始 raw completion 模式保留旧行为。独立对照额外使用 `--free-generation` 时，
同时记录固定历史 logits 对照与不喂 golden token 的自由生成，不混淆两种证据。
固定策略仍是显式实验选项，不因新的对话入口而升级为原始数值基线。

## 已接入的算子

| 名称 | 函数参数及关键语义 | 默认后端 |
|---|---|---|
| `act_quant` | BF16 激活，分组尺度与 inplace QDQ；量化 GEMM 使用 1×128 activation scale | TileLang |
| `fp4_act_quant` | FP4 E2M1 激活量化 / QDQ，分组 32；用于 Indexer，不代表专家 GEMM 是 W4A4 | TileLang |
| `fp8_gemm` | 已量化 FP8 A/B 及 scale；权重分组 128×128；输出 BF16 | TileLang |
| `fp4_gemm` | FP8 A、packed FP4 B 及 scale；权重分组 1×32；W4A8，输出 BF16 | TileLang |
| `sparse_attn` | BF16 query / latent、FP32 sink、int32 候选索引；-1 为无效槽 | TileLang |
| `hc_split_sinkhorn` | FP32 mixes、scale、base；输出 pre/post/combination | TileLang |
| `dense_linear` | `X[...,K], W[N,K]`，匹配的 BF16 或 FP32；不重新量化 | PyTorch |
| `grouped_linear` | `X[B,S,G,K], W[G,N,K]`，输出 `[B,S,G,N]` | PyTorch |
| `row_inv_rms` | `X, eps`；保留输入 dtype 的 square/mean/rsqrt；query 与 HC 使用 | PyTorch |
| `rms_norm` | `X, FP32 weight, eps`；FP32 归一化后转回输入 dtype | PyTorch |
| `rotary_inplace` | `X[B,S,(H),R], complex64 freqs[S,R/2], inverse`；邻接实虚数对，支持带 stride 的尾部视图 | PyTorch |
| `indexer_scores` | BF16 Q、压缩 K、head weights；保留 BF16 点积、ReLU、乘法和归约边界 | PyTorch |
| `indexer_topk` | `scores, k, tie_policy`；明确 published 或 index_ascending，不隐式换序 | PyTorch |
| `router` | FP32 logits、bias 或 hash IDs、top-k、score function、route scale | PyTorch |
| `compressor_pool` | FP32 values/scores；沿压缩组执行 softmax 和加权归约；支持 prefill/chunk 与单 token decode 的原始输入 rank | PyTorch |
| `expert_activation` | BF16 gate/up、clamp limit、可选 FP32 routing weight；FP32 SwiGLU，路由缩放在 BF16 cast / down projection 之前 | PyTorch |
| `moe_dispatch` | hidden、routing weights、expert IDs、expert 数；返回设备端 packed tensor 和 offsets | PyTorch |
| `moe_combine` | packed expert output 与 dispatch 描述；按 expert ID 递增顺序进行 FP32 累加 | PyTorch |
| `hc_sum` | FP32 HC 项与明确的归约轴；不改变 dtype | PyTorch |

`rms_norm` 是完整算子契约，`row_inv_rms` 是 query/HC 使用的独立契约。当前
`register_torch_model` 将同一个 inverse-RMS 数学实现传给两者，所以选择 fixed32
时，RMSNorm 内部也采用固定行策略；但在注册表中单独替换 `row_inv_rms`，不会改写
已经绑定的 `rms_norm` 闭包。不要依据调用次数把两者算作同一次 fused kernel。

## 替换单个算子

```python
from llaisys.models.deepseek_v4_backends import (
    CONTRACTS, MODEL_CONTRACTS, OperatorRegistry,
    register_tilelang, register_torch_model,
)

registry = OperatorRegistry(MODEL_CONTRACTS)
register_tilelang(registry, published_kernel_module, tilelang_version)
register_torch_model(registry)
registry.register(
    "my_operator_library", "fp4_gemm", my_w4a8_gemm,
    version="my-library-commit",
    contract=MODEL_CONTRACTS["fp4_gemm"],
)
selection = {
    name: "tilelang" if name in CONTRACTS else "torch"
    for name in MODEL_CONTRACTS
}
selection["fp4_gemm"] = "my_operator_library"
ops = registry.bind(selection)
# 将 ops 传给 DeepSeekV4Model；其他算子继续使用明确选择的基线。
```

六类 `CONTRACTS` 保留给上游参考 runner；独立模型使用完整 `MODEL_CONTRACTS`。
不完整选择、重复注册、未知算子或契约不匹配均失败。绑定后不自动改选实现。
每个算子的后端名称、版本、调用数、异常数和 fallback 状态可从 `ops.report()` 获取。
缺失库、kernel launch 失败必须显式报错，不能偷偷转 PyTorch 或另一个 GEMM。

## MoE packed 接口与所有权

输入：`hidden[T,D]`、`weights[T,K]`、每个 token 内互不重复且在 `[0,E)` 范围内的
`expert_ids[T,K]`。router 或经过校验的 hash 表负责满足 expert ID 前置条件。

`ExpertDispatch` 包含：

| 字段 | dtype / shape | 所有权与顺序 |
|---|---|---|
| `hidden` | BF16 `[T*K,D]` | 新分配，不修改或别名输入 hidden |
| `route_weights` | FP32 `[T*K,1]` | 与 packed hidden 同序 |
| `token_indices` | int64 `[T*K]` | packed 行对应原 token 行 |
| `expert_offsets` | int64 `[E+1]`，位于同一设备 | 按 expert ID 递增的半开区间；空专家允许相邻 offsets 相等 |
| `num_tokens` | 主机整数 | combine 的输出 token 数 |

同一 expert 内保持输入 token/choice 的稳定顺序。权重所有权仍属于模型；dispatch
不复制模型权重，也不修改 block manager。专家输出同 packed 顺序，已在 SwiGLU
内部应用 routing weight；combine 不能再乘一次。shared expert 在 combine 后单独
相加，最后统一转 BF16。

当前参考执行器读取 `expert_offsets.tolist()`，逐专家调用已有 Linear/GEMM；参考
combine 也读取 offsets，并按 expert ID 顺序累加。这里有可观察的主机同步，尚非
device-only metadata / grouped expert execution。未来应通过批专家执行接口消除
这些同步，不能把它们描述为已经实现的优化。多卡还需要 placement、rank/group、
通信、跨 rank 错误处理及 tensor/stream 生命周期合同，不是替换 dispatch 即完成 EP。

## 数值、布局和 stream 约束

- 不把 W4A8 改为 W4A4；不对已经量化的 GEMM 输入重复量化。
- 不用 FP32 近似替代 BF16 query normalization 或 Indexer 中间舍入而不记录差异。
- Router bias 仅影响选专家，不直接进入 routing weight；hash 层使用模型提供的 IDs。
- Expert 的路由缩放在 down projection 前，移到输出后通常不能保持量化语义。
- `rotary_inplace` 只改 RoPE 视图，不能覆盖同一 tensor 的非 RoPE 部分。
- 不复制、重复或展开 latent payload 来伪装普通 MHA K/V；候选 padding 只是 metadata。
- 输入 tensor 必须位于同一设备。参考 PyTorch 后端遵循当前 stream；自定义后端需要
  自行验证非默认 stream、输出存活期和临时 workspace，不得依赖默认流隐式同步。
- 返回 tensor / metadata 的生命周期至少覆盖后续消费；异步 kernel 不能引用已释放的
  workspace。当前接口尚未提供跨请求 workspace 池，不宣称已解决并发复用。

## 实验性形状稳定策略

`torch-fixed32` 显式将普通投影及 inverse RMS 按固定 32 行计算，并固定 HC 归约
顺序。配合 `indexer_tie_policy=index_ascending`、`attention_metadata_policy=fixed`
用于区分形状相关舍入和增量缓存错误。这是另一个数值策略，不是性能优化声明。

原始默认仍是 `torch + published + published`。固定策略目前在完整 43 层的
134-token、65/65/4 分块同策略对照中逐值一致，但与原始 golden 的 1% relative L2
门槛不一致，不能因此替换默认后端或声称所有精度问题已解决。
19 类接口接通后又验证了 2-token 分块：5-token 输入分成 2/2/1、3-token 输入分成
2/1、134-token 输入分成 67 个块；五例共 20 步均与同策略 full 逐值一致。与原始
golden 的 relative L2 仍为 3.084%–8.595%，总验收标志保持 false。

### 独立固定策略参考

`tools/deepseek_v4_reference/published_fixed_profile.py` 提供显式
`fixed32-reference-v1`。它在内存中对共享的上游 `model.py` 做严格 call-site 审计和
局部数学替换，复用上游模型控制流与权重映射；不导入 llaisys 模型实现，不修改
共享源码，不全局 monkey-patch PyTorch。它不是原始上游数值基线。

参考覆盖普通/分组投影的固定 32 行、包括 RMSNorm 在内的 inverse RMS、HC 升序
归约、Indexer 相同分数时按候选下标升序、固定 attention 候选分段。量化 GEMM 和
稀疏 attention 继续使用上游 TileLang，Hadamard 使用上游 CUDA 扩展。MTP 显式拒绝。

生成和消费参考时都必须传 `--reference-profile fixed32-reference-v1`；消费者还须
显式选择 `torch-fixed32 + index_ascending + fixed`。报告保存原始源码、转换后 AST、
策略实现的 SHA256，并校验完整权重与 golden 摘要。混用原始/固定策略报告、策略
源码变化或审计调用点变化均失败，不能把新策略结果重新标记为原始上游精度通过。

### 中间 chunk 不生成 logits

`DeepSeekV4Model.forward(..., emit_logits=False)` 执行全部 transformer 层并提交
cache/position，但跳过 HC head、最终 norm 和词表投影；返回 `ModelOutput.logits=None`。
调用者必须显式决定何时需要 logits，模型不管理 token budget、chunk 划分或采样。
`capture_layers=True` 仍可捕获该 chunk 的中间层输出。

默认模型 API 保留 `emit_logits=True`。独立分块 runner 默认只在最后一个 chunk
请求 logits；`--intermediate-prefill-logits` 恢复逐 chunk 输出用于诊断。报告记录
`intermediate_prefill_logits` 与每个 prefill 的 `skipped_prefill_heads`。此改动不是
完整 C++ batch runtime 接入，也不能单凭少执行 head 就宣称 TTFT 提升。

## 验收方法

1. 算子层：同输入测试布局、dtype、原地写入、mask、空专家和数值语义。
2. 模型层：小配置逐层与上游源码对照；完整真实权重固定 token 历史的 logits 对照。
3. 端到端：接入既有 runtime/serving 后，再测请求取消、cache 共享、正式 TTFT/TPOT
   与吞吐；当前参考步骤耗时不替代这一层验收。

新增测试在 `test/test_deepseek_v4_model_ops.py`、`test/test_deepseek_v4_model_backends.py`
和 `test/test_deepseek_v4_model.py`。原始 chunk 精度失败测试保留，不能仅执行新增
通过项后宣称全套测试通过。最新结果见 `B300_DEEPSEEK_V4_BRINGUP.md`。

## 参考产物与权重身份校验

新的参考生成命令可显式指定 `--hash-checkpoint`：以 8 MiB 缓冲区流式计算整个
MP1 文件（包含 header 和所有 tensor payload）的 SHA256，而不是只校验文件名或
safetensors header。不会把整份权重读入 Python 内存，也不使用仅靠 mtime 命中的
哈希缓存。计算前后及推理结束后校验文件状态，发现普通并发写入/替换则失败。

保存 golden 时，报告同时记录每个 `case_N.safetensors` 的 SHA256。独立执行器发现
参考报告带有完整 checkpoint 身份后，会在加载模型之前重新扫描实际权重并比较
摘要，也会比较 golden 文件摘要。即使文件大小、token 历史和 argmax 不变，payload
或非最大 logits 的变化仍会被拒绝。该机制是产物一致性校验，不是带签名的来源认证。

旧报告不具备完整 payload 摘要。它们仍可用于已有兼容回归，但每个 case 的
`checkpoint_payload_verified` 会明确为 false；不会反向补写成已校验。新报告每个
case 分别记录验证状态，混合新旧基线时保留身份限制说明。

长输入可用 `--reference-mode full-only` 生成原始 full prefill/decode golden。这不
执行逐 token replay，对应 `token_replay_tested=false`、`accuracy=null`、
`all_argmax_equal=null`；不把未测项目标为通过。默认模式仍为 `full-and-replay`，
原有 full/replay 检查没有删除。两种模式的 warmup 范围会分别记录。

## V4 paged latent 与 cache 算子接口

`deepseek_v4_model/paged.py` 提供 `PagedCachePool`。这是模型专属 payload 层，复用
`_C.CacheBlockPool` 中的既有 C++ BlockManager，不创建新的 Python free list、LRU
或前缀 Trie。默认连续路径保持不变；必须显式选择 pool：

```python
from llaisys.models.deepseek_v4_model import PagedCachePool

pool = PagedCachePool(model, num_blocks=32, block_size=128)
request = model.new_request(cache_pool=pool)
try:
    output = model(input_ids, request)
    output = model(next_token_ids, request)
finally:
    request.close()
print(pool.report())
```

每层每个物理 block 包含 `block_size` 个 full-resolution latent，以及
`block_size / ratio` 个 compressed latent；ratio=4 层另有 Indexer pool。RoPE 维度
仍在 latent 内，不展开成普通多头 K/V。block size 必须整除所有非零压缩 ratio。

Attention 将逻辑候选索引映射为物理槽索引，原 TileLang `sparse_attn` 直接读取
pool；保持原候选顺序、mask 和 dtype。不 gather 历史 attention latent，也不将
历史 latent 与新 token 拼接成连续 payload。Indexer 的评分路径仍显式 gather
压缩 key，报告中 `indexer_history_gather=true`，不能称为全流程无 gather。

独立 `CACHE_CONTRACTS` 使用同一注册/绑定机制，允许用户逐项替换：

| 接口 | 输入和输出 | 默认实现 |
|---|---|---|
| `cache_map_slots` | 逻辑位置、设备 block table、component width/stride/offset → int64 物理槽；无效位置保持 -1 | PyTorch |
| `cache_write` | BF16 pool、BF16 新 payload、互不重复的有效槽 → 原地写入所属 block | PyTorch `index_copy_` |
| `cache_read` | pool、有效物理槽 → 独立连续 key tensor；仅 Indexer 参考路径使用 | PyTorch `index_select` |
| `cache_copy_block` | 同一 BF16 component pool、不同 source/destination block ID、stride → 复制该组件完整 block；仅写目标 | PyTorch `copy_` |

将这四项注册到 `OperatorRegistry(CACHE_CONTRACTS)`，显式 `bind` 后作为
`PagedCachePool(..., ops=cache_ops)` 传入。缺失/不匹配的契约或执行错误均失败，
没有隐式后端替换。现有 19 类模型算子接口不因选择 paged 而改变。

### 所有权、错误与当前限制

- C++ lease 的 `append` 转移 block 引用，不增加/减少引用计数；拒绝跨 pool、自身
  追加或重复物理 ID。`mark_computed_counts` 批量校验后更新各 block 有效 token 数。
- 在整个模型前向开始前按需分配 block；成功后才更新 computed 和 request position。
  容量不足不会推进 position；kernel 错误使请求失效，必须 reset/close 后再用。
- 每个 pool 固定模型、权重 generation、设备和 CUDA stream。多个请求可以在该流上
  交错执行；跨流执行显式拒绝。目前不提供真正 mixed GPU batch。
- 压缩器的 unfinished/overlap 投影状态仍由请求持有。现已新增显式 block Prefix
  Cache 与 partial-block COW，包含必要的 overlap 边界状态；默认不开启，见下节。
- 这一版 full-resolution 槽保留到请求释放，尚未按滑动窗口淘汰；长上下文显存仍有
  优化空间。报告明确 `full_resolution_window_eviction=false`。
- payload 默认由 C++ PagedCacheStorage 持有，Python 只保留零拷贝 tensor 视图；可
  显式选择 `storage_backend="torch"` 做存储对照。Python 仍发起模型算子调用，模型
  权重和临时 tensor 还未整体迁入 C++，因此不是 model batch execution 完成。
  block manager 不感知任何 tensor shape。

runner 的 `--cache-layout paged` 强制此路径；`--compare-contiguous-in-process` 在
相同输入、相同 chunk 形状下对照连续布局，不把布局一致性当作 full/chunk 一致性。
原始 chunk/full 失败门槛继续保留。具体实测结果以工程记录和 JSON 为准。

## V4 block Prefix Cache、边界状态与 COW

`PagedCachePool(..., enable_prefix_cache=True)` 显式开启。该开关不修改模型数值
策略；原始策略与 fixed32 策略必须分别验收，不能因缓存命中就默认精度通过。
默认仍为关闭。已有 19 类模型契约不变，cache 契约新增 `cache_copy_block`。

```python
pool = PagedCachePool(model, num_blocks=32, block_size=128, enable_prefix_cache=True)
seed = model.new_request(cache_pool=pool)
try:
    model(input_ids, seed)
    seed.publish_prefix()
finally:
    seed.close()

request = model.new_request(cache_pool=pool)
try:
    matched = request.attach_prefix(input_ids[0].tolist())
    output = model(input_ids[:, matched:], request)
finally:
    request.close()
pool.prefix.clear()  # 清除缓存索引，不释放仍被其他活跃请求引用的 blocks
```

底层查找、哈希链、引用计数和 LRU 仍来自现有 C++ `BlockPrefixCache/BlockManager`，
没有新建 Trie 或另一套 allocator。Python `prefix.py` 只维护模型恢复状态和用于
碰撞复核的完整 token block。record 以 physical block ID 绑定 hash，受 pool
容量限制；正常回收重分配时清理旧 record。不得绕过 pool 在外部重新发布/重绑定
其底层 block，或擅自写共享 payload/snapshot。

V4 的恢复状态有两个要点：

1. BT 必须整除各层压缩比例。ratio=128 的完整 block 边界没有未完成组，不需要
   保存旧 scratch；ratio=4 则必须保存上一组 4 个 token 的 FP32 投影/score 第一半，
   attention compressor 与 Indexer compressor 都要保存。
2. 每个完整 block 都要有对应边界快照，不能用整个长请求的最终 compressor state
   代替。快照从本次实际投影提取，支持 full、跨边界 chunk 和单 token 补齐；不
   重新执行模型，也不改变现有 attention/GEMM 的数学策略。输入 token 在设备上
   记录，仅 publish 时转到主机，不为 prefix 哈希逐 decode token 增加 host 拷贝。

`publish_prefix()` 仅发布已计算完整 blocks，缺失任一必要层状态则失败；没有完整
block 或遇到由另一物理副本持有的既有前缀时返回 false，不覆盖既有 payload。
`attach_prefix(tokens)` 只接受空、有效、同 pool 的请求。命中后共享物理 blocks，
恢复独立 compressor scratch，并推进位置；总会留下至少一个 prompt token 计算
最终 hidden/head，因为缓存中没有保存最后 logits。

`request.fork()` 分享 block lease，但复制请求的 compressor scratch。继续写入
被多个请求共享的 partial block 时，先分配私有 block，通过可替换
`cache_copy_block` 复制每层 latent/压缩/Indexer 组件，再切换 block table 和引用。
完整共享 blocks 不被覆写。显存不足或复制出错不会修改原共享 payload；关闭、
reset、取消及正常结束均释放相应引用。`clear()`/LRU 不会回收仍有活跃引用的块。

生成 API 使用 `greedy_generate(..., reuse_prefix=True)`；聊天 CLI 使用
`--prefix-cache`。验证 runner 的 `--prefix-reuse` 会先执行并发布冷请求，再实际
跳过命中 token，分别比较 seed、命中后 logits、golden 和可选自由生成。报告中
`prefix_hit_tokens` 是本次命中，pool 统计还包含 warmup/自由生成的额外 lookup。
`resume_state_bytes` 单独记录 FP32 边界状态内存。

当前是真实单 stream 的模型缓存能力，仍不等于完成 serving 抢占恢复、跨 stream
共享、窗口淘汰或 C++ 模型批执行。真实 B300 精度结果见中文工程记录的最新章节。

## C++ cache storage 与 DLPack 所有权

默认构造 `PagedCachePool` 使用 `storage_backend="cpp"`；runner 对应
`--cache-layout paged --paged-storage cpp`。`--paged-storage torch` 是明确指定的
对照实现，不是静默 fallback。native 绑定未编译时，cpp 选择立即报错。

`src/models/deepseek_v4/cache_layout.hpp` 定义每层的
`InterleavedLayerLayout`。每层复用一个既有 `core::PagedCacheStorage`：

- `latent` component 的每块是 `[block_size + block_size/ratio, latent_dim]`，无
  压缩层只保留 block_size 行；ratio-4 的 `index` component 是 `[block_size/4, index_dim]`。
- cache 固定 BF16，两类 component 独立分配。物理形状与此前 Torch paged pool 完全
  一致，没有把其他旧 layout 的 pointer 强行解释为新布局。
- C++ 描述符和分配器负责 stride/offset/容量；三个 cache 算子契约仍保持不变。

`_C.V4PagedStorage.view(layer, component)` 返回实现 `__dlpack__` /
`__dlpack_device__` 的 C++ view。`torch.utils.dlpack.from_dlpack(view)` 只创建
共享显存的视图，TileLang 接收相同 pointer。支持 legacy 和 v1 versioned capsule，
明确拒绝要求复制、跨设备复制和不匹配 stream 的导出。

生命周期约定：

1. 每个 capsule 都持有独立 metadata 和 C++ shared_ptr；被消费后改名，消费一次。
2. 删除 Python storage/view 不会使仍存活的 Tensor 或切片悬空；最后一个持有者释放
   对应层的 C++ allocation。deleter 不持有 Python 对象、不依赖 GIL。
3. GPU 最终释放在所属 device 上同步所属 stream，并恢复调用线程原 device。这个
   同步发生于存储销毁，不在每个 token 热路径中；不是跨 stream 并发支持。
4. 导出后，调用者仍必须遵守 pool 的单 stream 合同；普通 Torch Tensor 不能阻止
   调用者擅自在另一个 stream 读写。不能把导出检查称为全生命周期自动 stream 调度。
5. 显存视图的存活不等于 logical block 租约存活。请求释放后，block 可被复用，不能
   将旧 view 当作不可变请求快照。下述 Prefix Cache/COW 仅保护通过请求 API 的访问，
   不会拦截用户对原始 pool Tensor 的任意写入或底层 block/prefix 的外部重绑定。

`_C.v4_storage_counters()` 提供原生 live_bytes/live_layers/destruction_errors。
显式 `PagedCacheStorage.release()` 会逐 component 释放、保留失败 pointer 供重试，
避免重复释放已成功的部分；析构不向外抛异常。若 CUDA context/stream 已失效，
原生 owner 明确记录销毁错误且不把未回收 bytes 记作已释放，不能静默声称回收成功。

构建时为现有 `--python-bindings=y` 增加 `--dlpack-include=<包含 dlpack/dlpack.h
的目录>`（DLPack >=1.0）。不依赖 libtorch C++ ABI；只使用 pybind、DLPack 头文件、
项目 runtime 和 CUDA runtime。CUDA include/lib 由当前 toolkit 探测，不复制旧架构。

runner 的 `--compare-paged-storage-in-process` 比较 cpp 与 torch pool 的同 schedule
logits，并要求逐值一致；在真实模型运行结束后释放 native pool，检查原生计数回到
开始值。原生 allocation 不属于 Torch caching allocator，因此报告把
`peak_memory_bytes` 明确标为 Torch allocator 范围，另列 `native_cache_allocation_bytes`，
不能把单独的 Torch peak 当作整个进程 GPU 显存峰值。

真实 43 层、2105-token/16 步已验证：cpp 存储的原始策略 full 对原始 golden、
Torch paged/连续缓存逐值一致；固定策略 257-token chunk 对独立固定策略 golden、
同策略 full 和 Torch paged 也逐值一致。两次 pool 销毁后的原生计数均归零。版本、
开关、范围及仍失败的原始 chunk 门槛详见工程记录，不将该结果称为完整 serving 验收。

DLPack 所有权与 capsule 约定参考其[官方 Python 规范](https://dmlc.github.io/dlpack/latest/python_spec.html)。

## 原生 C++ 算子接口与量化 Linear（2026-09-08）

新增接口不替换上述 19 类 Python 参考契约，也没有把 serving 策略移入 C++。
它是完整 C++ 模型执行迁移的起点，当前已实际接通六类核心 kernel 和量化 Linear：

- `src/backends/native/tensor.hpp`：使用既有 `core::Runtime/Storage` 持有设备内存，
  DLPack 描述真实 dtype、logical shape/stride；支持 packed FP4，不把量化权重转为 BF16。
- `src/backends/native/kernel.hpp`：不依赖 TVM/Torch/Python 的 `Kernel` 抽象，包含
  后端名、版本、operation、contract revision、调用与失败计数，以及所属 Runtime。
- `src/backends/tilelang/native_kernel.*`：载入已经导出的动态库，直接通过 C++ TVM-FFI
  调用，临时切换到现有 Runtime 的 stream，并在正常和异常路径恢复 FFI stream。
- `src/models/deepseek_v4/linear.*`：持有权重和 scale 的共享所有权；分别绑定量化及
  GEMM 后端；一次 C++ `forward` 连续执行 BF16 激活量化和 W8A8/W4A8 GEMM。

当前 `QuantizedLinear` 接口：

```cpp
// 使用 backends::native::Tensor / Kernel；构造与调用均在所属 Runtime 线程。
QuantizedLinear layer(weight_owner, weight_scale_owner, quant_backend, gemm_backend);
LinearWorkspace workspace(runtime, rows, layer.inputDim());
layer.forward(input_bf16, output_bf16, workspace);
// 同一 shape 的后续调用可复用 workspace；所有 tensor 保活至 stream 完成。
```

量化契约为 `act_quant@1`：BF16 `[M,K]` → FP8 E4M3 `[M,K]`、UE8M0 `[M,K/128]`。
GEMM 为 `fp8_gemm@1` 或 `fp4_gemm@1`：输出 BF16 `[M,N]`；FP8 权重 scale 为
`[ceil(N/128),K/128]`，FP4 权重 scale 为 `[N,K/32]`。这里的 C++ `Tensor` 权重
shape 始终为逻辑 `[N,K]`，FP4 payload 字节数为 `N*K/2`。

自定义算子库可继承 `Kernel` 并实现 `call/close/calls/failures`，再将实例传给该模型
组件。`call` 只能使用指定 Runtime stream，不能改变量化边界、重新乘 routing weight
或自动选择另一个后端。缺失实现、版本化契约不符、shape/dtype/runtime 错误应显式失败。
测试注入的用户后端抛错后，TileLang 量化/GEMM 调用数保持不变，验证了没有隐式 fallback。
这些接口不要求用户库一次实现全部算子。后续新增的 ATen/Hadamard 后端和完整原生
MoE 组合见下文；不能将组件级迁移等同于完整模型迁移。

生命周期要求：模型持有权重和 kernel；调用方持有 input/output/workspace，直到排队的
CUDA 操作完成。整个 Runtime 及其所属线程必须比这些对象活得更久；不允许跨线程释放
最后一个模型 tensor owner。Kernel `close` 同步后再卸载动态库；错误线程上的析构会
明确计错并保留 module，避免卸载仍在使用的设备代码。`forward` 本身不分配 workspace、
不做 D2H/H2D、不在量化与 GEMM 之间同步，但首次 module 加载和底层 allocator 的开销
没有在本轮性能验收中测量。尚不承诺 CUDA Graph、跨 stream 并发或任意 strided 输入。

### 固定版本 ABI 约束

当前原生 adapter 限定 TileLang 0.1.8；验证环境固定 TVM-FFI 0.1.8.post2。
TileLang 的该版 ArgBinder 将 sub-byte 参数解释为 packed byte shape，因此 adapter
仅在调用边界把 FP4 的逻辑 `[N,K]` 描述改成 uint8 `[N,K/2]`。数据地址、字节和模型层
dtype 都不变，未修改上游 kernel。奇数 K 或其他未经验证的 sub-byte 布局显式拒绝。
不能直接把这套 ABI 适配用于其他 TileLang 版本而不重新检查和测试。

`AnyView` 借用的是 `TensorView` 内部的 DLTensor；必须同时保活整个参数视图数组，
不能把临时 `TensorView` 填入参数数组后销毁。动态依赖还包含 `libtilelang` 的参数错误
helper 注册，不能仅链接 `libtvm`/`libtvm_ffi`。本地 TVM 存在旧 FFI 的相对 RUNPATH，
验证器显式选择兼容 FFI 的 library path 并记录最终 `ldd`。

### 构建与验证范围

在原有、按实际 GPU 配置的 xmake 参数上增加：

```text
--tilelang-native=y
--tilelang-root=<当前安装的 tilelang 包目录，包含 lib/>
--tvm-ffi-root=<兼容 tvm_ffi 包目录，包含 include/ 和 lib/>
```

构建 `xmake build llaisys-tilelang-native` 以及 `xmake build llaisys-deepseek-v4-native`。
后者持有与具体算子库无关的 Linear/MoE，前者只持有 TileLang adapter。
默认关闭相关选项，不给既有 Qwen/CPU 路径强加
TVM 依赖。`verify_native_tilelang.py --native-library <生成的静态库>` 实际链接这个构建
产物，而不是另编一份同名实现。Python 只负责离线 JIT/导出、准备参考数据；最终测试
子进程没有 Python/Torch 动态依赖，不经 pybind，也没有逐算子 Python callback。

最近一次 Linear 验收已加入非零 offset 输入、保护区、别名拒绝与零行输入：
23 个 primitive 用例 + 9 个实际权重 Linear 用例，193 项 tensor 字节对照
一致、113 项异常/契约检查及 9 项空输入检查通过。完整参数与证据见工程记录及
`benchmark_results/deepseek_v4_b300_native_tilelang.json`。这里尚未执行完整 C++
Transformer block、43 层模型或服务请求，不能复用先前 Python 整模型精度结果宣称
原生模型已经通过。

## 原生结构后端、Tensor 视图与完整 MoE 组合（2026-09-08）

### Tensor 与后端职责

`native::Tensor::asStrided(shape, strides, element_offset)` 产生共享既有 Storage 的
受检视图，不创建第二套 allocator。offset 相对当前视图，以逻辑元素为单位；FP4
offset 必须按字节对齐。拒绝负 stride、越界和尺寸溢出；允许零行/零元素 tensor，
此时不分配设备 payload。`upload/download` 只接受连续视图，保留精确字节数校验。
模型权重和 request workspace 仍须在所属 Runtime 线程上保活到 stream 完成。

TileLang adapter 将 `byte_offset` 合入借用描述符的 data pointer，保留底层 owner；
FP4 的 packed-byte ABI 转换仅发生在调用边界。量化 Linear 要求连续视图，不能把
Tensor 支持带 stride 理解为每个 kernel 都支持任意 stride。RoPE 则专门验证了
BF16/FP32、最后 64 维的非连续尾部视图，只改写目标区域。

`src/backends/aten/structural_kernel.*` 提供显式 `aten-reference` 后端。调用的是
libtorch/ATen 的 C++ 算子，而不是 Python，也不是自研 fused CUDA kernel。
它在 Runtime stream 上借用 native Tensor，不把内存所有权交给 ATen；临时结果由
ATen allocator 管理，再写回调用方输出。基线版本固定为 Torch 2.13.0+cu130、
CXX11 ABI=1，编译使用 C++20；外部 stream guard 在正常和异常路径都恢复。

原生 `call(vector<Tensor *>)` 的参数顺序如下，未注明者均为只读输入：

| operation@1 | 参数顺序及输出 |
|---|---|
| dense_linear | input, weight, output |
| grouped_linear | input, grouped_weight, output |
| row_inv_rms | input, output |
| rms_norm | input, FP32_weight, output |
| hc_sum | FP32_input, output；归约轴在构造时绑定 |
| rotary_inplace | input_output, complex64_frequencies；正/逆向在构造时绑定 |
| indexer_scores | query, compressed_keys, head_weights, output |
| indexer_topk | scores, output_ids；k 由输出 shape 指定，tie 策略显式绑定 |
| router | FP32_logits, bias 或 hash_ids, output_weights, output_ids |
| compressor_pool | FP32_values, FP32_scores, output |
| expert_activation | BF16_gate, BF16_up, [FP32_routing_weights], BF16_output |
| moe_dispatch | hidden, route_weights, expert_ids, packed_hidden, packed_weights, packed_rows, offsets；后四项为输出 |
| moe_combine | packed_expert_output, packed_rows, offsets, FP32_output |
| tensor_cast | input, output；BF16/FP32 或 int32/int64 同类别转换，I64→I32 检查溢出 |
| row_gather | table, integer_row_ids, output；越界显式失败 |
| moe_finalize | FP32_routed_output, BF16_shared_output, BF16_output |

前三个新组合工具不改变已有 19 类 Python 模型契约。router 的 score function、
route scale，以及激活的 clamp limit 由 `StructuralOptions` 显式绑定；hash IDs
必须在范围内且每个 token 不重复。动态 ID 校验、reference dispatch/combine 仍有
host 同步，不能把这个后端报告为 device-only 或 CUDA Graph 路径。

`src/backends/hadamard/native_kernel.*` 的 `cuda-hadamard/hadamard@1` 直接链接
未修改的 fast-hadamard-transform CUDA object，未链接其 Python extension。
接口是 `call({input, output})`，scale 在构造时固定；当前支持连续 BF16/FP32、
8～32768 的 2 次幂宽度、无输入输出别名。测试覆盖实际 Indexer 宽度 128 及宽度
512。保留上游 BSD-3-Clause 归属，不将其实现称为项目自研 kernel。

### 完整单卡 MoE 的模型接口

`src/models/deepseek_v4/moe.*` 新增 `Expert`、`MoE` 和显式 workspace：

1. BF16 hidden 转 FP32 后进行 gate 投影；hash 层按 token ID 查询路由表，其余层
   使用 learned top-k。实际 MP1 路由表是 I64，官方模型参数是 I32，因此加载期通过
   受检整数转换准备路由表；不改变 checkpoint，也不静默截断。
2. dispatch 保留每个 token 的 top-k 位置，按 expert ID 排序，生成 packed hidden、
   routing weight、原 token 行号和 offsets。模型层显式下载 offsets，按 expert ID
   递增执行全部非空专家，不接管 Python scheduler。
3. routed expert 用 W4A8，shared expert 用 W8A8；每个 Expert 的 w1/w3/w2 分别持有
   `QuantizedLinear`。FP32 routing weight 在 SwiGLU 之后、BF16 cast 与 down GEMM
   之前相乘，不能改为最后 combine 才乘。
4. routed 输出按 expert ID 顺序 FP32 累加，加 BF16 shared 输出后再转 BF16。

模型接口不引用 ATen/TileLang/Python 类型；调用方通过 `MoeBackends` 注入
cast/dense/gather/router/dispatch/combine/finalize，通过 `Expert` 注入三个量化
Linear 和 activation。量化及 GEMM 仍可分别替换，缺失或不匹配的契约在构造时拒绝。
这里保留逐专家 reference execution；未来 grouped GEMM 或 EP 需要显式增加执行
策略/通信接口，不会由 backend 名称暗中切换。

`MoeWorkspace` 持有 packed buffers、FP32 累加输出及按专家 token 数缓存的
`ExpertWorkspace`。同形状重入复用 scratch；每次成功执行的 offset 下载与专家执行
次数可读。backend 异常后 workspace 标记失效并拒绝重入；调用方应在所属线程等待
stream 完成后回收它，而不是带着部分写入继续执行下一请求。

验证器 `verify_native_moe.py` 链接 xmake 的实际静态库，在不加载 Python 的子进程中
完成上述整条 MoE 链路。参考直接调用同版本官方 `MoE.forward`，没有以项目自己的
MoE 实现生成 golden。第 0/3 层全部 256 routed experts 和 1 shared expert 均加载；
测试输入是确定性随机 hidden/token IDs，**不是 43 层端到端文本验收**。最新作业号、
用例计数、权重摘要和构建命令以工程记录与 native_moe JSON 为准。

## 原生 Compressor 与带标量的调用合同（2026-09-08）

`src/models/deepseek_v4/compressor.*` 将现有压缩模型语义迁入 C++，包括实际三种组件：
ratio=4、dimension=512 的重叠 latent 压缩；ratio=128、dimension=512 的普通压缩；
ratio=4、dimension=128、Hadamard/FP4 QDQ 的 Indexer 压缩。不是传统 MHA K/V 布局，
也不把这些模型维度引入 block manager。

### 模型、请求状态与输出所有权

- `Compressor` 持有投影、APE、norm 权重、频率 tensor 与显式后端。实际 checkpoint
  中的 BF16/FP32 参数在加载期通过 `tensor_cast` 准备 FP32 计算参数；不改写源权重。
- `CompressorState` 持有 KV/score 未完成分组及 ratio=4 的前一组重叠状态。必须由
  所属模型 `reset` 初始化；每次执行校验 owner、position、输入规模和频率容量。
- `CompressorWorkspace` 持有 FP32 投影/分组/池化结果、BF16 norm 与 QDQ staging。
  `forward` 不用逐 token 循环模拟多 token chunk，而是一次处理完整当前 chunk。
- 返回 `CompressedChunk{first_group,count,values}`。`values` 借用调用方 workspace，
  必须保活到下游 cache write 和当前 stream 完成；`count=0` 表示只更新未完成分组。
  压缩器不分配 block、不持有 paged pool，也不决定 Prefix Cache 共享策略。
- Attention 的 cache writer 根据 first_group/count 选择连续或 paged 写入。当前
  原生验证器写入受检连续 cache view；还不能据此宣称已经接入原生 paged Attention。

完整算子链是：BF16→FP32、两次投影、分组/状态准备、softmax pooling、转 BF16、
RMSNorm、尾部 RoPE，再执行非 RoPE FP8 QDQ 或 Hadamard+FP4 QDQ。
non-RoPE 切片在 TileLang 调用前显式整理为连续矩阵，计算后写回原视图；不能把
stride 为 512、有效宽度为 448 的切片直接作为连续 448 列输入。

### 新增标量元数据入口

`Kernel::callWithScalars(arguments, CallScalars)` 提供 C++ 已知的 launch 元数据，
`CallScalars` 仅含 `integers` 和 `reals` 两个向量，不涉及 Python 或 device→host 读取。
每个版本化操作定义参数位置与含义，未知参数显式拒绝。旧操作默认只接受空标量；
不能通过这个入口改变旧 kernel 的算法或自动选择其他后端。

| operation@1 | tensor 参数 | scalars |
|---|---|---|
| tensor_fill | output（BF16/FP32，原地写入） | integers 为空；reals={value}，允许 ±inf，不允许 NaN |
| compressor_prepare | projected_kv, projected_scores, ape, kv_state, score_state, grouped_kv, grouped_scores；state 原地更新，最后两项为输出 | integers={position,ratio}；reals 为空 |

`compressor_prepare` 的投影为 `[1,S,coff*D]`、APE 为 `[ratio,coff*D]`、state 为
`[1,coff*ratio,coff*D]`，其中 ratio=4 时 coff=2，否则 coff=1。初始 prefill 和
multi-token chunk 输出 `[1,complete,coff*ratio,D]`；单 token decode 命中压缩边界时
输出 `[1,coff*ratio,D]`，保留原始 pooling rank 与舍入语义。状态更新仍在明确指定的
Runtime stream 上执行，backend 只解释压缩组布局，不决定 serving token budget。

本次为 C++ Kernel 接口增加了虚函数，所有原生 host backend/model 静态库均已重编译。
不要将未重新编译的旧后端二进制与新头文件混用；尚未承诺跨版本 C++ 二进制 ABI。
现有 operation 的数学合同未改变，原生 MoE、结构算子及 TileLang/Linear 均需回归。

### 错误语义与验证范围

错误 position、未初始化或其他模型的 state，在排队前拒绝。计算阶段异常将 state
及 workspace 一并置为失效，后续拒绝重入。显式 reset 清理压缩器状态后，必须换
新的 workspace；Attention/request 层还需按其所有权合同处理自己的 cache，不能
把压缩器 reset 当作整个请求缓存已经清空。

`verify_native_compressor.py` 分开标记两类 oracle：

- `published`：未修改的官方 `Compressor.forward`，覆盖初始 prefill 和单 token decode。
- `incremental_python`：项目已有增量 Compressor，覆盖相同多 token chunk 调度；
  不是将项目 chunk 输出重新命名为官方 full-prefill golden。

使用真实三组权重、重复 reset/replay、非对齐 chunk、大 chunk 后 decode 边界以及
部分状态写入后的失败恢复。参数、投影、状态、norm 输入、输出和 cache 均可逐字节
检查。频率 tensor 目前由离线准备阶段的官方函数生成；完整 C++ 配置/权重入口与
Attention 组合仍待接通。该组件验收不改变整模型原始 full/chunk 精度失败门槛，
也不提供 TTFT/TPOT、CUDA Graph 或多卡结论。

## 原生 Indexer 组合与后端替换（2026-09-08）

`src/models/deepseek_v4/indexer.*` 新增完整单请求 C++ Indexer，保留实际 V4 的
W8A8 查询投影、RoPE、Hadamard、FP4 QDQ、独立 Compressor、BF16 head-weight
投影/缩放、BF16 打分、因果 mask、top-k 与 offset 映射。不是用普通 MHA 替代
压缩稀疏注意力；Indexer 只输出候选位置，真正 Attention 仍由下游消费。

### 配置、所有权与调用

- `IndexerConfig` 指定 hidden/query rank、heads、head dimension、RoPE dimension、
  top-k 和容量。当前单卡、batch=1、ratio=4；实际测试为 4096/1024/64/128/64/512。
- `IndexerWeights` 持有七个实际参数及频率 tensor。查询投影保留 FP8 权重/UE8M0
  scale，head projection 为 BF16，Compressor 的 FP32 转换只在模型构造时执行。
- `IndexerBackends` 显式注入 query activation quant/GEMM、rotary、Hadamard、
  FP4 QDQ、cast/fill/dense/scale、scores/mask/topk/remap 和 `CompressorBackends`。
  量化与 GEMM 可分别替换。模型头文件没有 Torch/TileLang 类型，不调用 Python。
- `IndexerState` 保留独立 Compressor 状态和连续 BF16 压缩 cache，必须先由模型
  `reset`；此处的连续存储是明确的迁移参考布局，不是新的 paged allocator。
- `IndexerWorkspace(runtime, config, position, tokens)` 持有当前形状的查询、量化、
  投影、分数、I64 top-k 和 I32 最终输出，复用已完成的 `LinearWorkspace` 与
  `CompressorWorkspace`。长 chunk 一次批量执行，不用逐 token replay 模拟。
- `forward(hidden, query_rank, offset, state, workspace)` 返回借用的 I32 tensor，
  shape 为 `[1,S,min(topk,(position+S)/4)]`；无完整压缩组时末维为 0，状态仍前进。
  `offset` 由 Attention 的 payload 布局决定，不能由 block manager 猜测。

原生调用中没有 token 级 Python/C++ 往返，但显式 ATen 参考后端仍有临时分配、
输出复制及 ID 校验同步，不宣称 CUDA Graph 或最优性能。frequency 仍由离线准备的
官方函数生成；原生 paged Indexer 存储、完整 Attention 和模型加载入口另行迁移。

### 新增原生合同

下列操作继续使用 `Kernel::callWithScalars`，不改变已有 19 类 Python 模型合同：

| operation@1 | tensor 参数 | scalars 与数值语义 |
|---|---|---|
| tensor_scale | BF16/FP32 input, 同 shape/dtype output | integers 为空；reals={finite_scale}；乘法保留输入 dtype 的舍入，禁止别名 |
| indexer_mask | BF16 scores，原地写入 | integers={position,4}；reals 为空；shape `[1,S,(position+S)/4]`；初始或多 token chunk 的未来组加 -inf，后续单 token decode 不改分数 |
| indexer_remap | I64 selected_ids, I32 output | integers={position,4,offset}；reals 为空；越界 ID/offset 显式失败；未来组输出 -1，有效组加 offset |

因果组数使用整数 floor-divide，不能让大 position 经 FP32 中转而丢失边界。
top-k 仍单独调用 `indexer_topk`：当前验收使用 published tie policy，不暗中改为
stable/index-ascending；将来更换 tie policy 必须独立报告数值策略和精度。

### 错误与验证口径

错误 owner、position、输入 dtype/shape、offset 溢出、跨线程和输入/可写 buffer
别名在执行前拒绝。后端失败可能已经写入 Compressor/cache；此时整个 Indexer
state 和 workspace 失效，不能继续 decode。显式 reset 清空本 Indexer 的连续 cache
及 Compressor，然后必须使用新 workspace；这仍不是整个 Attention/request reset。

`verify_native_indexer.py` 读取实际第 2/42 层全部 Indexer 参数。最终候选和缓存
分别对照未修改的官方 `Indexer.forward`（初始 prefill、单 token decode），以及
项目现有增量 Indexer（相同多 token chunk 调度）。query 诊断额外由 hook 捕获的
投影结合原始 RoPE/Hadamard/quant 计算，明确不是第二份整模型 golden。
Top-512 使用 2105-token 输入；同时覆盖短于压缩组、非对齐 chunk、不同 offset、
reset/replay、部分 cache 写入后的失败恢复及元数据合同负例。输入是确定性随机
hidden/query，不是文本端到端测试，不改变原始 full/chunk 精度门槛。

## 原生完整 Attention 组合（2026-09-08）

`src/models/deepseek_v4/attention.*` 提供单请求的完整 Attention 组件，支持实际
ratio=0 的纯滑窗、ratio=4 的 Compressor+Indexer 和 ratio=128 的压缩注意力。
它复用已完成的原生量化 Linear、Compressor、Indexer，不导入上游 Python model，
不实现 serving 队列，也不把 V4 latent layout 暴露给 block manager。

### 后端注入与执行顺序

`AttentionBackends` 分别接收 wq_a/wq_b/wkv/wo_b 的 activation-quant 与 GEMM，
以及 cast/fill/RMSNorm/inverse-RMS/row-multiply、正/逆 RoPE、latent QDQ、
metadata/cache prepare、sparse attention、grouped projection 和已有子组件后端。
`AttentionWeights` 按实际参数名称保存 Tensor owner；根据 ratio 严格校验所需的
12/23/16 个参数，拒绝缺失、额外参数及 shape/dtype/runtime 错误。

执行顺序为：

1. W8A8 wq_a → RMSNorm → W8A8 wq_b → BF16 inverse-RMS/乘法 → query RoPE。
2. W8A8 wkv → RMSNorm → latent RoPE → non-RoPE 部分 FP8 QDQ，block=64。
3. 可选 Indexer 与 Compressor 更新；候选 offset 取决于当前 payload 布局。
4. 准备候选及连续/环形 payload → TileLang sparse attention（含实际 FP32 sink）。
5. attention output 逆 RoPE → BF16 grouped wo_a → W8A8 wo_b。

query 的 square/mean/rsqrt 与乘法保留 BF16 舍入边界，不能与 FP32 RMSNorm 混为
一个契约。non-RoPE latent 使用显式连续 QDQ staging 后写回；RoPE 部分不量化。
grouped wo_a 使用 MP1 已转换的 BF16 权重，不在运行中把 FP8 GEMM 静默改为解量化。

### 请求与 workspace

`AttentionState` 持有连续参考 cache（滑窗 ring + 压缩组）、可选 Compressor/
Indexer state；`AttentionWorkspace` 持有当前 shape 的投影、QDQ、候选、payload、
输出以及子 workspace。输入是 `[1,S,hidden]` BF16；输出借用 workspace，同 shape。
一个 `forward` 在 C++ 内完成上述整条计算链，没有逐算子 Python callback。

当前数据路径明确为连续参考布局：

- 初始 prefill：当前 S 个 latent 加已完成压缩组；同时更新 ring 中最后一个窗口。
- 后续单 token decode：直接读取 ring + 压缩缓存，不构造第二份完整 payload。
- 后续多 token chunk：显式 gather 至多 window-1 个旧 latent，再拼接当前 chunk
  和当前压缩组；ring 只写最后 min(S,window) 个 token，避免重复写入物理槽。

这不是原生 paged Attention 的完成声明。既有 C++ paged pool/lease 仍需接入
原生模型，不能因参考 prepare 使用 gather 就撤销已完成的 Python direct-paged
执行路径。当前 metadata 数值策略固定为 published，不暗中启用 fixed padding。

### 新增原生合同

| operation@1 | tensor 参数 | scalars / 语义 |
|---|---|---|
| tensor_row_multiply | 可写 BF16/FP32 input、只读同 dtype factors | 无 scalars；factors 与 input 前缀 shape 相同、末维为 1；原地乘法；禁止 factors 别名 |
| attention_prepare | 只读 current_latent、可写 ring_and_compressed_cache、只读 indexer_candidates、可写 payload、可写 final_candidates | integers={position,window,ratio,index_topk}；reals 为空；限定 batch=1，ratio=0/4/128；显式连续参考准备 |

`attention_prepare` 中 current_latent 是 `[1,S,D]` BF16，cache 为
`[1,window+capacity/ratio,D]`（ratio=0 时只有 window），Indexer candidates
只在 ratio=4 时非空。最终 I32 candidates 保留有效窗口优先、-1 padding 在后的
原始排列；ratio=4 保留 Indexer top-k 顺序，ratio=128 按因果可见组生成候选。
初始 payload 是 `[1,S+groups,D]`，chunk payload 是 `[1,history+S+groups,D]`；
单 token decode 的 payload 输出零元素，下游直接使用 cache。

准备后端会校验候选范围、尺寸、整数溢出和读写别名；显式 ATen 参考实现包含
临时分配、ID 检查同步和 chunk gather，不声称 fused/device-only 或 CUDA Graph。

### 生命周期与验收边界

state 必须由所属模型 reset；position 单调递进，重复/跳跃执行、错误线程、输入
别名或缺失子状态应明确拒绝。后端错误可能发生在 Compressor/Indexer/cache 已经
写入后，整个 Attention state 和 workspace 一并失效。reset 清理全部所属子状态和
连续 cache，再使用新的 workspace；不能单独恢复某个子组件后继续旧 Attention。

`verify_native_attention.py` 覆盖真实第 0/2/3 层全部参数，比较最终输出、完整
cache、query rank、latent 投影、grouped 输出以及两类 Compressor/Indexer 状态。
初始/decode 使用未修改的官方 Attention；多 token chunk 使用已有项目的同分块
参考，明确不是原始 full golden。该组件验证不替代 HC/完整 Transformer block、
43 层文本推理、任务精度、原生 paged storage、TTFT/TPOT 或多卡验收。

## 原生 HC 与完整 Transformer Block（2026-09-08）

### HC 模型组合与独立后端

`src/models/deepseek_v4/hyperconnection.*` 新增 `HyperConnection`、`HCWeights`、
`HCWorkspace` 和 `HCBackends`。实际 hidden=4096、HC copies=4；支持 block 的
pre/post 配对执行，以及模型输出端的 HC head。模型接口不引用 Torch、TileLang 或
Python 类型，不创建另一套 scheduler，也不把 HC 当作普通 residual add。

执行顺序保持同版本官方实现：

1. BF16 `[1,S,C,D]` 展平为 `[1,S,C*D]`，显式转换 FP32。
2. FP32 inverse RMS、FP32 dense projection、逐行乘法得到 mixes。
3. block 分支调用上游 TileLang Sinkhorn，产生 FP32 pre/post/combination；
   head 分支使用 FP32 sigmoid/epsilon，不执行 Sinkhorn。
4. pre 的 FP32 加权归约结束后才转回 BF16；post 的分支扩展与残差矩阵乘加也在
   FP32 完成，再写入 BF16 隐藏状态。

`HCBackends` 分别注入 cast、dense、inverse-RMS、逐行乘法、Sinkhorn、pre 归约、
post 混合、head 权重计算。归一化 epsilon、Sinkhorn 次数/epsilon、head epsilon
在构造后端时显式绑定。Sinkhorn 使用 TileLang；新增组合算子的默认实现是 C++
ATen 参考后端，不宣称自研 fused CUDA。当前共有 26 类 ATen 原生 operation，原有
Python 19 类模型接口未被删除或替换。

新增三个 revision=1 原生算子合同：

| operation | 输入 | 输出与计算边界 |
|---|---|---|
| `hc_pre_reduce` | FP32 X `[B,S,C,D]`、pre `[B,S,C]` | BF16 `[B,S,D]`；先 FP32 乘法，再沿 C 归约，最后转换 |
| `hc_post_mix` | BF16 branch `[B,S,D]`、residual `[B,S,C,D]`；FP32 post `[B,S,C]`、comb `[B,S,C,C]` | BF16 `[B,S,C,D]`；comb 的倒数第二轴是源 copy，最后轴是目标 copy，不能转置 |
| `hc_head_weights` | FP32 mixes `[B,S,C]`、scale `[1]`、base `[C]` | FP32 `[B,S,C]`，`sigmoid(mixes*scale+base)+epsilon` |

三者不接受额外 launch scalar，输出不能与只读输入重叠。ATen 临时分配和输出
copy 是当前显式参考成本；不宣称零临时内存、CUDA Graph 或 fused 性能。

HC workspace 保留 pre 输入的 storage owner 供 post 使用，不复制整份 residual。
调用方在配对执行结束前不能修改该输入。重复 pre、缺少 pre 的 post、重复 post、
跨模型 workspace、跨线程调用及别名输入均拒绝。后端异常会使 workspace 失效，
必须创建新的 workspace；不能把已产生部分结果的旧 workspace 直接重用。
返回 tensor 借用 workspace，调用方需保留它直到 stream 完成。

### 完整原生 Block

`src/models/deepseek_v4/block.*` 组合已验证的 HC、Attention、MoE：

```text
BF16 HC hidden
  → HC pre → RMSNorm → Attention → HC post
  → HC pre → RMSNorm → MoE       → HC post
  → BF16 HC hidden
```

`Block` 独占 Attention/MoE 模型组件；`BlockState` 持有请求 Attention 状态；
`BlockWorkspace` 持有两套 HC、Attention、MoE 和归一化中间结果。
`BlockBackends` 保留两组 HC、cast、norm 接口，子组件继续各自持有可替换的
量化/GEMM、稀疏 attention、router/dispatch/combine 等后端。
norm 权重的 BF16→FP32 转换发生在装载阶段，不在每个 token 中重复进行。

整层成功后才推进 Block position。如果 Attention 已写 cache，而后续 MoE 或 HC
失败，Block state/workspace 都失效，即使其内部 Attention position 已前进。
恢复必须 reset 完整 Block state，并使用新的 workspace；不能从中间算子接着跑。
输入 hidden/token IDs 与可写的 HC、MoE、Attention/Indexer/Compressor workspace
及请求 cache 的别名会在执行前被拒绝，防止 MoE packing 或元数据写入污染输入。

此层接收 `[1,S,C,D]` BF16 hidden 和 `[1,S]` I32/I64 token IDs，不做 tokenizer、
embedding、词表输出或采样。token ID 的整模型词表校验仍属于入口层；hash MoE
查表会检查其实际范围，learned MoE 不使用 token ID 的值。

### 本阶段验证口径

`verify_native_hc.py` 使用第 0/2/3/42 层的两组 HC 及模型 HC head 实际参数；
直接调用未修改的官方 `Block.hc_pre/hc_post/hc_head` 生成 golden。
`verify_native_block.py` 严格加载第 0/2/3 层全部参数，包括每层 256 routed expert
和 1 shared expert；初始 prefill/decode 对照未修改的 `Block.forward`，多 token
chunk 对照项目已有相同 chunk 调度的 Python Block。原生子进程不加载 Python。

fixture 装载器只用于测试，不是完整模型 checkpoint loader。当前 Block Attention
仍使用连续/ring 参考布局；完整 43 层 C++ 模型、原生 paged/lease 接入、pybind batch、
HTTP、任务统计精度和性能、多卡 TP/EP 必须继续验收。相同 chunk 对照通过不改变
原始 full/chunk 数值不一致的已知结论。

## 原生整模型与批调用接口（2026-09-09）

`ModelConfig` / `Checkpoint` / `Model` 已把前述组件组合成完整 MP1 主模型，而非
fixture 装载器。C++ 生成 RoPE 表并执行 embedding、全部 Block 和词表输出。
Slurm 23085 的 2105 输入 / 16 输出、两次请求重置，32 步 logits 全部逐字节对齐
官方。该进程没有 Python。MTP 仍只有结构校验，没有推测执行。

`ModelBackends.layers[i]` 和 `.global` 是显式 `KernelBindings`；每项持有
`shared_ptr<backends::native::Kernel>`。shape 特化使用明确的 key，例如：

- `quant_hidden` / `quant_query_rank` / `quant_intermediate` / `quant_output`；
- `gemm_query_a` / `gemm_query_b` / `gemm_latent` / `gemm_output`；
- `fp4_gate` / `fp4_down` / `fp8_gate` / `fp8_down`；
- `sparse_attn` / `hc_split_sinkhorn` / `router` / `moe_dispatch` / `moe_combine`。

key 表示模型调用位置，`KernelIdentity.operation` 表示版本化算子合同。替换某个
shape 特化时不必替换其他形状；同一 kernel 可由多层共享，但必须遵守所属 runtime、
stream、输入/输出 dtype/layout 和生命周期合同。缺失 key、错误 operation/revision
或 runtime 不一致均报错，不临时改选后端。

`src/backends/v4_reference.*` 是默认组装器，不是模型层。用户后续可在 C++ 中提供
`Session::BackendFactory`，先调用默认组装器，再把某些绑定换成自己实现的 Kernel；
或者整体返回自己的 `ModelBackends`。接口不暴露 Torch/TileLang 的类型，不需要在
`qwen2.cpp` 添加 DeepSeek 算子，也不需要变更 Python scheduler。

`Session` 接受原有 `engine::SchedulePlan`，其专用线程只顺序执行调用并维护模型/
slot 的资源所有权，不负责 admission、token budget 或优先级。执行顺序为计划中的
reset、prefill、decode；先验证完整计划再修改状态。若执行中失败，丢弃所有此次
触及的 slot，不把先前已推进却没返回结果的请求留给调用方盲目重试。

独立 `_v4_native.so` 暴露 `Session.execute(plan, capture_logits=False)`，plan 使用
与上述 C++ 合同一致的 dict。Python 在一次调用外组织计划；C++ 内完成完整模型
执行，无逐算子 Python callback。正常输出为 request/slot/token IDs；仅诊断模式
下载整份 logits。构建产物暂留 build 目录；现有 serving facade 的原生接入见下节。

当前 Session 默认仅接受 greedy；多 slot 串行、连续/ring cache，不冒充 mixed
GPU batch。关闭、构造失败及所有 tensor/kernel 析构发生在拥有 runtime 的执行
线程；对外调用不依赖 Python 调用线程与该线程相同。原始 full/chunk gate 未通过，
因此非完整初始 prefill 需显式实验开关，不能通过 API 自动改变数值策略。

原生批接口已通过 Slurm 23093 的八个真实对话、352 步 logits 字节对照，包含不同
prompt 的两 slot、部分结束回收、GIL 释放与跨调用线程 close。报告见工程记录。

常规使用不需要测试 fixture 或 golden：在 Slurm 中运行
`tools/deepseek_v4_reference/export_native_backends.py` 得到版本化 `bundle.json`，
再用 `deepseek_v4_model.native_session.load_native_session` 装载真实模型。
现有 CLI `run_native_chat.py` 演示编码、原生 prefill/decode、EOS、解析与资源回收；
Slurm 23094 的新问题自由生成已通过。上游 kernel DSO 是可执行代码，只加载可信
本地产物；模型/硬件/ABI/摘要检查用于防止误配，不等于恶意代码沙箱。

## 原生 serving facade 与可观测性（2026-09-12）

`DeepSeekV4NativeServingModel` 消费已加载的 Session，不复制模型、不引入算子级
Python 往返。一次多 slot decode 转为一份原生 SchedulePlan。用法如下：

```python
from llaisys.models.deepseek_v4_model.native_batch import DeepSeekV4NativeServingModel
from server.engine import InferenceEngine, SamplingParams

facade = DeepSeekV4NativeServingModel(session, eos_id=codec.eos_id)
capacity = session.info()["capacity"]
engine = InferenceEngine(facade, max_batch_size=2, max_seq_per_slot=capacity,
                         prefill_chunk_size=capacity, enable_block_prefix_cache=False)
engine.start()
# 在 asyncio event loop 中 submit / 消费 stream_tokens / await future。
# 先停止 engine，再关闭调用方拥有的 session；不要并发直接操作同一 Session。
```

`Session.info()["operators"]` 按 backend/version/operation/contract_revision 聚合
真实调用数和失败数。同一个 Kernel 对象绑定到多层时只计一次，避免重复累加。
每次绑定都是显式选择，运行中不自动切换后端。正常路径只返回 token；仅提供
observer 时下载 logits，不能用这种观察模式报告正式 TTFT/TPOT。

当前原生 facade 明确报告 continuous/ring、paged=false、Prefix=false、preemption=false、
fused_gpu_batch=false；未支持的 engine 策略提前报错。Slurm 23097 的现有 worker
流式执行已对齐 352 步官方 logits，并通过取消/异常后恢复；Slurm 26564 提供首组
无观察器性能矩阵。详见中文工程记录，长输入波动与采样显存限制均保留。

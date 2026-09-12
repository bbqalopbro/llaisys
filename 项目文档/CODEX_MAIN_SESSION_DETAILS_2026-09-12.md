# 主长会话补充记录与调试证据（2026-09-12）

本文从本地会话记录恢复工程事实，并与当前源码和 JSON 报告核对。初稿核对时间为
2026-09-12 06:05 UTC（北京时间 14:05）；会话仍在继续，本文不把已启动的测试写成
完成。`B300_DEEPSEEK_V4_BRINGUP.md` 已有完整主体，本文补充定位索引、遗漏的环境
排错过程和最新性能测试的验收口径，不重复搬运全部工程记录。

最后增量检查为 2026-09-12 06:45 UTC：M 仍为 9541 个物理行，末尾时间
06:16:27.799 UTC。末尾仅发出的工具调用不作为完成依据；本文的 benchmark 完成
结论仍以已返回的 exit 0 和报告内容为准。

会话原文件保留在本机，不纳入仓库。下文的 `M:L9354` 表示会话 M 的 JSONL **物理
行号**，不是事件 ordinal。仅引用公开对话、工具调用和返回结果，不复制内部推理、
认证信息、SSH 公钥、其他用户目录、设备 UUID 或网络地址。时间均为 UTC；北京时间
为 UTC+8。

| 标记 | 会话 ID | 本地文件名 |
|---|---|---|
| M | `01a05815-0951-7743-a2a6-89877d637d7d` | `rollout-2026-08-31T13-49-28-01a05815-0951-7743-a2a6-89877d637d7d.jsonl` |
| Q | `01a0399b-78c7-7f43-b178-d89f03914156` | `rollout-2026-08-25T15-48-05-01a0399b-78c7-7f43-b178-d89f03914156.jsonl` |
| W | `01a039aa-79db-7c92-ac17-5dfb4c26d1a3` | `rollout-2026-08-25T16-04-28-01a039aa-79db-7c92-ac17-5dfb4c26d1a3.jsonl` |
| N | `01a05833-6921-7f20-8752-32161ab460c9` | `rollout-2026-08-31T14-22-39-01a05833-6921-7f20-8752-32161ab460c9.jsonl` |
| R | `01a09431-21a8-7ec0-9aaf-40ed1a1d16f8` | `rollout-2026-09-12T05-57-22-01a09431-21a8-7ec0-9aaf-40ed1a1d16f8.jsonl`；M 的原生 benchmark 审查子会话 |

文件按开始日期存放在本地 `.codex/sessions/YYYY/MM/DD/`。M 虽开始于 8 月 31 日，
仍包含 9 月 1 日、8 日、9 日和 12 日的工作；仅根据目录日期筛选会遗漏主线进展。

## 已完成成果的证据入口

以下项目现已在[主工程记录](B300_DEEPSEEK_V4_BRINGUP.md)中说明，本文补齐会话
时间和验收产物之间的对应关系。不同层次的通过数量不能累加成一个统一测试总数。

| 完成范围 | 会话证据 | 报告与边界 |
|---|---|---|
| C++ 正式配置/MP1 loader、43 层完整执行 | M:L8883，2026-09-09 00:21:06；Slurm 23085 | [native_model_top512](../benchmark_results/deepseek_v4_b300_native_model_top512.json)：2105 输入/16 输出，两次请求重置；32 步 logits 与生成 token 一致，含两张 RoPE 表共 34 次 tensor 字节比较。主模型 67569 个 tensor，MTP 4702 个 tensor 仅做结构校验。 |
| 独立 pybind `Session.execute(SchedulePlan)` | M:L9090，2026-09-09 00:42:34；Slurm 23093 | [native_session_chat](../benchmark_results/deepseek_v4_b300_native_session_chat.json)：8 个对话、352 步 logits、343 个执行计划、13 项负例；两个 slot 串行执行，最终 slot 归零。 |
| 不依赖 golden 的后端导出/原生对话入口 | M:L9263，2026-09-09 00:56:23；Slurm 23094 | [native_chat_smoke](../benchmark_results/deepseek_v4_b300_native_chat_smoke.json)：实际生成 `5` 后 EOS，正常解析与回收；`golden_required=false`、`teacher_forcing=false`。 |
| 整模型和八组原生组件联合回归收尾 | M:L9263；M:L9408，2026-09-12 05:58:58 补记 | Slurm 23094 正常退出。整模型 32 步 logits；HC/Block/Attention/Indexer/Compressor/MoE/结构/TileLang 为 1073/899/1106/614/705/170/374/193 次 tensor 字节比较。对应报告绑定各自执行时的源码。 |
| 既有 `InferenceEngine` 接原生 C++ Session | M:L9354，2026-09-09 01:08:02 的命令完成结果；M:L9377，2026-09-12 05:57:17 的报告复核 | [native_serving_chat](../benchmark_results/deepseek_v4_b300_native_serving_chat.json)：Slurm 23097，8 对话、352 步原始 golden logits、实际生成与流式 token 一致；取消、诊断回调异常后恢复通过。 |

本次独立核对 `native_serving_chat` 的 `provenance.file_sha256`，95 个文件、共
41,029,174 字节全部匹配，未发现缺失或摘要变化；报告 SHA256 为
`fd5b84fc84667be09cff02bd179ad227ac62f31c4bed1f3a874ba3efd6b5ebca`。
这是源码、模块、bundle 等被记录文件的核对；本次未重扫 168 GB checkpoint，完整
payload 的历史校验证据仍由报告中的 `checkpoint_identity` 和原始执行记录提供。

### 避免误读 serving 报告中的数字

- `logits_byte_comparisons=352` 是八个对话的逐步数值对照数。
- `model_calls=356` 还包含生命周期测试的四次模型执行；取消与诊断异常后恢复也会
  真正执行模型。它不能被写成“356 步 golden 比较”。
- `steps=389` 是 Session 计划计数，包含 reset/回收等计划；不等于输出 token 数。
- `sparse_attn` 的 15,308 次调用等于 `356×43`。共享 Kernel 按对象先去重再汇总，
  避免把同一个后端绑定到多层所产生的重复引用当成重复执行。
- `model_failures=0` 与 observer 故障并不冲突。M:L9354 的 traceback 来自刻意注入的
  `injected diagnostic failure after native execution`，发生在 Python 观察回调中，
  不证明经历过 CUDA kernel、device fault 或 allocator 故障注入。
- 八例数值/集成全部通过，不等于模型任务质量 8/8。既有严格任务检查仍为 5/8，包含
  官方参考输出自身的格式问题及 256-token 截断；没有改变任务判定。

上述实现使用 continuous/ring cache、原生串行 slots；原生 paged/Prefix、抢占快照、
HTTP、fused GPU batch 和 TP/EP 仍须单独验收。Python paged correctness executor
曾通过的结果不能自动迁移为原生 Session 的能力证明。

## 从首次失败到恢复：需要保留的因果链

### 最终 CUDA 二进制的架构不能只看编译命令

2026-08-31 第一次 B300 构建完成后，`cuobjdump` 仍发现 `sm_75`：静态 NVIDIA
target 设置了 `sm_103`，但顶层 shared target 再次编译的 CUDA/NCCL 源码没有继承
该参数。第一次补上编译参数后，编译行已显示 `-arch=sm_103`，最终 device-link
仍使用默认架构。最终同时修正 CUDA 编译与 device-link 参数，才得到只含 `sm_103`
的运行库，并完成 1 MiB H2D→D2D→D2H smoke。

证据：M:L412/L418（15:53:12/15:53:26，fatbin 为 `sm_75`），M:L432（15:54:18，
编译参数已正确但产物仍错误），M:L440（15:54:32，定位 device-link），M:L480
（15:56:24，修复后的运行验证）。当前构建位置为 `xmake.lua`、`xmake/nvidia.lua`。
这段已有工程说明，补充此索引便于追踪“两次修复”的区别。

### 正式 loader：存储 dtype 与计算 dtype 不是同一合同

M:L8818（2026-09-09 00:17:11）真实 checkpoint 验证首先报
`MP1 tensor dtype mismatch: layers.10.attn.compressor.wgate.weight`；
M:L8831（00:17:44）检查实际 safetensors header 后确认 Compressor 的投影以 BF16
存储。官方及已有 Compressor 在加载后以 FP32 计算，因此修复应针对允许的来源
dtype，而不是取消严格 dtype 检查。后续 Slurm 23085 才完成完整加载及 32 步比较。
这次失败发生于权重结构校验，不能归类为 GEMM 数值误差。

### 独立 executable 正常，不保证 pybind 的 DSO 加载也正常

1. M:L8971（2026-09-09 00:32:29）：首次 import `_v4_native.so` 缺少
   `llaisysGetRuntimeAPI`。M:L8977（00:32:46）确认提供方确实导出该符号；问题在
   ELF 链接依赖被 `--as-needed` 省略。仅在末尾加开关无效，必须保证保留库位于
   `--no-as-needed` 之后。
2. M:L9014（00:36:01）：修复 import 后，HC kernel 首次执行仍因
   `TVMBackendGetFuncFromEnv` 缺失而退出 127。M:L9027（00:36:32）检查导出 kernel
   的未定义符号，定位到 Python extension 局部符号作用域与缺失直接依赖的组合。
3. M:L9053（00:38:53）：原生 TileLang backend 通过已经链接的真实符号定位 runtime
   DSO，再显式提供全局符号与持有生命周期，随后 Slurm 23093 完成 352 步验证。

这两次链接修复已经进入主工程记录；此处保留失败顺序，避免把第二次错误误认为
第一次修复无效，或要求运行前先 import Python TileLang 来偶然补齐符号。

## 2026-09-12 正式性能测试启动前新增的排错

截至初稿时，正式性能 JSON 尚未形成完成结果。主窗口在 M:L9485（06:02:55）启动
`bench_native_serving.py`，计划矩阵为输入 32/256/2105 token、并发 1/2、最多输出
16 token、每格 warmup 1 次、测量 3 次。申请为单卡、8 CPU、96 GiB host memory、
40 分钟；只记录已启动，不能从命令推出已完成或性能好。

### GPU UUID 规范化与进程内状态丢失

第一次环境探针带入 `PYTHONPATH='undefined'`，在导入 torch 前失败（M:L9391，
05:57:38）。重新构造隔离路径后，torch 能识别 B300，但 `properties.uuid` 的文本
没有 `GPU-` 前缀，直接传入 `nvidia-smi -i` 返回退出码 6（M:L9398，05:57:56）。
因此“torch 识别 GPU 正常”不能推出“NVIDIA 管理工具接受相同标识”。

M:L9429（05:59:39）增加规范化：保留已有 `GPU-`/`MIG-`，否则添加 `GPU-`。
M:L9443（06:00:05）小型 Slurm probe 已成功读取显存样本并正常关闭，errors 为空。
该 smoke 只证明采样器基本可用，不能当成完整 benchmark 的覆盖率验收。

### 采样器曾有可能把不完整观测判成通过

初始判定只要求采样数非零和 errors 为空；若子进程提前正常 EOF，或中间长时间断档，
仍可能保留一个旧样本而通过。M:L9429 的修复增加：

- 非显式关闭时 EOF 必须写入错误；报告时线程与 `nvidia-smi` 子进程仍需存活。
- 采样区间 100 ms；把测量起点、样本、终点都纳入间隔检查。最大空档不得超过
  500 ms，记录首样本延迟、末样本年龄与 `coverage_valid`。
- `engine.stop()` 后检查 `_worker_error`；要求 runtime 报告存在，模型/算子失败为
  零，无 fallback、无 logits observer、无遗留 slot。
- 流式 token 必须等于 Future 返回值，输出不得超过预算；早于预算结束必须是 EOS。
- `Session.close()` 失败时仍用嵌套 finally 关闭采样器；失败报告保留已经完成的 case、
  checkpoint 身份和 provenance，不能被一个只有 error 的简短 JSON 覆盖。

本地代码位置：`tools/deepseek_v4_reference/bench_native_serving.py`。该脚本属于另一个
窗口的完整 V4 实现，本次文档与独立 GEMM 提交不包含其运行依赖。
M:L9432（05:59:44）语法检查通过；并未据此宣称统计测试或完整 GPU 性能矩阵通过。

### 性能数字应如何解释

脚本关闭 golden/observer/logits 下载，排除模型加载和 tokenization，以
`time.perf_counter()` 记录提交到 asyncio 流式 token 的时间。TTFT 包含排队；TPOT
是首 token 之后的到达间隔均值。`input_tokens/TTFT` 是有效 serving 速率，不是孤立
prefill kernel 的吞吐。尊重真实 EOS，报告实际输出长度；3 次重复的小样本分位数
只能用于描述，不能支撑稳定尾延迟结论。

显存来自设备范围的 `nvidia-smi` 100 ms 采样，可包含 C++ 原生分配，但也可能包含
同卡其他进程，且不是精确 allocator 高水位。多请求仍为原生串行 slots；并发 2
不能描述成 fused batch=2。正式结果还需检查源码前后摘要、采样覆盖、退出状态和
原始每次测量值。

### 审查子会话补充：启动失败也必须进入清理路径

R:L93（09-12 05:59:01）还复现了 `engine.start()` 位于 `try` 外的问题：启动超时
不会进入停止 worker 的 `finally`。修正应把启动本身纳入受保护区间，并在清理中
使用嵌套 `finally` 关闭 sampler。该项与 `engine.stop()` 不传播 worker 清理错误
是两个问题，不能只补后者就认定启动异常已覆盖。

R:L208（06:04:24）新增 `test_deepseek_v4_native_benchmark.py`，14 项 CPU 合同测试
覆盖采样 EOF/缺口、启动/停止失败、EOS 实际长度、单 token 时 TPOT=null，以及
分位数插值。该子会话只完成语法检查，明确未执行测试；不能将其最终答复本身当成
“14 项运行通过”的证据。这些测试也不能代替实机数值或 GPU 内存峰值验证。

### 原生 paged 后续设计：当时是只读方案，尚未落地

R:L171（06:01:44）给出了原生分页迁移审查意见，未修改文件或运行 GPU。其中两处
容易被后续实现遗漏的接口缺口是：Python cache binding 的 lease/pool/index 仍在
匿名 namespace，需要提取既有所有权实现供 Session 复用；native Tensor 需要持有
`PagedCacheStorage::componentPool()` 的共享 owner，`asStrided` 与 `overlaps` 也
必须保留 owner/别名关系，不能只包装裸 device pointer。

该方案要求复用当前 `InterleavedLayerLayout` 和统一逻辑 block IDs，不切换到指针
ABI 不同的另一套 layout。partial-block COW 要复制所有 43 层 component，包含
ratio-4 Indexer；完成 payload copy 和 device table 后才提交 lease replacement。
Prefix 恢复还需恢复 ratio-4 compressor 两路 overlap FP32 kv/scores，以及 Model、
Block、Attention、Compressor 各自的 position/owner/failed 状态。只恢复 block table
或顶层 position 不足以恢复模型。

这些是该时间点的设计/审查结论，未作为已实现成果计数；后续若执行，应独立覆盖
碎片化物理 block IDs、ratio=0/4/128、负候选、边界 token、双 slot 交错、失败回收
和 Prefix/COW。不得仅把 `paged=false` 改成 true，或用现有 Python paged 的通过
结果宣称原生分页已验收。

## 早期会话中未进入模型文档的环境经验

### 共享磁盘剩余容量不等于个人可写额度

Q:L37（2026-08-25 15:57:01）最初只查到共享文件系统尚余约 1.4 TiB，不能据此确认
个人可用空间。Q:L59（16:00:47）按不可见设备节点查询配额失败；Q:L65（16:01:03）
确认挂载启用用户配额，但 libc 未导出所需入口；Q:L71（16:01:15）通过目录文件
描述符读取内核配额成功：硬限制 100 GiB、软限制 95 GiB。这是当时的历史配置，
不能当作 9 月 12 日的最新配额查询。

W:L50（2026-08-25 16:05:33）已确认共享权重有完整 48 分片，约 155.43 GiB。
但 M:L1002（2026-08-31 17:11:16）与 M:L1154（17:17:48）仍以文件系统剩余空间
判断个人目录可容纳新增 MP1 文件，随后 M:L1178（17:19:37）实际收到
`Disk quota exceeded (os error 122)`。这说明“不同窗口已查明的 quota”没有自动
成为模型转换窗口的前置检查。最终在计算节点临时盘完成转换（M:L1224/L1252，
17:21:50/17:24:31）。复现时必须分别核对目标文件系统空间、个人 quota、节点本地
临时盘与实际目标路径；共享原始权重和新增 MP1 文件是两份占用。

### 远程终端卡顿的诊断并未证明服务器算力不足

N:L22/L36（2026-08-31 14:27:04/14:27:54）显示 CPU、内存和磁盘没有资源耗尽
特征；N:L49/L70（14:28:53/14:31:39）发现普通接入与 Tailscale 路径差异明显，
后者经 `DERP(sfo)` 中继，延迟 571–1256 ms 并伴随 TCP 重传。N:L107（15:06:04）
复测仍走中继，延迟 416–1172 ms，未证明已经恢复。

该会话完成的是只读定位与连接路径建议，没有重启、杀进程，也没有修复成功证据。
`ps` 在隔离环境报 `fatal library error, lookup self` 是另一类观测限制，不能拿来
解释模型计算慢。具体主机地址、客户端标识与连接信息不写入仓库。

## 本轮正式性能结果续记（09-12 06:15 UTC 复核）

初稿后，Slurm **26564** 已产出完整的
[native_serving_benchmark](../benchmark_results/deepseek_v4_b300_native_serving_benchmark.json)。
本窗口直接复核 6 个 case、全部原始 trial 和 79 个 provenance 文件摘要：
`all_passed=true`、`source_unchanged=true`，79 个文件全部匹配，6 个 case 的
`coverage_valid` 与 `repeated_greedy_ids_equal` 均为 true，采样器 errors 为空，
最终 active_slots=0、model_failures=0、无 fallback。报告 SHA256 为
`53474d23fde0d6dee14c8e6945f89ac6a6f31126cbc0022e9cfd618a2a699f81`。

| 输入 token | 并发 | TTFT p50（ms） | TPOT p50（ms） | 请求测量数 |
|---:|---:|---:|---:|---:|
| 32 | 1 | 1227.628 | 161.603 | 3 |
| 256 | 1 | 2271.856 | 162.217 | 3 |
| 2105 | 1 | 8200.510 | 359.193 | 3 |
| 32 | 2 | 2390.085 | 366.404 | 6 |
| 256 | 2 | 3407.000 | 398.832 | 6 |
| 2105 | 2 | 5594.444 | 457.963 | 6 |

长输入单并发测量本身波动较大：TTFT 为 6493.389–9823.809 ms，TPOT 为
164.719–386.838 ms。不能据单并发/双并发 p50 的交叉推断并发带来 prefill 加速，
也不能用三次重复声称稳定尾延迟。统计的是现有原生串行 slots serving 基线，
其中 FP4 GEMM 仍为 TileLang；本次另建的小 M CUDA 算子未接入这份测量。

退出状态的证据边界另记：本窗口 `squeue` 已不再列出该任务，报告完整且源码摘要
匹配；尝试 `sacct -j 26564,26568` 时，Slurm accounting 数据库连接被拒绝，未能
独立取回历史 ExitCode。随后在 **M:L9514（06:15:18.653 UTC）** 找到原窗口已收回的
对应 `write_stdin` 工具结果：列出六组上述实测值，`exit_code=0`。因此进程正常退出
已有直接会话证据；该结论不依赖空队列或 `all_passed` 字段，accounting 查询失败
也未触发重新启动已经完成的实验。

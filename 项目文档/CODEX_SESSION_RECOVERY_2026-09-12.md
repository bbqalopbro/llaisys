# Codex 会话恢复补记：2026-09-07 至 09-08

本记录在 2026-09-12 从本机其他 Codex 窗口的 JSONL 会话及当前磁盘产物恢复，补充
[B300 工程记录](B300_DEEPSEEK_V4_BRINGUP.md)、
[基础适配计划](DEEPSEEK_V4_BASE_ADAPTATION_PLAN.md)和
[算子后端说明](DEEPSEEK_V4_OPERATOR_BACKENDS.md)中未完整保留的调错顺序、失败状态
和证据边界。这里的历史结果不代替当前代码的新回归，也不把早期未完成状态覆盖到
后续已经实现的原生执行器上。

## 证据索引与读取规则

会话目录为本机 `$CODEX_HOME/sessions/2026/09/`，本次实际位于个人目录 `.codex` 下。
表中行号均为原始 JSONL 的 **1-based 物理行号**，时间为记录内 UTC 时间；北京时间
需加 8 小时，不能用会话文件名的日期推断每条事件发生日期。

| 简称 | 会话 ID / 文件 | 实际覆盖时间（UTC） | 内容 |
| --- | --- | --- | --- |
| S07 | `01a07c3e-3018-7ec3-ac70-a3641cae0c16`；`07/rollout-2026-09-07T14-20-45-01a07c3e-3018-7ec3-ac70-a3641cae0c16.jsonl` | 09-07 14:20:49 至 09-08 01:18:35 | 恢复 08-31 适配工作，Compressor/Indexer 实现与验证，启动 DeepGEMM 探针 |
| S08 | `01a08150-dd75-78f1-931d-2ab16cfe8395`；`08/rollout-2026-09-08T13-59-15-01a08150-dd75-78f1-931d-2ab16cfe8395.jsonl` | 09-08 14:00:00 至 14:29:35 | `active writer` 会话恢复冲突排查；没有新增推理算子实现 |

读取时将 `response_item` 中的用户请求、工具调用/返回和最终答复与当前文件对照。
没有将会话内再次打印的旧会话全文当成本窗口新执行的工作，也没有以助手进度说明
代替执行结果。下文只保存必要的工程事实，不复制整份会话、账户认证内容或个人信息。

## 已有结论的直接证据补全

Compressor 的归约修正、Indexer 算法和主要数值表已在 B300 工程记录中详细说明，
这里补上原始执行定位，便于区分“提议做”“正在做”和“已经结束”。

| 工作 | 原始工具证据 | 当时实际结论 |
| --- | --- | --- |
| Compressor 初版真实权重验收失败 | S07 L341，09-07 14:45:36；L365，14:47:30 | 六例中的 259-token 用例组件替换 cosine 为 0.9981496–0.9987136，低于 0.999；`all_compressor_comparisons_pass=false`，`E2E_EXIT=2`，即使所有 argmax 相同也未通过 |
| 归约顺序修正后的定向和整体验证 | S07 L411，14:50:01；L418，14:50:15；L435，14:51:23；L480，14:54:07；L491，14:54:55 | 62 次真实池化调用 FP32 最大差和 BF16 不同元素均为 0；8 项 Compressor 测试、45 项联合测试通过；六例 × 四步，共 24 次组件对照逐值一致，执行后校验了当时源码和动态库摘要 |
| Indexer 组件及真实模型对照 | S07 L592，15:10:46；L615，15:12:55；L985，15:35:25；L1318，15:56:54 | 先通过 7 项 Indexer 测试、52 项联合测试；七例、28 次 Indexer 组件对照逐值一致，最长输入 2105 token；最终核验 7 个模型用例、28 个对照和 5 个组件性能形状 |
| 长输入 full/replay 差异重复出现 | S07 L1256，15:53:33；L1275，15:54:14 | 2105-token logits cosine 再次为 0.974877119、最大绝对误差 8.113855362；同输入投影形状检查共 48 项，有些投影在多行/单行形状下不同，不能据此断言所有误差都来自投影 |
| 长 Compressor 状态回归及收尾 | S07 L1298，15:55:30；L1308，15:56:23；L1318，15:56:54 | 最终 53 项测试通过，`INDEXER_FINAL_TESTS_EXIT=0`；结束后该次 `squeue` 输出为空。此事实只说明当次收尾，不能用于判断当前作业状态 |

以上组件对照均属于当时显式 reference/cuBLAS 路径中的单组件替换验证。它们不等于
官方完整 TileLang golden 验收，也不等于生产 serving 性能。

## 环境调错中未完整记录的先后关系

### 缺失 Python 环境、uv 缓存权限和网络是三个独立问题

S07 L182/L189（09-07 14:35:18/29）确认旧共享 venv 的 Python 已不存在；L203
（14:35:54）确认系统 Python 可以导入 `torch 2.13.0+cu130`，但没有 `transformers`。
这不是 PyTorch/CUDA 算子导入失败。

第一次 `uv pip install` 在访问软件源前就失败：S07 L214/L216（14:36:12/13）中
默认缓存位于共享 `.uv-cache`，返回 `Could not acquire lock`，底层原因是创建临时
文件时 `Read-only file system (os error 30)`。后续显式设置
`UV_CACHE_DIR=$HOME/.cache/uv`，并将安装目标放在个人
`.local/deepseek-v4-tokenizer`，才越过本地缓存权限问题。

这次重试仍未成功：S07 L220/L232（14:36:25/55）显示 PyPI 请求在 3 次重试后以 DNS
解析失败退出。随后 S07 L235（14:37:46）明确请求沙箱外网络执行，L247/L254
（14:39:38/58）记录下载及安装完成。最终安装了 27 个包，关键版本为
`transformers 5.16.1`、`tokenizers 0.23.2`、`safetensors 0.8.0`。

初次失败命令曾写 `transformers==5.5.4`、`safetensors==0.7.0`，它没有完成安装；
成功命令使用 `transformers>=4.51,<6` 和 `safetensors>=0.6,<1`。因此复现依赖应以
成功安装记录及项目固定版本为准，不能把最早尝试的版本当作实际运行环境。

### `/tmp` 和 CUDA include 路径要在实际执行节点核实

S07 L99（09-07 14:29:12）在登录节点查询 MP1 转换目录返回不存在；L301
（14:42:21）的计算节点检查则列出了 tokenizer 文件和 `model0-mp1.safetensors`。
当时复用的是计算节点已有转换产物，没有因登录节点 `/tmp` 不可见而重新转换全部
权重。工程记录已说明节点区别，本补记保存了这条排查链。

S07 L535（15:03:40）按 `/usr/local/cuda/include/cub/device/...` 查找 CUB 分段排序
头失败；L542（15:03:48）找到真实位置：
`/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl/cub/device/device_segmented_radix_sort.cuh`。
后续 CUDA 13.0 自带 CCCL 构建已成功。这个查找失败不是“CUDA 没有 CUB”，也不是
Indexer 编译或数值错误。

## 长测试的取消、延迟返回和实际运行状态

### Top-512 测试曾主动替换输入

S07 L627（09-07 15:13:43）实际 tokenizer 输出最后一例为 1797 token、449 个
压缩候选，未触发 Top-512 截断。L633 将重复文本从原规模改为 300 次，L639/L642
（15:14:29/31）在 Slurm 22402 中对旧测试进程发送 SIGTERM，返回
`Stopped superseded benchmark`，随后启动修正用例。最终 JSON 确认输入为
2105 token，对应 526 个候选。

### 15:24 收到的退出码不能归到仍在运行的新测试

S07 L814/L816（15:24:40/45）轮询既有 PTY 后收到 `Terminated` 和
`INDEXER_E2E_EXIT=143`。这条迟到输出后，L820/L824 的 `ps -u` 辅助检查又返回
空进程表及该检查 step 的退出码 1；但两条输出均不足以证明修正后的计算已经结束。

于是 L828/L832（15:25:09）在同一 allocation 内重新查询，实际找到执行相同
`run_single_gpu.py` 命令的 PID 37025，同时 Slurm 22402 仍为 `R`。L865
（15:27:18）再次确认该进程处于 `Rl` 且累计 CPU 时间继续增加；L886
（15:28:45）返回 GPU 利用率 91%。L985 和 L1318 随后给出完整模型结果及最终核验。

证据支持的解释是：143 与前面主动终止旧用例相符，不能把它当成新用例失败，也
不能因辅助 `ps` 查询失败再次重启长测试。查验时应以 allocation 内实际命令、进程
存活和最终产物共同确定状态。这里没有从 143 推断 CUDA kernel 崩溃。

## DeepGEMM 首次接入前的环境和实际完成边界

### 开发头文件与源码准备

S07 L1413/L1422（09-07 16:17:53/16:18:04）尝试下载 Python 3.12 开发包和递归
克隆 DeepGEMM；初次沙箱下载受 DNS 限制。L1433（09-08 01:02:56）明确保留 Ubuntu
源解析失败，而不是编译失败。经单独批准网络后，L1447（01:15:02）确认开发包
下载成功，并打印 DeepGEMM 源码提交：
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`。

L1453（01:15:20）使用 `dpkg-deb -x` 将 `.deb` 解包到个人
`.cache/llaisys-python-dev/root`；L1461（01:15:38）设置 `CPLUS_INCLUDE_PATH`
指向该目录，使用 `DG_FORCE_BUILD=1 python3 setup.py build --build-base build-llaisys`。
这是个人目录内构建，没有执行系统级开发包安装。

L1495（01:16:36）记录扩展链接命令及生成的 `_C.cpython-312-x86_64-linux-gnu.so`；
L1511（01:17:14）在 B300 内导入后输出 `2.6.1 148 (10, 3)`，分别为 DeepGEMM
版本、查询到的 SM 数与 compute capability。构建期间出现 C++17 下 bit-field
默认初始化需要 C++20 的警告，但该轮扩展仍链接、导入成功；警告不能当作构建失败。

### 两类 Slurm 拒绝应分别记录

S07 L1350（09-07 16:14:49）的双卡申请返回 `QOSMaxGRESPerUser`，因此没有获得
多卡 EP 的执行证据。另一组错误与卡数无关：L1468/L1472（09-08 01:15:53/54）申请
4 小时、L1476/L1480（01:16:09）申请 2 小时，均返回 `Requested time limit is invalid`。
L1485 改为 1 小时后，L1495 确认分配到 Slurm 22520、B300、capability 10.3。
这两类错误分别限制卡数和作业时长，不能互相替代解释。

### 探针是启动证据，结果要从当前产物另行确认

S07 L1522/L1525（09-08 01:18:34/35）新增并启动
`tools/deepseek_v4_reference/probe_deepgemm.py`。本会话在命令发出后结束，没有收回
`DEEPGEMM_PROBE_EXIT` 数值。因此仅凭 S07 不能声称探针已完成。

2026-09-12 本次另行读取当前磁盘
`benchmark_results/deepseek_v4_b300_deepgemm_slices.json`，确认已有完整 10 行：

| 权重 | 格式 / N / K | M | 最大 relative L2 | 最大绝对误差 |
| --- | --- | --- | ---: | ---: |
| `layers.0.attn.wq_a` | FP8 / 1024 / 4096 | 1、5、32、129、259 | 0.000027974873 | 0.015625 |
| `layers.0.ffn.experts.0.w1` | FP4 / 2048 / 4096 | 1、5、32、129、259 | 0.000001212013 | 0.0009765625 |

10 行均满足当前探针代码的 `cosine >= 0.99999`、`relative_l2 <= 0.005` 门槛。
探针采用随机种子 908；A 按 128 元素一组量化为 FP8/UE8M0；FP8 B 的 recipe 为
`(128,128)`，FP4 B 为 `(1,32)`；参考值为同份量化输入解量化后的矩阵乘，最终转
BF16。它验证这两个权重切片与这些形状的格式和数值，并非整个模型、所有形状或吞吐
测量。JSON 没有完整 checkpoint 摘要、源码摘要、执行时间或 kernel 时延，故不扩展
其证明范围，也不从文件存在倒推它一定由 S07 的最后一次命令成功生成。

后续 DeepGEMM 整模型的失败已写在 B300 工程记录中：16 步 argmax 虽然相同，
relative L2 约为 2.72%–8.16%，`all_golden_comparisons_pass=false`。小切片通过
不能覆盖该失败；DeepGEMM 仍须保持显式实验后端，进一步 GEMM 工作需要独立数值与
性能验收。

## 09-08 的会话写入锁故障不属于模型适配结果

S08 L9（09-08 14:00:00）的用户错误指向 08-31 会话
`01a05815-0951-7743-a2a6-89877d637d7d`，消息是已有 active writer，错误码 `-32600`。
S08 L33（14:29:03）列出了对应 `thread-writer-locks/<会话ID>.lock`，L44
（14:29:16）再读该文件时已返回不存在。日志输出同时含有 app-server control socket
已被占用的消息，但输出被截断，未建立该消息与具体会话持有者之间的完整关联。

S08 L54（14:29:34）最终答复明确承认尚未确认具体占用进程，只建议重试恢复及正常
退出其他界面。本会话没有给出恢复成功的后续输出，也没有删除 JSONL、强行删锁或
修改推理仓库。因此这段历史应记为“恢复冲突被观察到，锁随后消失；原因和重试结果
未确认”，不能写成“清理锁后已修复”，更不能据此宣称会话数据损坏。

## 2026-09-12 的产物复核与历史版本边界

本次复核直接解析了失败/成功 JSON 的逐例对照、测试日志中的完成行、探针全部
10 行以及源码摘要映射，未重新运行 43 层 GPU 测试。Compressor 失败记录仍保留
`false`，修正后 24 次组件比较与 Indexer 28 次组件比较的最大/平均误差仍均为 0。
长输入的 logits 差异与上表一致，未把现存失败字段改写为成功。

以下摘要对应本次读取的磁盘文件，用于后续核对；完整路径均以仓库
`benchmark_results/` 为前缀。

| 文件 | SHA256 |
| --- | --- |
| `deepseek_v4_b300_native_compressor_before_reduction_fix.json` | `b64c9da82ecdb54559419f1a87b2a6d01ac7f8002174021dac61bfd760004395` |
| `deepseek_v4_b300_native_compressor_accuracy.json` | `cd854c6da152bd260610e55ff507213dba4b5e191dd021006859b4ee5edae64c` |
| `deepseek_v4_b300_native_indexer_accuracy.json` | `48e0c8f962cbc7cc3e701d71b2b27cddb03f65dbb956c823a0b10ff8b2b0b18f` |
| `deepseek_v4_b300_indexer_long_layer_diagnostics.json` | `818a30db37707d37767e4355bcb83aad3fcca92b4b2f5104518cd0f38e8ede75` |
| `deepseek_v4_b300_projection_shape_diagnostics.json` | `c115f96fc59190cb5efb4502603bea20b1c4f241b57a616518b98e1c66f8c7c6` |
| `deepseek_v4_b300_deepgemm_slices.json` | `58ee6103a2719ad6192b9bce3fbd5f8fce86ec1c258accd9a0d44db3232a1a8a` |
| `deepseek_v4_b300_indexer_regression_final.log` | `4203377e6269b1261ff5eae502b38add686dd8718c7823d3eda101ba0af815ec` |

这些历史结果里的 `provenance.file_sha256` 记录的是当时文件。当前
`run_single_gpu.py` 和 `libllaisys.so` 摘要已与 09-07 记录不同；原始官方 `model.py`
仍匹配，Indexer 轮次的 `deepseek_v4_native.py` 与投影诊断脚本也仍匹配。
相对路径的 `python/llaisys/libllaisys/libllaisys.so` 必须以仓库根目录解析，不能因为
从个人目录检查而误报文件缺失。S07 L491/L1318 的“摘要核验通过”是当时核验结果，
不得改写为今天的二进制已通过同一测试。

本次核对时，上述历史 JSON 被 `.gitignore` 的 `*.json` 规则忽略，日志被 `*.log`
忽略；文档写出文件名不代表它们已随 Git 发布。本文保留核心事实和摘要供追踪，原始
会话与完整日志继续留在本机。若需随提交提供完整验收数据，应明确选择所需产物并
检查内容，不能默认 `git add` 已覆盖这些文件。

## 交接时应保留的未完成与不确定事项

- 09-07 组件替换成功与 full/replay 长输入失败是两种独立比较；后续开发不能以
  `all_argmax_equal=true` 代替完整 logits 精度门槛。
- 原生 Indexer 的组件计时只在 4096 候选的单 token 形状下约快 1.26 倍；2105-token
  prefill 总时间约为 PyTorch 路径的 1.82 倍。168 个寄存器/336 字节栈的静态
  `reduce_heads` 记录是优化线索，尚不是 profiler 已确认的主要瓶颈。
- S07 末尾仅证明 DeepGEMM 扩展导入和探针启动；当前探针结果的完成性由本次磁盘
  复核单独支持。它没有给出新的 GEMM 性能结论。
- 该阶段未获得双卡分配，不能用单卡正确性测试代替 TP/EP 真实通信验证；当前配额
  需以新的 Slurm 查询为准。
- S08 active-writer 故障的具体持有进程与重试是否成功没有后续证据。不能从历史
  锁文件或最后一条会话消息推断今天仍有进程运行。

后续原生 Compressor、Indexer、Attention、Block、整模型与 serving 的进展，以
B300 工程记录的 09-08、09-09 及 09-12 复核章节为准；本文不把 09-07 的阶段性待办
误标成最新开发状态。

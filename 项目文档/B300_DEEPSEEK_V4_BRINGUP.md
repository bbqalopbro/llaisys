# B300 DeepSeek-V4-Flash 适配工程记录

日期：2026-09-01 UTC；最近续接：2026-09-08 UTC

## 工作范围与基线

- 仓库：`git@github.com:bbqalopbro/llaisys.git`
- 分支：`feat/vllm-paged-attention`
- 起始提交：`05784b4da6483f77821a061e521ea1b98fa271e1`
- GPU 通过 Slurm 的 `gpu` 分区分配；当前 QOS 限制每个用户最多使用一张 GPU。
- 根据项目所有者要求，本轮没有重复执行 Qwen 测试。

## 当前摘要（2026-09-08）

- 独立主模型 43 层已加载真实 MP1 权重；默认 TileLang 核心 + PyTorch 模型算子。
- 已接入实际 V4 对话编码与无 golden 依赖的 greedy CLI；8 例对话、352 步 logits
  对原始基线逐值一致，自由生成 token 也一致；严格功能检查 5/8，与官方相同。
- 新增 V4 paged latent 执行：C++ block 管理、物理槽写入、TileLang attention 直接读
  pool；2105-token 输入、16-token 输出与原始 full golden 逐值一致。
- 默认 257-token 分块与原始 full golden 的 logits 仍不通过精度门槛；paged 与同
  分块连续缓存逐值一致，不能把两种对照混为一谈。
- 新增独立上游固定数值策略参考；真实 43 层、2105-token/257-token 分块、16 步
  输出与该参考逐值一致。这不是原始上游数值策略验收，更不是统计任务精度通过。
- 19 类模型算子 + 4 类 cache 算子均提供显式后端接口；最新回归 193 方法中 192 通过，
  仍保留原始非对齐 chunk/full 失败项。中间 chunk 可只更新 cache，不执行输出 head。
- paged payload 已改由既有 C++ PagedCacheStorage 持有，通过 DLPack 共享给 TileLang；
  模型权重/计算仍未整体迁入 C++。新增显式 V4 Prefix Cache/partial-block COW，以及既有
  InferenceEngine 的 V4 参考后端接入；窗口回收、C++ model runtime、HTTP 接入、正式性能
  验收和 TP/EP 未完成。完整证据及限制见文末最新章节。
- 新增纯 C++ TileLang backend 及可替换 Kernel/Tensor 接口；23 个核心 kernel 用例、
  9 个实际权重的量化 Linear 用例通过。Slurm 23074 回归为 193 项字节对照、113 项
  异常检查及 9 项空输入检查通过，原生 TileLang 子进程无 Python/Torch。
- 新增 13 类 ATen C++ 结构后端、3 类组合工具和原生 Hadamard；联合回归 129 用例、
  374 项字节对照与 391 项异常检查通过。进一步完成原生 C++ MoE 组合：实际第 0/3
  层各加载 256 routed + 1 shared expert，1/33/137-token hidden 输入共 6 组，每组
  重复两次，路由/中间结果/最终输出对官方逐字节一致；170 项字节对照、24 项异常
  与 2 项空输入检查通过。此进程无 Python，但显式使用 ATen C++；不是完整 C++
  Transformer、文本端到端或 EP 验收。详见文末本次续接记录。
- 完成三类原生 Compressor：ratio-4/128 latent 和旋转 Indexer 压缩；96 step、
  705 次字节对照、132 项异常检查和 3 组部分写入恢复通过。
- 进一步完成原生 Indexer：真实第 2/42 层、76 step、614 次字节对照、102 项异常
  和 17 项元数据合同检查通过，包含 2105-token Top-512；两组失败恢复通过。
  原生回归最新为 Slurm 23074。Indexer 当前为连续缓存参考组合，不能
  宣称完整原生 paged Attention、43 层 C++ 执行或原始 full/chunk 精度已完成。
- 完成原生连续 Attention 的三种 ratio=0/4/128 组合，实际第 0/2/3 层共 132 step、
  1106 次字节对照、175 项异常、18 项元数据合同及 3 组部分写入恢复通过；复用
  TileLang、ATen、原生 Compressor/Indexer。原生 paged 接入和 43 层 C++ 模型仍未
  完成，不能把单组件通过表述为完整 runtime 验收。
- 完成原生 HC pre/post/head：9 组实际参数、108 次用例执行、1073 项 tensor 字节
  对照、174 项异常、17 项结构合同及 17 组失败恢复通过。
- 完成原生 Transformer Block：实际第 0/2/3 层全部参数，包含每层全部 256 routed
  和 1 shared expert；84 step、899 项 tensor 字节对照、126 项异常和 3 组整层失败
  恢复通过。Slurm 23074 的八组原生联合回归全部通过；43 层 C++ 模型装载/执行、
  原生 paged 与 pybind batch、正式性能及多卡仍待完成。

## 初始 B300 构建环境（后续变更见文末）

- GPU：一张 NVIDIA B300 SXM6 AC，显存 287,428,640,768 字节
- 计算能力：10.3（`sm_103`）
- 驱动：580.126.09
- CUDA Toolkit：13.0（`nvcc` 13.0.88）
- NCCL：2.29.3+cuda13.1
- Python：3.12.3，路径为 `/home/lcpu/62385178/venv`
- 共享虚拟环境默认 PyTorch：2.11.0+cu128；单卡正确性 runner 通过隔离依赖使用
  PyTorch 2.13.0+cu130。两者都能在 CC 10.3 上执行 CUDA，但不将参考 runner 的
  PyTorch 执行视为 llaisys 原生 B300 优化。
- xmake：3.1.1

项目现已在 CUDA 编译和设备链接阶段同时传入 `sm_103`。执行
`cuobjdump --list-elf` 后，`libllaisys.so` 中仅发现 `sm_103` cubin。

使用的构建命令：

```bash
xmake f -c --nv-gpu=y --cuda-arch=sm_103 --flashinfer=y \
  --python-bindings=n --dist-nccl=y -m release
xmake build llaisys
xmake install llaisys
```

初始构建时未启用 Python bindings，使用 ctypes 完成相互隔离的算子正确性测试。
该状态已更新：09-08 使用 PyTorch 随附 pybind11 headers 重建 `_C.so`，详见文末。

## 实际模型与权重结构

模型目录：`/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731`

- `model_type`：`deepseek_v4`；架构：`DeepseekV4ForCausalLM`
- 43 个主干层，hidden size 为 4096，64 个 attention head
- 每个 token 使用一个 512 维 latent KV，其中最后 64 维承载 RoPE
- 128 token 滑动窗口
- 每层学习式压缩比例为 0、4 或 128
- ratio-4 层额外包含 64 head、每 head 128 维、top-k 512 的学习式 Indexer
- 256 个 routed expert、top-6、一个 shared expert；前三层使用 token-ID hash routing
- Dense 权重为 FP8 E4M3，scale 为 UE8M0；Expert 权重为打包 FP4
- inference 配置及权重命名空间中包含三个 MTP/DSpark stage
- 48 个 safetensors 分片、72,317 个 tensor，声明大小 166,878,536,440 字节

该格式不是经典的 DeepSeek-V3 MLA。它的滑动窗口缓存、学习式压缩缓存和 ratio-4
索引缓存不能表示为 Qwen K/V，也不能表示为传统的 latent 与独立 RoPE cache。

## 已实现的正确性边界

当前适配包括：

1. 严格解析模型配置，并与官方 inference 配置交叉校验。
2. 对全部 72,317 个权重键进行 header 校验和所有权映射。
3. 支持 safetensors 切片加载；已使用真实 BF16、I32 和 FP8 数据验证，无需加载完整分片。
4. 新增模型专属 `DeepSeekV4CacheLayout`，分别描述 window latent、compressed latent
   和 ratio-4 index latent；通用 block manager 不感知 tensor shape。
5. 新增解量化 PyTorch 正确性参考实现，覆盖 RMSNorm、YaRN/RoPE、窗口与压缩索引、
   学习式压缩、稀疏 latent attention、routing、SwiGLU expert 和 routed/shared expert 合并。
6. 新增仅用于正确性验证的 CPU/CUDA 稀疏 latent attention 和非 hash
   sqrt-softplus top-k router。接口名称显式包含 `Reference`，未接入 serving 热路径。
7. 完成一个合成裁剪单层对照，覆盖 Hyper-Connection、低秩 Q/O 投影、latent attention、
   routed/shared MoE 和残差合成，并与官方模型源码对比。

RoPE、索引生成、ratio-4 压缩和 router 均已直接与官方参考源码进行一致性测试。

## 单卡真实权重端到端验证

已将 48 个原始分片转换为官方 inference 代码需要的 MP1 部署格式，转换后文件位于
`/tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors`，大小约 157 GiB。在一张
B300 上加载全部 43 层真实权重，完成了 full prefill、逐 token replay、KV/压缩
状态复用和 greedy decode。

执行后端显式标记为 `dequantized-pytorch-reference`：按 checkpoint scale 真实解码
FP8 E4M3 dense 权重和打包 FP4 expert 权重，模拟激活量化，再用 BF16 PyTorch
linear 执行；稀疏 attention 和 Hyper-Connection 使用项目的 PyTorch 正确性实现。
这不是原生 llaisys serving 热路径，也不是性能后端。

四个冒烟样例结果：

| 提示 | 生成摘要 | full/replay argmax | logits cosine |
| --- | --- | --- | --- |
| `The capital of France is` | ` Paris. The capital` | 一致 | 0.999679 |
| `1 + 1 =` | ` 2, ` | 一致 | 0.999759 |
| `The opposite of hot is` | ` cold. The opposite` | 一致 | 0.999773 |
| `中国的首都是` | `北京，而北京` | 一致 | 0.999838 |

另外使用 134 token 提示跨过 ratio-128 的首个完整压缩组：full prefill 与
token replay 的 argmax 一致，logits cosine 为 0.997872，最大绝对差为 1.93437。
该结果证明 ratio-128 状态能跨 prefill/decode 路径运行，但更大的数值差异也表明
量化顺序或压缩更新顺序需要继续和官方 kernel 对齐。详细记录位于
`benchmark_results/deepseek_v4_b300_long_context_accuracy.json`。

逐层诊断显示，5-token 用例在第 0 层已有 BF16 量级的微小差异（max abs
0.00390625，cosine 0.9999987），之后随 43 层累积；这不是某个 ratio-4/ratio-128
层突然产生的首次偏差。第 42 层 hidden max abs 为 9.0，cosine 仍为 0.999127，
最终 logits argmax 一致。详细数据位于
`benchmark_results/deepseek_v4_b300_layer_diagnostics.json`。因为 full prefill 与单 token replay
的 GEMM 形状、归约顺序和激活量化边界不同，必须基于官方 kernel golden 设定合理
容差，不应直接要求逐 bit 相等。

- 权重加载：28.05 s
- 峰值显存：179,363,362,816 字节，约 167.04 GiB
- 3/5 token 短 prompt 首次 prefill：719.53–1034.05 ms
- 单 token decode：约 359–384 ms/token

上述耗时只表示解量化 PyTorch 正确性后端的运行成本，不能作为生产 TTFT/TPOT
或 B300 性能指标。完整 JSON 记录位于
`benchmark_results/deepseek_v4_b300_accuracy.json`。full prefill 与 token replay 的最终
logits 余弦相似度高且 argmax 全部一致，但最大绝对差为 0.49–0.67；这一差异仍需
用可运行的官方量化 kernel 或外部 golden logits 进一步定位，不将其表述为完整数值验收。

## 原生 BF16 稀疏 MLA 和 Hyper-Connection 接入结果

已将原生 CUDA 稀疏 latent attention 扩展到 BF16 输入/输出，并实现了原生
FP32 Hyper-Connection split/Sinkhorn。两者通过显式参数
`--native-structural-library` 接入完整模型 runner。如果动态库或 typed symbol
不存在会立即报错，不会静默回退到 PyTorch。

单卡构建命令：

```bash
xmake f --nv-gpu=y --cuda-arch=sm_103 --flashinfer=y \
  --python-bindings=n --dist-nccl=n -m release
xmake build llaisys
xmake install llaisys
```

B300 上已通过 8 项原生 CPU/CUDA 对照，包括 BF16 路径、Hyper-Connection
Sinkhorn 和正式
`H=64, D=512, top-k=640` 形状。随后使用同一份 43 层真实权重重跑四个
端到端样例，执行路径为
`dequantized-pytorch-linear+native-bf16-sparse-attention+native-hc`：

- 四个样例的 full prefill/token replay argmax 全部一致。
- `Paris`、`2`、`cold`、`北京` 四项预期语义全部命中。
- logits cosine 范围为 0.999223–0.999940。
- 峰值显存仍为约 167.04 GiB。
- 解码步耗时约 300–324 ms/token，仍由整个参考后端主导，不作为 TPOT。

完整记录位于
`benchmark_results/deepseek_v4_b300_native_structural_accuracy.json`。相比纯 PyTorch 稀疏
attention，第三个解码 token 后的个别生成 token 可因 BF16 归约顺序不同而变化，
但本轮验收的首 token、full/replay argmax 和预期语义保持一致。后续仍需用官方
kernel logits 设定精度阈值。

## 原生 FP8/FP4 权重计算接入

后续审查发现，早期解量化 PyTorch linear 虽然正确解码了权重，但漏掉了官方
`linear()` 在量化 GEMM 之前对激活执行的 FP8 QAT 边界。该问题已修正：解量化
参考路径和原生路径现在都会先按 128 元素 block 量化/反量化激活。因此早期
JSON 只保留为 bring-up 过程记录，最新精度口径以本节为准。

已实现 correctness-only fused CUDA GEMM：

- BF16 activation × FP8 E4M3 weight，权重 scale 为 E8M0。
- BF16 activation × packed FP4 E2M1 weight，每 32 个权重一个 E8M0 scale。
- 不在显存中长期保留整份 BF16 解量化权重。
- FP4 激活临界值按官方有序查表 `argmin` 规则处理，不使用硬件
  ties-to-even 来隐藏舍入差异。

B300 原生算子测试现为 12 项全部通过，新增覆盖 FP8/FP4 激活量化、非连续
view 和 FP8/FP4 权重 GEMM。使用完整 43 层真实权重运行四个样例，实际后端为
`native-fp8-fp4-linear+native-bf16-attention-hc-router-actquant`：

- 四个样例 full prefill/token replay argmax 全部一致。
- `Paris`、`2`、`cold`、`北京` 预期语义全部命中。
- logits cosine 为 0.999343–0.999676。
- 3/5 token prefill 为 347.80–522.35 ms。
- 单 token decode 为 172.20–238.46 ms/token。
- 峰值显存 170,254,715,392 字节，约 158.56 GiB，比长期缓存解量化权重的
  参考路径低约 8.48 GiB。

为防止只根据 backend 名称误判实际路径，runner 现在会写入原生算子调用计数。
本轮四个样例实际调用为：

- sparse attention：1,634 次
- Hyper-Connection：3,268 次
- sqrt-softplus router：1,520 次（前三个 hash layer 保留官方查表路径）
- activation quant：59,428 次
- FP8 linear：12,236 次
- FP4 linear：44,214 次

完整结果位于 `benchmark_results/deepseek_v4_b300_native_linear_accuracy.json`。
这仍是直接解码权重并用 FP32 归约的 correctness kernel，不是 Tensor Core 优化的生产
GEMM；上述耗时只用于当前后端内部回归，不宣称为正式 TTFT/TPOT。

## 算子库 GEMM 接入

根据后续约束，GEMM 不再依赖手写点积作为主验收路径。已新增显式
`--operator-library-linear` 后端：CUDA kernel 只负责将 checkpoint 的 FP8/FP4
权重按 E8M0 scale 解码为 FP32，矩阵乘法交给 cuBLAS SGEMM，最后转换为 BF16
输出。cuBLAS 使用 `CUBLAS_PEDANTIC_MATH`，本轮精度测试没有启用 TF32。

不能直接复用仓库内 FlashInfer 的通用 FP4 GEMM：该接口是 W4A4/NVFP4，实际
模型是 FP8 activation × FP4 weight，激活按 K=128 分组、FP4 权重按 K=32
分组。两者的数据类型和 scale layout 不相同。强行套用会改变模型量化语义。

B300 上 FP8、FP4 小矩阵单元测试均与 FP32 解码后的 PyTorch oracle 在 BF16
输出上逐值一致。完整 43 层、单卡、真实权重四用例结果为：

- 后端：`cublas-fp32-gemm+native-quant-decode-attention-hc-router-actquant`
- 四个用例 full prefill/token replay 首 token argmax 全部一致。
- `Paris`、`2`、`cold`、`北京` 四项预期语义全部命中。
- logits cosine 为 0.998677–0.999578。
- 3/5 token 短 prompt prefill 为 422.77–603.98 ms，平均 534.63 ms。
- 单 token decode 为 211.25–276.53 ms，平均 222.46 ms，中位数 215.10 ms。
- 峰值显存 170,254,715,392 字节，约 158.56 GiB。
- 实际调用：cuBLAS FP8 linear 12,236 次、cuBLAS FP4 linear 44,310 次；
  手写 FP8/FP4 linear 调用均为 0。

完整结果位于 `benchmark_results/deepseek_v4_b300_cublas_accuracy.json`。这些输入
只有 3/5 个 prompt token，且当前仍是逐层 Python reference runner、每次 GEMM 前
临时解码权重，因此这里只是正确性后端耗时，不是 serving TTFT/TPOT 或生产吞吐指标。
生产路径仍需实现匹配实际 W4A8 scale layout 的 B300 Tensor Core kernel，或接入明确
支持该格式的算子库。

## 其他验证结果

- Python 配置、权重清单、reference 和 scheduler 测试：通过
- C++ cache-core 测试：通过
- 真实 checkpoint header 和小规模权重切片测试：通过
- B300 上 PyTorch CPU/GPU reference 对比：通过
- B300 上原生 CPU/CUDA 稀疏 attention 与 PyTorch 对比：通过
- 正式 attention shape（`H=64`、`D=512`、`topk=640`）测试：通过
- B300 上原生 CPU/CUDA router（`E=256`、`topk=6`）对比并接入完整模型：通过
- B300 上原生 FP8/FP4 激活量化和权重 GEMM 对比：通过
- B300 上 cuBLAS FP32 GEMM 后端与 FP8/FP4 解码 oracle 对比：通过
- 完整合成裁剪单层与官方源码对比：通过
- CUDA 二进制检查：仅包含 `sm_103` cubin

目前没有生产级 TTFT、TPOT 和吞吐量数据，因为尚未形成原生 llaisys
DeepSeek runner 和优化量化 kernel。当前的四个语义冒烟样例也不等价于 MMLU/GSM8K
等统计准确率评测。

## 已知限制和下一阶段闭环（截至本节；09-08 更新见文末）

- TileLang 0.1.8 的简单量化算子可在 B300 上运行，但官方复杂 `sparse_attn`
  和 `fp4_gemm` 在 TVM `DecoupleTypeCast` 阶段触发内部错误，所以暂时无法生成
  官方 golden logits。
- `fast-hadamard-transform` 源码构建因环境缺少 Python 开发头文件失败；当前使用
  明确标记的 PyTorch FWHT 正确性实现。
- 当前 CUDA attention 和 router 是正确性参考算子，不是生产级 kernel，不能作为未报告的
  静默 fallback 使用。
- 真实 FP8/FP4 权重、43 层模型、ratio-4 Indexer 和流式 compressor state 已在参考
  后端闭环；原生 BF16 稀疏 attention 和 Hyper-Connection 已接入受控端到端
  runner；原生 correctness 量化 GEMM 和 router 也已接入。尚未完成的是生产级
  Tensor Core GEMM、原生 DeepSeek runner 和 serving 调度。2026-09-07 已继续实现
  compressor 的原生流式池化及状态更新，以及 Indexer 的 cuBLAS 打分和 CUB Top-K，
  组件替换均通过真实权重对照，具体边界和长输入残余误差见后文。
- 完整 checkpoint 可在单张 B300 上容纳，参考执行峰值约 167.04 GiB；该结论不代表
  生产后端的显存开销。
- 当前单 GPU QOS 无法测试 EP；至少需要两张 GPU 的 Slurm 配额后才能验证 EP。

下一阶段最小闭环为：固化长序列与 ratio-128 回归 → 定位 full/replay logits 数值差异
→ 用官方 golden logits 验证原生 FP8/FP4 GEMM 容差 → 优化原生 Indexer
→ 接入 llaisys 单卡 serving runner
与 Python scheduler → 官方 golden logits 对比 → 获得多卡配额后进行双卡 EP。

## 2026-09-07：恢复会话并接入原生流式 compressor

恢复依据为 `.codex/sessions/2026/08/31/rollout-2026-08-31T13-49-28-01a05815-0951-7743-a2a6-89877d637d7d.jsonl`。
仓库仍在 `feat/vllm-paged-attention`，HEAD 为 `05784b4`；此前 DeepSeek 适配改动尚未
提交。本轮在这些改动之上继续实现，没有覆盖已有工作或改写提交历史。

本轮增加 `llaisysDeepSeekV4CompressProjectedReference` CUDA 算子，输入是 FP32 的
`wkv/wgate` 投影和 APE，输出是归一化之前的 FP32 加权池化结果。算子同时维护请求
独占的 `kv_state/score_state`，覆盖：

- ratio-4：上一组前半特征与当前组后半特征的重叠压缩；首组的上一窗口使用负无穷分数屏蔽。
- ratio-128：每满 128 token 输出一个压缩向量。
- 未满一组时只写入状态；支持跨组、非对齐分块以及逐 token 追加。
- `start_pos=0` 明确重置状态；Python 适配器拒绝跳跃或重复的后续位置。
- CUDA stream 由调用方传入，支持非默认流；绑定检查 dtype、shape、连续性和状态内存别名。

通过 `--native-compressor` 显式替换官方模型中 Attention 和 Indexer 使用的 Compressor。
投影继续使用已有矩阵乘法路径；池化后的 BF16 舍入、官方 RMSNorm、组起始位置 RoPE、
Attention 的 FP8 QAT 和 Indexer 的 Hadamard/FP4 QAT 顺序保持与原实现一致。新增调用
计数 `compress_projected`，缺少算子或参数不合法时直接报错。

这里完成的是 compressor 的池化和流式状态更新。它不代表整个 compressor 全部原生化，
也不代表 Indexer 打分/Top-K、整个模型的多 token chunked prefill、paged cache 或
serving 已完成。现有模型 `Attention.forward` 的增量路径仍仅支持单 token。

### 验证口径

新增测试文件为 `test/test_deepseek_v4_compressor.py`。B300 上 8 项新增测试全部通过，
合并原有配置、真实权重切片、reference、原生 attention/router/GEMM 后共 45 项通过，
没有跳过项。记录位于 `benchmark_results/deepseek_v4_b300_compressor_regression.log`。

- 独立 PyTorch softmax oracle：覆盖 D=128/512、ratio=4/128，FP32 池化使用
  `atol=rtol=3e-6` 验收。
- 同一份投影输入下，完整序列、逐 token 和非对齐分块得到逐值一致的池化结果及最终状态。
- 官方 Compressor 源码对照：覆盖未满组的 prefill、跨边界 decode、完整 prefill 重置，
  包括真实形状下的 RoPE、FP8/FP4 QAT 和 Indexer 旋转路径。量化后 BF16 输出容差为
  `atol=0.032, rtol=0.008`；逐 token 路径状态逐值一致。
- 极端分数稳定性、空输出、状态重置、内存别名拒绝和非默认 CUDA stream 均通过。
- 新增 ratio-128 BF16 舍入回归：10 个随机种子、259 token、D=512，舍入后的池化值
  必须与 PyTorch 逐值一致，不能只用 FP32 `allclose` 掩盖舍入边界差异。

新增 `--compare-compressor` 在同一模型实例、同一权重和相同 GEMM 后端下切换官方
Compressor 与新增 CUDA Compressor，比较完整模型的 prefill 和固定 token 的 decode
logits。此对照只评估 compressor 替换带来的误差；其余 reference/量化 GEMM 与之前
一致，不将其称为官方完整量化 kernel golden logits。验收要求各步 argmax 一致且
cosine >= 0.999；JSON 同时保留最大/平均绝对误差。

### 长序列精度失败与修正

首次 6 用例测试中，前 5 个用例（3/5/134 token）替换前后的完整模型 logits 逐值一致，
但 259 token 用例仅 argmax 一致，四个步骤的 cosine 为 0.99815–0.99871，未通过
0.999 验收线。保留失败记录为
`benchmark_results/deepseek_v4_b300_native_compressor_before_reduction_fix.json`，没有
降低验收阈值。

问题在于原生池化初版使用串行 FP32 求和；PyTorch 的维度归约使用四个独立累加器，
ratio-128 的真实形状还包含四路分组归约。两者微小的 FP32 差异会跨过 BF16 舍入
中点，在后续量化和多层 MoE 中放大。实现改为与当前参考环境相同的累加/归并顺序，
保留乘法和加法各自的舍入。

归约结构核对依据为当前安装的
`torch/include/ATen/native/cuda/Reduce.cuh`，并参考
[PyTorch Reduce.cuh](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Reduce.cuh)
和 [SoftMax.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu)。
这项逐值一致性针对本轮 PyTorch 2.13.0+cu130 与模型形状，不保证跨版本归约实现逐 bit 不变。

`tools/deepseek_v4_reference/diagnose_compressor_pool.py` 可逐次比较真实投影输入的
原生池化和独立 PyTorch 计算，在首次 BF16 不一致时保存小型诊断 tensor。修正后的
259 token prefill 中，62 次 compressor 调用（Attention 与 Indexer）全部得到
FP32 最大误差 0、BF16 不一致数量 0，记录位于
`benchmark_results/deepseek_v4_b300_compressor_diagnostics.log`。

### 修正后真实 43 层结果

最终结果为 `benchmark_results/deepseek_v4_b300_native_compressor_accuracy.json`，
进程退出码 0，三个总验收字段全部为 true。6 个用例的 full/replay 首 token argmax
全部一致，四个语义冒烟样例全部命中；更关键的是，同输入下替换 compressor 前后
的 24 个 prefill/decode 步骤，logits 最大和平均绝对误差均为 0。

| 用例 | 输入 token | 输出 token | prefill 单步 ms | decode 平均 ms/token | compressor 替换前后最大 logits 误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| France → Paris | 5 | 4 | 424.48 | 194.33 | 0 |
| 1 + 1 → 2 | 5 | 4 | 414.06 | 174.15 | 0 |
| hot → cold | 5 | 4 | 453.90 | 174.50 | 0 |
| 中国首都 → 北京 | 3 | 4 | 321.55 | 173.76 | 0 |
| 重复 hello，跨首个 ratio-128 组 | 134 | 4 | 1212.29 | 250.18 | 0 |
| 重复 hello，跨两个 ratio-128 组 | 259 | 4 | 1813.64 | 249.28 | 0 |

上述每行只有一次计时生成、3 个 decode 样本，属于正确性 runner 单步成本，不能
视为 serving TTFT/TPOT 或吞吐基准。18 个 decode 步骤平均 202.70 ms，范围
170.18–257.52 ms。最终一次权重加载 31.93 s；首次冷文件缓存加载为 135.52 s，
记录在失败轮 JSON，二者不能混入推理时延比较。

- 峰值显存 170,378,900,480 字节（158.68 GiB）。
- `compress_projected` 实际调用 28,830 次。
- cuBLAS FP8/FP4 linear 分别调用 157,458/453,858 次；手写 FP8/FP4 GEMM 调用均为 0。
- 计数包含 full/replay、计时生成以及 compressor 前后对照；不是单条请求的调用量。
- 完整模型仍使用显式 reference runner 和 cuBLAS FP32 GEMM；没有静默 fallback。

必须区分两种对照：**替换 compressor 前后**的 logits 已逐值一致；**完整 prefill 与
逐 token replay**仍有差异，134/259 token 的 cosine 分别为 0.997786/0.997133，
最大绝对误差分别为 1.67695/2.78001。后者的执行形状与量化顺序仍需继续诊断，
不能因此宣称完成官方整模型数值验收。下一步继续推进原生 Indexer 与独立
DeepSeek runner，并对其余 full/replay 差异做逐层定位。

### 环境恢复与复现

原共享 `/home/lcpu/62385178/venv` 已不存在。改用系统 `/usr/bin/python3`（3.12.3），
保留个人目录下已有的 PyTorch 2.13.0+cu130，并将缺失的 tokenizer 依赖安装到独立个人
目录。本轮实际版本为 transformers 5.16.1、tokenizers 0.23.2、safetensors 0.8.0。
计算节点上的 `/tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors` 仍存在，可以
复用；登录节点的 `/tmp` 中不可见该文件。

```bash
cd /home/lcpu/57682329/workspace/llaisys
export PYTHONPATH="$PWD/python:$PWD/tools/deepseek_v4_reference:/home/lcpu/57682329/.local/deepseek-v4-reference:/home/lcpu/57682329/.local/deepseek-v4-tokenizer"
/home/lcpu/62385178/.local/bin/xmake build llaisys
/home/lcpu/62385178/.local/bin/xmake install llaisys
srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=256G --time=00:30:00 \
  python3 tools/deepseek_v4_reference/run_single_gpu.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --converted-model /tmp/llaisys-deepseek-v4-flash-mp1 \
  --cases-file test/data/deepseek_v4_compressor_cases.json \
  --max-new-tokens 4 --max-seq-len 384 \
  --torch-reference-structural-ops --dequantized-pytorch-linear \
  --native-structural-library python/llaisys/libllaisys/libllaisys.so \
  --operator-library-linear --native-compressor --compare-compressor \
  --output benchmark_results/deepseek_v4_b300_native_compressor_accuracy.json
```

本轮结果额外记录分支、提交、工作区是否修改、动态库/绑定/runner/官方源码 SHA256 和
实际命令。时延在 GPU 同步后开始计时，明确标记为正确性 runner 的单步耗时；每个用例
先执行一次 full prefill 和一次 token replay，再测一次生成序列，batch/concurrency=1，
CUDA Graph 和 Prefix Cache 均未开启。该口径仍不是 serving TTFT/TPOT。

## 2026-09-07：继续接入原生 Indexer 打分与 Top-K

新增 `src/ops/deepseek_v4/nvidia/indexer.cu`，并通过 `--native-indexer` 显式接入
完整模型；后端名称包含 `cublas-bf16-indexer+cub-stable-topk`。固定支持本 checkpoint
的 H=64、D=128、压缩比例 4，当前适配器仅支持单 rank，多 rank 会明确报错。

### 算子与数值边界

1. cuBLAS `cublasGemmStridedBatchedEx` 计算 BF16 Q 与 BF16 压缩 K 的点积，FP32
   累加后输出 BF16。没有增加手写 GEMM。
2. CUDA 执行 ReLU、逐 head 权重乘法和 head 维度归约。乘法结果先舍入到 BF16，
   归约按当前参考环境的累加顺序执行，再舍入到 BF16。对外用 FP32 保存这些已舍入
   分数，供排序使用；不能将整个过程融合成一次无中间舍入的 FP32 计算。
3. CUB `DeviceSegmentedRadixSort::SortPairsDescending` 在每个 query 内排序，取前
   `min(512, 候选数)` 项。排序前应用因果屏蔽，尚未产生的压缩位置输出 `-1`。
4. 并列分数采用稳定规则：候选 index 小的优先。原始 `torch.topk` 不保证并列
   index 的稳定性，因此独立测试同时检查分数、选中分数和显式并列规则，不把未知的
   并列顺序当成数学错误。真实模型仍与未经修改的官方 Indexer 做单独对照。

C ABI 新增打分、排序 workspace 大小查询和 Top-K 接口。绑定检查 dtype、shape、
连续性、位置和 int32 容量；空候选直接返回空结果。打分的 cuBLAS handle 按主机线程
和设备管理，算子使用调用方 CUDA stream，临时 tensor 由调用方持有，排序缓存仅保存
workspace 大小。调用计数新增 `indexer_scores_cublas` 和 `indexer_topk_cub`。

接口语义参考 [NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex)
与 [CUB 分段排序源码](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_segmented_radix_sort.cuh)，
实际构建使用本机 CUDA 13.0 附带版本。并列行为核对了当前安装的 PyTorch 2.13
`torch.topk.__doc__`；[PyTorch 在线说明](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)
也说明并列项的 index 不保证稳定。在线文档版本与本机版本分开记录。

### 结构和数值回归

新增 `test/test_deepseek_v4_indexer.py`，7 项测试在 B300 全部通过。合并之前的
compressor、attention/router/GEMM、配置和权重测试后共 52 项通过，没有跳过项，
日志为 `benchmark_results/deepseek_v4_b300_indexer_regression.log`。随后针对长输入
数值差异新增 compressor 状态回归，最终 53 项测试全部通过，耗时 3.321 s，无跳过项；
最终日志为 `benchmark_results/deepseek_v4_b300_indexer_regression_final.log`。

- 候选数 1/2/3/4/32/64/65/512/513/1024 下，打分与 PyTorch BF16 路径逐值一致。
- Top-512 从 513/1024/4096 个候选中实际筛选，与稳定排序 oracle 逐值一致。
- 正负分数、完全并列、前 3 个 token 没有完整压缩组、未来高分候选屏蔽均通过。
- 非默认 CUDA stream、重复 workspace 尺寸、空候选和非法参数检查均通过。

`--compare-indexer` 保持同一模型权重、原生 compressor 和其余算子不变，仅切换
官方与原生 Indexer，对比 prefill 和固定 token 的 decode logits。验收线保持
argmax 一致、cosine >= 0.999，记录最大/平均绝对误差。这仍是组件替换对照，
不是官方完整量化 kernel golden logits。

长用例经过实际 tokenizer 校验为 2105 token、526 个压缩候选，确保超过 Top-512
截断边界。用例位于 `test/data/deepseek_v4_indexer_cases.json`。不能根据文本重复
次数估计 token 数：初版重复 256 次的文本只有 1797 token，已停止该测试并更正输入。

### 当前实现边界

原生化范围是 Indexer 的打分、因果屏蔽和 Top-K。Q/权重投影、RoPE、Hadamard、
QAT 和模型层调用仍在显式 reference runner 内；压缩由上一轮原生 compressor
承担。CUB 路径会排序全部候选，并使用临时点积分数/workspace，尚不是为长上下文
优化的流式、分页或 fused Indexer。整个模型的多 token chunked prefill、原生
DeepSeek runner、serving 调度和多卡 EP 仍需继续实现。

独立组件性能脚本为 `tools/deepseek_v4_reference/bench_indexer.py`，默认 warmup=5、
repeat=30，比较官方 PyTorch 路径与原生路径，分别记录 CUDA event 和 wall-clock
的平均/P50/P95。它只覆盖 Indexer，不作为 serving TTFT/TPOT。

### 43 层真实权重结果

结果文件 `benchmark_results/deepseek_v4_b300_native_indexer_accuracy.json` 的退出码为 0。
7 个用例的 full/replay 首 token argmax 全部一致，5 个设置了语义预期的用例全部命中。
与官方 Indexer 的 28 个同输入 prefill/decode 对照步骤，**logits 最大和平均绝对误差
全部为 0**，组件替换验收通过。这包括 2105 token、526 个候选截取 Top-512 的用例。

| 用例 | 输入 token | 输出 token | prefill 单步 ms | decode 平均 ms/token | full/replay logits cosine | Indexer 替换最大误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| France → Paris | 5 | 4 | 558.02 | 222.00 | 0.998677 | 0 |
| 1 + 1 → 2 | 5 | 4 | 544.25 | 219.53 | 0.999578 | 0 |
| hot → cold | 5 | 4 | 598.58 | 219.07 | 0.999204 | 0 |
| 中国首都 → 北京 | 3 | 4 | 422.21 | 216.82 | 0.999265 | 0 |
| 重复 hello，跨 ratio-128 | 134 | 4 | 1813.12 | 325.45 | 0.997612 | 0 |
| 重复 hello，跨两个 ratio-128 组 | 259 | 4 | 2910.22 | 336.17 | 0.995825 | 0 |
| 重复 France/Paris，超过 Top-512 | 2105 | 4 | 33040.30 | 518.59 | 0.974877 | 0 |

每个用例仅一次计时生成、3 个 decode 样本；上述数值是当前正确性 runner 的单步
成本。长输入 prefill 约 33.04 s，不能标为生产 serving TTFT。当前构建、缓存长度与
之前轮次不同，跨轮耗时差异也不能直接归因于本次 Indexer 替换。独立同进程组件对照
的耗时见下一节。

- 权重加载 32.21 s；峰值显存 171,293,393,920 字节（159.53 GiB）。
- Indexer 打分和 CUB Top-K 各实际调用 53,655 次。
- 原生 compressor 调用 161,634 次；cuBLAS FP8/FP4 linear 分别调用
  839,454/2,160,030 次，手写量化 GEMM 调用均为 0。
- 调用数包含完整 prefill、逐 token 重放、计时生成和替换前后对照，不是每条请求调用数。
- 结果中的 runner、官方源码、原生绑定与动态库 SHA256 均已和执行后文件核对一致。

**整模型数值验收仍未完成。** 长输入 full/replay 的最大绝对误差为 8.11386，cosine
为 0.974877，明显大于短输入误差。总字段 `all_argmax_equal=true` 只说明首 token
选择一致；`all_indexer_comparisons_pass=true` 只说明 Indexer 替换符合组件验收。
两个字段都不代表 full/replay logits 通过 0.999 余弦阈值，也不代表官方完整量化
kernel golden logits 已通过。为此增加该长用例的逐层诊断，见后文。

### Indexer 组件性能

结果文件为 `benchmark_results/deepseek_v4_b300_indexer_benchmark.json`，退出码 0。
所有形状的打分值和所选分数值均与 PyTorch 逐值一致。batch=1、H=64、D=128，
FP4 QAT 后的 BF16 输入，每种路径预热 5 次、同步测量 30 次，没有 CUDA Graph。

| query token | 压缩候选 | PyTorch 总耗时平均 ms | 原生总耗时平均 ms | PyTorch/原生 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.12390 | 0.14145 | 0.88× |
| 1 | 513 | 0.14930 | 0.14897 | 1.00× |
| 1 | 4096 | 0.18847 | 0.14917 | 1.26× |
| 259 | 64 | 0.13772 | 0.19017 | 0.72× |
| 2105 | 526 | 0.99662 | 1.81494 | 0.55× |

本轮原生 Indexer 仅在 4096 候选的单 token 测试中更快，小候选和 prefill 尚无普遍
性能收益。原生路径包含完整候选排序、临时点积/workspace 和 ctypes 调用；需要通过
profiler 继续区分各项开销，再推进分段选择、归约和 workspace 优化。上述是组件总
耗时，不是完整模型 TPOT；JSON 另存 CUDA event 时间及 P50/P95。

静态二进制检查还发现 `reduce_heads` 使用 168 个寄存器、336 字节栈，而准备排序和
收集 Top-K 的 kernel 分别使用 28/27 个寄存器且没有栈。该信息来自当前动态库的
`cuobjdump --dump-resource-usage`，提示 head 归约的临时数组值得优化；尚未通过
kernel 时间分解证明它是主要瓶颈。

### 长输入误差的进一步定位

`benchmark_results/deepseek_v4_b300_indexer_long_layer_diagnostics.json` 在同一动态库下
重新运行 2105 token 用例，并记录各层最后一个 token 的四路 HC hidden。最终 logits
的最大/平均绝对误差和 cosine 均与前一轮完全复现，排除了本次结果只是偶发变化。

| 层编号（从 0 开始） | hidden 最大绝对误差 | hidden cosine |
| ---: | ---: | ---: |
| 0 | 0.000488281 | 1.000000 |
| 1 | 0.001953125 | 0.999992 |
| 2 | 0.003906250 | 0.999967 |
| 3 | 0.012496948 | 0.998994 |
| 7 | 0.136718750 | 0.982909 |
| 12 | 0.583984375 | 0.749308 |
| 16 | 2.500000000 | 0.472085 |
| 42 | 33.000000000 | 0.864562 |

第 0 层已经出现微小偏差，其时尚未进入 ratio-4/128 compressor；之后出现明显误差
放大。表中第 0 层 cosine 四舍五入显示为 1，但输出并非逐值一致。最终 logits 的
cosine 0.974877 和首 token 一致不能掩盖中间 hidden 的较大差异，也不能作为长上下文
完整数值验收通过的依据。

新增 `tools/deepseek_v4_reference/diagnose_projection_shapes.py`，仅执行前 4 层
prefill。每次以**同一行实际输入**分别计算多 token 投影和单 token 投影，保持原路径
输出供后续模型计算，额外记录影子调用结果。检查包含 attention、compressor、Indexer、
shared expert 投影以及 HC preprocessing；不包含 routed expert 的逐项对照。

`benchmark_results/deepseek_v4_b300_projection_shape_diagnostics.json` 共记录 48 项
对照（40 项投影、8 项 HC），退出码为 0。环境为 float32 matmul precision=`highest`、
TF32=false。观察到：

- 40 项投影中 12 项存在逐值差异，包含 6 项 FP8/cuBLAS 投影与 6 项 FP32 compressor
  投影。第 0 层 `wq_a/wq_b` 各有 1 个元素不同，最大差均为 0.0000610352。
- 第 3 层 `wq_b` 有 3 个元素不同，最大差为 0.000244141；该层 compressor
  `wgate` 的 FP32 最大差为 0.00000429153。
- 4 项 BF16 分组 `wo_a` 投影在该检查中逐值一致。
- HC 的 FP32 post/combination 已有微小差异；第 1/3 层 attention 前的 HC BF16
  输出各有 1 个元素不同，最大差分别为 0.000000953674 和 0.0000152588。

这些数据证明在输入相同、没有缓存读写参与的投影/HC 子步骤中，执行形状本身就能
产生误差；**尚不能证明它们解释了全部整模型误差**。后续应固定上游输入，逐项对照
稀疏 attention、量化边界、压缩输出和 expert 选择，并继续寻求官方量化 kernel golden。

另外在 `test/test_deepseek_v4_compressor.py` 补入 2105 token 的定向回归，覆盖
ratio-4/D=128、ratio-4/D=512、ratio-128/D=512。相同投影输入下，完整计算、2105 次
单 token 追加及跨 127/128/129 等边界的非对齐分块，FP32 池化输出和最终状态均逐值
一致，且满足独立 softmax oracle 的 3e-6 容差。这降低了投影后长序列状态更新出错的
可能性，但不替代整层或整模型的数值验收。

诊断复现：使用下节整模型命令，将 cases 改为
`test/data/deepseek_v4_indexer_long_case.json`、输出 token 改为 1，去掉
`--compare-indexer`、加入 `--layer-diagnostics` 即可获得逐层结果。
同输入投影检查则将入口换为 `diagnose_projection_shapes.py`，使用相同模型/后端参数，
增加 `--probe-layers 4 --probe-output <结果路径>`。结果 JSON 记录了完整实际命令及
相关文件 SHA256。

### 本轮复现命令

在上节相同的 `PYTHONPATH` 和已构建、安装的 `sm_103` 动态库环境中执行。
更新动态库时先复制到临时文件再原子替换，避免覆盖正在被其他测试映射的 `.so`。

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=256G --time=01:00:00 \
  python3 tools/deepseek_v4_reference/run_single_gpu.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --converted-model /tmp/llaisys-deepseek-v4-flash-mp1 \
  --cases-file test/data/deepseek_v4_indexer_cases.json \
  --max-new-tokens 4 --max-seq-len 3072 \
  --torch-reference-structural-ops --dequantized-pytorch-linear \
  --native-structural-library python/llaisys/libllaisys/libllaisys.so \
  --operator-library-linear --native-compressor --native-indexer --compare-indexer \
  --output benchmark_results/deepseek_v4_b300_native_indexer_accuracy.json

srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:10:00 \
  python3 tools/deepseek_v4_reference/bench_indexer.py \
  --library python/llaisys/libllaisys/libllaisys.so \
  --output benchmark_results/deepseek_v4_b300_indexer_benchmark.json
```

合并回归命令如下，同样需要在已分配 GPU 的 Slurm 作业内运行：

```bash
python3 -m unittest test.test_deepseek_v4_indexer \
  test.test_deepseek_v4_compressor test.test_deepseek_v4_native \
  test.test_deepseek_v4_reference test.test_deepseek_v4_manifest -v
```

真实模型测试为 batch=1、输出 4 token、最大缓存长度 3072，
没有启用 Prefix Cache 或 CUDA Graph；完整重放会执行超过两千次整模型前向。

## 2026-09-08：恢复完整 TileLang 基线与多算子后端接口

### 编译错误根因与环境修复

先前 `DecoupleTypeCast` 的 `condition.dtype().is_bool()` 失败不是模型算子数值错误。
最小 CPU TIR 构造已经可以复现：

- 原 `apache-tvm-ffi==0.1.2` 把 `tir.const(True)` 编码为 code=1、bits=1，即旧式 uint1；
- TileLang 内置 TVM 的 `DataType::is_bool()` 要求 code=6；
- TIR 比较表达式在同一进程生成 code=6、bits=8，能够用于 `Allocate`；
- 独立安装 `apache-tvm-ffi==0.1.8.post2` 后，常量也使用 code=6、bits=8，
  `Allocate`、原始 FP4 GEMM 和 sparse attention 均通过编译与运行。

没有删除编译 pass、没有修改模型 `kernel.py`、没有绕过类型检查。
兼容版本装在个人目录 `.local/deepseek-v4-tilelang-compat`，原环境和共享目录未覆盖。
固定依赖见 `tools/deepseek_v4_reference/requirements-tilelang.txt`。

同时通过只解包 Ubuntu `libpython3.12-dev=3.12.3-1ubuntu0.16` 到个人目录，补齐开发
头文件；使用 `build_hadamard.py` 构建上游
`Dao-AILab/fast-hadamard-transform` 的原始 C++/CUDA 源码。
源码 commit：`e7706faf8d1c3b9f241e36860640ad1dac644ede`。
架构来自已分配 GPU 的 capability，实际为 `sm_103`，没有沿用 4060 参数。

Hadamard 现在默认要求 CUDA 扩展；若要重现旧 PyTorch FWHT 路径，必须明确指定
`--hadamard-backend torch-reference`，JSON 标记 fallback=true。导入失败不会静默回退。

### 完整模型参考基线

`benchmark_results/deepseek_v4_b300_tilelang_accuracy.json`：

- 实际模型：`DeepSeek-V4-Flash-0731`，43 层、完整转换后的 MP1 权重；
- B300 SXM6 AC × 1，CC 10.3；CUDA 13.0；PyTorch 2.13.0+cu130；
- TileLang 0.1.8、TVM-FFI 0.1.8.post2、上游 Hadamard CUDA；
- dense W8A8、expert W4A8，BF16 tensor/缓存，量化 scale 为 UE8M0；
- batch=concurrency=1，输入为 5/5/5/3 token，输出各 4 token，最大缓存 512；
- 无 Prefix Cache、无 CUDA Graph、无 fallback；
- 四个语义样例均通过：Paris、2、cold、北京；
- 逐算子实际调用：FP8 GEMM 12,236、FP4 GEMM 44,352、sparse attention 1,634、
  HC 3,268、FP8 quant 58,495、FP4 quant 1,071、Hadamard CUDA 1,071；
- 四份逐步 logits 已保存为 safetensors（每份约 2 MB，未纳入 Git）。

本次不再使用自写结构算子或 FP32 解量化 cuBLAS 路径作为唯一正确性参考。
但模型组织仍来自随权重提供的 `model.py`，不能宣称独立 llaisys runner/serving 已完成。

### 重要精度口径修正

完整 TileLang 基线自身，full prefill 与逐 token replay 的 logits cosine 为
0.998664–0.999514，最大绝对差约 0.921–2.037，四例首 token argmax 一致。
因此“full/replay 不逐值相同”不能单独证明是 llaisys 自写算子出错，也不能单凭
argmax 相同宣称数值验证完成。还需要相同执行形状、相同输入历史的后端对照。

runner 新增 `--dump-logits-dir` 与 `--compare-logits-dir`。比较时将基线的生成 token
作为后续输入，避免采样分叉掩盖误差；逐步报告 cosine、relative L2、max absolute
error 和 argmax。当前替换验收门槛为 cosine≥0.999、relative L2≤0.01、argmax 相同。
这是一组明确的短输入回归门槛，不是完整模型统计精度评价，也不是通用量化容差。

### 测试与接口

新增六类核心算子的版本化契约、注册、显式绑定和计数，并实现 TileLang、DeepGEMM
GEMM adapter。DeepGEMM 接口消费现成 FP8 activation，不重复量化，明确区分
W8A8 的 128×128 weight scale 与 W4A8 的 1×32 weight scale。

B300 回归 69 项通过、0 跳过、30.093 秒，包含首次 JIT；日志：
`benchmark_results/deepseek_v4_b300_tilelang_regression.log`。覆盖：

- 六类 TileLang 核心算子及不同 GEMM 行数、非对齐 candidate/mask；
- FP8/FP4 packed quant 与 inplace QDQ 一致性；
- 上游 Hadamard CUDA 与独立 FWHT 数值对照；
- DeepGEMM 与 TileLang 同输入 GEMM 对照；
- 9 项注册/选择/契约/异常/不可见 fallback 防护测试；
- 原有 DeepSeek manifest/reference/native/compressor/indexer 回归。

完整目标与剩余验收见 `DEEPSEEK_V4_BASE_ADAPTATION_PLAN.md`，本轮没有修改
scheduler、block manager、Qwen 模型或提交历史。

### 参考步骤耗时（非 serving 性能）

| 输入 token | 输出 token | TileLang prefill ms | 平均 decode ms/token |
|---:|---:|---:|---:|
| 5（France） | 4 | 748.250 | 284.787 |
| 5（算术） | 4 | 734.156 | 282.742 |
| 5（反义词） | 4 | 815.972 | 281.208 |
| 3（中文） | 4 | 568.781 | 282.533 |

commit=`05784b4da6483f77821a061e521ea1b98fa271e1`，dirty worktree，相关文件
SHA256 与完整命令记录在 JSON。每例 warmup 为一次 full prefill + 一次逐 token replay，
计时生成一次，decode 仅 3 个样本，不提供缺乏样本意义的 P95/P99；这些不是正式
TTFT/TPOT。PyTorch allocator 峰值约 158.565 GiB，不等同于所有 GPU 原生分配总量。

### 复现环境与命令

先以个人目录优先设置 `PYTHONPATH`，顺序为：

```text
/home/lcpu/57682329/.local/deepseek-v4-tilelang-compat
/home/lcpu/57682329/.local/deepseek-v4-hadamard
/home/lcpu/57682329/workspace/llaisys/third_party/DeepGEMM/build-llaisys/lib.linux-x86_64-cpython-312
/home/lcpu/57682329/.local/deepseek-v4-reference
/home/lcpu/57682329/.local/deepseek-v4-tokenizer
```

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=256G --time=00:35:00 \
  python3 tools/deepseek_v4_reference/run_single_gpu.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --converted-model /tmp/llaisys-deepseek-v4-flash-mp1 \
  --cases-file test/data/deepseek_v4_accuracy_cases.json \
  --max-new-tokens 4 --max-seq-len 512 \
  --gemm-backend tilelang --hadamard-backend cuda \
  --dump-logits-dir benchmark_results/deepseek_v4_tilelang_golden \
  --output benchmark_results/deepseek_v4_b300_tilelang_accuracy.json
```

同输入后端对照时，将 `--gemm-backend` 改为 `deepgemm`，将 `--dump-logits-dir`
改为 `--compare-logits-dir`，输出写入独立结果文件。不能把不同 checkpoint、不同
tokenizer 或不同模型配置下的 logits 文件混用。

### DeepGEMM 整模型替换：执行完成，但精度门槛未通过

结果文件：`benchmark_results/deepseek_v4_b300_deepgemm_tilelang_comparison.json`。
只将 FP8/FP4 GEMM 切换到 DeepGEMM 2.6.1；激活量化、sparse attention、HC 仍使用
相同 TileLang 源码，Hadamard 仍使用同一 CUDA 扩展。4 例 × 4 步，以保存的 TileLang
生成 token 作为后续输入：

- 16/16 步预测 argmax 相同，四例预期语义均命中；
- logits cosine：0.996876–0.999642；
- relative L2：0.027192–0.081554，即约 2.72%–8.16%；
- max absolute error：0.772237–1.820665；
- `all_golden_comparisons_pass=false`，进程按验收门槛退出码 2；
- 没有异常 fallback，实际 DeepGEMM FP8/FP4 调用 12,236/44,268 次。

不能因为单算子测试通过或 token 相同就认定 DeepGEMM 整模型数值验收通过。
本次不放宽阈值、不把实验结果标绿。默认仍是 TileLang，DeepGEMM 只作为显式实验
后端。尚不能把误差唯一归因为 GEMM 累加次序、MoE 路由放大或某个具体算子；应先
确认基线跨进程复现，再抓取相同输入的真实逐层投影做定位。

随后完成了独立进程的 TileLang 自身复测，结果为
`benchmark_results/deepseek_v4_b300_tilelang_reproducibility.json`：四例共 16 步的
完整 logits 与已保存基线逐值一致，max absolute error=0、relative L2=0，退出码 0。
因此这组短输入已经具有可重复的参考输出；DeepGEMM 本轮差异不是由这组基线的
跨进程不稳定造成。此结论仍不扩展到尚未跑完的长序列/不同 shape。

Hadamard 构建复现（头文件已解包到上述个人目录）：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:15:00 \
  python3 tools/deepseek_v4_reference/build_hadamard.py \
  --source /home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --build-dir /home/lcpu/57682329/.local/deepseek-v4-hadamard \
  --python-include /home/lcpu/57682329/.local/deepseek-v4-python-dev/usr/include/python3.12 \
  --python-include /home/lcpu/57682329/.local/deepseek-v4-python-dev/usr/include
```

新环境依赖可以使用 `uv pip install --python /usr/bin/python3 --no-deps --target
/home/lcpu/57682329/.local/deepseek-v4-tilelang-compat apache-tvm-ffi==0.1.8.post2`
独立安装，并使该目录排在旧依赖目录之前。不要直接升级共享环境。

## 2026-09-08：独立模型完整权重接入与增量 chunk 验证

### 本轮完成的独立执行边界

新增 `python/llaisys/models/deepseek_v4_model/`，不再通过上游 `model.py` 的全局变量
或 monkey patch 来组织独立模型。配置、权重模块和请求状态分开，算子实现由模型
实例持有 `BoundOperators`；调度、采样和 tokenizer 不放进模型 forward。
保留参考方程及权重名称兼容性，并在目录 `NOTICE` 中保留 DeepSeek MIT 许可与归属。

- `config.py`：实际配置解析和维度检查。
- `layers.py` / `model.py`：独立 Attention、Compressor、Indexer、MoE、HC、输出 head。
- `weights.py`：先检查完整 header，再逐 tensor 搬运到 GPU；不创建巨大 CPU 权重字典，
  不将整份 FP4 权重展开成 FP32。
- `state.py`：请求级窗口/压缩/Indexer 状态，模型及权重代际 ownership、reset、close；
  执行异常后状态标为 invalid，重新加载权重后旧请求不能继续使用。
- `run_independent.py`：只导入显式 TileLang kernel，不导入上游模型代码，固定输入历史
  对照保存的 logits，可保存每一层最后一个 token 的 hidden tensor。

这仍是 Python correctness executor，并非已经完成 C++/pybind batch runtime 或
serving 接入。普通 MoE 暂为逐 expert 的 PyTorch 调度，不宣称 fused/grouped MoE。

### 严格加载的实际结果

GPU 节点本地文件：`/tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors`。

- 文件大小：168,430,204,680 字节。
- header SHA256：`80fb82525d75f1580d3bfca8d0f5340fc8cc4a8f243a1e6ae3bb36cf6a6df22c`。
- 主模型参数 tensor：67,569；实际 GPU 参数字节：159,117,270,492。
- 主模型 missing/unexpected：均为空；不使用 `strict=False` 掩盖未知权重。
- 额外 4,702 个 MTP/DSpark tensor 的名称、shape、dtype 按实际 3-stage 配置校验，
  但不加载、不执行。不能宣称 MTP 推测解码已经实现。
- 共 363 项显式 dtype 转换，主要是 BF16 norm/投影到 FP32，以及 hash expert 表
  I64 到 I32；hash 表转换前检查 expert ID 范围。
- safetensors F4 header 保存的是逻辑 K，PyTorch 张量最后一维是 K/2 packed；loader
  对这两种 shape 分别解释，没有错误地二次压缩或将 FP4 当成普通 uint8 权重。
- 首次完整加载 24.376 秒，后续同节点复测约 24.2–24.3 秒；这是热本地文件环境，
  不能当作共享存储冷启动时间。

### 独立执行器整模型对照通过的范围

结果：`benchmark_results/deepseek_v4_b300_independent_accuracy.json`，增加 chunk 代码后
再次复查：`benchmark_results/deepseek_v4_b300_independent_full_recheck.json`。

环境仍为 B300 SXM6 AC × 1、CC 10.3、CUDA 13.0、PyTorch 2.13.0+cu130、
TileLang 0.1.8、TVM-FFI 0.1.8.post2、上游 Hadamard CUDA。
主模型 43 层，dense W8A8、expert W4A8、UE8M0 scales、BF16 hidden/cache。
batch=concurrency=1，max_seq_len=512，输入长度 5/5/5/3/134，输出各 4 token。

两次独立执行器运行的 20 步 logits 均与保存的上游 TileLang 输出逐值一致：
max absolute error=0、relative L2=0、全部 argmax 相同。
134-token 样例跨过 ratio-128 压缩边界，但还没有触发真实 Indexer Top-512 候选裁剪。
这是明确输入集合下的实现一致性验证，不等于完整统计精度评价。

独立 GEMM adapter 现在显式创建 BF16 输出，调用同一 TileLang 原始 kernel，
不依赖 `torch.get_default_dtype()`。整模型运行时全局默认仍为 FP32，运行前后不变。
新增非默认 CUDA stream 测试与原 wrapper 逐值对照均通过。

首次运行（不抓取中间层）仅供诊断的步骤耗时：

| 输入 token | 输出 token | full prefill ms | decode 平均 ms/token |
|---:|---:|---:|---:|
| 5（France） | 4 | 616.075 | 216.884 |
| 5（算术） | 4 | 603.521 | 216.847 |
| 5（反义词） | 4 | 679.399 | 216.536 |
| 3（中文） | 4 | 463.987 | 217.370 |
| 134（重复词） | 4 | 1257.158 | 218.159 |

commit 仍为 `05784b4da6483f77821a061e521ea1b98fa271e1`，dirty worktree，源码 SHA256
与完整命令见 JSON。每例 warmup 一次 full prefill + 一次 decode；测量 1 次，decode
只有 3 个样本，不给没有统计意义的 P95/P99。这不是正式 serving TTFT/TPOT。
无 CUDA Graph、无 Prefix Cache、无 paged kernel、无 fallback。
PyTorch allocator 峰值约 148.340 GiB；本 runner 没加载 MTP，不能将相对上游 runner
的显存下降直接当成同功能下的缓存优化收益。

### 多 token 增量 prefill：已执行，但精度未验收

新增真正按 chunk 计算投影/Attention/MoE 的增量路径，并非逐 token 重放：

- Compressor 将未完成组的投影状态与新 chunk 拼接，按完整组输出；ratio-4 保留
  上一组重叠状态，ratio-128 保留未完成组，不重复计算历史 prompt 投影。
- window 环形缓存只保存最后一个窗口，单次写入的物理 slot 不重复。
- Attention 临时视图由至多 window-1 个旧原分辨率 slot、新 token 和压缩缓存组成，
  明确是连续参考视图，尚未接入 paged backend。
- Indexer 按每个 query 的绝对位置进行因果屏蔽。
- 新增单次多 token chunk 只执行一次 layer forward 的结构检查。

实际 134-token 输入切为 65/65/4，4 步输出 token 与 full 基线相同，但数值失败。
初版结果 `deepseek_v4_b300_independent_chunk65_accuracy.json` 的 relative L2 为
6.029%–9.467%，cosine 为 0.996010–0.998189。
将窗口候选统一为“有效索引在前、-1 在后”以匹配 full prefill 的分块顺序后，结果
`deepseek_v4_b300_independent_chunk65_ordered_accuracy.json` 仍失败：
relative L2=5.543%–7.550%，cosine=0.997428–0.998468。
门槛仍为 relative L2≤1%、cosine≥0.999、argmax 一致；没有放宽阈值或静默回退。

逐层文件由 `compare_layer_dumps.py` 离线比较，结果保存在
`deepseek_v4_independent_chunk65_ordered_layer_comparison.json`：prefill 第 0 层末 token
仅有约 1.19e-7 的绝对差，第 1 层已有约 0.917% relative L2，第 3 层约 8.962%。
第 0/1 层均不使用压缩缓存，因此不能把误差直接归咎于 MLA Compressor。
这只是定位范围，不是最终根因。需要继续比较同输入投影、QAT、窗口 Attention
及 MoE 的中间张量，区分形状相关计算差异和缓存逻辑错误。

随机 BF16 小配置也保留了失败回归：4 种非对齐切分的 logits relative L2 约
2.330%–4.739%。同一小配置逐 token/full 自身也有差异；改变 Indexer 候选集合大小
会影响并列 Top-K 选择，不能只用最终 token 是否相同判断正确。
单独 Compressor 在相同投影输入、ratio-4/128 和多种非对齐 chunk 下的 cache 对照
逐值一致，但这个局部结果不能替代整模型验收。

最终 B300 回归共 **84 项，83 项通过，1 项测试中的 4 个 chunk 数值子用例失败**，
0 跳过。日志 `benchmark_results/deepseek_v4_b300_independent_regression.log`，退出码 1。
失败测试没有删除、跳过或改为 expected-failure。普通 full prefill/decode 独立模型
对照仍通过，chunk 路径只能通过 `--prefill-chunk-size` 显式进入实验测试。

### 参考产物与复现注意

`run_independent.py` 检查模型路径、上游 config/kernel/model 源码 hash、最大序列长度、
tokenizer 编码、生成 token 历史、logits shape 与有限性。基础 tokenizer 词表为
128000，包含新增 token 后为 129280，logits 宽度按模型配置 129280 验证。
产物记录实际 checkpoint header hash、每个 golden 文件 hash、执行源码 hash。
旧 golden 报告没有 checkpoint payload digest，当前明确记录这一身份绑定限制；
下一轮需要在参考生成时同步固化更强的权重身份，不能反向补写成已经验证过。

除之前环境路径外，`PYTHONPATH` 最前面加本仓库 `python` 目录：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=256G --time=00:35:00 \
  python3 tools/deepseek_v4_reference/run_independent.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --baseline benchmark_results/deepseek_v4_b300_tilelang_accuracy.json \
    benchmark_results/deepseek_v4_tilelang_golden \
  --baseline benchmark_results/deepseek_v4_b300_tilelang_long_accuracy.json \
    benchmark_results/deepseek_v4_tilelang_long_golden \
  --max-seq-len 512 \
  --output benchmark_results/deepseek_v4_b300_independent_accuracy.json
```

使用 `--prefill-chunk-size 65` 显式测试实验 chunk；`--dump-layer-dir <目录>` 记录中间层，
此时步骤计时包含 GPU 中间层 clone 的开销，不与未抓取版本比较性能。
权重、中间 tensor safetensors 和大日志保持忽略，仅小型 JSON 结果加入可跟踪列表。
未修改 scheduler、Qwen、通用 block manager 或提交历史；完整基础适配目标仍未完成。

## 2026-09-08：增量精度定位与 19 类算子后端接口

### 分开回答两个数值问题

本轮不把“同策略 full/chunk 一致”和“与原始 TileLang golden 一致”混成一项。
前者检查增量执行与缓存的一致性，后者检查后端替换后的数值偏移。原始通过阈值
保持 relative L2≤1%、cosine≥0.999 且 argmax 相同；默认后端没有改动。

逐层追踪得到的证据链：

1. 相同输入下，早期纯窗口 Attention 可以逐值一致；微小差异先出现在普通 FP32
   投影与 HC 路径，不能仅据最终 logits 把问题归咎于压缩缓存。
2. 固定普通 GEMM 的 M 维后，早期差异显著减少。量化临界值可把微小输入舍入
   放大为 BF16/FP8/FP4 输出差异，后续 MoE 会继续传播差异。
3. 在压缩投影、pooling 输出、Indexer scores/indices 都一致时，短首块仍可能改变
   window/压缩候选的分段位置，进而改变 TileLang 64-candidate online-softmax 分组。
   固定 metadata 段宽可以消除这一组差异，不需要重复或展开 latent payload。
4. 剩余尾部 token 的差异进一步定位到随 tensor 形状变化的 inverse RMS 计算。
   固定 32 行计算并保留原 dtype 的 square/mean/rsqrt 后，前八层全部观测点逐值一致。

这里的固定策略包括 `torch-fixed32`、`index_ascending` 和固定 Attention metadata。
不通过全局关闭量化、改变权重 dtype、移除编译 pass 或静默 fallback 获得结果。

### 真实完整权重的固定策略结果

`benchmark_results/deepseek_v4_b300_fixed_rms_profile_accuracy.json`：

- 单卡 B300、完整 43 层主模型、W8A8 dense / W4A8 expert、BF16 hidden/cache。
- 输入 5/5/5/3/134 token，输出各 4 token；batch=concurrency=1，max_seq_len=512。
- chunk size=65。134-token 实际分成 65/65/4；四个短例没有真正分块，不能据它们
  宣称短例分块通过。
- 134-token 的 chunk/full 四步 logits 在同一固定策略下逐值一致，relative L2=0。
- 与原始 golden 的比较仍失败；总结果 `all_passed=false`，退出码 2。

19 类接口接通后，补测
`benchmark_results/deepseek_v4_b300_model_ops_chunk2_accuracy.json`：

- chunk size=2：5-token 为 2/2/1，3-token 为 2/1，134-token 为 67 个块。
- 五例共 20 步均与同策略 full 的完整 logits 逐值一致，max absolute error=0，
  relative L2=0；prefill 后三个 decode 步骤使用相同的参考 token 历史。
- 与原始 golden 的 relative L2=3.084%–8.595%，20 步 argmax 相同；仍未通过原始
  数值门槛，`all_passed=false`，没有更换默认后端。

这是具体样例下的增量一致性进展，不是长序列、长输出或任务级统计精度验收。
尤其尚未覆盖实际 Top-512 裁剪边界，不能用这些样例宣称支持全部上下文长度。

### 模型算子接口扩展

目前共有 19 类真实接入的契约：原六类 TileLang 核心算子，普通/分组 Linear、
inverse RMS、RMSNorm、RoPE、Indexer scores/top-k、router、compressor pooling、
expert activation、MoE dispatch/combine 和 HC sum。

新增 `deepseek_v4_model_ops.py` 提供显式 PyTorch 参考实现。模型层继续持有权重和
请求状态，后端仅处理计算。Hadamard 仍是独立显式 CUDA 函数接口。

MoE dispatch 返回独立 packed hidden / routing weights、token indices 和设备端
expert offsets。参考执行器逐专家执行，combine 按 expert ID 递增顺序做 FP32
累加，保留 route weight 在 SwiGLU 内、down projection 前应用的量化语义。
不使用可能改变归约顺序的跨 expert 原子累加来冒充逐值一致。

当前仍有 offsets 到主机的同步及逐专家 Python 循环，不宣称已经是高性能 grouped
GEMM、device-only metadata 或 EP。完整接口与后续算子库接入方式记录在
`DEEPSEEK_V4_OPERATOR_BACKENDS.md`。

### 默认路径完整权重复测

结果：`benchmark_results/deepseek_v4_b300_model_ops_accuracy.json`。

- B300 SXM6 AC × 1，CC 10.3；CUDA 13.0，PyTorch 2.13.0+cu130，TileLang 0.1.8，
  TVM-FFI 0.1.8.post2，上游 Hadamard CUDA。
- 严格加载 67,569 个主模型 tensor，4,702 个 MTP tensor 仅校验结构；MTP 不执行。
- 原始数值策略、完整 43 层、输入 5/5/5/3/134、输出各 4；没有分块、Prefix Cache、
  CUDA Graph、paged kernel 或 fallback；全局默认 dtype 保持 FP32。
- 20 步 logits 与保存的原始 TileLang golden 逐值一致；退出码 0。
- 19 类接口都有真实调用且无异常：例如 router、dispatch、combine 各 1,290 次，
  RMSNorm 5,818 次、RoPE 5,128 次、pooling 628 次。
- 报告记录 branch、commit、dirty worktree 和源码 SHA256。执行后核对所有记录的
  SHA256 与当前文件一致，包括新结构算子文件；旧 golden 缺少权重 payload digest
  的限制仍明确保留。

默认路径的诊断步骤耗时（不是正式 serving TTFT/TPOT）：

| 输入 token | 输出 token | full prefill ms | decode 平均 ms/token |
|---:|---:|---:|---:|
| 5（France） | 4 | 567.074 | 207.362 |
| 5（算术） | 4 | 557.125 | 206.925 |
| 5（反义词） | 4 | 625.199 | 206.839 |
| 3（中文） | 4 | 430.374 | 206.866 |
| 134 | 4 | 1140.345 | 208.946 |

commit=`05784b4da6483f77821a061e521ea1b98fa271e1`，dirty worktree。每例 warmup
一次 prefill 和一次 decode，测量一次，decode 只有三个样本；没有有效的 P95/P99，
不据此宣称优化收益。完整运行命令、峰值分配和后端统计以 JSON 为准。实验 chunk
报告中的旧 `warmup_kind` 标签仍写 full prefill；实际 warmup 跟随所选 chunk 模式，
这一标签不能用来推断未执行的 full warmup，后续 runner 需同步修正。

### 单元与结构回归

Slurm 单卡命令为 `python3 -m unittest discover -s test -p 'test_deepseek_v4*.py' -v`。
环境路径同前，含 DeepGEMM 扩展目录。完整日志：
`benchmark_results/deepseek_v4_b300_model_ops_regression.log`。

- 共 99 项，98 项通过，0 跳过，耗时 6.389 秒（已有 JIT 缓存）。
- 唯一失败方法仍是原始数值策略的非对齐 chunk 测试，其四个 subtest 失败。
  没有删除、跳过或放宽这组验收。
- 新增 Norm/RoPE、Indexer BF16 边界、router/hash/selection bias、FP64 pooling
  oracle、expert activation 量化顺序、空专家/负载不均 packed dispatch/combine、
  dtype/layout 错误及后端统计测试均通过。
- 新增固定策略小模型测试覆盖 CPU 和本次分配的 CUDA：四种跨 window、ratio-4、
  ratio-128 的非对齐分块，逐层末 token、缓存和三个 decode 步骤均逐值一致。

### 尚未完成

固定策略仍需要独立参考、更多真实输入和任务精度验证；原始 chunk 门槛仍为失败。
还需真实长输入 Top-512、权重身份绑定、paged/Prefix Cache 执行、批专家执行、
C++/pybind runtime 与既有 scheduler/serving 接通、正式性能矩阵；多卡 TP/EP 仍需
配额和真实多卡验证。没有把本轮接口扩展或短样例一致性等同于完整基础适配完成。

## 2026-09-08：完整权重身份、Top-512 长输入与真实 paged latent 接入

### 原始参考与权重身份补强

此前待办中的 Top-512 筛选边界已用真实权重覆盖：输入 2105 token，ratio-4 层有
526 个压缩候选，超过 Top-512；完整 43 层、输出 16 token，原始 full prefill/decode
golden 保存在 `deepseek_v4_b300_tilelang_top512_accuracy.json` 及对应忽略的
safetensors 目录中。参考生成使用 `--reference-mode full-only --hash-checkpoint`；
没有执行 token replay，相关报告项保持 null/false，不补写为通过。

- 实际 MP1 权重大小：168,430,204,680 bytes。
- 完整文件 SHA256：`024836b412ccd5c539bb130842842247d9c427f99c3a1e21ed35ea08ddcc0aa2`。
- golden 文件 SHA256：`86ae149a3e019e8737266dcdda44f701eb20d0f4f03cfafc692cc37fc4fa9915`。
- 生产者与消费者分别以有限缓冲区扫描全部 header 和 tensor payload，运行结束再次
  检查文件状态；消费者校验 golden 摘要。这是产物一致性，不是签名来源认证。
- 独立连续执行器的对照报告为 `deepseek_v4_b300_independent_top512_accuracy.json`：
  16 步 logits 逐值一致、relative L2=0，且 `checkpoint_payload_verified=true`。

这只是具体长输入样例的正确性证据；原始 HF 分片直接加载、更多任务、长输出和
全上下文精度仍需验证。旧报告缺少完整权重摘要的限制没有被反向抹除。

### 现有 pybind 与 C++ 资源管理恢复

使用已有共享 xmake，而不是重新安装构建工具：

```text
/home/lcpu/62385178/.local/bin/xmake f -y --nv-gpu=y --flashinfer=y --cuda-arch=sm_103
  --python-bindings=y
  --python-include=/home/lcpu/57682329/.local/deepseek-v4-python-dev/usr/include/python3.12
  --pybind11-include=/home/lcpu/57682329/.local/deepseek-v4-reference/torch/include
/home/lcpu/62385178/.local/bin/xmake build -j 8 llaisys-python
```

`build_python_runtime.py` 根据 Slurm 分配 GPU 的实际 CC 生成 sm_103，并为解包的
Python 开发头补充 multiarch include 根。没有变更当前关闭的 MPI/NCCL 构建选项。
最初 smoke 误用了字符串 request_id；真实 SchedulePlan 合同为 uint64，脚本改为
101/102 后通过。这是 smoke 脚本问题，不是推理或 pybind ABI 故障。

`python/bindings/cache.cpp` 绑定真实 BlockManager / BlockPrefixCache，而非 Python
仿制实现：lease 持有强引用、自动释放、share、容量不足原子失败、前缀 retain/LRU，
拒绝跨 pool 或不同前缀复用同一已发布 payload。原有 C++ cache-core test 已通过。
本轮增加 `append` 所有权转移与 `mark_computed_counts` 批量有效长度更新，实际用于
V4 请求跨 block 追加。元数据操作保持 GIL，不宣称底层 manager 是并发容器。

最新构建报告：`deepseek_v4_b300_paged_bindings_build.json`，Slurm job 22960；
模块 SHA256 为 `dbb451e9817309be953e8cb1643ad95a19007cb7b8682eea09abbd182868cad3`。
构建和 SchedulePlan roundtrip 均通过，没有创建 Qwen 推理上下文或执行 Qwen 测试。

### V4 实际 paged latent 执行

新增 `deepseek_v4_model/paged.py`，显式选择 `--cache-layout paged` 后：

1. 模型前向开始前批量取得 C++ physical block IDs；block table 仅在增长时上传。
2. 新 full-resolution/compressed/index latent 写入对应物理槽，压缩投影的未完成组
   和 overlap 状态继续跨 chunk 保存在请求中。
3. 将窗口和压缩候选的逻辑索引映射为物理槽索引，保留原候选顺序和 mask；
   TileLang sparse attention 直接读取共享 pool，不 gather 历史 attention payload。
4. 全部模型层成功后更新 block validity 和 position；失败使请求失效，reset/close
   释放资源；容量不足不会推进请求状态。

新增三个可替换 cache 算子合同：map slots、write、read。与已有 19 类模型算子
共存，用户可用自己的算子库逐项注册替换；错误必须上抛，没有静默 fallback。

### 真实 43 层 full prefill/decode 结果

报告：`benchmark_results/deepseek_v4_b300_paged_top512_accuracy.json`。

- B300 SXM6 AC × 1，CC 10.3；PyTorch 2.13.0+cu130、CUDA 13.0、TileLang 0.1.8、
  TVM-FFI 0.1.8.post2、上游 Hadamard CUDA。
- `feat/vllm-paged-attention`，HEAD `05784b4da6483f77821a061e521ea1b98fa271e1`，
  dirty worktree；报告记录本轮 Python 源码、cache.cpp 和 `_C.so` 摘要。
- 模型 `DeepSeek-V4-Flash-0731`，完整 43 层，严格加载 67,569 个主模型 tensor。
  BF16 hidden/cache，量化 dense W8A8、expert W4A8；MTP 不执行。
- 输入 2105、输出 16、batch/concurrency=1、max sequence=4096、block size=128，
  32 个物理 block。无 chunk、Prefix Cache、CUDA Graph 或 fallback。
- 与保存的原始 TileLang golden **16 步 logits 逐值一致**；与同进程连续 cache
  路径也逐值一致。`all_passed=true`，每步最大绝对误差及 relative L2 均为 0。
- 运行结束 `num_free=32/32`。cache payload pool 为 208,535,552 bytes，包含所有
  层的 full-resolution/compressed/index pool；不包含参数及其他临时 tensor。
- 三类 cache 接口都有真实调用：map 2956 次、write 1066 次、read 378 次，异常 0。
  这里包含 warmup；read 来自 Indexer，不属于 attention 历史 gather。

诊断耗时：paged full prefill 3878.134 ms，后续 15 次 decode 平均 226.001 ms/token；
进程 peak allocated memory 为 160,446,034,432 bytes。每条路径 warmup 一次 prefill
加一次 decode，仅测量一次生成；没有统计可靠的 P95/P99，**不是 serving TTFT/TPOT
或优化收益声明**。完整命令和每步数据在 JSON 中。

### 明确保留的未完成边界

- 这是执行路径接入，不是仅定义 layout；但 payload 暂由 PyTorch tensor 持有，
  Python 仍发起模型计算，还需迁入 C++ PagedCacheStorage / batch runtime。
- attention 无历史 payload gather；Indexer 仍 gather 压缩 key，明确记录为 true。
- full-resolution 槽保留到请求释放，尚未按滑动窗口回收；不宣称长上下文显存最优。
- pool 限定单一 CUDA stream；多请求可在该流交错执行，跨流显式失败；尚无 mixed batch。
- Prefix Cache 必须连同压缩器 unfinished/overlap 状态接入，不能只复用 latent 就
  启用前缀跳过；当前 V4 prefix sharing/COW 尚未完成。
- 原始 chunk/full 的精度门槛不变；paged/contiguous 同形状一致性不能替代它。
- serving、正式性能矩阵、多卡 TP/EP 仍未完成；本轮不标记完整基础适配完成。

### 非对齐长 chunk 对照与回归

报告：`deepseek_v4_b300_paged_chunk257_accuracy.json`。相同真实 43 层权重和
2105-token 输入，按 **257×8 + 49** 切分，输出 16 token。block size 128，既跨
physical block，也跨 ratio-4/ratio-128 边界；实际仍包含 Top-512 筛选。

- paged 与相同分块顺序的连续缓存：16 步 logits 逐值一致，relative L2=0。
- 相对原始 full prefill golden：relative L2 范围 **6.356%–39.087%**，16 步 argmax
  相同，但未通过 1% 门槛，`all_passed=false`，进程按预期退出 2。
- 这证明本次 paged 布局在该输入下没有新增可观察数值差异，不证明 full/chunk 已
  一致，也不能仅凭 argmax 相同宣称任务精度正确；较大的差异仍需逐层定位。
- 没有改为逐 token replay、没有放宽门槛或更换默认数值策略；cache 全部释放回 pool。

runner 现在分别记录所选路径、可选 full 对照和可选同 schedule 连续布局对照的
warmup 方式，避免把分块对照误标为 full warmup。上面的两份 paged 报告在标签修正
后重新执行，最终命令/耗时/源码摘要以 JSON 为准。

新增 `run_tests.py` 保存完整回归日志、每个通过方法、失败 subtest、版本与源码摘要，
按方法和 subtest 分别计数，不把四个 subtest 错算成四个独立测试方法。
报告为 `deepseek_v4_b300_paged_regression.json`，Slurm job 22970：

- 共 **125 个测试方法，124 通过，1 个失败，0 error、0 skip**；7.936 秒（已有 JIT
  缓存）。源码在执行前后摘要一致。
- 唯一失败方法仍为默认非对齐 chunk/full 测试，四个 subtest relative L2 分别为
  2.768%、4.739%、2.860%、2.330%；原门槛保持不变。
- 新增 paged 模型/算子测试 11 项、native cache binding 测试共 10 项均通过。
  覆盖物理槽映射、共享 pool 指针、真正非连续 block table、请求交错、同 schedule
  full/chunk/decode 对照、容量不足原子失败、异常状态、reset/GC 回收、模型 generation
  隔离、算子契约版本拒绝、显式失败无 fallback，以及 GPU 非默认流的拒绝。

下一步需优先解决增量数值策略及其独立参考验收，补齐压缩器状态与 block Prefix
Cache 的真实共享/COW，然后接入既有 C++ batch execution 和 Python scheduler。
当前新增 payload 排列并非已有 `DeepSeekV4CacheLayout` 三 component 存储的可互换
指针格式；迁移到 C++ storage 时必须显式匹配 stride/offset 和 attention descriptor，
不能依据相同 block ID 就直接混用两种物理布局。

## 长分块诊断、独立数值策略参考与中间输出合同（2026-09-08）

### 长输入的最早可观察差异

`diagnose_independent_chunks.py` 新增 `--max-seq-len`，支持实际 2105-token 输入，
在加载权重前校验 token tensor，并要求 Slurm GPU。真实前三层（4704 个 tensor）、
max-seq-len=4096、257-token 分块的报告：

- `deepseek_v4_longchunk_default_trace.json`：layer 0 attention 输出一致，最早非零
  观测点为 `layers.0.ffn_norm.input`，relative L2 约 0.0000190947，即 FFN 前的 HC
  处理已经存在小差异；它早于 ratio-4 Indexer 层，不能全部归因于 Top-k 或页表。
- `deepseek_v4_longchunk_fixed_trace.json`：选择 `torch-fixed32`、候选同分按下标
  升序及固定 attention metadata 后，121 个观测字段全部逐值一致，且覆盖真实
  Top-512 候选数。此处只有三层，不能外推完整 43 层精度。

### 独立策略参考的边界

新增 `published_fixed_profile.py`，以显式 `fixed32-reference-v1` 生成独立参考。
它在内存中转换上游源码的已审计调用点，不改共享文件、不导入本项目模型控制流：

- 4 个普通投影调用点、4 个 inverse-RMS 调用点（包括 RMSNorm）；
- 3 个 HC 归约、1 个 grouped projection、1 个 Indexer Top-k；
- 1 个窗口 metadata、2 个 sparse attention 调用点。

保持 BF16/FP32 边界和 W8A8/W4A8 量化合同，不通过 FP64 或重复量化更换精度。
报告标记 `unmodified_published_baseline=false`、MTP 不支持；`forward_spec` 明确失败。
源文件、转换后 AST、策略实现和 golden 摘要绑定到报告，消费者拒绝混用策略。
这一参考用于区分“模型/缓存接入错误”和“不同数值策略的差异”，不是把原始失败
报告改成通过。默认仍为原始 TileLang/PyTorch 数值路径。

真实 43 层固定策略 full golden 已生成于
`deepseek_v4_b300_fixed_oracle_top512.json`，输入 2105、输出 16、batch/concurrency=1，
max-seq-len=4096，CUDA Graph/Prefix Cache 均关闭、无 fallback。主模型权重完整
SHA256 为 `024836b412ccd5c539bb130842842247d9c427f99c3a1e21ed35ea08ddcc0aa2`；
missing/unexpected 均为空。该运行是 `full-only`，没有执行真实全长逐 token replay，
对应字段为 null，不记作通过。

### 只在最后一个 chunk 生成输出

新增 `emit_logits=False`：跳过 HC head、最终 norm 和词表投影，所有主模型层、
compressed/Indexer cache 写入、block validity 和 request position 正常执行。返回
`logits=None`，不隐式采样或补算。模型默认保留输出；分块 runner 默认只请求最后
一个 chunk 的 logits，诊断选项 `--intermediate-prefill-logits` 可恢复旧行为。

验证覆盖 CPU/GPU、连续/paged cache、3/2/7/115/10 非对齐分块、层输出捕获、后续
三个 decode、资源释放、非法输出策略的无副作用拒绝，以及计算异常后的 reset。
通过 spy 和算子调用统计验证：中间 chunk 没有调用最终 norm/head，但全部 attention
层仍执行。独立上游固定策略小配置测试也改用这一接口，full、逐 token replay、
独立 paged chunk/decode 仍逐值一致。

`deepseek_v4_b300_final_chunk_regression.json`：**132 方法，131 通过，1 失败，
0 error、0 skip**，源码前后摘要一致。失败仍是原始 chunk/full 方法的四个 subtest；
没有放宽原 1% 门槛。本轮不把该独立 Python correctness executor 称为最终 C++
runtime 或 serving 完成。

### 完整 43 层 paged 分块对独立固定策略参考

消费者报告：`deepseek_v4_b300_fixed_oracle_chunk257.json`。

| 条件 | 实际执行 |
|---|---|
| 代码 | `feat/vllm-paged-attention`，HEAD `05784b4da6483f77821a061e521ea1b98fa271e1`，dirty worktree；源码摘要逐文件记录 |
| 硬件/软件 | 单张 B300 SXM6 AC，CC 10.3；PyTorch 2.13.0+cu130，CUDA 13.0，TileLang 0.1.8 |
| 模型/权重 | DeepSeek-V4-Flash-0731，完整 43 主干层、67569 个主模型 tensor；W8A8 dense、W4A8 experts、BF16 hidden/cache、FP32 结构运算 |
| 输入/输出 | 2105 / 16 token；batch=1，concurrency=1；257×8+49 分块 |
| 算子与数值策略 | 六类 TileLang 核心、十三类 torch-fixed32 结构算子；index_ascending + fixed metadata；CUDA Hadamard |
| 缓存 | paged，block size=128、32 blocks；attention 直接读 pool，Indexer 仍显式 gather |
| 其他开关 | Prefix Cache=off、CUDA Graph=off、fallback=false；中间 8 个 chunk 不计算输出 head |
| 正确性测量 | 每条路径 warmup 一次对应 prefill+decode，再运行一次 16 步；另有同策略 full 和同 schedule 连续缓存对照，不含调度/网络 |

结果：

- paged 分块对独立上游固定策略 full golden：16 步全部逐值一致，relative L2=0。
- paged 分块对同策略独立 full：16 步全部逐值一致。
- paged 分块对同 schedule 连续缓存：16 步全部逐值一致。
- `all_passed=true`、`all_logits_exact=true`，进程退出 0；所有 blocks 释放后为
  32/32 free、0 cached。完整 checkpoint 与 golden 身份分别由生成器、消费者校验。
- 这次对照不再仅是“本模型 full 对本模型 chunk”的自洽证明：参考控制流来自
  独立上游模型。但明确修改了其数值策略，`unmodified_published_baseline_tested=false`。

生成参考的核心参数：

```bash
python3 tools/deepseek_v4_reference/run_single_gpu.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --converted-model /tmp/llaisys-deepseek-v4-flash-mp1 \
  --cases-file test/data/deepseek_v4_indexer_long_case.json \
  --max-new-tokens 16 --max-seq-len 4096 \
  --gemm-backend tilelang --hadamard-backend cuda \
  --reference-profile fixed32-reference-v1 --reference-mode full-only --hash-checkpoint \
  --dump-logits-dir benchmark_results/deepseek_v4_fixed_oracle_top512_golden \
  --output benchmark_results/deepseek_v4_b300_fixed_oracle_top512.json
```

消费参考的核心参数：

```bash
python3 tools/deepseek_v4_reference/run_independent.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --baseline benchmark_results/deepseek_v4_b300_fixed_oracle_top512.json \
    benchmark_results/deepseek_v4_fixed_oracle_top512_golden \
  --max-seq-len 4096 --reference-profile fixed32-reference-v1 \
  --model-linear-backend torch-fixed32 --indexer-tie-policy index_ascending \
  --attention-metadata-policy fixed --cache-layout paged --prefill-chunk-size 257 \
  --compare-full-in-process --compare-contiguous-in-process \
  --output benchmark_results/deepseek_v4_b300_fixed_oracle_chunk257.json
```

两者都在 `srun --partition=gpu --gres=gpu:1 --cpus-per-task=12 --mem=80G
--time=00:20:00` 分配内执行，并使用前述隔离 PYTHONPATH；不是登录节点执行 CUDA。

### 仍未通过的原始策略与任务精度边界

对两份已保存的 full golden 做 CPU 只读分析，确认 input IDs、16 个生成 token
及对应 decode 历史均相同；固定策略相对原始上游 logits 的 relative L2 为
**6.316%–28.251%**，最小 cosine 约 **0.968856**，仍不通过原始 1% 门槛。
这不是同输入下的逐值正确性通过，也不允许将旧失败报告改标。

该重复文本样本的原始下一 token 概率全部高于 0.9999；其分布 KL/total variation
虽很小，也不能证明多任务精度。下一步必须增加使用实际 V4 chat 编码的多样输入、
更长输出及任务级验证，尤其覆盖非饱和分布和 Top-k 变化，不能靠重复文本通过扩大
精度声明。共享 tokenizer_config 没有 chat_template；官方 generate.py 使用
`encoding/encoding_dsv4.py` 的 `encode_messages(..., thinking_mode="chat")`，不能
把既有 raw completion 测试当作真实 chat 端到端验收。

后续继续完成压缩器 overlap 状态与 Prefix Cache 共享/COW、窗口回收、C++ storage
和 batch execution、现有 Python scheduler 接入及正式性能测试。固定 32 行策略的
额外 kernel/Python 调用也仍需优化；本节仅给正确性结论，不给 serving TTFT/TPOT
或性能提升结论。

## V4 paged payload 迁入既有 C++ storage（2026-09-08）

### 实现与所有权

新增模型专属 `src/models/deepseek_v4/cache_layout.hpp`，按每层定义 TileLang 所需的
interleaved latent/index descriptor，再复用既有 `core::PagedCacheStorage` 分配
和持有 CPU/CUDA 内存。没有新增 block allocator，没有修改 Qwen 模型实现，也没有
把 V4 cache 变成普通 K/V。模型专属布局、通用 block 生命周期、算子实现仍分离。

`python/bindings/paged_storage.cpp` 提供 `V4PagedStorage/V4CacheView`：

- 每层 latent/index 由 C++ allocation 持有；PyTorch 仅建立零拷贝 BF16 视图。
- DLPack legacy/versioned capsule 各有独立 metadata 与纯 C++ shared_ptr owner；
  未消费 capsule 会回收，已消费 capsule 由 consumer 回收，拒绝重复消费。
- Tensor/切片可比 Python storage/view 存活更久；deleter 不依赖 Python/GIL，验证了
  另一线程释放的情形。最终销毁在所属 device 同步所属 stream 并恢复线程原设备。
- 跨设备复制、显式 copy、不匹配 stream 的导出均失败。导出后仍要求调用者遵守
  单 stream 约定，不宣称普通 Tensor 可以自动阻止跨 stream 使用。
- 所有 block ID 和请求租约仍归既有 BlockManager 管理。Tensor 存活只保证底层
  allocation 不被释放，不保证对应逻辑 block 在请求结束后不会被复用。

通用 `PagedCacheStorage` 新增显式 `release()`：各 component 独立释放，成功的
pointer 立即置空；失败可报告、重试，重复释放不 double-free，访问已释放数据报错。
析构捕获释放错误并记录，不让异常直接终止进程。V4 owner 若遇到 CUDA context/stream
失效，保留错误计数和未回收 bytes，不把资源泄漏隐瞒为正常回收。

Python `PagedCachePool` 默认选择 `storage_backend="cpp"`；缺失 native capability
时明确失败。`storage_backend="torch"` 保留原实现作显式对照。三个 cache 算子契约
不变，仍可逐项替换为用户算子库。Python 还在调用模型算子；本次只完成 paged
payload 的 C++ 所有权，不等于完整 C++ model runner、临时 tensor 或权重迁移。

### 构建与单元验证

`deepseek_v4_b300_native_storage_build.json`，Slurm job 23009：

- 从实际 B300 capability 生成 sm_103，编译现有 shared runtime 和 pybind 模块。
- 增加 `--dlpack-include=/home/lcpu/57682329/.local/deepseek-v4-tilelang-compat/tvm_ffi/include`；
  CUDA include/link 目录由 xmake 的 toolkit 探测得到，不新增写死的 CUDA 架构。
- 构建后安装项目 runtime 到本仓库的 Python package，避免新模块加载旧 runtime ABI。
- `_C.so` SHA256：`75c6ee368a6a5767487b625f57b7064f57c6f9b363adac3fef7de94a667b75e1`。
- `libllaisys.so` SHA256：`43302daa19357dfb8eff84c15c1557d4e1dd33d112a8810903015dc1de6f44a2`。
- 既有 SchedulePlan roundtrip 和 native CPU BF16 DLPack import smoke 通过。
- 新增 `llaisys-cache-storage-test` 纯 C++ 测试通过：布局/stride、越界、尺寸溢出、
  部分分配失败回滚、释放失败后继续释放其他 component、重试与重复释放、已释放
  pointer 的访问拒绝。不执行模型推理。

第一次构建本身成功，但构建检查进程的 PYTHONPATH 未包含已有 safetensors 包，
import smoke 失败；补齐已验证依赖路径后重新运行成功。不是 CUDA/kernel 或数值问题。

`deepseek_v4_b300_native_storage_regression.json`：**140 个方法，139 通过、1 失败，
0 error/skip**，最终 Slurm job 23013，源码前后摘要一致。新增八个 native storage
测试方法全部通过，覆盖 CPU/GPU 零拷贝与
pointer/stride、legacy/versioned capsule 生命周期、越界/非法布局、显式无 fallback、
非默认 CUDA stream、跨线程释放，以及真实小配置 full/chunk/decode 对 Torch storage
逐值一致。还验证了原生 allocation 不计入 Torch allocator 的 memory_allocated。
唯一失败仍是原始默认非对齐 chunk/full 方法；没有放宽精度门槛。

显存统计现在明确区分 Torch allocator peak 与 `native_cache_allocation_bytes`。
后续正式进程显存峰值测试必须覆盖外部分配，不能继续仅用 Torch peak 表示全部显存。

纯 C++ storage lifecycle 测试还使用 `g++ -fsanitize=address,undefined
-fno-omit-frame-pointer` 编译运行，退出 0，无 sanitizer 报错。这只覆盖该 C++
布局/分配/释放测试，不把它扩展成全部 Python/CUDA 算子的 sanitizer 验收。

### 原始数值策略的真实完整权重对照

`deepseek_v4_b300_native_storage_top512.json`：单张 B300、完整 43 层，严格加载
67569 个主模型 tensor，2105-token 输入、16-token 输出，batch/concurrency=1，
max-seq-len=4096，原始 TileLang/PyTorch 数值策略，paged block size=128、32 blocks，
CUDA Graph/Prefix Cache 关闭，无 fallback。权重格式仍为 W8A8 dense、W4A8 experts，
BF16 hidden/cache；完整 checkpoint 与原始 golden 的 SHA256 校验通过。

- C++ payload 对原始上游 full golden：16 步 logits 全部逐值一致。
- C++ payload 对同 schedule Torch paged storage：16 步全部逐值一致。
- C++ payload 对连续缓存：16 步全部逐值一致。
- `all_passed=true`，进程退出 0；请求释放后 blocks 为 32/32 free。
- 真实 pool 创建前原生计数为 0 bytes/0 layers；执行期间为 208535552 bytes/43
  layers；销毁 pool 后回到 0/0，destruction_errors 始终为 0。没有只验证 Python
  对象被删除，就假定原生显存已回收。

核心新增命令参数为 `--cache-layout paged --paged-storage cpp
--compare-paged-storage-in-process --compare-contiguous-in-process`，其余模型、权重、
原始 baseline、Slurm 和隔离依赖参数见 JSON。每条路径 warmup 一次 prefill+decode，
再测一次生成；这是正确性对照，不是正式 TTFT/TPOT 或统计性能结果。

### 固定策略的真实长 chunk 与释放对照

`deepseek_v4_b300_native_storage_chunk257.json`：同一真实模型与输入输出规模，
使用独立 `fixed32-reference-v1` golden，显式选择 `torch-fixed32`、
`index_ascending`、固定 attention metadata；257×8+49 分块，中间八块不生成 logits。

- C++ payload 分块对独立固定策略 full golden：16 步 logits 逐值一致。
- 对同策略独立 full：16 步逐值一致。
- 对同 schedule Torch paged storage：16 步逐值一致。
- `all_passed=true`，进程退出 0；请求归还全部 32 blocks，pool 销毁后原生计数从
  208535552 bytes/43 layers 回到 0/0，destruction_errors=0。
- 原始上游数值策略没有在此分块运行中重新验收；报告明确
  `unmodified_published_baseline_tested=false`，原始 chunk/full 失败仍保留。

两条真实权重路径使用同一批已编译 C++ storage/pybind 实现与源文件摘要。运行中
没有编辑相应源码或覆盖已加载的共享库。新增 JSON 是小型验收报告；权重、golden
safetensors、二进制和日志仍受 Git ignore 保护，没有提交或改写 Git 历史。

本轮达成的是 **paged payload 的真实 C++ 所有权与可替换算子视图接口**，而不是
仅新增未使用的 layout 类；它已经进入完整模型的 attention/cache 执行。模型权重、
临时激活、层执行和部分 metadata 仍由 Python/Torch 控制，下一步还须完成真正的
C++ model/batch runner、压缩器状态共享/Prefix Cache/COW、窗口回收，以及 chat
编码/任务精度和已有 scheduler 的接入。

## 官方对话编码与独立自由生成（2026-09-08）

### 修复的验收缺口

此前的完整权重样例主要是 raw completion：直接 `tokenizer.encode(prompt)`。
实际模型的 `tokenizer_config.json` 没有 chat_template，随模型发布的 `generate.py`
会先调用 `encoding/encoding_dsv4.py::encode_messages`。因此，原有文本续写正确性
不能替代真正的对话输入/多轮上下文验证。

新增 `deepseek_v4_model/chat.py`，读取明确指定且可信的本地模型编码源码，执行与
记录同一份字节的 SHA256，不修改共享模型目录。BOS/EOS 必须与 tokenizer 的独立
token 编码一致；已含 BOS 的 prompt 显式关闭 tokenizer 的额外特殊 token 添加。
工具调用仅编码和解析，不会执行任何函数或网络请求。

新增 `deepseek_v4_model/generation.py`，在独立模型上实现实际 token 反馈的 greedy
生成，保留可注入的多算子后端与 C++ paged pool。请求状态在正常、EOS、长度上限、
取消、执行异常及输出回调异常时均释放。此处是单请求生成驱动，不是新 scheduler，
也不宣称 C++ batch runtime 或已有 serving API 已完成。

EOS 必须真正由模型生成。达到 max_new_tokens 或上下文容量上限不补伪造的 EOS；
返回 `finish_reason=length` 与原始截断文本，`complete=false`。有 EOS 但思考/DSML
结构错误时明确记录 parse_error，不修补输出再宣称格式正确。

### 测试与可复现入口

- CPU 定向测试 25 项全部通过。
- Slurm 23016 完整 V4 回归：156 方法、155 通过，0 error/skip，源码前后摘要一致。
- 唯一失败仍是原有非对齐 chunk/full 方法的四个子项，relative L2 分别为
  0.0276797079、0.0473939367、0.0286046304、0.0233045444；1% 门槛保持不变。
- 新增测试覆盖精确角色标记、单 BOS、chat/thinking/三种 effort、多轮历史、工具
  结果排序、不修改输入、tokenizer/编码身份不匹配、EOS/截断/错误格式，以及取消、
  异常释放、真实小模型生成对照与 C++ block 全量归还。
- 基线消费者重新编码消息，核对编码源码/tokenizer、渲染 prompt、实际 token IDs、
  完成状态和完整 checkpoint/golden 摘要；不接受只有相同文件名的伪基线。

新增 `run_chat.py` 不要求任何 golden 文件，可直接运行所选本地对话 JSON。GPU 通过
Slurm 分配；沿用本文前述隔离依赖 PYTHONPATH，命令主体如下：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=12 --mem=80G --time=00:20:00 \
  python3 tools/deepseek_v4_reference/run_chat.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --cases-file test/data/deepseek_v4_chat_cases.json \
  --max-new-tokens 256 --max-seq-len 4096 \
  --output benchmark_results/deepseek_v4_b300_chat_standalone.json
```

默认使用 TileLang 核心算子、Torch 结构算子、CUDA Hadamard 和 C++ paged storage，
不启用 Prefix Cache、CUDA Graph 或 MTP，不发生静默 fallback。单次未预热的总
wall time 仅用于运行观察，不作为正式 TTFT/TPOT 或吞吐 benchmark。

真实模型用例固定在 `test/data/deepseek_v4_chat_cases.json`：中文计算、多轮变量
更新、英文抽取、排序、代码解释、工具结果、180 条独立记录的长账本检索，以及
thinking 模式。输入长度分别为 19、41、48、28、32、62、2011、21 token，输出
上限 256。这是 8 个确定性功能样例，不代表公开基准或统计泛化精度。

### 原始官方基线的实际结果

Slurm 23017，`deepseek_v4_b300_chat_published.json`：原始模型控制流、原始数值策略、
完整 43 层 MP1 权重，checkpoint 全量 SHA256 仍为
`024836b412ccd5c539bb130842842247d9c427f99c3a1e21ed35ea08ddcc0aa2`。
full-only 模式不执行逐 token replay，相关精度字段保持 null，而不是填通过。

| 用例 | 输入 token | 实际输出 token（含真实 EOS） | 结束方式 | 预设检查 |
|---|---:|---:|---|---|
| 中文计算 | 19 | 2 | EOS | 391，通过 |
| 多轮变量更新 | 41 | 54 | EOS | 算出 57，但附加解释，不符合只输出整数，失败 |
| 英文抽取 | 48 | 3 | EOS | teal，通过 |
| 排序 | 28 | 14 | EOS | 输出字符串数组而非预设数值数组，失败 |
| 代码与解释 | 32 | 256 | 长度上限，无 EOS | 代码正文含正确 `% 2`，但回答未结束，严格检查失败 |
| 工具结果 | 62 | 2 | EOS | 24，通过；没有执行工具 |
| 长账本检索 | 2011 | 2 | EOS | 86，通过 |
| thinking 算术 | 21 | 19 | EOS | 正确解析思考结束标记与最终答案 15，通过 |

预设功能检查为 **5/8**；不能把额外解释、字符串数组或截断输出事后重新解释成
预设检查通过。参考进程退出 2，报告 `all_expected_match=false`。这是上游基线
本身的任务/格式表现，不是独立执行器的数值差异证据。后续对照须分别报告模型
任务表现、相同历史的 logits 精度、自由生成 token 一致性，保留失败项。

### 无 golden 依赖的独立聊天入口

Slurm 23018，`deepseek_v4_b300_chat_standalone.json`：`run_chat.py` 独立加载完整真实
主模型，未导入上游模型，也未读取参考报告/golden 文件。8 例共 **352 个实际生成
token** 与事后对照的原始官方基线完全一致，8 个 completion 结构也完全一致。
这里的逐 token 相同来自实际自由生成，而不是固定参考历史下的 argmax 列表。

- 后端：六类 TileLang、十三类 Torch、上游 CUDA Hadamard；C++ paged storage，
  BT=128、32 blocks；full prefill，max_seq_len=4096，max_new_tokens=256。
- 每例新建请求，不共享前缀；没有 Prefix Cache、CUDA Graph、MTP 或 fallback。
- 结束后 32/32 blocks 全部空闲；pool 销毁前后计数均为 0 bytes、0 layers、
  destruction_errors=0；源码前后摘要一致。
- 预设检查仍为 5/8。两项格式不符及一个长度截断均与官方一致，未放宽预期；
  报告 `all_passed=false`、`all_conversations_completed=false`，进程退出 2。
- 能证明独立模型的对话生成与结束状态适配；不能把它描述为模型任务精度全部
  通过、正式 serving 完成或 C++ model execution 迁移完成。

### 独立模型逐步 logits 与自由生成双重对照

Slurm 23019，`deepseek_v4_b300_chat_independent.json`：严格加载 67,569 个主模型
tensor，完整 checkpoint SHA256 与每份 golden 文件 SHA256 验证通过。源码及已
加载二进制摘要在执行结束后另行复核一致，三个真实模型进程均已结束。

- 8 例共 **352 步 logits 全部逐值一致**：max_abs_error=0、relative L2=0，
  各例 `numerical_gate_passed=true`、`all_logits_exact=true`。
- `--free-generation` 另起新请求，反馈自己的预测 token；8 例生成 IDs 全部等于
  原始 golden 的实际生成 IDs，不使用 teacher forcing 伪装自由生成。
- 同一 C++ pool 被多次请求复用后，32 blocks 全部空闲。原生分配计数从
  208535552 bytes/43 layers 回到 0/0，destruction_errors=0。
- 模型任务检查仍为 5/8；报告保留 `all_passed=false`、退出码 2。该退出码来自
  与上游相同的任务/格式/截断结果，不是本轮 logits 门槛失败，也不是执行崩溃。
- 默认 full prefill、原始数值策略，**没有**在本轮重新声明 chunk/full 通过；
  156 项回归中的原有失败项仍保留。

复现对照命令主体（Slurm 与 PYTHONPATH 配置同上）：

```bash
python3 tools/deepseek_v4_reference/run_independent.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --baseline benchmark_results/deepseek_v4_b300_chat_published.json \
    benchmark_results/deepseek_v4_chat_golden \
  --max-seq-len 4096 --cache-layout paged --paged-storage cpp \
  --free-generation --output benchmark_results/deepseek_v4_b300_chat_independent.json
```

本轮新增能力是 **真实对话格式 → 独立权重加载 → 单请求生成 → EOS/截断/解析 →
资源释放**，以及与原始参考分离的数值/生成/任务三类验收证据。没有改写共享权重
或官方源码，没有提交代码，174 MiB golden 与日志仍被 Git 忽略。

完整基础适配仍未结束。下一步继续 V4 压缩器状态与 Prefix Cache/COW、窗口回收、
C++ 模型批执行和既有 scheduler 接入。已确认服务接点是
`python/server/engine.py` 的 `create_batch_context`、`prefill_chunk` 与
`decode_per_request`，应复用其队列/调度策略，不另建 serving scheduler。还需处理
服务线程的 CUDA stream 所有权、V4 EOS=1、取消、抢占与压缩状态恢复；当前独立
生成驱动的取消测试不替代这些服务级验证。TP/EP、MTP 和正式性能验收没有在本轮
实现或补写为完成。

## V4 Prefix Cache 与 partial-block COW（2026-09-08）

### 实现与所有权

新增模型层 `deepseek_v4_model/prefix.py`，复用既有 C++ BlockPrefixCache 的哈希链、
BlockManager 的 refcount/LRU 和 C++ PagedCacheStorage 的 payload。没有恢复 Trie，
也没有给 block manager 添加 Qwen/V4 tensor shape 依赖。

此前只共享 latent blocks 不足以恢复 V4 推理：ratio-4 层在下一组压缩时还会读取
上一组投影状态。新实现从真实前向中捕获每个完整 block 边界的最小 FP32 状态：
前四个投影 token 的第一半及对应 score；Attention 与 Indexer compressor 分别保存。
ratio-128 在对齐 block 边界没有未完成组，旧 scratch 无需保存。

支持 full prefill、非对齐 chunk 和单 token 补齐边界。不能拿长请求最终的
compressor state 代替较短前缀的状态。实际 token IDs 在设备记录，publish 时统一
转到主机；发布仅接受已完整计算且状态齐全的 block。命中后共享 payload、复制
恢复所需的 compressor scratch，并跳过对应 token；总留下至少一个 prompt token
计算最终输出 head。每个 pool 固定权重 generation、配置/数值策略、device 和 stream。

`request.fork()` 共享 block lease、独立复制 compressor scratch。继续写入共享
partial block 时，先复制各层 full/compressed latent 与 Indexer 组件，再提交新
block table；父请求内容不被覆盖。新增第四类 cache 后端接口 `cache_copy_block`，
默认 Torch `copy_`，后续算子库可独立替换。19 类模型算子接口不变。

C++ 绑定新增 lease `prefix/replace` 和 block `uncache`，仍只是资源所有权操作。
原子校验后转移/释放对应引用，不把调度策略放入 C++。活跃引用保护 shared blocks，
LRU 只淘汰无活跃引用的缓存；物理 ID 复用时丢弃旧模型状态 record。不得绕过模型
pool 任意重绑定底层 index 或写入共享原始 Tensor。

默认 `enable_prefix_cache=False`。生成 API 使用 `reuse_prefix=True`，独立聊天
入口使用 `--prefix-cache`；原始数值策略没有自动被替换成 fixed32。

### 构建与结构/小配置验证

- Slurm 23022：按实际 sm_103 增量重建 pybind，原生 storage lifecycle 测试通过。
  新 `_C.so` SHA256 为
  `40861b588608f0206a5fb969744931b198712237204ac610b8c32916ada8e6ee`；
  runtime library 未变，SHA256 为
  `43302daa19357dfb8eff84c15c1557d4e1dd33d112a8810903015dc1de6f44a2`。
- 两组 CPU 定向测试分别为 34 项、33 项通过，包含既有 paged/对话/基线身份回归。
- Slurm 23023 完整 GPU 回归：170 方法、169 通过，0 error/skip，源码前后摘要一致。
  唯一失败仍是原始非对齐 chunk/full 方法的四个子项，没有删除或放宽 1% 门槛。
- 新增 12 项 Prefix/COW 方法和 2 项原生 lease 方法。CPU/CUDA 小配置覆盖全量及
  短前缀命中、每个边界快照、127+1 等非对齐分块、真实跳过 prefill、交错请求、
  partial-block fork/COW、显存不足/复制失败、LRU 过期状态、错误发布、取消与回收。
- 缓存本身的数值隔离测试明确使用 fixed32 + stable Top-K + fixed metadata 小配置，
  不将这些通过项称为原始数值策略已经通过。

### 完整 43 层固定策略实测

Slurm 23024，`deepseek_v4_b300_prefix_fixed_top512.json`：B300 SXM6 AC ×1，CC 10.3，
CUDA 13.0、PyTorch 2.13.0+cu130、TileLang 0.1.8、TVM-FFI 0.1.8.post2，CUDA Hadamard。
67569 个真实主模型 tensor、dense W8A8 / experts W4A8、BF16 latent，完整 payload
SHA256 与独立 fixed32-reference-v1 golden 摘要均验证通过。

- 输入 2105 token，输出 16 token，max_seq_len=4096，batch/concurrency=1。
- BT=128、32 blocks，先真实执行冷 full prefill 并发布完整 blocks/边界状态。
- 命中 2048 token，只执行剩余 57 token 的 prefill，之后正常 decode。
- 冷请求与独立固定策略 golden 的 16 步 logits 逐值一致。
- 命中请求与冷请求、独立固定策略 golden 的 16 步 logits 均逐值一致。
- 另起请求自由生成（不喂 golden token），16 个生成 IDs 与参考相同。
- 16 个 record 的恢复状态共 6881280 bytes；C++ latent pool 为 208535552 bytes。
  pool 汇总的 3 次命中/6144 token 包括 warmup、正式对照和自由生成，不是单次命中。
- 结束后 32/32 blocks 均无活跃引用；pool 销毁后原生计数归零，destruction_errors=0。
- `all_passed=true`，退出 0；`unmodified_published_baseline_tested=false`，严格标记
  独立固定数值策略，不声称原始策略或统计任务精度通过。

没有 CUDA Graph、MTP 或 fallback。warmup 每条路径一次 prefill+一次 decode，
计时重复一次；lookup/publish 不在 step latency 内。数据用于正确性验收，不能
直接作为正式 Prefix Cache TTFT 加速比或 serving benchmark。

### 原始策略：命中相对 full golden 未通过

Slurm 23025，`deepseek_v4_b300_prefix_published_top512.json`：输入/输出、权重、
物理 pool 均同上，但使用原始 `torch + published + published` 数值策略。

- 冷 full 请求的 16 步 logits 与原始 golden 仍逐值一致，说明启用边界捕获没有
  改变这条冷执行路径。
- 实际命中 2048 token、执行尾部 57 token 后，相对 full golden 的 relative L2
  为 4.2929%–94.9614%，最低 cosine 为 0.7042103，明确未过原始精度门槛。
- 16 步 argmax 和另起请求的自由生成 IDs 均相同，但本例是高置信重复文本，不能
  因 token 相同就声称数值或统计任务精度通过。
- 生命周期仍通过：32 blocks 无活跃引用，pool 销毁后原生计数归零。
- `all_passed=false`、退出 2；没有把原始策略自动切到 fixed32，也没有放宽门槛。

该结果本身还不能唯一归因于缓存恢复或既有形状相关数值变化，需另用相同前缀
分块方式对照 seed 与命中请求。默认 Prefix Cache 保持关闭，原始策略不标为已
完成数值验收。

### 原始策略的同分块隔离对照

补充产物 `deepseek_v4_b300_prefix_published_boundary2048.json`：冷请求明确分为
2048+57，命中请求复用前 2048、执行同样的尾部 57。两条路径后续 16 步 logits
逐值一致，`prefix_reuse_comparison.passed=true`，没有观察到恢复引入的额外误差。
但两者相对原始 full golden 的 relative L2 仍为 4.9826%–41.9813%，总标志为 false。

此例把差异定位到改变 prefill 执行形状后的数值路径，而不是这次前缀恢复本身。
这只是该输入/配置的隔离证据，不证明所有前缀或 chunk 都正确，也不消除既有精度
门槛。32 blocks 归还、native live bytes/layers 归零，未启用 fallback。

## 既有 serving 调度器接入 V4 参考后端（2026-09-08）

本轮推进运行时接口与生命周期，不将任务缩减为几项算子。新增
`deepseek_v4_model/batch.py`、`snapshot.py` 和 `run_serving.py`；修改既有
`python/server/engine.py` 的异常、抢占与线程退出处理，不新建 serving scheduler。

### 接口、执行与所有权

- `DeepSeekV4ServingModel` 包装已经严格加载的独立模型，EOS 必须显式传入，V4 使用 1。
- `create_batch_context` 在 worker 内创建专属 CUDA stream 与 C++ paged pool；等待
  权重生产 stream 的 event。每个调用显式进入所属 stream，跨 worker 访问拒绝。
- 对接现有 `prefill/prefill_chunk/decode_per_request/slot_reset/slot_save/slot_restore`
  与 `prefix_lookup/prefix_publish`。slot 独立持有请求状态，复用 19+4 类算子契约。
- 当前按 slot 串行执行，不宣称 fused/mixed GPU batch；仍由 Python 模型发起算子
  调用，C++ model execution、权重和临时 tensor 所有权迁移仍未完成。
- 完整 prefill 默认可用；chunk/Prefix 必须显式 opt-in，服务默认分块不会悄悄替换
  原始数值策略。中间 chunk 不生成 logits，最终 chunk 才采样。
- 支持 greedy 及显式 temperature/top-k/top-p 采样，每 slot 独立 RNG；采样不等于
  与官方随机序列逐值一致。整模型精度验收继续固定 greedy 和原始量化语义。
- 抢占快照为进程内 CPU tensor，保存逻辑 latent/Indexer、压缩器 scratch、前缀
  边界状态、pending token 与 RNG state；不持有 GPU block lease。恢复重新分配
  physical blocks，经已有 cache read/write 契约搬运，不把 shape 写进 block manager。
- 先创建完整快照再释放原 slot；保存失败不驱逐、不从 prompt 重新开始生成。
  恢复失败归还新 blocks；原快照仍可重试。跨模型 generation/不匹配容量拒绝。

### 调度器缺口修复

保留原来的 admission、chunk 策略与最长输出请求优先抢占规则，补齐以下边界：

- prefill 错误先 reset slot；decode 返回 token 数不匹配则显式失败。
- 完全超出 pool 的请求显式拒绝，不无限重排队；admission 预留运行请求后续增长。
- 抢占后允许 decode 取得进展，避免两个请求在同一次 admission 中无限互换。
- 等待中取消也终结 Future/stream；活跃取消在 chunk/token 边界处理，取消后不继续
  输出 token；Future 的 done 检查放在 asyncio 线程，避免取消竞态。
- 初始化失败、停止与运行异常均清理队列和 slots；stop 超时保留真实 live worker
  handle，不把仍在执行的线程标为退出。关闭后的请求不再持有整个 native pool。
- 新 API 只是既有 InferenceEngine 的模型适配与测试入口，HTTP app 尚未切换到 V4。

### 结构与小配置数值验证

Slurm 23027，`deepseek_v4_b300_serving_regression.json`：193 方法、192 通过，
1 个原始 chunk/full 方法失败（4 个子项），0 error/skip，执行期间源码摘要一致。
新增 16 项 batch/serving 测试通过；另包含既有 7 项无模型 scheduler 测试，不运行 Qwen。

CPU/CUDA 小配置实际覆盖独立生成对照、多个串行 slots、worker 非默认 stream、
主机快照跨物理 block 重映射、压缩器恢复与采样 RNG 一致、pool 释放、请求/线程
所有权、显式 chunk 与 Prefix、异常后继续服务、取消、抢占压力下取得进展。
初始化失败、stop 超时、Future 取消、保存快照失败等生命周期场景另有 CPU 测试。
测试中注入的错误日志是预期断言，不是被隐藏的 fallback。

本轮源码已有新增与修改，旧报告只代表当时记录的 SHA256；不能把旧产物自动当作
当前所有文件已重新验收。新增报告记录此次执行实际文件、命令、dirty HEAD 和版本。

### 完整权重：真实抢占/恢复验收

Slurm 23028，`deepseek_v4_b300_serving_preemption.json`，退出 0、all_passed=true：

- B300 SXM6 AC ×1、CC 10.3；CUDA 13.0 / PyTorch 2.13.0+cu130 / TileLang 0.1.8 /
  TVM-FFI 0.1.8.post2，上游 CUDA Hadamard。原始 `torch + published + published` 策略。
- 完整 43 层与 67,569 主模型 tensors，dense W8A8、experts W4A8、BF16 latent。
  权重完整 payload SHA256 为
  `024836b412ccd5c539bb130842842247d9c427f99c3a1e21ed35ea08ddcc0aa2`；golden 身份也校验。
- 两个相同的 2105-token prompt 请求，每个自由生成 16 token；max_seq_len=4096，
  concurrency=2，物理执行 batch=1，BT=128、pool=32 blocks、watermark=0。
- 32 blocks 不足以同时容纳两个请求，实际触发 **29 次保存、29 次恢复**。模型只
  prefill 两次、decode 30 次，没有通过重复 prompt 计算伪装恢复，也没有喂 golden token。
- 两个请求生成及 stream 输出均与原始参考相同；**32 步 logits 全部逐值一致**。
- 32/32 blocks 归还，native live bytes/layers 均从 0 返回 0，destruction_errors=0；
  等待/活跃队列归零，执行期间源码 SHA256 未变。
- 无 chunk、Prefix、CUDA Graph、MTP 或 fallback，未修改量化/Top-K/归约策略。

这是缓存压力的正确性测试，29 次换出不是吞吐优化成果。该完整权重用例使用相同
prompt 的两个请求；不同 prompt 的交错隔离已在 CPU/CUDA 小配置覆盖，后续还需真实
多样请求验收。计时包含 logits 的 GPU→CPU 对照、未预热、只执行一次，因此不报告
正式 TTFT/TPOT 或抢占性能收益。报告中的 unwarmed wall time 仅供排查运行流程。

### 完整权重：两个 slot 同时驻留对照

Slurm 23029，`deepseek_v4_b300_serving_top512.json`，退出 0、all_passed=true。
权重、原始数值策略、2105/16 输入输出、concurrency=2 等与压力测试完全相同，
只把 pool 扩到 64 blocks（417071104 bytes），实际 save/restore/preemption 均为 0。
两个 slot 各自保留状态，按已有 scheduler 的 decode 批接口交错执行，底层仍逐 slot
串行执行模型。32 步 logits、自由生成 IDs 和 stream IDs 全部与原始 golden 一致。
64/64 blocks 归还，native live bytes/layers 和等待/活跃队列归零。

两组完整权重测试合计 64 个输出步，均为固定输入的正确性对照，不是统计任务精度。
未运行 Qwen、未重新编译本轮未改动的 C++ 二进制、未启用 fallback。最终核对三份
新报告（193 项回归、32-block 抢占、64-block 同驻留）记录的全部源码/二进制 SHA256
与当前执行文件一致；`git diff --check` 通过。工程记录和使用说明保持中文。

完整权重复现入口（需沿用记录的隔离 PYTHONPATH；CUDA 仍只能在 Slurm 内执行）：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=12 --mem=80G --time=00:20:00 \
  python3 tools/deepseek_v4_reference/run_serving.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --baseline benchmark_results/deepseek_v4_b300_tilelang_top512_accuracy.json \
             benchmark_results/deepseek_v4_tilelang_top512_golden \
  --max-seq-len 4096 --concurrency 2 --num-blocks 64 \
  --output benchmark_results/deepseek_v4_b300_serving_top512.json
```

压力对照将 `--num-blocks` 改为 32、输出文件改为 `serving_preemption.json` 对应路径。
完整命令及真实输出路径以每个 JSON 的 provenance 为准。工具支持选择既有对话
golden，用于后续多样输入的服务级对照；当前还不能将不同 case 同时混入同一组请求。

下一阶段仍须完成 C++ model execution 与权重/临时 tensor 所有权迁移、HTTP 入口、
更多不同真实 prompt 及异常后显存释放测试、原始 chunk/Prefix 数值门槛和正式性能
矩阵。TP/EP 受当前单卡配额限制，不能用这两组单 GPU 多请求结果替代多卡验收。

## 原生 TileLang 调用与实际权重量化 Linear（2026-09-08）

本轮沿既有 Runtime/Storage 推进模型执行迁移，未重新实现 scheduler、block manager
或 Prefix Cache，也未修改共享模型 `kernel.py`。新增：

1. 后端无关的 C++ `Tensor/Kernel` 接口：实际低精度布局、显式 backend/version/
   operation/revision、错误传播与调用统计，供后续自定义算子库继承和逐项替换。
2. TileLang 原生 adapter：离线导出 kernel 动态库，C++ 直接载入并调用；模型执行侧
   不需要 Python 解释器、Torch、pybind 或逐算子 callback。
3. C++ `QuantizedLinear`：保持原始 W8A8/W4A8 权重和 UE8M0 scales，持有权重所有权，
   使用调用方可复用的 workspace 连续执行激活量化与 GEMM。两个 kernel 之间不做
   host roundtrip、显式同步或临时显存分配。
4. 可选 xmake 静态库 `llaisys-tilelang-native` 及独立原生进程验证器。

### 本轮实际遇到并修正的问题

- PyTorch 不支持对 packed FP4 直接 `zeros`：测试初始化改为 uint8 零字节视图，
  不改变数据布局与量化语义。
- TileLang 命中磁盘缓存时返回已导出的 `runtime.Module`，再次 `export_library`
  会失败；对这一路径复制已有库并核对 SHA256，未重新编译或更换 kernel。
- 主动态库未导出内部 Runtime/Storage 符号：链接现有 core/device 静态库；没有为
  测试扩大公开 C ABI 或另写一套分配器。
- 参数错误 helper 注册位于 `libtilelang`，仅链接 TVM/FFI 不够；原生链接明确保留
  `libtilelang` 的初始化，并显式选择兼容 TVM-FFI 的动态库路径。
- `AnyView` 借用 `TensorView` 内部描述符：临时视图会悬空，现保留整组视图到调用结束。
- TileLang 0.1.8 的 FP4 ArgBinder 接收 packed byte shape；adapter 将逻辑 DLPack
  描述转换成调用专用 byte 描述，数据不搬运、不解量化。此行为限制在已验证版本。

这些早期失败分别发生于测试准备、导出、链接和参数绑定阶段，不是模型算子数值
比较失败。验证器现在也会在准备/编译异常时写入失败报告，避免输出路径残留旧绿色结果。

### 实测结果

最终报告：`benchmark_results/deepseek_v4_b300_native_tilelang.json`，Slurm **23041**。
它实际链接 xmake 生成的静态库，执行期间报告涉及的源码/库文件保持不变。

| 项目 | 实际范围与结果 |
|---|---|
| 硬件 | B300 SXM6 AC ×1，CC 10.3；driver 580.126.09 |
| 软件 | CUDA 13.0；参考准备进程 PyTorch 2.13.0+cu130；TileLang 0.1.8；TVM-FFI 0.1.8.post2；G++ 13.3.0 |
| 核心 kernel | 六类算子、23 个用例；量化/QDQ、W8A8/W4A8 GEMM、mHC、稀疏 attention；覆盖 M=1/33/137，attention S=1/3、候选数 64/576 及 -1 mask |
| 真实权重 | `layers.0.attn.wq_a`：FP8 `[1024,4096]`；`layers.0.ffn.experts.0.w1`：FP4 `[2048,4096]`；`w2`：FP4 `[4096,2048]`，均保持实际 scale payload |
| 连续 Linear | 上述三组权重 × M=1/33/137，共 9 用例；每例复用 workspace 连续执行两遍 |
| 数值 | 184 项 tensor 字节对照全部一致，包含量化激活、scales、BF16 输出及只读输入/权重未被改写 |
| 错误与资源 | 107 项参数/线程/close/契约/自定义后端失败检查通过；权重 owner 随 layer 释放；kernel teardown error=0 |
| 原生依赖 | 子进程执行前后均检查无 Python/Torch runtime；调用路径无 fallback |
| 性能范围 | 正确性测试，不是 TTFT/TPOT、吞吐或峰值显存 benchmark；没有 CUDA Graph、请求并发或多卡测试 |

MP1 文件大小 168,430,204,680 字节。本轮只读取所列三个 Linear 的权重及 scale，
记录 checkpoint header SHA256、每个选中 payload SHA256、输入/输出文件摘要、
kernel 动态库/可执行文件/静态库摘要及 `ldd`；没有加载完整模型，也没有重新扫描
整份 168 GB payload。报告明确 `whole_model_loaded=false`、`whole_payload_hashed=false`。

当前原生静态库 SHA256：
`2967d1988bd24594db8c77c6c39cb234fbba4a824605828a887fa7254d221627`。
既有 `_C.so` 和 `libllaisys.so` 未安装覆盖，摘要与上一轮 serving 验收一致。
未重复 Qwen 推理，也未把上一轮 193 方法的回归结果当成本轮新执行的结果。

### 本机复现入口

以下命令均在 Slurm GPU allocation 内执行。路径取自本机实际安装位置，其他服务器
须重新探测，不能沿用旧 Python include 或 CUDA arch。xmake 配置时保留原有参数：

```bash
/home/lcpu/62385178/.local/bin/xmake f -y \
  --nv-gpu=y --cuda-arch=sm_103 --flashinfer=y --python-bindings=y \
  --python-include=/home/lcpu/57682329/.local/deepseek-v4-python-dev/usr/include/python3.12 \
  --pybind11-include=/home/lcpu/57682329/.local/deepseek-v4-reference/torch/include \
  --dlpack-include=/home/lcpu/57682329/.local/deepseek-v4-tilelang-compat/tvm_ffi/include \
  --tilelang-native=y \
  --tilelang-root=/home/lcpu/57682329/.local/deepseek-v4-reference/tilelang \
  --tvm-ffi-root=/home/lcpu/57682329/.local/deepseek-v4-tilelang-compat/tvm_ffi
/home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-tilelang-native

# 使用既有隔离参考依赖的 PYTHONPATH，兼容 FFI 排在旧参考环境之前。
python3 tools/deepseek_v4_reference/verify_native_tilelang.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --native-library build/linux/x86_64/release/libllaisys-tilelang-native.a \
  --output benchmark_results/deepseek_v4_b300_native_tilelang.json
```

Slurm 参数为 `--partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00`。
只将小型 JSON 报告加入允许列表，JIT 库、权重测试片段、原生可执行文件和日志均继续
被 Git 忽略。本轮没有提交代码或改写已有提交。

### 尚未完成的原生边界

已验证的 C++ Linear 尚未替换 Python 模型执行路径。下一步必须完成其余结构算子的
C++ 后端、Hadamard、完整单层/中间 tensor 对照，再迁移 43 层模型与 batch 接口，
并复用既有 paged cache 和 scheduler。不能把六类 kernel 的原生调用或三组真实
Linear 权重的正确性，写成“完整 C++ DeepSeek 推理已经完成”。
原始 chunk/Prefix 精度门槛、HTTP 接入、正式性能矩阵和真实 TP/EP 验收仍未完成。

## 原生结构后端与完整 C++ MoE 层（2026-09-08）

### 本次范围与结果

本次沿用 TileLang 为默认量化/稀疏算子基线，补齐原生模型组合所需的结构后端，
并将路由、dispatch、routed/shared expert 和 combine 串成完整 C++ MoE 层。
未改写 Python scheduler、block manager、Prefix Cache、Qwen 模型或共享上游代码。

最终联合验收为 **Slurm 23054**，作业已正常退出。全部 GPU 编译/测试经 Slurm
`gpu` 分区、单 GPU 分配；测试命令使用 8 CPU、48 GiB host memory、25 分钟时限。
实际环境：NVIDIA B300 SXM6 AC、CC 10.3（sm_103）、driver 580.126.09、CUDA 13.0、
G++ 13.3.0、Python 3.12.3、Torch 2.13.0+cu130、TileLang 0.1.8、TVM-FFI 0.1.8.post2。
ATen 与 native 测试的 TF32/cuBLAS 开关均为 false，未运行 NCCL collective。

| 验证范围 | 结果 | 不能推导出的结论 |
|---|---|---|
| 13 类 ATen 结构算子、3 类组合工具、原生 CUDA Hadamard | 129 用例（含 10 组应拒绝输入）、374 次 tensor 字节对照、391 项异常/线程/生命周期检查通过 | 不是全部 fused kernel，也不是完整模型执行 |
| 原生 TileLang + 实际权重量化 Linear | 23 primitive、9 Linear 用例各重复 2 次；193 字节对照、113 异常与 9 空输入检查通过 | 不是完整 Attention 或 Transformer block |
| 完整原生 MoE，第 0/3 层 | 6 个规模/层组合各重复 2 次，170 字节对照、24 异常与 2 空输入检查通过 | 不是 43 层文本端到端、融合专家或 EP |

每项最终 JSON 的 `all_passed/source_unchanged` 均为 true。MoE 验证额外复核了
全部选中权重 tensor 的 payload SHA256；没有只比较 checkpoint header。
原生 MoE/结构子进程没有 Python interpreter 或 libtorch_python，显式链接 ATen C++；
独立 TileLang/Linear 子进程还不依赖 Torch。三个验证器均链接 xmake 的实际静态库。

### 实际权重与数值口径

来源为 `/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731` 的配置和原始推理代码；
权重来自节点本地 `/tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors`。
测试没有加载其余 41 层或 MTP，也没有把裁剪专家集当作完整 MoE：

| 层 | 路由 | 加载 tensor 数 | 选中权重字节数 | routed/shared expert |
|---|---|---:|---:|---|
| 0 | 按 token ID 的 hash 路由 | 1544 | 3,456,022,016 | 256 / 1，全部加载 |
| 3 | learned sqrtsoftplus + top-k | 1544 | 3,449,817,600 | 256 / 1，全部加载 |

hidden=4096、intermediate=2048、top-k=6、route scale=1.5、SwiGLU limit=10。
输入为固定种子的 BF16 hidden 与真实 vocabulary 范围内的 token IDs；测试的
1/33/137 指一次 MoE 调用的 token 行数，不是完整模型 prompt length 的性能口径。

- gate 投影与 routing weight 为 FP32；hidden 与专家投影输出为 BF16。
- routed experts 为 W4A8（packed E2M1 权重、E4M3 激活）；shared expert 为 W8A8。
- activation scale 为 1×128 UE8M0；FP4 weight scale 为 1×32；FP8 weight scale
  为 128×128。没有将权重展开为 BF16 来绕开量化 kernel。
- routing weight 在 FP32 SwiGLU 后、BF16 cast 与 down projection 前相乘；
  combine 按 expert ID 递增做 FP32 累加，最后加 shared 输出并转 BF16。

参考直接调用未替换的官方 `MoE.forward`。hook 只取路由、专家和 shared 输出；
诊断重组先与官方最终输出逐字节比较，再用于定位原生差异。项目的 MoE 实现
没有用于生成 golden。原生每组比较 14 项：input、token IDs、logits、routing
weights/IDs、hash rows 或 bias、packed hidden/weights/rows/offsets、packed expert
output、FP32 routed output、shared output 和最终 BF16 output。

| 输入行数 | 第 0 层实际非空专家 | 第 3 层实际非空专家 |
|---:|---:|---:|
| 1 | 6 | 6 |
| 33 | 140 | 130 |
| 137 | 247 | 214 |

全部专家权重都已加载，但每次只执行实际被选中的专家，不能把权重加载数写成
每组执行数。六组各执行两次，共 1486 次 routed expert 执行；shared expert 另计。
逐 backend 的名称、版本、调用数、失败数与 fallback 状态已写入 JSON，计数明确
包含构造时准备和失败注入前已执行的操作，不是纯性能测量计数。

### 新增接口及发现的问题

- `src/models/deepseek_v4/moe.*`：独立的 `Expert/MoE`、显式 `MoeBackends` 和
  `MoeWorkspace`，模型代码不引用 Torch/TileLang/Python 类型。每个 Linear 的
  quant/GEMM、activation、router、dispatch、combine 均可分别替换。
- `src/backends/aten/structural_kernel.*`：13 类结构操作及 cast/gather/finalize。
  `src/backends/hadamard/native_kernel.*` 直接链接未修改的上游 CUDA object，保留
  BSD-3-Clause 来源，不使用上游 Python extension 作为原生执行路径。
- `native::Tensor`：共享 Storage 的 checked view、offset/stride/范围校验、零元素
  无 payload；TileLang adapter 处理非零 byte offset，FP4 packed ABI 仅在后端解释。
- 首次 MoE 验证在加载环节明确失败：实际 `gate.tid2eid` 为 I64，而官方模型参数为
  I32。现已增加受检加载期转换，验证范围和整数溢出，不修改 checkpoint、不做静默
  截断。该失败不是算子精度失败；修正后两类完整 MoE 对照均通过。
- scratch 按实际专家 token 行数缓存，重复相同输入规模没有新增 scratch shape。
  backend 异常后 workspace 标记失效，后续重入直接拒绝；注入失败的 dispatch 没有
  调用默认 dispatch，输出保护区保持不变。模型配置、缺失专家、错误契约、空 selector、
  输入输出别名和错误线程也有拒绝测试。

### 构建与复现

`xmake/native_model.lua` 将 Linear/MoE 放入独立的
`llaisys-deepseek-v4-native` 静态库，不再将模型组件塞进 TileLang backend target。
Tensor 单独复用 `llaisys-native-tensor`，没有新增设备 allocator。ATen 使用 C++20，
模型及 TileLang adapter 保持 C++17。以下配置/命令在 Slurm 分配的 shell 中执行：

```bash
export CPLUS_INCLUDE_PATH=/home/lcpu/57682329/.local/deepseek-v4-python-dev/usr/include
/home/lcpu/62385178/.local/bin/xmake f -y \
  --nv-gpu=y --cuda-arch=sm_103 --flashinfer=y --python-bindings=y \
  --python-include=/home/lcpu/57682329/.local/deepseek-v4-python-dev/usr/include/python3.12 \
  --pybind11-include=/home/lcpu/57682329/.local/deepseek-v4-reference/torch/include \
  --dlpack-include=/home/lcpu/57682329/.local/deepseek-v4-tilelang-compat/tvm_ffi/include \
  --tilelang-native=y \
  --tilelang-root=/home/lcpu/57682329/.local/deepseek-v4-reference/tilelang \
  --tvm-ffi-root=/home/lcpu/57682329/.local/deepseek-v4-tilelang-compat/tvm_ffi \
  --aten-native=y --aten-cxx11-abi=1 \
  --torch-root=/home/lcpu/57682329/.local/deepseek-v4-reference/torch \
  --hadamard-source=/home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --hadamard-object=/home/lcpu/57682329/.local/deepseek-v4-hadamard/fast_hadamard_transform_cuda.cuda.o
/home/lcpu/62385178/.local/bin/xmake build llaisys-deepseek-v4-native
/home/lcpu/62385178/.local/bin/xmake build llaisys-aten-native
/home/lcpu/62385178/.local/bin/xmake build llaisys-tilelang-native

# 使用既有隔离参考环境的 PYTHONPATH，兼容 TVM-FFI 目录优先。
python3 tools/deepseek_v4_reference/verify_native_moe.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --output benchmark_results/deepseek_v4_b300_native_moe.json
python3 tools/deepseek_v4_reference/verify_native_structure.py \
  --hadamard-source=/home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --hadamard-object=/home/lcpu/57682329/.local/deepseek-v4-hadamard/fast_hadamard_transform_cuda.cuda.o \
  --native-library build/linux/x86_64/release/libllaisys-aten-native.a \
  --output benchmark_results/deepseek_v4_b300_native_structure.json
python3 tools/deepseek_v4_reference/verify_native_tilelang.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --native-library build/linux/x86_64/release/libllaisys-tilelang-native.a \
  --output benchmark_results/deepseek_v4_b300_native_tilelang.json
```

完整 C++ 链接命令、各静态库/源码 SHA256、导出库/可执行文件/manifest、测试输入输出
的校验值见三份 JSON。MoE 的临时目录只保存测试 hidden/输出和导出的 kernel，不另存
巨大专家权重；原生进程按 manifest 的受检 extent 读取既有 checkpoint。所有临时
原生可执行文件、模型权重、JIT 库及日志继续被 Git 忽略。新增 JSON 约 1.96 MB，
主要为两层 3088 个实际权重记录；不包含权重 payload。

已有 `_C.so` 与 `libllaisys.so` 的 SHA256 仍分别为
`40861b588608f0206a5fb969744931b198712237204ac610b8c32916ada8e6ee` 和
`43302daa19357dfb8eff84c15c1557d4e1dd33d112a8810903015dc1de6f44a2`。
未覆盖安装这些模块、未提交或改写 Git 历史。先前 Python 整模型/serving 回归并未在
本轮重跑，仍保留原来的范围与失败项，不能改写为本轮新证据。

### 剩余工作

下一步将已验证的结构/量化/sparse kernels 组合为原生 Attention、压缩器、Indexer、
HC 与完整 Transformer block，再迁移 43 层模型、权重装载和 batch 入口，复用现有
paged storage 与 scheduler。当前逐专家循环、offset D2H、ATen 临时分配和输出复制
均为显式参考成本；没有运行正式 TTFT/TPOT、吞吐或 peak memory 基准，也没有
CUDA Graph、grouped MoE、双卡 EP 或完整 TP×EP 证据。原始 chunk/Prefix 精度门槛
仍未通过，不因本次完整 MoE 层正确而改变。

## 原生 Compressor 与 Indexer 的实际权重闭环（2026-09-08）

### 本次新增与保留边界

在已有原生 Linear/MoE 和可替换 Kernel 接口上继续组合模型，而不是另写一套
简单 CUDA 推理实现。本次没有改动 Qwen、Python scheduler、block manager、
既有 paged storage 或共享模型源码。

1. 补录已完成的 `Compressor`：投影、增量分组、重叠状态、pooling、RMSNorm、
   RoPE、FP8/FP4 QDQ；输出明确的 first_group/count 和借用 workspace。
2. 新增 `Indexer`：W8A8 query projection → RoPE → Hadamard/FP4 QDQ → 原生
   Compressor/cache 更新 → head projection/scale → BF16 score → mask/top-k/offset。
3. 新增独立 `IndexerWeights/IndexerBackends/IndexerState/IndexerWorkspace`，
   C++ 模型代码不含 ATen/TileLang 类型；量化、GEMM、结构算子均可以逐项替换。
4. ATen 新增 `tensor_scale/indexer_mask/indexer_remap` 三个显式合同；position、
   ratio、offset 使用受检整数元数据，因果边界不经 FP32。旧合同没有隐式改选后端。
5. backend 异常后整体 Indexer state/workspace 失效；reset 清空其连续 cache 与
   Compressor 后只能使用新 workspace，防止部分 cache 写入继续参与下一次 decode。

当前 Indexer 组件持有连续 BF16 压缩缓存。它不提供第二套 block 管理，也不冒充
完整原生 paged Attention：后续需将它与既有 C++ paged pool/lease 接通，保持此前
Python paged Indexer 已明确的 gather 路径可观测。完整 Attention、HC、Transformer
block、43 层 C++ 权重入口/batch 执行仍未完成。频率 tensor 暂由离线官方函数生成。

### 实际验证与精度口径

原生组件构建 **Slurm 23058**，Indexer 首轮 **23059**，最终五组联合回归
**Slurm 23060**，均正常退出。所有 CUDA/GPU 编译与测试使用 `gpu` 分区单卡；
联合回归分配 8 CPU、48 GiB host memory、25 分钟时限。

硬件/软件与同作业 TileLang 报告一致：B300 SXM6 AC ×1、CC 10.3（sm_103）、
driver 580.126.09、CUDA 13.0、Python 3.12.3、Torch 2.13.0+cu130、TileLang 0.1.8、
TVM-FFI 0.1.8.post2、ATen CXX11 ABI=1/C++20；TF32-cuBLAS=false。
源码分支 `feat/vllm-paged-attention`、HEAD `05784b4da6483f77821a061e521ea1b98fa271e1`，
工作区包含未提交改动，因此报告另外绑定实际源码和静态库 SHA256。

| 范围 | 最终结果 | 独立性/限制 |
|---|---|---|
| Indexer 第 2/42 层 | 10 个计划各重复 2 次，76 step；614 字节对照、102 异常和 17 元数据合同检查通过；8 次空候选、12 次候选超过 top-k、2 组部分写入恢复 | native 子进程无 Python；连续缓存；不是文本端到端 |
| Compressor 三类 | 12 个计划各重复 2 次，96 step；705 字节对照、132 异常、3 组恢复；36 step 只更新未完成组 | native 子进程无 Python；独立于 Attention cache writer |
| MoE 第 0/3 层 | 6 用例、170 字节对照、24 异常、2 空输入、1486 routed expert 执行 | 全部 256 routed + 1 shared；显式 offsets D2H，不是 EP |
| ATen/Hadamard 原有覆盖 | 129 用例、374 字节对照、391 异常检查通过 | 不将其单独当作新增 Indexer 元数据测试 |
| TileLang/实际 Linear | 23 primitive、9 Linear 各重复两次，193 字节对照、113 异常、9 空输入通过 | 此 native 子进程既无 Python 也无 Torch |

五份 JSON 的 `all_passed/source_unchanged` 均为 true。Indexer/Compressor/MoE 还
在执行前后复核全部选中权重的 payload SHA256，不仅检查 header。native 子进程
读取 checkpoint 的受检 extent，不另存权重 payload；参考计算在单独 Python 准备
进程中完成，native forward 没有 Python callback，也没有 fallback。

Indexer 每层加载 7 个实际参数，均为 **13,112,064 字节**。配置为 hidden=4096、
query rank=1024、64 heads、head dimension=128、RoPE=64、ratio=4、top-k=512、
capacity=2112、batch=1。查询 Linear 使用原始 FP8 权重与 UE8M0 scales，W8A8；
查询/cache/打分/head projection 为 BF16，压缩投影和 pooling 为 FP32，查询及
压缩输出经 Hadamard 与 FP4 QDQ（block=32）。不是把整个模型换成 BF16 推理。

每层测试以下计划，输入为固定种子的随机 hidden/query，不是文本 prompt：

| 计划 | 各 step token 数 | 最终 indices/cache 的参考 |
|---|---|---|
| published_short | 3, 1, 1 | 未修改的官方 Indexer.forward |
| published_partial | 137, 1, 1 | 未修改的官方 Indexer.forward |
| published_top512 | 2105, 1, 1 | 未修改的官方 Indexer.forward；实际候选超过 512 |
| chunk_non_aligned | 3, 2, 5, 127, 1, 133 | 现有增量 Python Indexer，相同分块调度 |
| chunk_large | 257, 257, 17, 1 | 现有增量 Python Indexer，相同分块调度 |

offset 交替使用 0/128/257，比较最终候选、完整 cache、Compressor KV/score state、
head projection、FP4 QDQ query，以及输入未被改写。query 诊断由 hook 的投影结果
经原始 RoPE/Hadamard/quant 生成；最终 indices/cache 始终直接来自相应 forward。
额外测试涵盖整数大位置（67,108,862）的因果边界、非法 ratio/ID/offset、输出
别名、跨线程、错误 state/workspace、部分 cache 写入后失败及 reset 恢复。

Compressor 分别加载 `layers.2.attn.compressor`（16,794,624 字节）、
`layers.3.attn.compressor`（8,651,776 字节）和
`layers.2.attn.indexer.compressor`（4,198,656 字节）；各 4 个参数，使用实际
ratio=4/D512、ratio=128/D512、ratio=4/D128/rotate 三种配置。其初始/decode
参考来自官方，增量 chunk 参考来自已有项目实现，报告明确区分二者。

上述同调度对照不代表原始 full 与 chunk 等价，不改变现有失败门槛。未运行正式
TTFT/TPOT、吞吐、CUDA Graph 或 peak-memory benchmark，不提供性能提升声明。

### 构建与复现

使用上一节完整 xmake 配置（保留全部 native/ATen/TileLang 参数），本次没有覆盖
安装 `_C.so/libllaisys.so`。以下命令在 Slurm 分配的 shell 内执行，不能在登录节点
直接启动 CUDA。`PYTHONPATH` 使用此前隔离环境，TVM-FFI 兼容目录优先。

```bash
/home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-deepseek-v4-native
/home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-aten-native

python3 tools/deepseek_v4_reference/verify_native_indexer.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --hadamard-source /home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --hadamard-object /home/lcpu/57682329/.local/deepseek-v4-hadamard/fast_hadamard_transform_cuda.cuda.o \
  --output benchmark_results/deepseek_v4_b300_native_indexer.json

python3 tools/deepseek_v4_reference/verify_native_compressor.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --hadamard-source /home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --hadamard-object /home/lcpu/57682329/.local/deepseek-v4-hadamard/fast_hadamard_transform_cuda.cuda.o \
  --output benchmark_results/deepseek_v4_b300_native_compressor.json
```

随后按上一节命令回归 `verify_native_moe.py`、`verify_native_structure.py` 和
`verify_native_tilelang.py`。各报告保存完整 C++ 链接命令、实际静态库、参考源码、
参数记录、golden 文件、导出库和原生可执行文件 SHA256；临时目录、可执行文件、
二进制 golden、模型权重、JIT 缓存和日志仍被 Git 忽略，不加入版本库。

下一步直接推进原生 Attention/HC 和完整 Transformer block，与既有 paged pool
及请求所有权对接，再迁移完整模型/batch/权重入口。保留已有 Python 正确性路径
作为独立对照，不能把本节组件通过表述为完整基础适配已经结束。

## 原生完整 Attention 的三种模式（2026-09-08）

### 本次实现

新增 `src/models/deepseek_v4/attention.*`，将实际模型的完整 Attention 数值链
组合为一次 C++ forward，包含 W8A8 查询/latent/输出投影、归一化、RoPE、latent
QDQ、可选 Compressor/Indexer、候选准备、TileLang sparse attention、逆 RoPE
和 BF16 grouped 输出投影。没有用传统 MHA K/V 代替 V4 压缩 latent。

`AttentionWeights` 严格按 ratio 校验参数集合；`AttentionBackends` 分别注入四个
Linear 的量化/GEMM 及结构/缓存准备/稀疏算子。新模型组件不包含 ATen/TileLang
类型，也不通过 Python callback 发起 kernel；全部沿用现有 Runtime/Tensor。

新 ATen 合同为 `tensor_row_multiply` 和 `attention_prepare`：前者保留查询 BF16
归约后的乘法边界；后者显式实现连续参考 payload、候选元数据和环形 cache 更新。
block manager、paged pool、Python scheduler、Qwen 及共享模型源码均未修改。

当前 cache 是 ring window + compressed groups。初始 prefill 用当前 latent 加
压缩输出；decode 直接读 ring/cache；multi-token chunk 只 gather 至多 window-1
个历史全分辨率 latent，再加当前 chunk 和压缩组。ring 只保留最后一个窗口，避免
长 chunk 向相同环形槽重复写入。这个参考布局不等于原生 paged 路径已经接通。

错误 owner、position、输入 shape/dtype、别名、线程及缺失子状态在执行前拒绝。
计算异常后整个 Attention state/workspace 失效，包含可能已经更新的 Compressor、
Indexer 和 cache；必须整体 reset，并换用新 workspace，不允许只恢复某个子模块。

### 硬件、版本与权重

初次构建 Slurm 23061、首轮对照 23062、补强后重编译 23063，最终六组联合回归
**Slurm 23064**，均已退出。联合作业为 gpu 分区、单卡、8 CPU、48 GiB host memory、
25 分钟时限；没有在登录节点启动 CUDA，也没有执行 Qwen 测试。

NVIDIA B300 SXM6 AC ×1，CC 10.3/sm_103；driver 580.126.09，CUDA 13.0；
Python 3.12.3、Torch 2.13.0+cu130、TileLang 0.1.8、TVM-FFI 0.1.8.post2；
G++ 13.3、ATen CXX11 ABI=1、TF32-cuBLAS=false。模型仍为实际
`DeepSeek-V4-Flash-0731`，权重读取既有完整 MP1 checkpoint 的指定 tensor extent。
HEAD 仍为 `05784b4da6483f77821a061e521ea1b98fa271e1`，分支
`feat/vllm-paged-attention`，工作区有未提交改动，另存源码/静态库摘要用于复现。

| Attention 层 | ratio | 完整参数数 | 加载的实际权重字节 |
|---|---:|---:|---:|
| layers.0.attn | 0 | 12 | 140,516,992 |
| layers.2.attn | 4 | 23，包含完整 Indexer/两个 Compressor | 170,423,680 |
| layers.3.attn | 128 | 16 | 149,168,768 |

实际配置：hidden=4096、heads=64、head dim=512、query rank=1024、output groups=8、
output rank=1024、RoPE dim=64、window=128、Indexer top-k=512、capacity=2112、batch=1。
量化 Linear 为 W8A8，activation scale 为 1×128 UE8M0，weight scale 为 128×128；
query/cache/grouped projection 使用 BF16，RMSNorm/压缩投影使用 FP32，query 自身的
square/mean/rsqrt 保留 BF16 边界。latent 非 RoPE 448 维经 block=64 FP8 QDQ，
RoPE 部分不量化；Indexer 继续 Hadamard + FP4 QDQ。
`wo_a` 使用 MP1 已转换的 BF16 权重，未声称完成原始 HF FP8 `wo_a` 原生加载。

### 最终结果与比较口径

最终 `deepseek_v4_b300_native_attention.json`：18 个计划各重复两次，共 **132 step**；
**1106 次逐字节对照、175 项异常检查、18 项元数据合同检查、3 组部分写入恢复**
全部通过。每步比较输出、完整 cache、query rank、latent 原始投影、grouped 输出；
有压缩时再比较 Compressor KV/score state，ratio=4 额外比较 Indexer cache/state。

| 每层计划 | token 数 | oracle |
|---|---|---|
| published_short | 3,1,1 | 未修改的官方 Attention.forward |
| published_window | 127,1,1 | 官方，覆盖滑窗与 ratio-128 边界 |
| published_partial | 137,1,1 | 官方，覆盖非对齐压缩状态 |
| published_long | ratio-4 为 2105,1,1；其余为 257,1,1 | 官方，ratio-4 实际候选超过 Top-512 |
| chunk_non_aligned | 3,2,5,127,1,133 | 现有项目增量 Attention，相同分块 |
| chunk_large | 257,257,17,1 | 现有项目增量 Attention，相同分块 |

官方 forward 未替换算子；hook 只捕获中间 tensor。输入为确定性随机 BF16 hidden，
不是文本输入，不包含 HC、MoE 或完整 Transformer block。chunk 的 oracle 和
published full/decode 分开标记，不能把本次逐字节一致当作原始 full/chunk 一致性。

18 项元数据合同检查包括：初始因果候选、decode 环形回绕、chunk 历史窗口与候选
顺序、压缩 cache 保留、非法 position/window/ratio/top-k/scalars/ID、读写别名、
row-multiply 数值和别名拒绝。稀疏后端注入异常发生在 cache 已更新后，验证无
fallback、旧 state/workspace 不可重入，以及整体 reset 后输出/cache 恢复一致。

同一 Slurm 23064 作业中重新验证：

| 既有组件 | 回归结果 |
|---|---|
| Indexer | 76 step、614 字节对照、102 异常、17 元数据、2 组恢复通过 |
| Compressor | 96 step、705 字节对照、132 异常、3 组恢复通过 |
| 完整 MoE | 170 字节对照、24 异常、2 空输入；1486 routed expert 执行通过 |
| 原有 ATen/Hadamard | 129 用例、374 字节对照、391 异常通过 |
| TileLang/Linear | 23 primitive、9 Linear，193 字节对照、113 异常、9 空输入通过 |

六份报告均 `all_passed/source_unchanged=true`；四份模型组件报告另复核所选权重
payload SHA256。原生执行子进程均无 Python interpreter；ATen 是明确选定的参考
计算后端，不是 silent fallback。未运行正式 TTFT/TPOT、吞吐、peak memory 或
CUDA Graph benchmark，不给出性能提升声明，也没有多卡通信证据。

### 复现与后续

保留上一节完整 xmake 配置，在 Slurm 分配的 shell 中执行（兼容 TVM-FFI 的隔离
PYTHONPATH 优先）：

```bash
/home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-deepseek-v4-native
/home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-aten-native
python3 tools/deepseek_v4_reference/verify_native_attention.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --hadamard-source /home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --hadamard-object /home/lcpu/57682329/.local/deepseek-v4-hadamard/fast_hadamard_transform_cuda.cuda.o \
  --output benchmark_results/deepseek_v4_b300_native_attention.json
```

继续运行前节的五个原生 verifier，即本次联合回归顺序。所有实际编译/链接参数、
模型参数记录、源码/库/manifest/golden/可执行文件摘要保存在 JSON；权重不复制进
临时目录，二进制 golden/JIT 库/原生可执行文件和日志仍由 Git 忽略。

下一步完成原生 HC 与 Transformer block，接通既有 paged pool/lease，再推进完整
43 层 C++ 模型、权重入口和 pybind batch。既有 Python paged/serving 基线继续保留，
原始 chunk/Prefix 精度失败项、正式性能验收和 TP/EP 仍不能标记完成。

## 原生 HC 与完整 Transformer Block 验证（2026-09-08）

本次从已通过的原生 Attention/MoE 继续组合，没有修改 scheduler、block manager、
Prefix Cache 或 Qwen 路径。HC 首轮实际权重验证为 Slurm 23067，完整 Block 首轮为
Slurm 23072；补充结构合同、嵌套缓冲区别名与整层失败恢复检查后，最终八组联合
回归统一在 **Slurm 23074** 完成，作业已正常退出，exit code=0。

### 实现与边界

- 新增 `src/models/deepseek_v4/hyperconnection.*`：HC pre/post 和输出 HC head，
  每个数学阶段均通过明确的 `HCBackends` 调用；Sinkhorn 复用原始 TileLang，
  其余 FP32 结构运算默认使用原生 C++ ATen。
- 新增 `src/models/deepseek_v4/block.*`：HC pre→norm→Attention→HC post→HC pre→
  norm→MoE→HC post，整层无 Python callback。
- 新增三个版本化原生算子合同：`hc_pre_reduce`、`hc_post_mix`、`hc_head_weights`。
  后续可单独替换它们或已有量化/GEMM/router/dispatch 等实现，不要求一次替换整库。
- HC 的归约保持 FP32，不能误用 query 的 BF16 inverse RMS 舍入边界；combination
  明确按源 copy→目标 copy 混合，最后转换 BF16。
- HC workspace 持有 residual storage 的引用，遵守 pre/post 顺序、所有者和线程
  约束。Block 对所有可写子组件 workspace/cache 做输入别名检查，避免 MoE packing
  或元数据更新覆盖调用方 hidden/token IDs。
- 整个 Block 成功后才推进其 position。故障注入位于最后一个 HC post，此时
  Attention cache 已经更新、MoE 已执行；异常使整层状态和 workspace 失效，reset
  完整请求状态并换新 workspace 后，输出与干净执行逐字节一致，无静默 fallback。

仍然是显式连续/ring Attention 参考布局。这里没有宣称原生 paged、完整 43 层
C++ loader/model、pybind batch、HTTP 或多卡已经完成。测试用例加载器不是面向用户
的整模型加载入口；HC head 不包含最后的 norm/vocabulary projection 或生成驱动。

### 实际配置、权重与测试范围

- 模型：共享目录 `/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731`。
- 权重：GPU 节点现有 `/tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors`，
  完整文件 168430204680 字节。按 header extent 读取选定真实参数，没有伪造小模型
  权重或只加载被路由到的少数专家；没有重读并解包全部 43 层巨型权重。
- 分支 `feat/vllm-paged-attention`，HEAD `05784b4da6483f77821a061e521ea1b98fa271e1`，
  工作区有未提交修改；每份报告绑定实际执行源文件和静态库 SHA256。
- B300 SXM6 AC 单卡，CC 10.3 / sm_103；CUDA 13.0、Torch 2.13.0+cu130、
  TileLang 0.1.8、TVM-FFI 0.1.8.post2；原生上游 CUDA Hadamard。
- batch=1，单 runtime stream；W8A8 dense/shared、W4A8 routed experts，BF16
  hidden/cache，FP32 HC/normalization/router/compressor；TF32 关闭。
- Prefix Cache/CUDA Graph 未启用；后端全部显式选择，无 fallback。

Block 每层严格消费的参数如下（不含单独准备的 RoPE 频率 tensor）：

| 实际层 | Attention ratio | MoE 路由 | 参数 tensor 数 | 读取参数字节数 |
|---|---:|---|---:|---:|
| 0 | 0 | hash | 1564 | 3599701336 |
| 2 | 4 | hash | 1575 | 3629608024 |
| 3 | 128 | learned top-k | 1568 | 3602148696 |

HC 测试使用第 0/2/3/42 层的 attention/FFN HC，以及模型 HC head，共 9 组参数。
每组输入长度为 1、3、32、33、137、257，分别包含普通随机、全零和放大激活，
每个用例执行两次；对照未修改的官方 `Block.hc_pre/hc_post/hc_head`。

Block 每层四种执行计划，各重复两次：

| 计划 | token 序列 | 参考来源 |
|---|---|---|
| short | 3、1、1 | 原始官方完整 Block，初始 prefill 后两个 decode |
| window | 127、1、1 | 原始官方完整 Block，覆盖窗口边界 |
| long | ratio=4 为 2105、1、1；其他为 257、1、1 | 原始官方完整 Block，覆盖 Top-512 与压缩/窗口 |
| 非对齐 chunk | 3、2、5、127、1 | 现有独立 Python Block，相同分块计划 |

输入为确定性 BF16 hidden 和真实词表范围内的 token IDs，不是文本生成测试。
逐字节对照包括输入不变、最终 hidden、norm 后 Attention/MoE 输入、两者输出、
完整 latent cache、Compressor 状态和 Indexer cache/压缩状态。
相同 chunk 参考不是原始 full golden，不能据此删除 full/chunk 精度失败门槛。

### 最终八组联合回归

报告均位于 `benchmark_results/deepseek_v4_b300_native_<名称>.json`，Slurm 23074。

| 组件 | 主用例/step | tensor 字节对照 | 异常检查 | 其他检查 |
|---|---:|---:|---:|---|
| HC | 108 次用例执行 | 1073 | 174 | 17 结构合同、17 失败恢复 |
| Block | 84 step | 899 | 126 | 3 整层部分执行恢复 |
| Attention | 132 step | 1106 | 175 | 18 元数据合同、3 恢复 |
| Indexer | 76 step | 614 | 102 | 17 元数据合同、2 恢复 |
| Compressor | 96 step | 705 | 132 | 3 恢复 |
| MoE | 6 组，两次执行 | 170 | 24 | 2 空输入 |
| ATen/Hadamard 结构 | 129 用例 | 374 | 391 | 含拒绝用例 |
| TileLang/Linear | 23 核心 + 9 Linear | 193 | 113 | 9 空输入，0 析构错误 |

每份报告记录实际命令、源码/静态库/依赖库/manifest/golden/原生可执行文件摘要，
并验证执行期间源码不变。HC/Block 的选中 checkpoint payload 前后哈希一致，
不冒充本轮重新哈希了整个巨型权重文件。原生子进程没有 Python；ATen 是显式
C++ 参考依赖。既有 Python `_C.so` 与 `libllaisys.so` 未安装覆盖，SHA256 未改变。

### 复现及下一步

复用前节已验证的完整 sm_103 xmake 配置和隔离 PYTHONPATH。以下命令全部在
Slurm `--partition=gpu --gres=gpu:1` 分配内运行；最终回归使用 8 CPU、96 GiB
host memory、50 分钟时限，八个脚本顺序执行并在任一失败时停止：

```bash
/home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-deepseek-v4-native
/home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-aten-native
python3 tools/deepseek_v4_reference/verify_native_hc.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --output benchmark_results/deepseek_v4_b300_native_hc.json
python3 tools/deepseek_v4_reference/verify_native_block.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --hadamard-source /home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --hadamard-object /home/lcpu/57682329/.local/deepseek-v4-hadamard/fast_hadamard_transform_cuda.cuda.o \
  --output benchmark_results/deepseek_v4_b300_native_block.json
```

随后依次运行已有 Attention、Indexer、Compressor、MoE、结构、TileLang verifier。
没有重新测试 Qwen，没有提交权重或二进制 golden/profiler 文件。

下一步是将这些原生组件接成可装载完整权重的 C++ 模型入口，补 embedding/final
norm/vocab head、43 层执行和逐层/整模型对照，并接通既有 paged pool/lease 与
pybind batch。不能停留在独立测试可执行文件；最终高频路径须在一次 batch 调用
内执行完整模型。原始增量精度、不同请求并发、HTTP、正式 TTFT/TPOT 及 TP/EP
仍是未完成项。本轮没有性能 warmup/repeat/分位数测量，不报告 TTFT/TPOT。

## 原生整模型配置、checkpoint 与 43 层入口（2026-09-09）

本次继续上一节的原生迁移，不重写 scheduler、block manager 或 Prefix Cache。
默认计算仍是上游 TileLang、显式 ATen C++ 参考算子及上游 CUDA Hadamard。

### 新增生产接口

- `src/models/deepseek_v4/config.*`：同时解析实际 HF / inference 配置，核对模型结构、
  压缩比、量化合同、路由、RoPE 与 HC 参数。MTP/DSpark 仅作为辅助权重结构识别。
- `checkpoint.*`：从实际配置生成主模型和辅助参数清单；先检查全部名字、shape、
  dtype 和 safetensors 数据区间，再允许上传主模型权重。拒绝重复 JSON 键、未知参数、
  越界、重叠、空洞和未覆盖数据。读取期间保留原文件描述符，检查文件身份及修改。
- `model.*`：原生 embedding、43 层 Block、HC head、final norm、vocab head 与 greedy
  argmax。`ModelBackends` 按层、按算子注入实现；公共接口没有 Torch/TileLang/Python
  类型。`ModelState` 持有请求状态，任一层或输出 head 失败使整个请求状态失效。
- `rotary_frequencies`、`token_embedding`、`greedy_argmax` 是新增 ATen C++ 合同。
  RoPE 表由 C++ 计算，测试中的官方表仅用作输出对照，不作为模型输入。

加载器使用环境中现有 RapidJSON，构建时通过 `--rapidjson-include` 明确提供头文件
根目录。该依赖为私有 system include，不自行实现 JSON，也不污染公共模型接口。
现阶段只接入转换后的 MP1 checkpoint，不冒充直接支持原始 HF 多分片部署格式。

### 实际发现与验证状态

首次真实 checkpoint 校验拒绝了 `layers.10.attn.compressor.wgate.weight`。静态检查
确认实际压缩器 wkv/wgate 为 BF16 存储，而官方及既有原生 Compressor 在加载后用
FP32 执行。已在参数合同中明确允许这些投影的 BF16 来源；辅助 confidence head
也遵循其官方注释中的 BF16 存储 / FP32 参数语义。没有放宽为任意 dtype 转换。

Slurm 23085 已确认：43 层配置、67569 个主模型参数、4702 个辅助参数全部通过
无 GPU 分配的原生结构校验；14 个损坏配置/配置与实际权重不匹配用例均被拒绝。
本次主库和独立验证入口编译通过。

Slurm 23085 随后完成了完整权重加载及数值验证。验证程序
`verify_native_model.py` 对 checkpoint 全文件做 SHA256，检查既有官方 golden 的
源码、权重、输入、输出及容量身份，随后启动无 Python 的独立 C++ 进程。
本次用例是完整 43 层、2105 输入 / 16 输出、两次冷请求重置。67569 个主参数共
157458450652 字节成功装载；4702 个辅助参数只做结构校验，不执行 MTP。32 步
输出 logits 和 greedy token 全部与官方一致；加上标准与压缩 RoPE 表，共 34 次
tensor 字节对照通过。36 项模型/位置验证检查通过。decode 使用实际生成 token，
不向原生模型回灌 golden token。

报告：`benchmark_results/deepseek_v4_b300_native_model_top512.json`。全 checkpoint
168430204680 字节的 SHA256 为
`024836b412ccd5c539bb130842842247d9c427f99c3a1e21ed35ea08ddcc0aa2`。
原生执行进程未加载 Python 或 libtorch_python；ATen C++ 为显式参考依赖；TF32 关闭。
报告内源码、静态库、第三方动态库、导出 kernel、输入和 golden 的哈希均已另行复核，
当时无不匹配。该结果对应 Slurm 23085 的源码/构建快照，后续接口修改须另行回归。

当前 cache 仍是显式连续/环形参考布局；Prefix Cache、CUDA Graph 关闭，batch 和
concurrency 均为 1。本测试不是正式 TTFT/TPOT，不验收原始 full/chunk 等价，不等于
pybind batch、HTTP、多请求 fused execution 或 TP/EP 完成。随后仍须补充故障恢复、
原生 paged 接入和实际对话/任务测试。

### 整模型 pybind SchedulePlan 接口

新增 `Session` 消费既有 `src/engine/schedule_plan.hpp`，没有新增请求准入、token
budget、chunk 分配或抢占策略。一个专用 C++ 执行线程拥有模型、runtime 和所有
请求状态；跨 Python 线程调用和 close 通过该执行线程完成，资源在所属线程及其
thread-local Context 析构之前释放。此处的命令串行化不等于 serving 调度策略。

`src/backends/v4_reference.*` 负责显式组装默认 TileLang/ATen/Hadamard 后端；模型
和 Session 只依赖 `ModelBackends`/`Kernel`，用户自己的 C++ 算子库可以提供不同的
BackendFactory，不需要修改模型层数学组合。

`python/bindings/v4_native.cpp` 生成独立 `_v4_native.so`，目前只放在 build 目录，
不替换已安装 `_C.so`。构建需要实际 Python multiarch include 根目录：

```bash
# 在 Slurm GPU allocation 内，复用本节完整 xmake 配置。
CPLUS_INCLUDE_PATH=/home/lcpu/57682329/.local/deepseek-v4-python-dev/usr/include \
  /home/lcpu/62385178/.local/bin/xmake build -j 4 llaisys-v4-python
```

Python 一次 `execute(plan)` 进入 C++ 后完成整层/整模型执行，期间释放 GIL。
`capture_logits=True` 仅供诊断，会下载整 vocabulary logits，不应开启后用于正式
性能测量。当前多 slot 在一个设备上串行执行、使用独立连续/ring cache；不是 fused
batch，也尚未接回现有 InferenceEngine facade。默认只支持显式 greedy，原始
chunk/full 未通过，因此默认拒绝增量 prefill，必须明确实验开关；无静默 fallback。

首次 import 遇到 `llaisysGetRuntimeAPI` 缺失，ELF 依赖检查确认链接器省略了
`libllaisys.so`。需把必须保留的运行时库放到 `--no-as-needed` 之后；xmake 将 shflags
放在 add_links 之后，不能只在末尾单独添加该开关。这属于绑定链接问题，不是模型
数值失败；既有运行时库内容未改变。

另一个加载边界问题是 Python extension 默认为局部符号作用域，而上游导出的 kernel
只声明 TVM helper 未定义符号、没有对应的 DT_NEEDED。独立 C++ executable 没有
此问题，但 pybind 首次执行 HC 报 `TVMBackendGetFuncFromEnv` 缺失。原生 TileLang
backend 现在通过真实已链接符号定位运行时 DSO，并用 NOLOAD/GLOBAL 显式提供
符号；不猜库搜索路径，也不依赖先导入 Python TileLang。持有库引用直到 module/
function 释放，且卸载前仍同步 stream。

Slurm 23093：`verify_native_session.py` 已通过 8 组官方对话 golden，共 352 步
logits 字节对照及实际 greedy token 对照。输入长度为 19、41、48、28、32、62、
2011、21，输出长度分别为 2、54、3、14、256、2、2、19。每轮同时提交两个不同
prompt，覆盖短请求先结束并回收、另一个继续 decode；343 个含模型执行的计划，
另有 4 个最终纯回收计划。最终 active_slots=0，13 项非法计划/关闭后调用检查通过。
加载与执行释放 GIL，跨 Python 调用线程执行和关闭通过；重复 close 可安全返回。

报告 `benchmark_results/deepseek_v4_b300_native_session_chat.json` 绑定完整权重
payload 身份及实际 `_v4_native.so`、源码和 bundle。此处 8 例是数值/集成测试全部
通过，不是宣称任务得分为 8/8；原始功能检查仍为 5/8，包含参考输出自身的格式问题
和 256-token 截断。没有重新定义原有任务判定，也不把它当统计 benchmark。

### 不依赖 golden 的原生对话入口

- `export_native_backends.py` 只读取实际配置并导出 17 个 kernel 特化，不读取权重，
  不依赖测试报告。输出 `bundle.json` 包含模型配置、源文件/DSO 哈希、GPU CC 和
  Torch/TileLang/TVM-FFI/CUDA 版本；目标目录必须新建，避免覆盖既有 bundle。
- `deepseek_v4_model/native_session.py` 校验 bundle 与当前模型、GPU 和 ABI，再加载
  独立原生模块；只应加载可信的本地 DSO，摘要不是针对恶意发布者的安全签名。
- `run_native_chat.py` 使用已有实际 V4 ChatCodec 和 tokenizer，提交原生计划并用
  实际输出 token 继续生成，遇到 EOS 停止；正常路径不下载整份 logits。

Slurm 23094 已完成首次独立导出与真实对话：问题“请只回答一个数字：2 + 3 等于几？”
生成 `5` 并正常 EOS，解析无错误、slot 回收。报告为
`benchmark_results/deepseek_v4_b300_native_chat_smoke.json`。这证明入口不依赖
golden，而非新增统计精度或性能结论。

```bash
# 所有命令在 Slurm 分配内，并设置工程记录中的隔离 PYTHONPATH。
python3 tools/deepseek_v4_reference/export_native_backends.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --hadamard-source /home/lcpu/57682329/.local/src/fast-hadamard-transform \
  --output-directory <新的_bundle_目录> --capacity 4096
python3 tools/deepseek_v4_reference/run_native_chat.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --bundle <新的_bundle_目录>/bundle.json \
  --prompt '请只回答一个数字：2 + 3 等于几？' --max-new-tokens 32 \
  --output benchmark_results/deepseek_v4_b300_native_chat_smoke.json
```

Slurm 23094 随后的原生整模型及八组既有组件回归全部完成，退出码 0，各报告
`all_passed=true`、`source_unchanged=true`。整模型仍为 32 步 logits、34 次字节对照，
HC / Block / Attention / Indexer / Compressor / MoE / 结构 / TileLang 分别为
1073 / 899 / 1106 / 614 / 705 / 170 / 374 / 193 次 tensor 字节对照。
各报告绑定其执行时源码与二进制；后续 Session 改动不反向改写这些历史证据。

## 原生模型接回既有 serving 调度（2026-09-09 执行，2026-09-12 复核）

新增 `deepseek_v4_model/native_batch.py`，由已加载的原生 Session 提供现有
`InferenceEngine` batch context 协议：prefill、整批 decode、slot reset/position。
不新增请求队列，也不在 Python 中执行模型算子。多个活跃 slot 的 decode 合并为
一份 SchedulePlan，一次 pybind 调用内完成所有原生模型步骤；当前 GPU 内仍串行。

Session 的执行线程继续独占 Runtime/设备状态；Python facade 只允许一个 serving
worker 使用该 Session。停止 engine 回收全部 slot，但不销毁共享的模型权重；调用方
在 engine 停止后显式关闭 Session，可在同一权重实例上重新创建 engine/context。
诊断 observer 仅在显式启用时下载 logits；正常 serving 只返回 token IDs。

`InferenceEngine` 新增可选的模型能力检查钩子。原生 V4 当前必须显式关闭 Prefix
Cache；未启用实验性 chunk 时，chunk size 必须覆盖 slot capacity。对不支持的策略
在创建 engine 时失败，而不是悄悄忽略参数。原生缓存仍为 continuous/ring，block
size 返回既有协议的 0，表示非 paged；没有伪造 block 容量，也没有宣称支持抢占快照。

原生 `Session.info()` 新增 slot 容量、chunk 开关、整模型调用/失败计数，以及带
backend/version/contract revision 的逐算子调用/失败统计。同一个 Kernel 对象可能被
43 层及多个别名共享，统计先按对象去重再按身份聚合，避免把绑定次数误计成执行次数。

验证结果：

- CPU 协议双替身测试与既有 scheduler 共 19 项通过；覆盖整批调用、phase/采样/
  容量校验、线程所有权、重复上下文、明确拒绝未支持策略、部分执行失败回收、
  流式输出、EOS/长度结束、取消与后续请求恢复。双替身不是模型数值证据。
- Slurm **23097**：完整 43 层 / 67569 主模型 tensor，通过现有 InferenceEngine
  执行 8 个不同 prompt 的两两组合；**352 步 logits 逐字节一致**，真实生成 token
  与流式 token 全部等于官方 golden。覆盖 2011-token prompt 和 256-token 输出。
- 额外验证活跃请求在原生 prefill 完成后的取消、诊断回调异常，以及同一权重实例
  后续请求恢复。最终 356 次模型执行，15308 次 sparse attention 调用（356×43），
  模型/算子内部失败为 0；主动注入的是 Python 诊断回调异常，不混称 CUDA 故障注入。
- 最终 active_slots=0、waiting/active 队列清空；未发生 fallback。2026-09-12 再次
  检查报告中的全部文件摘要，与当前源码、原生模块、静态库及 kernel bundle 一致。

报告：`benchmark_results/deepseek_v4_b300_native_serving_chat.json`。
这些是含 logits 观察器的正确性测试，不报告其耗时为 TTFT/TPOT；任务功能判定仍
沿用原始 5/8 的边界，不把数值对齐升级为统计任务精度通过。

```bash
# 在 Slurm GPU 分配内，使用前述隔离 PYTHONPATH。
python3 tools/deepseek_v4_reference/verify_native_serving.py \
  --source-model /home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --bundle benchmark_results/native_backend_20260909_cpp/bundle.json \
  --module build/linux/x86_64/release/_v4_native.so \
  --baseline benchmark_results/deepseek_v4_b300_chat_published.json benchmark_results/deepseek_v4_chat_golden \
  --output benchmark_results/deepseek_v4_b300_native_serving_chat.json
```

尚待完成：原生 paged/Prefix 与请求快照、原始 full/chunk 数值门槛、HTTP 接入、
正式性能矩阵和多卡 TP/EP。上述原生 serving 链路不是完整适配目标的终点。

# DeepSeek-V4 完整基础适配与多算子后端验收

## 目标与边界

目标不是仅运行上游脚本或实现几项 CUDA 算子，而是让实际
`DeepSeek-V4-Flash-0731` 权重在 llaisys 的模型、执行与 serving 边界中形成完整
单卡推理闭环，并允许后续逐项替换算子库后端。所有能力以测试证据为准。

保留 Python scheduler、C++ runtime、与 payload 无关的 block manager 和 block
Prefix Cache；不重新实现第一阶段，也不把 V4 压缩稀疏缓存当成 Qwen K/V。
根据用户后续要求，不再重复 Qwen 测试，所有 GPU 工作使用 Slurm 分配。

## 2026-09-09 原生整模型进展

Slurm 23085 已完成正式 C++ config/checkpoint/Model 入口验证：完整加载 67569 个
主模型参数，校验 4702 个辅助参数；2105 输入 / 16 输出重复两次，32 步 logits 和
实际 greedy 输出逐字节对齐官方。标准/压缩 RoPE 表也由 C++ 生成并逐字节对齐。
结构负例 14 项、模型/位置负例 36 项通过，全 checkpoint payload 身份已验证。
这已超过下表 Slurm 23074 的组件/fixture 范围；报告与命令见工程记录末节。

独立 `_v4_native.so` / `Session.execute(SchedulePlan)` 已通过 Slurm 23093：两个
不同 prompt 的 slot 组合、8 组对话、352 步 logits 逐字节一致，13 项非法计划/
关闭检查通过，最终 slot 归零。后端组装独立于模型，仍保留逐算子替换接口；执行
线程只保证 runtime/资源所有权，不接管 Python scheduler 策略。Slurm 23094 已
验证不依赖 golden 的 kernel 导出、原生 Session 装载及实际对话入口，随后继续回归。

原生 InferenceEngine facade 已接回并经 Slurm 23097 验证：不同 prompt 两两同驻留，
352 步 logits 和流式 token 与原始 golden 一致；取消、诊断错误恢复、slot 回收通过。
新增原生逐算子去重计数；2026-09-12 复核报告全文件摘要仍与当前实现一致。

仍待完成原生 paged/Prefix 接入、请求抢占快照、原始 full/chunk
数值门槛、HTTP、稳定性能复测和多卡 TP/EP。Slurm 26564 已提供首组原生 serving
性能矩阵（32/256/2105 输入、16 输出、并发 1/2；warmup=1/repeat=3），但长输入
波动明显、显存仅 100 ms 采样，不能称为稳定 SLO 或精确 allocator 峰值。
Slurm 26578 的 33 项新增/既有调度与 benchmark 合同测试通过。
不能因为单卡整模型一组通过而结束
整个基础适配目标。下表旧报告按原测试范围保留，不视为新源码的全套回归。

## 验收矩阵（2026-09-08 历史快照；新增原生结果见上节）

| 能力 | 当前证据 | 尚需完成的验收 |
|---|---|---|
| 实际配置、量化与权重映射 | 独立 header-first MP1 loader 严格加载 67,569 个主模型 tensor；4,702 个 MTP tensor 仅校验结构；missing/unexpected 均为空；2105-token 基线与消费者分别完整扫描 checkpoint payload SHA256，并校验 golden 文件摘要 | 原始 HF 分片直接加载、其他 placement；旧基线仍缺少完整 payload 身份，不能反向标记已验证 |
| 独立参考基线 | 完整 43 层 TileLang + 上游 Hadamard CUDA；短输入/134-token 共 20 步，以及 2105-token 输入、16 步输出的原始 golden；另有独立固定策略同规模 golden；新增 8 例官方对话/352 步原始 golden，独立无 golden 入口自由生成 token 全部一致 | 8 例严格功能检查为 5/8，官方同样存在两项格式不符和一项长度截断；仍需更多任务/统计精度及固定策略对照，不能把功能样例当公开 benchmark |
| 可替换算子接口 | 19 类模型契约及真实调用，另有 4 类独立 cache map/write/read/copy-block 契约；paged 与原始 golden 的真实 16 步逐值一致；各后端显式选择并计数 | 批专家执行、device-only Indexer、workspace 与跨 stream 生命周期合同；Hadamard 目前是独立显式函数接口 |
| 独立 DeepSeek 模型执行器 | 独立 config/loader/model/request state；不导入上游 model.py，19 类接口接通后完整权重 20 步 logits 与参考逐值一致；新增实际对话 352 步原始 logits 全部逐值一致、自由生成一致 | C++/pybind 批执行迁移与 serving 接通；不能把 Python correctness executor 当作最终 runtime |
| 单卡 Attention/MoE | 独立 Python 模型 full prefill/decode 对照通过，BF16 小配置逐层对照通过 | 长序列、更多模型任务、真实专家调度优化与完整统计精度评价 |
| 增量请求状态 | ownership/reset/close 等测试通过；真实 43 层、2105-token/257-token 分块 paged 对独立上游固定策略 full golden 的 16 步逐值一致；同策略 full、连续缓存也一致；中间 chunk 跳过 HC head/norm/vocab 输出且维持完整 cache 更新 | 默认长 chunk 相对原始 full golden 的 relative L2 为 6.356%–39.087%，不通过 1% 门槛；固定策略 full 相对原始 full 为 6.316%–28.251%，也未通过，不替换默认基线；还需任务精度及取消/并发验证 |
| V4 paged cache / Prefix Cache | 原生存储真实 43 层 full 对原始 golden 逐值一致；已新增完整 block 的压缩器边界快照、共享/COW 与 LRU 状态清理。固定策略 2105-token 命中 2048、仅执行尾部 57，16 步与独立固定策略 golden 逐值一致。原始策略相同 2048+57 分块下命中与冷分块请求逐值一致 | 原始策略命中相对 full golden 仍未通过精度门槛；默认关闭 Prefix Cache，不能用固定策略或同分块通过替代原始精度验收。Indexer 仍 gather，full-resolution 槽未按窗口淘汰 |
| Python/C++ 执行边界 | 已按 sm_103 重建 pybind；C++ block/prefix RAII、block lease/validity 以及 paged payload 所有权已接入；DLPack 设备/stream/视图生命周期测试通过 | DeepSeek model batch execution、权重和临时 tensor 迁移；当前 Python 模型仍发起算子调用，不宣称已完成 C++ runtime |
| 原生算子与模型组件 | Slurm 23074：通用 C++ Tensor/Kernel、六类 TileLang 核心与实际权重 Linear 回归通过（193 字节对照、113 异常检查、9 空输入）；13 类 ATen 结构算子、3 类组合工具及原生 Hadamard 的 129 用例通过（374 字节对照、391 异常检查）；完整 C++ MoE，第 0/3 层各加载 256 routed + 1 shared expert，1/33/137 token 共 6 组、170 字节对照全部一致，24 异常和 2 空输入检查通过；执行进程无 Python，ATen C++ 是显式参考后端 | 尚未完成原生 paged Attention 接入、43 层模型和 batch 入口；完整 MoE 仍按专家顺序执行并下载 offsets，不是融合 MoE 或 EP；Python 端整模型基线保持独立，不能将组件结果升级为整模型验收 |
| 原生 Compressor/Indexer | Slurm 23074：实际三种 Compressor，96 step/705 字节对照/132 异常/3 组部分写入恢复通过；完整 Indexer 第 2/42 层，76 step/614 字节对照/102 异常/17 元数据合同检查/2 组恢复通过，含 2105-token Top-512；量化/GEMM/结构算子独立注入，无 Python callback 或 fallback | 原生 Indexer 当前使用连续缓存，需接入既有 paged pool/lease；频率及选中权重仍由离线 fixture 准备。published full/decode 与项目相同 chunk oracle 分开报告，不能据此宣称原始 full/chunk 已一致或文本端到端已完成 |
| 原生完整 Attention | Slurm 23074：实际第 0/2/3 层，ratio=0/4/128；132 step、1106 字节对照、175 异常、18 元数据合同与 3 组部分写入恢复通过；完整 C++ 调用投影、Compressor、Indexer、sparse attention 和输出投影，量化/GEMM/结构/prepare 均可替换 | 当前为显式连续/ring 参考布局，chunk gather 有限历史；原生 paged 存储和 43 层仍待接入。初始/decode 对官方，chunk 对同分块现有参考，不改变原始 full/chunk 失败门槛；没有 TTFT/TPOT 或完整文本验收 |
| 原生 HC / Transformer Block | Slurm 23074：HC pre/post/head，9 组实际参数、108 次用例执行、1073 项 tensor 字节对照、174 异常、17 结构合同和 17 失败恢复通过；完整 Block 第 0/2/3 层各加载全部参数（含 256 routed + 1 shared expert），84 step、899 tensor 字节对照、126 异常、3 组末尾失败恢复通过；全部无 Python callback、无 fallback | 尚未组合为 43 层 C++ 模型入口；fixture reader 不是正式模型 loader，连续/ring cache 不是原生 paged 接入；原始 full/chunk gate、文本/统计精度和正式性能仍须验收 |
| Serving 单卡端到端 | V4 facade 接通既有 InferenceEngine：worker 专属 stream、串行 slots、主机抢占快照（latent/压缩器/RNG）、取消/异常/停止回收。Slurm 23028/23029：完整 43 层，两份相同 2105-token prompt、各生成 16 token，32-block 压力下 29 次抢占恢复及 64-block 同驻留两组共 64 步 logits 均与原始 golden 逐值一致，资源归零 | 不同真实 prompt 的并发验收及更多 CUDA 错误路径显存检查；仍为 Python correctness executor，不是最终 C++ model batch runtime；HTTP 服务入口、真正 mixed GPU batch 尚未接通 |
| 精度与性能 | Slurm 23027：V4 + 既有无模型 scheduler 共 193 方法，192 通过、1 个原有 chunk/full 方法失败，0 error/skip；新增 16 项 V4 batch/serving 方法全部通过，执行期间源码未变 | 不同数值策略的任务精度；固定输入矩阵、warmup/repeat/分位数、正式 TTFT/TPOT、吞吐/包含 native allocation 的显存与后端对照 |
| 多卡 TP/EP | 尚未完成；当前 GPU QOS 仅允许单卡 | 单卡正确后双卡 EP，再实际 B300 多卡 TP×EP；必须获得并验证多卡配额，不用单卡测试替代 |

## 后端接口使用方式

`python/llaisys/models/deepseek_v4_backends.py` 目前提供执行适配接口，不接管调度策略。
完整函数、dtype/layout、MoE packed 所有权与替换示例见
`DEEPSEEK_V4_OPERATOR_BACKENDS.md`：

1. 以算子名称注册实现，携带后端名称、版本及 `OperatorContract`。
2. 明确指定每个算子的后端；缺失实现、重复注册、契约不匹配均拒绝。
3. 一次性绑定为固定调用表，执行过程中不自动改选后端。
4. 报告逐算子调用次数、异常次数、版本与 fallback 状态。
5. 自定义算子库可以只替换一个算子，其他算子继续显式使用基线实现。

示例：

```python
registry = OperatorRegistry(MODEL_CONTRACTS)
register_tilelang(registry, published_kernel_module, tilelang_version)
register_torch_model(registry)
registry.register(
    "my_operator_library", "fp4_gemm", my_w4a8_gemm,
    version="my-library-commit", contract=MODEL_CONTRACTS["fp4_gemm"],
)
selection = {name: "tilelang" if name in CONTRACTS else "torch" for name in MODEL_CONTRACTS}
selection["fp4_gemm"] = "my_operator_library"
ops = registry.bind(selection)
```

注册接口并不自动证明精度、stream 安全、并发安全或性能。实现必须遵守实际 W4A8
与 scale layout，先通过同输入算子测试，再通过逐层和整模型对照。
当前 `install_on` 仅服务于上游参考 runner。新的独立 `DeepSeekV4Model` 通过实例持有
`BoundOperators`，不再修改上游模型模块；这仍不代表独立生产 runtime 已完成。
DeepGEMM 目前是显式实验后端：整模型四个短输入、16 步 argmax 相同，但 logits
relative L2 超过预设 1% 门槛，尚未通过数值验收；默认基线保持 TileLang。

## 下一步顺序

1. 保持已通过的独立 full prefill/decode 路径；保留原始 chunk 失败门槛。已定位 FP32
   投影、inverse RMS 形状相关舍入及候选分段影响，固定策略同形状对照已通过；继续
   独立上游策略参考及完整长 chunk 对照已通过，接下来补任务精度证据，不把固定
   策略参考通过当作原始 golden 验收。实际 V4 chat 编码的 8 例/352 步原始数值与
   自由生成对照已完成；官方与独立执行器的严格任务检查均为 5/8，需要继续更广泛
   任务与固定策略对照，不把这些样例当作统计模型质量评价。
2. 保留已完成的完整权重/golden 身份和 2105-token Top-512 证据，继续扩展非对齐
   chunk、长输出和任务精度。DeepGEMM 保持显式实验后端。
3. 保留已完成的 block/压缩器快照、Prefix Cache/COW 与同策略真实权重证据；继续
   窗口回收、原始增量数值验收与任务精度，不新建 Trie、不默认切换数值策略。
4. 既有 scheduler 的 V4 参考后端接口及完整权重抢占/同驻留验收已完成首组证据；继续
   不同 prompt 的真实并发、错误路径显存验收与 C++ 模型执行迁移。随后迁移
   C++ model batch execution、权重/临时 tensor 所有权并接入 HTTP API。不能把 Python
   slot 循环、C++ cache 存储或一组通过测试改称完整高性能 runtime。
   原生迁移已打通 C++ → TileLang、ATen、Hadamard、实际量化 Linear、完整 MoE、
   三类 Compressor、完整 Indexer、三种 ratio 的连续 Attention、HC pre/post/head 和完整
   Transformer Block；新增正式配置/权重入口、embedding/final norm/vocab head、
   43 层执行与 pybind SchedulePlan 已分别对照 32/352 步真实 logits。下一步接通
   既有 paged pool/lease 及请求缓存所有权。原生 Session 已通过 facade 接回
   InferenceEngine 并验证流式/取消；继续原生分页、快照和 HTTP，不能停在现有
   continuous/ring 路径，不用逐算子 Python callback 替代 C++ 执行。
5. 正式单卡端到端正确性与性能矩阵；量化说明各后端收益及代价。
6. 配额满足后双卡 EP、完整多卡 TP×EP；随后再推进 serving 性能优化。

简历中的技术贡献应来自这些经过验证的模型接入、执行接口、缓存生命周期、精度
诊断、后端替换与性能测量，不把调用上游 TileLang 或 DeepGEMM 宣称为自研 kernel。

# 2026-09-12 会话补录与 B300 GEMM 交付索引

本次整理本地 Codex 会话中已经发生的工程工作、失败原因、恢复过程和验证边界，
并在北京时间 15:00 前的剩余时间中开发、实测两条 B300 GEMM 实验路径。
上传范围为本次补录、相关工程文档、独立 GEMM 源码/测试与选定证据报告；其他窗口
尚未提交的完整 V4 runtime、模型、Serving、Paged Cache 实现仍保留在本地工作树。
历史文档中的这些代码位置描述的是本地工程状态，不代表本次已发布全部实现。

## 会话恢复入口

- [主长会话与早期环境调错](CODEX_MAIN_SESSION_DETAILS_2026-09-12.md)：主会话跨日期
  追踪，编译/device-link、dtype、DSO 加载、配额、网络及最新原生服务 benchmark。
- [9 月 7/8 日调试补录](CODEX_SESSION_RECOVERY_2026-09-12.md)：量化尺度、归约精度、
  长输入 Indexer、DeepGEMM 探针、环境兼容和中断恢复的事实边界。
- 相关工程文档快照：[Bring-up](B300_DEEPSEEK_V4_BRINGUP.md)、
  [适配计划](DEEPSEEK_V4_BASE_ADAPTATION_PLAN.md)、
  [后端记录](DEEPSEEK_V4_OPERATOR_BACKENDS.md)。

会话原始 JSONL、认证材料、模型权重、golden tensor 和动态库均不随提交上传。
选定 JSON/编译日志保留原始结果及执行时的源码摘要，包括失败对照证据；源码摘要
不等于完整模型或整个工作树均已通过同一种验收。各报告的比较次数和门槛分别解释。

## 本轮 GEMM 成果

| 路径 | 实现与验证 | 证据入口 |
|---|---|---|
| W4A8 小 M | 自写 SM103 CUDA SIMT，M=1…8；定位并修正 K32 归约顺序后，完整 43 层/352 步 logits 逐值一致，8/8 自由生成 token 一致 | [修正版结果](B300_W4A8_ORDERED_2026-09-12.md)、[初版与调错](B300_GEMM_CUDA_2026-09-12.md) |
| FP8 预填充 | 自写 CUDA scale 布局转换，使用 cuBLASLt MXFP8 Tensor Core GEMM；5 类真实投影×3 个 M，共 15 例 | [设计与完整结果](B300_FP8_PREFILL_2026-09-12.md) |
| 独立构建 | 只复制 6 个源文件，清除额外 Python/include 路径后构建；22 项主机 ABI 通过，两条 CUDA 源未启宏时可在 sm80 下空编译 | [独立构建报告](../benchmark_results/deepseek_v4_cuda/standalone_release_build_20260912.json) |

FP8 包含动态 scale 准备及必要 padding/copy 的时间较原始 TileLang 约加速
1.33–6.66 倍。W4A8 小 M 的优势和适用边界见逐形状冷暖计时表；这些是独立算子
GPU 测量，不能直接换算为整模型吞吐、TTFT 或 TPOT 的提升。

[工具依赖说明](../tools/deepseek_v4_cuda/README.md)区分单算子复现与完整模型探针。
两条路径均通过显式独立构建/选择使用，尚未成为默认模型后端。W4A8 的完整模型
探针结果另在其设计文档记录；FP8 仍只完成投影切片验证。

## 上传约定

目标分支为 `feat/vllm-paged-attention`。本次仅按显式文件清单提交，避免把共享
工作树中其他窗口的进行中修改一并纳入。实际推送时间、提交 ID 和远端核验结果
以 Git 返回结果和本轮最终交付说明为准。

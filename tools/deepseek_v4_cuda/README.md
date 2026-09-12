# B300 独立 GEMM 实验

本目录提供 DeepSeek-V4-Flash-0731 的两条实验路径：

- [W4A8 小 M CUDA 修正版](../../项目文档/B300_W4A8_ORDERED_2026-09-12.md)：
  M=1…8，向量加载、原生 FP8/FP4 解码与顺序 FP32 归约；C ABI 和独立 Python
  wrapper。显式 variant 1 已通过完整模型数值与自由生成验证。
  [初版 tree 与调错记录](../../项目文档/B300_GEMM_CUDA_2026-09-12.md)保留失败证据。
- [FP8 预填充](README_FP8_PREFILL.md)：CUDA scale 转换与 cuBLASLt MXFP8
  Tensor Core GEMM，固定形状 plan、预分配 workspace 和 CUDA Graph。

两者都需要 CUDA 13 和实际 CC 10.3 GPU。源码用 `LLAISYS_B300_STANDALONE` 宏隔离，
默认构建不会注册这些后端。构建命令和完整数值、性能边界见上述文档。

## 依赖与可复现范围

实测环境为 Python 3.12、Torch `2.13.0+cu130`、CUDA toolkit `13.0.88`、
TileLang `0.1.8`、`apache-tvm-ffi==0.1.8.post2`。真实权重测试另需 safetensors
和 NumPy。安装与已有兼容环境说明见[工程记录](../../项目文档/B300_DEEPSEEK_V4_BRINGUP.md)。
模型、转换后的 MP1 checkpoint、golden tensors、第三方源码和二进制均需在本地准备。

| 工具 | 额外输入/依赖 |
|---|---|
| `test_w4a8_abi.py --compile` | 仅 Python 标准库和 CUDA 13；无 GPU 也可运行拒绝路径验证 |
| `test_w4a8_cuda.py` | Torch、同目录 `bench_w4a8_decode.py` 与独立 wrapper；无需模型权重 |
| `bench_w4a8_decode.py` | 官方 `inference/kernel.py`、真实 MP1 权重、TileLang |
| `bench_w4a8_memory.py` | 上述输入，以及已构建 DeepGEMM 和对应 Git checkout |
| `bench_fp8_prefill.py` | 官方 `inference/config.json`、`kernel.py` 与 TileLang；合成模式无需大权重 |

这五个算子工具不依赖完整 V4 模型实现。W4A8 工具通过文件路径加载 wrapper，
不会执行 `llaisys` 包的 runtime 导入链。它们可以从本次提交独立构建和运行，
前提是具备表中的外部输入及 B300 环境。

DeepGEMM 对照实测版本为 `2.6.1`，源码提交
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`，CUTLASS 子模块提交
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`，fmt 子模块提交
`553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`。可在任意目录准备官方
`deepseek-ai/DeepGEMM` checkout，检出该版本并初始化子模块，按其构建说明生成
`deep_gemm` 包。将构建包加入 `PYTHONPATH`，运行 memory 工具时显式传入
`--deepgemm-source /path/to/DeepGEMM`；本次提交不包含默认的
`third_party/DeepGEMM` 目录。报告记录实际导入包、扩展库摘要和源码状态。

## 完整模型探针的额外前提

`probe_w4a8_model.py` 是独立验收入口，`diagnose_w4a8_model.py` 比较同一真实
activation 上的 CUDA/TileLang 输出并固定将 TileLang 结果交给模型。两者依赖
另一个窗口的完整本地 V4 实现：
`tools/deepseek_v4_reference/run_independent.py`、Hadamard adapter 与 CUDA 扩展、
`python/llaisys/models/deepseek_v4_*` 模型/后端/证据模块，以及可正常导入的
`llaisys` runtime。它还需要真实 MP1 checkpoint、tokenizer、官方模型源码与
原始 golden 报告及 tensor 文件。这些依赖没有随本次独立 GEMM 提交全部发布。
因此精简 checkout 可以运行上表的算子验证，完整模型探针需使用已准备好的 V4
工程环境；其报告记录了执行时的全部本地源码摘要。

探针显式按 M 分派：M=1…8 使用固定 CUDA variant，M>8 使用原 TileLang，
错误直接报告，不改用其他后端。数值门槛与任务回答命中率分别记录；开启
`--free-generation` 时另要求生成 token 与原始 golden 一致。

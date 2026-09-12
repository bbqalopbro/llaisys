# DeepSeek-V4 阶段保存与续接记录（2026-09-12）

## 保存原因与验收口径

用户要求暂停开发并把现有改动上传 GitHub。本次为阶段保存，不是“完整基础适配
全部验收通过”的发布。保留已经通过验证的实现、失败报告和正在接入的代码，
不把旧报告的通过结论应用到其后修改的源码。

## 已有功能与基线

- 实际模型为共享目录中的 `DeepSeek-V4-Flash-0731`，主模型 43 层，真实 MP1 权重。
- 默认核心算子为 TileLang 0.1.8；结构算子为 ATen C++；Hadamard 使用上游 CUDA。
- dense/shared W8A8，routed experts W4A8，hidden/cache BF16；不是统一 FP16 推理。
- 已有独立模型、C++ 整模型、pybind Session、既有 InferenceEngine 接入和版本化
  多算子后端接口。模型层不依赖具体算子库类型，不重写 Python scheduler。
- 先前完整 prefill/decode、原生 serving 对话和首轮 TTFT/TPOT 的证据见
  `B300_DEEPSEEK_V4_BRINGUP.md`。这些报告绑定运行时源码哈希与环境，不代表本次
  尚未重新编译的分页桥接版本通过 GPU 回归。
- 原始 full/chunk 数值一致性门槛仍未通过；固定数值策略仅作显式实验，不替代原始基线。
- 原生 paged/Prefix/抢占接入尚未完成，HTTP、真正 mixed GPU batch、TP/EP 仍需继续。

## 本次中断前新增内容

1. 将 Python binding 内已有 block lease/Prefix 包装提取至 `src/core/cache/block_lease.*`，
   供 Python 和 C++ 共用；复用既有 BlockManager/BlockPrefixCache，没有另写 allocator。
2. 修复 Prefix match 在 retain 后 vector 分配异常可能遗失引用的问题。
3. 新增原生 `PagedStorage` 所有权包装及 native Tensor 的共享显存视图，目标是让
   既有 PagedCacheStorage 直接供 C++/TileLang 使用，关闭时拒绝活动视图并同步所属流。
4. 增强 TileLang 验证器，拟覆盖 ratio=0/4/128 的 paged-backed sparse attention 输入。
   第 3、4 项尚未完成 GPU 编译和运行，不宣称原生 paged 模型已经接通。

## 已执行与未执行的验证

- 提取后的纯 CPU lease 契约：128 checks 通过。
- 原有 cache core 测试通过。
- ASan/UBSan：128 checks 通过；`detect_leaks=0`，不能宣称 LSan 已通过。
- 隔离重编译的纯 CPU pybind 模块：现有全部 12 项 binding 测试通过；未覆盖已安装
  `_C.so` 或 `libllaisys.so`。
- 上述 CPU 报告保留于本机 `/tmp/llaisys-cache-pybind-review.sMfPy9/`，临时二进制不入 Git。
- 最新 native Tensor/PagedStorage 改动还需重新编译所有使用其头文件的原生 targets，
  并重跑 TileLang、ATen、整模型和 serving 回归；不能直接复用旧 `.so` 判断通过。
- BlockManager 底层极端 host OOM 下的 recycle/cache 异常安全仍有既存风险，当前
  lease 故障注入测试不覆盖整个 manager 的全部分配点。

## 续接顺序

先完成原生分页所有权/布局和实际 TileLang 消费验证，再接 Attention/Indexer 的
逻辑到物理槽映射、Model/Session 的 block lease，以及 Prefix/COW/状态恢复。
随后继续解决原始 full/chunk 精度门槛、任务精度评测与完整基础适配剩余项。
GPU 编译和运行必须经 Slurm；用户已明确不再要求 Qwen 测试。

DeepGEMM 作为可选实验后端以 Git submodule 固定到
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`。新环境使用
`git submodule update --init --recursive` 获取源代码；其本地 `build-llaisys/`
编译产物、模型权重、凭据、profiler SQLite 和大型测试产物均不随本次提交上传。

# B300 GEMM 复核与暂停交接（2026-09-12）

用户质疑较大加速比是否建立在达标精度上，要求继续复核；随后因额度不足要求先上传
当前改动。本轮停止新增实现与 GPU 作业。此前已启动的长输入任务在收尾期间完成，
以下严格区分已验证结果和未完成计划。

## 已完成：W4A8 长输入整模型验证

使用已发布 ordered-K32 库、固定 variant 1、真实 43 层模型和原始 top512 golden，
输入 2105 token，覆盖超过 2048 token 后的 Indexer top512 场景。

- 16/16 teacher-forced logits 逐值一致，argmax 全部一致，relative L2 全部为 0。
- 独立自由生成的完整 token IDs 与 golden 一致。
- 43 层各执行 32 次；CUDA small-M 37944 次，TileLang large-M 20022 次。
- 完整 checkpoint 身份校验通过，运行期间源码和 checkpoint 未变；进程 exit 0。

[完整报告](../benchmark_results/deepseek_v4_b300_w4a8_ordered_top512_recheck_20260912.json)
SHA256：`f576c85f267379539abcb4bbb793178a60795e63ca050ee6295083e76e516bd5`。
实际命令为原 `probe_w4a8_model.py --suite top512 --variant 1 --free-generation`，
库路径及环境、源码、权重身份均记录在报告中。此结果补充此前 chat 352 步通过，
没有扩大成所有输入精度保证，也没有新的端到端性能结论。

## 已完成的计时口径审查

此前 2.9–17.7 倍来自同一权重反复执行的热缓存 CUDA Graph 微基准，不能当作
真实服务或整模型加速。原 decode 基准按固定后端顺序测量；memory 基准每个 case
只随机一次后端顺序，然后连续采样，尚未完成逐轮配对交错复核。图内 GPU 事件不
包含 Python 提交和主机输出分配成本；既有 eager 均值也不足以描述延迟分布。
DeepGEMM 对照包含 SA/SB 转 FP32 和内部布局处理，是 adapter 总 GPU 工作量。

拟补充的公平复测尚未实现/运行：每轮随机顺序与充分预热、GPU 时钟/功耗/温度记录、
64 个真实专家权重轮换、同输入逐轮精度检查、同步单请求与批量提交的主机耗时分布。
这些审查发现不推翻已记录的同次算子测量，但限制其可推广范围。

## FP8 整模型探针：未完成草稿

新增 [probe_fp8_model.py](../tools/deepseek_v4_cuda/probe_fp8_model.py)，明确标为 WIP。
它依赖尚未实现的 `python/llaisys/models/deepseek_v4_fp8_cuda.py`，目前不可运行，
未做 CPU runtime 或完整模型 GPU 验证。只做了保存前的 Python 语法检查。

已审查到的设计约束：B-scale buffer 归属于 plan，不能仅按 device/stream/M/N/K
复用不同权重的计划；缓存还必须绑定实际 B/SB tensor 身份、地址、版本与生命周期。
原定显式 M>=32 使用 FP8 native、较小 M 使用 TileLang，无错误回退，维持 cosine
≥0.999、relative L2≤0.01、argmax 一致的完整模型门槛。该包装和验证均未完成。
原 FP8 15 个投影测试不能作为完整模型精度证明。

## 尚未落地的扩展精度验证

多层、多专家、多 seed，以及零/次正规/极值、跨 K32 scale、抵消和 BF16 舍入边界
压力测试尚未实现/运行。后续应同时比较 TileLang 与独立手工解码的 FP64 参考，
区分数学误差与参考实现一致性，并保存真实模型 A/SA 供重放。真实激活范围生成的
合成数据不能标为真实模型激活；溢出压力域与正常模型输入域应分别报告。

本次上传保留当前已完成证据和明确标记的草稿；没有切换模型默认后端，也没有把
共享工作树中其他窗口的进行中实现纳入这次提交。

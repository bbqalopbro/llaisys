# B300 FP8 prefill 独立 CUDA 后端

该候选通过 CUDA 尺度布局转换和 cuBLASLt MXFP8 Tensor Core GEMM 实现
DeepSeek-V4-Flash-0731 的五类 FP8 投影。CUDA 库不依赖 Python、Torch、ATen 或
TileLang；Python 仅用于独立测试与对照。它不是默认模型后端，也不代表整模型精度
已经验收；Tensor Core GEMM 由 cuBLASLt 提供，不把调用库描述为自研 GEMM 主内核。

## 数值与布局合同

`C[M,N] = A[M,K] @ B[N,K]^T`，具体是：

| 数据 | 连续 row-major 形状 | dtype/缩放 |
|---|---|---|
| A | M×K | E4M3；每行 K128 一个 UE8M0 scale |
| A scale | M×(K/128) | UE8M0 原始字节 |
| B | N×K | E4M3；每 128×128 块一个 UE8M0 scale |
| B scale | (N/128)×(K/128) | UE8M0 原始字节 |
| C | M×N | BF16；FP32 accumulation |

CUDA 13 文档把 `VEC128_32F`/`BLK128x128_32F` 限定于 CC 9.0；B300 CC 10.3
使用支持 Blackwell 的 `VEC32_UE8M0`。本实现将每个 K128 的 scale 复制到四个 K32
位置；B 的 scale 还沿对应 128 行复制，再写入 cuBLASLt 的 tiled scale 布局。
FP8 payload 没有重新量化，也没有转成 BF16 权重。尺度值相同不意味着归约路径逐位
相同，因此必须实测与原始 TileLang、独立 FP32 参考的差异。
[NVIDIA CUDA 13 scaling 文档](https://docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html#narrow-precision-data-types-usage)

只接受实际 B300 CC 10.3；N/K 为 128 的倍数。M 非 16 倍数时内部复制并补齐 FP8
输入，输出再截回 M 行，复制计入 `native_gemm` 和 inclusive 时间。矩阵指针至少
16B 对齐，A/B/C 与动态 A scale 不允许部分重叠。UE8M0 `0xff` 是无效 NaN 输入，
库不会静默修正为普通尺度。

## 构建

在仓库根目录运行；需要 CUDA 13 的 `nvcc`、CUDA headers 和 cuBLASLt：

```bash
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_103 -DLLAISYS_B300_STANDALONE \
  -shared -Xcompiler=-fPIC \
  src/ops/deepseek_v4/nvidia/fp8_prefill_sm103.cu \
  -lcublasLt -o build/libv4_fp8_prefill_sm103.so
```

`LLAISYS_B300_STANDALONE` 必须显式开启。既有 xmake 自动搜集 CUDA 文件时，未开启
该宏会编译为空 translation unit，不把 CUDA 13/cuBLASLt 依赖引入默认 runtime。
CUDA runtime 由上述 nvcc 默认静态链接，运行时需要能找到 `libcublasLt.so.13`。
共享目录下的库可供 Slurm 计算节点读取；登录环境 `/tmp` 不保证与计算节点相同。

## 调用与生命周期

C ABI 见 [fp8_prefill_sm103.h](../../src/ops/deepseek_v4/nvidia/fp8_prefill_sm103.h)。

1. 在将要执行的设备上创建固定 M/N/K、固定 CUDA stream 的 plan。初始化分配
   descriptor、workspace、scale buffers 和必要的 padding buffers，并查询候选算法。
2. 用 `prepare_b(plan, b_scale)` 一次性准备权重尺度；权重尺度变化后重新准备。
3. 可在初始化阶段测量 `algorithm_count` 个候选并用 `select_algorithm` 选择；正式
   测量与 graph 建立之前固定选择。harness 默认最多测量前四个 heuristic。
4. 每次调用 `run(plan, A, A_scale, B, C)`；它包括 A scale 转换与 GEMM。
   `prepare_a`/`gemm` 两个入口供调度复用或分开计时，调用 `gemm` 前应已准备尺度。
5. 停止使用所有 graph 后销毁 plan。销毁同步其 stream；返回 2 表示发生 stream
   错误但 plan 已消费且各资源释放都已尝试，不能再次使用或销毁同一指针。验证阶段
   返回 1 则不消费 plan，调用者应纠正设备/capture 状态。

执行路径不分配显存、不创建 descriptor、不做 device synchronization。一个 plan
只供一个 host 调用者和一个 stream 串行使用；并发请求/stream 各自使用独立 plan。
CUDA Graph 前先在目标 stream warmup 一次；graph replay 保留同一 plan、stream、
输入和输出地址，数据内容可以更新。创建、切算法与销毁须位于 capture 之外；不要
在另一个 stream 的 global capture 期间初始化 plan。计划不持有 PyTorch 对象，
外部 tensor 的所有权与依赖事件仍由调用者维护。

## 独立测量

Python harness 需要项目已验证的隔离环境：Torch（支持 E4M3/E8M0）、TileLang 0.1.8、
TVM-FFI 0.1.8.post2。按[工程记录](../../项目文档/B300_DEEPSEEK_V4_BRINGUP.md)设置
`PYTHONPATH`；以下命令在 Slurm GPU 分配内执行。

先做最小合成数据 smoke：

```bash
python3 tools/deepseek_v4_cuda/bench_fp8_prefill.py \
  --library build/libv4_fp8_prefill_sm103.so \
  --m 32 --shapes query_a --tune-candidates 1 \
  --graph-nodes 5 --replays 5 \
  --output benchmark_results/fp8_prefill_smoke.json
```

五类真实投影的 N/K 从本地 `inference/config.json` 读取；原始矩阵合同为 query_a
1024×4096、query_b 32768×1024、latent 512×4096、output 4096×8192、indexer
8192×1024。默认测试 M=32/256/2105。可选转换后的 MP1 checkpoint，读取第 2 层
对应 `wq_a/wq_b/wkv/wo_b/indexer.wq_b` 的真实 FP8 权重和尺度：

```bash
python3 tools/deepseek_v4_cuda/bench_fp8_prefill.py \
  --library build/libv4_fp8_prefill_sm103.so \
  --checkpoint /tmp/llaisys-deepseek-v4-flash-mp1/model0-mp1.safetensors \
  --layer 2 --m 32 256 2105 --tune-candidates 4 \
  --graph-nodes 20 --replays 20 \
  --output benchmark_results/fp8_prefill_real_weights.json
```

`--checkpoint` 只使 B 与 B scale 使用真实权重，A 仍是固定随机种子的合成激活。
读取严格检查所选 shape/dtype/范围，记录完整 header 和选中 payload 的 SHA256，
并核对文件身份前后不变；没有扫描或验证 checkpoint 的全部 payload。

报告分别保留 A scale 转换、GEMM、包含 A scale 的调用时间和 TileLang 时间。
B scale 转换、初始化、算法选择、JIT、正确性 FP32 反量化与输出 allocation 位于
计时区间外。两条路径均使用预分配输出。CUDA events 包围包含重复调用的 graph，
得到五次样本；这是 graph 下的单算子时延，不是 eager API、TTFT、TPOT 或服务吞吐。

验收同时检查 native-vs-TileLang、native-vs-FP32、TileLang-vs-FP32 的有限值与
relative L2；阈值默认 1%。capture 后先污染输出，native/TileLang 重放都必须重写
正确结果；另在相同地址将 A scale 加倍后重放，验证 graph 实际读取新的 scale。
还检查 16B offset 视图与部分别名/未对齐拒绝。最终源码/DSO/参考文件摘要必须不变。
即使这些检查通过，也不能替代使用真实模型激活、逐层 logits 和整模型生成的验收。

# 第一阶段 Runtime 与调度边界重构

## 阶段结论

第一阶段把项目收敛为以下职责边界：

- Python 负责请求队列、准入、抢占、chunk 切分、采样参数组织和调度策略。
- C++ 负责模型执行、设备内存、KV/MLA cache 存储、物理 block 生命周期和算子调用。
- Python 与 C++ 的批推理主链路优先使用 pybind11；原 ctypes 模型加载与旧快照接口暂时作为迁移兼容层保留。
- Qwen2 是 `ModelRunner` 的一个标准 KV attention 实现，不再被视为所有模型共用的数据格式。
- block id 的分配、引用计数、缓存和淘汰，与 block 中存储的是 K/V 还是 MLA latent 完全分离。

本阶段不包含 DeepSeek 模型权重映射、MLA kernel、MoE/EP、跨卡 all-to-all 或 B300 专用 kernel。这些属于下一阶段；当前接口已经为它们预留边界。

## 调用关系

```text
HTTP / API
    |
Python InferenceEngine
    |  admission / preemption / chunk policy / SchedulePlan
    v
pybind11 Qwen2BatchRuntime
    |  RAII, exception translation, GIL release
    v
C++ ModelRunner (Qwen2; later DeepSeek)
    |
    +-- BlockManager          physical id / refcount / LRU
    +-- PagedCacheStorage     layout-driven device storage
    +-- CacheLayout           StandardKV or MLA latent
    +-- attention backend     Qwen paged attention or later MLA kernel
```

## 1. Python/C++ 边界

`src/engine/schedule_plan.hpp` 定义不包含 HTTP、线程或队列概念的纯数据合同：

- `PrefillItem`：slot、token chunk、`start_pos`、是否为最后一个 chunk、采样参数；
- `DecodeItem`：slot、当前 token 和逐请求采样参数；
- `SchedulePlan`：一次调度 step 的 reset、prefill 和 decode 工作；
- `StepResult`：与 request id、slot 对齐的输出。

`python/bindings/module.cpp` 提供 `Qwen2BatchRuntime`：

- 上下文由 C++ RAII 管理；
- 参数由 pybind11 做类型转换，不再手工拼 ctypes 指针数组；
- 执行期间释放 GIL；
- C++ 异常转换为 Python 异常；
- 支持整份 `SchedulePlan`，也保留细粒度方法供现有 scheduler 渐进迁移。

`python/llaisys/models/qwen2.py` 在 `_C.so` 存在时优先使用 pybind runtime；否则回退到 ctypes。模型创建、权重装载和少量旧 cache API 仍使用 ctypes，下一阶段可继续迁移，避免本阶段同时改动权重转换、TP 和服务链路。

构建绑定示例：

```bash
xmake f --nv-gpu=y --python-bindings=y \
  --cuda-arch=sm_80 \
  --python-include=/usr/include/python3.10 \
  --pybind11-include=/path/to/pybind11/include
xmake build llaisys-python
```

当前环境也可使用 PyTorch 随附的 pybind11 headers。

## 2. 多后端和模型隔离

新增的边界文件包括：

- `src/models/common/model_runner.hpp`：模型族、attention 族和能力声明；
- `src/backends/backend.hpp`：CPU、NVIDIA、MetaX 的后端能力查询；
- `src/core/cache/cache_layout.*`：cache payload 描述；
- `src/core/cache/paged_cache_storage.*`：按 layout 分配设备存储；
- `src/core/cache/block_manager.*`：不感知数据格式的 block 生命周期。

Qwen2 声明为 `StandardKV` 模型。未来 DeepSeek runner 应声明为 `MLA`，创建 `MLACacheLayout` 并选择 MLA attention backend，而不是复制或修改 Qwen 的 block 调度逻辑。

建议后续逐步把现有大文件拆成：

```text
src/models/qwen2/       config, weights, runner
src/models/deepseek/    config, weights, mla, moe, runner
src/models/common/      model contract, sampling/output types
src/backends/nvidia/    CUDA kernels and launchers
src/backends/metax/     MetaX kernels and launchers
src/core/cache/         block lifecycle and layout-driven storage
src/engine/             C++ execution contract only
python/server/          scheduling policy and service
```

本阶段未物理移动 `qwen2.cpp`，以免和当前 CUDA Graph、cache kernel 等在途改动产生高风险冲突；但依赖方向和扩展接口已经建立。

## 3. Block 管理与 attention 数据格式解耦

旧 `BlockAllocator` 同时维护 free list 和固定 K/V pool。现在它是兼容 facade，内部组合：

- `BlockManager`：allocate、retain、release、computed 状态、prefix hash、LRU 淘汰；
- `PagedCacheStorage`：根据 `CacheLayout` 创建若干命名 component pool；
- `StandardKVCacheLayout`：`key`、`value` 两个 component；
- `MLACacheLayout`：`latent` 和可选 `rope` component，允许逐层不同字节数。

因此可以直接复用给 MLA 的是 block id、PageTable、引用计数、prefix hash 和淘汰策略；不能直接复用的是 Qwen 的 K/V stride 假设、K/V 双 pool 参数和 paged-attention kernel。DeepSeek 接入时应新增 MLA kernel adapter，不应让 `BlockManager` 感知 MLA 维度。

## 4. Chunked prefill

新增 `llaisysQwen2BatchPrefillChunk`：

- `start_pos` 必须严格等于 slot 当前位置；
- chunk 追加到现有 paged cache，不会重置 slot；
- 中间 chunk 不产生用户输出，最后一个 chunk 才按请求参数采样；
- Python scheduler 决定 `prefill_chunk_size` 并维护 chunk 边界。

当前 C++ 实现已经是 multi-token 路径：

- 每个 chunk 的 embedding、QKV、RoPE 和 MLP 均以 `[chunk_tokens, ...]` 批量执行；
- 每层先将当前 chunk 的 K/V scatter 到共享 block pool；
- 通用 `ops::paged_prefill_*` 接口隔离 model runner 与具体 GPU backend；
- NVIDIA 默认使用 FlashInfer `BatchPrefillWithPagedKVCache` 直接消费 block pool，不再 gather 历史 K/V 到连续临时张量；
- dense PageTable 到 CSR `indices/kv_indptr/qo_indptr/last_page_len` 的转换和 prefill plan 每个 chunk 只准备一次，所有 transformer layers 复用；
- 编译时禁用 FlashInfer、运行时不支持 shape/dtype，或设置 `LLAISYS_DISABLE_FLASHINFER_PREFILL=1` 时，回退到 gather+GEMM 路径；
- 中间 chunk 不执行 final norm、vocab lm_head、sampling 或 D2H，仅最后一个 chunk 产生输出。

该实现保持 block pool 为唯一权威 cache 存储，与 vLLM/SGLang 的生产路径一样由 prefill kernel 直接读 paged KV metadata。RTX 4060 Laptop 上，512-token prompt 按 256+256 切分时，direct paged 路径为 84.115 ms，gather fallback 为 108.910 ms，延迟降低约 22.8%，且输出一致。

## 5. Block Prefix Cache

`BlockPrefixCache` 使用带 parent hash 和 cache salt 的稳定链式 hash，只发布完整且已计算的 block：

- cache block 自身不属于某个请求；
- slot 命中时对 block 增加引用并直接 attach 到 PageTable；
- slot reset 时减少引用；
- 无请求引用的 cached block 可按 LRU 淘汰；
- partial block 不进入共享 cache。

查找时最多匹配到倒数第二个 prompt token 所在的完整 block。最后一个 prompt token必须经过执行路径，因为 KV cache 并不保存它对应的输出 logits。Python scheduler 随后只对未命中尾部调用增量 prefill。

旧的 token Trie + CPU KV 深拷贝前缀池已删除，Prefix Cache 只保留 block-level hash/refcount/LRU 实现。CacheSnapshot 本身仍保留，仅用于抢占恢复和 session 状态，不再用于前缀匹配。

## 6. 验证和已知边界

已执行：

```text
xmake build llaisys
xmake build llaisys-python
xmake build/run llaisys-cache-core-test
python3 -m compileall -q python scripts
pytest -q test/test_scheduler.py
pybind SchedulePlan import/round-trip smoke test
```

第一阶段的已知边界：

- 4060 环境不能验证 B300、NVLink/NVSwitch、FP8、DeepSeek MLA 和多卡 EP；
- NVIDIA 目标架构已由 `--cuda-arch` 配置，到 B300 环境后应按当地 CUDA toolkit/GPU 设置对应架构并重新编译；
- NVIDIA chunked prefill 已使用 FlashInfer direct paged 路径；gather+GEMM 仅作为可选 fallback；
- 当前 C++ batch prefill API 每次仍执行一个 slot，还未把多请求 mixed prefill/decode 合并为一次 GPU batch；
- scheduler/PageTable 仍先产生 host dense table，adapter 每个 chunk 转为 CSR 并上传；后续应直接维护 device-side metadata；
- Python scheduler 已拥有策略，但现有 worker 仍使用细粒度 batch runtime 方法；可继续收敛为每轮构造一个 `SchedulePlan`；
- prefix cache 当前是单 BatchContext、单进程内共享；跨 rank 一致性需结合 TP/EP placement 设计；
- MoE/EP 应在模型与通信层接入 all-to-all、expert placement 和容量策略，不应侵入 block manager。

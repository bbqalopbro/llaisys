# TP/PP 关键问题与业界实现方案

> 本文档记录 llaisys 项目 TP 实现的关键问题，以及业界（vLLM / Megatron-LM / TensorRT-LLM）的标准解决方案。
> 可用于后续对话窗口参考。

---

## 零、Phase 1 实施记录：FP16 AllReduce（已完成）

### 改动概览

| 文件 | 改动内容 |
|------|---------|
| `src/distributed/comm.hpp` | 新增 `CommDataType` 枚举(F32/F16/BF16)；新增 `allReduceSum(void*, count, CommDataType)` 纯虚接口；旧 `allReduceSum(float*, count)` 保留为兼容层 |
| `src/distributed/nccl_comm.cu` | 新增 `toNcclDataType()` 映射函数；`allReduceSum` 改为多精度版本，传入 `ncclFloat16`/`ncclBfloat16`/`ncclFloat32` |
| `src/distributed/mpi_comm.cpp` | FP32 走原路径；FP16/BF16 走 FP32 中转归约(标准 MPI 无 half 归约) |
| `src/distributed/mock_comm.cpp` | 签名对齐新接口，逻辑不变(空操作) |
| `include/llaisys/distributed.h` | 新增 `llaisysDistDataType_t` 枚举；新增 `llaisysDistAllReduceSum(comm, data, count, dtype)` C API |
| `src/llaisys/distributed.cc` | 新增 `_to_comm_dtype()` 转换；实现 `llaisysDistAllReduceSum` C API |
| `src/llaisys/models/qwen2.cpp` | `allReduceIfTP` 根据 `act_dtype` 自动选择 `CommDataType::F16/F32`；**去掉 TP 模式强制 FP32** |

### 关键设计决策

1. **兼容性优先**：旧 `allReduceSum(float*, count)` 作为非纯虚默认实现保留，内部转发到新接口。已有测试代码（`dist_smoke.cpp`）和 C API（`llaisysDistAllReduceSumF32`）无需任何改动。

2. **NCCL 零开销**：NCCL 原生支持 `ncclFloat16`/`ncclBfloat16`，直接传给 `ncclAllReduce`，无任何数据拷贝或类型转换。

3. **MPI FP32 中转**：标准 MPI 没有 FP16 的 `MPI_SUM` 操作。FP16/BF16 数据先展开为 FP32 → `MPI_Allreduce` → 压缩回原精度。虽有额外开销，但 MPI 后端本身是兼容方案。

4. **qwen2.cpp 核心变更**：
   - 删除 `tp_sz <= 1` 限制条件，GPU 上无论是否 TP 都使用 FP16 激活
   - `allReduceIfTP` 不再 `(float*)` 强转，而是读取 `act_dtype` 选择 `CommDataType`
   - `LLAISYS_FORCE_FP32=1` 环境变量仍可强制 FP32（用于 A/B 性能对比）

### 预期性能收益

| 指标 | 改造前 | 改造后 | 提升 |
|------|--------|--------|------|
| AllReduce 数据量 (hidden=4096) | 16 KB/call | 8 KB/call | **50%↓** |
| 通信带宽利用率 | 1× | 2× | **2×** |
| TP=2 端到端延迟 | baseline | ~-20% | **~20%↓** |

---

## 零-B、Phase 2 实施记录：通信原语补全（已完成）

### 新增原语一览

| 原语 | 签名 | NCCL 实现 | MPI 实现 | Mock 实现 |
|------|------|-----------|---------|-----------|
| AllGather | `allGather(sendbuf, recvbuf, sendcount, dtype)` | `ncclAllGather` | `MPI_Allgather` (MPI_BYTE) | memcpy |
| ReduceScatter | `reduceScatter(sendbuf, recvbuf, recvcount, dtype)` | `ncclReduceScatter` | `MPI_Reduce_scatter_block` (FP16→FP32中转) | memcpy |
| Broadcast | `broadcast(data, count, dtype, root)` | `ncclBroadcast` | `MPI_Bcast` (MPI_BYTE) | noop |
| Send | `send(data, count, dtype, dst)` | `ncclSend` | `MPI_Send` (MPI_BYTE) | throw (单卡不可用) |
| Recv | `recv(data, count, dtype, src)` | `ncclRecv` | `MPI_Recv` (MPI_BYTE) | throw (单卡不可用) |

### 改动文件

| 文件 | 改动 |
|------|------|
| `src/distributed/comm.hpp` | 新增 5 个纯虚函数 (allGather/reduceScatter/broadcast/send/recv) |
| `src/distributed/nccl_comm.cu` | NCCL 原生实现，直接映射到 ncclAllGather/ncclReduceScatter/ncclBroadcast/ncclSend/ncclRecv |
| `src/distributed/mpi_comm.cpp` | MPI 实现 + halfToFloat/floatToHalf 辅助函数；新增 GPU↔CPU 数据搬运 |
| `src/distributed/mock_comm.cpp` | AllGather/ReduceScatter 用 memcpy，Broadcast noop，Send/Recv throw |
| `include/llaisys/distributed.h` | 新增 5 个 C API 声明 |
| `src/llaisys/distributed.cc` | 新增 5 个 C API 实现 |

### 用途说明

| 原语 | 典型用途 | 数据流 |
|------|---------|--------|
| AllGather | Sequence Parallel 激活拼接 | 各 rank 的分片 → 完整序列 |
| ReduceScatter | Sequence Parallel 归约分片 | AllReduce 的前半部分 |
| Broadcast | 权重同步、采样结果广播、控制信令 | rank 0 → 所有 rank |
| Send/Recv | Pipeline Parallelism stage 间传递激活 | stage i → stage i+1 |

---

## 零-C、Phase 3 实施记录：NCCL Broadcast 替代 TCP 同步（已完成）

### 背景

原 TP 协调使用 TCP JSON 消息（`tp_worker.py` 中的 `TPCoordinator`），存在以下问题：
- TCP 连接断开 → 全部死锁
- 延迟 ~100μs（socket send/recv + JSON 序列化）
- 无错误传播机制

Phase 3 实现了用 NCCL/MPI Broadcast 替代 TCP 的 C++ 同步控制器。

### 同步协议设计

**两阶段 Broadcast 协议**：

```
阶段 1: 广播固定大小 Header (TPSyncHeader, 16 × int32 = 64 字节)
         ┌───────────────────────────────────────┐
         │ cmd | slot_id | batch_size | payload_count │
         │ temperature×1000 | top_k | top_p×1000 │
         │ reserved[9]                             │
         └───────────────────────────────────────┘
         Comm::broadcast(header, 16, CommDataType::F32, root=0)

阶段 2: 若 payload_count > 0, 广播变长 Payload (int32 数组)
         ┌──────────────────────────┐
         │ token_ids[] (prefill)     │
         │ 或 active_slots[]+current_tokens[] (decode) │
         └──────────────────────────┘
         Comm::broadcast(payload, payload_count, CommDataType::F32, root=0)
```

**命令类型**（`TPSyncCmd` 枚举）：

| 枚举值 | 命令 | Payload 内容 | 用途 |
|--------|------|-------------|------|
| 0 | Idle | 无 | 空闲心跳 |
| 1 | Prefill | token_ids[n_tokens] | 预填充请求 |
| 2 | Decode | active_slots[bs] + current_tokens[bs] | 解码步 |
| 3 | SlotReset | 无 | 重置 KV cache slot |
| 4 | Shutdown | 无 | 关闭所有 worker |

**浮点参数整数化**：temperature 和 top_p 乘以 1000 转为 int32，避免 GPU 上的浮点 broadcast 类型问题。

### 改动文件

| 文件 | 改动 |
|------|------|
| `src/distributed/tp_sync.hpp` | **新增** TPSyncController 类 (~200 行)，封装 Header+Payload 两阶段 broadcast |
| `include/llaisys/distributed.h` | 新增 `llaisysTPSync_t` 句柄 + 13 个 C API 函数 (Create/Destroy/Fill*/Broadcast/Get*) |
| `src/llaisys/distributed.cc` | 新增 `LlaisysTPSync` 包装结构 + 13 个 C API 实现 |

### 核心类：TPSyncController

```cpp
class TPSyncController {
    Comm &comm_;                     // 引用通信后端 (NCCL/MPI/Mock)
    std::vector<int> header_buf_;    // 16 × int32 固定 header
    std::vector<int> payload_buf_;   // 可变长度 payload
    size_t max_payload_;             // 最大 payload 元素数

    // rank 0 调用: 填充命令
    void fillPrefillCommand(slot_id, token_ids, n_tokens, temperature, top_k, top_p);
    void fillDecodeCommand(active_slots, current_tokens, batch_size, temperature, top_k, top_p);
    void fillSlotResetCommand(slot_id);
    void fillShutdownCommand();

    // 所有 rank 调用: 执行同步
    void broadcastSync();  // header broadcast + 条件 payload broadcast

    // 读取结果
    TPSyncCmd cmd();
    int slotId(), batchSize(), payloadCount();
    float temperature(), topP();
    int topK();
    const int* payloadData();
};
```

### C API 使用示例

```c
// 创建
llaisysTPSync_t sync = llaisysTPSyncCreate(comm, 4096);

// Rank 0: 填充 prefill 命令
int tokens[] = {1, 2, 3, 4, 5};
llaisysTPSyncFillPrefill(sync, /*slot=*/0, tokens, 5, /*temp=*/0.7, /*top_k=*/50, /*top_p=*/0.9);

// 所有 rank: 同步
llaisysTPSyncBroadcast(sync);

// 读取结果
int cmd = llaisysTPSyncGetCmd(sync);           // 1 (Prefill)
int slot = llaisysTPSyncGetSlotId(sync);       // 0
int n = llaisysTPSyncGetPayloadCount(sync);    // 5
const int* ids = llaisysTPSyncGetPayload(sync); // [1,2,3,4,5]
float temp = llaisysTPSyncGetTemperature(sync); // 0.7

// 销毁
llaisysTPSyncDestroy(sync);
```

### 预期性能收益

| 指标 | TCP JSON (旧) | NCCL Broadcast (新) | 提升 |
|------|--------------|---------------------|------|
| 同步延迟 | ~100μs | ~5μs (NVLink) | **20×↓** |
| 错误传播 | 无 (死锁) | NCCL 统一错误处理 | **可靠性↑** |
| 协议开销 | JSON 序列化/反序列化 | 固定 64 字节 Header | **零序列化** |
| 代码复杂度 | TCP server + 连接管理 | 2 行 broadcast 调用 | **大幅简化** |

### 后续集成步骤

Phase 3 完成了 C++ 基础设施和 C API。后续需要修改 Python 层：
1. `tp_worker.py`: 将 `TPCoordinator` 的 TCP 发送替换为 `llaisysTPSyncFill*` + `llaisysTPSyncBroadcast`
2. `tp_worker.py`: 将 `run_follower` 的 TCP 接收循环替换为循环调用 `llaisysTPSyncBroadcast` + `llaisysTPSyncGet*`
3. 可保留 TCP 作为 fallback（环境变量 `LLAISYS_USE_TCP_SYNC=1` 切换）

---

## 零-D、Phase 4 实施记录：多节点 NCCL 支持（已完成）

### 背景

原 NCCL 初始化通过文件系统 (`/tmp/llaisys_nccl_id`) 共享 `ncclUniqueId`，仅限所有进程在同一台机器上。
Phase 4 实现了通过 TCP socket 跨节点分发 `ncclUniqueId`，支持多机分布式推理。

### 架构设计

```
┌─────────────────────┐     TCP (128 bytes)     ┌──────────────────────┐
│  Rank 0 (Master)    │ ──────────────────────→ │  Rank 1 (Worker)     │
│  ncclGetUniqueId()  │                         │  recv ncclUniqueId   │
│  listen on port     │ ──────────────────────→ │  ncclCommInitRank()  │
│  accept & send ID   │                         └──────────────────────┘
│  ncclCommInitRank() │ ──────────────────────→ ┌──────────────────────┐
└─────────────────────┘                         │  Rank N (Worker)     │
                                                │  connect & recv ID   │
                                                │  ncclCommInitRank()  │
                                                └──────────────────────┘
```

**模式选择（自动回退）**：

| 优先级 | 条件 | 模式 | 说明 |
|--------|------|------|------|
| 1 | `config.master_addr` 非空 | TCP 分发 | 跨节点，兼容 PyTorch Distributed |
| 2 | 环境变量 `MASTER_ADDR` 非空 | TCP 分发 | 自动读取，无需修改代码 |
| 3 | 以上都不设置 | 文件共享 | `/tmp/llaisys_nccl_id`，仅限单机 |

### 改动文件

| 文件 | 改动 |
|------|------|
| `src/distributed/tcp_id_store.hpp` | **新增** TCP socket 收发 ncclUniqueId（~220 行）|
| `src/distributed/comm.hpp` | Config 新增 `master_addr` (string) 和 `master_port` (int, 默认 29400) |
| `src/distributed/nccl_comm.cu` | 三分支初始化: 单卡 / TCP 多节点 / 文件单机 |
| `include/llaisys/distributed.h` | `LlaisysDistConfig` 新增 `master_addr` (const char*) 和 `master_port` (int) |
| `src/llaisys/distributed.cc` | CommCreate 读取新字段 + 回退到 `MASTER_ADDR`/`MASTER_PORT` 环境变量 |
| `python/llaisys/libllaisys/distributed.py` | ctypes `LlaisysDistConfig` 新增两个字段 |

### TCP ID Store 设计要点

1. **RAII Socket**：`ScopedSocket` 类自动关闭 fd，防止泄漏
2. **完整读写**：`sendAll`/`recvAll` 处理 partial write/read + EINTR
3. **IPv6 Dual-Stack**：优先 IPv6（同时支持 v4/v6），回退纯 IPv4
4. **带重试连接**：Worker 每 500ms 重试连接 master，总超时 60 秒
5. **超时保护**：Master 使用 `poll()` 等待连接，防止无限阻塞

### 使用示例

```bash
# 多节点启动 (2 机器, 每台 1 GPU)
# 机器 A (rank 0, IP: 10.0.0.1):
MASTER_ADDR=10.0.0.1 MASTER_PORT=29400 \
  python -m server.app --tp-size 2 --rank 0 --device nvidia

# 机器 B (rank 1):
MASTER_ADDR=10.0.0.1 MASTER_PORT=29400 \
  python -m server.app --tp-size 2 --rank 1 --device nvidia
```

```python
# Python ctypes 用法
from llaisys.libllaisys.distributed import LlaisysDistConfig, DistBackend

cfg = LlaisysDistConfig()
cfg.backend = DistBackend.NCCL
cfg.world_size = 2
cfg.rank = 0
cfg.local_device = 0
cfg.master_addr = b"10.0.0.1"  # bytes string
cfg.master_port = 29400
```

### 兼容性

- **完全向后兼容**：不设 `master_addr` 和 `MASTER_ADDR` 时，行为与改造前完全一致（文件共享模式）
- **C API 兼容**：`LlaisysDistConfig` 使用 `{}` 零初始化时 `master_addr=NULL, master_port=0`，自动走默认路径
- **Python 兼容**：ctypes 新增字段默认为 `None/0`，现有代码无需修改

## 一、当前 TP 实现概况（Phase 1 后更新）

| 组件 | 状态 | 文件 |
|------|------|------|
| 权重分片 (Column/Row Parallel) | ✅ 正确 | `src/llaisys/models/qwen2.cpp` L432-548 |
| AllReduce 放置位置 | ✅ 正确 | `qwen2.cpp` L684, 694, 817, 840 |
| FP16/BF16 AllReduce | ✅ 已实现 | `comm.hpp` CommDataType + `nccl_comm.cu` ncclFloat16 |
| AllGather/ReduceScatter | ✅ 已实现 | `comm.hpp` + 三后端 (NCCL/MPI/Mock) |
| Broadcast | ✅ 已实现 | 可替代 TCP 同步控制信号 |
| Send/Recv (P2P) | ✅ 已实现 | 为 Pipeline Parallelism 预留 |
| NCCL 后端 | ✅ 支持多节点 | `src/distributed/nccl_comm.cu` + `tcp_id_store.hpp` |
| MPI 后端 | ⚠️ FP16走FP32中转 | `src/distributed/mpi_comm.cpp` |
| NCCL Broadcast 同步控制器 | ✅ 已实现 | `src/distributed/tp_sync.hpp` + C API |
| 多进程协调 (TCP, 遗留) | ⚠️ TCP 无认证 | `scripts/tp_worker.py` (待替换) |
| Pipeline Parallelism | ❌ 未实现 | 无模型层切分逻辑 |

---

## 二、关键问题与业界方案

### 问题 1：FP16 AllReduce 缺失（最严重）

**现状**：`allReduceSum(float* data, size_t count)` 写死 FP32，TP 模式强制激活转 FP32，浪费 50% 带宽。

**业界方案**：

```cpp
// vLLM / TensorRT-LLM 做法：模板化 AllReduce
class Comm {
public:
    // 方案A：模板化（推荐）
    template<typename T>
    void allReduceSum(T* data, size_t count, cudaStream_t stream);
    
    // 方案B：dtype 枚举
    void allReduceSum(void* data, size_t count, DataType dtype, cudaStream_t stream);
};

// NCCL 原生支持 FP16：
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat16, ncclSum, comm, stream);
// 还支持 BF16：
ncclAllReduce(sendbuf, recvbuf, count, ncclBfloat16, ncclSum, comm, stream);
```

**NCCL dtype 映射**：

| 数据类型 | ncclDataType_t | 带宽利用率 |
|----------|---------------|-----------|
| FP32 | `ncclFloat32` | 1× (baseline) |
| FP16 | `ncclFloat16` | 2× |
| BF16 | `ncclBfloat16` | 2× |
| INT8 | `ncclInt8` | 4× |

**实现步骤**：
1. 扩展 `Comm` 接口，增加 `dtype` 参数
2. NCCL 后端直接映射到 `ncclFloat16` / `ncclBfloat16`
3. MPI 后端使用 `MPI_Type_contiguous` 或直接按 `MPI_BYTE` 发送
4. 删除 `qwen2.cpp` 中 TP 模式强制 FP32 的代码

---

### 问题 2：AllGather 原语缺失

**现状**：只有 `allReduceSum`，无法实现某些 TP 变体。

**业界方案**：

```cpp
// Megatron-LM / vLLM 的通信原语集
class Comm {
    void allReduceSum(void* data, size_t count, DataType dtype, cudaStream_t stream);
    void allGather(void* sendbuf, void* recvbuf, size_t sendcount, DataType dtype, cudaStream_t stream);
    void reduceScatter(void* sendbuf, void* recvbuf, size_t recvcount, DataType dtype, cudaStream_t stream);
    void broadcast(void* data, size_t count, DataType dtype, int root, cudaStream_t stream);
    void barrier();  // 全局同步
};
```

**用途说明**：

| 原语 | 典型用途 | 数据流 |
|------|---------|--------|
| AllReduce | Row-Parallel 层后聚合 | 每个 rank 的部分和 → 全局和 |
| AllGather | Sequence Parallel (SP) 的激活拼接 | 每个 rank 的片段 → 完整序列 |
| ReduceScatter | SP 前的规约 + 分片 | 全局和 → 每个 rank 获取一片 |
| Broadcast | 权重同步、采样结果广播 | rank 0 → 所有 rank |

**vLLM 的 Custom AllReduce**：
- 小消息 (<256KB) 使用自定义 CUDA kernel（通过 shared memory / IPC）
- 大消息使用 NCCL（高效带宽利用）
- 自动切换阈值

---

### 问题 3：多节点 NCCL 支持

**现状**：NCCL Unique ID 通过 `/tmp/llaisys_nccl_id` 文件传递，仅限单机。

**业界方案（Megatron-LM / PyTorch Distributed）**：

```python
# 标准做法：通过环境变量 + TCP store 传递 NCCL Unique ID
# Rank 0 生成 unique_id，通过 TCP 广播给其他 rank

import os
os.environ["MASTER_ADDR"] = "10.0.0.1"  # Rank 0 的 IP
os.environ["MASTER_PORT"] = "29500"
os.environ["WORLD_SIZE"] = "8"
os.environ["RANK"] = str(rank)

# C++ 端：
// 1. Rank 0: ncclGetUniqueId(&id)
// 2. 通过 TCP socket 发送 id 到所有 rank
// 3. 所有 rank: ncclCommInitRank(&comm, world_size, id, rank)
```

**关键改动**：
1. 从环境变量读取 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`
2. Rank 0 创建 TCP server，广播 `ncclUniqueId`（128 bytes）
3. 其他 rank 连接到 Rank 0 获取 ID
4. 所有 rank 调用 `ncclCommInitRank`

---

### 问题 4：TCP 协调脆弱

**现状**：rank 间用裸 TCP JSON 消息同步，无认证、无错误传播。

**业界方案**：

| 方案 | 使用者 | 特点 |
|------|--------|------|
| **gRPC** | TensorRT-LLM | 强类型、有超时、TLS 支持 |
| **NCCL Group Call** | vLLM | 在 AllReduce 之前用 NCCL broadcast 同步控制信号 |
| **Ray** | vLLM v2 | 分布式调度框架，自动处理进程管理 |
| **torch.distributed** | Megatron-LM | PyTorch 内置，成熟稳定 |

**最简改进**：用 NCCL broadcast 替代 TCP 同步
```cpp
// Rank 0 填充 cmd_buf (e.g., 0=idle, 1=prefill, 2=decode)
int cmd_buf[2] = {CMD_DECODE, seq_len};
ncclBroadcast(cmd_buf, cmd_buf, 2, ncclInt, 0, comm, stream);
cudaStreamSynchronize(stream);
// 所有 rank 现在有相同的 cmd_buf
```

---

### 问题 5：Pipeline Parallelism（PP）未实现

**业界标准实现（Megatron-LM GPipe / 1F1B）**：

#### PP 核心概念

```
Pipeline Parallelism: 将模型按层切分到不同设备

Stage 0 (GPU 0): Embedding + Layer 0-6
Stage 1 (GPU 1): Layer 7-13  
Stage 2 (GPU 2): Layer 14-20
Stage 3 (GPU 3): Layer 21-27 + LM_Head

微批次调度 (1F1B Schedule):
Time →
GPU 0: F0 F1 F2 F3 B3 B2 B1 B0   (F=Forward, B=Backward)
GPU 1:    F0 F1 F2 F3 B3 B2 B1 B0
GPU 2:       F0 F1 F2 F3 B3 B2 B1 B0
GPU 3:          F0 F1 F2 F3 B3 B2 B1 B0
```

#### 推理场景 PP 实现要点

```cpp
// 1. 模型分段
struct PipelineStage {
    int stage_id;
    int start_layer, end_layer;   // 本 stage 负责的层范围
    int prev_rank, next_rank;     // 上游/下游 rank (-1 if none)
};

// 2. Forward 通信：stage i 的输出 → stage i+1 的输入
// 使用 P2P Send/Recv（不是 AllReduce）
if (next_rank >= 0) {
    ncclSend(hidden_states, count, ncclFloat16, next_rank, comm, stream);
}
if (prev_rank >= 0) {
    ncclRecv(hidden_states, count, ncclFloat16, prev_rank, comm, stream);
}

// 3. 推理时不需要 Backward，PP 退化为简单的 stage 串行
//    主要优势：每个 GPU 只加载部分层，减少显存
```

#### PP + TP 混合并行（3D Parallelism）

```
Megatron-LM 的标准做法：

8 GPU 集群 = TP=2 × PP=2 × DP=2

GPU 0,1: TP group, Stage 0, Data parallel group 0
GPU 2,3: TP group, Stage 1, Data parallel group 0  
GPU 4,5: TP group, Stage 0, Data parallel group 1
GPU 6,7: TP group, Stage 1, Data parallel group 1

通信组：
- TP group: 同一 stage 内的 AllReduce (NVLink, 高带宽)
- PP group: 跨 stage 的 P2P Send/Recv (PCIe/NVLink)
- DP group: 跨副本的 AllReduce (网络)
```

#### 对 llaisys 的改造建议

```cpp
// 需要新增的接口
class Comm {
    // P2P 通信（PP 必需）
    void send(void* data, size_t count, DataType dtype, int dst, cudaStream_t stream);
    void recv(void* data, size_t count, DataType dtype, int src, cudaStream_t stream);
    
    // 通信组管理
    static CommGroup createGroup(std::vector<int> ranks);  // 子组
};

// qwen2.cpp 改造
struct Qwen2Model {
    int pp_size, pp_rank;
    int start_layer, end_layer;  // 本 stage 的层范围
    
    void forward() {
        if (pp_rank > 0) recv_from_prev_stage();
        for (int l = start_layer; l < end_layer; l++) {
            transformer_block(l);
        }
        if (pp_rank < pp_size - 1) send_to_next_stage();
    }
};
```

---

## 三、优先级路线图

| 优先级 | 任务 | 难度 | 预期收益 | 状态 |
|--------|------|------|---------|------|
| P0 | FP16 AllReduce | 低 | TP 带宽翻倍 | ✅ 已完成 |
| P1 | AllGather + ReduceScatter | 中 | 支持 Sequence Parallel | ✅ 已完成 |
| P1 | 多节点 NCCL | 中 | 支持分布式推理 | ✅ 已完成 |
| P2 | NCCL 替代 TCP 同步 | 低 | 消除 TCP 脆弱性 | ✅ 已完成 |
| P3 | Pipeline Parallelism | 高 | 支持超大模型 | ❌ 未开始 (Send/Recv 已就绪) |
| P3 | 3D 混合并行 | 极高 | 企业级分布式 | ❌ 未开始 |

---

## 四、面试关键话术

> 面试时被问到 TP/PP 时，可以这样描述：

**TP**：
"我实现了基于 Megatron-LM 范式的 Tensor Parallelism。QKV/gate/up 做 Column-Parallel，O/down 做 Row-Parallel，Row-Parallel 后插入 AllReduce。通信层抽象了 NCCL/MPI 两个后端。目前的已知限制是 AllReduce 接口只支持 FP32，需要扩展为泛型以支持 FP16 通信。"

**PP**：
"PP 的核心是将模型按层切分到不同 stage，stage 间用 P2P Send/Recv 通信。推理场景比训练简单，不需要 1F1B 调度，只需串行 forward。关键挑战是 micro-batch 调度和跨 stage 的 KV-Cache 管理。"

---

## 五、GPU 推理性能验证

> 详见 [`PERFORMANCE_REPORT.md`](PERFORMANCE_REPORT.md) 第 10-11 节

Phase 1-4 TP 基础设施实现完毕后，在 RTX 4060 8GB 上进行了端到端推理性能验证：

| 精度 | 短输入 (16 tok) | 长输入 (512 tok) | 模型显存 |
|------|----------------|-----------------|---------|
| FP16 | 58.4 tok/s | 31.5 tok/s | ~3973 MB |
| INT8 | 17.6 tok/s | OOM | 3082 MB |
| INT4-g128 | **105.1 tok/s** | **41.9 tok/s** | **2340 MB** |

**关键结论**: INT4 量化在 bandwidth-bound 的 decode 阶段表现最优（比 FP16 快 1.3~1.8×），同时节省 41% 显存。FP16 AllReduce (Phase 1) 的通信带宽优势在 TP 场景下可叠加。

---

## 六、参考资源

| 资源 | 链接/路径 | 说明 |
|------|----------|------|
| Megatron-LM 论文 | "Efficient Large-Scale Language Model Training on GPU Clusters" | TP+PP+DP 3D 并行 |
| vLLM TP 实现 | `vllm/distributed/` | Custom AllReduce + NCCL |
| NCCL API | NVIDIA NCCL docs | `ncclAllReduce`, `ncclSend`, `ncclRecv` |
| 本项目通信抽象 | `src/distributed/comm.hpp` | 当前接口定义 |
| 本项目 NCCL 后端 | `src/distributed/nccl_comm.cu` | 当前 NCCL 实现 |
| 本项目 TP 集成 | `src/llaisys/models/qwen2.cpp` L108-113, 432-548 | 权重分片逻辑 |

# TP 代码导读（带注释）

这份文档按代码链路整理 Tensor Parallel（TP）实现。重点不是把每一行源码复制一遍，而是把所有和 TP 相关的关键代码块标出来，并解释它们在推理链路中的作用。

核心路径：

```text
创建模型时传入 tp_size / tp_rank
-> 构造函数校验 TP 维度并计算 local_nh/local_nkvh/local_di
-> 加载权重时按 Column/Row Parallel 切分
-> 初始化本地 Q/K/V、MLP、KV cache buffer
-> forward 中本 rank 只算本地 shard
-> Row Parallel 后调用 allReduceIfTP
-> Comm 抽象层转到 NCCL/MPI/Mock 后端
-> NCCL/MPI 执行 collective
```

## 1. 模型里的 TP 状态

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:107)

```cpp
// TP (Tensor Parallel) 字段
int tp_size = 1;   // TP 组大小：一个模型切到几张卡
int tp_rank = 0;   // 当前 rank 在 TP 组里的编号

// TP 后的本地维度
size_t local_nh;   // 本 rank 负责的 Q head 数 = meta.nh / tp_size
size_t local_nkvh; // 本 rank 负责的 KV head 数 = meta.nkvh / tp_size
size_t local_di;   // 本 rank 负责的 MLP intermediate 维度 = meta.di / tp_size

// TP 通信句柄；tp_size=1 或未初始化通信时为空
llaisys::distributed::comm_t comm = nullptr;
```

这里的 `local_*` 是 TP 的核心。模型整体的 head/intermediate 维度被切开，每个 rank 只保存和计算自己的那一片。

## 2. 构造函数：校验 TP 可整除并计算本地维度

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:230)

```cpp
LlaisysQwen2Model(..., int tp_sz = 1, int tp_rk = 0)
    : ..., tp_size(tp_sz), tp_rank(tp_rk) {
    ...
    if (tp_size < 1) tp_size = 1;
    if (tp_rank < 0 || tp_rank >= tp_size) tp_rank = 0;

    if (tp_size > 1) {
        if (meta.nh % tp_size != 0 ||
            meta.nkvh % tp_size != 0 ||
            meta.di % tp_size != 0) {
            // 当前实现比较 naive：不能整除就回退到单卡 TP=1
            tp_size = 1;
            tp_rank = 0;
        }
    }

    local_nh = meta.nh / tp_size;
    local_nkvh = meta.nkvh / tp_size;
    local_di = meta.di / tp_size;
}
```

这说明当前项目的 TP 依赖简单整除：

```text
num_heads % tp_size == 0
num_kv_heads % tp_size == 0
intermediate_size % tp_size == 0
```

如果不满足，不做复杂 padding 或不均匀切分，而是回退到 `tp_size=1`。

## 3. TP 下的 KV Cache 和激活 buffer

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:323)

```cpp
void init_cache() {
    // TP: 每个 rank 只存 local_nkvh 个 KV head
    cfg.nkvh = local_nkvh;
    cfg.dh = meta.dh;
    ...
}
```

KV cache 被按 KV head 切分：

```text
完整 KV: [tokens, num_kv_heads, head_dim]
本地 KV: [tokens, local_nkvh, head_dim]
```

所以 TP 不只降低权重显存，也降低每张卡的 KV cache 显存。

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:363)

```cpp
void init_buffers() {
    hidden_states = Tensor::create({1, meta.hs}, ...);
    residual      = Tensor::create({1, meta.hs}, ...);
    norm_out      = Tensor::create({1, meta.hs}, ...);

    // Q/K/V 使用本地 head 数
    size_t q_dim  = local_nh * meta.dh;
    size_t kv_dim = local_nkvh * meta.dh;

    q = Tensor::create({1, q_dim}, ...);
    k = Tensor::create({1, kv_dim}, ...);
    v = Tensor::create({1, kv_dim}, ...);

    attn_out = Tensor::create({1, local_nh, meta.dh}, ...);

    // MLP intermediate 使用本地 local_di
    gate    = Tensor::create({1, local_di}, ...);
    up      = Tensor::create({1, local_di}, ...);
    mlp_act = Tensor::create({1, local_di}, ...);
}
```

注意：

```text
hidden_states/residual/norm_out: 仍是完整 hidden_size
q/k/v/attn_out/gate/up/mlp_act: 是本 rank 的 shard
```

这正是 Megatron 风格 TP：层边界处保持完整 hidden，Column Parallel 后进入 shard 状态，Row Parallel 后 AllReduce 回完整 hidden。

## 4. 权重分类：哪些 Column，哪些 Row

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:478)

```cpp
enum class TpSlice { None, ColDim0, RowDim1 };
```

这里命名和数学里的 `W: [in, out]` 要稍微换一下脑子。项目里权重 tensor 常按 `[out_features, in_features]` 存。

```cpp
static TpSlice classifyWeight(const std::string& suffix) {
    // Column Parallel：按 out_features 切
    // 在当前 [out, in] 存储里就是切 dim0
    if (suffix == "self_attn.q_proj.weight" ||
        suffix == "self_attn.k_proj.weight" ||
        suffix == "self_attn.v_proj.weight" ||
        suffix == "mlp.gate_proj.weight" ||
        suffix == "mlp.up_proj.weight" ||
        ... scale/qzeros/bias ...) {
        return TpSlice::ColDim0;
    }

    // Row Parallel：按 in_features 切
    // 在当前 [out, in] 存储里就是切 dim1
    if (suffix == "self_attn.o_proj.weight" ||
        suffix == "mlp.down_proj.weight" ||
        ... qzeros ...) {
        return TpSlice::RowDim1;
    }

    // embedding、norm、lm_head 等不切，复制到各 rank
    return TpSlice::None;
}
```

对应关系：

```text
Column Parallel:
  q_proj / k_proj / v_proj
  mlp.gate_proj / mlp.up_proj

Row Parallel:
  self_attn.o_proj
  mlp.down_proj

不切分:
  embedding / norm / lm_head / 部分 scale
```

面试记法：

```text
QKV 和 gate/up 切输出维，本地继续 attention/SwiGLU
o_proj 和 down_proj 切输入维，输出 partial hidden，后面 AllReduce
```

## 5. 权重切片函数

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:502)

```cpp
static std::vector<uint8_t> slice2D(..., TpSlice how,
                                    int tp_size, int tp_rank,
                                    size_t& out_rows, size_t& out_cols) {
    if (how == TpSlice::ColDim0) {
        // Column Parallel：按 dim0/输出行切
        out_rows = rows / tp_size;
        out_cols = cols;

        // [out, in] 的 dim0 连续，所以可以一次 memcpy
        size_t offset = tp_rank * out_rows * cols * elem_size;
        size_t bytes = out_rows * cols * elem_size;
        memcpy(buf.data(), data + offset, bytes);
    } else {
        // Row Parallel：按 dim1/输入列切
        out_rows = rows;
        out_cols = cols / tp_size;

        // dim1 不是整块连续，需要逐行拷贝当前 rank 的列范围
        for (size_t r = 0; r < rows; ++r) {
            src = data + (r * cols + offset_cols) * elem_size;
            dst = buf + r * row_bytes;
            memcpy(dst, src, row_bytes);
        }
    }
}
```

1D 切片用于 Column Parallel 的 bias/scale：

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:533)

```cpp
static std::vector<uint8_t> slice1D(...) {
    out_size = size / tp_size;
    offset = tp_rank * out_size * elem_size;
    memcpy(buf.data(), data + offset, bytes);
}
```

## 6. 加载权重时应用 TP 切分

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:994)

```cpp
TpSlice how = TpSlice::None;
if (model->tp_size > 1) {
    if (key.find("model.layers.") == 0) {
        // 根据权重名判断 Column/Row/None
        std::string suffix = key.substr(second_dot + 1);
        how = classifyWeight(suffix);

        // AWQ 原生布局和标准 [out, in] 相反，需要翻转切分方向
        if (model->has_awq_native && how != TpSlice::None &&
            suffix.find(".weight") != std::string::npos) {
            how = (how == TpSlice::ColDim0)
                ? TpSlice::RowDim1
                : TpSlice::ColDim0;
        }
    }
}
```

真正创建 tensor：

```cpp
if (how == TpSlice::None || model->tp_size <= 1) {
    // 不切分：完整复制到本 rank
    tensor = Tensor::create(shape_vec, dtype, ...);
    tensor->load(data);
} else if (ndim == 2) {
    // 2D 权重切片后再 load
    auto sliced = slice2D(data, rows, cols, elem_size,
                          how, model->tp_size, model->tp_rank,
                          out_rows, out_cols);
    tensor = Tensor::create({out_rows, out_cols}, dtype, ...);
    tensor->load(sliced.data());
} else if (ndim == 1 && how == TpSlice::ColDim0) {
    // Column 的 bias/scale 也按输出维切
    auto sliced = slice1D(...);
    tensor = Tensor::create({out_size}, dtype, ...);
    tensor->load(sliced.data());
}
```

所以你的项目是 **加载时切权重**，不是 forward 时先算完整权重再切结果。

## 7. 单请求 decode 中的 TP forward

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:784)

```cpp
auto q_3d = model->q->reshape({1, model->local_nh,   model->meta.dh});
auto k_3d = model->k->reshape({1, model->local_nkvh, model->meta.dh});
auto v_3d = model->v->reshape({1, model->local_nkvh, model->meta.dh});
```

QKV Column Parallel 后，本 rank 只得到本地 heads：

```text
q: [1, local_nh, head_dim]
k/v: [1, local_nkvh, head_dim]
```

本地 RoPE、本地 KV 写入、本地 PagedAttention：

```cpp
ops::rope(q_3d, q_3d, pos_ids_buf, theta);
ops::rope(k_3d, k_3d, pos_ids_buf, theta);

ops::reshape_and_cache(... local_nkvh ...);

ops::paged_attention_device(... local_nh, local_nkvh ...);
```

attention 输出也是本地 head slice：

```cpp
auto attn_flat = model->attn_out->reshape(
    {1, model->local_nh * model->meta.dh});
```

### 7.1 attention o_proj 后 AllReduce

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:820)

```cpp
if (model->tp_size <= 1) {
    // 单卡：可以把 linear + residual add 融合
    linear_maybe_dequant(hidden_states, attn_flat, o_proj, ..., residual);
} else {
    // TP：o_proj 是 Row Parallel
    // 每个 rank 得到完整 hidden shape 的 partial sum
    linear_maybe_dequant(hidden_states, attn_flat, o_proj, ...);

    // 对所有 rank 的 partial hidden 做逐元素 sum
    model->allReduceIfTP(model->hidden_states, model->meta.hs);

    // AllReduce 后才和 residual 相加
    ops::add(model->hidden_states, model->hidden_states, model->residual);
}
```

为什么 TP 分支不融合 residual？

```text
Row Parallel 输出还只是 partial sum
必须先 AllReduce 得到完整 hidden
再 residual add
```

### 7.2 MLP down_proj 后 AllReduce

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:833)

```cpp
// gate/up 是 Column Parallel，输出 local_di
linear_maybe_dequant(gate, norm_out, gate_proj, ...);
linear_maybe_dequant(up,   norm_out, up_proj,   ...);

// SwiGLU 是逐元素操作，可以直接吃本地 shard
ops::swiglu(mlp_act, gate, up);

if (model->tp_size <= 1) {
    linear_maybe_dequant(hidden_states, mlp_act, down_proj, ..., residual);
} else {
    // down_proj 是 Row Parallel，输出 partial hidden
    linear_maybe_dequant(hidden_states, mlp_act, down_proj, ...);

    // 汇总各 rank 的 partial hidden
    model->allReduceIfTP(model->hidden_states, model->meta.hs);

    // 再 residual add
    ops::add(model->hidden_states, model->hidden_states, model->residual);
}
```

## 8. Batch decode 中的 TP forward

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:1670)

QKV 后 reshape 成本地 heads：

```cpp
auto q_3d = b_q->reshape({B, model->local_nh, meta.dh});
auto k_3d = b_k->reshape({B, model->local_nkvh, meta.dh});
auto v_3d = b_v->reshape({B, model->local_nkvh, meta.dh});
```

PagedAttention 也只看本地 heads 和本地 KV cache：

```cpp
ops::paged_attention(...,
    static_cast<int>(model->local_nh),
    static_cast<int>(model->local_nkvh),
    ...);
```

attention o_proj 后：

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:1723)

```cpp
ctx->linear_maybe_dequant(b_hidden, b_attn, o_proj, ...);
model->allReduceIfTP(b_hidden, B * meta.hs);
ops::add(b_hidden, b_hidden, b_resid);
```

MLP down_proj 后：

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:1738)

```cpp
ctx->linear_maybe_dequant(b_gate, b_norm, gate_proj, ...);
ctx->linear_maybe_dequant(b_up,   b_norm, up_proj,   ...);
ops::swiglu(b_mlp, b_gate, b_up);

ctx->linear_maybe_dequant(b_hidden, b_mlp, down_proj, ...);
model->allReduceIfTP(b_hidden, B * meta.hs);
ops::add(b_hidden, b_hidden, b_resid);
```

## 9. allReduceIfTP：模型层到通信层的入口

位置：[src/llaisys/models/qwen2.cpp](/home/bbq/llaisys/src/llaisys/models/qwen2.cpp:213)

```cpp
void allReduceIfTP(tensor_t t, size_t count) {
    if (tp_size > 1 && comm) {
        // 根据激活 dtype 选择通信 dtype
        auto comm_dtype = CommDataType::F32;
        if (act_dtype == LLAISYS_DTYPE_F16) {
            comm_dtype = CommDataType::F16;
        } else if (act_dtype == LLAISYS_DTYPE_BF16) {
            comm_dtype = CommDataType::BF16;
        }

        // 统一通信接口；底层可能是 NCCL/MPI/Mock
        comm->allReduceSum(t->data(), count, comm_dtype);
    }
}
```

关键点：

```text
count 是元素个数，不是字节数
F16/BF16 每元素 2B，F32 每元素 4B
TP 主路径只调用 AllReduce，不用 AllGather/ReduceScatter
```

## 10. 通信抽象 Comm

位置：[src/distributed/comm.hpp](/home/bbq/llaisys/src/distributed/comm.hpp:19)

```cpp
enum class Backend {
    Mock = 0, // 单卡测试，通信是空操作
    Nccl = 1, // GPU 间高速通信
    Mpi  = 2, // 通用多进程通信，GPU 数据通常要 CPU 中转
};

enum class CommDataType {
    F32 = 0,  // 4 bytes
    F16 = 1,  // 2 bytes
    BF16 = 2, // 2 bytes，动态范围接近 FP32
};
```

统一接口：

位置：[src/distributed/comm.hpp](/home/bbq/llaisys/src/distributed/comm.hpp:68)

```cpp
class Comm {
public:
    virtual int worldSize() const = 0;
    virtual int rank() const = 0;

    // TP 主路径使用这个：Row Parallel 后聚合 partial sum
    virtual void allReduceSum(void *data, size_t count,
                              CommDataType dtype) = 0;

    // 后续 SP/扩展用，目前 Qwen2 主路径基本不用
    virtual void allGather(...) = 0;
    virtual void reduceScatter(...) = 0;
    virtual void broadcast(...) = 0;
    virtual void send(...) = 0;
    virtual void recv(...) = 0;
    virtual void barrier() = 0;
};
```

模型层只依赖 `Comm`，不直接调用 NCCL/MPI，这就是通信后端隔离。

## 11. NCCL 初始化：rank 如何加入通信组

位置：[src/distributed/nccl_comm.cu](/home/bbq/llaisys/src/distributed/nccl_comm.cu:112)

```cpp
NcclComm::NcclComm(const Config &config)
    : _world_size(config.world_size),
      _rank(config.rank),
      _device(config.local_device) {

    cudaSetDevice(_device);
    cudaStreamCreate(&_stream);

    ncclUniqueId nccl_id;

    if (_world_size == 1) {
        ncclGetUniqueId(&nccl_id);
    } else if (!config.master_addr.empty()) {
        // 多节点 TCP 模式
        if (_rank == 0) {
            ncclGetUniqueId(&nccl_id);
        }
        tcpExchangeNcclId(nccl_id.internal, _rank, _world_size,
                          config.master_addr, config.master_port);
    } else {
        // 单机共享文件模式
        if (_rank == 0) {
            ncclGetUniqueId(&nccl_id);
            writeNcclId(nccl_id, id_file);
        } else {
            nccl_id = readNcclId(id_file);
        }
    }

    // 所有 rank 使用同一个 nccl_id + 各自 rank 初始化 communicator
    ncclCommInitRank(&_nccl_comm, _world_size, nccl_id, _rank);
}
```

`ncclUniqueId` 只用于初始化，真正 tensor 数据不走文件/TCP。

```text
rank0 生成 unique id
其他 rank 读取/接收同一个 id
所有 rank 调 ncclCommInitRank
之后 collective 都在 _nccl_comm 上执行
```

## 12. NCCL AllReduce 实现

位置：[src/distributed/nccl_comm.cu](/home/bbq/llaisys/src/distributed/nccl_comm.cu:199)

```cpp
void allReduceSum(void *data, size_t count, CommDataType dtype) override {
    if (count == 0) return;

    cudaSetDevice(_device);

    // F16/BF16/F32 -> ncclFloat16/ncclBfloat16/ncclFloat32
    ncclDataType_t nccl_dtype = toNcclDataType(dtype);

    // 就地 AllReduce：sendbuf == recvbuf
    ncclAllReduce(
        data, data, count,
        nccl_dtype, ncclSum,
        _nccl_comm, _stream);

    // 当前项目使用同步调用，等待通信完成再返回
    cudaStreamSynchronize(_stream);
}
```

这里的协同语义：

```text
所有 rank 必须在同一个 communicator 上，以相同顺序调用 ncclAllReduce
每个 rank 的 data shape/count/dtype 必须匹配
NCCL 内部用 ring/tree 等算法把各 rank 的 data 做逐元素 sum
最终每个 rank 的 data 都变成 sum 后的完整 hidden
```

如果某个 rank 没有进入同一次 collective，其他 rank 会等待、报错或 hang，不会自动回退 TP=1。

## 13. NCCL AllGather / ReduceScatter / Broadcast

位置：[src/distributed/nccl_comm.cu](/home/bbq/llaisys/src/distributed/nccl_comm.cu:236)

```cpp
void allGather(sendbuf, recvbuf, sendcount, dtype) {
    ncclAllGather(sendbuf, recvbuf, sendcount, nccl_dtype,
                  _nccl_comm, _stream);
}
```

用途：

```text
把各 rank 的 shard 拼接起来，让每个 rank 都拿到完整 tensor
当前 Qwen2 TP 主路径基本不用
```

位置：[src/distributed/nccl_comm.cu](/home/bbq/llaisys/src/distributed/nccl_comm.cu:268)

```cpp
void reduceScatter(sendbuf, recvbuf, recvcount, dtype) {
    ncclReduceScatter(sendbuf, recvbuf, recvcount,
                      nccl_dtype, ncclSum,
                      _nccl_comm, _stream);
}
```

用途：

```text
先 sum，再把结果按 rank 切片
主要给 Sequence Parallel 预留
当前 Qwen2 TP 主路径不用
```

Broadcast/Send/Recv 也是通信抽象层能力，当前 TP forward 的核心不是它们。

## 14. MPI 后端

位置：[src/distributed/mpi_comm.cpp](/home/bbq/llaisys/src/distributed/mpi_comm.cpp:134)

```cpp
void allReduceSum(void *data, size_t count, CommDataType dtype) override {
    if (dtype == CommDataType::F32) {
        allReduceSumF32(static_cast<float*>(data), count);
        return;
    }

    // FP16/BF16: 标准 MPI 没有 half SUM，转成 FP32 归约再转回
    // 如果 data 是 GPU pointer，还要 GPU -> CPU -> MPI -> CPU -> GPU
}
```

MPI 后端是兼容/fallback，不是高性能 GPU TP 首选。

```text
NCCL: GPU direct，适合 TP 主路径
MPI: 通用进程通信；GPU 数据通常 CPU 中转，慢
Mock: 单卡测试，通信为空操作
```

## 15. TP 主流程总图

```text
每一层 Transformer:

完整 hidden
-> RMSNorm
-> q/k/v Column Parallel linear
   每 rank 得到 local Q/K/V heads
-> RoPE on local Q/K
-> 写本地 local_nkvh KV cache
-> local paged attention
   输出 local attention heads
-> o_proj Row Parallel
   每 rank 得到 hidden partial sum
-> AllReduce(sum)
   每 rank 得到完整 hidden
-> residual add

完整 hidden
-> RMSNorm
-> gate/up Column Parallel linear
   每 rank 得到 local_di intermediate shard
-> local SwiGLU
-> down_proj Row Parallel
   每 rank 得到 hidden partial sum
-> AllReduce(sum)
   每 rank 得到完整 hidden
-> residual add
```

## 16. 面试防守边界

当前项目已经实现：

```text
Megatron 风格基础 TP
Column/Row Parallel 权重加载切分
Q/K/V heads 和 KV cache 按 TP 切分
Row Parallel 后 AllReduce
NCCL/MPI/Mock 通信抽象
F16/BF16/F32 多精度通信 dtype
```

当前项目没有完整实现：

```text
Sequence Parallel 主路径
TP scheduler / distributed executor 级别的复杂调度
QKV heads 不整除时的复杂复制/不均匀切分
AllGather/ReduceScatter 在 Qwen2 forward 主路径中的实际使用
通信与计算 overlap
严谨的 compute stream 与 NCCL stream event 依赖
```

一句话总结：

> 我的 TP 实现是基础 Megatron 风格：加载时切权重，Column 输出 shard 本地继续算，Row 输出 partial hidden 后 AllReduce。通信层通过统一 Comm 接口接 NCCL/MPI/Mock，Qwen2 主路径实际核心通信是 AllReduce。

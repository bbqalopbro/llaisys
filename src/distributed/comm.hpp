// ============================================================================
// comm.hpp — 分布式通信抽象层
// 定义了 Comm 基类接口, 所有通信后端 (Mock/NCCL/MPI) 都继承此接口
// 用途: Tensor Parallelism (张量并行) 中多卡间的 allReduce 归约操作
//
// Phase 1 改造: 新增 CommDataType 枚举, allReduceSum 支持 FP16/BF16/FP32
//   - 保留旧 allReduceSum(float*, count) 作为兼容接口 (内部调用新接口)
//   - 新增 allReduceSum(void*, count, CommDataType) 支持多精度通信
//   - TP 模式不再需要强制 FP32, 可直接 FP16 AllReduce, 带宽翻倍
// ============================================================================
#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace llaisys::distributed {

// 通信后端枚举: 三种实现可选
enum class Backend {
    Mock = 0,   // 单卡测试用, allReduce 是空操作
    Nccl = 1,   // NVIDIA GPU 间高速通信 (NVLink/PCIe 直通)
    Mpi = 2,    // 通用多进程通信 (GPU 数据需经 CPU 中转)
};

// 通信数据类型枚举: 指定 allReduce 等操作的元素精度
// 与 NCCL 的 ncclFloat32/ncclFloat16/ncclBfloat16 一一对应
enum class CommDataType {
    F32 = 0,    // 32-bit 浮点 (4 字节/元素)
    F16 = 1,    // 16-bit 半精度浮点 (2 字节/元素, 带宽利用率翻倍)
    BF16 = 2,   // 16-bit BFloat16 (2 字节/元素, 与 FP32 指数范围相同)
};

// 获取 CommDataType 对应的每个元素字节数
inline size_t commDataTypeSize(CommDataType dtype) {
    switch (dtype) {
        case CommDataType::F32:  return 4;
        case CommDataType::F16:  return 2;
        case CommDataType::BF16: return 2;
        default:                 return 4;
    }
}

// CommDataType 转字符串 (用于日志)
inline const char* commDataTypeToString(CommDataType dtype) {
    switch (dtype) {
        case CommDataType::F32:  return "F32";
        case CommDataType::F16:  return "F16";
        case CommDataType::BF16: return "BF16";
        default:                 return "UNKNOWN";
    }
}

// 通信配置: 创建 Comm 时传入
struct Config {
    Backend backend;       // 使用哪种后端
    int world_size;        // 总进程数 (即总 GPU 数)
    int rank;              // 当前进程编号 [0, world_size)
    int local_device;      // 当前进程绑定的 GPU 设备号

    // Phase 4: 多节点 NCCL 支持
    // 跨节点场景下, rank 0 通过 TCP 将 ncclUniqueId 分发给其他 rank
    // 兼容 PyTorch Distributed 环境变量: MASTER_ADDR, MASTER_PORT
    std::string master_addr;  // rank 0 的 IP 地址 (空字符串 = 单机文件模式)
    int master_port = 29400;  // rank 0 的 TCP 端口 (默认 29400, 避免与 PyTorch 29500 冲突)
};

// 通信接口基类 — 纯虚函数, 由具体后端实现
class Comm {
public:
    virtual ~Comm() = default;

    virtual Backend backend() const = 0;    // 返回后端类型
    virtual int worldSize() const = 0;      // 返回总进程数
    virtual int rank() const = 0;           // 返回当前 rank

    // ══════════════════════════════════════════════════════════════
    // AllReduce: 就地全归约求和 (Phase 1 已实现)
    // ══════════════════════════════════════════════════════════════
    // data[i] = sum(data[i] on all ranks)
    // 用途: Row-Parallel 层后聚合各 rank 的部分和
    virtual void allReduceSum(void *data, size_t count, CommDataType dtype) = 0;

    // 兼容接口: 旧的 float* 签名, 内部转发到新接口
    virtual void allReduceSum(float *data, size_t count) {
        allReduceSum(static_cast<void*>(data), count, CommDataType::F32);
    }

    // ══════════════════════════════════════════════════════════════
    // AllGather: 全收集 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // 每个 rank 提供 sendcount 个元素 (sendbuf)
    // 所有 rank 收到拼接后的结果 (recvbuf), 大小 = sendcount * world_size
    // recvbuf 布局: [rank0_data | rank1_data | ... | rankN_data]
    //
    // 用途: Sequence Parallel 中将分片的激活拼接为完整序列
    //   例: LayerNorm 后各 rank 持有 seq_len/tp_size 的片段
    //       AllGather 后每个 rank 得到完整 seq_len 的序列
    virtual void allGather(const void *sendbuf, void *recvbuf, size_t sendcount,
                           CommDataType dtype) = 0;

    // ══════════════════════════════════════════════════════════════
    // ReduceScatter: 归约后分发 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // 先对所有 rank 的 sendbuf 做逐元素求和 (Reduce)
    // 然后将结果均分给各 rank (Scatter), 每个 rank 得到 recvcount 个元素
    // sendbuf 大小 = recvcount * world_size
    //
    // 用途: Sequence Parallel 中替代 AllReduce
    //   AllReduce = ReduceScatter + AllGather (可拆分流水)
    //   SP 只需 ReduceScatter → 各 rank 处理自己的分片 → AllGather
    virtual void reduceScatter(const void *sendbuf, void *recvbuf, size_t recvcount,
                               CommDataType dtype) = 0;

    // ══════════════════════════════════════════════════════════════
    // Broadcast: 广播 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // root rank 的 data 广播到所有其他 rank (就地操作)
    // 非 root rank 的 data 被覆盖为 root 的数据
    //
    // 用途:
    //   1. 权重同步: rank 0 的权重广播给所有 rank
    //   2. 采样结果广播: rank 0 做 sampling 后广播 token_id
    //   3. 替代 TCP 同步: 用 broadcast 传递控制信令 (Phase 3 计划)
    virtual void broadcast(void *data, size_t count, CommDataType dtype, int root) = 0;

    // ══════════════════════════════════════════════════════════════
    // Send/Recv: 点对点通信 (Phase 2 新增, 为 PP 预留)
    // ══════════════════════════════════════════════════════════════
    // send: 将 data 发送到 dst rank
    // recv: 从 src rank 接收数据到 data
    //
    // 用途: Pipeline Parallelism 中 stage 间的激活传递
    //   stage i 的输出 send 到 stage i+1
    //   stage i+1 从 stage i recv 输入
    virtual void send(const void *data, size_t count, CommDataType dtype, int dst) = 0;
    virtual void recv(void *data, size_t count, CommDataType dtype, int src) = 0;

    // 路障同步: 所有 rank 阻塞直到全部到达此点
    virtual void barrier() = 0;
};

// 智能指针别名, 工厂函数返回此类型
using comm_t = std::shared_ptr<Comm>;

// 工厂函数: 根据 config.backend 创建对应的 Comm 实例
comm_t createComm(const Config &config);

// 查询编译时是否启用了指定后端
bool backendAvailable(Backend backend);

// 后端枚举转字符串 (用于日志)
const char *backendToString(Backend backend);

} // namespace llaisys::distributed

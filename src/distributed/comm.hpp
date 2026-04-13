// ============================================================================
// comm.hpp — 分布式通信抽象层
// 定义了 Comm 基类接口, 所有通信后端 (Mock/NCCL/MPI) 都继承此接口
// 用途: Tensor Parallelism (张量并行) 中多卡间的 allReduce 归约操作
// ============================================================================
#pragma once

#include <cstddef>
#include <memory>

namespace llaisys::distributed {

// 通信后端枚举: 三种实现可选
enum class Backend {
    Mock = 0,   // 单卡测试用, allReduce 是空操作
    Nccl = 1,   // NVIDIA GPU 间高速通信 (NVLink/PCIe 直通)
    Mpi = 2,    // 通用多进程通信 (GPU 数据需经 CPU 中转)
};

// 通信配置: 创建 Comm 时传入
struct Config {
    Backend backend;       // 使用哪种后端
    int world_size;        // 总进程数 (即总 GPU 数)
    int rank;              // 当前进程编号 [0, world_size)
    int local_device;      // 当前进程绑定的 GPU 设备号
};

// 通信接口基类 — 纯虚函数, 由具体后端实现
class Comm {
public:
    virtual ~Comm() = default;

    virtual Backend backend() const = 0;    // 返回后端类型
    virtual int worldSize() const = 0;      // 返回总进程数
    virtual int rank() const = 0;           // 返回当前 rank

    // 就地全归约求和: data[0..count-1] 在所有 rank 间求和, 结果覆盖原数据
    // 这是张量并行中每个 Transformer 层调用 2 次的核心操作
    virtual void allReduceSum(float *data, size_t count) = 0;

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

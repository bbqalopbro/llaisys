// ============================================================================
// distributed.cc — 分布式通信 C API 包装层
// 将 C++ Comm 接口暴露为 C API, 供 Python ctypes 调用
// 设计模式: 不透明句柄 (opaque handle)
//   LlaisysDistComm 结构体内含 shared_ptr<Comm>, 外部只看到指针
//   Python 端通过 ctypes 持有此指针, 调用 Create/Destroy 管理生命周期
// ============================================================================
#include "llaisys/distributed.h"

#include "../distributed/comm.hpp"

#include <memory>
#include <stdexcept>

// 不透明句柄: 外部只看到 LlaisysDistComm*, 不知道内部结构
struct LlaisysDistComm {
    llaisys::distributed::comm_t impl;  // shared_ptr<Comm>
};

// C 枚举 → C++ 枚举转换
static llaisys::distributed::Backend _to_backend(llaisysDistBackend_t backend) {
    switch (backend) {
        case LLAISYS_DIST_BACKEND_MOCK:
            return llaisys::distributed::Backend::Mock;
        case LLAISYS_DIST_BACKEND_NCCL:
            return llaisys::distributed::Backend::Nccl;
        case LLAISYS_DIST_BACKEND_MPI:
            return llaisys::distributed::Backend::Mpi;
        default:
            throw std::invalid_argument("unknown dist backend");
    }
}

// C++ 枚举 → C 枚举转换
static llaisysDistBackend_t _from_backend(llaisys::distributed::Backend backend) {
    switch (backend) {
        case llaisys::distributed::Backend::Mock:
            return LLAISYS_DIST_BACKEND_MOCK;
        case llaisys::distributed::Backend::Nccl:
            return LLAISYS_DIST_BACKEND_NCCL;
        case llaisys::distributed::Backend::Mpi:
            return LLAISYS_DIST_BACKEND_MPI;
        default:
            throw std::invalid_argument("unknown dist backend");
    }
}

// 创建通信句柄: 分配 LlaisysDistComm, 调用工厂函数创建 Comm
__C llaisysDistComm_t llaisysDistCommCreate(struct LlaisysDistConfig config) {
    auto handle = new LlaisysDistComm;
    llaisys::distributed::Config cfg;
    cfg.backend = _to_backend(config.backend);
    cfg.world_size = config.world_size;
    cfg.rank = config.rank;
    cfg.local_device = config.local_device;
    handle->impl = llaisys::distributed::createComm(cfg);
    return handle;
}

// 销毁通信句柄: delete 会触发 shared_ptr 析构, 释放 Comm
__C void llaisysDistCommDestroy(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        return;
    }
    delete comm;
}

// 以下为属性查询接口, Python 端用于验证通信状态
__C llaisysDistBackend_t llaisysDistCommBackend(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    return _from_backend(comm->impl->backend());
}

__C int llaisysDistCommWorldSize(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    return comm->impl->worldSize();
}

__C int llaisysDistCommRank(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    return comm->impl->rank();
}

// 查询编译时后端可用性 (Python 端用于自动选择后端)
__C int llaisysDistBackendAvailable(llaisysDistBackend_t backend) {
    return llaisys::distributed::backendAvailable(_to_backend(backend)) ? 1 : 0;
}

// 核心操作: 就地 allReduce 求和 (张量并行中每层调用 2 次)
__C void llaisysDistAllReduceSumF32(llaisysDistComm_t comm, float *data, size_t count) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    comm->impl->allReduceSum(data, count);
}

// 路障同步
__C void llaisysDistBarrier(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    comm->impl->barrier();
}

// 获取内部 Comm 指针 (高级用法: C++ 内部跨模块传递 shared_ptr)
__C void *llaisysDistCommGetImplPtr(llaisysDistComm_t comm) {
    if (comm == nullptr) return nullptr;
    return &comm->impl;
}

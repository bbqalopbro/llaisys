// ============================================================================
// distributed.cc — 分布式通信 C API 包装层
// 将 C++ Comm 接口暴露为 C API, 供 Python ctypes 调用
// 设计模式: 不透明句柄 (opaque handle)
//   LlaisysDistComm 结构体内含 shared_ptr<Comm>, 外部只看到指针
//   Python 端通过 ctypes 持有此指针, 调用 Create/Destroy 管理生命周期
// ============================================================================
#include "llaisys/distributed.h"

#include "../distributed/comm.hpp"
#include "../distributed/tp_sync.hpp"

#include <cstdlib>
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

// C 数据类型枚举 → C++ CommDataType 转换 (Phase 1 新增)
static llaisys::distributed::CommDataType _to_comm_dtype(llaisysDistDataType_t dtype) {
    switch (dtype) {
        case LLAISYS_DIST_DTYPE_F32:
            return llaisys::distributed::CommDataType::F32;
        case LLAISYS_DIST_DTYPE_F16:
            return llaisys::distributed::CommDataType::F16;
        case LLAISYS_DIST_DTYPE_BF16:
            return llaisys::distributed::CommDataType::BF16;
        default:
            throw std::invalid_argument("unknown dist data type");
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

    // Phase 4: 多节点配置
    // 优先使用 C API 传入的地址; 若为空, 回退到环境变量 MASTER_ADDR/MASTER_PORT
    if (config.master_addr && config.master_addr[0] != '\0') {
        cfg.master_addr = config.master_addr;
    } else {
        const char *env_addr = std::getenv("MASTER_ADDR");
        if (env_addr && env_addr[0] != '\0') {
            cfg.master_addr = env_addr;
        }
        // 否则 master_addr 为空, 走单机文件模式
    }
    if (config.master_port > 0) {
        cfg.master_port = config.master_port;
    } else {
        const char *env_port = std::getenv("MASTER_PORT");
        if (env_port && env_port[0] != '\0') {
            cfg.master_port = std::atoi(env_port);
        }
        // 否则使用 Config 默认值 29400
    }

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

// Phase 1 新增: 多数据类型 AllReduce
// 支持 F32/F16/BF16, NCCL 后端直接映射, MPI 后端走 FP32 中转
__C void llaisysDistAllReduceSum(llaisysDistComm_t comm, void *data, size_t count, llaisysDistDataType_t dtype) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    comm->impl->allReduceSum(data, count, _to_comm_dtype(dtype));
}

// 路障同步
__C void llaisysDistBarrier(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    comm->impl->barrier();
}

// ══════════════════════════════════════════════════════════════
// Phase 2 新增: 通信原语 C API 实现
// ══════════════════════════════════════════════════════════════

// AllGather: 全收集
__C void llaisysDistAllGather(llaisysDistComm_t comm, const void *sendbuf, void *recvbuf,
                               size_t sendcount, llaisysDistDataType_t dtype) {
    if (comm == nullptr) throw std::invalid_argument("comm is null");
    comm->impl->allGather(sendbuf, recvbuf, sendcount, _to_comm_dtype(dtype));
}

// ReduceScatter: 归约后分发
__C void llaisysDistReduceScatter(llaisysDistComm_t comm, const void *sendbuf, void *recvbuf,
                                    size_t recvcount, llaisysDistDataType_t dtype) {
    if (comm == nullptr) throw std::invalid_argument("comm is null");
    comm->impl->reduceScatter(sendbuf, recvbuf, recvcount, _to_comm_dtype(dtype));
}

// Broadcast: 广播
__C void llaisysDistBroadcast(llaisysDistComm_t comm, void *data, size_t count,
                               llaisysDistDataType_t dtype, int root) {
    if (comm == nullptr) throw std::invalid_argument("comm is null");
    comm->impl->broadcast(data, count, _to_comm_dtype(dtype), root);
}

// Send: 点对点发送
__C void llaisysDistSend(llaisysDistComm_t comm, const void *data, size_t count,
                           llaisysDistDataType_t dtype, int dst) {
    if (comm == nullptr) throw std::invalid_argument("comm is null");
    comm->impl->send(data, count, _to_comm_dtype(dtype), dst);
}

// Recv: 点对点接收
__C void llaisysDistRecv(llaisysDistComm_t comm, void *data, size_t count,
                           llaisysDistDataType_t dtype, int src) {
    if (comm == nullptr) throw std::invalid_argument("comm is null");
    comm->impl->recv(data, count, _to_comm_dtype(dtype), src);
}

// 获取内部 Comm 指针 (高级用法: C++ 内部跨模块传递 shared_ptr)
__C void *llaisysDistCommGetImplPtr(llaisysDistComm_t comm) {
    if (comm == nullptr) return nullptr;
    return &comm->impl;
}

// ══════════════════════════════════════════════════════════════
// Phase 3: TP 同步控制器 C API
// ══════════════════════════════════════════════════════════════

// TPSync 不透明句柄
struct LlaisysTPSync {
    std::unique_ptr<llaisys::distributed::TPSyncController> impl;
};

__C llaisysTPSync_t llaisysTPSyncCreate(llaisysDistComm_t comm, size_t max_payload) {
    if (comm == nullptr) throw std::invalid_argument("comm is null");
    auto handle = new LlaisysTPSync;
    handle->impl = std::make_unique<llaisys::distributed::TPSyncController>(
        comm->impl, max_payload > 0 ? max_payload : 8192
    );
    return handle;
}

__C void llaisysTPSyncDestroy(llaisysTPSync_t sync) {
    if (sync) delete sync;
}

__C void llaisysTPSyncFillPrefill(llaisysTPSync_t sync, int slot_id,
                                    const int *token_ids, size_t n_tokens,
                                    float temperature, int top_k, float top_p) {
    if (!sync) throw std::invalid_argument("sync is null");
    sync->impl->fillPrefillCommand(slot_id, token_ids, n_tokens, temperature, top_k, top_p);
}

__C void llaisysTPSyncFillDecode(llaisysTPSync_t sync,
                                   const int *active_slots, const int *current_tokens,
                                   int batch_size,
                                   float temperature, int top_k, float top_p) {
    if (!sync) throw std::invalid_argument("sync is null");
    sync->impl->fillDecodeCommand(active_slots, current_tokens, batch_size, temperature, top_k, top_p);
}

__C void llaisysTPSyncFillSlotReset(llaisysTPSync_t sync, int slot_id) {
    if (!sync) throw std::invalid_argument("sync is null");
    sync->impl->fillSlotResetCommand(slot_id);
}

__C void llaisysTPSyncFillShutdown(llaisysTPSync_t sync) {
    if (!sync) throw std::invalid_argument("sync is null");
    sync->impl->fillShutdownCommand();
}

__C void llaisysTPSyncBroadcast(llaisysTPSync_t sync) {
    if (!sync) throw std::invalid_argument("sync is null");
    sync->impl->broadcastSync();
}

__C int llaisysTPSyncGetCmd(llaisysTPSync_t sync) {
    if (!sync) throw std::invalid_argument("sync is null");
    return static_cast<int>(sync->impl->cmd());
}

__C int llaisysTPSyncGetSlotId(llaisysTPSync_t sync) {
    if (!sync) return -1;
    return sync->impl->slotId();
}

__C int llaisysTPSyncGetBatchSize(llaisysTPSync_t sync) {
    if (!sync) return 0;
    return sync->impl->batchSize();
}

__C int llaisysTPSyncGetPayloadCount(llaisysTPSync_t sync) {
    if (!sync) return 0;
    return sync->impl->payloadCount();
}

__C float llaisysTPSyncGetTemperature(llaisysTPSync_t sync) {
    if (!sync) return 0.0f;
    return sync->impl->temperature();
}

__C int llaisysTPSyncGetTopK(llaisysTPSync_t sync) {
    if (!sync) return 0;
    return sync->impl->topK();
}

__C float llaisysTPSyncGetTopP(llaisysTPSync_t sync) {
    if (!sync) return 0.0f;
    return sync->impl->topP();
}

__C const int *llaisysTPSyncGetPayload(llaisysTPSync_t sync) {
    if (!sync) return nullptr;
    return sync->impl->payload();
}

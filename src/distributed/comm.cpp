// ============================================================================
// comm.cpp — 通信后端工厂函数实现
// 使用条件编译 (#ifdef) 控制后端可用性:
//   ENABLE_DIST_NCCL → 编译 NCCL 后端
//   ENABLE_DIST_MPI  → 编译 MPI 后端
//   Mock 始终可用
// ============================================================================
#include "comm.hpp"

#include <stdexcept>

namespace llaisys::distributed {

// 前向声明: 各后端的创建函数, 实现在各自的 .cpp/.cu 中
std::shared_ptr<Comm> createMockComm(const Config &config);

#ifdef ENABLE_DIST_NCCL
std::shared_ptr<Comm> createNcclComm(const Config &config);
#endif

#ifdef ENABLE_DIST_MPI
std::shared_ptr<Comm> createMpiComm(const Config &config);
#endif

// 工厂函数: 根据 config.backend 分发到具体创建函数
std::shared_ptr<Comm> createComm(const Config &config) {
    switch (config.backend) {
        case Backend::Mock:
            return createMockComm(config);
        case Backend::Nccl:
#ifdef ENABLE_DIST_NCCL
            return createNcclComm(config);
#else
            throw std::runtime_error("NCCL backend is disabled at compile-time");
#endif
        case Backend::Mpi:
#ifdef ENABLE_DIST_MPI
            return createMpiComm(config);
#else
            throw std::runtime_error("MPI backend is disabled at compile-time");
#endif
        default:
            throw std::invalid_argument("unknown backend");
    }
}

// 查询后端是否在编译时启用
bool backendAvailable(Backend backend) {
    switch (backend) {
        case Backend::Mock:
            return true;  // Mock 永远可用
        case Backend::Nccl:
#ifdef ENABLE_DIST_NCCL
            return true;
#else
            return false;
#endif
        case Backend::Mpi:
#ifdef ENABLE_DIST_MPI
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

// 后端名字符串 (用于日志输出)
const char *backendToString(Backend backend) {
    switch (backend) {
        case Backend::Mock:
            return "mock";
        case Backend::Nccl:
            return "nccl";
        case Backend::Mpi:
            return "mpi";
        default:
            return "unknown";
    }
}
} // namespace llaisys::distributed

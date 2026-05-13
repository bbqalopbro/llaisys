// ============================================================================
// nccl_comm.cu — NCCL GPU-Direct 通信后端
// 使用 ncclAllReduce 进行 GPU 多卡集合通信 (数据始终在 GPU 上, 零 CPU 拷贝)
// 特点:
//   1. ncclUniqueId 通过文件系统共享 (rank 0 写, 其他 rank 轮询读)
//   2. 专用 CUDA stream 避免与计算 stream 冲突
//   3. barrier 用单元素 allReduce 模拟 (NCCL 无原生 barrier)
//
// Phase 1 改造: 新增 CommDataType → ncclDataType_t 映射
//   - allReduceSum(void*, count, CommDataType) 直接调用 ncclAllReduce
//   - FP16 使用 ncclFloat16, BF16 使用 ncclBfloat16
//   - 无需任何数据拷贝或类型转换, NCCL 原生支持这些精度
// ============================================================================

#include "comm.hpp"

#ifdef ENABLE_DIST_NCCL

#include "tcp_id_store.hpp"

#include <nccl.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

namespace llaisys::distributed {

// CommDataType → ncclDataType_t 映射
// NCCL 原生支持 FP32/FP16/BF16, 无需任何手动转换
static ncclDataType_t toNcclDataType(CommDataType dtype) {
    switch (dtype) {
        case CommDataType::F32:  return ncclFloat32;
        case CommDataType::F16:  return ncclFloat16;
        case CommDataType::BF16: return ncclBfloat16;
        default:
            throw std::runtime_error("Unsupported CommDataType for NCCL");
    }
}

// NCCL 错误检查宏
#define NCCL_CHECK(cmd) do {                                               \
    ncclResult_t r = (cmd);                                                \
    if (r != ncclSuccess) {                                                \
        char msg[256];                                                     \
        snprintf(msg, sizeof(msg), "NCCL error at %s:%d: %s",             \
                 __FILE__, __LINE__, ncclGetErrorString(r));               \
        throw std::runtime_error(msg);                                     \
    }                                                                      \
} while(0)

#define CUDA_CHECK(cmd) do {                                               \
    cudaError_t e = (cmd);                                                 \
    if (e != cudaSuccess) {                                                \
        char msg[256];                                                     \
        snprintf(msg, sizeof(msg), "CUDA error at %s:%d: %s",             \
                 __FILE__, __LINE__, cudaGetErrorString(e));               \
        throw std::runtime_error(msg);                                     \
    }                                                                      \
} while(0)

// ncclUniqueId 在进程间通过文件共享
// ID 文件路径: 环境变量 LLAISYS_NCCL_ID_FILE 或默认 /tmp/llaisys_nccl_id
static std::string getNcclIdFilePath() {
    const char *env = std::getenv("LLAISYS_NCCL_ID_FILE");
    if (env && env[0] != '\0') return env;
    return "/tmp/llaisys_nccl_id";
}

static void writeNcclId(const ncclUniqueId &id, const std::string &path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot write NCCL unique ID to " + path);
    }
    f.write(id.internal, NCCL_UNIQUE_ID_BYTES);
    f.close();
}

static ncclUniqueId readNcclId(const std::string &path, int max_wait_ms = 30000) {
    ncclUniqueId id;
    int waited = 0;
    const int poll_ms = 100;
    while (waited < max_wait_ms) {
        std::ifstream f(path, std::ios::binary);
        if (f.is_open()) {
            f.read(id.internal, NCCL_UNIQUE_ID_BYTES);
            if (f.gcount() == NCCL_UNIQUE_ID_BYTES) {
                return id;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        waited += poll_ms;
    }
    throw std::runtime_error("Timeout waiting for NCCL unique ID at " + path);
}

class NcclComm : public Comm {
private:
    ncclComm_t _nccl_comm;
    cudaStream_t _stream;
    int _world_size;
    int _rank;
    int _device;

public:
    explicit NcclComm(const Config &config)
        : _nccl_comm(nullptr),
          _stream(nullptr),
          _world_size(config.world_size),
          _rank(config.rank),
          _device(config.local_device) {

        if (_world_size <= 0) {
            throw std::invalid_argument("world_size must be > 0");
        }
        if (_rank < 0 || _rank >= _world_size) {
            throw std::invalid_argument("rank out of range");
        }

        // 设置 CUDA 设备
        CUDA_CHECK(cudaSetDevice(_device));

        // 创建专用 CUDA stream
        CUDA_CHECK(cudaStreamCreate(&_stream));

        // 获取或接收 NCCL unique ID
        ncclUniqueId nccl_id;

        if (_world_size == 1) {
            // 单卡模式: 直接初始化, 无需跨进程同步
            NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        } else if (!config.master_addr.empty()) {
            // ── Phase 4: 多节点 TCP 模式 ──────────────────────────
            // 通过 TCP socket 分发 ncclUniqueId, 支持跨机器
            // 兼容 PyTorch Distributed 模式 (MASTER_ADDR/MASTER_PORT)
            if (_rank == 0) {
                NCCL_CHECK(ncclGetUniqueId(&nccl_id));
                fprintf(stderr, "[NCCL] rank 0: generated unique ID, "
                                "distributing via TCP %s:%d\n",
                        config.master_addr.c_str(), config.master_port);
            }
            // rank 0 分发, 其他 rank 接收
            tcpExchangeNcclId(nccl_id.internal, _rank, _world_size,
                              config.master_addr, config.master_port);
            if (_rank != 0) {
                fprintf(stderr, "[NCCL] rank %d: received unique ID via TCP "
                                "from %s:%d\n",
                        _rank, config.master_addr.c_str(), config.master_port);
            }
        } else {
            // ── 传统单机文件模式 ──────────────────────────────────
            // rank 0 写文件, 其他 rank 轮询读取 (仅限共享文件系统)
            std::string id_file = getNcclIdFilePath();
            if (_rank == 0) {
                NCCL_CHECK(ncclGetUniqueId(&nccl_id));
                writeNcclId(nccl_id, id_file);
                fprintf(stderr, "[NCCL] rank 0: generated unique ID → %s\n", id_file.c_str());
            } else {
                nccl_id = readNcclId(id_file);
                fprintf(stderr, "[NCCL] rank %d: read unique ID ← %s\n", _rank, id_file.c_str());
            }
        }

        // 初始化 NCCL communicator (阻塞集合操作, 所有 rank 需同时调用)
        NCCL_CHECK(ncclCommInitRank(&_nccl_comm, _world_size, nccl_id, _rank));

        fprintf(stderr, "[NCCL] rank %d/%d initialized on GPU %d\n",
                _rank, _world_size, _device);
    }

    ~NcclComm() override {
        if (_nccl_comm) {
            ncclCommDestroy(_nccl_comm);
        }
        if (_stream) {
            cudaStreamDestroy(_stream);
        }
    }

    Backend backend() const override { return Backend::Nccl; }
    int worldSize() const override { return _world_size; }
    int rank() const override { return _rank; }

    // 多数据类型就地全归约求和 (Phase 1 核心改造)
    // data: GPU 显存上的缓冲区指针 (类型由 dtype 指定)
    // count: 元素个数 (不是字节数)
    // dtype: F32/F16/BF16 → 映射到 ncclFloat32/ncclFloat16/ncclBfloat16
    //
    // FP16 AllReduce 相比 FP32:
    //   - 传输数据量减半 → NVLink/PCIe 带宽利用率翻倍
    //   - NCCL 内部 ring/tree reduce 开销也减半
    //   - 精度损失在推理场景可忽略 (FP16 已是标准推理精度)
    void allReduceSum(void *data, size_t count, CommDataType dtype) override {
        if (count == 0) return;

        // 确保在正确的设备上操作
        CUDA_CHECK(cudaSetDevice(_device));

        // 转换数据类型枚举
        ncclDataType_t nccl_dtype = toNcclDataType(dtype);

        // 就地 allReduce (sendbuf == recvbuf)
        NCCL_CHECK(ncclAllReduce(
            data, data, count,
            nccl_dtype, ncclSum,
            _nccl_comm, _stream));

        // 同步等待完成 (当前架构是同步调用)
        CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    void barrier() override {
        // NCCL 没有原生 barrier, 用一个单元素 allReduce 模拟
        CUDA_CHECK(cudaSetDevice(_device));

        float *dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&dummy, sizeof(float)));
        CUDA_CHECK(cudaMemset(dummy, 0, sizeof(float)));

        NCCL_CHECK(ncclAllReduce(
            dummy, dummy, 1,
            ncclFloat32, ncclSum,
            _nccl_comm, _stream));

        CUDA_CHECK(cudaStreamSynchronize(_stream));
        CUDA_CHECK(cudaFree(dummy));
    }

    // ══════════════════════════════════════════════════════════════
    // AllGather: 全收集 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // NCCL ncclAllGather:
    //   每个 rank 的 sendbuf[0..sendcount-1] → recvbuf 中按 rank 顺序拼接
    //   recvbuf 大小 = sendcount * world_size
    //   数据始终在 GPU 显存上, 零 CPU 拷贝
    void allGather(const void *sendbuf, void *recvbuf, size_t sendcount,
                   CommDataType dtype) override {
        if (sendcount == 0) return;
        CUDA_CHECK(cudaSetDevice(_device));
        ncclDataType_t nccl_dtype = toNcclDataType(dtype);

        NCCL_CHECK(ncclAllGather(
            sendbuf, recvbuf, sendcount,
            nccl_dtype,
            _nccl_comm, _stream));

        CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    // ══════════════════════════════════════════════════════════════
    // ReduceScatter: 归约后分发 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // NCCL ncclReduceScatter:
    //   所有 rank 的 sendbuf 逐元素求和, 结果均分给各 rank
    //   sendbuf 大小 = recvcount * world_size
    //   每个 rank 的 recvbuf 获得总和的第 rank 个分片
    //
    // Sequence Parallel 用法:
    //   AllReduce 可拆分为 ReduceScatter + AllGather
    //   SP 在 ReduceScatter 后各 rank 独立做 LayerNorm/Dropout
    //   最后再 AllGather 拼接 → 激活显存降为 1/tp_size
    void reduceScatter(const void *sendbuf, void *recvbuf, size_t recvcount,
                       CommDataType dtype) override {
        if (recvcount == 0) return;
        CUDA_CHECK(cudaSetDevice(_device));
        ncclDataType_t nccl_dtype = toNcclDataType(dtype);

        NCCL_CHECK(ncclReduceScatter(
            sendbuf, recvbuf, recvcount,
            nccl_dtype, ncclSum,
            _nccl_comm, _stream));

        CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    // ══════════════════════════════════════════════════════════════
    // Broadcast: 广播 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // NCCL ncclBroadcast (等价于旧版 ncclBcast):
    //   root rank 的 data → 所有 rank 的 data (就地覆盖)
    //   NVLink 上延迟极低 (~5μs), 可替代 TCP 同步控制信号
    void broadcast(void *data, size_t count, CommDataType dtype, int root) override {
        if (count == 0) return;
        CUDA_CHECK(cudaSetDevice(_device));
        ncclDataType_t nccl_dtype = toNcclDataType(dtype);

        // ncclBroadcast: sendbuf=recvbuf 时为就地操作
        NCCL_CHECK(ncclBroadcast(
            data, data, count,
            nccl_dtype, root,
            _nccl_comm, _stream));

        CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    // ══════════════════════════════════════════════════════════════
    // Send/Recv: 点对点通信 (Phase 2 新增, 为 PP 预留)
    // ══════════════════════════════════════════════════════════════
    // NCCL ncclSend/ncclRecv:
    //   GPU-Direct P2P 传输, 同一 NVLink 域内延迟极低
    //   必须成对调用: rank A 调 send(dst=B) 同时 rank B 调 recv(src=A)
    //   否则会死锁 (NCCL 的 P2P 语义要求配对)
    //
    // PP 用法:
    //   stage i: send(hidden_states, count, F16, next_stage_rank)
    //   stage i+1: recv(hidden_states, count, F16, prev_stage_rank)
    void send(const void *data, size_t count, CommDataType dtype, int dst) override {
        if (count == 0) return;
        CUDA_CHECK(cudaSetDevice(_device));
        ncclDataType_t nccl_dtype = toNcclDataType(dtype);

        NCCL_CHECK(ncclSend(
            data, count, nccl_dtype,
            dst, _nccl_comm, _stream));

        CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    void recv(void *data, size_t count, CommDataType dtype, int src) override {
        if (count == 0) return;
        CUDA_CHECK(cudaSetDevice(_device));
        ncclDataType_t nccl_dtype = toNcclDataType(dtype);

        NCCL_CHECK(ncclRecv(
            data, count, nccl_dtype,
            src, _nccl_comm, _stream));

        CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
};

std::shared_ptr<Comm> createNcclComm(const Config &config) {
    return std::make_shared<NcclComm>(config);
}

} // namespace llaisys::distributed

#endif // ENABLE_DIST_NCCL

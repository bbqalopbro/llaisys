// ============================================================================
// nccl_comm.cu — NCCL GPU-Direct 通信后端
// 使用 ncclAllReduce 进行 GPU 多卡集合通信 (数据始终在 GPU 上, 零 CPU 拷贝)
// 特点:
//   1. ncclUniqueId 通过文件系统共享 (rank 0 写, 其他 rank 轮询读)
//   2. 专用 CUDA stream 避免与计算 stream 冲突
//   3. barrier 用单元素 allReduce 模拟 (NCCL 无原生 barrier)
// ============================================================================

#include "comm.hpp"

#ifdef ENABLE_DIST_NCCL

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
            // 单卡模式: 直接初始化
            NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        } else {
            // 多卡模式: rank 0 生成并写文件, 其他 rank 从文件读取
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

    void allReduceSum(float *data, size_t count) override {
        if (count == 0) return;

        // 确保在正确的设备上操作
        CUDA_CHECK(cudaSetDevice(_device));

        // 就地 allReduce (sendbuf == recvbuf)
        NCCL_CHECK(ncclAllReduce(
            data, data, count,
            ncclFloat32, ncclSum,
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
};

std::shared_ptr<Comm> createNcclComm(const Config &config) {
    return std::make_shared<NcclComm>(config);
}

} // namespace llaisys::distributed

#endif // ENABLE_DIST_NCCL

// ============================================================================
// mpi_comm.cpp — MPI 通信后端
// 使用 MPI_Allreduce 进行多进程集合通信
// 与 NCCL 的关键区别:
//   - MPI 只能操作 CPU 内存 (标准 MPI, 非 GPU-aware MPI)
//   - GPU 数据需要 GPU→CPU→MPI_Allreduce→CPU→GPU 三次拷贝
//   - 因此性能远低于 NCCL 的 GPU-Direct 模式
// 适用场景: 没有 NCCL 时的兼容方案, 或 CPU 推理场景
// ============================================================================

#include "comm.hpp"

#ifdef ENABLE_DIST_MPI

#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef ENABLE_DIST_NCCL
// 如果同时有 NCCL (CUDA), 使用 CUDA API 进行 GPU↔CPU 数据搬运
#include <cuda_runtime.h>

#define CUDA_CHECK(cmd) do {                                               \
    cudaError_t e = (cmd);                                                 \
    if (e != cudaSuccess) {                                                \
        char msg[256];                                                     \
        snprintf(msg, sizeof(msg), "CUDA error at %s:%d: %s",             \
                 __FILE__, __LINE__, cudaGetErrorString(e));               \
        throw std::runtime_error(msg);                                     \
    }                                                                      \
} while(0)

static bool isDevicePointer(const void *ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError(); // clear error
        return false;
    }
    return attr.type == cudaMemoryTypeDevice;
}
#endif // ENABLE_DIST_NCCL

namespace llaisys::distributed {

// 全局 MPI 初始化状态追踪 (MPI_Init 只能调用一次)
static bool g_mpi_initialized_by_us = false;

class MpiComm : public Comm {
private:
    MPI_Comm _mpi_comm;
    int _world_size;
    int _rank;
    int _device;
    std::vector<float> _cpu_buf; // CPU 端缓冲区, 复用以避免反复分配

public:
    explicit MpiComm(const Config &config)
        : _mpi_comm(MPI_COMM_WORLD),
          _world_size(config.world_size),
          _rank(config.rank),
          _device(config.local_device) {

        if (_world_size <= 0) {
            throw std::invalid_argument("world_size must be > 0");
        }
        if (_rank < 0 || _rank >= _world_size) {
            throw std::invalid_argument("rank out of range");
        }

        // MPI_Init (幂等: 只在第一次调用时执行)
        int already_init = 0;
        MPI_Initialized(&already_init);
        if (!already_init) {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            if (provided < MPI_THREAD_MULTIPLE) {
                fprintf(stderr, "[MPI] Warning: MPI_THREAD_MULTIPLE not supported "
                        "(got %d), multi-threaded usage may be unsafe\n", provided);
            }
            g_mpi_initialized_by_us = true;
        }

        // 验证 MPI world 与 config 一致
        int mpi_size, mpi_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        if (mpi_rank != _rank) {
            fprintf(stderr, "[MPI] Warning: config.rank=%d but MPI_Comm_rank=%d, "
                    "using MPI rank\n", _rank, mpi_rank);
            _rank = mpi_rank;
        }
        if (mpi_size != _world_size) {
            fprintf(stderr, "[MPI] Warning: config.world_size=%d but MPI_Comm_size=%d, "
                    "using MPI size\n", _world_size, mpi_size);
            _world_size = mpi_size;
        }

#ifdef ENABLE_DIST_NCCL
        // 设置 CUDA 设备
        CUDA_CHECK(cudaSetDevice(_device));
#endif

        fprintf(stderr, "[MPI] rank %d/%d initialized (device %d)\n",
                _rank, _world_size, _device);
    }

    ~MpiComm() override {
        // 注意: MPI_Finalize 会影响全局, 只在我们自己初始化且尚未 finalize 时调用
        if (g_mpi_initialized_by_us) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Finalize();
                g_mpi_initialized_by_us = false;
            }
        }
    }

    Backend backend() const override { return Backend::Mpi; }
    int worldSize() const override { return _world_size; }
    int rank() const override { return _rank; }

    void allReduceSum(float *data, size_t count) override {
        if (count == 0) return;

#ifdef ENABLE_DIST_NCCL
        // GPU 场景: data 可能在 GPU 内存, MPI 需要 CPU 缓冲区
        if (isDevicePointer(data)) {
            CUDA_CHECK(cudaSetDevice(_device));

            // 确保 CPU 缓冲区足够大
            if (_cpu_buf.size() < count) {
                _cpu_buf.resize(count);
            }

            // GPU → CPU
            CUDA_CHECK(cudaMemcpy(_cpu_buf.data(), data,
                                  count * sizeof(float), cudaMemcpyDeviceToHost));

            // MPI 就地 allreduce
            MPI_Allreduce(MPI_IN_PLACE, _cpu_buf.data(),
                          static_cast<int>(count), MPI_FLOAT, MPI_SUM, _mpi_comm);

            // CPU → GPU
            CUDA_CHECK(cudaMemcpy(data, _cpu_buf.data(),
                                  count * sizeof(float), cudaMemcpyHostToDevice));

            return;
        }
#endif
        // CPU 场景: 直接就地 allreduce
        MPI_Allreduce(MPI_IN_PLACE, data,
                      static_cast<int>(count), MPI_FLOAT, MPI_SUM, _mpi_comm);
    }

    void barrier() override {
        MPI_Barrier(_mpi_comm);
    }
};

std::shared_ptr<Comm> createMpiComm(const Config &config) {
    return std::make_shared<MpiComm>(config);
}

} // namespace llaisys::distributed

#endif // ENABLE_DIST_MPI

// ============================================================================
// mpi_comm.cpp — MPI 通信后端
// 使用 MPI_Allreduce 进行多进程集合通信
// 与 NCCL 的关键区别:
//   - MPI 只能操作 CPU 内存 (标准 MPI, 非 GPU-aware MPI)
//   - GPU 数据需要 GPU→CPU→MPI_Allreduce→CPU→GPU 三次拷贝
//   - 因此性能远低于 NCCL 的 GPU-Direct 模式
// 适用场景: 没有 NCCL 时的兼容方案, 或 CPU 推理场景
//
// Phase 1 改造: 支持 FP16/BF16 数据类型
//   - FP16/BF16 归约: 先转为 FP32 做 MPI_Allreduce, 再转回原精度
//     (标准 MPI 不支持 FP16 的 SUM 操作, 必须走 FP32 中转)
//   - FP32: 直接 MPI_Allreduce (与改造前相同)
//   - GPU 场景: 仍需 GPU↔CPU 拷贝, 但拷贝量按 dtype 大小减半
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

    // 多数据类型就地全归约求和 (Phase 1 改造)
    // MPI 策略:
    //   FP32 → 直接 MPI_Allreduce(MPI_FLOAT, MPI_SUM) [最快]
    //   FP16/BF16 → 拷贝到 FP32 缓冲区 → MPI_Allreduce → 转回原精度
    //     (标准 MPI 没有 FP16 归约操作, 必须走 FP32 中转)
    //     (虽然有额外转换开销, 但 MPI 后端本身就是兼容方案, 性能非首要)
    void allReduceSum(void *data, size_t count, CommDataType dtype) override {
        if (count == 0) return;

        // ── FP32 直通路径 (与改造前完全一致) ──
        if (dtype == CommDataType::F32) {
            allReduceSumF32(static_cast<float*>(data), count);
            return;
        }

        // ── FP16/BF16 路径: 需要 FP32 中转归约 ──
        // 步骤: 原数据 → FP32 → MPI_Allreduce → FP32 → 原精度写回
        size_t elem_size = commDataTypeSize(dtype);  // 2 bytes for F16/BF16

#ifdef ENABLE_DIST_NCCL
        // GPU 场景: data 在 GPU 内存
        if (isDevicePointer(data)) {
            CUDA_CHECK(cudaSetDevice(_device));

            // 1. GPU 原精度 → CPU 临时缓冲区 (按原精度大小拷贝, 节省带宽)
            std::vector<char> cpu_raw(count * elem_size);
            CUDA_CHECK(cudaMemcpy(cpu_raw.data(), data,
                                  count * elem_size, cudaMemcpyDeviceToHost));

            // 2. CPU 上 FP16/BF16 → FP32 展开
            if (_cpu_buf.size() < count) {
                _cpu_buf.resize(count);
            }
            if (dtype == CommDataType::F16) {
                // FP16 → FP32: 逐元素转换
                const uint16_t* src = reinterpret_cast<const uint16_t*>(cpu_raw.data());
                for (size_t i = 0; i < count; i++) {
                    // IEEE 754 FP16 → FP32 硬件转换 (GCC __fp16 扩展)
                    _Float16 val;
                    std::memcpy(&val, &src[i], sizeof(_Float16));
                    _cpu_buf[i] = static_cast<float>(val);
                }
            } else {
                // BF16 → FP32: 左移 16 位即可 (BF16 就是 FP32 截断高 16 位)
                const uint16_t* src = reinterpret_cast<const uint16_t*>(cpu_raw.data());
                for (size_t i = 0; i < count; i++) {
                    uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
                    std::memcpy(&_cpu_buf[i], &bits, sizeof(float));
                }
            }

            // 3. FP32 MPI AllReduce
            MPI_Allreduce(MPI_IN_PLACE, _cpu_buf.data(),
                          static_cast<int>(count), MPI_FLOAT, MPI_SUM, _mpi_comm);

            // 4. FP32 → FP16/BF16 压缩写回
            if (dtype == CommDataType::F16) {
                uint16_t* dst = reinterpret_cast<uint16_t*>(cpu_raw.data());
                for (size_t i = 0; i < count; i++) {
                    _Float16 val = static_cast<_Float16>(_cpu_buf[i]);
                    std::memcpy(&dst[i], &val, sizeof(_Float16));
                }
            } else {
                uint16_t* dst = reinterpret_cast<uint16_t*>(cpu_raw.data());
                for (size_t i = 0; i < count; i++) {
                    uint32_t bits;
                    std::memcpy(&bits, &_cpu_buf[i], sizeof(float));
                    dst[i] = static_cast<uint16_t>(bits >> 16);
                }
            }

            // 5. CPU → GPU 写回 (原精度)
            CUDA_CHECK(cudaMemcpy(data, cpu_raw.data(),
                                  count * elem_size, cudaMemcpyHostToDevice));
            return;
        }
#endif
        // CPU 场景: data 在 CPU 内存
        // 同样需要 FP32 中转 (MPI 不支持 half 归约)
        if (_cpu_buf.size() < count) {
            _cpu_buf.resize(count);
        }

        // 展开为 FP32
        if (dtype == CommDataType::F16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(data);
            for (size_t i = 0; i < count; i++) {
                _Float16 val;
                std::memcpy(&val, &src[i], sizeof(_Float16));
                _cpu_buf[i] = static_cast<float>(val);
            }
        } else {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(data);
            for (size_t i = 0; i < count; i++) {
                uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
                std::memcpy(&_cpu_buf[i], &bits, sizeof(float));
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, _cpu_buf.data(),
                      static_cast<int>(count), MPI_FLOAT, MPI_SUM, _mpi_comm);

        // 压缩写回
        if (dtype == CommDataType::F16) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(data);
            for (size_t i = 0; i < count; i++) {
                _Float16 val = static_cast<_Float16>(_cpu_buf[i]);
                std::memcpy(&dst[i], &val, sizeof(_Float16));
            }
        } else {
            uint16_t* dst = reinterpret_cast<uint16_t*>(data);
            for (size_t i = 0; i < count; i++) {
                uint32_t bits;
                std::memcpy(&_cpu_buf[i], &bits, sizeof(float));
                dst[i] = static_cast<uint16_t>(bits >> 16);
            }
        }
    }

private:
    // FP32 专用路径 (原始逻辑, 无额外开销)
    void allReduceSumF32(float *data, size_t count) {
#ifdef ENABLE_DIST_NCCL
        if (isDevicePointer(data)) {
            CUDA_CHECK(cudaSetDevice(_device));
            if (_cpu_buf.size() < count) {
                _cpu_buf.resize(count);
            }
            CUDA_CHECK(cudaMemcpy(_cpu_buf.data(), data,
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
            MPI_Allreduce(MPI_IN_PLACE, _cpu_buf.data(),
                          static_cast<int>(count), MPI_FLOAT, MPI_SUM, _mpi_comm);
            CUDA_CHECK(cudaMemcpy(data, _cpu_buf.data(),
                                  count * sizeof(float), cudaMemcpyHostToDevice));
            return;
        }
#endif
        MPI_Allreduce(MPI_IN_PLACE, data,
                      static_cast<int>(count), MPI_FLOAT, MPI_SUM, _mpi_comm);
    }

public:

    void barrier() override {
        MPI_Barrier(_mpi_comm);
    }

    // ══════════════════════════════════════════════════════════════
    // AllGather: 全收集 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // MPI_Allgather: 每个 rank 提供 sendcount 个元素 → 合并到所有 rank
    // GPU 数据需要 GPU→CPU→MPI_Allgather→CPU→GPU
    // 注意: MPI 不支持 FP16, 按 MPI_BYTE 传输 (不涉及归约, 只是数据搬运)
    void allGather(const void *sendbuf, void *recvbuf, size_t sendcount,
                   CommDataType dtype) override {
        if (sendcount == 0) return;
        size_t elem_size = commDataTypeSize(dtype);
        size_t send_bytes = sendcount * elem_size;
        size_t recv_bytes = send_bytes * _world_size;

#ifdef ENABLE_DIST_NCCL
        if (isDevicePointer(sendbuf)) {
            CUDA_CHECK(cudaSetDevice(_device));
            // GPU → CPU
            std::vector<char> cpu_send(send_bytes);
            std::vector<char> cpu_recv(recv_bytes);
            CUDA_CHECK(cudaMemcpy(cpu_send.data(), sendbuf, send_bytes, cudaMemcpyDeviceToHost));

            MPI_Allgather(cpu_send.data(), static_cast<int>(send_bytes), MPI_BYTE,
                          cpu_recv.data(), static_cast<int>(send_bytes), MPI_BYTE, _mpi_comm);

            // CPU → GPU
            CUDA_CHECK(cudaMemcpy(recvbuf, cpu_recv.data(), recv_bytes, cudaMemcpyHostToDevice));
            return;
        }
#endif
        // CPU 场景: 直接 MPI_Allgather (按字节传输)
        MPI_Allgather(sendbuf, static_cast<int>(send_bytes), MPI_BYTE,
                      recvbuf, static_cast<int>(send_bytes), MPI_BYTE, _mpi_comm);
    }

    // ══════════════════════════════════════════════════════════════
    // ReduceScatter: 归约后分发 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // MPI_Reduce_scatter_block: 先归约再分片
    // FP16/BF16: 需要 FP32 中转归约 (MPI 无 half SUM)
    // FP32: 直接 MPI_Reduce_scatter_block
    void reduceScatter(const void *sendbuf, void *recvbuf, size_t recvcount,
                       CommDataType dtype) override {
        if (recvcount == 0) return;
        size_t total_count = recvcount * _world_size;

        if (dtype == CommDataType::F32) {
            // FP32 直通路径
#ifdef ENABLE_DIST_NCCL
            if (isDevicePointer(sendbuf)) {
                CUDA_CHECK(cudaSetDevice(_device));
                std::vector<float> cpu_send(total_count);
                std::vector<float> cpu_recv(recvcount);
                CUDA_CHECK(cudaMemcpy(cpu_send.data(), sendbuf,
                                      total_count * sizeof(float), cudaMemcpyDeviceToHost));
                MPI_Reduce_scatter_block(cpu_send.data(), cpu_recv.data(),
                                         static_cast<int>(recvcount), MPI_FLOAT, MPI_SUM, _mpi_comm);
                CUDA_CHECK(cudaMemcpy(recvbuf, cpu_recv.data(),
                                      recvcount * sizeof(float), cudaMemcpyHostToDevice));
                return;
            }
#endif
            MPI_Reduce_scatter_block(sendbuf, recvbuf,
                                     static_cast<int>(recvcount), MPI_FLOAT, MPI_SUM, _mpi_comm);
        } else {
            // FP16/BF16: FP32 中转
            size_t elem_size = commDataTypeSize(dtype);
#ifdef ENABLE_DIST_NCCL
            if (isDevicePointer(sendbuf)) {
                CUDA_CHECK(cudaSetDevice(_device));
                // GPU → CPU (原精度)
                std::vector<char> cpu_raw(total_count * elem_size);
                CUDA_CHECK(cudaMemcpy(cpu_raw.data(), sendbuf,
                                      total_count * elem_size, cudaMemcpyDeviceToHost));

                // 展开为 FP32
                std::vector<float> fp32_send(total_count);
                std::vector<float> fp32_recv(recvcount);
                halfToFloat(cpu_raw.data(), fp32_send.data(), total_count, dtype);

                MPI_Reduce_scatter_block(fp32_send.data(), fp32_recv.data(),
                                         static_cast<int>(recvcount), MPI_FLOAT, MPI_SUM, _mpi_comm);

                // FP32 → 原精度 → GPU
                std::vector<char> cpu_result(recvcount * elem_size);
                floatToHalf(fp32_recv.data(), cpu_result.data(), recvcount, dtype);
                CUDA_CHECK(cudaMemcpy(recvbuf, cpu_result.data(),
                                      recvcount * elem_size, cudaMemcpyHostToDevice));
                return;
            }
#endif
            // CPU 场景
            std::vector<float> fp32_send(total_count);
            std::vector<float> fp32_recv(recvcount);
            halfToFloat(sendbuf, fp32_send.data(), total_count, dtype);
            MPI_Reduce_scatter_block(fp32_send.data(), fp32_recv.data(),
                                     static_cast<int>(recvcount), MPI_FLOAT, MPI_SUM, _mpi_comm);
            floatToHalf(fp32_recv.data(), recvbuf, recvcount, dtype);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Broadcast: 广播 (Phase 2 新增)
    // ══════════════════════════════════════════════════════════════
    // MPI_Bcast: root 的 data → 所有 rank (就地操作)
    // 使用 MPI_BYTE 传输, 不涉及归约, 支持任意 dtype
    void broadcast(void *data, size_t count, CommDataType dtype, int root) override {
        if (count == 0) return;
        size_t total_bytes = count * commDataTypeSize(dtype);

#ifdef ENABLE_DIST_NCCL
        if (isDevicePointer(data)) {
            CUDA_CHECK(cudaSetDevice(_device));
            std::vector<char> cpu_buf(total_bytes);
            // 所有 rank 都需要 GPU→CPU (root 的数据要发, 非 root 需要接收缓冲区)
            CUDA_CHECK(cudaMemcpy(cpu_buf.data(), data, total_bytes, cudaMemcpyDeviceToHost));
            MPI_Bcast(cpu_buf.data(), static_cast<int>(total_bytes), MPI_BYTE, root, _mpi_comm);
            // CPU → GPU (非 root rank 写回广播数据)
            CUDA_CHECK(cudaMemcpy(data, cpu_buf.data(), total_bytes, cudaMemcpyHostToDevice));
            return;
        }
#endif
        MPI_Bcast(data, static_cast<int>(total_bytes), MPI_BYTE, root, _mpi_comm);
    }

    // ══════════════════════════════════════════════════════════════
    // Send/Recv: 点对点通信 (Phase 2 新增, 为 PP 预留)
    // ══════════════════════════════════════════════════════════════
    // MPI_Send/MPI_Recv: 阻塞式点对点通信
    // GPU 数据需要 GPU→CPU→MPI→CPU→GPU 中转
    void send(const void *data, size_t count, CommDataType dtype, int dst) override {
        if (count == 0) return;
        size_t total_bytes = count * commDataTypeSize(dtype);

#ifdef ENABLE_DIST_NCCL
        if (isDevicePointer(data)) {
            CUDA_CHECK(cudaSetDevice(_device));
            std::vector<char> cpu_buf(total_bytes);
            CUDA_CHECK(cudaMemcpy(cpu_buf.data(), data, total_bytes, cudaMemcpyDeviceToHost));
            MPI_Send(cpu_buf.data(), static_cast<int>(total_bytes), MPI_BYTE, dst, 0, _mpi_comm);
            return;
        }
#endif
        MPI_Send(data, static_cast<int>(total_bytes), MPI_BYTE, dst, 0, _mpi_comm);
    }

    void recv(void *data, size_t count, CommDataType dtype, int src) override {
        if (count == 0) return;
        size_t total_bytes = count * commDataTypeSize(dtype);

#ifdef ENABLE_DIST_NCCL
        if (isDevicePointer(data)) {
            CUDA_CHECK(cudaSetDevice(_device));
            std::vector<char> cpu_buf(total_bytes);
            MPI_Recv(cpu_buf.data(), static_cast<int>(total_bytes), MPI_BYTE, src, 0, _mpi_comm, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(data, cpu_buf.data(), total_bytes, cudaMemcpyHostToDevice));
            return;
        }
#endif
        MPI_Recv(data, static_cast<int>(total_bytes), MPI_BYTE, src, 0, _mpi_comm, MPI_STATUS_IGNORE);
    }

private:
    // FP16/BF16 → FP32 批量转换辅助函数
    static void halfToFloat(const void *src, float *dst, size_t count, CommDataType dtype) {
        const uint16_t *s = reinterpret_cast<const uint16_t*>(src);
        if (dtype == CommDataType::F16) {
            for (size_t i = 0; i < count; i++) {
                _Float16 val;
                std::memcpy(&val, &s[i], sizeof(_Float16));
                dst[i] = static_cast<float>(val);
            }
        } else { // BF16
            for (size_t i = 0; i < count; i++) {
                uint32_t bits = static_cast<uint32_t>(s[i]) << 16;
                std::memcpy(&dst[i], &bits, sizeof(float));
            }
        }
    }

    // FP32 → FP16/BF16 批量转换辅助函数
    static void floatToHalf(const float *src, void *dst, size_t count, CommDataType dtype) {
        uint16_t *d = reinterpret_cast<uint16_t*>(dst);
        if (dtype == CommDataType::F16) {
            for (size_t i = 0; i < count; i++) {
                _Float16 val = static_cast<_Float16>(src[i]);
                std::memcpy(&d[i], &val, sizeof(_Float16));
            }
        } else { // BF16
            for (size_t i = 0; i < count; i++) {
                uint32_t bits;
                std::memcpy(&bits, &src[i], sizeof(float));
                d[i] = static_cast<uint16_t>(bits >> 16);
            }
        }
    }
};

std::shared_ptr<Comm> createMpiComm(const Config &config) {
    return std::make_shared<MpiComm>(config);
}

} // namespace llaisys::distributed

#endif // ENABLE_DIST_MPI

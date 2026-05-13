#ifndef LLAISYS_DISTRIBUTED_H
#define LLAISYS_DISTRIBUTED_H

#include "../llaisys.h"

__C {
    typedef enum {
        LLAISYS_DIST_BACKEND_MOCK = 0,
        LLAISYS_DIST_BACKEND_NCCL = 1,
        LLAISYS_DIST_BACKEND_MPI = 2,
    } llaisysDistBackend_t;

    // 通信数据类型枚举 (Phase 1 新增)
    // 与 C++ 内部 CommDataType 一一对应
    typedef enum {
        LLAISYS_DIST_DTYPE_F32 = 0,     // 32-bit 浮点
        LLAISYS_DIST_DTYPE_F16 = 1,     // 16-bit 半精度
        LLAISYS_DIST_DTYPE_BF16 = 2,    // 16-bit BFloat16
    } llaisysDistDataType_t;

    struct LlaisysDistConfig {
        llaisysDistBackend_t backend;
        int world_size;
        int rank;
        int local_device;

        // Phase 4: 多节点 NCCL 支持
        // master_addr: rank 0 的 IP 地址 (NULL 或空字符串 = 单机文件模式)
        // master_port: rank 0 的 TCP 端口 (默认 29400)
        // 兼容 PyTorch Distributed 环境变量: MASTER_ADDR, MASTER_PORT
        const char *master_addr;  // 可为 NULL (单机模式)
        int master_port;          // 0 时使用默认端口 29400
    };

    typedef struct LlaisysDistComm *llaisysDistComm_t;

    __export llaisysDistComm_t llaisysDistCommCreate(struct LlaisysDistConfig config);
    __export void llaisysDistCommDestroy(llaisysDistComm_t comm);

    __export llaisysDistBackend_t llaisysDistCommBackend(llaisysDistComm_t comm);
    __export int llaisysDistCommWorldSize(llaisysDistComm_t comm);
    __export int llaisysDistCommRank(llaisysDistComm_t comm);

    __export int llaisysDistBackendAvailable(llaisysDistBackend_t backend);

    // 旧接口: FP32 专用 (保留兼容性)
    __export void llaisysDistAllReduceSumF32(llaisysDistComm_t comm, float *data, size_t count);

    // Phase 1 新增: 多数据类型 AllReduce
    // data: GPU/CPU 上的缓冲区 (类型由 dtype 指定)
    // count: 元素个数 (不是字节数)
    // dtype: LLAISYS_DIST_DTYPE_F32/F16/BF16
    __export void llaisysDistAllReduceSum(llaisysDistComm_t comm, void *data, size_t count, llaisysDistDataType_t dtype);

    __export void llaisysDistBarrier(llaisysDistComm_t comm);

    // Phase 2 新增: 通信原语
    // AllGather: 每个 rank 提供 sendcount 个元素, 所有 rank 得到拼接结果
    __export void llaisysDistAllGather(llaisysDistComm_t comm, const void *sendbuf, void *recvbuf,
                                       size_t sendcount, llaisysDistDataType_t dtype);

    // ReduceScatter: 先归约再分片, 每个 rank 得到 recvcount 个元素
    __export void llaisysDistReduceScatter(llaisysDistComm_t comm, const void *sendbuf, void *recvbuf,
                                            size_t recvcount, llaisysDistDataType_t dtype);

    // Broadcast: root rank 的数据广播到所有 rank
    __export void llaisysDistBroadcast(llaisysDistComm_t comm, void *data, size_t count,
                                       llaisysDistDataType_t dtype, int root);

    // Send: 点对点发送到 dst rank
    __export void llaisysDistSend(llaisysDistComm_t comm, const void *data, size_t count,
                                   llaisysDistDataType_t dtype, int dst);

    // Recv: 从 src rank 接收数据
    __export void llaisysDistRecv(llaisysDistComm_t comm, void *data, size_t count,
                                   llaisysDistDataType_t dtype, int src);

    // 获取内部 comm 实现指针 (供 C++ 内部使用, 返回 shared_ptr<Comm>*)
    __export void *llaisysDistCommGetImplPtr(llaisysDistComm_t comm);

    // ══════════════════════════════════════════════════════════════
    // Phase 3: TP 同步控制器 (替代 TCP JSON 同步)
    // ══════════════════════════════════════════════════════════════
    // 用 NCCL/MPI Broadcast 替代 TCP socket 传递推理控制命令
    // 延迟从 ~100μs(TCP) 降至 ~5μs(NVLink broadcast)

    typedef struct LlaisysTPSync *llaisysTPSync_t;

    // 创建同步控制器 (需要传入已创建的 comm, max_payload 为 token_ids 最大长度)
    __export llaisysTPSync_t llaisysTPSyncCreate(llaisysDistComm_t comm, size_t max_payload);
    __export void llaisysTPSyncDestroy(llaisysTPSync_t sync);

    // Rank 0: 填充命令
    __export void llaisysTPSyncFillPrefill(llaisysTPSync_t sync, int slot_id,
                                            const int *token_ids, size_t n_tokens,
                                            float temperature, int top_k, float top_p);
    __export void llaisysTPSyncFillDecode(llaisysTPSync_t sync,
                                           const int *active_slots, const int *current_tokens,
                                           int batch_size,
                                           float temperature, int top_k, float top_p);
    __export void llaisysTPSyncFillSlotReset(llaisysTPSync_t sync, int slot_id);
    __export void llaisysTPSyncFillShutdown(llaisysTPSync_t sync);

    // 所有 Rank: 执行 Broadcast 同步
    __export void llaisysTPSyncBroadcast(llaisysTPSync_t sync);

    // 读取同步结果
    __export int llaisysTPSyncGetCmd(llaisysTPSync_t sync);           // 返回 TPSyncCmd 枚举值
    __export int llaisysTPSyncGetSlotId(llaisysTPSync_t sync);
    __export int llaisysTPSyncGetBatchSize(llaisysTPSync_t sync);
    __export int llaisysTPSyncGetPayloadCount(llaisysTPSync_t sync);
    __export float llaisysTPSyncGetTemperature(llaisysTPSync_t sync);
    __export int llaisysTPSyncGetTopK(llaisysTPSync_t sync);
    __export float llaisysTPSyncGetTopP(llaisysTPSync_t sync);
    __export const int *llaisysTPSyncGetPayload(llaisysTPSync_t sync); // int32 数组指针
}

#endif // LLAISYS_DISTRIBUTED_H

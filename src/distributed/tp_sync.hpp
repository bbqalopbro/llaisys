// ============================================================================
// tp_sync.hpp — TP 控制面同步器 (Phase 3: 替代 TCP JSON 同步)
// ============================================================================
// 用 NCCL/MPI Broadcast 替代 TCP socket 传递推理控制命令
//
// 旧方案 (TCP):
//   rank 0 → followers: JSON 消息 (socket send/recv)
//   问题: TCP 连接断开 → 全部死锁; 延迟 ~100μs; 无错误传播
//
// 新方案 (NCCL Broadcast):
//   rank 0 填充 GPU 上的命令缓冲区 → ncclBroadcast → 所有 rank 读取
//   优势: NVLink 上延迟 ~5μs; 原子操作, 不存在半接收; 与通信后端统一
//
// 同步协议:
//   1. Broadcast 固定大小 Header (TPSyncHeader, 16 个 int32)
//      内含 cmd_type + 参数 + payload_count (变长数据的元素数)
//   2. 若 payload_count > 0, Broadcast 变长 Payload (int32 数组)
//      内含 token_ids (prefill) 或 active_slots+current_tokens (decode)
//
// 使用方式:
//   rank 0: fillPrefillCommand(...) → broadcastSync()
//   rank 1-N: broadcastSync() → 读取 header() 和 payload()
// ============================================================================
#pragma once

#include "comm.hpp"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace llaisys::distributed {

// 命令类型枚举
enum class TPSyncCmd : int {
    Idle = 0,       // 空闲 (无操作)
    Prefill = 1,    // 预填充: 需要 slot_id + token_ids
    Decode = 2,     // 解码: 需要 active_slots + current_tokens
    SlotReset = 3,  // 重置 slot
    Shutdown = 4,   // 关闭
};

// 固定大小的命令头 (16 个 int32 = 64 字节)
// 所有字段为 int32, 方便 GPU broadcast
struct TPSyncHeader {
    int cmd;             // TPSyncCmd 枚举值
    int slot_id;         // prefill/slot_reset 用的 slot ID
    int batch_size;      // decode 时的活跃 batch 大小
    int payload_count;   // 后续 payload 的 int32 元素数 (0 表示无 payload)
    int temperature_x1000; // 温度 × 1000 (整数化, 避免浮点 broadcast)
    int top_k;           // top-k 采样参数
    int top_p_x1000;     // top-p × 1000
    int reserved[9];     // 保留字段 (未来扩展, 初始化为 0)
};

static_assert(sizeof(TPSyncHeader) == 64, "TPSyncHeader must be 64 bytes");

// TP 同步控制器
// 封装 header + payload 的 broadcast 逻辑
class TPSyncController {
public:
    // 构造: 传入通信句柄, 内部分配 CPU 缓冲区
    // max_payload: payload 数组最大元素数 (prefill 的 token_ids 最大长度)
    explicit TPSyncController(comm_t comm, size_t max_payload = 8192)
        : _comm(std::move(comm)),
          _max_payload(max_payload) {
        if (!_comm) {
            throw std::invalid_argument("TPSyncController: comm is null");
        }
        // 预分配 payload 缓冲区
        _payload.resize(_max_payload);
        // 清零 header
        std::memset(&_header, 0, sizeof(_header));
    }

    // ── Rank 0 端: 填充命令 ──

    // 填充 Prefill 命令
    // slot_id: 目标 slot
    // token_ids: prefill 的 token 序列
    // temperature/top_k/top_p: 采样参数
    void fillPrefillCommand(int slot_id, const int *token_ids, size_t n_tokens,
                            float temperature = 0.8f, int top_k = 50, float top_p = 0.9f) {
        if (n_tokens > _max_payload) {
            throw std::runtime_error("TPSyncController: token_ids exceeds max_payload");
        }
        std::memset(&_header, 0, sizeof(_header));
        _header.cmd = static_cast<int>(TPSyncCmd::Prefill);
        _header.slot_id = slot_id;
        _header.payload_count = static_cast<int>(n_tokens);
        _header.temperature_x1000 = static_cast<int>(temperature * 1000.0f);
        _header.top_k = top_k;
        _header.top_p_x1000 = static_cast<int>(top_p * 1000.0f);

        // 拷贝 token_ids 到 payload 缓冲区
        std::memcpy(_payload.data(), token_ids, n_tokens * sizeof(int));
    }

    // 填充 Decode 命令
    // active_slots: 活跃 slot ID 数组
    // current_tokens: 每个 slot 当前的 token
    // batch_size: 活跃 slot 数
    void fillDecodeCommand(const int *active_slots, const int *current_tokens,
                           int batch_size,
                           float temperature = 0.8f, int top_k = 50, float top_p = 0.9f) {
        if (static_cast<size_t>(batch_size * 2) > _max_payload) {
            throw std::runtime_error("TPSyncController: batch data exceeds max_payload");
        }
        std::memset(&_header, 0, sizeof(_header));
        _header.cmd = static_cast<int>(TPSyncCmd::Decode);
        _header.batch_size = batch_size;
        _header.payload_count = batch_size * 2; // active_slots + current_tokens 连续存放
        _header.temperature_x1000 = static_cast<int>(temperature * 1000.0f);
        _header.top_k = top_k;
        _header.top_p_x1000 = static_cast<int>(top_p * 1000.0f);

        // payload 布局: [active_slots[0..B-1] | current_tokens[0..B-1]]
        std::memcpy(_payload.data(), active_slots, batch_size * sizeof(int));
        std::memcpy(_payload.data() + batch_size, current_tokens, batch_size * sizeof(int));
    }

    // 填充 SlotReset 命令
    void fillSlotResetCommand(int slot_id) {
        std::memset(&_header, 0, sizeof(_header));
        _header.cmd = static_cast<int>(TPSyncCmd::SlotReset);
        _header.slot_id = slot_id;
        _header.payload_count = 0;
    }

    // 填充 Shutdown 命令
    void fillShutdownCommand() {
        std::memset(&_header, 0, sizeof(_header));
        _header.cmd = static_cast<int>(TPSyncCmd::Shutdown);
        _header.payload_count = 0;
    }

    // ── 所有 Rank: 执行 Broadcast 同步 ──

    // broadcastSync: rank 0 发送, 其他 rank 接收
    // 调用后所有 rank 的 _header 和 _payload 内容相同
    //
    // 实现:
    //   1. Broadcast header (16 个 int32, 用 CommDataType::F32 传 int 数据)
    //      注: int32 和 float32 同为 4 字节, NCCL broadcast 只关心字节数不关心类型
    //   2. 若 payload_count > 0, Broadcast payload
    void broadcastSync() {
        // 第一步: 广播 header (固定 16 个 int32 = 64 字节)
        _comm->broadcast(
            &_header,
            sizeof(TPSyncHeader) / sizeof(int),  // 16 个元素
            CommDataType::F32,  // int32 和 float32 同为 4 字节, 这里只做字节搬运
            0  // root = rank 0
        );

        // 第二步: 若有 payload, 广播 payload
        if (_header.payload_count > 0) {
            if (static_cast<size_t>(_header.payload_count) > _max_payload) {
                throw std::runtime_error("TPSyncController: received payload_count exceeds max_payload");
            }
            _comm->broadcast(
                _payload.data(),
                _header.payload_count,
                CommDataType::F32,  // int32 当作 float32 广播 (字节数相同)
                0
            );
        }
    }

    // ── 只读访问器 ──

    const TPSyncHeader& header() const { return _header; }
    TPSyncCmd cmd() const { return static_cast<TPSyncCmd>(_header.cmd); }
    int slotId() const { return _header.slot_id; }
    int batchSize() const { return _header.batch_size; }
    int payloadCount() const { return _header.payload_count; }

    // 采样参数还原 (整数 → 浮点)
    float temperature() const { return _header.temperature_x1000 / 1000.0f; }
    int topK() const { return _header.top_k; }
    float topP() const { return _header.top_p_x1000 / 1000.0f; }

    // Payload 原始指针 (int32 数组)
    const int* payload() const { return _payload.data(); }

    // Prefill: 获取 token_ids
    const int* tokenIds() const { return _payload.data(); }
    int numTokens() const { return _header.payload_count; }

    // Decode: 获取 active_slots 和 current_tokens
    const int* activeSlots() const { return _payload.data(); }
    const int* currentTokens() const { return _payload.data() + _header.batch_size; }

private:
    comm_t _comm;
    size_t _max_payload;
    TPSyncHeader _header;
    std::vector<int> _payload;  // CPU 缓冲区
};

} // namespace llaisys::distributed

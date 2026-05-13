// ============================================================================
// tcp_id_store.hpp — 跨节点 TCP ncclUniqueId 分发
// ============================================================================
// 背景:
//   原 nccl_comm.cu 通过文件系统 (/tmp/llaisys_nccl_id) 共享 ncclUniqueId,
//   仅限单机 (所有进程访问同一文件系统).
//
// 本模块实现:
//   使用 TCP socket 在跨节点场景下分发 ncclUniqueId.
//   Rank 0 (master) 启动 TCP server, 等待 (world_size-1) 个 worker 连接;
//   收到连接后发送 128 字节的 ncclUniqueId.
//   Worker rank 连接到 master, 接收 ncclUniqueId.
//
// 使用方式:
//   环境变量 MASTER_ADDR 和 MASTER_PORT 设定 rank 0 的 IP 和端口,
//   兼容 PyTorch Distributed 标准 (torch.distributed.init_process_group).
//
// 安全须知:
//   - ncclUniqueId 本身是一次性 token, 仅在初始化时使用
//   - 建议在受信网络中使用, 或配合 VPN/SSH tunnel
//   - 连接超时默认 60 秒, 防止无限等待
// ============================================================================
#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

// POSIX socket 头文件
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <poll.h>

namespace llaisys::distributed {

// ncclUniqueId 的固定大小 (128 字节, NCCL 定义)
static constexpr size_t NCCL_ID_BYTES = 128;

// ============================================================================
// TCP socket 辅助函数 (RAII 封装, 防止 fd 泄漏)
// ============================================================================

// 自动关闭的 socket 持有器
class ScopedSocket {
public:
    explicit ScopedSocket(int fd = -1) : fd_(fd) {}
    ~ScopedSocket() { if (fd_ >= 0) ::close(fd_); }

    // 禁止拷贝
    ScopedSocket(const ScopedSocket &) = delete;
    ScopedSocket &operator=(const ScopedSocket &) = delete;

    // 允许移动
    ScopedSocket(ScopedSocket &&other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    ScopedSocket &operator=(ScopedSocket &&other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    int fd() const { return fd_; }
    int release() { int f = fd_; fd_ = -1; return f; }

private:
    int fd_;
};

// 完整发送 n 字节 (处理 partial write)
static inline void sendAll(int fd, const void *buf, size_t n) {
    const char *p = static_cast<const char *>(buf);
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(fd, p + sent, n - sent, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("TCP send failed: ") + strerror(errno));
        }
        sent += static_cast<size_t>(r);
    }
}

// 完整接收 n 字节 (处理 partial read)
static inline void recvAll(int fd, void *buf, size_t n) {
    char *p = static_cast<char *>(buf);
    size_t recvd = 0;
    while (recvd < n) {
        ssize_t r = ::recv(fd, p + recvd, n - recvd, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("TCP recv failed: ") + strerror(errno));
        }
        if (r == 0) {
            throw std::runtime_error("TCP connection closed prematurely");
        }
        recvd += static_cast<size_t>(r);
    }
}

// ============================================================================
// Master (Rank 0): 启动 TCP server, 分发 ncclUniqueId
// ============================================================================
// nccl_id: 已生成的 ncclUniqueId (128 字节)
// port: 监听端口
// world_size: 总进程数 (需接受 world_size-1 个连接)
// timeout_sec: 等待所有 worker 连接的超时 (秒)
static inline void tcpMasterDistributeId(const void *nccl_id, int port,
                                          int world_size, int timeout_sec = 60) {
    // 创建 TCP server socket
    ScopedSocket server_fd(::socket(AF_INET6, SOCK_STREAM, 0));
    if (server_fd.fd() < 0) {
        // 回退到 IPv4
        server_fd = ScopedSocket(::socket(AF_INET, SOCK_STREAM, 0));
        if (server_fd.fd() < 0) {
            throw std::runtime_error(
                std::string("Cannot create TCP socket: ") + strerror(errno));
        }

        // 设置 SO_REUSEADDR (避免 TIME_WAIT 问题)
        int opt = 1;
        setsockopt(server_fd.fd(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        // 绑定 IPv4
        struct sockaddr_in addr4 {};
        addr4.sin_family = AF_INET;
        addr4.sin_addr.s_addr = INADDR_ANY;
        addr4.sin_port = htons(static_cast<uint16_t>(port));

        if (::bind(server_fd.fd(), reinterpret_cast<struct sockaddr *>(&addr4),
                   sizeof(addr4)) < 0) {
            throw std::runtime_error(
                std::string("TCP bind failed on port ") + std::to_string(port) +
                ": " + strerror(errno));
        }
    } else {
        // IPv6 dual-stack (同时支持 v4/v6)
        int opt = 1;
        setsockopt(server_fd.fd(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        int v6only = 0;
        setsockopt(server_fd.fd(), IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only));

        struct sockaddr_in6 addr6 {};
        addr6.sin6_family = AF_INET6;
        addr6.sin6_addr = in6addr_any;
        addr6.sin6_port = htons(static_cast<uint16_t>(port));

        if (::bind(server_fd.fd(), reinterpret_cast<struct sockaddr *>(&addr6),
                   sizeof(addr6)) < 0) {
            throw std::runtime_error(
                std::string("TCP bind failed on port ") + std::to_string(port) +
                ": " + strerror(errno));
        }
    }

    // 开始监听
    if (::listen(server_fd.fd(), world_size) < 0) {
        throw std::runtime_error(
            std::string("TCP listen failed: ") + strerror(errno));
    }

    fprintf(stderr, "[TCP-ID-Store] master: listening on port %d, "
                    "waiting for %d workers...\n", port, world_size - 1);

    // 使用 poll 等待连接 (带超时)
    int accepted = 0;
    int remaining = world_size - 1;

    while (remaining > 0) {
        struct pollfd pfd;
        pfd.fd = server_fd.fd();
        pfd.events = POLLIN;

        int ret = ::poll(&pfd, 1, timeout_sec * 1000);
        if (ret < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("TCP poll failed: ") + strerror(errno));
        }
        if (ret == 0) {
            throw std::runtime_error(
                "Timeout waiting for workers to connect. "
                "Expected " + std::to_string(world_size - 1) +
                " workers, got " + std::to_string(accepted));
        }

        // 接受连接
        ScopedSocket client_fd(::accept(server_fd.fd(), nullptr, nullptr));
        if (client_fd.fd() < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("TCP accept failed: ") + strerror(errno));
        }

        // 发送 ncclUniqueId (128 字节)
        sendAll(client_fd.fd(), nccl_id, NCCL_ID_BYTES);

        accepted++;
        remaining--;

        fprintf(stderr, "[TCP-ID-Store] master: sent ID to worker %d/%d\n",
                accepted, world_size - 1);
    }

    fprintf(stderr, "[TCP-ID-Store] master: all %d workers connected\n",
            world_size - 1);
}

// ============================================================================
// Worker (Rank > 0): 连接到 master, 接收 ncclUniqueId
// ============================================================================
// nccl_id: 输出缓冲区 (128 字节, 接收到的 ncclUniqueId)
// master_addr: rank 0 的 IP 地址或主机名
// port: rank 0 的监听端口
// timeout_sec: 连接超时 (秒), 含重试
static inline void tcpWorkerReceiveId(void *nccl_id, const std::string &master_addr,
                                       int port, int timeout_sec = 60) {
    // DNS 解析 master_addr
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;     // IPv4 或 IPv6
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo *result = nullptr;
    std::string port_str = std::to_string(port);

    int gai_err = getaddrinfo(master_addr.c_str(), port_str.c_str(), &hints, &result);
    if (gai_err != 0) {
        throw std::runtime_error(
            "Cannot resolve master address '" + master_addr + "': " +
            gai_strerror(gai_err));
    }

    // RAII 清理 addrinfo
    struct AddrInfoGuard {
        struct addrinfo *p;
        ~AddrInfoGuard() { if (p) freeaddrinfo(p); }
    } guard{result};

    // 重试连接 (master 可能还没 listen)
    const int retry_interval_ms = 500;
    int elapsed_ms = 0;

    while (elapsed_ms < timeout_sec * 1000) {
        for (struct addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
            ScopedSocket sock(::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol));
            if (sock.fd() < 0) continue;

            if (::connect(sock.fd(), rp->ai_addr, rp->ai_addrlen) == 0) {
                // 连接成功, 接收 ncclUniqueId
                recvAll(sock.fd(), nccl_id, NCCL_ID_BYTES);

                fprintf(stderr, "[TCP-ID-Store] worker: received ID from %s:%d\n",
                        master_addr.c_str(), port);
                return;
            }
        }

        // 连接失败, 等待后重试
        usleep(static_cast<useconds_t>(retry_interval_ms) * 1000);
        elapsed_ms += retry_interval_ms;
    }

    throw std::runtime_error(
        "Timeout connecting to master at " + master_addr + ":" +
        std::to_string(port) + " after " + std::to_string(timeout_sec) + "s");
}

// ============================================================================
// 统一接口: 根据 rank 自动选择 master/worker 角色
// ============================================================================
// nccl_id: rank 0 提供已生成的 ID, 其他 rank 接收
// rank: 当前进程 rank
// world_size: 总进程数
// master_addr: rank 0 的 IP 地址 (所有 rank 必须一致)
// master_port: rank 0 的监听端口
// timeout_sec: 超时 (秒)
static inline void tcpExchangeNcclId(void *nccl_id, int rank, int world_size,
                                      const std::string &master_addr, int master_port,
                                      int timeout_sec = 60) {
    if (rank == 0) {
        // Master: 分发自己生成的 ncclUniqueId
        tcpMasterDistributeId(nccl_id, master_port, world_size, timeout_sec);
    } else {
        // Worker: 从 master 接收 ncclUniqueId
        tcpWorkerReceiveId(nccl_id, master_addr, master_port, timeout_sec);
    }
}

} // namespace llaisys::distributed

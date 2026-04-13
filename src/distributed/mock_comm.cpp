// ============================================================================
// mock_comm.cpp — Mock (空操作) 通信后端
// 用途: 单卡开发和测试时使用, allReduce 直接返回 (不做任何归约)
// 适合调试模型正确性, 不需要多卡环境
// ============================================================================
#include "comm.hpp"

#include <stdexcept>

namespace llaisys::distributed {

// MockComm: 不做任何实际通信, 单卡时 allReduce 是恒等操作
class MockComm : public Comm {
private:
    int _world_size;
    int _rank;

public:
    explicit MockComm(const Config &config)
        : _world_size(config.world_size),
          _rank(config.rank) {
        if (_world_size <= 0) {
            throw std::invalid_argument("world_size must be > 0");
        }
        if (_rank < 0 || _rank >= _world_size) {
            throw std::invalid_argument("rank out of range");
        }
    }

    Backend backend() const override { return Backend::Mock; }
    int worldSize() const override { return _world_size; }
    int rank() const override { return _rank; }

    // 空操作: 单卡时数据本身就是完整的, 无需归约
    void allReduceSum(float *data, size_t count) override {
        if (data == nullptr && count != 0) {
            throw std::invalid_argument("data is null");
        }
        // 什么都不做 — 数据原封不动返回
    }

    // 空操作: 单卡不需要同步
    void barrier() override {
    }
};

std::shared_ptr<Comm> createMockComm(const Config &config) {
    return std::make_shared<MockComm>(config);
}
} // namespace llaisys::distributed

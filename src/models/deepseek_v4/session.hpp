#pragma once
#include "model.hpp"
#include "src/engine/schedule_plan.hpp"
#include <functional>

namespace llaisys::models::deepseek_v4 {
struct SessionOutput {
    engine::SequenceOutput sequence;
    std::vector<float> logits; // Optional diagnostic D2H; off on normal execution.
};
struct SessionResult { uint64_t step_id; std::vector<SessionOutput> outputs; };
struct SessionOperatorInfo {
    backends::native::KernelIdentity identity;
    uint64_t calls, failures;
};
struct SessionInfo {
    int64_t layers, vocabulary, capacity;
    size_t loaded_weights, auxiliary_weights, active_slots;
    uint64_t loaded_bytes, steps;
    size_t slots;
    bool experimental_chunking;
    uint64_t model_calls, model_failures;
    std::vector<SessionOperatorInfo> operators;
};

// Owns one native execution thread so Python finalization/calls from different
// threads cannot outlive a thread-local Runtime. This is a serialized command
// executor, NOT request admission, token budgeting, batching policy or an EP
// worker. The existing Python scheduler supplies complete SchedulePlans.
class Session {
public:
    using BackendFactory = std::function<ModelBackends(core::Runtime &, const ModelConfig &)>;
    Session(std::string source, std::string checkpoint, int64_t capacity, size_t slots,
            BackendFactory factory, bool experimental_chunking = false);
    ~Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    SessionResult execute(engine::SchedulePlan plan, bool capture_logits = false);
    SessionInfo info();
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace llaisys::models::deepseek_v4

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::engine {

struct SamplingParams {
    float temperature = 0.8F;
    int top_k = 50;
    float top_p = 0.9F;
};

struct PrefillItem {
    uint64_t request_id = 0;
    size_t slot_id = 0;
    std::vector<int64_t> token_ids;
    int64_t start_pos = 0;
    bool is_last_chunk = false;
    SamplingParams sampling;
};

struct DecodeItem {
    uint64_t request_id = 0;
    size_t slot_id = 0;
    int64_t token_id = 0;
    SamplingParams sampling;
};

// Built by Python policy code and executed by the C++ model runtime. It is a
// plain data contract: no queueing, admission, priority, or HTTP concepts.
struct SchedulePlan {
    uint64_t step_id = 0;
    std::vector<size_t> reset_slots;
    std::vector<PrefillItem> prefills;
    std::vector<DecodeItem> decodes;
};

struct SequenceOutput {
    uint64_t request_id = 0;
    size_t slot_id = 0;
    int64_t token_id = -1;
};

struct StepResult {
    uint64_t step_id = 0;
    std::vector<SequenceOutput> outputs;
};

} // namespace llaisys::engine

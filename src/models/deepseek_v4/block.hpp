#pragma once

#include "attention.hpp"
#include "hyperconnection.hpp"
#include "moe.hpp"

namespace llaisys::models::deepseek_v4 {

struct BlockConfig { AttentionConfig attention; MoeConfig moe; int64_t copies; };
struct BlockWeights {
    HCWeights attention_hc, feedforward_hc;
    std::shared_ptr<backends::native::Tensor> attention_norm, feedforward_norm;
};
struct BlockBackends {
    HCBackends attention_hc, feedforward_hc;
    std::shared_ptr<backends::native::Kernel> cast, norm;
};

class Block;
class BlockState {
public:
    BlockState(core::Runtime &runtime, BlockConfig config);
    AttentionState attention;
    int64_t position() const { return position_; }
    bool failed() const { return failed_; }
private:
    friend class Block;
    const Block *owner_ = nullptr;
    int64_t position_ = 0;
    bool failed_ = true;
};

class BlockWorkspace {
public:
    using Tensor = backends::native::Tensor;
    BlockWorkspace(core::Runtime &runtime, BlockConfig config, int64_t position, int64_t tokens);
    const BlockConfig config;
    HCWorkspace attention_hc, feedforward_hc;
    AttentionWorkspace attention;
    MoeWorkspace moe;
    Tensor attention_input, feedforward_input, feedforward_output;
    bool failed() const { return failed_; }
private:
    friend class Block;
    bool failed_ = false;
};

// Complete single-rank Transformer block. Owns model components, not serving
// policy. Attention state commits only as part of a successful whole block:
// any later HC/MoE failure poisons BlockState even if its nested cache advanced.
// Current Attention storage is the explicit continuous/ring reference backend.
class Block {
public:
    using Tensor = backends::native::Tensor;
    Block(std::unique_ptr<Attention> attention, std::unique_ptr<MoE> moe, int64_t copies,
          BlockWeights weights, BlockBackends backends);
    const BlockConfig &config() const { return config_; }
    void reset(BlockState &state);
    Tensor &forward(Tensor &hidden, Tensor &token_ids, BlockState &state, BlockWorkspace &workspace);
    uint64_t calls() const { return calls_; }
    uint64_t failures() const { return failures_; }
private:
    std::unique_ptr<Attention> attention_;
    std::unique_ptr<MoE> moe_;
    const BlockConfig config_;
    BlockWeights weights_;
    BlockBackends backends_;
    HyperConnection attention_hc_, feedforward_hc_;
    Tensor attention_norm_, feedforward_norm_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::models::deepseek_v4

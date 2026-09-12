#pragma once

#include "indexer.hpp"
#include <map>

namespace llaisys::models::deepseek_v4 {

struct AttentionConfig {
    int64_t hidden, heads, dimension, query_rank, output_groups, output_rank;
    int64_t rope_dimension, window, ratio, capacity, index_heads, index_dimension, index_topk;
};
struct LinearBackends { std::shared_ptr<backends::native::Kernel> quantize, gemm; };
struct AttentionBackends {
    using Kernel = backends::native::Kernel;
    LinearBackends query_a, query_b, latent, output;
    std::shared_ptr<Kernel> cast, fill, norm, inverse_rms, row_multiply, rotary, inverse_rotary;
    std::shared_ptr<Kernel> latent_qdq, prepare, sparse, grouped;
    CompressorBackends compression;
    IndexerBackends indexer;
};
using AttentionWeights = std::map<std::string, std::shared_ptr<backends::native::Tensor>>;

class Attention;
class AttentionState {
public:
    AttentionState(core::Runtime &runtime, AttentionConfig config);
    const AttentionConfig config;
    // Explicit reference storage: circular full-resolution window followed by
    // compressed latent groups. Not a paged-cache allocator/production claim.
    backends::native::Tensor cache;
    std::unique_ptr<CompressorState> compression;
    std::unique_ptr<IndexerState> indexer;
    int64_t position() const { return position_; }
    bool failed() const { return failed_; }
private:
    friend class Attention;
    const Attention *owner_ = nullptr;
    int64_t position_ = 0;
    bool failed_ = true;
};

class AttentionWorkspace {
public:
    using Tensor = backends::native::Tensor;
    AttentionWorkspace(core::Runtime &runtime, AttentionConfig config, int64_t position, int64_t tokens);
    const AttentionConfig config;
    const int64_t position, tokens, history, groups, window_candidates, extra_candidates;
    LinearWorkspace query_a, query_b, latent_linear, output_linear;
    Tensor query_rank_raw, query_rank, query, inverse_rms, latent_raw, latent;
    Tensor quant_input, quant_output, quant_scales, empty_indices, candidates, payload, attended, grouped, output;
    std::unique_ptr<CompressorWorkspace> compression;
    std::unique_ptr<IndexerWorkspace> indexer;
    bool failed() const { return failed_; }
private:
    friend class Attention;
    bool failed_ = false;
};

// Full native Attention composition for all actual 0/4/128 ratios, using
// explicit contiguous reference storage. Does not import upstream model code.
// Kernel bindings are independent of model/request ownership and scheduling.
class Attention {
public:
    using Tensor = backends::native::Tensor;
    Attention(AttentionConfig config, AttentionWeights weights, std::shared_ptr<Tensor> frequencies, AttentionBackends backends);
    void reset(AttentionState &state);
    Tensor &forward(Tensor &input, AttentionState &state, AttentionWorkspace &workspace);
    uint64_t calls() const { return calls_; }
    uint64_t failures() const { return failures_; }
    const AttentionConfig &config() const { return config_; }
    core::Runtime &runtime() const { return query_a_.runtime(); }
private:
    const AttentionConfig config_;
    AttentionWeights weights_;
    std::shared_ptr<Tensor> frequencies_;
    AttentionBackends backends_;
    QuantizedLinear query_a_, query_b_, latent_, output_;
    Tensor query_norm_, latent_norm_, output_group_weight_;
    std::unique_ptr<Compressor> compressor_;
    std::unique_ptr<Indexer> indexer_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::models::deepseek_v4

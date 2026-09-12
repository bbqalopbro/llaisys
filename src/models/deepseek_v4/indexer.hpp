#pragma once

#include "compressor.hpp"
#include "linear.hpp"

namespace llaisys::models::deepseek_v4 {

struct IndexerConfig {
    int64_t hidden, query_rank, heads, dimension, rope_dimension, topk, capacity;
};

struct IndexerWeights {
    using Tensor = backends::native::Tensor;
    std::shared_ptr<Tensor> query, query_scales, head_weights;
    std::shared_ptr<Tensor> compressor_kv, compressor_gate, ape, norm, frequencies;
};

struct IndexerBackends {
    using Kernel = backends::native::Kernel;
    std::shared_ptr<Kernel> activation_quant, gemm, rotary, hadamard, fp4_qdq;
    std::shared_ptr<Kernel> cast, fill, dense, scale, scores, mask, topk, remap;
    CompressorBackends compression;
};

class Indexer;
class IndexerState {
public:
    IndexerState(core::Runtime &runtime, IndexerConfig config);
    const IndexerConfig config;
    CompressorState compression;
    // Explicit contiguous reference cache, NOT a replacement block allocator.
    // Native paged-storage integration remains a separate Attention task.
    backends::native::Tensor cache;
    int64_t position() const { return compression.position(); }
    bool failed() const { return failed_; }
private:
    friend class Indexer;
    const Indexer *owner_ = nullptr;
    bool failed_ = true;
};

class IndexerWorkspace {
public:
    using Tensor = backends::native::Tensor;
    IndexerWorkspace(core::Runtime &runtime, IndexerConfig config, int64_t position, int64_t tokens);
    const IndexerConfig config;
    const int64_t position, tokens, candidates, selected;
    LinearWorkspace linear;
    CompressorWorkspace compression;
    Tensor query, rotated, quantized, quant_scales, projected_weights, weights, scores, indices, output;
    bool failed() const { return failed_; }
private:
    friend class Indexer;
    bool failed_ = false;
};

// Single-request complete Indexer composition. All math crosses explicit
// Kernel contracts; neither serving policy nor ATen/TileLang types live here.
class Indexer {
public:
    using Tensor = backends::native::Tensor;
    Indexer(IndexerConfig config, IndexerWeights weights, IndexerBackends backends);
    void reset(IndexerState &state);
    // Returns a borrowed int32 [1,S,min(topk,end/4)] workspace view. -1 denotes
    // masked candidates; offset belongs to the consuming Attention layout.
    Tensor &forward(Tensor &hidden, Tensor &query_rank, int64_t offset, IndexerState &state, IndexerWorkspace &workspace);
    uint64_t calls() const { return calls_; }
    uint64_t failures() const { return failures_; }
private:
    const IndexerConfig config_;
    IndexerWeights weights_;
    IndexerBackends backends_;
    QuantizedLinear query_;
    Compressor compressor_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::models::deepseek_v4

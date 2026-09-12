#pragma once

#include "src/backends/native/kernel.hpp"

namespace llaisys::models::deepseek_v4 {

struct CompressorConfig {
    int64_t hidden, dimension, rope_dimension, ratio;
    bool rotate;
};

struct CompressorBackends {
    using Kernel = backends::native::Kernel;
    std::shared_ptr<Kernel> cast, fill, dense, prepare, pool, norm, rotary, quantize;
    std::shared_ptr<Kernel> hadamard; // Required only for rotated Indexer cache.
};

class Compressor;
class CompressorState {
public:
    using Tensor = backends::native::Tensor;
    CompressorState(core::Runtime &runtime, CompressorConfig config);
    const CompressorConfig config;
    Tensor kv, scores;
    int64_t position() const { return position_; }
    bool failed() const { return failed_; }
private:
    friend class Compressor;
    const Compressor *owner_ = nullptr;
    int64_t position_ = 0;
    bool failed_ = true; // Must be reset by its owning model before execution.
};

// Explicit shape-dependent buffers, retained until stream completion. No
// Python/ATen object owns model weights or request compression state.
class CompressorWorkspace {
public:
    using Tensor = backends::native::Tensor;
    CompressorWorkspace(core::Runtime &runtime, CompressorConfig config, int64_t position, int64_t tokens);
    const CompressorConfig config;
    const int64_t position, tokens, emitted;
    Tensor input_fp32, projected_kv, projected_scores, grouped_kv, grouped_scores;
    Tensor pooled_fp32, pooled_bf16, normalized, rotated, quant_input, quant_output, quant_scales;
    bool failed() const { return failed_; }
private:
    friend class Compressor;
    bool failed_ = false;
};

struct CompressedChunk {
    int64_t first_group, count;
    // Borrowed from caller-owned workspace. The Attention cache writer chooses
    // contiguous or paged storage; compressor does NOT allocate/manage blocks.
    backends::native::Tensor &values;
};

// Complete single-request compression: projection -> incremental group packing
// -> pooling -> RMSNorm -> RoPE -> optional Hadamard -> QDQ. Cache layout and
// block ownership remain outside this model-specific component.
class Compressor {
public:
    using Tensor = backends::native::Tensor;
    Compressor(CompressorConfig config, std::shared_ptr<Tensor> wkv, std::shared_ptr<Tensor> wgate,
               std::shared_ptr<Tensor> ape, std::shared_ptr<Tensor> norm_weight,
               std::shared_ptr<Tensor> frequencies, CompressorBackends backends);
    void reset(CompressorState &state);
    CompressedChunk forward(Tensor &input, CompressorState &state, CompressorWorkspace &workspace);
    uint64_t calls() const { return calls_; }
    uint64_t failures() const { return failures_; }
private:
    const CompressorConfig config_;
    std::shared_ptr<Tensor> wkv_source_, wgate_source_, ape_source_, norm_source_, frequencies_;
    Tensor wkv_, wgate_, ape_, norm_weight_;
    CompressorBackends backends_;
    uint64_t calls_ = 0, failures_ = 0;
};

} // namespace llaisys::models::deepseek_v4

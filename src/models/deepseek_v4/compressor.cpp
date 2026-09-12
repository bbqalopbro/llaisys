#include "compressor.hpp"

#include <limits>

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
constexpr DLDataType bf16{kDLBfloat, 16, 1}, fp32{kDLFloat, 32, 1}, ue8{kDLFloat8_e8m0fnu, 8, 1};
void check(bool c, const char *message) { if (!c) throw std::invalid_argument(message); }
bool dtype(const Tensor &t, DLDataType expected) {
    const auto d = t.dtype(); return d.code == expected.code && d.bits == expected.bits && d.lanes == expected.lanes;
}
CompressorConfig valid(CompressorConfig c) {
    check(c.hidden > 0 && c.dimension > c.rope_dimension && c.rope_dimension > 0 && c.rope_dimension % 2 == 0
          && (c.ratio == 4 || c.ratio == 128) && c.dimension <= 32768 && c.dimension % 32 == 0
          && (c.rotate ? (c.ratio == 4 && !(c.dimension & (c.dimension - 1)))
                       : (c.dimension - c.rope_dimension) % 64 == 0), "invalid native compressor config");
    return c;
}
int64_t coff(CompressorConfig c) { return c.ratio == 4 ? 2 : 1; }
int64_t quantDim(CompressorConfig c) { return c.rotate ? c.dimension : c.dimension - c.rope_dimension; }
bool same(CompressorConfig a, CompressorConfig b) {
    return a.hidden == b.hidden && a.dimension == b.dimension && a.rope_dimension == b.rope_dimension
        && a.ratio == b.ratio && a.rotate == b.rotate;
}
int64_t checkedPosition(int64_t p) { check(p >= 0, "negative compressor position"); return p; }
int64_t checkedTokens(int64_t n, int64_t p, CompressorConfig c) {
    check(n > 0 && n <= INT64_MAX - p && n <= INT64_MAX - c.ratio, "invalid/overflowing compressor token count");
    return n;
}
std::vector<int64_t> groupShape(CompressorConfig c, int64_t p, int64_t n) {
    const int64_t groups = (p % c.ratio + n) / c.ratio;
    return p > 0 && n == 1 && groups ? std::vector<int64_t>{1, coff(c) * c.ratio, c.dimension}
                                   : std::vector<int64_t>{1, groups, coff(c) * c.ratio, c.dimension};
}
core::Runtime &weightRuntime(const std::shared_ptr<Tensor> &w) {
    check(static_cast<bool>(w), "native compressor requires projection weights"); return w->runtime();
}
void kernel(const std::shared_ptr<backends::native::Kernel> &k, const char *op, core::Runtime &runtime) {
    check(k && k->identity().operation == op && k->identity().contract_revision == 1 && &k->runtime() == &runtime,
          "missing/mismatched explicit compressor backend");
}
}

CompressorState::CompressorState(core::Runtime &runtime, CompressorConfig cfg)
    : config(valid(cfg)), kv(runtime, {1, coff(config) * config.ratio, coff(config) * config.dimension}, fp32),
      scores(runtime, kv.shape(), fp32) {}

CompressorWorkspace::CompressorWorkspace(core::Runtime &runtime, CompressorConfig cfg, int64_t p, int64_t n)
    : config(valid(cfg)), position(checkedPosition(p)), tokens(checkedTokens(n, position, config)),
      emitted((position % config.ratio + tokens) / config.ratio),
      input_fp32(runtime, {1, tokens, config.hidden}, fp32),
      projected_kv(runtime, {1, tokens, coff(config) * config.dimension}, fp32),
      projected_scores(runtime, projected_kv.shape(), fp32),
      grouped_kv(runtime, groupShape(config, position, tokens), fp32),
      grouped_scores(runtime, grouped_kv.shape(), fp32),
      pooled_fp32(runtime, {1, emitted, config.dimension}, fp32),
      pooled_bf16(runtime, pooled_fp32.shape(), bf16), normalized(runtime, pooled_fp32.shape(), bf16),
      rotated(runtime, {1, config.rotate ? emitted : 0, config.dimension}, bf16),
      quant_input(runtime, {emitted, quantDim(config)}, bf16), quant_output(runtime, quant_input.shape(), bf16),
      quant_scales(runtime, {emitted, quantDim(config) / (config.rotate ? 32 : 64)}, ue8) {}

Compressor::Compressor(CompressorConfig cfg, std::shared_ptr<Tensor> wkv, std::shared_ptr<Tensor> wgate,
                       std::shared_ptr<Tensor> ape, std::shared_ptr<Tensor> norm,
                       std::shared_ptr<Tensor> frequencies, CompressorBackends backends)
    : config_(valid(cfg)), wkv_source_(std::move(wkv)), wgate_source_(std::move(wgate)), ape_source_(std::move(ape)),
      norm_source_(std::move(norm)), frequencies_(std::move(frequencies)),
      wkv_(weightRuntime(wkv_source_), {coff(config_) * config_.dimension, config_.hidden}, fp32),
      wgate_(wkv_.runtime(), wkv_.shape(), fp32), ape_(wkv_.runtime(), {config_.ratio, coff(config_) * config_.dimension}, fp32),
      norm_weight_(wkv_.runtime(), {config_.dimension}, fp32), backends_(std::move(backends)) {
    auto &runtime = wkv_.runtime();
    for (const auto &pair : {std::make_pair(wkv_source_.get(), &wkv_), std::make_pair(wgate_source_.get(), &wgate_),
                           std::make_pair(ape_source_.get(), &ape_), std::make_pair(norm_source_.get(), &norm_weight_)}) {
        check(pair.first && &pair.first->runtime() == &runtime && pair.first->shape() == pair.second->shape()
              && (dtype(*pair.first, bf16) || dtype(*pair.first, fp32)), "invalid compressor weight shape/dtype/runtime");
        (void)pair.first->view();
    }
    check(frequencies_ && &frequencies_->runtime() == &runtime && frequencies_->shape().size() == 2
          && frequencies_->shape()[0] > 0 && frequencies_->shape()[1] == config_.rope_dimension / 2
          && dtype(*frequencies_, {kDLComplex, 64, 1}) && frequencies_->isContiguous(), "invalid compressor RoPE frequencies");
    kernel(backends_.cast, "tensor_cast", runtime); kernel(backends_.fill, "tensor_fill", runtime);
    kernel(backends_.dense, "dense_linear", runtime); kernel(backends_.prepare, "compressor_prepare", runtime);
    kernel(backends_.pool, "compressor_pool", runtime); kernel(backends_.norm, "rms_norm", runtime);
    kernel(backends_.rotary, "rotary_inplace", runtime);
    kernel(backends_.quantize, config_.rotate ? "fp4_act_quant" : "act_quant", runtime);
    if (config_.rotate) kernel(backends_.hadamard, "hadamard", runtime);
    // The real checkpoint stores some compressor parameters as BF16 while
    // the published model computes projections/norm in FP32. Convert once.
    for (const auto &pair : {std::make_pair(wkv_source_.get(), &wkv_), std::make_pair(wgate_source_.get(), &wgate_),
                           std::make_pair(ape_source_.get(), &ape_), std::make_pair(norm_source_.get(), &norm_weight_)})
        backends_.cast->call({pair.first, pair.second});
}

void Compressor::reset(CompressorState &state) {
    check(same(config_, state.config) && &state.kv.runtime() == &wkv_.runtime(), "foreign compressor state");
    (void)state.kv.view();
    state.failed_ = true;
    backends_.fill->callWithScalars({&state.kv}, {{}, {0.0}});
    backends_.fill->callWithScalars({&state.scores}, {{}, {-std::numeric_limits<double>::infinity()}});
    state.owner_ = this; state.position_ = 0; state.failed_ = false;
}

CompressedChunk Compressor::forward(Tensor &input, CompressorState &state, CompressorWorkspace &w) {
    check(state.owner_ == this && !state.failed_ && !w.failed_, "foreign/uninitialized/failed compressor state or workspace");
    check(same(config_, state.config) && same(config_, w.config) && w.position == state.position_
          && input.shape() == std::vector<int64_t>{1, w.tokens, config_.hidden} && dtype(input, bf16)
          && w.position <= frequencies_->shape()[0] - w.tokens, "compressor position/shape/frequency capacity mismatch");
    for (auto *t : {&input, &state.kv, &w.input_fp32}) {
        check(&t->runtime() == &wkv_.runtime() && t->isContiguous(), "foreign/strided compressor input or workspace");
        (void)t->view();
    }
    ++calls_;
    try {
        backends_.cast->call({&input, &w.input_fp32});
        backends_.dense->call({&w.input_fp32, &wkv_, &w.projected_kv});
        backends_.dense->call({&w.input_fp32, &wgate_, &w.projected_scores});
        backends_.prepare->callWithScalars({&w.projected_kv, &w.projected_scores, &ape_, &state.kv, &state.scores,
                                           &w.grouped_kv, &w.grouped_scores}, {{w.position, config_.ratio}, {}});
        Tensor &output = config_.rotate ? w.rotated : w.normalized;
        if (w.emitted) {
            backends_.pool->call({&w.grouped_kv, &w.grouped_scores, &w.pooled_fp32});
            backends_.cast->call({&w.pooled_fp32, &w.pooled_bf16});
            backends_.norm->call({&w.pooled_bf16, &norm_weight_, &w.normalized});
            const int64_t d = config_.dimension, rd = config_.rope_dimension;
            auto tail = w.normalized.asStrided({1, w.emitted, rd}, {w.emitted * d, d, 1}, d - rd);
            auto freq = frequencies_->asStrided({w.emitted, rd / 2}, {config_.ratio * rd / 2, 1},
                                                 (w.position / config_.ratio) * config_.ratio * rd / 2);
            backends_.rotary->call({tail.get(), freq.get()});
            if (config_.rotate) backends_.hadamard->call({&w.normalized, &w.rotated});
            auto quantized_part = output.asStrided({w.emitted, quantDim(config_)}, {d, 1});
            // Preserve the published contiguous staging around QDQ: a
            // strided non-RoPE slice must not be passed as dense to TileLang.
            backends_.cast->call({quantized_part.get(), &w.quant_input});
            backends_.quantize->call({&w.quant_input, &w.quant_output, &w.quant_scales});
            backends_.cast->call({&w.quant_output, quantized_part.get()});
        }
        state.position_ += w.tokens;
        return {w.position / config_.ratio, w.emitted, output};
    } catch (...) {
        ++failures_; state.failed_ = true; w.failed_ = true; throw;
    }
}

} // namespace llaisys::models::deepseek_v4

#include "block.hpp"
#include <tuple>

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
constexpr DLDataType bf16{kDLBfloat, 16, 1}, fp32{kDLFloat, 32, 1};
void check(bool ok, const char *msg) { if (!ok) throw std::invalid_argument(msg); }
bool dtype(const Tensor &t, DLDataType d) { auto v = t.dtype(); return v.code == d.code && v.bits == d.bits && v.lanes == d.lanes; }
HCConfig hc(BlockConfig c) { return {c.attention.hidden, c.copies, false}; }
BlockConfig config(const std::unique_ptr<Attention> &a, const std::unique_ptr<MoE> &m, int64_t copies) {
    check(a && m && &a->runtime() == &m->runtime() && a->config().hidden == m->config().hidden
          && copies > 0 && copies <= 64, "incompatible Block Attention/MoE/runtime/HC config");
    return {a->config(), m->config(), copies};
}
auto fields(const AttentionConfig &c) {
    return std::make_tuple(c.hidden, c.heads, c.dimension, c.query_rank, c.output_groups, c.output_rank,
        c.rope_dimension, c.window, c.ratio, c.capacity, c.index_heads, c.index_dimension, c.index_topk);
}
auto fields(const MoeConfig &c) { return std::make_tuple(c.hidden, c.intermediate, c.experts, c.top_k, c.vocabulary, c.hash_routing); }
bool same(BlockConfig a, BlockConfig b) { return a.copies == b.copies && fields(a.attention) == fields(b.attention) && fields(a.moe) == fields(b.moe); }
void kernel(const std::shared_ptr<backends::native::Kernel> &k, const char *op, core::Runtime &r) {
    check(k && k->identity().operation == op && k->identity().contract_revision == 1 && &k->runtime() == &r,
          "missing/mismatched Block backend");
}
void disjoint(Tensor &input, std::initializer_list<Tensor *> storage) {
    for (auto *t : storage) check(!input.overlaps(*t), "Block input aliases mutable cache/workspace");
}
void disjoint(Tensor &input, CompressorWorkspace &w) {
    disjoint(input, {&w.input_fp32, &w.projected_kv, &w.projected_scores, &w.grouped_kv, &w.grouped_scores,
        &w.pooled_fp32, &w.pooled_bf16, &w.normalized, &w.rotated, &w.quant_input, &w.quant_output, &w.quant_scales});
}
void disjoint(Tensor &input, HCWorkspace &w) {
    disjoint(input, {&w.input_fp32, &w.inverse_rms, &w.mixes, &w.pre_weights, &w.post_weights, &w.combination, &w.reduced, &w.expanded});
}
void disjoint(Tensor &input, AttentionWorkspace &w) {
    disjoint(input, {&w.query_rank_raw, &w.query_rank, &w.query, &w.inverse_rms, &w.latent_raw, &w.latent,
        &w.quant_input, &w.quant_output, &w.quant_scales, &w.empty_indices, &w.candidates, &w.payload, &w.attended, &w.grouped, &w.output});
    for (auto *q : {&w.query_a, &w.query_b, &w.latent_linear, &w.output_linear}) disjoint(input, {&q->activation, &q->scales});
    if (w.compression) disjoint(input, *w.compression);
    if (w.indexer) {
        auto &i = *w.indexer;
        disjoint(input, {&i.query, &i.rotated, &i.quantized, &i.quant_scales, &i.projected_weights, &i.weights,
                        &i.scores, &i.indices, &i.output, &i.linear.activation, &i.linear.scales});
        disjoint(input, i.compression);
    }
}
}

BlockState::BlockState(core::Runtime &r, BlockConfig cfg) : attention(r, cfg.attention) {}
BlockWorkspace::BlockWorkspace(core::Runtime &r, BlockConfig cfg, int64_t position, int64_t tokens)
    : config(cfg), attention_hc(r, hc(cfg), tokens), feedforward_hc(r, hc(cfg), tokens),
      attention(r, cfg.attention, position, tokens), moe(r, cfg.moe, tokens),
      attention_input(r, {1, tokens, cfg.attention.hidden}, bf16), feedforward_input(r, attention_input.shape(), bf16),
      feedforward_output(r, attention_input.shape(), bf16) {
    check(cfg.attention.hidden == cfg.moe.hidden, "Block workspace model dimensions disagree");
}

Block::Block(std::unique_ptr<Attention> attention, std::unique_ptr<MoE> moe, int64_t copies,
             BlockWeights weights, BlockBackends backends)
    : attention_(std::move(attention)), moe_(std::move(moe)), config_(deepseek_v4::config(attention_, moe_, copies)),
      weights_(std::move(weights)), backends_(std::move(backends)),
      attention_hc_(hc(config_), weights_.attention_hc, backends_.attention_hc),
      feedforward_hc_(hc(config_), weights_.feedforward_hc, backends_.feedforward_hc),
      attention_norm_(attention_->runtime(), {config_.attention.hidden}, fp32),
      feedforward_norm_(attention_->runtime(), attention_norm_.shape(), fp32) {
    auto &r = attention_->runtime();
    for (auto t : {weights_.attention_norm, weights_.feedforward_norm})
        check(t && &t->runtime() == &r && t->isContiguous() && t->shape() == attention_norm_.shape()
              && (dtype(*t, fp32) || dtype(*t, bf16)), "Block requires valid norm weights");
    // HC owns no runtime: explicitly require both HC bundles to match this block.
    for (auto k : {backends_.attention_hc.cast, backends_.feedforward_hc.cast})
        check(k && &k->runtime() == &r, "Block HC runtime mismatch");
    kernel(backends_.cast, "tensor_cast", r); kernel(backends_.norm, "rms_norm", r);
    backends_.cast->call({weights_.attention_norm.get(), &attention_norm_});
    backends_.cast->call({weights_.feedforward_norm.get(), &feedforward_norm_});
}

void Block::reset(BlockState &state) {
    check((!state.owner_ || state.owner_ == this) && fields(state.attention.config) == fields(config_.attention), "foreign Block state");
    state.failed_ = true;
    attention_->reset(state.attention); state.owner_ = this; state.position_ = 0; state.failed_ = false;
}

Block::Tensor &Block::forward(Tensor &hidden, Tensor &ids, BlockState &state, BlockWorkspace &w) {
    check(state.owner_ == this && !state.failed_ && !w.failed_ && same(config_, w.config)
          && state.position_ == w.attention.position && state.attention.position() == state.position_
          && !state.attention.failed() && !w.attention.failed() && !w.moe.failed()
          && !w.attention_hc.failed() && !w.feedforward_hc.failed(), "invalid/foreign/failed Block state or workspace");
    const int64_t n = w.attention.tokens, h = config_.attention.hidden;
    auto &r = attention_->runtime();
    check(hidden.shape() == std::vector<int64_t>{1, n, config_.copies, h} && dtype(hidden, bf16)
          && ids.shape() == std::vector<int64_t>{1, n} && ids.dtype().code == kDLInt
          && (ids.dtype().bits == 32 || ids.dtype().bits == 64) && ids.dtype().lanes == 1, "Block hidden/token shape or dtype mismatch");
    for (auto *input : {&hidden, &ids}) {
        check(&input->runtime() == &r && input->isContiguous(), "foreign/strided Block input"); (void)input->view();
        disjoint(*input, {&w.attention_input, &w.feedforward_input, &w.feedforward_output, &state.attention.cache});
        disjoint(*input, w.attention_hc); disjoint(*input, w.feedforward_hc); disjoint(*input, w.attention);
        auto &m = w.moe;
        disjoint(*input, {&m.input_fp32, &m.logits, &m.hash_ids, &m.route_weights, &m.route_ids, &m.packed_hidden,
            &m.packed_weights, &m.packed_rows, &m.offsets, &m.packed_output, &m.routed_output, &m.shared_output});
        if (state.attention.compression) disjoint(*input, {&state.attention.compression->kv, &state.attention.compression->scores});
        if (state.attention.indexer) disjoint(*input, {&state.attention.indexer->cache,
            &state.attention.indexer->compression.kv, &state.attention.indexer->compression.scores});
    }
    ++calls_;
    try {
        auto &reduced = attention_hc_.pre(hidden, w.attention_hc);
        backends_.norm->call({&reduced, &attention_norm_, &w.attention_input});
        auto &attended = attention_->forward(w.attention_input, state.attention, w.attention);
        auto &residual = attention_hc_.post(attended, w.attention_hc);
        auto &ffn = feedforward_hc_.pre(residual, w.feedforward_hc);
        backends_.norm->call({&ffn, &feedforward_norm_, &w.feedforward_input});
        auto value = w.feedforward_input.asStrided({n, h}, {h, 1});
        auto output = w.feedforward_output.asStrided({n, h}, {h, 1});
        auto tokens = ids.asStrided({n}, {1});
        moe_->forward(*value, *tokens, *output, w.moe);
        auto &result = feedforward_hc_.post(w.feedforward_output, w.feedforward_hc);
        state.position_ += n; return result;
    } catch (...) {
        // Never reuse an advanced Attention cache if a later stage failed.
        state.failed_ = true; w.failed_ = true; ++failures_; throw;
    }
}

} // namespace llaisys::models::deepseek_v4

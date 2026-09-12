#include "attention.hpp"

#include <algorithm>
#include <set>

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
constexpr DLDataType bf16{kDLBfloat, 16, 1}, fp32{kDLFloat, 32, 1}, ue8{kDLFloat8_e8m0fnu, 8, 1}, i32{kDLInt, 32, 1};
void check(bool ok, const char *msg) { if (!ok) throw std::invalid_argument(msg); }
bool dtype(const Tensor &t, DLDataType d) { const auto v = t.dtype(); return v.code == d.code && v.bits == d.bits && v.lanes == d.lanes; }
AttentionConfig valid(AttentionConfig c) {
    check(c.hidden > 0 && c.hidden % 128 == 0 && c.heads >= 16 && c.dimension > c.rope_dimension
          && c.dimension % 128 == 0 && c.rope_dimension > 0 && c.rope_dimension % 2 == 0
          && (c.dimension - c.rope_dimension) % 64 == 0 && c.query_rank > 0 && c.query_rank % 128 == 0
          && c.output_groups > 0 && c.heads % c.output_groups == 0 && c.output_rank > 0 && c.output_rank % 128 == 0
          && c.window > 0 && c.window <= INT32_MAX / 4 && c.capacity >= 128 && c.capacity <= INT32_MAX / 4
          && (c.ratio == 0 || c.ratio == 4 || c.ratio == 128) && c.index_topk > 0
          && c.heads <= INT32_MAX / c.dimension && c.output_groups <= INT32_MAX / c.output_rank,
          "invalid native Attention config");
    return c;
}
CompressorConfig comp(AttentionConfig c) { return {c.hidden, c.dimension, c.rope_dimension, c.ratio, false}; }
IndexerConfig index(AttentionConfig c) { return {c.hidden, c.query_rank, c.index_heads, c.index_dimension, c.rope_dimension, c.index_topk, c.capacity}; }
int64_t cacheSlots(AttentionConfig c) { return c.window + (c.ratio ? c.capacity / c.ratio : 0); }
int64_t position(int64_t p, AttentionConfig c) { check(p >= 0 && p < c.capacity, "invalid Attention position"); return p; }
int64_t tokens(int64_t n, int64_t p, AttentionConfig c) { check(n > 0 && n <= c.capacity - p, "invalid Attention token count"); return n; }
bool same(AttentionConfig a, AttentionConfig b) {
    return a.hidden == b.hidden && a.heads == b.heads && a.dimension == b.dimension && a.query_rank == b.query_rank
        && a.output_groups == b.output_groups && a.output_rank == b.output_rank && a.rope_dimension == b.rope_dimension
        && a.window == b.window && a.ratio == b.ratio && a.capacity == b.capacity && a.index_heads == b.index_heads
        && a.index_dimension == b.index_dimension && a.index_topk == b.index_topk;
}
std::shared_ptr<Tensor> weight(const AttentionWeights &weights, const std::string &name) {
    auto it = weights.find(name); check(it != weights.end() && it->second, "missing native Attention weight"); return it->second;
}
void kernel(const std::shared_ptr<backends::native::Kernel> &k, const char *op, core::Runtime &r) {
    check(k && k->identity().operation == op && k->identity().contract_revision == 1 && &k->runtime() == &r,
          "missing/mismatched Attention backend");
}
void linear(QuantizedLinear &layer, Tensor &input, Tensor &output, LinearWorkspace &w, int64_t rows) {
    auto in = input.asStrided({rows, layer.inputDim()}, {layer.inputDim(), 1});
    auto out = output.asStrided({rows, layer.outputDim()}, {layer.outputDim(), 1});
    layer.forward(*in, *out, w);
}
}

AttentionState::AttentionState(core::Runtime &r, AttentionConfig cfg)
    : config(valid(cfg)), cache(r, {1, cacheSlots(config), config.dimension}, bf16) {
    if (config.ratio) compression = std::make_unique<CompressorState>(r, comp(config));
    if (config.ratio == 4) indexer = std::make_unique<IndexerState>(r, index(config));
}

AttentionWorkspace::AttentionWorkspace(core::Runtime &r, AttentionConfig cfg, int64_t p, int64_t n)
    : config(valid(cfg)), position(deepseek_v4::position(p, config)), tokens(deepseek_v4::tokens(n, position, config)),
      history(position > 0 && tokens > 1 ? std::min(position, config.window - 1) : 0),
      groups(config.ratio ? (position + tokens) / config.ratio : 0),
      window_candidates(position == 0 ? std::min(tokens, config.window) : config.window),
      extra_candidates(config.ratio == 4 ? std::min(config.index_topk, groups) : groups),
      query_a(r, tokens, config.hidden), query_b(r, tokens, config.query_rank), latent_linear(r, tokens, config.hidden),
      output_linear(r, tokens, config.output_groups * config.output_rank),
      query_rank_raw(r, {1, tokens, config.query_rank}, bf16), query_rank(r, query_rank_raw.shape(), bf16),
      query(r, {1, tokens, config.heads, config.dimension}, bf16), inverse_rms(r, {1, tokens, config.heads, 1}, bf16),
      latent_raw(r, {1, tokens, config.dimension}, bf16), latent(r, latent_raw.shape(), bf16),
      quant_input(r, {tokens, config.dimension - config.rope_dimension}, bf16), quant_output(r, quant_input.shape(), bf16),
      quant_scales(r, {tokens, (config.dimension - config.rope_dimension) / 64}, ue8), empty_indices(r, {1, tokens, 0}, i32),
      candidates(r, {1, tokens, window_candidates + extra_candidates}, i32),
      payload(r, {1, position > 0 && tokens == 1 ? 0 : history + tokens + groups, config.dimension}, bf16),
      attended(r, query.shape(), bf16), grouped(r, {1, tokens, config.output_groups, config.output_rank}, bf16),
      output(r, {1, tokens, config.hidden}, bf16) {
    if (config.ratio) compression = std::make_unique<CompressorWorkspace>(r, comp(config), position, tokens);
    if (config.ratio == 4) indexer = std::make_unique<IndexerWorkspace>(r, index(config), position, tokens);
}

Attention::Attention(AttentionConfig cfg, AttentionWeights weights, std::shared_ptr<Tensor> frequencies, AttentionBackends b)
    : config_(valid(cfg)), weights_(std::move(weights)), frequencies_(std::move(frequencies)), backends_(std::move(b)),
      query_a_(weight(weights_, "wq_a.weight"), weight(weights_, "wq_a.scale"), backends_.query_a.quantize, backends_.query_a.gemm),
      query_b_(weight(weights_, "wq_b.weight"), weight(weights_, "wq_b.scale"), backends_.query_b.quantize, backends_.query_b.gemm),
      latent_(weight(weights_, "wkv.weight"), weight(weights_, "wkv.scale"), backends_.latent.quantize, backends_.latent.gemm),
      output_(weight(weights_, "wo_b.weight"), weight(weights_, "wo_b.scale"), backends_.output.quantize, backends_.output.gemm),
      query_norm_(query_a_.runtime(), {config_.query_rank}, fp32), latent_norm_(query_a_.runtime(), {config_.dimension}, fp32),
      output_group_weight_(query_a_.runtime(), {config_.output_groups * config_.output_rank, config_.heads * config_.dimension / config_.output_groups}, bf16) {
    auto &r = query_a_.runtime();
    std::set<std::string> expected{"attn_sink", "wq_a.weight", "wq_a.scale", "q_norm.weight", "wq_b.weight", "wq_b.scale",
        "wkv.weight", "wkv.scale", "kv_norm.weight", "wo_a.weight", "wo_b.weight", "wo_b.scale"};
    if (config_.ratio) for (const auto &name : {"ape", "wkv.weight", "wgate.weight", "norm.weight"}) expected.insert(std::string("compressor.") + name);
    if (config_.ratio == 4) for (const auto &name : {"wq_b.weight", "wq_b.scale", "weights_proj.weight", "compressor.ape",
         "compressor.wkv.weight", "compressor.wgate.weight", "compressor.norm.weight"}) expected.insert(std::string("indexer.") + name);
    check(weights_.size() == expected.size(), "unexpected native Attention weights");
    for (const auto &name : expected) {
        auto t = weight(weights_, name); check(&t->runtime() == &r && t->isContiguous(), "foreign/strided Attention weight"); (void)t->view();
    }
    for (auto *q : {&query_a_, &query_b_, &latent_, &output_}) check(!q->packedFp4() && &q->runtime() == &r, "Attention requires single-runtime W8A8 projections");
    check(query_a_.inputDim() == config_.hidden && query_a_.outputDim() == config_.query_rank
          && query_b_.inputDim() == config_.query_rank && query_b_.outputDim() == config_.heads * config_.dimension
          && latent_.inputDim() == config_.hidden && latent_.outputDim() == config_.dimension
          && output_.inputDim() == config_.output_groups * config_.output_rank && output_.outputDim() == config_.hidden,
          "Attention projection dimensions disagree with config");
    check(frequencies_ && &frequencies_->runtime() == &r && frequencies_->isContiguous()
          && frequencies_->shape() == std::vector<int64_t>{config_.capacity, config_.rope_dimension / 2}
          && dtype(*frequencies_, {kDLComplex, 64, 1}), "invalid Attention RoPE frequencies");
    auto sink = weight(weights_, "attn_sink"); check(dtype(*sink, fp32) && sink->shape() == std::vector<int64_t>{config_.heads}, "invalid Attention sink");
    for (const auto &pair : {std::make_pair(backends_.cast, "tensor_cast"), {backends_.fill, "tensor_fill"}, {backends_.norm, "rms_norm"},
         {backends_.inverse_rms, "row_inv_rms"}, {backends_.row_multiply, "tensor_row_multiply"}, {backends_.rotary, "rotary_inplace"},
         {backends_.inverse_rotary, "rotary_inplace"}, {backends_.latent_qdq, "act_quant"}, {backends_.prepare, "attention_prepare"},
         {backends_.sparse, "sparse_attn"}, {backends_.grouped, "grouped_linear"}}) kernel(pair.first, pair.second, r);
    for (const auto &pair : {std::make_pair("q_norm.weight", &query_norm_), {"kv_norm.weight", &latent_norm_}, {"wo_a.weight", &output_group_weight_}}) {
        auto input = weight(weights_, pair.first);
        check(input->shape() == pair.second->shape() && (dtype(*input, bf16) || dtype(*input, fp32)), "Attention dense weight shape/dtype mismatch");
        backends_.cast->call({input.get(), pair.second});
    }
    if (config_.ratio) compressor_ = std::make_unique<Compressor>(comp(config_), weight(weights_, "compressor.wkv.weight"),
        weight(weights_, "compressor.wgate.weight"), weight(weights_, "compressor.ape"), weight(weights_, "compressor.norm.weight"), frequencies_, backends_.compression);
    if (config_.ratio == 4) {
        IndexerWeights iw{weight(weights_, "indexer.wq_b.weight"), weight(weights_, "indexer.wq_b.scale"), weight(weights_, "indexer.weights_proj.weight"),
            weight(weights_, "indexer.compressor.wkv.weight"), weight(weights_, "indexer.compressor.wgate.weight"), weight(weights_, "indexer.compressor.ape"),
            weight(weights_, "indexer.compressor.norm.weight"), frequencies_};
        indexer_ = std::make_unique<Indexer>(index(config_), iw, backends_.indexer);
    }
}

void Attention::reset(AttentionState &state) {
    check(same(config_, state.config) && &state.cache.runtime() == &query_a_.runtime(), "foreign Attention state");
    check(static_cast<bool>(state.compression) == static_cast<bool>(compressor_)
          && static_cast<bool>(state.indexer) == static_cast<bool>(indexer_), "incomplete Attention request components");
    (void)state.cache.view(); state.failed_ = true;
    backends_.fill->callWithScalars({&state.cache}, {{}, {0.0}});
    if (compressor_) compressor_->reset(*state.compression);
    if (indexer_) indexer_->reset(*state.indexer);
    state.owner_ = this; state.position_ = 0; state.failed_ = false;
}

Tensor &Attention::forward(Tensor &input, AttentionState &state, AttentionWorkspace &w) {
    check(state.owner_ == this && !state.failed_ && !w.failed_ && same(config_, state.config) && same(config_, w.config)
          && state.position_ == w.position, "foreign/uninitialized/failed Attention state, workspace or position");
    check(static_cast<bool>(state.compression) == static_cast<bool>(compressor_)
          && static_cast<bool>(w.compression) == static_cast<bool>(compressor_)
          && static_cast<bool>(state.indexer) == static_cast<bool>(indexer_)
          && static_cast<bool>(w.indexer) == static_cast<bool>(indexer_), "incomplete Attention state/workspace components");
    check(dtype(input, bf16) && input.shape() == std::vector<int64_t>{1, w.tokens, config_.hidden}, "invalid Attention input shape/dtype");
    for (auto *t : {&input, &state.cache, &w.query}) { check(&t->runtime() == &query_a_.runtime() && t->isContiguous(), "foreign/strided Attention input/workspace"); (void)t->view(); }
    check(!input.overlaps(state.cache), "Attention input aliases mutable cache");
    for (auto *t : {&w.query_rank_raw, &w.query_rank, &w.query, &w.inverse_rms, &w.latent_raw, &w.latent, &w.quant_input,
         &w.quant_output, &w.payload, &w.attended, &w.grouped, &w.output}) check(!input.overlaps(*t), "Attention input aliases mutable workspace");
    ++calls_;
    try {
        const int64_t s = w.tokens, h = config_.heads, d = config_.dimension, rd = config_.rope_dimension;
        auto frequencies = frequencies_->asStrided({s, rd / 2}, {rd / 2, 1}, w.position * rd / 2);
        linear(query_a_, input, w.query_rank_raw, w.query_a, s);
        backends_.norm->call({&w.query_rank_raw, &query_norm_, &w.query_rank});
        linear(query_b_, w.query_rank, w.query, w.query_b, s);
        backends_.inverse_rms->call({&w.query, &w.inverse_rms});
        backends_.row_multiply->call({&w.query, &w.inverse_rms});
        auto q_tail = w.query.asStrided({1, s, h, rd}, {s * h * d, h * d, d, 1}, d - rd);
        backends_.rotary->call({q_tail.get(), frequencies.get()});
        linear(latent_, input, w.latent_raw, w.latent_linear, s);
        backends_.norm->call({&w.latent_raw, &latent_norm_, &w.latent});
        auto kv_tail = w.latent.asStrided({1, s, rd}, {s * d, d, 1}, d - rd);
        backends_.rotary->call({kv_tail.get(), frequencies.get()});
        auto non_rope = w.latent.asStrided({s, d - rd}, {d, 1});
        backends_.cast->call({non_rope.get(), &w.quant_input});
        backends_.latent_qdq->call({&w.quant_input, &w.quant_output, &w.quant_scales});
        backends_.cast->call({&w.quant_output, non_rope.get()});
        Tensor *extra = &w.empty_indices;
        if (indexer_) {
            const int64_t offset = w.position > 0 && s == 1 ? config_.window : w.history + s;
            extra = &indexer_->forward(input, w.query_rank, offset, *state.indexer, *w.indexer);
        }
        if (compressor_) {
            auto chunk = compressor_->forward(input, *state.compression, *w.compression);
            if (chunk.count) {
                auto destination = state.cache.asStrided({1, chunk.count, d}, {cacheSlots(config_) * d, d, 1}, (config_.window + chunk.first_group) * d);
                backends_.cast->call({&chunk.values, destination.get()});
            }
        }
        backends_.prepare->callWithScalars({&w.latent, &state.cache, extra, &w.payload, &w.candidates},
                                           {{w.position, config_.window, config_.ratio, config_.index_topk}, {}});
        Tensor *payload = w.position > 0 && s == 1 ? &state.cache : &w.payload;
        backends_.sparse->call({&w.query, payload, &w.attended, weight(weights_, "attn_sink").get(), &w.candidates});
        auto o_tail = w.attended.asStrided({1, s, h, rd}, {s * h * d, h * d, d, 1}, d - rd);
        backends_.inverse_rotary->call({o_tail.get(), frequencies.get()});
        const int64_t g = config_.output_groups, rank = config_.output_rank, gd = h * d / g;
        auto grouped_input = w.attended.asStrided({1, s, g, gd}, {s * h * d, h * d, gd, 1});
        auto grouped_weights = output_group_weight_.asStrided({g, rank, gd}, {rank * gd, gd, 1});
        backends_.grouped->call({grouped_input.get(), grouped_weights.get(), &w.grouped});
        linear(output_, w.grouped, w.output, w.output_linear, s);
        state.position_ += s; return w.output;
    } catch (...) { ++failures_; state.failed_ = true; w.failed_ = true; throw; }
}

} // namespace llaisys::models::deepseek_v4

#include "indexer.hpp"

#include <algorithm>
#include <cmath>

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
constexpr DLDataType bf16{kDLBfloat, 16, 1}, ue8{kDLFloat8_e8m0fnu, 8, 1};
void check(bool ok, const char *message) { if (!ok) throw std::invalid_argument(message); }
bool dtype(const Tensor &t, DLDataType d) {
    return t.dtype().code == d.code && t.dtype().bits == d.bits && t.dtype().lanes == d.lanes;
}
IndexerConfig valid(IndexerConfig c) {
    check(c.hidden > 0 && c.query_rank > 0 && c.query_rank % 128 == 0 && c.heads > 0
          && c.dimension > c.rope_dimension && c.rope_dimension > 0 && c.rope_dimension % 2 == 0
          && c.dimension >= 32 && c.dimension <= 32768 && !(c.dimension & (c.dimension - 1))
          && c.heads <= INT32_MAX / c.dimension && c.topk > 0 && c.capacity >= 4 && c.capacity <= INT32_MAX,
          "invalid native Indexer config");
    return c;
}
CompressorConfig compression(IndexerConfig c) { return {c.hidden, c.dimension, c.rope_dimension, 4, true}; }
bool same(IndexerConfig a, IndexerConfig b) {
    return a.hidden == b.hidden && a.query_rank == b.query_rank && a.heads == b.heads && a.dimension == b.dimension
        && a.rope_dimension == b.rope_dimension && a.topk == b.topk && a.capacity == b.capacity;
}
int64_t position(int64_t p, IndexerConfig c) { check(p >= 0 && p < c.capacity, "invalid Indexer position"); return p; }
int64_t tokens(int64_t n, int64_t p, IndexerConfig c) { check(n > 0 && n <= c.capacity - p, "invalid Indexer token count"); return n; }
void kernel(const std::shared_ptr<backends::native::Kernel> &k, const char *op, core::Runtime &r) {
    check(k && k->identity().operation == op && k->identity().contract_revision == 1 && &k->runtime() == &r,
          "missing/mismatched explicit Indexer backend");
}
}

IndexerState::IndexerState(core::Runtime &r, IndexerConfig cfg)
    : config(valid(cfg)), compression(r, deepseek_v4::compression(config)), cache(r, {1, config.capacity / 4, config.dimension}, bf16) {}

IndexerWorkspace::IndexerWorkspace(core::Runtime &r, IndexerConfig cfg, int64_t p, int64_t n)
    : config(valid(cfg)), position(deepseek_v4::position(p, config)), tokens(deepseek_v4::tokens(n, position, config)),
      candidates((position + tokens) / 4), selected(std::min(config.topk, candidates)),
      linear(r, tokens, config.query_rank), compression(r, deepseek_v4::compression(config), position, tokens),
      query(r, {1, tokens, config.heads, config.dimension}, bf16), rotated(r, query.shape(), bf16), quantized(r, query.shape(), bf16),
      quant_scales(r, {tokens * config.heads, config.dimension / 32}, ue8),
      projected_weights(r, {1, tokens, config.heads}, bf16), weights(r, projected_weights.shape(), bf16),
      scores(r, {1, tokens, candidates}, bf16), indices(r, {1, tokens, selected}, {kDLInt, 64, 1}),
      output(r, indices.shape(), {kDLInt, 32, 1}) {}

Indexer::Indexer(IndexerConfig cfg, IndexerWeights weights, IndexerBackends backends)
    : config_(valid(cfg)), weights_(std::move(weights)), backends_(std::move(backends)),
      query_(weights_.query, weights_.query_scales, backends_.activation_quant, backends_.gemm),
      compressor_(compression(config_), weights_.compressor_kv, weights_.compressor_gate, weights_.ape, weights_.norm,
                  weights_.frequencies, backends_.compression) {
    auto &r = query_.runtime();
    check(!query_.packedFp4() && query_.inputDim() == config_.query_rank && query_.outputDim() == config_.heads * config_.dimension,
          "Indexer query must use matching W8A8 projection");
    check(weights_.head_weights && &weights_.head_weights->runtime() == &r && dtype(*weights_.head_weights, bf16)
          && weights_.head_weights->shape() == std::vector<int64_t>{config_.heads, config_.hidden}
          && weights_.head_weights->isContiguous(), "invalid Indexer head weight tensor");
    check(weights_.frequencies && &weights_.frequencies->runtime() == &r && weights_.frequencies->shape()[0] >= config_.capacity,
          "Indexer frequency capacity/runtime mismatch");
    for (const auto &pair : {std::make_pair(backends_.rotary, "rotary_inplace"), {backends_.hadamard, "hadamard"},
         {backends_.fp4_qdq, "fp4_act_quant"}, {backends_.cast, "tensor_cast"}, {backends_.fill, "tensor_fill"},
         {backends_.dense, "dense_linear"}, {backends_.scale, "tensor_scale"}, {backends_.scores, "indexer_scores"},
         {backends_.mask, "indexer_mask"}, {backends_.topk, "indexer_topk"}, {backends_.remap, "indexer_remap"}})
        kernel(pair.first, pair.second, r);
    (void)weights_.head_weights->view();
}

void Indexer::reset(IndexerState &state) {
    check(same(config_, state.config) && &state.cache.runtime() == &query_.runtime(), "foreign Indexer state");
    (void)state.cache.view(); state.failed_ = true;
    compressor_.reset(state.compression);
    backends_.fill->callWithScalars({&state.cache}, {{}, {0.0}});
    state.owner_ = this; state.failed_ = false;
}

Tensor &Indexer::forward(Tensor &hidden, Tensor &qr, int64_t offset, IndexerState &state, IndexerWorkspace &w) {
    check(state.owner_ == this && !state.failed_ && !w.failed_, "foreign/uninitialized/failed Indexer state or workspace");
    check(same(config_, state.config) && same(config_, w.config) && state.position() == w.position
          && offset >= 0 && offset <= INT32_MAX - w.candidates, "Indexer position/config/offset mismatch");
    check(hidden.shape() == std::vector<int64_t>{1, w.tokens, config_.hidden} && qr.shape() == std::vector<int64_t>{1, w.tokens, config_.query_rank}
          && dtype(hidden, bf16) && dtype(qr, bf16), "Indexer requires matching BF16 hidden and low-rank query");
    for (auto *t : {&hidden, &qr, &state.cache, &w.query}) {
        check(&t->runtime() == &query_.runtime() && t->isContiguous(), "Indexer foreign/strided input or state"); (void)t->view();
    }
    for (auto *t : {&hidden, &qr}) {
        check(!t->overlaps(state.cache) && !t->overlaps(state.compression.kv) && !t->overlaps(state.compression.scores), "Indexer input aliases request state");
        for (auto *out : {&w.query, &w.rotated, &w.quantized, &w.projected_weights, &w.weights, &w.scores, &w.indices, &w.output})
            check(!t->overlaps(*out), "Indexer input aliases mutable workspace");
    }
    ++calls_;
    try {
        const int64_t h = config_.heads, d = config_.dimension, rd = config_.rope_dimension;
        auto input = qr.asStrided({w.tokens, config_.query_rank}, {config_.query_rank, 1});
        auto projected = w.query.asStrided({w.tokens, h * d}, {h * d, 1});
        query_.forward(*input, *projected, w.linear);
        auto tail = w.query.asStrided({1, w.tokens, h, rd}, {w.tokens * h * d, h * d, d, 1}, d - rd);
        auto freq = weights_.frequencies->asStrided({w.tokens, rd / 2}, {rd / 2, 1}, w.position * rd / 2);
        backends_.rotary->call({tail.get(), freq.get()});
        backends_.hadamard->call({&w.query, &w.rotated});
        auto rotated = w.rotated.asStrided({w.tokens * h, d}, {d, 1});
        auto quantized = w.quantized.asStrided({w.tokens * h, d}, {d, 1});
        backends_.fp4_qdq->call({rotated.get(), quantized.get(), &w.quant_scales});
        auto chunk = compressor_.forward(hidden, state.compression, w.compression);
        if (chunk.count) {
            auto destination = state.cache.asStrided({1, chunk.count, d}, {config_.capacity / 4 * d, d, 1}, chunk.first_group * d);
            backends_.cast->call({&chunk.values, destination.get()});
        }
        backends_.dense->call({&hidden, weights_.head_weights.get(), &w.projected_weights});
        backends_.scale->callWithScalars({&w.projected_weights, &w.weights}, {{}, {std::pow(d, -0.5) * std::pow(h, -0.5)}});
        auto keys = state.cache.asStrided({1, w.candidates, d}, {config_.capacity / 4 * d, d, 1});
        backends_.scores->call({&w.quantized, keys.get(), &w.weights, &w.scores});
        backends_.mask->callWithScalars({&w.scores}, {{w.position, 4}, {}});
        backends_.topk->call({&w.scores, &w.indices});
        backends_.remap->callWithScalars({&w.indices, &w.output}, {{w.position, 4, offset}, {}});
        return w.output;
    } catch (...) { ++failures_; state.failed_ = true; w.failed_ = true; throw; }
}

} // namespace llaisys::models::deepseek_v4

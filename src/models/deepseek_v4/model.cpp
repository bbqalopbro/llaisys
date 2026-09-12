#include "model.hpp"

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
constexpr DLDataType bf16{kDLBfloat, 16, 1}, fp32{kDLFloat, 32, 1}, c64{kDLComplex, 64, 1}, i64{kDLInt, 64, 1};
void check(bool ok, const char *msg) { if (!ok) throw std::invalid_argument(msg); }
ModelConfig valid(ModelConfig c, Checkpoint &checkpoint) { c.validate(); checkpoint.validate(c); return c; }
int64_t validPosition(const ModelConfig &c, int64_t p, int64_t n) {
    c.validate(); check(p >= 0 && n > 0 && p < c.capacity && n <= c.capacity - p, "invalid native model position/token count");
    return p;
}
HCConfig hc(const ModelConfig &c) { return {c.hidden, c.copies, true}; }
HCBackends hcBackends(const KernelBindings &ops, bool head) {
    return {ops.at("tensor_cast"), ops.at("dense_linear"), ops.at("row_inv_rms"), ops.at("tensor_row_multiply"),
            ops.at("hc_pre_reduce"), head ? nullptr : ops.at("hc_split_sinkhorn"), head ? nullptr : ops.at("hc_post_mix"),
            head ? ops.at("hc_head_weights") : nullptr};
}
void kernel(const std::shared_ptr<backends::native::Kernel> &k, const char *op, core::Runtime &r) {
    check(k && k->identity().operation == op && k->identity().contract_revision == 1 && &k->runtime() == &r, "missing/mismatched native model backend");
}
std::unique_ptr<Block> assembleBlock(BlockConfig cfg, AttentionWeights weights, const KernelBindings &ops) {
    auto take = [&](const std::string &name) {
        auto it = weights.find(name); check(it != weights.end(), "missing loaded Block weight");
        auto t = it->second; weights.erase(it); return t;
    };
    AttentionWeights aw;
    for (auto it = weights.begin(); it != weights.end();) {
        if (it->first.rfind("attn.", 0) == 0) { aw.emplace(it->first.substr(5), it->second); it = weights.erase(it); }
        else ++it;
    }
    CompressorBackends cb{ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("dense_linear"), ops.at("compressor_prepare"),
        ops.at("compressor_pool"), ops.at("rms_norm"), ops.at("rotary_inplace"), ops.at("latent_qdq"), nullptr};
    auto icb = cb; icb.quantize = ops.at("indexer_qdq"); icb.hadamard = ops.at("hadamard");
    IndexerBackends ib{ops.at("quant_query_rank"), ops.at("indexer_gemm"), ops.at("rotary_inplace"), ops.at("hadamard"), ops.at("indexer_qdq"),
        ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("dense_linear"), ops.at("tensor_scale"), ops.at("indexer_scores"),
        ops.at("indexer_mask"), ops.at("indexer_topk"), ops.at("indexer_remap"), icb};
    AttentionBackends ab{{ops.at("quant_hidden"), ops.at("gemm_query_a")}, {ops.at("quant_query_rank"), ops.at("gemm_query_b")},
        {ops.at("quant_hidden"), ops.at("gemm_latent")}, {ops.at("quant_output"), ops.at("gemm_output")},
        ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("rms_norm"), ops.at("row_inv_rms"), ops.at("tensor_row_multiply"),
        ops.at("rotary_inplace"), ops.at("inverse_rotary"), ops.at("latent_qdq"), ops.at("attention_prepare"), ops.at("sparse_attn"),
        ops.at("grouped_linear"), cb, ib};
    auto attention = std::make_unique<Attention>(cfg.attention, aw, take("frequencies"), ab);
    auto makeExpert = [&](const std::string &prefix, bool shared) {
        auto projection = [&](const std::string &name, bool down) {
            auto w = take(prefix + "." + name + ".weight"), s = take(prefix + "." + name + ".scale");
            return std::make_shared<QuantizedLinear>(w, s, ops.at(down ? "quant_intermediate" : "quant_hidden"),
                ops.at(std::string(shared ? "fp8_" : "fp4_") + (down ? "down" : "gate")));
        };
        auto gate = projection("w1", false), up = projection("w3", false), down = projection("w2", true);
        return std::make_shared<Expert>(gate, up, down, ops.at("expert_activation"));
    };
    auto gate = take("ffn.gate.weight"), selector = take(cfg.moe.hash_routing ? "ffn.gate.tid2eid" : "ffn.gate.bias");
    std::vector<std::shared_ptr<Expert>> experts;
    for (int64_t e = 0; e < cfg.moe.experts; ++e) experts.push_back(makeExpert("ffn.experts." + std::to_string(e), false));
    auto shared = makeExpert("ffn.shared_experts", true);
    MoeBackends mb{ops.at("tensor_cast"), ops.at("dense_linear"), ops.at("row_gather"), ops.at("router"),
        ops.at("moe_dispatch"), ops.at("moe_combine"), ops.at("moe_finalize")};
    auto moe = std::make_unique<MoE>(cfg.moe, gate, selector, experts, shared, mb);
    BlockWeights bw{{take("hc_attn_fn"), take("hc_attn_scale"), take("hc_attn_base")},
                    {take("hc_ffn_fn"), take("hc_ffn_scale"), take("hc_ffn_base")}, take("attn_norm.weight"), take("ffn_norm.weight")};
    auto hcb = hcBackends(ops, false);
    check(weights.empty(), "unconsumed native Block parameters");
    auto result = std::make_unique<Block>(std::move(attention), std::move(moe), cfg.copies, bw,
                                        BlockBackends{hcb, hcb, ops.at("tensor_cast"), ops.at("rms_norm")});
    ops.at("tensor_cast")->runtime().synchronize(); // Finish load-time conversions before temporary source owners die.
    return result;
}
}

ModelState::ModelState(core::Runtime &r, const ModelConfig &c) {
    c.validate(); layers.reserve(c.layers);
    for (int64_t i = 0; i < c.layers; ++i) layers.push_back(std::make_unique<BlockState>(r, c.block(i)));
}
ModelWorkspace::ModelWorkspace(core::Runtime &r, const ModelConfig &cfg, int64_t p, int64_t n, bool emit)
    : position(validPosition(cfg, p, n)), tokens(n), emit_logits(emit), hidden(r, {1, n, cfg.copies, cfg.hidden}, bf16),
      normalized(r, {1, emit ? n : 0, cfg.hidden}, bf16), head_input(r, {emit ? 1 : 0, cfg.hidden}, fp32),
      logits(r, {emit ? 1 : 0, cfg.vocabulary}, fp32), next_ids(r, {emit ? 1 : 0}, i64) {
    layers.reserve(cfg.layers);
    for (int64_t i = 0; i < cfg.layers; ++i) layers.push_back(std::make_unique<BlockWorkspace>(r, cfg.block(i), p, n));
    if (emit) head = std::make_unique<HCWorkspace>(r, hc(cfg), n);
}

Model::Model(core::Runtime &r, ModelConfig cfg, Checkpoint &checkpoint, ModelBackends backends)
    : runtime_(r), config_(valid(std::move(cfg), checkpoint)), backends_(std::move(backends)),
      norm_(r, {config_.hidden}, fp32), head_weight_(r, {config_.vocabulary, config_.hidden}, fp32) {
    check(backends_.layers.size() == static_cast<size_t>(config_.layers), "missing per-layer native model backends");
    auto &ops = backends_.global;
    for (auto op : {"tensor_cast", "rms_norm", "dense_linear", "token_embedding", "greedy_argmax", "rotary_frequencies"}) kernel(ops.at(op), op, r);
    embedding_ = checkpoint.load("embed.weight", r);
    auto norm = checkpoint.load("norm.weight", r), head = checkpoint.load("head.weight", r);
    ops.at("tensor_cast")->call({norm.get(), &norm_}); ops.at("tensor_cast")->call({head.get(), &head_weight_});
    HCWeights hw{checkpoint.load("hc_head_fn", r), checkpoint.load("hc_head_scale", r), checkpoint.load("hc_head_base", r)};
    head_ = std::make_unique<HyperConnection>(hc(config_), hw, hcBackends(ops, true));
    for (bool compressed : {false, true}) {
        auto f = std::make_shared<Tensor>(r, std::vector<int64_t>{config_.capacity, config_.rope_dimension / 2}, c64);
        ops.at("rotary_frequencies")->callWithScalars({f.get()}, {{compressed ? config_.original_sequence : 0},
            {compressed ? config_.compressed_theta : config_.rope_theta, config_.rope_factor, config_.beta_fast, config_.beta_slow}});
        frequencies_.emplace(compressed, f);
    }
    for (int64_t i = 0; i < config_.layers; ++i) {
        const auto prefix = "layers." + std::to_string(i) + "."; AttentionWeights weights;
        for (auto it = checkpoint.records().lower_bound(prefix); it != checkpoint.records().end() && it->first.rfind(prefix, 0) == 0; ++it)
            weights.emplace(it->first.substr(prefix.size()), checkpoint.load(it->first, r));
        weights.emplace("frequencies", frequencies_.at(config_.ratios[i] != 0));
        layers_.push_back(assembleBlock(config_.block(i), std::move(weights), backends_.layers[i]));
    }
    checkpoint.finish(); r.synchronize();
}

void Model::reset(ModelState &state) {
    check((!state.owner_ || state.owner_ == this) && state.layers.size() == layers_.size(), "foreign native model state");
    for (const auto &s : state.layers) check(static_cast<bool>(s), "missing model layer state");
    state.failed_ = true;
    for (size_t i = 0; i < layers_.size(); ++i) layers_[i]->reset(*state.layers[i]);
    state.owner_ = this; state.position_ = 0; state.failed_ = false;
}

ModelOutput Model::forward(Tensor &ids, ModelState &state, ModelWorkspace &w) {
    check(state.owner_ == this && !state.failed_ && !w.failed_ && state.position_ == w.position
          && w.tokens > 0 && w.tokens <= config_.capacity - w.position && state.layers.size() == layers_.size() && w.layers.size() == layers_.size()
          && static_cast<bool>(w.head) == w.emit_logits, "invalid native model state/workspace");
    check(ids.shape() == std::vector<int64_t>{1, w.tokens} && ids.dtype().code == kDLInt && ids.dtype().lanes == 1
          && (ids.dtype().bits == 32 || ids.dtype().bits == 64) && ids.isContiguous() && &ids.runtime() == &runtime_, "invalid native model token IDs");
    (void)ids.view();
    for (size_t i = 0; i < layers_.size(); ++i)
        check(state.layers[i] && w.layers[i] && state.layers[i]->position() == w.position && !state.layers[i]->failed()
              && !w.layers[i]->failed() && w.layers[i]->attention.position == w.position && w.layers[i]->attention.tokens == w.tokens,
              "inconsistent nested model state/workspace");
    ++calls_;
    try {
        auto &ops = backends_.global;
        ops.at("token_embedding")->call({embedding_.get(), &ids, &w.hidden});
        Tensor *hidden = &w.hidden;
        for (size_t i = 0; i < layers_.size(); ++i) hidden = &layers_[i]->forward(*hidden, ids, *state.layers[i], *w.layers[i]);
        if (w.emit_logits) {
            auto &reduced = head_->head(*hidden, *w.head); ops.at("rms_norm")->call({&reduced, &norm_, &w.normalized});
            auto last = w.normalized.asStrided({1, config_.hidden}, {config_.hidden, 1}, (w.tokens - 1) * config_.hidden);
            ops.at("tensor_cast")->call({last.get(), &w.head_input});
            ops.at("dense_linear")->call({&w.head_input, &head_weight_, &w.logits});
            ops.at("greedy_argmax")->call({&w.logits, &w.next_ids});
        }
        state.position_ += w.tokens;
        return {w.emit_logits ? &w.logits : nullptr, w.emit_logits ? &w.next_ids : nullptr};
    } catch (...) { state.failed_ = true; w.failed_ = true; ++failures_; throw; }
}
} // namespace llaisys::models::deepseek_v4

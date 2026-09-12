#include "hyperconnection.hpp"

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
constexpr DLDataType bf16{kDLBfloat, 16, 1}, fp32{kDLFloat, 32, 1};
void check(bool ok, const char *msg) { if (!ok) throw std::invalid_argument(msg); }
bool dtype(const Tensor &t, DLDataType d) { auto v = t.dtype(); return v.code == d.code && v.bits == d.bits && v.lanes == d.lanes; }
HCConfig valid(HCConfig c) {
    check(c.hidden > 0 && c.copies > 0 && c.copies <= 64 && c.hidden <= INT32_MAX / c.copies,
          "invalid native HC config"); return c;
}
int64_t tokens(int64_t n) { check(n > 0 && n <= INT32_MAX, "HC requires positive token count"); return n; }
int64_t mixes(HCConfig c) { return c.head ? c.copies : (2 + c.copies) * c.copies; }
bool same(HCConfig a, HCConfig b) { return a.hidden == b.hidden && a.copies == b.copies && a.head == b.head; }
void kernel(const std::shared_ptr<backends::native::Kernel> &k, const char *op, core::Runtime &r) {
    check(k && k->identity().operation == op && k->identity().contract_revision == 1 && &k->runtime() == &r,
          "missing/mismatched HC backend");
}
}

HCWorkspace::HCWorkspace(core::Runtime &r, HCConfig cfg, int64_t n)
    : config(valid(cfg)), tokens(deepseek_v4::tokens(n)),
      input_fp32(r, {1, tokens, config.copies * config.hidden}, fp32), inverse_rms(r, {1, tokens, 1}, fp32),
      mixes(r, {1, tokens, deepseek_v4::mixes(config)}, fp32), pre_weights(r, {1, tokens, config.copies}, fp32),
      post_weights(r, {1, config.head ? 0 : tokens, config.copies}, fp32),
      combination(r, {1, config.head ? 0 : tokens, config.copies, config.copies}, fp32),
      reduced(r, {1, tokens, config.hidden}, bf16), expanded(r, {1, config.head ? 0 : tokens, config.copies, config.hidden}, bf16) {}

HyperConnection::HyperConnection(HCConfig cfg, HCWeights weights, HCBackends backends)
    : config_(valid(cfg)), weights_(std::move(weights)), backends_(std::move(backends)) {
    check(weights_.projection && weights_.scale && weights_.base, "missing HC weights");
    auto &r = weights_.projection->runtime();
    for (auto t : {weights_.projection, weights_.scale, weights_.base}) {
        check(&t->runtime() == &r && dtype(*t, fp32) && t->isContiguous(), "HC weights must be contiguous FP32 in one runtime");
        (void)t->view();
    }
    check(weights_.projection->shape() == std::vector<int64_t>{mixes(config_), config_.copies * config_.hidden}
          && weights_.scale->shape() == std::vector<int64_t>{config_.head ? 1 : 3}
          && weights_.base->shape() == std::vector<int64_t>{mixes(config_)}, "HC weight shape mismatch");
    kernel(backends_.cast, "tensor_cast", r); kernel(backends_.dense, "dense_linear", r);
    kernel(backends_.inverse_rms, "row_inv_rms", r); kernel(backends_.row_multiply, "tensor_row_multiply", r);
    kernel(backends_.pre_reduce, "hc_pre_reduce", r);
    if (config_.head) kernel(backends_.head_weights, "hc_head_weights", r);
    else { kernel(backends_.split, "hc_split_sinkhorn", r); kernel(backends_.post_mix, "hc_post_mix", r); }
}

void HyperConnection::validate(Tensor &input, HCWorkspace &w, bool post) const {
    check(same(config_, w.config) && !w.failed_ && (!w.owner_ || w.owner_ == this), "foreign/failed HC workspace");
    auto &r = weights_.projection->runtime();
    check(&input.runtime() == &r && dtype(input, bf16) && input.isContiguous(), "HC hidden must be contiguous BF16 in owning runtime");
    check(input.shape() == (post ? std::vector<int64_t>{1, w.tokens, config_.hidden}
                               : std::vector<int64_t>{1, w.tokens, config_.copies, config_.hidden}), "HC input shape mismatch");
    (void)input.view();
    for (auto *t : {&w.input_fp32, &w.inverse_rms, &w.mixes, &w.pre_weights, &w.post_weights, &w.combination, &w.reduced, &w.expanded}) {
        check(&t->runtime() == &r, "foreign HC workspace runtime"); (void)t->view();
        // post reads the branch only and writes expanded. Reduced is a valid
        // branch input (e.g. identity branch), other workspace aliases are not.
        if (!post || t != &w.reduced) check(!input.overlaps(*t), "HC input aliases mutable workspace");
    }
    check(post ? w.ready_ && w.residual_ && w.owner_ == this : !w.ready_, "HC pre/post ordering violation");
    if (post) check(!input.overlaps(*w.residual_), "HC branch aliases retained residual");
}

void HyperConnection::project(Tensor &input, HCWorkspace &w) {
    auto flat = input.asStrided({1, w.tokens, config_.copies * config_.hidden},
                               {w.tokens * config_.copies * config_.hidden, config_.copies * config_.hidden, 1});
    backends_.cast->call({flat.get(), &w.input_fp32});
    backends_.inverse_rms->call({&w.input_fp32, &w.inverse_rms});
    backends_.dense->call({&w.input_fp32, weights_.projection.get(), &w.mixes});
    backends_.row_multiply->call({&w.mixes, &w.inverse_rms});
}

HyperConnection::Tensor &HyperConnection::pre(Tensor &input, HCWorkspace &w) {
    check(!config_.head, "head-only HC has no pre/post pair"); validate(input, w, false); ++calls_;
    try {
        w.owner_ = this;
        w.residual_ = input.asStrided(input.shape(), {w.tokens * config_.copies * config_.hidden, config_.copies * config_.hidden, config_.hidden, 1});
        project(input, w);
        // The exported TileLang ABI flattens [B,S] into its symbolic M.
        auto m = w.mixes.asStrided({w.tokens, mixes(config_)}, {mixes(config_), 1});
        auto pre = w.pre_weights.asStrided({w.tokens, config_.copies}, {config_.copies, 1});
        auto post = w.post_weights.asStrided({w.tokens, config_.copies}, {config_.copies, 1});
        auto comb = w.combination.asStrided({w.tokens, config_.copies, config_.copies}, {config_.copies * config_.copies, config_.copies, 1});
        backends_.split->call({m.get(), weights_.scale.get(), weights_.base.get(), pre.get(), post.get(), comb.get()});
        auto fp = w.input_fp32.asStrided(input.shape(), {w.tokens * config_.copies * config_.hidden, config_.copies * config_.hidden, config_.hidden, 1});
        backends_.pre_reduce->call({fp.get(), &w.pre_weights, &w.reduced});
        w.ready_ = true; return w.reduced;
    } catch (...) { w.failed_ = true; w.ready_ = false; ++failures_; throw; }
}

HyperConnection::Tensor &HyperConnection::post(Tensor &branch, HCWorkspace &w) {
    check(!config_.head, "head-only HC has no pre/post pair"); validate(branch, w, true); ++calls_;
    try {
        backends_.post_mix->call({&branch, w.residual_.get(), &w.post_weights, &w.combination, &w.expanded});
        w.ready_ = false; return w.expanded;
    } catch (...) { w.failed_ = true; w.ready_ = false; ++failures_; throw; }
}

HyperConnection::Tensor &HyperConnection::head(Tensor &input, HCWorkspace &w) {
    check(config_.head, "mixing HC requires pre/post, not head"); validate(input, w, false); ++calls_;
    try {
        w.owner_ = this; project(input, w);
        backends_.head_weights->call({&w.mixes, weights_.scale.get(), weights_.base.get(), &w.pre_weights});
        auto fp = w.input_fp32.asStrided(input.shape(), {w.tokens * config_.copies * config_.hidden, config_.copies * config_.hidden, config_.hidden, 1});
        backends_.pre_reduce->call({fp.get(), &w.pre_weights, &w.reduced}); return w.reduced;
    } catch (...) { w.failed_ = true; ++failures_; throw; }
}

} // namespace llaisys::models::deepseek_v4

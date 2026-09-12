#include "moe.hpp"

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
constexpr DLDataType bf16{kDLBfloat, 16, 1}, fp32{kDLFloat, 32, 1}, i32{kDLInt, 32, 1}, i64{kDLInt, 64, 1};
void check(bool value, const char *message) { if (!value) throw std::invalid_argument(message); }
bool dtype(Tensor &x, DLDataType expected) {
    const auto d = x.dtype();
    return d.code == expected.code && d.bits == expected.bits && d.lanes == expected.lanes;
}
MoeConfig valid(MoeConfig c) {
    check(c.hidden > 0 && c.hidden % 128 == 0 && c.intermediate > 0 && c.intermediate % 128 == 0
          && c.experts > 0 && c.experts <= INT32_MAX && c.top_k > 0 && c.top_k <= c.experts
          && c.vocabulary > 0 && c.vocabulary <= INT32_MAX, "invalid native MoE config");
    return c;
}
int64_t validRows(int64_t rows, MoeConfig c) {
    check(rows >= 0 && rows <= INT64_MAX / c.top_k, "native MoE row count overflows packed capacity");
    return rows;
}
core::Runtime &weightRuntime(const std::shared_ptr<Tensor> &weight) {
    check(static_cast<bool>(weight), "native MoE requires gate weights");
    return weight->runtime();
}
void kernel(const std::shared_ptr<backends::native::Kernel> &k, const char *op, core::Runtime &runtime) {
    check(k && k->identity().operation == op && k->identity().contract_revision == 1 && &k->runtime() == &runtime,
          "missing/mismatched explicit native MoE backend");
}
bool same(MoeConfig a, MoeConfig b) {
    return a.hidden == b.hidden && a.intermediate == b.intermediate && a.experts == b.experts
        && a.top_k == b.top_k && a.vocabulary == b.vocabulary && a.hash_routing == b.hash_routing;
}
std::shared_ptr<Tensor> rowsView(Tensor &x, int64_t start, int64_t count) {
    return x.asStrided({count, x.shape()[1]}, {x.shape()[1], 1}, start * x.shape()[1]);
}
}

ExpertWorkspace::ExpertWorkspace(core::Runtime &runtime, int64_t rows, int64_t hidden, int64_t intermediate)
    : input_quant(runtime, rows, hidden), intermediate_quant(runtime, rows, intermediate),
      gate(runtime, {rows, intermediate}, bf16), up(runtime, {rows, intermediate}, bf16),
      activation(runtime, {rows, intermediate}, bf16) {}

Expert::Expert(std::shared_ptr<QuantizedLinear> gate, std::shared_ptr<QuantizedLinear> up,
               std::shared_ptr<QuantizedLinear> down, std::shared_ptr<Kernel> activation)
    : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)), activation_(std::move(activation)) {
    check(gate_ && up_ && down_ && activation_, "native Expert requires all projections/backends");
    check(gate_->inputDim() == up_->inputDim() && gate_->outputDim() == up_->outputDim()
          && down_->inputDim() == gate_->outputDim() && down_->outputDim() == gate_->inputDim(),
          "native Expert projection dimensions do not compose");
    kernel(activation_, "expert_activation", activation_->runtime());
    check(&gate_->runtime() == &activation_->runtime() && &up_->runtime() == &activation_->runtime()
          && &down_->runtime() == &activation_->runtime(), "native Expert cannot mix projection runtimes");
}

void Expert::forward(Tensor &input, Tensor &output, ExpertWorkspace &workspace, Tensor *routing_weights) {
    gate_->forward(input, workspace.gate, workspace.input_quant);
    up_->forward(input, workspace.up, workspace.input_quant);
    if (routing_weights) activation_->call({&workspace.gate, &workspace.up, routing_weights, &workspace.activation});
    else activation_->call({&workspace.gate, &workspace.up, &workspace.activation});
    down_->forward(workspace.activation, output, workspace.intermediate_quant);
}

MoeWorkspace::MoeWorkspace(core::Runtime &runtime, MoeConfig cfg, int64_t count)
    : config(valid(cfg)), rows(validRows(count, config)),
      input_fp32(runtime, {rows, config.hidden}, fp32), logits(runtime, {rows, config.experts}, fp32),
      hash_ids(runtime, {rows, config.top_k}, i32), route_weights(runtime, {rows, config.top_k}, fp32),
      route_ids(runtime, {rows, config.top_k}, config.hash_routing ? i32 : i64),
      packed_hidden(runtime, {rows * config.top_k, config.hidden}, bf16),
      packed_weights(runtime, {rows * config.top_k, 1}, fp32), packed_rows(runtime, {rows * config.top_k}, i64),
      offsets(runtime, {config.experts + 1}, i64), packed_output(runtime, {rows * config.top_k, config.hidden}, bf16),
      routed_output(runtime, {rows, config.hidden}, fp32), shared_output(runtime, {rows, config.hidden}, bf16),
      host_offsets_(config.experts + 1) {}

ExpertWorkspace &MoeWorkspace::scratch(int64_t count) {
    check(count > 0 && count <= rows, "invalid expert batch size (duplicate per-token expert IDs?)");
    auto &entry = scratch_[count];
    if (!entry) entry = std::make_unique<ExpertWorkspace>(input_fp32.runtime(), count, config.hidden, config.intermediate);
    return *entry;
}

MoE::MoE(MoeConfig cfg, std::shared_ptr<Tensor> gate_weight, std::shared_ptr<Tensor> selector,
         std::vector<std::shared_ptr<Expert>> experts, std::shared_ptr<Expert> shared, MoeBackends backends)
    : config_(valid(cfg)), gate_weight_(std::move(gate_weight)), selector_(std::move(selector)),
      gate_fp32_(weightRuntime(gate_weight_), {config_.experts, config_.hidden}, fp32),
      experts_(std::move(experts)), shared_(std::move(shared)), backends_(std::move(backends)) {
    auto &runtime = gate_fp32_.runtime();
    check(gate_weight_->shape() == std::vector<int64_t>{config_.experts, config_.hidden}
          && (dtype(*gate_weight_, bf16) || dtype(*gate_weight_, fp32)), "invalid native MoE gate weights");
    check(selector_ && &selector_->runtime() == &runtime, "native MoE selector belongs to another runtime");
    check(config_.hash_routing
          ? (selector_->shape() == std::vector<int64_t>{config_.vocabulary, config_.top_k}
             && (dtype(*selector_, i32) || dtype(*selector_, i64)))
          : (selector_->shape() == std::vector<int64_t>{config_.experts} && dtype(*selector_, fp32)),
          "invalid native MoE hash table or routing bias");
    check(experts_.size() == static_cast<size_t>(config_.experts) && shared_, "native MoE requires every expert and shared expert");
    auto validateExpert = [&](const std::shared_ptr<Expert> &e) {
        check(e && e->hiddenDim() == config_.hidden && e->intermediateDim() == config_.intermediate
              && &e->runtime() == &runtime, "native MoE expert shape/runtime mismatch");
    };
    for (auto &e : experts_) validateExpert(e);
    validateExpert(shared_);
    kernel(backends_.cast, "tensor_cast", runtime); kernel(backends_.dense, "dense_linear", runtime);
    kernel(backends_.gather, "row_gather", runtime); kernel(backends_.router, "router", runtime);
    kernel(backends_.dispatch, "moe_dispatch", runtime); kernel(backends_.combine, "moe_combine", runtime);
    kernel(backends_.finalize, "moe_finalize", runtime);
    // Low-frequency preparation once per loaded layer, not once per token.
    backends_.cast->call({gate_weight_.get(), &gate_fp32_});
    if (config_.hash_routing && dtype(*selector_, i64)) {
        // Actual MP1 checkpoint stores tid2eid as I64; published model uses
        // I32. This is a checked, explicit load-time conversion, not truncation.
        auto table = std::make_shared<Tensor>(runtime, selector_->shape(), i32);
        backends_.cast->call({selector_.get(), table.get()});
        selector_ = std::move(table);
    }
}

void MoE::forward(Tensor &input, Tensor &token_ids, Tensor &output, MoeWorkspace &w) {
    check(!w.failed(), "native MoE workspace invalid after a previous execution error; discard it");
    check(same(config_, w.config) && input.shape() == std::vector<int64_t>{w.rows, config_.hidden}
          && output.shape() == input.shape() && dtype(input, bf16) && dtype(output, bf16)
          && token_ids.shape() == std::vector<int64_t>{w.rows} && (dtype(token_ids, i32) || dtype(token_ids, i64)),
          "native MoE input/output/workspace shape or dtype mismatch");
    auto &runtime = gate_fp32_.runtime();
    for (Tensor *t : {&input, &token_ids, &output, &w.input_fp32}) {
        check(&t->runtime() == &runtime && t->isContiguous(), "native MoE requires contiguous tensors in its runtime");
        (void)t->view();
    }
    for (Tensor *t : {&input, &token_ids, gate_weight_.get(), selector_.get(), &w.input_fp32, &w.logits,
                     &w.hash_ids, &w.route_weights, &w.route_ids, &w.packed_hidden, &w.packed_weights,
                     &w.packed_rows, &w.offsets, &w.packed_output, &w.routed_output, &w.shared_output})
        check(!output.overlaps(*t), "native MoE output aliases input/weights/workspace");
    ++calls_;
    if (!w.rows) return;
    try {
        backends_.cast->call({&input, &w.input_fp32});
        backends_.dense->call({&w.input_fp32, &gate_fp32_, &w.logits});
        Tensor *selector = selector_.get();
        if (config_.hash_routing) {
            backends_.gather->call({selector, &token_ids, &w.hash_ids});
            selector = &w.hash_ids;
        }
        backends_.router->call({&w.logits, selector, &w.route_weights, &w.route_ids});
        backends_.dispatch->call({&input, &w.route_weights, &w.route_ids, &w.packed_hidden,
                                 &w.packed_weights, &w.packed_rows, &w.offsets});
        // Deliberate reference barrier. Future grouped/EP implementations must
        // replace this execution strategy explicitly, not hide host transfers.
        w.offsets.download(w.host_offsets_.data(), w.offsets.bytes());
        ++w.offset_downloads_;
        check(w.host_offsets_.front() == 0 && w.host_offsets_.back() == w.rows * config_.top_k,
              "invalid native MoE dispatch endpoints");
        for (int64_t e = 0; e < config_.experts; ++e) {
            const auto start = w.host_offsets_[e], end = w.host_offsets_[e + 1];
            check(start >= 0 && end >= start && end <= w.rows * config_.top_k && end - start <= w.rows,
                  "invalid native MoE dispatch offsets");
        }
        for (int64_t e = 0; e < config_.experts; ++e) {
            const auto start = w.host_offsets_[e], count = w.host_offsets_[e + 1] - start;
            if (!count) continue;
            auto x = rowsView(w.packed_hidden, start, count), weights = rowsView(w.packed_weights, start, count);
            auto y = rowsView(w.packed_output, start, count);
            experts_[e]->forward(*x, *y, w.scratch(count), weights.get());
            ++w.expert_executions_;
        }
        backends_.combine->call({&w.packed_output, &w.packed_rows, &w.offsets, &w.routed_output});
        shared_->forward(input, w.shared_output, w.scratch(w.rows));
        backends_.finalize->call({&w.routed_output, &w.shared_output, &output});
    } catch (...) {
        ++failures_;
        w.failed_ = true;
        throw;
    }
}

} // namespace llaisys::models::deepseek_v4

#include <ATen/Context.h>
#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/backends/hadamard/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/block.hpp"
#include "native_fixture_utils.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

using namespace native_fixture;
using namespace llaisys::models::deepseek_v4;
using Kernel = llaisys::backends::native::Kernel;
using StructuralKernel = llaisys::backends::aten::StructuralKernel;
using TileKernel = llaisys::backends::tilelang::NativeKernel;
using Bindings = std::map<std::string, std::shared_ptr<Kernel>>;

// Test-only assembly from exact fixture weights. The production full-model
// config/checkpoint loader is a separate deliverable, not this fixture parser.
static std::unique_ptr<Block> makeBlock(BlockConfig cfg, AttentionWeights weights, const Bindings &ops,
                                      BlockBackends block_backends) {
    auto take = [&](const std::string &name) {
        auto it = weights.find(name); check(it != weights.end(), "missing Block weight: " + name);
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
    check(weights.empty(), "unconsumed Block weights");
    return std::make_unique<Block>(std::move(attention), std::move(moe), cfg.copies, bw, block_backends);
}

class OnceFail final : public Kernel {
public:
    explicit OnceFail(std::shared_ptr<Kernel> k) : Kernel(k->runtime(), {"test-user-library", "v1", k->identity().operation, 1}), next_(k) {}
    void call(const std::vector<Tensor *> &a) override { if (++calls_ == 1) throw std::runtime_error("injected final HC post failure"); next_->call(a); }
    void close() override {}
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return calls_ ? 1 : 0; }
private:
    std::shared_ptr<Kernel> next_; uint64_t calls_ = 0;
};
struct Step { int64_t position, tokens; std::map<std::string, Value> values; };

int main(int argc, char **argv) {
    try {
        check(argc == 2 && std::getenv("SLURM_JOB_ID"), "Block verification requires Slurm allocation");
        noPython(); llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0); auto &r = llaisys::core::context().runtime();
        std::ifstream manifest(argv[1]); std::string magic, tl, aten, had; size_t layers;
        manifest >> magic >> tl >> aten >> had >> layers;
        check(magic == "LLAISYS_NATIVE_BLOCK_V1" && layers == 3 && aten == StructuralKernel::compiledVersion(), "invalid Block fixture");
        size_t comparisons = 0, negatives = 0, steps_run = 0, recoveries = 0; std::ostringstream counts; counts << '[';
        auto reject = [&](auto &&fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
            check(failed, "invalid Block call accepted"); ++negatives; };
        for (size_t layer = 0; layer < layers; ++layer) {
            std::string name, score; BlockConfig cfg; auto &a = cfg.attention; auto &m = cfg.moe;
            double eps, hc_eps, route, limit; int hash; int64_t iters; size_t nweights;
            manifest >> name >> a.hidden >> a.heads >> a.dimension >> a.query_rank >> a.output_groups >> a.output_rank
                     >> a.rope_dimension >> a.window >> a.ratio >> a.capacity >> a.index_heads >> a.index_dimension >> a.index_topk >> eps
                     >> cfg.copies >> m.intermediate >> m.experts >> m.top_k >> m.vocabulary >> hash >> route >> limit >> score >> hc_eps >> iters >> nweights;
            m.hidden = a.hidden; m.hash_routing = hash; check(nweights > 1500 && nweights < 1600 && iters == 20, "incomplete actual Block weights");
            AttentionWeights weights;
            for (size_t i = 0; i < nweights; ++i) { auto value = read(manifest, r); check(weights.emplace(value.first, value.second.tensor).second, "duplicate Block weight"); }
            Bindings ops; llaisys::backends::aten::StructuralOptions options;
            options.epsilon = eps; options.hash_routing = hash; options.route_scale = route; options.activation_limit = limit; options.score_function = score;
            for (auto op : {"tensor_cast", "tensor_fill", "dense_linear", "compressor_prepare", "compressor_pool", "rms_norm",
                 "rotary_inplace", "tensor_scale", "indexer_scores", "indexer_mask", "indexer_topk", "indexer_remap",
                 "row_inv_rms", "tensor_row_multiply", "attention_prepare", "grouped_linear", "row_gather", "router",
                 "moe_dispatch", "moe_combine", "moe_finalize", "expert_activation", "hc_pre_reduce", "hc_post_mix"})
                ops.emplace(op, std::make_shared<StructuralKernel>(r, llaisys::backends::native::KernelIdentity{"aten-reference", aten, op, 1}, options));
            options.inverse_rotary = true;
            ops.emplace("inverse_rotary", std::make_shared<StructuralKernel>(r, llaisys::backends::native::KernelIdentity{"aten-reference", aten, "rotary_inplace", 1}, options));
            size_t bundles; manifest >> bundles; check(bundles == 17, "wrong Block TileLang specialization count");
            for (size_t i = 0; i < bundles; ++i) {
                std::string key, op, path; manifest >> key >> op >> std::quoted(path);
                check(ops.emplace(key, std::make_shared<TileKernel>(r, path, llaisys::backends::native::KernelIdentity{"tilelang", tl, op, 1})).second, "duplicate Block kernel");
            }
            ops.emplace("hadamard", std::make_shared<llaisys::backends::hadamard::NativeKernel>(r,
                llaisys::backends::native::KernelIdentity{"cuda-hadamard", had, "hadamard", 1}, static_cast<float>(std::pow(a.index_dimension, -0.5))));
            HCBackends hc{ops.at("tensor_cast"), ops.at("dense_linear"), ops.at("row_inv_rms"), ops.at("tensor_row_multiply"),
                ops.at("hc_pre_reduce"), ops.at("hc_split_sinkhorn"), ops.at("hc_post_mix"), nullptr};
            BlockBackends bb{hc, hc, ops.at("tensor_cast"), ops.at("rms_norm")};
            auto model = makeBlock(cfg, weights, ops, bb);
            size_t plans; manifest >> plans; check(plans == 4, "wrong Block plan count");
            for (size_t plan = 0; plan < plans; ++plan) {
                std::string pname, oracle; size_t nsteps; manifest >> pname >> oracle >> nsteps;
                check(nsteps > 0 && nsteps < 64, "invalid step count"); std::vector<Step> steps(nsteps);
                for (auto &step : steps) {
                    size_t nt; manifest >> step.position >> step.tokens >> nt; check(nt == static_cast<size_t>(a.ratio == 4 ? 13 : a.ratio ? 10 : 8), "wrong Block trace count");
                    for (size_t i = 0; i < nt; ++i) check(step.values.insert(read(manifest, r)).second, "duplicate trace");
                }
                auto execute = [&](Block &block, BlockState &state, BlockWorkspace &w, Step &step) {
                    auto &x = *step.values.at("input").tensor, &ids = *step.values.at("input_ids").tensor;
                    auto &y = block.forward(x, ids, state, w);
                    std::map<std::string, Tensor *> actual{{"input", &x}, {"input_ids", &ids}, {"output", &y}, {"cache", &state.attention.cache},
                        {"attn_input", &w.attention_input}, {"attn_output", &w.attention.output}, {"moe_input", &w.feedforward_input}, {"moe_output", &w.feedforward_output}};
                    if (a.ratio) { actual.emplace("kv_state", &state.attention.compression->kv); actual.emplace("score_state", &state.attention.compression->scores); }
                    if (a.ratio == 4) { actual.emplace("index_cache", &state.attention.indexer->cache); actual.emplace("index_kv_state", &state.attention.indexer->compression.kv); actual.emplace("index_score_state", &state.attention.indexer->compression.scores); }
                    for (const auto &[tag, t] : actual) { exact(*t, step.values.at(tag).expected, name + "/" + pname + "/" + tag); ++comparisons; }
                    check(!state.failed() && !w.failed() && state.position() == step.position + step.tokens, "wrong Block committed state");
                };
                BlockState state(r, cfg);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    model->reset(state);
                    for (auto &step : steps) {
                        BlockWorkspace w(r, cfg, step.position, step.tokens); execute(*model, state, w, step); ++steps_run;
                        reject([&] { model->forward(*step.values.at("input").tensor, *step.values.at("input_ids").tensor, state, w); });
                        check(!state.failed(), "position validation poisoned state");
                    }
                }
                if (!plan) {
                    auto &first = steps.front(); auto &x = *first.values.at("input").tensor, &ids = *first.values.at("input_ids").tensor;
                    BlockState uninitialized(r, cfg); BlockWorkspace w(r, cfg, 0, first.tokens);
                    reject([&] { model->forward(x, ids, uninitialized, w); }); model->reset(uninitialized);
                    reject([&] { model->forward(w.feedforward_hc.expanded, ids, uninitialized, w); });
                    auto packed_alias = w.moe.packed_hidden.asStrided(x.shape(), {first.tokens * cfg.copies * a.hidden, cfg.copies * a.hidden, a.hidden, 1});
                    reject([&] { model->forward(*packed_alias, ids, uninitialized, w); });
                    auto token_alias = w.moe.offsets.asStrided({1, first.tokens}, {first.tokens, 1});
                    reject([&] { model->forward(x, *token_alias, uninitialized, w); });
                    auto attended_alias = w.attention.attended.asStrided(x.shape(), {first.tokens * cfg.copies * a.hidden, cfg.copies * a.hidden, a.hidden, 1});
                    reject([&] { model->forward(*attended_alias, ids, uninitialized, w); });
                    Tensor wrong_dtype(r, ids.shape(), {kDLFloat, 32, 1});
                    reject([&] { model->forward(x, wrong_dtype, uninitialized, w); });
                    reject([&] { BlockWorkspace empty(r, cfg, 0, 0); });
                    reject([&] { BlockWorkspace beyond(r, cfg, a.capacity, 1); });
                    reject([&] { auto bad = cfg; bad.moe.hidden += 128; BlockWorkspace incompatible(r, bad, 0, first.tokens); });
                    check(!uninitialized.failed(), "shape/alias validation poisoned healthy Block");
                    auto foreign = makeBlock(cfg, weights, ops, bb); reject([&] { foreign->reset(uninitialized); });
                    std::exception_ptr error; std::thread t([&] { try { model->forward(x, ids, uninitialized, w); } catch (...) { error = std::current_exception(); } });
                    t.join(); check(static_cast<bool>(error), "cross-thread Block accepted"); ++negatives;
                    auto bad = bb; auto injected = std::make_shared<OnceFail>(hc.post_mix); bad.feedforward_hc.post_mix = injected;
                    auto broken = makeBlock(cfg, weights, ops, bad); BlockState partial(r, cfg); broken->reset(partial); BlockWorkspace pw(r, cfg, 0, first.tokens);
                    auto before = hc.post_mix->calls();
                    reject([&] { broken->forward(x, ids, partial, pw); });
                    check(partial.failed() && pw.failed() && partial.position() == 0 && partial.attention.position() == first.tokens
                          && injected->calls() == 1 && hc.post_mix->calls() == before + 1, "Block did not poison partially committed state/no-fallback");
                    reject([&] { broken->forward(x, ids, partial, pw); }); broken->reset(partial);
                    reject([&] { broken->forward(x, ids, partial, pw); });
                    BlockWorkspace recovered(r, cfg, 0, first.tokens); execute(*broken, partial, recovered, first); ++recoveries;
                }
                std::cout << "PASS " << name << '/' << pname << " oracle=" << oracle << " steps=" << nsteps << " repeats=2\n";
            }
            if (layer) counts << ','; counts << "{\"layer\":" << std::quoted(name) << ",\"kernels\":{"; size_t index = 0;
            for (const auto &[name, op] : ops) {
                check(!op->failures(), "unexpected Block baseline backend failure"); if (index++) counts << ',';
                counts << std::quoted(name) << ":{\"backend\":" << std::quoted(op->identity().backend) << ",\"calls\":" << op->calls() << ",\"fallback\":false}";
                op->close();
            }
            counts << "}}";
        }
        counts << ']'; noPython(); check(!TileKernel::destructionErrors(), "TileLang teardown error");
        std::cout << "{\"all_passed\":true,\"layers\":" << layers << ",\"steps\":" << steps_run << ",\"byte_comparisons\":" << comparisons
                  << ",\"negative_checks\":" << negatives << ",\"failure_recoveries\":" << recoveries
                  << ",\"python_loaded\":false,\"aten_cpp\":true,\"contiguous_cache\":true,\"fallback\":false,\"allow_tf32_cublas\":"
                  << (at::globalContext().allowTF32CuBLAS() ? "true" : "false") << ",\"backends\":" << counts.str() << "}\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}

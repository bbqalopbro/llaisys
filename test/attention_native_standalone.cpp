#include <ATen/Context.h>
#include "native_fixture_utils.hpp"
#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/hadamard/native_kernel.hpp"
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/attention.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

using namespace llaisys::models::deepseek_v4;
using namespace native_fixture;
using Kernel = llaisys::backends::native::Kernel;
using StructuralKernel = llaisys::backends::aten::StructuralKernel;
using TileKernel = llaisys::backends::tilelang::NativeKernel;
class FailingSparse final : public Kernel {
public:
    explicit FailingSparse(llaisys::core::Runtime &r) : Kernel(r, {"test-user-library", "v1", "sparse_attn", 1}) {}
    void call(const std::vector<Tensor *> &) override { ++calls_; throw std::runtime_error("injected sparse Attention failure"); }
    void close() override {}
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return calls_; }
private: uint64_t calls_ = 0;
};
struct Step { int64_t position, tokens; std::map<std::string, Value> values; };

static size_t metadataChecks(llaisys::core::Runtime &r, const std::string &version) {
    StructuralKernel prepare(r, {"aten-reference", version, "attention_prepare", 1});
    StructuralKernel multiply(r, {"aten-reference", version, "tensor_row_multiply", 1});
    size_t checks = 0;
    auto reject = [&](auto &&fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
        check(failed, "invalid Attention metadata accepted"); ++checks; };
    auto compareIds = [&](Tensor &t, const std::vector<int32_t> &expected) {
        std::vector<int32_t> actual(expected.size()); t.download(actual.data(), actual.size() * 4);
        check(actual == expected, "Attention candidate order/mask mismatch"); ++checks;
    };
    Tensor x(r, {1, 4, 2}, {kDLBfloat, 16, 1}), cache(r, {1, 6, 2}, {kDLBfloat, 16, 1});
    Tensor extra(r, {1, 4, 1}, {kDLInt, 32, 1}), payload(r, {1, 5, 2}, {kDLBfloat, 16, 1}), ids(r, {1, 4, 5}, {kDLInt, 32, 1});
    std::vector<uint16_t> hidden{0x3f80, 0x3f80, 0x4000, 0x4000, 0x4040, 0x4040, 0x4080, 0x4080}, cached(12, 0);
    cached[8] = cached[9] = 0x4100; cached[10] = cached[11] = 0x4110;
    std::vector<int32_t> selected{-1, -1, -1, 4};
    x.upload(hidden.data(), 16); cache.upload(cached.data(), 24); extra.upload(selected.data(), 16);
    prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{0, 4, 4, 512}, {}});
    compareIds(ids, {0, -1, -1, -1, -1, 0, 1, -1, -1, -1, 0, 1, 2, -1, -1, 0, 1, 2, 3, 4});
    std::vector<uint16_t> actual(10), expected = hidden; expected.insert(expected.end(), {0x4100, 0x4100});
    payload.download(actual.data(), 20); check(actual == expected, "initial Attention payload incorrect"); ++checks;
    std::copy(hidden.begin(), hidden.end(), cached.begin()); actual.resize(12); cache.download(actual.data(), 24);
    check(actual == cached, "initial circular cache/compressed preservation incorrect"); ++checks;
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{-1, 4, 4, 512}, {}}); });
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{0, INT64_MAX, 4, 512}, {}}); });
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{0, 4, 16, 512}, {}}); });
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{0, 4, 4, 0}, {}}); });
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{0, 4, 4, 512}, {1.0}}); });
    auto aliased = cache.asStrided({1, 5, 2}, {12, 2, 1});
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, aliased.get(), &ids}, {{0, 4, 4, 512}, {}}); });
    selected[0] = -2; extra.upload(selected.data(), 16);
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{0, 4, 4, 512}, {}}); });
    selected[0] = 5; extra.upload(selected.data(), 16);
    reject([&] { prepare.callWithScalars({&x, &cache, &extra, &payload, &ids}, {{0, 4, 4, 512}, {}}); });
    Tensor dx(r, {1, 1, 2}, {kDLBfloat, 16, 1}), de(r, {1, 1, 1}, {kDLInt, 32, 1}),
           dp(r, {1, 0, 2}, {kDLBfloat, 16, 1}), di(r, {1, 1, 5}, {kDLInt, 32, 1});
    std::vector<uint16_t> decoded{0x40a0, 0x40a0}; int32_t dc = 4;
    dx.upload(decoded.data(), 4); de.upload(&dc, 4);
    prepare.callWithScalars({&dx, &cache, &de, &dp, &di}, {{4, 4, 4, 512}, {}});
    compareIds(di, {1, 2, 3, 0, 4}); cached[0] = cached[1] = 0x40a0; cache.download(actual.data(), 24);
    check(actual == cached, "decode wraparound corrupted compressed cache"); ++checks;
    Tensor cx(r, {1, 3, 2}, {kDLBfloat, 16, 1}), ce(r, {1, 3, 2}, {kDLInt, 32, 1}),
           cp(r, {1, 8, 2}, {kDLBfloat, 16, 1}), ci(r, {1, 3, 6}, {kDLInt, 32, 1});
    std::vector<uint16_t> chunk{0x40c0, 0x40c0, 0x40e0, 0x40e0, 0x4100, 0x4100};
    std::vector<int32_t> chunk_extra{6, -1, 6, -1, 6, 7}; cx.upload(chunk.data(), 12); ce.upload(chunk_extra.data(), 24);
    prepare.callWithScalars({&cx, &cache, &ce, &cp, &ci}, {{5, 4, 4, 512}, {}});
    compareIds(ci, {0, 1, 2, 3, 6, -1, 1, 2, 3, 4, 6, -1, 2, 3, 4, 5, 6, 7});
    expected = {0x4040, 0x4040, 0x4080, 0x4080, 0x40a0, 0x40a0};
    expected.insert(expected.end(), chunk.begin(), chunk.end()); expected.insert(expected.end(), {0x4100, 0x4100, 0x4110, 0x4110});
    actual.resize(16); cp.download(actual.data(), 32); check(actual == expected, "chunk did not preserve bounded historical window"); ++checks;
    Tensor rows(r, {2, 3}, {kDLFloat, 32, 1}), factors(r, {2, 1}, {kDLFloat, 32, 1});
    std::vector<float> rv{1, 2, 3, 4, 5, 6}, fv{2, 0.5}; rows.upload(rv.data(), 24); factors.upload(fv.data(), 8);
    multiply.call({&rows, &factors}); rows.download(rv.data(), 24);
    check(rv == std::vector<float>{2, 4, 6, 2, 2.5, 3}, "row inverse RMS multiply mismatch"); ++checks;
    auto factor_alias = rows.asStrided({2, 1}, {3, 1});
    reject([&] { multiply.call({&rows, factor_alias.get()}); });
    reject([&] { multiply.callWithScalars({&rows, &factors}, {{1}, {}}); });
    return checks;
}

int main(int argc, char **argv) {
    try {
        check(argc == 2 && std::getenv("SLURM_JOB_ID"), "Attention verification requires Slurm GPU allocation");
        noPython(); llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &runtime = llaisys::core::context().runtime();
        std::ifstream manifest(argv[1]); std::string magic, tl_version, aten_version, hadamard_version; size_t models;
        manifest >> magic >> tl_version >> aten_version >> hadamard_version >> models;
        check(magic == "LLAISYS_NATIVE_ATTENTION_V1" && models == 3 && aten_version == StructuralKernel::compiledVersion(), "invalid manifest/version");
        const auto metadata_checks = metadataChecks(runtime, aten_version);
        size_t compared = 0, negative = 0, steps_run = 0, recoveries = 0;
        auto reject = [&](auto &&fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
            check(failed, "invalid Attention call accepted"); ++negative; };
        std::ostringstream reports; reports << '[';
        for (size_t layer = 0; layer < models; ++layer) {
            std::string name; AttentionConfig cfg; double eps; size_t weight_count;
            manifest >> name >> cfg.hidden >> cfg.heads >> cfg.dimension >> cfg.query_rank >> cfg.output_groups >> cfg.output_rank
                     >> cfg.rope_dimension >> cfg.window >> cfg.ratio >> cfg.capacity >> cfg.index_heads >> cfg.index_dimension >> cfg.index_topk >> eps >> weight_count;
            check(weight_count == static_cast<size_t>(cfg.ratio == 4 ? 24 : cfg.ratio ? 17 : 13), "wrong actual Attention weight count");
            std::map<std::string, Value> loaded;
            for (size_t i = 0; i < weight_count; ++i) check(loaded.insert(read(manifest, runtime)).second, "duplicate weight");
            AttentionWeights weights; for (const auto &[key, v] : loaded) if (key != "frequencies") weights.emplace(key, v.tensor);
            auto frequencies = loaded.at("frequencies").tensor;
            std::map<std::string, std::shared_ptr<Kernel>> ops;
            llaisys::backends::aten::StructuralOptions options; options.epsilon = eps;
            for (const auto &op : {"tensor_cast", "tensor_fill", "dense_linear", "compressor_prepare", "compressor_pool", "rms_norm",
                 "rotary_inplace", "tensor_scale", "indexer_scores", "indexer_mask", "indexer_topk", "indexer_remap",
                 "row_inv_rms", "tensor_row_multiply", "attention_prepare", "grouped_linear"})
                ops.emplace(op, std::make_shared<StructuralKernel>(runtime,
                    llaisys::backends::native::KernelIdentity{"aten-reference", aten_version, op, 1}, options));
            options.inverse_rotary = true;
            ops.emplace("inverse_rotary", std::make_shared<StructuralKernel>(runtime,
                llaisys::backends::native::KernelIdentity{"aten-reference", aten_version, "rotary_inplace", 1}, options));
            size_t bundles; manifest >> bundles; check(bundles == 11, "wrong TileLang specialization count");
            for (size_t i = 0; i < bundles; ++i) {
                std::string key, op, path; manifest >> key >> op >> std::quoted(path);
                check(ops.emplace(key, std::make_shared<TileKernel>(runtime, path,
                    llaisys::backends::native::KernelIdentity{"tilelang", tl_version, op, 1})).second, "duplicate kernel");
            }
            ops.emplace("hadamard", std::make_shared<llaisys::backends::hadamard::NativeKernel>(runtime,
                llaisys::backends::native::KernelIdentity{"cuda-hadamard", hadamard_version, "hadamard", 1}, static_cast<float>(std::pow(cfg.index_dimension, -0.5))));
            CompressorBackends cb{ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("dense_linear"), ops.at("compressor_prepare"),
                ops.at("compressor_pool"), ops.at("rms_norm"), ops.at("rotary_inplace"), ops.at("latent_qdq"), nullptr};
            auto icb = cb; icb.quantize = ops.at("indexer_qdq"); icb.hadamard = ops.at("hadamard");
            IndexerBackends ib{ops.at("quant_query_rank"), ops.at("indexer_gemm"), ops.at("rotary_inplace"), ops.at("hadamard"), ops.at("indexer_qdq"),
                ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("dense_linear"), ops.at("tensor_scale"), ops.at("indexer_scores"),
                ops.at("indexer_mask"), ops.at("indexer_topk"), ops.at("indexer_remap"), icb};
            AttentionBackends b{{ops.at("quant_hidden"), ops.at("gemm_query_a")}, {ops.at("quant_query_rank"), ops.at("gemm_query_b")},
                {ops.at("quant_hidden"), ops.at("gemm_latent")}, {ops.at("quant_output"), ops.at("gemm_output")},
                ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("rms_norm"), ops.at("row_inv_rms"), ops.at("tensor_row_multiply"),
                ops.at("rotary_inplace"), ops.at("inverse_rotary"), ops.at("latent_qdq"), ops.at("attention_prepare"), ops.at("sparse_attn"),
                ops.at("grouped_linear"), cb, ib};
            Attention model(cfg, weights, frequencies, b);
            reject([&] { auto bad = cfg; bad.ratio = 16; AttentionState invalid(runtime, bad); });
            reject([&] { AttentionWorkspace invalid(runtime, cfg, -1, 1); });
            reject([&] { AttentionWorkspace invalid(runtime, cfg, cfg.capacity - 1, 2); });
            reject([&] { AttentionWorkspace invalid(runtime, cfg, 0, 0); });
            reject([&] { auto bad = weights; bad.erase("attn_sink"); Attention invalid(cfg, bad, frequencies, b); });
            reject([&] { auto bad = b; bad.prepare = bad.sparse; Attention invalid(cfg, weights, frequencies, bad); });
            size_t plans; manifest >> plans; check(plans == 6, "expected six full/decode/chunk plans");
            for (size_t plan = 0; plan < plans; ++plan) {
                std::string plan_name, oracle; size_t count; manifest >> plan_name >> oracle >> count;
                check(count > 0 && count < 64, "invalid plan"); std::vector<Step> steps(count);
                for (auto &step : steps) {
                    size_t tensors; manifest >> step.position >> step.tokens >> tensors;
                    check(tensors == static_cast<size_t>(cfg.ratio == 4 ? 11 : cfg.ratio ? 8 : 6), "wrong trace count");
                    for (size_t i = 0; i < tensors; ++i) check(step.values.insert(read(manifest, runtime)).second, "duplicate trace");
                }
                AttentionState state(runtime, cfg);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    model.reset(state);
                    for (auto &step : steps) {
                        AttentionWorkspace w(runtime, cfg, step.position, step.tokens);
                        auto &x = *step.values.at("input").tensor;
                        auto &result = model.forward(x, state, w);
                        check(state.position() == step.position + step.tokens && !state.failed() && !w.failed(), "invalid Attention state");
                        std::map<std::string, Tensor *> actual{{"input", &x}, {"output", &result}, {"cache", &state.cache},
                            {"query_rank", &w.query_rank}, {"latent_raw", &w.latent_raw}, {"grouped", &w.grouped}};
                        if (cfg.ratio) { actual.emplace("kv_state", &state.compression->kv); actual.emplace("score_state", &state.compression->scores); }
                        if (cfg.ratio == 4) {
                            actual.emplace("index_cache", &state.indexer->cache); actual.emplace("index_kv_state", &state.indexer->compression.kv);
                            actual.emplace("index_score_state", &state.indexer->compression.scores);
                        }
                        // Intermediate values first so a final output mismatch is localized.
                        for (const auto &[tag, tensor] : actual) {
                            exact(*tensor, step.values.at(tag).expected, name + "/" + plan_name + "/" + tag); ++compared;
                        }
                        reject([&] { model.forward(x, state, w); });
                        check(!state.failed(), "validation poisoned healthy Attention state"); ++steps_run;
                    }
                }
                std::cout << "PASS " << name << '/' << plan_name << " oracle=" << oracle << " steps=" << count << " repeats=2\n";
                if (plan == 2) {
                    auto &first = steps.front(); auto &x = *first.values.at("input").tensor;
                    AttentionState fresh(runtime, cfg); AttentionWorkspace w(runtime, cfg, 0, first.tokens);
                    reject([&] { model.forward(x, fresh, w); }); model.reset(fresh);
                    reject([&] { model.forward(w.output, fresh, w); }); // Same shape but mutable workspace alias.
                    if (cfg.ratio) {
                        AttentionState incomplete(runtime, cfg); incomplete.compression.reset();
                        reject([&] { model.reset(incomplete); });
                        AttentionWorkspace incomplete_w(runtime, cfg, 0, first.tokens); incomplete_w.compression.reset();
                        reject([&] { model.forward(x, fresh, incomplete_w); });
                    }
                    Attention foreign(cfg, weights, frequencies, b); reject([&] { foreign.forward(x, fresh, w); });
                    std::exception_ptr thread_error;
                    std::thread thread([&] { try { model.forward(x, fresh, w); } catch (...) { thread_error = std::current_exception(); } });
                    thread.join(); check(static_cast<bool>(thread_error), "cross-thread Attention accepted"); ++negative;
                    auto bad = b; auto injected = std::make_shared<FailingSparse>(runtime); bad.sparse = injected;
                    Attention broken(cfg, weights, frequencies, bad); AttentionState partial(runtime, cfg); broken.reset(partial);
                    AttentionWorkspace failed_workspace(runtime, cfg, 0, first.tokens);
                    const auto baseline_calls = b.sparse->calls();
                    reject([&] { broken.forward(x, partial, failed_workspace); });
                    reject([&] { broken.forward(x, partial, failed_workspace); });
                    check(partial.failed() && failed_workspace.failed() && injected->calls() == 1 && broken.failures() == 1
                          && b.sparse->calls() == baseline_calls, "partial Attention failure reused state or silently fell back");
                    model.reset(partial); reject([&] { model.forward(x, partial, failed_workspace); });
                    AttentionWorkspace recovered(runtime, cfg, 0, first.tokens);
                    auto &result = model.forward(x, partial, recovered);
                    exact(result, first.values.at("output").expected, "reset recovery/output");
                    exact(partial.cache, first.values.at("cache").expected, "reset recovery/cache");
                    compared += 2; ++recoveries;
                }
            }
            if (layer) reports << ',';
            reports << "{\"component\":" << std::quoted(name) << ",\"ratio\":" << cfg.ratio << ",\"kernels\":{";
            size_t index = 0;
            for (const auto &[op_name, op] : ops) {
                check(op->failures() == 0, "unexpected baseline failure: " + op_name);
                if (index++) reports << ',';
                reports << std::quoted(op_name) << ":{\"backend\":" << std::quoted(op->identity().backend)
                        << ",\"operation\":" << std::quoted(op->identity().operation) << ",\"version\":" << std::quoted(op->identity().version)
                        << ",\"calls\":" << op->calls() << ",\"fallback\":false}";
                op->close();
            }
            reports << "}}";
        }
        reports << ']'; noPython(); check(!TileKernel::destructionErrors(), "TileLang teardown failed");
        std::cout << "{\"all_passed\":true,\"layers\":" << models << ",\"steps\":" << steps_run
                  << ",\"metadata_contract_checks\":" << metadata_checks
                  << ",\"tensor_byte_comparisons\":" << compared << ",\"negative_checks\":" << negative
                  << ",\"partial_write_recovery_checks\":" << recoveries
                  << ",\"python_interpreter\":false,\"aten_cpp\":true,\"fallback\":false,\"contiguous_cache\":true,\"allow_tf32_cublas\":"
                  << (at::globalContext().allowTF32CuBLAS() ? "true" : "false") << ",\"backend_reports\":" << reports.str() << "}\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}

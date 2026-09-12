#include <ATen/Context.h>
#include "native_fixture_utils.hpp"
#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/hadamard/native_kernel.hpp"
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/indexer.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

using namespace llaisys::models::deepseek_v4;
using namespace native_fixture;
using Kernel = llaisys::backends::native::Kernel;
using CallScalars = llaisys::backends::native::CallScalars;
using StructuralKernel = llaisys::backends::aten::StructuralKernel;
using TileKernel = llaisys::backends::tilelang::NativeKernel;
class FailingBackend final : public Kernel {
public:
    explicit FailingBackend(llaisys::core::Runtime &r) : Kernel(r, {"test-user-library", "v1", "indexer_remap", 1}) {}
    void call(const std::vector<Tensor *> &a) override { callWithScalars(a, {}); }
    void callWithScalars(const std::vector<Tensor *> &, const CallScalars &) override { ++calls_; throw std::runtime_error("injected remap failure"); }
    void close() override {}
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return calls_; }
private: uint64_t calls_ = 0;
};
struct Step { int64_t position, tokens, offset; std::map<std::string, Value> values; };

static size_t metadataChecks(llaisys::core::Runtime &r, const std::string &version) {
    auto make = [&](const char *op) { return std::make_unique<StructuralKernel>(r,
        llaisys::backends::native::KernelIdentity{"aten-reference", version, op, 1}); };
    auto scale = make("tensor_scale"), mask = make("indexer_mask"), remap = make("indexer_remap");
    size_t tests = 0;
    auto reject = [&](auto &&fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
        check(failed, "invalid Indexer metadata contract accepted"); ++tests; };
    Tensor input(r, {2}, {kDLFloat, 32, 1}), output(r, {2}, {kDLFloat, 32, 1});
    std::vector<float> values{1.5f, -2.f}, actual(2); input.upload(values.data(), 8);
    scale->callWithScalars({&input, &output}, {{}, {0.5}}); output.download(actual.data(), 8);
    check(actual == std::vector<float>{0.75f, -1.f}, "incorrect tensor scale"); ++tests;
    input.download(actual.data(), 8); check(actual == values, "scale mutated read-only input"); ++tests;
    reject([&] { scale->call({&input, &output}); });
    reject([&] { scale->callWithScalars({&input, &output}, {{1}, {0.5}}); });
    reject([&] { scale->callWithScalars({&input, &output}, {{}, {INFINITY}}); });
    reject([&] { scale->callWithScalars({&input, &input}, {{}, {0.5}}); });
    Tensor scores(r, {1, 4, 1}, {kDLBfloat, 16, 1});
    std::vector<uint16_t> s{0x3f80, 0x4000, 0x4040, 0x4080}, masked(4); scores.upload(s.data(), 8);
    mask->callWithScalars({&scores}, {{0, 4}, {}}); scores.download(masked.data(), 8);
    check(masked == std::vector<uint16_t>{0xff80, 0xff80, 0xff80, 0x4080}, "future compressed group not masked"); ++tests;
    reject([&] { mask->callWithScalars({&scores}, {{-1, 4}, {}}); });
    reject([&] { mask->callWithScalars({&scores}, {{0, 128}, {}}); });
    reject([&] { mask->callWithScalars({&scores}, {{4, 4}, {}}); }); // Candidate shape mismatch.
    reject([&] { mask->callWithScalars({&scores}, {{0, 4}, {0.0}}); });
    Tensor ids(r, {1, 4, 1}, {kDLInt, 64, 1}), mapped(r, {1, 4, 1}, {kDLInt, 32, 1});
    std::vector<int64_t> selected(4, 0); std::vector<int32_t> mapped_host(4); ids.upload(selected.data(), 32);
    remap->callWithScalars({&ids, &mapped}, {{0, 4, 128}, {}}); mapped.download(mapped_host.data(), 16);
    check(mapped_host == std::vector<int32_t>{-1, -1, -1, 128}, "mask/offset remapping is incorrect"); ++tests;
    reject([&] { remap->callWithScalars({&ids, &mapped}, {{0, 4, -1}, {}}); });
    reject([&] { remap->callWithScalars({&ids, &mapped}, {{0, 4, INT32_MAX}, {}}); });
    selected[0] = 1; ids.upload(selected.data(), 32);
    reject([&] { remap->callWithScalars({&ids, &mapped}, {{0, 4, 128}, {}}); });
    selected[0] = -1; ids.upload(selected.data(), 32);
    reject([&] { remap->callWithScalars({&ids, &mapped}, {{0, 4, 128}, {}}); });
    // The integer causal boundary must not round through FP32 at long positions.
    Tensor long_ids(r, {1, 2, 1}, {kDLInt, 64, 1}), long_output(r, {1, 2, 1}, {kDLInt, 32, 1});
    std::vector<int64_t> long_values{16777215, 16777215}; std::vector<int32_t> long_actual(2);
    long_ids.upload(long_values.data(), 16);
    remap->callWithScalars({&long_ids, &long_output}, {{67108862, 4, 7}, {}}); long_output.download(long_actual.data(), 8);
    check(long_actual == std::vector<int32_t>{-1, 16777222}, "integer position precision was lost"); ++tests;
    return tests;
}

int main(int argc, char **argv) {
    try {
        check(argc == 2 && std::getenv("SLURM_JOB_ID"), "Indexer verification requires Slurm GPU allocation");
        noPython(); llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &runtime = llaisys::core::context().runtime();
        std::ifstream manifest(argv[1]); std::string magic, tl_version, aten_version, hadamard_version; size_t models;
        manifest >> magic >> tl_version >> aten_version >> hadamard_version >> models;
        check(magic == "LLAISYS_NATIVE_INDEXER_V1" && models == 2 && aten_version == StructuralKernel::compiledVersion(), "invalid manifest/version");
        const auto metadata_checks = metadataChecks(runtime, aten_version);
        size_t compared = 0, negative = 0, steps_run = 0, empty_steps = 0, top512_steps = 0, recoveries = 0;
        auto reject = [&](auto &&fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
            check(failed, "invalid Indexer call accepted"); ++negative; };
        std::ostringstream reports; reports << '[';
        for (size_t layer = 0; layer < models; ++layer) {
            std::string name; IndexerConfig cfg; double eps; size_t weight_count;
            manifest >> name >> cfg.hidden >> cfg.query_rank >> cfg.heads >> cfg.dimension >> cfg.rope_dimension >> cfg.topk >> cfg.capacity >> eps >> weight_count;
            check(weight_count == 8, "expected seven weights and frequencies");
            std::map<std::string, Value> loaded;
            for (size_t i = 0; i < weight_count; ++i) check(loaded.insert(read(manifest, runtime)).second, "duplicate weight");
            std::map<std::string, std::shared_ptr<Kernel>> ops;
            llaisys::backends::aten::StructuralOptions options; options.epsilon = eps;
            for (const auto &op : {"tensor_cast", "tensor_fill", "dense_linear", "compressor_prepare", "compressor_pool", "rms_norm",
                 "rotary_inplace", "tensor_scale", "indexer_scores", "indexer_mask", "indexer_topk", "indexer_remap"})
                ops.emplace(op, std::make_shared<StructuralKernel>(runtime,
                    llaisys::backends::native::KernelIdentity{"aten-reference", aten_version, op, 1}, options));
            size_t bundles; manifest >> bundles; check(bundles == 3, "expected three TileLang specializations");
            for (size_t i = 0; i < bundles; ++i) {
                std::string op, path; manifest >> op >> std::quoted(path);
                check(ops.emplace(op, std::make_shared<TileKernel>(runtime, path,
                    llaisys::backends::native::KernelIdentity{"tilelang", tl_version, op, 1})).second, "duplicate kernel");
            }
            ops.emplace("hadamard", std::make_shared<llaisys::backends::hadamard::NativeKernel>(runtime,
                llaisys::backends::native::KernelIdentity{"cuda-hadamard", hadamard_version, "hadamard", 1}, static_cast<float>(std::pow(cfg.dimension, -0.5))));
            CompressorBackends cb{ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("dense_linear"), ops.at("compressor_prepare"),
                ops.at("compressor_pool"), ops.at("rms_norm"), ops.at("rotary_inplace"), ops.at("fp4_act_quant"), ops.at("hadamard")};
            IndexerBackends b{ops.at("act_quant"), ops.at("fp8_gemm"), ops.at("rotary_inplace"), ops.at("hadamard"), ops.at("fp4_act_quant"),
                ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("dense_linear"), ops.at("tensor_scale"), ops.at("indexer_scores"),
                ops.at("indexer_mask"), ops.at("indexer_topk"), ops.at("indexer_remap"), cb};
            IndexerWeights weights{loaded.at("wq_b.weight").tensor, loaded.at("wq_b.scale").tensor, loaded.at("weights_proj.weight").tensor,
                loaded.at("compressor.wkv.weight").tensor, loaded.at("compressor.wgate.weight").tensor, loaded.at("compressor.ape").tensor,
                loaded.at("compressor.norm.weight").tensor, loaded.at("frequencies").tensor};
            Indexer model(cfg, weights, b);
            reject([&] { auto bad = cfg; bad.heads = 0; IndexerState invalid(runtime, bad); });
            reject([&] { IndexerWorkspace invalid(runtime, cfg, -1, 1); });
            reject([&] { IndexerWorkspace invalid(runtime, cfg, cfg.capacity - 1, 2); });
            reject([&] { IndexerWorkspace invalid(runtime, cfg, 0, 0); });
            reject([&] { auto bad = b; bad.mask = bad.topk; Indexer invalid(cfg, weights, bad); });
            size_t plans; manifest >> plans; check(plans == 5, "expected five full/decode/chunk plans");
            for (size_t plan = 0; plan < plans; ++plan) {
                std::string plan_name, oracle; size_t count; manifest >> plan_name >> oracle >> count;
                check(count > 0 && count < 64, "invalid plan"); std::vector<Step> steps(count);
                for (auto &step : steps) {
                    size_t tensors; manifest >> step.position >> step.tokens >> step.offset >> tensors;
                    check(tensors == 8, "expected eight Indexer traces");
                    for (size_t i = 0; i < tensors; ++i) check(step.values.insert(read(manifest, runtime)).second, "duplicate trace");
                }
                IndexerState state(runtime, cfg);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    model.reset(state);
                    for (auto &step : steps) {
                        IndexerWorkspace w(runtime, cfg, step.position, step.tokens);
                        auto &x = *step.values.at("input").tensor, &qr = *step.values.at("query_rank").tensor;
                        auto &result = model.forward(x, qr, step.offset, state, w);
                        check(state.position() == step.position + step.tokens && !state.failed() && !w.failed(), "invalid final Indexer state");
                        for (const auto &pair : {std::make_pair("input", &x), {"query_rank", &qr}, {"output", &result}, {"cache", &state.cache},
                             {"kv_state", &state.compression.kv}, {"score_state", &state.compression.scores},
                             {"projected_weights", &w.projected_weights}, {"quantized_query", &w.quantized}}) {
                            exact(*pair.second, step.values.at(pair.first).expected, name + "/" + plan_name + "/" + pair.first); ++compared;
                        }
                        if (!w.selected) ++empty_steps;
                        if (w.candidates > cfg.topk) ++top512_steps;
                        reject([&] { (void)model.forward(x, qr, step.offset, state, w); });
                        check(!state.failed(), "position validation poisoned healthy state"); ++steps_run;
                    }
                }
                std::cout << "PASS " << name << '/' << plan_name << " oracle=" << oracle << " steps=" << count << " repeats=2\n";
                if (plan == 1) {
                    auto &first = steps.front(); auto &x = *first.values.at("input").tensor, &qr = *first.values.at("query_rank").tensor;
                    IndexerState fresh(runtime, cfg); IndexerWorkspace w(runtime, cfg, 0, first.tokens);
                    reject([&] { model.forward(x, qr, 0, fresh, w); }); model.reset(fresh);
                    reject([&] { model.forward(x, qr, -1, fresh, w); });
                    reject([&] { model.forward(x, qr, INT32_MAX, fresh, w); });
                    Indexer foreign(cfg, weights, b); reject([&] { foreign.forward(x, qr, 0, fresh, w); });
                    std::exception_ptr thread_error;
                    std::thread thread([&] { try { model.forward(x, qr, 0, fresh, w); } catch (...) { thread_error = std::current_exception(); } });
                    thread.join(); check(static_cast<bool>(thread_error), "cross-thread Indexer accepted"); ++negative;
                    auto bad = b; auto injected = std::make_shared<FailingBackend>(runtime); bad.remap = injected;
                    Indexer broken(cfg, weights, bad); IndexerState partial(runtime, cfg); broken.reset(partial);
                    IndexerWorkspace failed_workspace(runtime, cfg, 0, first.tokens);
                    const auto remap_calls = b.remap->calls();
                    reject([&] { broken.forward(x, qr, first.offset, partial, failed_workspace); });
                    reject([&] { broken.forward(x, qr, first.offset, partial, failed_workspace); });
                    check(partial.failed() && failed_workspace.failed() && injected->calls() == 1 && broken.failures() == 1
                          && partial.position() == first.tokens && b.remap->calls() == remap_calls,
                          "failure after cache write reused state or silently fell back");
                    model.reset(partial);
                    reject([&] { model.forward(x, qr, first.offset, partial, failed_workspace); });
                    IndexerWorkspace recovered(runtime, cfg, 0, first.tokens);
                    auto &result = model.forward(x, qr, first.offset, partial, recovered);
                    exact(result, first.values.at("output").expected, "reset recovery/output");
                    exact(partial.cache, first.values.at("cache").expected, "reset recovery/cache");
                    exact(partial.compression.kv, first.values.at("kv_state").expected, "reset recovery/compressor");
                    compared += 3; ++recoveries;
                }
            }
            if (layer) reports << ',';
            reports << "{\"component\":" << std::quoted(name) << ",\"kernels\":{";
            size_t index = 0;
            for (const auto &[op_name, op] : ops) {
                check(op->failures() == 0, "unexpected baseline failure: " + op_name);
                if (index++) reports << ',';
                reports << std::quoted(op_name) << ":{\"backend\":" << std::quoted(op->identity().backend)
                        << ",\"version\":" << std::quoted(op->identity().version) << ",\"calls\":" << op->calls() << ",\"fallback\":false}";
                op->close();
            }
            reports << "}}";
        }
        reports << ']'; noPython(); check(!TileKernel::destructionErrors(), "TileLang teardown failed");
        std::cout << "{\"all_passed\":true,\"layers\":" << models << ",\"steps\":" << steps_run
                  << ",\"metadata_contract_checks\":" << metadata_checks
                  << ",\"tensor_byte_comparisons\":" << compared << ",\"negative_checks\":" << negative
                  << ",\"empty_candidate_steps\":" << empty_steps << ",\"top512_steps\":" << top512_steps
                  << ",\"partial_write_recovery_checks\":" << recoveries
                  << ",\"python_interpreter\":false,\"aten_cpp\":true,\"fallback\":false,\"contiguous_cache\":true,\"allow_tf32_cublas\":"
                  << (at::globalContext().allowTF32CuBLAS() ? "true" : "false") << ",\"backend_reports\":" << reports.str() << "}\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}

#include <ATen/Context.h>
#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/hadamard/native_kernel.hpp"
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/compressor.hpp"
#include <cstdlib>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

using namespace llaisys::models::deepseek_v4;
using Tensor = llaisys::backends::native::Tensor;
using Kernel = llaisys::backends::native::Kernel;
using CallScalars = llaisys::backends::native::CallScalars;
using StructuralKernel = llaisys::backends::aten::StructuralKernel;
using StructuralOptions = llaisys::backends::aten::StructuralOptions;
using TileKernel = llaisys::backends::tilelang::NativeKernel;
static void check(bool c, const std::string &message) { if (!c) throw std::runtime_error(message); }
static void noPython() {
    check(!dlsym(RTLD_DEFAULT, "Py_IsInitialized"), "Python interpreter unexpectedly loaded");
    std::ifstream maps("/proc/self/maps"); check(maps.good(), "cannot inspect process mappings");
    std::string line;
    while (std::getline(maps, line)) check(line.find("libpython") == std::string::npos
        && line.find("libtorch_python") == std::string::npos, "Python runtime mapping unexpectedly loaded");
}
static std::vector<unsigned char> bytes(const std::string &path, size_t count, uint64_t offset = 0) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    check(in.good() && offset <= static_cast<uint64_t>(in.tellg()) && count <= static_cast<uint64_t>(in.tellg()) - offset,
          "invalid tensor file/extent: " + path);
    in.seekg(offset); std::vector<unsigned char> value(count);
    if (count) in.read(reinterpret_cast<char *>(value.data()), count);
    check(in.good(), "truncated tensor file"); return value;
}
struct Value { std::shared_ptr<Tensor> tensor; std::string expected; };
static std::pair<std::string, Value> read(std::istream &in, llaisys::core::Runtime &runtime) {
    std::string name, path, expected; unsigned code, bits, lanes; size_t rank, count; uint64_t offset;
    in >> name >> code >> bits >> lanes >> rank;
    check(in.good() && rank > 0 && rank <= 16 && code < 256 && bits < 256 && lanes < 65536, "invalid tensor descriptor");
    std::vector<int64_t> shape(rank); for (auto &dim : shape) in >> dim;
    in >> count >> offset >> std::quoted(path) >> std::quoted(expected);
    check(in.good() && count < (1ull << 31), "invalid tensor record");
    auto value = std::make_shared<Tensor>(runtime, shape, DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
    check(value->bytes() == count, "tensor shape/dtype byte mismatch");
    auto host = bytes(path, count, offset); value->upload(host.data(), host.size());
    return {name, {std::move(value), expected}};
}
static void exact(Tensor &value, const std::string &path, const std::string &label) {
    auto wanted = bytes(path, value.bytes()); std::vector<unsigned char> actual(value.bytes());
    value.download(actual.data(), actual.size());
    if (wanted != actual) {
        size_t first = 0, changed = 0;
        for (size_t i = 0; i < actual.size(); ++i) if (wanted[i] != actual[i]) { if (!changed) first = i; ++changed; }
        throw std::runtime_error("non-exact " + label + ": changed_bytes=" + std::to_string(changed) + " first=" + std::to_string(first));
    }
}
class FailingBackend final : public Kernel {
public:
    explicit FailingBackend(llaisys::core::Runtime &r, std::string operation)
        : Kernel(r, {"test-user-library", "v1", std::move(operation), 1}) {}
    void call(const std::vector<Tensor *> &args) override { callWithScalars(args, {}); }
    void callWithScalars(const std::vector<Tensor *> &, const CallScalars &) override {
        ++calls_; throw std::runtime_error("injected compressor backend failure");
    }
    void close() override {}
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return calls_; }
private: uint64_t calls_ = 0;
};
struct Step {
    int64_t position, tokens, emitted;
    std::map<std::string, Value> values;
};

int main(int argc, char **argv) {
    try {
        check(argc == 2 && std::getenv("SLURM_JOB_ID"), "compressor test requires a Slurm GPU allocation");
        noPython(); llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &runtime = llaisys::core::context().runtime();
        std::ifstream manifest(argv[1]);
        std::string magic, tl_version, aten_version, hadamard_version; size_t models;
        manifest >> magic >> tl_version >> aten_version >> hadamard_version >> models;
        check(magic == "LLAISYS_NATIVE_COMPRESSOR_V1" && models == 3 && aten_version == StructuralKernel::compiledVersion(), "invalid manifest/version");
        size_t compared = 0, negative = 0, steps_run = 0, plans_run = 0, no_output_steps = 0, recoveries = 0;
        std::ostringstream backend_reports; backend_reports << '[';
        for (size_t model_index = 0; model_index < models; ++model_index) {
            std::string name, quant_path; CompressorConfig cfg; int64_t capacity; double eps; size_t weights;
            manifest >> name >> cfg.hidden >> cfg.dimension >> cfg.rope_dimension >> cfg.ratio >> cfg.rotate >> capacity >> eps >> weights;
            check(weights == 5, "expected four weights plus precomputed frequencies");
            std::map<std::string, Value> loaded;
            for (size_t i = 0; i < weights; ++i) check(loaded.insert(read(manifest, runtime)).second, "duplicate compressor weight");
            manifest >> std::quoted(quant_path);
            std::map<std::string, std::shared_ptr<Kernel>> ops;
            StructuralOptions options; options.epsilon = eps;
            for (const auto &op : {"tensor_cast", "tensor_fill", "dense_linear", "compressor_prepare", "compressor_pool", "rms_norm", "rotary_inplace"})
                ops.emplace(op, std::make_shared<StructuralKernel>(runtime,
                    llaisys::backends::native::KernelIdentity{"aten-reference", aten_version, op, 1}, options));
            ops.emplace("quantize", std::make_shared<TileKernel>(runtime, quant_path,
                llaisys::backends::native::KernelIdentity{"tilelang", tl_version, cfg.rotate ? "fp4_act_quant" : "act_quant", 1}));
            if (cfg.rotate) ops.emplace("hadamard", std::make_shared<llaisys::backends::hadamard::NativeKernel>(runtime,
                llaisys::backends::native::KernelIdentity{"cuda-hadamard", hadamard_version, "hadamard", 1}, static_cast<float>(1.0 / std::sqrt(cfg.dimension))));
            CompressorBackends b{ops.at("tensor_cast"), ops.at("tensor_fill"), ops.at("dense_linear"), ops.at("compressor_prepare"),
                ops.at("compressor_pool"), ops.at("rms_norm"), ops.at("rotary_inplace"), ops.at("quantize"), cfg.rotate ? ops.at("hadamard") : nullptr};
            auto makeModel = [&](CompressorBackends backends) {
                return std::make_unique<Compressor>(cfg, loaded.at("wkv.weight").tensor, loaded.at("wgate.weight").tensor,
                    loaded.at("ape").tensor, loaded.at("norm.weight").tensor, loaded.at("frequencies").tensor, backends);
            };
            auto model = makeModel(b);
            auto reject = [&](auto &&function) {
                bool failed = false; try { function(); } catch (const std::exception &) { failed = true; }
                check(failed, "invalid compressor call was accepted"); ++negative;
            };
            reject([&] { auto bad = cfg; bad.ratio = 16; CompressorState invalid(runtime, bad); });
            reject([&] { CompressorWorkspace invalid(runtime, cfg, -1, 1); });
            reject([&] { CompressorWorkspace invalid(runtime, cfg, 0, 0); });
            reject([&] { auto bad = b; bad.pool = bad.dense; (void)makeModel(bad); });
            reject([&] { b.quantize->callWithScalars({}, {{1}, {}}); }); // Old kernel contract rejects metadata.
            size_t plans; manifest >> plans; check(plans == 4, "expected published/decode and chunked plans");
            for (size_t plan_index = 0; plan_index < plans; ++plan_index) {
                std::string plan_name, oracle; size_t count;
                manifest >> plan_name >> oracle >> count; check(count > 0 && count < 64, "invalid compressor plan");
                std::vector<Step> steps(count);
                for (auto &step : steps) {
                    size_t tensors; manifest >> step.position >> step.tokens >> step.emitted >> tensors;
                    check(tensors == static_cast<size_t>(step.emitted ? 8 : 6), "invalid compressor trace count");
                    for (size_t i = 0; i < tensors; ++i) check(step.values.insert(read(manifest, runtime)).second, "duplicate fixture");
                }
                CompressorState state(runtime, cfg);
                Tensor cache(runtime, {1, capacity / cfg.ratio, cfg.dimension}, {kDLBfloat, 16, 1});
                for (int repeat = 0; repeat < 2; ++repeat) {
                    model->reset(state); b.fill->callWithScalars({&cache}, {{}, {0.0}});
                    for (auto &step : steps) {
                        auto &input = *step.values.at("input").tensor;
                        CompressorWorkspace w(runtime, cfg, step.position, step.tokens);
                        auto result = model->forward(input, state, w);
                        check(result.first_group == step.position / cfg.ratio && result.count == step.emitted
                              && state.position() == step.position + step.tokens && !state.failed() && !w.failed(), "compressor state/emit count mismatch");
                        if (result.count) {
                            auto destination = cache.asStrided({1, result.count, cfg.dimension}, {capacity / cfg.ratio * cfg.dimension, cfg.dimension, 1},
                                                               result.first_group * cfg.dimension);
                            b.cast->call({&result.values, destination.get()});
                            exact(result.values, step.values.at("output").expected, plan_name + "/output"); ++compared;
                            exact(w.pooled_bf16, step.values.at("pooled_bf16").expected, plan_name + "/norm_input"); ++compared;
                        } else ++no_output_steps;
                        for (const auto &pair : {std::make_pair("input", &input), std::make_pair("projected_kv", &w.projected_kv),
                            std::make_pair("projected_scores", &w.projected_scores), std::make_pair("kv_state", &state.kv),
                            std::make_pair("score_state", &state.scores), std::make_pair("cache", &cache)}) {
                            exact(*pair.second, step.values.at(pair.first).expected, plan_name + "/" + pair.first); ++compared;
                        }
                        reject([&] { (void)model->forward(input, state, w); }); // Replayed position cannot overwrite state/cache.
                        check(!state.failed(), "validation-only failure poisoned valid state");
                        ++steps_run;
                    }
                }
                std::cout << "PASS " << name << '/' << plan_name << " oracle=" << oracle << " steps=" << count << " repeats=2\n";
                ++plans_run;
                if (!plan_index) {
                    auto &step = steps.front(); auto &input = *step.values.at("input").tensor;
                    CompressorState uninitialized(runtime, cfg); CompressorWorkspace w(runtime, cfg, 0, step.tokens);
                    reject([&] { (void)model->forward(input, uninitialized, w); });
                    model->reset(uninitialized);
                    auto foreign = makeModel(b);
                    reject([&] { (void)foreign->forward(input, uninitialized, w); });
                    std::exception_ptr thread_error;
                    std::thread thread([&] { try { model->forward(input, uninitialized, w); } catch (...) { thread_error = std::current_exception(); } });
                    thread.join(); check(static_cast<bool>(thread_error), "foreign-thread compressor execution accepted"); ++negative;
                    auto bad = b; auto injected = std::make_shared<FailingBackend>(runtime, "compressor_prepare"); bad.prepare = injected;
                    auto broken = makeModel(bad); CompressorState poisoned(runtime, cfg); broken->reset(poisoned);
                    CompressorWorkspace poisoned_workspace(runtime, cfg, 0, step.tokens);
                    const auto original_calls = b.prepare->calls();
                    reject([&] { broken->forward(input, poisoned, poisoned_workspace); });
                    reject([&] { broken->forward(input, poisoned, poisoned_workspace); });
                    check(poisoned.failed() && poisoned_workspace.failed() && injected->calls() == 1 && broken->failures() == 1
                          && b.prepare->calls() == original_calls, "failed compressor reused workspace or silently fell back");
                    // Fail AFTER state packing/norm/RoPE have already written
                    // request buffers; reset must discard all partial state.
                    auto bad_quant = b;
                    auto quant_failure = std::make_shared<FailingBackend>(runtime, b.quantize->identity().operation);
                    bad_quant.quantize = quant_failure;
                    auto partial = makeModel(bad_quant); CompressorState partial_state(runtime, cfg); partial->reset(partial_state);
                    CompressorWorkspace partial_workspace(runtime, cfg, 0, step.tokens);
                    const auto quant_calls = b.quantize->calls();
                    reject([&] { partial->forward(input, partial_state, partial_workspace); });
                    check(partial_state.failed() && partial_workspace.failed() && quant_failure->calls() == 1
                          && b.quantize->calls() == quant_calls, "partial compressor failure silently switched quantization");
                    model->reset(partial_state);
                    reject([&] { model->forward(input, partial_state, partial_workspace); });
                    CompressorWorkspace recovered_workspace(runtime, cfg, 0, step.tokens);
                    auto recovered = model->forward(input, partial_state, recovered_workspace);
                    exact(recovered.values, step.values.at("output").expected, "reset after partial state write/output");
                    exact(partial_state.kv, step.values.at("kv_state").expected, "reset after partial state write/kv");
                    exact(partial_state.scores, step.values.at("score_state").expected, "reset after partial state write/scores");
                    compared += 3; ++recoveries;
                }
            }
            if (model_index) backend_reports << ',';
            backend_reports << "{\"component\":" << std::quoted(name) << ",\"kernels\":{";
            size_t op_index = 0;
            for (const auto &[op_name, op] : ops) {
                check(!op->failures(), "unexpected baseline backend failure: " + op_name);
                if (op_index++) backend_reports << ',';
                backend_reports << std::quoted(op_name) << ":{\"backend\":" << std::quoted(op->identity().backend)
                    << ",\"operation\":" << std::quoted(op->identity().operation) << ",\"version\":" << std::quoted(op->identity().version)
                    << ",\"calls\":" << op->calls() << ",\"failures\":" << op->failures() << ",\"fallback\":false}";
                op->close();
            }
            backend_reports << "}}";
        }
        backend_reports << ']'; noPython(); check(!TileKernel::destructionErrors(), "TileLang destruction error");
        std::cout << "{\"all_passed\":true,\"components\":" << models << ",\"plans\":" << plans_run << ",\"steps\":" << steps_run
                  << ",\"no_output_steps\":" << no_output_steps << ",\"tensor_byte_comparisons\":" << compared << ",\"negative_checks\":" << negative
                  << ",\"partial_write_recovery_checks\":" << recoveries
                  << ",\"python_interpreter\":false,\"aten_cpp\":true,\"fallback\":false,\"allow_tf32_cublas\":"
                  << (at::globalContext().allowTF32CuBLAS() ? "true" : "false") << ",\"backend_counts_include_setup_and_injection\":true,\"backend_reports\":" << backend_reports.str() << "}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}

// Joint native TileLang + ATen process; no Python interpreter/model callbacks.
#include <ATen/Context.h>
#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/moe.hpp"
#include <cstdlib>
#include <cstring>
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
using StructuralKernel = llaisys::backends::aten::StructuralKernel;
using StructuralOptions = llaisys::backends::aten::StructuralOptions;
using TileKernel = llaisys::backends::tilelang::NativeKernel;
static void check(bool c, const std::string &message) { if (!c) throw std::runtime_error(message); }
static void noPython() {
    check(!dlsym(RTLD_DEFAULT, "Py_IsInitialized"), "unexpected Python interpreter");
    std::ifstream maps("/proc/self/maps"); check(maps.good(), "cannot inspect process mappings");
    std::string line;
    while (std::getline(maps, line)) check(line.find("libpython") == std::string::npos
        && line.find("libtorch_python") == std::string::npos, "unexpected Python mapping");
}
static std::vector<unsigned char> bytes(const std::string &path, size_t count, uint64_t offset = 0) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(file.good() && offset <= static_cast<uint64_t>(file.tellg())
          && count <= static_cast<uint64_t>(file.tellg()) - offset, "invalid binary file/extent: " + path);
    file.seekg(offset);
    std::vector<unsigned char> value(count);
    if (count) file.read(reinterpret_cast<char *>(value.data()), count);
    check(file.good(), "binary read failed"); return value;
}
struct Value { std::shared_ptr<Tensor> tensor; std::string expected; };
static std::pair<std::string, Value> read(std::istream &in, llaisys::core::Runtime &runtime) {
    std::string name, path, expected; unsigned code, bits, lanes; size_t rank, count; uint64_t offset;
    in >> name >> code >> bits >> lanes >> rank;
    check(in.good() && rank > 0 && rank <= 16 && code < 256 && bits < 256 && lanes < 65536, "invalid tensor record");
    std::vector<int64_t> shape(rank); for (auto &dim : shape) in >> dim;
    in >> count >> offset >> std::quoted(path) >> std::quoted(expected);
    check(in.good() && count < (1ull << 31), "invalid/oversized test tensor");
    auto tensor = std::make_shared<Tensor>(runtime, shape, DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
    check(tensor->bytes() == count, "tensor byte layout mismatch");
    auto host = bytes(path, count, offset); tensor->upload(host.data(), host.size());
    return {name, Value{std::move(tensor), expected}};
}
static void exact(Tensor &tensor, const std::string &path, const std::string &label) {
    auto expected = bytes(path, tensor.bytes()); std::vector<unsigned char> actual(tensor.bytes());
    tensor.download(actual.data(), actual.size());
    if (actual != expected) {
        size_t first = 0, changed = 0;
        for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) { if (!changed) first = i; ++changed; }
        throw std::runtime_error("non-exact " + label + ": changed_bytes=" + std::to_string(changed) + " first=" + std::to_string(first));
    }
}
class FailingDispatch final : public Kernel {
public:
    explicit FailingDispatch(llaisys::core::Runtime &r) : Kernel(r, {"test-user-library", "v1", "moe_dispatch", 1}) {}
    void call(const std::vector<Tensor *> &) override { ++calls_; throw std::runtime_error("injected dispatch failure"); }
    void close() override {}
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return calls_; }
private: uint64_t calls_ = 0;
};

int main(int argc, char **argv) {
    try {
        check(argc == 2 && std::getenv("SLURM_JOB_ID"), "native MoE test requires a Slurm allocation");
        noPython();
        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &runtime = llaisys::core::context().runtime();
        std::ifstream manifest(argv[1]);
        std::string magic, tl_version, aten_version; size_t layers;
        manifest >> magic >> tl_version >> aten_version >> layers;
        check(magic == "LLAISYS_NATIVE_MOE_V1" && layers == 2 && aten_version == StructuralKernel::compiledVersion(), "invalid manifest/version");
        size_t cases = 0, compared = 0, negative = 0, empty = 0; uint64_t total_experts = 0;
        std::ostringstream backend_reports;
        backend_reports << '[';
        for (size_t l = 0; l < layers; ++l) {
            int layer; MoeConfig cfg; double route_scale, clamp; std::string score; size_t weight_count;
            manifest >> layer >> cfg.hidden >> cfg.intermediate >> cfg.experts >> cfg.top_k >> cfg.vocabulary >> cfg.hash_routing
                     >> route_scale >> clamp >> score >> weight_count;
            check(manifest.good() && weight_count < 10000, "invalid MoE layer record");
            std::map<std::string, Value> weights;
            for (size_t i = 0; i < weight_count; ++i) check(weights.insert(read(manifest, runtime)).second, "duplicate weight key");
            auto take = [&](const std::string &name) {
                auto it = weights.find(name); check(it != weights.end(), "missing weight " + name);
                auto value = it->second.tensor; weights.erase(it); return value;
            };
            std::map<std::string, std::shared_ptr<Kernel>> ops;
            size_t kernels; manifest >> kernels; check(kernels == 6, "expected six quantized Linear specializations");
            for (size_t i = 0; i < kernels; ++i) {
                std::string name, op, path; manifest >> name >> op >> std::quoted(path);
                check(ops.emplace(name, std::make_shared<TileKernel>(runtime, path,
                    llaisys::backends::native::KernelIdentity{"tilelang", tl_version, op, 1})).second, "duplicate kernel");
            }
            StructuralOptions opt; opt.hash_routing = cfg.hash_routing; opt.route_scale = route_scale;
            opt.activation_limit = clamp; opt.score_function = score;
            for (const auto &op : {"tensor_cast", "dense_linear", "row_gather", "router", "moe_dispatch", "moe_combine", "moe_finalize", "expert_activation"})
                ops.emplace(op, std::make_shared<StructuralKernel>(runtime,
                    llaisys::backends::native::KernelIdentity{"aten-reference", aten_version, op, 1}, opt));
            auto makeExpert = [&](const std::string &prefix, bool shared) {
                auto projection = [&](const std::string &name, bool down) {
                    auto w = take(prefix + "." + name + ".weight"), s = take(prefix + "." + name + ".scale");
                    return std::make_shared<QuantizedLinear>(w, s, ops.at(down ? "quant_intermediate" : "quant_hidden"),
                        ops.at(std::string(shared ? "fp8_" : "fp4_") + (down ? "down" : "gate")));
                };
                auto gate = projection("w1", false), up = projection("w3", false), down = projection("w2", true);
                return std::make_shared<Expert>(gate, up, down, ops.at("expert_activation"));
            };
            auto gate = take("gate.weight"), selector = take(cfg.hash_routing ? "gate.tid2eid" : "gate.bias");
            std::vector<std::shared_ptr<Expert>> experts;
            for (int64_t e = 0; e < cfg.experts; ++e) experts.push_back(makeExpert("experts." + std::to_string(e), false));
            auto shared = makeExpert("shared_experts", true);
            check(weights.empty(), "unconsumed model weights");
            MoeBackends backends{ops.at("tensor_cast"), ops.at("dense_linear"), ops.at("row_gather"), ops.at("router"),
                                 ops.at("moe_dispatch"), ops.at("moe_combine"), ops.at("moe_finalize")};
            MoE model(cfg, gate, selector, experts, shared, backends);
            auto reject = [&](auto &&fn) {
                bool failed = false;
                try { fn(); } catch (const std::invalid_argument &) { failed = true; }
                check(failed, "invalid native MoE configuration/contract accepted"); ++negative;
            };
            const auto setup_calls = backends.cast->calls();
            reject([&] { auto bad = cfg; bad.top_k = cfg.experts + 1; MoeWorkspace invalid(runtime, bad, 1); });
            reject([&] { auto missing = experts; missing.pop_back(); MoE invalid(cfg, gate, selector, missing, shared, backends); });
            reject([&] { auto bad = backends; bad.dispatch = bad.router; MoE invalid(cfg, gate, selector, experts, shared, bad); });
            reject([&] { MoE invalid(cfg, gate, nullptr, experts, shared, backends); });
            check(setup_calls == backends.cast->calls(), "invalid model enqueued backend preparation");
            size_t count; manifest >> count; check(count == 3, "expected decode/small-prefill/longer-prefill cases");
            for (size_t c = 0; c < count; ++c) {
                std::string name; int64_t rows; size_t tensors;
                manifest >> name >> rows >> tensors; check(rows > 0 && tensors == 14, "invalid MoE test case");
                std::map<std::string, Value> values;
                for (size_t i = 0; i < tensors; ++i) check(values.insert(read(manifest, runtime)).second, "duplicate fixture");
                auto &input = *values.at("input").tensor, &ids = *values.at("tokens").tensor, &out = *values.at("output").tensor;
                MoeWorkspace w(runtime, cfg, rows);
                std::map<std::string, Tensor *> trace{{"logits", &w.logits}, {"weights", &w.route_weights}, {"ids", &w.route_ids},
                    {"packed_hidden", &w.packed_hidden}, {"packed_weights", &w.packed_weights}, {"packed_rows", &w.packed_rows},
                    {"offsets", &w.offsets}, {"packed_output", &w.packed_output}, {"routed", &w.routed_output}, {"shared", &w.shared_output},
                    {"output", &out}, {"input", &input}, {"tokens", &ids}};
                // The extra 'hash_or_bias' fixture validates selected table rows
                // for hashed layers and the actual learned bias otherwise.
                trace["hash_or_bias"] = cfg.hash_routing ? &w.hash_ids : selector.get();
                size_t shapes = 0;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    model.forward(input, ids, out, w);
                    runtime.synchronize();
                    for (const auto &[tag, value] : trace) { exact(*value, values.at(tag).expected, name + "/" + tag); ++compared; }
                    if (repeat) check(shapes == w.scratchShapes(), "scratch unexpectedly reallocated on repeated shape");
                    shapes = w.scratchShapes();
                }
                total_experts += w.expertExecutions(); ++cases;
                check(!w.failed() && w.offsetDownloads() == 2, "incorrect reference metadata accounting");
                bool rejected = false;
                try { model.forward(input, ids, input, w); } catch (const std::invalid_argument &) { rejected = true; }
                check(rejected && !w.failed(), "aliased output not rejected before work"); ++negative;
                std::exception_ptr foreign;
                std::thread thread([&] { try { model.forward(input, ids, out, w); } catch (...) { foreign = std::current_exception(); } });
                thread.join(); check(static_cast<bool>(foreign), "foreign-thread execution accepted"); ++negative;
                if (!c) {
                    auto failing = std::make_shared<FailingDispatch>(runtime);
                    auto injected_backends = backends; injected_backends.dispatch = failing;
                    MoE injected(cfg, gate, selector, experts, shared, injected_backends);
                    MoeWorkspace failed_workspace(runtime, cfg, rows);
                    const auto original_dispatches = backends.dispatch->calls();
                    for (int attempt = 0; attempt < 2; ++attempt) {
                        rejected = false;
                        try { injected.forward(input, ids, out, failed_workspace); } catch (const std::exception &) { rejected = true; }
                        check(rejected, "failed backend/poisoned workspace was accepted"); ++negative;
                    }
                    check(failed_workspace.failed() && failing->calls() == 1 && injected.failures() == 1
                          && original_dispatches == backends.dispatch->calls(), "silent fallback or failed-workspace reuse");
                    exact(out, values.at("output").expected, "injected failure output guard"); ++compared;
                }
                std::cout << "PASS layer=" << layer << " rows=" << rows << " exact=14 repeats=2 scratch_shapes=" << shapes
                          << " expert_executions=" << w.expertExecutions() << '\n';
            }
            Tensor zero(runtime, {0, cfg.hidden}, {kDLBfloat, 16, 1}), zero_out(runtime, {0, cfg.hidden}, {kDLBfloat, 16, 1});
            Tensor zero_ids(runtime, {0}, {kDLInt, 64, 1}); MoeWorkspace zero_workspace(runtime, cfg, 0);
            const auto before = backends.cast->calls();
            model.forward(zero, zero_ids, zero_out, zero_workspace);
            check(before == backends.cast->calls() && !zero_workspace.scratchShapes(), "empty MoE launched kernels"); ++empty;
            if (l) backend_reports << ',';
            backend_reports << "{\"layer\":" << layer << ",\"kernels\":{";
            size_t op_index = 0;
            for (const auto &[name, op] : ops) {
                check(!op->failures(), "baseline backend failure: " + name);
                if (op_index++) backend_reports << ',';
                backend_reports << std::quoted(name) << ":{\"backend\":" << std::quoted(op->identity().backend)
                    << ",\"version\":" << std::quoted(op->identity().version) << ",\"operation\":" << std::quoted(op->identity().operation)
                    << ",\"contract_revision\":" << op->identity().contract_revision << ",\"calls\":" << op->calls()
                    << ",\"failures\":" << op->failures() << ",\"fallback\":false}";
                op->close();
            }
            backend_reports << "}}";
            runtime.synchronize();
        }
        noPython(); check(!TileKernel::destructionErrors(), "TileLang destruction error");
        backend_reports << ']';
        std::cout << "{\"all_passed\":true,\"complete_moe_layers\":" << layers << ",\"cases\":" << cases
                  << ",\"tensor_byte_comparisons\":" << compared << ",\"negative_checks\":" << negative
                  << ",\"empty_moe_checks\":" << empty << ",\"expert_executions\":" << total_experts
                  << ",\"python_interpreter\":false,\"aten_cpp\":true,\"fallback\":false,\"host_offsets\":true,\"allow_tf32_cublas\":"
                  << (at::globalContext().allowTF32CuBLAS() ? "true" : "false")
                  << ",\"backend_counts_include_setup_and_failure_injection\":true,\"backend_reports\":" << backend_reports.str() << "}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}

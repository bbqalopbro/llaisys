#include <ATen/Context.h>
#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/hyperconnection.hpp"
#include "native_fixture_utils.hpp"
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

static size_t contractChecks(llaisys::core::Runtime &r, const std::string &version) {
    llaisys::backends::aten::StructuralOptions options; options.epsilon = 0.125;
    auto make = [&](const char *op) { return std::make_unique<StructuralKernel>(r,
        llaisys::backends::native::KernelIdentity{"aten-reference", version, op, 1}, options); };
    auto pre = make("hc_pre_reduce"), post = make("hc_post_mix"), head = make("hc_head_weights");
    size_t checked = 0;
    auto reject = [&](auto &&fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
        check(failed, "invalid HC structural contract accepted"); ++checked; };
    auto f32 = [&](std::vector<int64_t> shape, std::vector<float> data) {
        auto t = std::make_shared<Tensor>(r, shape, DLDataType{kDLFloat, 32, 1});
        check(t->bytes() == data.size() * 4, "invalid synthetic shape"); t->upload(data.data(), t->bytes()); return t;
    };
    auto bf16 = [&](std::vector<int64_t> shape, std::vector<uint16_t> data) {
        auto t = std::make_shared<Tensor>(r, shape, DLDataType{kDLBfloat, 16, 1});
        check(t->bytes() == data.size() * 2, "invalid synthetic shape"); t->upload(data.data(), t->bytes()); return t;
    };
    auto x = f32({1, 1, 2, 2}, {1, 2, 3, 4}), pw = f32({1, 1, 2}, {0.25, 0.75});
    auto reduced = bf16({1, 1, 2}, {0, 0}); pre->call({x.get(), pw.get(), reduced.get()});
    std::vector<uint16_t> actual(2); reduced->download(actual.data(), 4);
    check(actual == std::vector<uint16_t>{0x4020, 0x4060}, "HC weighted pre sum mismatch"); ++checked;
    auto residual = bf16({1, 1, 2, 2}, {0x3f80, 0x4000, 0x4040, 0x4080});
    auto branch = bf16({1, 1, 2}, {0x4120, 0x41a0});
    auto postw = f32({1, 1, 2}, {2, 3}), comb = f32({1, 1, 2, 2}, {1, 2, 3, 4});
    auto expanded = bf16({1, 1, 2, 2}, {0, 0, 0, 0});
    post->call({branch.get(), residual.get(), postw.get(), comb.get(), expanded.get()});
    actual.resize(4); expanded->download(actual.data(), 8);
    check(actual == std::vector<uint16_t>{0x41f0, 0x4258, 0x4230, 0x42a0}, "HC combination source/destination axes transposed"); ++checked;
    auto mixes = f32({1, 1, 2}, {0, 0}), scale = f32({1}, {1}), base = f32({2}, {0, 0}), out = f32({1, 1, 2}, {0, 0});
    head->call({mixes.get(), scale.get(), base.get(), out.get()}); std::vector<float> floats(2); out->download(floats.data(), 8);
    check(floats == std::vector<float>{0.625, 0.625}, "HC head sigmoid/epsilon mismatch"); ++checked;
    reject([&] { pre->call({x.get(), pw.get()}); });
    reject([&] { pre->call({residual.get(), pw.get(), reduced.get()}); });
    reject([&] { pre->call({x.get(), base.get(), reduced.get()}); });
    reject([&] { pre->callWithScalars({x.get(), pw.get(), reduced.get()}, {{1}, {}}); });
    reject([&] { post->call({branch.get(), residual.get(), postw.get(), comb.get()}); });
    reject([&] { post->call({branch.get(), residual.get(), postw.get(), comb.get(), residual.get()}); });
    reject([&] { post->call({branch.get(), x.get(), postw.get(), comb.get(), expanded.get()}); });
    reject([&] { post->call({branch.get(), residual.get(), base.get(), comb.get(), expanded.get()}); });
    reject([&] { post->callWithScalars({branch.get(), residual.get(), postw.get(), comb.get(), expanded.get()}, {{}, {1}}); });
    reject([&] { head->call({mixes.get(), scale.get(), base.get()}); });
    reject([&] { head->call({mixes.get(), scale.get(), base.get(), mixes.get()}); });
    reject([&] { head->call({mixes.get(), base.get(), base.get(), out.get()}); });
    reject([&] { head->call({reduced.get(), scale.get(), base.get(), out.get()}); });
    reject([&] { head->callWithScalars({mixes.get(), scale.get(), base.get(), out.get()}, {{}, {1}}); });
    pre->close(); post->close(); head->close(); return checked;
}

class OnceFail final : public Kernel {
public:
    explicit OnceFail(std::shared_ptr<Kernel> delegate)
        : Kernel(delegate->runtime(), {"injected-user-library", "test-v1", delegate->identity().operation, 1}), delegate_(delegate) {}
    void call(const std::vector<Tensor *> &args) override {
        ++calls_; if (!failures_++) throw std::runtime_error("injected HC failure"); delegate_->call(args);
    }
    void close() override {}
    uint64_t calls() const override { return calls_; }
    uint64_t failures() const override { return calls_ ? 1 : 0; }
private:
    std::shared_ptr<Kernel> delegate_;
    uint64_t calls_ = 0, failures_ = 0;
};

int main(int argc, char **argv) {
    try {
        check(argc == 2 && std::getenv("SLURM_JOB_ID"), "HC verification requires Slurm allocation");
        noPython(); llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &r = llaisys::core::context().runtime();
        std::ifstream manifest(argv[1]); std::string magic, tl, aten, split_path; size_t components;
        int64_t hidden, copies, iterations; double norm_eps, hc_eps;
        manifest >> magic >> tl >> aten >> hidden >> copies >> iterations >> norm_eps >> hc_eps >> std::quoted(split_path) >> components;
        check(magic == "LLAISYS_NATIVE_HC_V1" && components == 9 && hidden == 4096 && copies == 4 && iterations == 20
              && aten == StructuralKernel::compiledVersion(), "wrong HC fixture/configuration");
        const auto contract_checks = contractChecks(r, aten);
        size_t comparisons = 0, negative = 0, cases = 0, recoveries = 0;
        auto reject = [&](auto &&fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
            check(failed, "invalid HC call accepted"); ++negative; };
        std::map<std::string, std::shared_ptr<Kernel>> ops;
        for (auto op : {"tensor_cast", "dense_linear", "row_inv_rms", "tensor_row_multiply", "hc_pre_reduce", "hc_post_mix", "hc_head_weights"}) {
            llaisys::backends::aten::StructuralOptions options;
            options.epsilon = std::string(op) == "hc_head_weights" ? hc_eps : norm_eps;
            ops.emplace(op, std::make_shared<StructuralKernel>(r,
                llaisys::backends::native::KernelIdentity{"aten-reference", aten, op, 1}, options));
        }
        ops.emplace("hc_split_sinkhorn", std::make_shared<TileKernel>(r, split_path,
            llaisys::backends::native::KernelIdentity{"tilelang", tl, "hc_split_sinkhorn", 1}));
        HCBackends b{ops.at("tensor_cast"), ops.at("dense_linear"), ops.at("row_inv_rms"), ops.at("tensor_row_multiply"),
                     ops.at("hc_pre_reduce"), ops.at("hc_split_sinkhorn"), ops.at("hc_post_mix"), ops.at("hc_head_weights")};
        for (size_t component = 0; component < components; ++component) {
            std::string name; int head; size_t count; manifest >> name >> head;
            HCConfig cfg{hidden, copies, head != 0};
            std::map<std::string, Value> weights;
            for (int i = 0; i < 3; ++i) check(weights.insert(read(manifest, r)).second, "duplicate HC weight");
            HCWeights wts{weights.at("projection").tensor, weights.at("scale").tensor, weights.at("base").tensor};
            HyperConnection model(cfg, wts, b);
            reject([&] { auto bad = cfg; bad.copies = 0; HCWorkspace invalid(r, bad, 1); });
            reject([&] { auto bad = cfg; bad.hidden = INT64_MAX; HyperConnection invalid(bad, wts, b); });
            reject([&] { HCWorkspace invalid(r, cfg, 0); });
            reject([&] { auto bad = wts; bad.base.reset(); HyperConnection invalid(cfg, bad, b); });
            reject([&] { auto bad = b; bad.dense = bad.cast; HyperConnection invalid(cfg, wts, bad); });
            manifest >> count; check(count == 6, "expected six HC token/activation cases");
            for (size_t i = 0; i < count; ++i) {
                std::string case_name; int64_t tokens; size_t tensor_count; manifest >> case_name >> tokens >> tensor_count;
                check(tensor_count == static_cast<size_t>(head ? 5 : 9), "wrong HC trace count");
                std::map<std::string, Value> values;
                for (size_t j = 0; j < tensor_count; ++j) check(values.insert(read(manifest, r)).second, "duplicate HC trace");
                auto &x = *values.at("input").tensor;
                auto execute = [&](HyperConnection &layer, HCWorkspace &w) {
                    auto &out = head ? layer.head(x, w) : layer.pre(x, w);
                    std::map<std::string, Tensor *> actual{{"input", &x}, {"reduced", &out}, {"mixes", &w.mixes},
                        {"inverse_rms", &w.inverse_rms}, {"pre_weights", &w.pre_weights}};
                    if (!head) {
                        auto &branch = *values.at("branch").tensor;
                        actual.emplace("branch", &branch); actual.emplace("post_weights", &w.post_weights);
                        actual.emplace("combination", &w.combination); actual.emplace("expanded", &layer.post(branch, w));
                    }
                    for (const auto &[tag, tensor] : actual) { exact(*tensor, values.at(tag).expected, name + "/" + case_name + "/" + tag); ++comparisons; }
                    check(!w.failed() && !w.awaitingPost(), "HC workspace state incorrect");
                };
                HCWorkspace w(r, cfg, tokens);
                for (int repeat = 0; repeat < 2; ++repeat) { execute(model, w); ++cases; }
                if (i == 1) {
                    HCWorkspace fresh(r, cfg, tokens);
                    Tensor wrong(r, x.shape(), {kDLFloat, 32, 1});
                    reject([&] { if (head) model.head(wrong, fresh); else model.pre(wrong, fresh); });
                    reject([&] { if (head) model.pre(x, fresh); else model.head(x, fresh); });
                    reject([&] { model.post(fresh.reduced, fresh); });
                    auto alias = fresh.reduced.asStrided({1, tokens, copies, hidden / copies}, {tokens * hidden, hidden, hidden / copies, 1});
                    reject([&] { if (head) model.head(*alias, fresh); else model.pre(*alias, fresh); });
                    HCWorkspace wrong_tokens(r, cfg, tokens + 1);
                    reject([&] { if (head) model.head(x, wrong_tokens); else model.pre(x, wrong_tokens); });
                    HyperConnection foreign(cfg, wts, b);
                    reject([&] { if (head) foreign.head(x, w); else foreign.pre(x, w); });
                    std::exception_ptr error;
                    std::thread t([&] { try { if (head) model.head(x, fresh); else model.pre(x, fresh); } catch (...) { error = std::current_exception(); } });
                    t.join(); check(static_cast<bool>(error), "cross-thread HC accepted"); ++negative;
                    if (!head) {
                        model.pre(x, fresh); reject([&] { model.pre(x, fresh); });
                        auto residual_alias = x.asStrided({1, tokens, hidden}, {tokens * hidden, hidden, 1});
                        reject([&] { model.post(*residual_alias, fresh); });
                        model.post(*values.at("branch").tensor, fresh);
                        reject([&] { model.post(*values.at("branch").tensor, fresh); });
                        reject([&] { model.pre(fresh.expanded, fresh); });
                    }
                    auto bad = b; auto target = head ? b.head_weights : b.split;
                    auto injected = std::make_shared<OnceFail>(target);
                    if (head) bad.head_weights = injected; else bad.split = injected;
                    HyperConnection broken(cfg, wts, bad); HCWorkspace partial(r, cfg, tokens);
                    auto before = target->calls();
                    reject([&] { if (head) broken.head(x, partial); else broken.pre(x, partial); });
                    check(partial.failed() && target->calls() == before && broken.failures() == 1, "HC failure/fallback not propagated");
                    reject([&] { if (head) broken.head(x, partial); else broken.pre(x, partial); });
                    HCWorkspace recovered(r, cfg, tokens); execute(broken, recovered); ++recoveries;
                    if (!head) {
                        auto bad_post = b; bad_post.post_mix = std::make_shared<OnceFail>(b.post_mix);
                        HyperConnection interrupted(cfg, wts, bad_post); HCWorkspace pending(r, cfg, tokens);
                        interrupted.pre(x, pending); before = b.post_mix->calls();
                        reject([&] { interrupted.post(*values.at("branch").tensor, pending); });
                        check(pending.failed() && b.post_mix->calls() == before, "post failure silently fell back");
                        reject([&] { interrupted.post(*values.at("branch").tensor, pending); });
                        HCWorkspace reset(r, cfg, tokens); execute(interrupted, reset); ++recoveries;
                    }
                }
            }
            std::cout << "PASS " << name << " cases=6 repeats=2\n";
        }
        noPython(); std::ostringstream backend; backend << '{'; size_t index = 0;
        for (const auto &[name, op] : ops) {
            check(op->failures() == 0, "unexpected HC baseline kernel failure");
            if (index++) backend << ',';
            backend << std::quoted(name) << ":{\"backend\":" << std::quoted(op->identity().backend)
                    << ",\"calls\":" << op->calls() << ",\"fallback\":false}"; op->close();
        }
        backend << '}'; check(!TileKernel::destructionErrors(), "TileLang teardown error");
        std::cout << "{\"all_passed\":true,\"components\":" << components << ",\"cases\":" << cases
                  << ",\"byte_comparisons\":" << comparisons << ",\"negative_checks\":" << negative
                  << ",\"failure_recoveries\":" << recoveries << ",\"python_loaded\":false,\"aten_cpp\":true,\"fallback\":false"
                  << ",\"structural_contract_checks\":" << contract_checks
                  << ",\"allow_tf32_cublas\":" << (at::globalContext().allowTF32CuBLAS() ? "true" : "false")
                  << ",\"backends\":" << backend.str() << "}\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

#include <ATen/Context.h>
#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/backends/hadamard/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/model.hpp"
#include "native_fixture_utils.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace native_fixture;
using namespace llaisys::models::deepseek_v4;
using Kernel = llaisys::backends::native::Kernel;
using Identity = llaisys::backends::native::KernelIdentity;
using Structural = llaisys::backends::aten::StructuralKernel;
using Tile = llaisys::backends::tilelang::NativeKernel;

// The fixture contains only explicit backend bundles, token IDs and expected
// outputs. Config parsing, parameter schema, loading and 43-layer execution
// are production Model/Checkpoint responsibilities, not fixture assembly.
int main(int argc, char **argv) {
    try {
        check(std::getenv("SLURM_JOB_ID"), "native model verification requires Slurm"); noPython();
        if (argc == 5 && std::string(argv[1]) == "--validate") {
            auto config = ModelConfig::fromDirectory(argv[2], std::stoll(argv[4]));
            Checkpoint checkpoint(argv[3]); checkpoint.validate(config);
            std::cout << "{\"all_passed\":true,\"layers\":" << config.layers << ",\"required_weights\":"
                      << checkpoint.records().size() - checkpoint.auxiliaryCount() << ",\"auxiliary_weights\":"
                      << checkpoint.auxiliaryCount() << ",\"loaded_weights\":0,\"gpu_allocated\":false}\n";
            return 0;
        }
        check(argc == 2, "expected native model manifest");
        std::ifstream in(argv[1]); std::string magic, tl, aten, had, source, path; int64_t capacity; size_t bundles;
        in >> magic >> tl >> aten >> had >> std::quoted(source) >> std::quoted(path) >> capacity >> bundles;
        check(in.good() && magic == "LLAISYS_NATIVE_MODEL_V1" && aten == Structural::compiledVersion() && bundles == 17,
              "invalid native model manifest/version");
        auto cfg = ModelConfig::fromDirectory(source, capacity); Checkpoint checkpoint(path); checkpoint.validate(cfg);
        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0); auto &r = llaisys::core::context().runtime();
        KernelBindings shared;
        for (size_t i = 0; i < bundles; ++i) {
            std::string key, op, library; in >> key >> op >> std::quoted(library);
            check(shared.emplace(key, std::make_shared<Tile>(r, library, Identity{"tilelang", tl, op, 1})).second, "duplicate bundle");
        }
        auto make = [&](bool hash) {
            auto ops = shared; llaisys::backends::aten::StructuralOptions options;
            options.epsilon = cfg.norm_epsilon; options.hash_routing = hash; options.route_scale = cfg.route_scale;
            options.activation_limit = cfg.activation_limit; options.score_function = cfg.score_function;
            for (auto op : {"tensor_cast", "tensor_fill", "dense_linear", "compressor_prepare", "compressor_pool", "rms_norm",
                 "rotary_inplace", "tensor_scale", "indexer_scores", "indexer_mask", "indexer_topk", "indexer_remap",
                 "row_inv_rms", "tensor_row_multiply", "attention_prepare", "grouped_linear", "row_gather", "router",
                 "moe_dispatch", "moe_combine", "moe_finalize", "expert_activation", "hc_pre_reduce", "hc_post_mix",
                 "token_embedding", "greedy_argmax", "rotary_frequencies"})
                ops.emplace(op, std::make_shared<Structural>(r, Identity{"aten-reference", aten, op, 1}, options));
            options.inverse_rotary = true;
            ops.emplace("inverse_rotary", std::make_shared<Structural>(r, Identity{"aten-reference", aten, "rotary_inplace", 1}, options));
            options.inverse_rotary = false; options.epsilon = cfg.hc_epsilon;
            ops.emplace("hc_head_weights", std::make_shared<Structural>(r, Identity{"aten-reference", aten, "hc_head_weights", 1}, options));
            ops.emplace("hadamard", std::make_shared<llaisys::backends::hadamard::NativeKernel>(r,
                Identity{"cuda-hadamard", had, "hadamard", 1}, static_cast<float>(std::pow(cfg.index_dimension, -0.5))));
            return ops;
        };
        auto learned = make(false), hash = make(true); ModelBackends backends; backends.global = learned;
        for (int64_t i = 0; i < cfg.layers; ++i) backends.layers.push_back(i < cfg.hash_layers ? hash : learned);
        std::cout << "loading " << cfg.layers << " layers and " << checkpoint.records().size() - checkpoint.auxiliaryCount()
                  << " main parameters directly from MP1\n" << std::flush;
        Model model(r, cfg, checkpoint, backends); size_t comparisons = 0, steps = 0, negatives = 0;
        std::cout << "loaded " << checkpoint.loadedCount() << " weights, " << checkpoint.loadedBytes() << " bytes\n" << std::flush;
        for (bool compressed : {false, true}) {
            std::string expected; in >> std::quoted(expected);
            auto &f = const_cast<Tensor &>(model.frequencies(compressed)); exact(f, expected, compressed ? "YaRN frequencies" : "standard frequencies"); ++comparisons;
        }
        size_t cases, repeats; in >> cases >> repeats;
        check(in.good() && cases > 0 && cases <= 64 && repeats > 0 && repeats <= 10, "invalid model cases/repeats");
        auto reject = [&](auto &&fn) {
            bool threw = false; try { fn(); } catch (const std::exception &) { threw = true; }
            check(threw, "invalid native model input accepted"); ++negatives;
        };
        std::ostringstream results; results << '[';
        ModelState state(r, cfg);
        for (size_t c = 0; c < cases; ++c) {
            std::string id, ids_path; int64_t tokens; size_t outputs;
            in >> id >> tokens >> outputs >> std::quoted(ids_path);
            check(in.good() && tokens > 0 && outputs > 0 && tokens + outputs <= capacity, "invalid token sequence");
            std::vector<std::string> logits_paths(outputs); std::vector<int64_t> wanted_ids(outputs);
            for (size_t i = 0; i < outputs; ++i) in >> wanted_ids[i] >> std::quoted(logits_paths[i]);
            check(in.good(), "truncated output manifest");
            auto host_ids = bytes(ids_path, tokens * sizeof(int64_t)); Tensor initial(r, {1, tokens}, {kDLInt, 64, 1});
            initial.upload(host_ids.data(), host_ids.size());
            if (!c) {
                reject([&] { ModelWorkspace bad(r, cfg, 0, 0); });
                reject([&] { ModelWorkspace bad(r, cfg, capacity, 1); });
                reject([&] { ModelWorkspace bad(r, cfg, -1, 1); });
                ModelWorkspace fresh(r, cfg, 0, tokens);
                reject([&] { model.forward(initial, state, fresh); });
            }
            for (size_t repeat = 0; repeat < repeats; ++repeat) {
                model.reset(state); Tensor next(r, {1, 1}, {kDLInt, 64, 1});
                std::vector<int64_t> generated;
                for (size_t i = 0; i < outputs; ++i) {
                    ModelWorkspace workspace(r, cfg, state.position(), i ? 1 : tokens);
                    auto result = model.forward(i ? next : initial, state, workspace);
                    check(result.logits && result.next_ids && !state.failed() && !workspace.failed(), "incomplete model output");
                    // Actual model output feeds the next step. Golden IDs are
                    // compared only; they are never substituted into decoding.
                    int64_t actual; result.next_ids->download(&actual, sizeof(actual)); generated.push_back(actual);
                    exact(*result.logits, logits_paths[i], id + "/logits/" + std::to_string(i)); ++comparisons;
                    check(actual == wanted_ids[i] && state.position() == tokens + static_cast<int64_t>(i), "wrong greedy output/position");
                    for (const auto &s : state.layers) check(!s->failed() && s->position() == state.position(), "layer/model state mismatch");
                    ++steps; next.upload(&actual, sizeof(actual));
                    reject([&] { model.forward(i ? next : initial, state, workspace); });
                    check(!state.failed(), "position validation poisoned healthy model");
                    std::cout << "PASS " << id << " repeat=" << repeat << " step=" << i << " token=" << actual << '\n' << std::flush;
                }
                if (c || repeat) results << ',';
                results << "{\"case\":" << std::quoted(id) << ",\"repeat\":" << repeat << ",\"input_tokens\":" << tokens
                        << ",\"output_tokens\":" << outputs << ",\"generated_ids\":[";
                for (size_t i = 0; i < generated.size(); ++i) { if (i) results << ','; results << generated[i]; } results << "]}";
            }
        }
        in >> std::ws; check(in.eof(), "unconsumed model manifest"); results << ']';
        std::map<Kernel *, std::shared_ptr<Kernel>> unique;
        for (const auto *ops : {&learned, &hash}) for (const auto &[key, op] : *ops) unique.emplace(op.get(), op);
        std::ostringstream stats; stats << '['; size_t n = 0;
        for (const auto &[pointer, op] : unique) {
            check(!op->failures(), "unexpected baseline backend failure"); if (n++) stats << ',';
            stats << "{\"backend\":" << std::quoted(op->identity().backend) << ",\"operation\":" << std::quoted(op->identity().operation)
                  << ",\"calls\":" << op->calls() << ",\"fallback\":false}"; op->close();
        }
        stats << ']'; noPython(); check(!Tile::destructionErrors(), "TileLang teardown failed");
        std::cout << "{\"all_passed\":true,\"layers\":" << cfg.layers << ",\"loaded_weights\":" << checkpoint.loadedCount()
                  << ",\"loaded_bytes\":" << checkpoint.loadedBytes() << ",\"auxiliary_weights\":" << checkpoint.auxiliaryCount()
                  << ",\"steps\":" << steps << ",\"byte_comparisons\":" << comparisons << ",\"negative_checks\":" << negatives
                  << ",\"python_loaded\":false,\"aten_cpp\":true,\"fallback\":false,\"free_generation\":true,\"allow_tf32_cublas\":"
                  << (at::globalContext().allowTF32CuBLAS() ? "true" : "false") << ",\"cases\":" << results.str() << ",\"backends\":" << stats.str() << "}\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}

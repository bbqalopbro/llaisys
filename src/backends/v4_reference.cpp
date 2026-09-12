#include "v4_reference.hpp"
#include "aten/structural_kernel.hpp"
#include "tilelang/native_kernel.hpp"
#include "hadamard/native_kernel.hpp"
#include <cmath>

namespace llaisys::backends::v4_reference {
using namespace models::deepseek_v4;
ModelBackends make(core::Runtime &r, const ModelConfig &cfg, const std::map<std::string, Bundle> &bundles,
                   const std::string &tl, const std::string &had) {
    const std::map<std::string, std::string> expected{{"quant_hidden", "act_quant"}, {"quant_query_rank", "act_quant"},
        {"quant_output", "act_quant"}, {"quant_intermediate", "act_quant"}, {"latent_qdq", "act_quant"}, {"indexer_qdq", "fp4_act_quant"},
        {"gemm_query_a", "fp8_gemm"}, {"gemm_query_b", "fp8_gemm"}, {"gemm_latent", "fp8_gemm"}, {"gemm_output", "fp8_gemm"},
        {"indexer_gemm", "fp8_gemm"}, {"fp8_gate", "fp8_gemm"}, {"fp8_down", "fp8_gemm"}, {"fp4_gate", "fp4_gemm"},
        {"fp4_down", "fp4_gemm"}, {"sparse_attn", "sparse_attn"}, {"hc_split_sinkhorn", "hc_split_sinkhorn"}};
    if (bundles.size() != expected.size() || tl.empty() || had.empty()) throw std::invalid_argument("incomplete explicit native reference bundles");
    KernelBindings shared;
    for (const auto &[key, op] : expected) {
        auto it = bundles.find(key);
        if (it == bundles.end() || it->second.operation != op) throw std::invalid_argument("native reference bundle operation mismatch");
        shared.emplace(key, std::make_shared<tilelang::NativeKernel>(r, it->second.library, native::KernelIdentity{"tilelang", tl, op, 1}));
    }
    auto bind = [&](bool hash) {
        auto ops = shared; aten::StructuralOptions options; options.epsilon = cfg.norm_epsilon;
        options.hash_routing = hash; options.route_scale = cfg.route_scale; options.activation_limit = cfg.activation_limit; options.score_function = cfg.score_function;
        auto add = [&](const char *key, const char *op) {
            ops.emplace(key, std::make_shared<aten::StructuralKernel>(r, native::KernelIdentity{"aten-reference", aten::StructuralKernel::compiledVersion(), op, 1}, options));
        };
        for (auto op : {"tensor_cast", "tensor_fill", "dense_linear", "compressor_prepare", "compressor_pool", "rms_norm", "rotary_inplace",
             "tensor_scale", "indexer_scores", "indexer_mask", "indexer_topk", "indexer_remap", "row_inv_rms", "tensor_row_multiply",
             "attention_prepare", "grouped_linear", "row_gather", "router", "moe_dispatch", "moe_combine", "moe_finalize", "expert_activation",
             "hc_pre_reduce", "hc_post_mix", "token_embedding", "greedy_argmax", "rotary_frequencies"}) add(op, op);
        options.inverse_rotary = true; add("inverse_rotary", "rotary_inplace");
        options.inverse_rotary = false; options.epsilon = cfg.hc_epsilon; add("hc_head_weights", "hc_head_weights");
        ops.emplace("hadamard", std::make_shared<hadamard::NativeKernel>(r, native::KernelIdentity{"cuda-hadamard", had, "hadamard", 1},
            static_cast<float>(std::pow(cfg.index_dimension, -0.5)))); return ops;
    };
    auto learned = bind(false), hash = bind(true); ModelBackends result; result.global = learned;
    for (int64_t i = 0; i < cfg.layers; ++i) result.layers.push_back(i < cfg.hash_layers ? hash : learned);
    return result;
}
}

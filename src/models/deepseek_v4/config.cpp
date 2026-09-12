#include "config.hpp"
#include "json_detail.hpp"
#include <algorithm>
#include <set>

namespace llaisys::models::deepseek_v4 {
using namespace json_detail;
ModelConfig ModelConfig::fromDirectory(const std::string &directory, int64_t limit) {
    auto root = file(directory + "/config.json"), inference = file(directory + "/inference/config.json");
    require(string(root, "model_type") == "deepseek_v4", "expected actual deepseek_v4 config");
    const auto &arch = get(root, "architectures");
    require(arch.IsArray() && arch.Size() == 1 && string(arch[0]) == "DeepseekV4ForCausalLM", "unsupported V4 architectures");
    const auto &quant = get(root, "quantization_config"), &rope = get(root, "rope_scaling");
    require(string(quant, "quant_method") == "fp8" && string(quant, "fmt") == "e4m3"
            && string(quant, "scale_fmt") == "ue8m0" && string(quant, "activation_scheme") == "dynamic"
            && integers(quant, "weight_block_size") == std::vector<int64_t>{128, 128}, "unsupported V4 dense quantization contract");
    require(string(root, "expert_dtype") == "fp4" && string(root, "torch_dtype") == "bfloat16"
            && string(root, "hidden_act") == "silu" && string(root, "topk_method") == "noaux_tc"
            && string(rope, "type") == "yarn", "unsupported V4 dtype/activation/routing/RoPE contract");
    boolean(root, "attention_bias", false); boolean(root, "tie_word_embeddings", false); boolean(root, "norm_topk_prob", true);
    require(integer(root, "num_key_value_heads") == 1 && integer(root, "n_shared_experts") == 1
            && number(root, "attention_dropout") == 0, "unsupported V4 attention/shared expert configuration");
    ModelConfig c; c.capacity = limit;
    auto alias = [&](const char *hf, const char *infer) {
        auto v = integer(root, hf); require(integer(inference, infer) == v, std::string("HF/inference config mismatch: ") + hf); return v;
    };
    auto num_alias = [&](const char *hf, const char *infer) {
        auto v = number(root, hf); require(number(inference, infer) == v, std::string("HF/inference number mismatch: ") + hf); return v;
    };
    c.hidden = alias("hidden_size", "dim"); c.vocabulary = alias("vocab_size", "vocab_size"); c.layers = alias("num_hidden_layers", "n_layers");
    c.heads = alias("num_attention_heads", "n_heads"); c.dimension = alias("head_dim", "head_dim"); c.query_rank = alias("q_lora_rank", "q_lora_rank");
    c.output_groups = alias("o_groups", "o_groups"); c.output_rank = alias("o_lora_rank", "o_lora_rank");
    c.rope_dimension = alias("qk_rope_head_dim", "rope_head_dim"); c.window = alias("sliding_window", "window_size");
    c.index_heads = alias("index_n_heads", "index_n_heads"); c.index_dimension = alias("index_head_dim", "index_head_dim"); c.index_topk = alias("index_topk", "index_topk");
    c.intermediate = alias("moe_intermediate_size", "moe_inter_dim"); c.experts = alias("n_routed_experts", "n_routed_experts");
    c.top_k = alias("num_experts_per_tok", "n_activated_experts"); c.hash_layers = alias("num_hash_layers", "n_hash_layers");
    c.copies = alias("hc_mult", "hc_mult"); c.sinkhorn_iterations = alias("hc_sinkhorn_iters", "hc_sinkhorn_iters");
    c.norm_epsilon = number(root, "rms_norm_eps"); c.hc_epsilon = number(root, "hc_eps");
    // These fields are omitted in the published inference config and default
    // to 1e-6 in its ModelArgs. Explicit overrides must agree with HF config.
    for (auto item : {std::pair<const char *, double>{"norm_eps", c.norm_epsilon}, {"hc_eps", c.hc_epsilon}})
        require((inference.HasMember(item.first) ? number(inference, item.first) : 1e-6) == item.second, "epsilon config mismatch");
    c.route_scale = num_alias("routed_scaling_factor", "route_scale"); c.activation_limit = num_alias("swiglu_limit", "swiglu_limit");
    c.rope_theta = num_alias("rope_theta", "rope_theta"); c.compressed_theta = num_alias("compress_rope_theta", "compress_rope_theta");
    c.score_function = string(root, "scoring_func"); require(string(inference, "score_func") == c.score_function, "routing score config mismatch");
    c.max_position = integer(root, "max_position_embeddings");
    c.original_sequence = integer(rope, "original_max_position_embeddings");
    c.rope_factor = number(rope, "factor"); c.beta_fast = number(rope, "beta_fast"); c.beta_slow = number(rope, "beta_slow");
    require(integer(inference, "original_seq_len") == c.original_sequence && number(inference, "rope_factor") == c.rope_factor
            && number(inference, "beta_fast") == c.beta_fast && number(inference, "beta_slow") == c.beta_slow, "YaRN config mismatch");
    require(string(inference, "dtype") == "fp8" && string(inference, "scale_fmt") == "ue8m0"
            && string(inference, "expert_dtype") == "fp4" && integer(inference, "n_shared_experts") == 1, "MP1 quantization config mismatch");
    auto ratios = integers(root, "compress_ratios"); require(ratios == integers(inference, "compress_ratios"), "compression ratios disagree");
    c.mtp_stages = integer(inference, "n_mtp_layers"); c.markov_rank = alias("dspark_markov_rank", "dspark_markov_rank");
    c.declared_nextn_layers = integer(root, "num_nextn_predict_layers");
    c.dspark_targets = integers(root, "dspark_target_layer_ids");
    require(c.dspark_targets == integers(inference, "dspark_target_layer_ids") && c.layers > 0 && c.mtp_stages >= 0
            && c.layers <= 4096 && c.mtp_stages <= 64 && ratios.size() == static_cast<size_t>(c.layers + c.mtp_stages), "invalid main/MTP compression coverage");
    for (size_t i = 0; i < ratios.size(); ++i) require(i < static_cast<size_t>(c.layers) ? ratios[i] == 0 || ratios[i] == 4 || ratios[i] == 128 : ratios[i] == 0,
                                                    "unsupported main/MTP attention ratio");
    c.ratios.assign(ratios.begin(), ratios.begin() + c.layers); c.validate(); return c;
}

void ModelConfig::validate() const {
    for (auto v : {hidden, vocabulary, layers, heads, dimension, query_rank, output_groups, output_rank, rope_dimension,
                   window, capacity, max_position, index_heads, index_dimension, index_topk, intermediate, experts, top_k, copies, sinkhorn_iterations, markov_rank})
        require(v > 0 && v <= INT32_MAX / 4, "invalid/overflowing native model dimension");
    require(layers <= 4096 && experts <= 65536 && copies <= 64 && sinkhorn_iterations <= 1000
            && capacity <= max_position && capacity >= 128 && hash_layers >= 0 && hash_layers <= layers
            && top_k <= experts && hidden % 128 == 0 && intermediate % 128 == 0 && query_rank % 128 == 0
            && output_rank % 128 == 0 && heads >= 16 && heads % output_groups == 0 && dimension % 128 == 0
            && rope_dimension % 2 == 0 && dimension > rope_dimension && (dimension - rope_dimension) % 64 == 0
            && index_dimension >= rope_dimension && index_dimension % 128 == 0 && (index_dimension & (index_dimension - 1)) == 0,
            "unsupported native model dimensions");
    require(hidden <= INT32_MAX / copies && heads <= INT32_MAX / dimension && output_groups <= INT32_MAX / output_rank
            && index_heads <= INT32_MAX / index_dimension, "model projection size overflow");
    require(ratios.size() == static_cast<size_t>(layers), "missing main-layer ratios");
    for (auto ratio : ratios) require(ratio == 0 || ratio == 4 || ratio == 128, "unsupported attention ratio");
    for (auto v : {norm_epsilon, hc_epsilon, route_scale, rope_theta, compressed_theta, rope_factor, beta_fast, beta_slow})
        require(std::isfinite(v) && v > 0, "invalid model numerical option");
    require(rope_theta > 1 && compressed_theta > 1 && original_sequence >= 0 && std::isfinite(activation_limit) && activation_limit >= 0
            && (score_function == "softmax" || score_function == "sigmoid" || score_function == "sqrtsoftplus"), "invalid RoPE/router/activation options");
    require(mtp_stages >= 0 && mtp_stages <= 64 && declared_nextn_layers >= 0 && (!mtp_stages || !dspark_targets.empty()), "invalid MTP metadata");
    require((layers + mtp_stages) * (6 * (experts + 1) + 64) <= 1000000, "model schema exceeds native loader tensor-count limit");
    std::set<int64_t> targets;
    for (auto id : dspark_targets) require(id >= 0 && id < layers && targets.insert(id).second, "invalid/duplicate DSpark target layer");
}

BlockConfig ModelConfig::block(int64_t layer) const {
    require(layer >= 0 && layer < layers, "model layer outside config");
    return {{hidden, heads, dimension, query_rank, output_groups, output_rank, rope_dimension, window, ratios.at(layer), capacity,
             index_heads, index_dimension, index_topk}, {hidden, intermediate, experts, top_k, vocabulary, layer < hash_layers}, copies};
}
} // namespace llaisys::models::deepseek_v4

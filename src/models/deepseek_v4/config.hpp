#pragma once
#include "block.hpp"
#include <string>

namespace llaisys::models::deepseek_v4 {

// MP1 base-model contract, read from BOTH actual HF and inference configs.
// No inference of architecture from model directory/name. MTP is validated
// as auxiliary checkpoint structure, not enabled for execution.
struct ModelConfig {
    int64_t hidden = 0, vocabulary = 0, layers = 0, heads = 0, dimension = 0, query_rank = 0;
    int64_t output_groups = 0, output_rank = 0, rope_dimension = 0, window = 0, capacity = 0, max_position = 0;
    int64_t index_heads = 0, index_dimension = 0, index_topk = 0;
    int64_t intermediate = 0, experts = 0, top_k = 0, hash_layers = 0;
    int64_t copies = 0, sinkhorn_iterations = 0, original_sequence = 0;
    int64_t mtp_stages = 0, markov_rank = 0, declared_nextn_layers = 0;
    double norm_epsilon = 0, hc_epsilon = 0, route_scale = 0, activation_limit = 0;
    double rope_theta = 0, compressed_theta = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0;
    std::string score_function;
    std::vector<int64_t> ratios, dspark_targets;
    static ModelConfig fromDirectory(const std::string &directory, int64_t capacity);
    void validate() const;
    BlockConfig block(int64_t layer) const;
};

} // namespace llaisys::models::deepseek_v4

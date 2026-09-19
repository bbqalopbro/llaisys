#include "sample_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void sample(tensor_t out_idx, tensor_t logits,
            float temperature, int top_k, float top_p, uint64_t seed) { llmops::metax::sample(view(out_idx), view(logits), temperature, top_k, top_p, seed, nullptr); }
}

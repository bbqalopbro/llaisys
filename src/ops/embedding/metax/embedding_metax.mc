#include "embedding_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void embedding(tensor_t out, tensor_t index, tensor_t weight) { llmops::metax::embedding(view(out), view(index), view(weight), nullptr); }
}

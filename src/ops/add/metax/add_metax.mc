#include "add_metax.hpp"
#include "../../llmops.hpp"
namespace llaisys::ops::metax {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) { llmops::metax::add(c, a, b, kernel_dtype(type), numel, nullptr); }
}

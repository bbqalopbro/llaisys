#include "cache_layout.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::core {

StandardKVCacheLayout::StandardKVCacheLayout(
    size_t block_size, size_t num_layers, size_t num_kv_heads,
    size_t head_dim, size_t element_size)
    : block_size_(block_size), num_layers_(num_layers),
      num_kv_heads_(num_kv_heads), head_dim_(head_dim),
      element_size_(element_size),
      layer_bytes_(block_size * num_kv_heads * head_dim * element_size) {
    if (block_size_ == 0 || num_layers_ == 0 || num_kv_heads_ == 0 ||
        head_dim_ == 0 || element_size_ == 0) {
        throw std::invalid_argument("StandardKVCacheLayout: dimensions must be non-zero");
    }
}

const std::string &StandardKVCacheLayout::componentName(size_t component) const {
    return component_names_.at(component);
}

size_t StandardKVCacheLayout::layerBytes(size_t component, size_t layer) const {
    if (component >= component_names_.size() || layer >= num_layers_) {
        throw std::out_of_range("StandardKVCacheLayout: index out of range");
    }
    return layer_bytes_;
}

MLACacheLayout::MLACacheLayout(
    size_t block_size, size_t num_layers, size_t latent_dim,
    size_t rope_dim, size_t element_size,
    std::vector<size_t> layer_compression_divisors)
    : block_size_(block_size), num_layers_(num_layers),
      latent_dim_(latent_dim), rope_dim_(rope_dim),
      element_size_(element_size),
      layer_compression_divisors_(std::move(layer_compression_divisors)) {
    if (block_size_ == 0 || num_layers_ == 0 || latent_dim_ == 0 ||
        element_size_ == 0) {
        throw std::invalid_argument("MLACacheLayout: dimensions must be non-zero");
    }
    if (layer_compression_divisors_.empty()) {
        layer_compression_divisors_.assign(num_layers_, 1);
    }
    if (layer_compression_divisors_.size() != num_layers_) {
        throw std::invalid_argument("MLACacheLayout: one compression divisor is required per layer");
    }
    for (size_t divisor : layer_compression_divisors_) {
        if (divisor == 0) {
            throw std::invalid_argument("MLACacheLayout: compression divisor must be non-zero");
        }
    }
    component_names_.push_back("latent");
    if (rope_dim_ > 0) component_names_.push_back("rope");
}

const std::string &MLACacheLayout::componentName(size_t component) const {
    return component_names_.at(component);
}

size_t MLACacheLayout::layerBytes(size_t component, size_t layer) const {
    if (component >= component_names_.size() || layer >= num_layers_) {
        throw std::out_of_range("MLACacheLayout: index out of range");
    }
    const size_t divisor = layer_compression_divisors_[layer];
    const size_t tokens = (block_size_ + divisor - 1) / divisor;
    const size_t dim = component == 0 ? latent_dim_ : rope_dim_;
    return tokens * dim * element_size_;
}

} // namespace llaisys::core

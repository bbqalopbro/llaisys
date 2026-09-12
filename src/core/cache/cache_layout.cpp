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

DeepSeekV4CacheLayout::DeepSeekV4CacheLayout(
    size_t block_size, size_t latent_dim, size_t index_dim,
    size_t element_size, std::vector<size_t> compression_ratios)
    : block_size_(block_size), latent_dim_(latent_dim), index_dim_(index_dim),
      element_size_(element_size),
      compression_ratios_(std::move(compression_ratios)) {
    if (block_size_ == 0 || latent_dim_ == 0 || element_size_ == 0 ||
        compression_ratios_.empty()) {
        throw std::invalid_argument(
            "DeepSeekV4CacheLayout: dimensions and layers must be non-zero");
    }
    bool has_compressed = false;
    bool has_index = false;
    for (const size_t ratio : compression_ratios_) {
        if (ratio != 0 && ratio != 4 && ratio != 128) {
            throw std::invalid_argument(
                "DeepSeekV4CacheLayout: compression ratio must be 0, 4, or 128");
        }
        has_compressed = has_compressed || ratio != 0;
        has_index = has_index || ratio == 4;
    }
    if (has_compressed) component_names_.push_back("compressed_latent");
    if (has_index) {
        if (index_dim_ == 0) {
            throw std::invalid_argument(
                "DeepSeekV4CacheLayout: ratio-4 layers require index_dim");
        }
        component_names_.push_back("index_latent");
    }
}

const std::string &
DeepSeekV4CacheLayout::componentName(size_t component) const {
    return component_names_.at(component);
}

size_t DeepSeekV4CacheLayout::compressionRatio(size_t layer) const {
    return compression_ratios_.at(layer);
}

size_t DeepSeekV4CacheLayout::layerBytes(size_t component,
                                         size_t layer) const {
    if (component >= component_names_.size() ||
        layer >= compression_ratios_.size()) {
        throw std::out_of_range("DeepSeekV4CacheLayout: index out of range");
    }
    const auto &name = component_names_[component];
    if (name == "window_latent") {
        return block_size_ * latent_dim_ * element_size_;
    }
    const size_t ratio = compression_ratios_[layer];
    if (name == "compressed_latent") {
        if (ratio == 0) return 0;
        return ((block_size_ + ratio - 1) / ratio) * latent_dim_ *
               element_size_;
    }
    if (name == "index_latent") {
        if (ratio != 4) return 0;
        return ((block_size_ + ratio - 1) / ratio) * index_dim_ *
               element_size_;
    }
    throw std::logic_error("DeepSeekV4CacheLayout: unknown component");
}

} // namespace llaisys::core

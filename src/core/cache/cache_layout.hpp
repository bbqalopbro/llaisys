#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llaisys::core {

// Describes the physical payload stored for one logical paged-cache block.
// Block scheduling only deals with block ids; attention backends interpret the
// named components through this interface.
class CacheLayout {
public:
    virtual ~CacheLayout() = default;

    virtual const char *name() const = 0;
    virtual size_t numLayers() const = 0;
    virtual size_t blockSize() const = 0;
    virtual size_t numComponents() const = 0;
    virtual const std::string &componentName(size_t component) const = 0;
    virtual size_t layerBytes(size_t component, size_t layer) const = 0;
};

using cache_layout_t = std::shared_ptr<const CacheLayout>;

// Standard GQA/MHA layout used by Qwen2:
//   key/value: [block_size, num_kv_heads, head_dim]
class StandardKVCacheLayout final : public CacheLayout {
public:
    StandardKVCacheLayout(size_t block_size, size_t num_layers,
                          size_t num_kv_heads, size_t head_dim,
                          size_t element_size);

    const char *name() const override { return "standard-kv"; }
    size_t numLayers() const override { return num_layers_; }
    size_t blockSize() const override { return block_size_; }
    size_t numComponents() const override { return component_names_.size(); }
    const std::string &componentName(size_t component) const override;
    size_t layerBytes(size_t component, size_t layer) const override;

    size_t numKVHeads() const { return num_kv_heads_; }
    size_t headDim() const { return head_dim_; }
    size_t elementSize() const { return element_size_; }

private:
    size_t block_size_;
    size_t num_layers_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t element_size_;
    size_t layer_bytes_;
    std::vector<std::string> component_names_{"key", "value"};
};

// MLA-compatible storage description. It intentionally describes storage
// only; the DeepSeek attention kernel remains a separate backend. Each token
// stores a compressed latent vector and, optionally, a decoupled RoPE vector.
class MLACacheLayout final : public CacheLayout {
public:
    MLACacheLayout(size_t block_size, size_t num_layers,
                   size_t latent_dim, size_t rope_dim,
                   size_t element_size,
                   std::vector<size_t> layer_compression_divisors = {});

    const char *name() const override { return "mla-latent"; }
    size_t numLayers() const override { return num_layers_; }
    size_t blockSize() const override { return block_size_; }
    size_t numComponents() const override { return component_names_.size(); }
    const std::string &componentName(size_t component) const override;
    size_t layerBytes(size_t component, size_t layer) const override;

private:
    size_t block_size_;
    size_t num_layers_;
    size_t latent_dim_;
    size_t rope_dim_;
    size_t element_size_;
    std::vector<size_t> layer_compression_divisors_;
    std::vector<std::string> component_names_;
};

} // namespace llaisys::core

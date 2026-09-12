#pragma once

#include "src/core/cache/cache_layout.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaisys::models::deepseek_v4 {

inline size_t checkedProduct(size_t lhs, size_t rhs) {
    const auto maximum = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (rhs && lhs > maximum / rhs) throw std::overflow_error("V4 cache size exceeds int64");
    return lhs * rhs;
}

// One layer per PagedCacheStorage. This preserves the TileLang pool's exact
// contiguous [1, blocks * (block_size + block_size/ratio), latent_dim] ABI.
// It is not pointer-compatible with the older three-component all-layer layout.
class InterleavedLayerLayout final : public core::CacheLayout {
public:
    InterleavedLayerLayout(size_t block_size, size_t latent_dim, size_t index_dim, size_t ratio)
        : block_size_(block_size), latent_dim_(latent_dim), index_dim_(index_dim), ratio_(ratio) {
        if (!block_size || !latent_dim || !index_dim || (ratio != 0 && ratio != 4 && ratio != 128))
            throw std::invalid_argument("invalid V4 interleaved dimensions or compression ratio");
        if (ratio && block_size % ratio)
            throw std::invalid_argument("V4 block size must be divisible by compression ratio");
        // The sum is bounded before addition as well as before byte multiplication.
        checkedProduct(block_size, 2);
        slots_ = block_size + (ratio ? block_size / ratio : 0);
        bytes_.push_back(checkedProduct(checkedProduct(slots_, latent_dim), 2));
        if (ratio == 4) {
            names_.push_back("index");
            bytes_.push_back(checkedProduct(checkedProduct(block_size / 4, index_dim), 2));
        }
    }

    const char *name() const override { return "v4-interleaved-full-and-compressed-latent-v1"; }
    size_t numLayers() const override { return 1; }
    size_t blockSize() const override { return block_size_; }
    size_t numComponents() const override { return names_.size(); }
    const std::string &componentName(size_t component) const override { return names_.at(component); }
    size_t layerBytes(size_t component, size_t layer) const override {
        if (layer != 0) throw std::out_of_range("one model layer per V4 interleaved storage");
        return bytes_.at(component);
    }
    size_t slots(size_t component) const { return component == 0 ? slots_ : (checkIndex(component), block_size_ / 4); }
    size_t width(size_t component) const { return component == 0 ? latent_dim_ : (checkIndex(component), index_dim_); }
    size_t ratio() const { return ratio_; }

private:
    void checkIndex(size_t component) const { (void)names_.at(component); }
    size_t block_size_, latent_dim_, index_dim_, ratio_, slots_;
    std::vector<std::string> names_{"latent"};
    std::vector<size_t> bytes_;
};

} // namespace llaisys::models::deepseek_v4

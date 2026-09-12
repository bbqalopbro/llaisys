#pragma once
#include "config.hpp"
#include <map>
#include <set>

namespace llaisys::models::deepseek_v4 {
struct WeightSpec {
    std::vector<int64_t> shape;
    std::string dtype;
    bool bf16_source_allowed = false, int64_source_allowed = false;
};
struct WeightRecord { std::vector<int64_t> shape; std::string dtype; uint64_t offset, bytes; };
std::map<std::string, WeightSpec> expectedWeights(const ModelConfig &config, bool include_mtp);

// Header-first MP1 loader. Reject unknown/missing keys, duplicate JSON names,
// mismatched dtypes/shapes, out-of-range/overlapping/holey payload extents.
// The file descriptor is retained; loading never follows a changed path.
class Checkpoint {
public:
    explicit Checkpoint(const std::string &path);
    ~Checkpoint();
    Checkpoint(const Checkpoint &) = delete;
    Checkpoint &operator=(const Checkpoint &) = delete;
    void validate(const ModelConfig &config);
    std::shared_ptr<backends::native::Tensor> load(const std::string &key, core::Runtime &runtime);
    void finish();
    const std::map<std::string, WeightRecord> &records() const { return records_; }
    const std::string &headerBytes() const { return header_; }
    uint64_t sizeBytes() const;
    size_t loadedCount() const { return loaded_.size(); }
    uint64_t loadedBytes() const { return loaded_bytes_; }
    size_t auxiliaryCount() const { return auxiliary_; }
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string header_;
    std::map<std::string, WeightRecord> records_;
    std::set<std::string> required_, loaded_;
    size_t auxiliary_ = 0;
    uint64_t loaded_bytes_ = 0;
    bool validated_ = false, failed_ = false;
};
} // namespace llaisys::models::deepseek_v4

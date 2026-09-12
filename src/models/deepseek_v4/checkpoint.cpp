#include "checkpoint.hpp"
#include "json_detail.hpp"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace llaisys::models::deepseek_v4 {
using json_detail::require;
namespace {
DLDataType dtype(const std::string &name) {
    if (name == "BF16") return {kDLBfloat, 16, 1};
    if (name == "F32") return {kDLFloat, 32, 1};
    if (name == "I32") return {kDLInt, 32, 1};
    if (name == "I64") return {kDLInt, 64, 1};
    if (name == "F8_E4M3") return {kDLFloat8_e4m3fn, 8, 1};
    if (name == "F8_E8M0") return {kDLFloat8_e8m0fnu, 8, 1};
    if (name == "F4") return {kDLFloat4_e2m1fn, 4, 1};
    throw std::invalid_argument("unsupported checkpoint dtype: " + name);
}
}

std::map<std::string, WeightSpec> expectedWeights(const ModelConfig &c, bool include_mtp) {
    c.validate(); std::map<std::string, WeightSpec> result;
    auto add = [&](const std::string &key, std::vector<int64_t> shape, const char *dt, bool bf = false, bool i64 = false) {
        require(result.emplace(key, WeightSpec{std::move(shape), dt, bf, i64}).second, "duplicate expected weight");
    };
    auto linear = [&](const std::string &p, int64_t n, int64_t k, bool fp4) {
        add(p + ".weight", {n, k}, fp4 ? "F4" : "F8_E4M3");
        add(p + ".scale", {fp4 ? n : (n + 127) / 128, k / (fp4 ? 32 : 128)}, "F8_E8M0");
    };
    auto norm = [&](const std::string &p, int64_t n) { add(p + ".weight", {n}, "F32", true); };
    auto compressor = [&](const std::string &p, int64_t ratio, int64_t dim) {
        int64_t width = (ratio == 4 ? 2 : 1) * dim;
        add(p + ".ape", {ratio, width}, "F32");
        // Published MP1 stores projection weights in BF16; Compressor owns
        // their one-time FP32 promotion, just as the official parameter load.
        add(p + ".wkv.weight", {width, c.hidden}, "F32", true);
        add(p + ".wgate.weight", {width, c.hidden}, "F32", true); norm(p + ".norm", dim);
    };
    auto block = [&](const std::string &p, int64_t ratio, bool hash) {
        for (auto part : {"attn", "ffn"}) {
            const auto h = p + ".hc_" + part; const int64_t width = (2 + c.copies) * c.copies;
            add(h + "_fn", {width, c.copies * c.hidden}, "F32"); add(h + "_scale", {3}, "F32"); add(h + "_base", {width}, "F32");
            norm(p + "." + part + "_norm", c.hidden);
        }
        const auto a = p + ".attn";
        add(a + ".attn_sink", {c.heads}, "F32"); linear(a + ".wq_a", c.query_rank, c.hidden, false); norm(a + ".q_norm", c.query_rank);
        linear(a + ".wq_b", c.heads * c.dimension, c.query_rank, false); linear(a + ".wkv", c.dimension, c.hidden, false); norm(a + ".kv_norm", c.dimension);
        add(a + ".wo_a.weight", {c.output_groups * c.output_rank, c.heads * c.dimension / c.output_groups}, "BF16");
        linear(a + ".wo_b", c.hidden, c.output_groups * c.output_rank, false);
        if (ratio) compressor(a + ".compressor", ratio, c.dimension);
        if (ratio == 4) {
            linear(a + ".indexer.wq_b", c.index_heads * c.index_dimension, c.query_rank, false);
            add(a + ".indexer.weights_proj.weight", {c.index_heads, c.hidden}, "BF16");
            compressor(a + ".indexer.compressor", 4, c.index_dimension);
        }
        const auto f = p + ".ffn";
        add(f + ".gate.weight", {c.experts, c.hidden}, "BF16");
        if (hash) add(f + ".gate.tid2eid", {c.vocabulary, c.top_k}, "I32", false, true);
        else add(f + ".gate.bias", {c.experts}, "F32");
        for (int64_t e = 0; e <= c.experts; ++e) {
            bool routed = e < c.experts; auto prefix = f + (routed ? ".experts." + std::to_string(e) : ".shared_experts");
            linear(prefix + ".w1", c.intermediate, c.hidden, routed); linear(prefix + ".w3", c.intermediate, c.hidden, routed);
            linear(prefix + ".w2", c.hidden, c.intermediate, routed);
        }
    };
    auto hc_head = [&](const std::string &p) {
        add(p + "hc_head_fn", {c.copies, c.copies * c.hidden}, "F32"); add(p + "hc_head_scale", {1}, "F32"); add(p + "hc_head_base", {c.copies}, "F32");
    };
    add("embed.weight", {c.vocabulary, c.hidden}, "BF16"); norm("norm", c.hidden); add("head.weight", {c.vocabulary, c.hidden}, "F32", true); hc_head("");
    for (int64_t i = 0; i < c.layers; ++i) block("layers." + std::to_string(i), c.ratios[i], i < c.hash_layers);
    if (include_mtp) {
        require(c.mtp_stages > 0, "unexpected auxiliary MTP weights");
        for (int64_t i = 0; i < c.mtp_stages; ++i) block("mtp." + std::to_string(i), 0, false);
        linear("mtp.0.main_proj", c.hidden, c.hidden * static_cast<int64_t>(c.dspark_targets.size()), false); norm("mtp.0.main_norm", c.hidden);
        const auto last = "mtp." + std::to_string(c.mtp_stages - 1) + ".";
        norm(last + "norm", c.hidden); hc_head(last);
        add(last + "markov_head.markov_w1.weight", {c.vocabulary, c.markov_rank}, "BF16");
        add(last + "markov_head.markov_w2.weight", {c.vocabulary, c.markov_rank}, "F32", true);
        add(last + "confidence_head.proj.weight", {1, c.hidden + c.markov_rank}, "F32", true);
    }
    return result;
}

struct Checkpoint::Impl {
    int fd = -1;
    struct stat identity{};
    explicit Impl(const std::string &path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) throw std::invalid_argument("cannot open MP1 checkpoint");
        if (::fstat(fd, &identity) || !S_ISREG(identity.st_mode) || identity.st_size < 8) {
            ::close(fd); fd = -1; throw std::invalid_argument("checkpoint must be a regular safetensors file");
        }
    }
    ~Impl() { if (fd >= 0) ::close(fd); }
    void unchanged() const {
        struct stat now{};
        require(!::fstat(fd, &now) && now.st_dev == identity.st_dev && now.st_ino == identity.st_ino && now.st_size == identity.st_size
                && now.st_mtim.tv_sec == identity.st_mtim.tv_sec && now.st_mtim.tv_nsec == identity.st_mtim.tv_nsec
                && now.st_ctim.tv_sec == identity.st_ctim.tv_sec && now.st_ctim.tv_nsec == identity.st_ctim.tv_nsec,
                "checkpoint changed during loading");
    }
    std::vector<char> read(uint64_t offset, uint64_t bytes) const {
        unchanged(); require(offset <= static_cast<uint64_t>(identity.st_size) && bytes <= static_cast<uint64_t>(identity.st_size) - offset,
                             "checkpoint read extent outside file");
        std::vector<char> data(bytes); size_t done = 0;
        while (done < bytes) {
            auto count = ::pread(fd, data.data() + done, bytes - done, static_cast<off_t>(offset + done));
            if (count < 0 && errno == EINTR) continue;
            require(count > 0, "truncated/unreadable checkpoint payload"); done += static_cast<size_t>(count);
        }
        unchanged(); return data;
    }
};

Checkpoint::Checkpoint(const std::string &path) : impl_(std::make_unique<Impl>(path)) {
    auto length = impl_->read(0, 8); uint64_t size = 0;
    for (size_t i = 0; i < 8; ++i) size |= static_cast<uint64_t>(static_cast<unsigned char>(length[i])) << (8 * i);
    require(size > 0 && size <= 128 * 1024 * 1024 && size <= sizeBytes() - 8, "invalid safetensors header size");
    auto raw = impl_->read(8, size); header_.assign(raw.begin(), raw.end()); auto document = json_detail::parse(header_);
    std::vector<std::pair<uint64_t, uint64_t>> extents;
    for (auto it = document.MemberBegin(); it != document.MemberEnd(); ++it) {
        auto key = json_detail::string(it->name); const auto &entry = it->value;
        if (key == "__metadata__") {
            require(entry.IsObject(), "safetensors metadata must be an object");
            for (auto i = entry.MemberBegin(); i != entry.MemberEnd(); ++i) require(i->value.IsString(), "safetensors metadata values must be strings");
            continue;
        }
        require(entry.IsObject() && entry.MemberCount() == 3, "invalid safetensors tensor descriptor");
        auto shape = json_detail::integers(entry, "shape"), offsets = json_detail::integers(entry, "data_offsets");
        auto dt = json_detail::string(entry, "dtype");
        require(!shape.empty() && shape.size() <= 16 && offsets.size() == 2 && offsets[0] >= 0 && offsets[1] >= offsets[0]
                && static_cast<uint64_t>(offsets[1]) <= sizeBytes() - 8 - size, "invalid safetensors tensor extent");
        if (dt == "F4") require(shape.back() >= 0 && shape.back() % 2 == 0, "unaligned packed FP4 shape");
        auto bytes = backends::native::Tensor::checkedBytes(shape, dtype(dt));
        require(static_cast<uint64_t>(offsets[1] - offsets[0]) == bytes, "safetensors shape/dtype byte mismatch");
        require(records_.emplace(key, WeightRecord{std::move(shape), dt, 8 + size + static_cast<uint64_t>(offsets[0]), bytes}).second,
                "duplicate safetensors tensor");
        extents.emplace_back(offsets[0], offsets[1]);
    }
    std::sort(extents.begin(), extents.end()); uint64_t cursor = 0;
    for (auto [start, end] : extents) { require(start == cursor, "overlap/hole in safetensors payload"); cursor = end; }
    require(cursor == sizeBytes() - 8 - size && !records_.empty(), "unclaimed/truncated safetensors payload");
}
Checkpoint::~Checkpoint() = default;
uint64_t Checkpoint::sizeBytes() const { return static_cast<uint64_t>(impl_->identity.st_size); }

void Checkpoint::validate(const ModelConfig &config) {
    require(!failed_ && loaded_.empty(), "cannot revalidate used/failed checkpoint loader"); validated_ = false; required_.clear(); auxiliary_ = 0;
    const bool has_mtp = records_.lower_bound("mtp.") != records_.end() && records_.lower_bound("mtp.")->first.rfind("mtp.", 0) == 0;
    auto expected = expectedWeights(config, has_mtp);
    require(expected.size() == records_.size(), "MP1 tensor count differs from actual main/auxiliary schema");
    for (const auto &[key, spec] : expected) {
        auto found = records_.find(key); require(found != records_.end(), "missing MP1 tensor: " + key); const auto &value = found->second;
        require(value.shape == spec.shape, "MP1 tensor shape mismatch: " + key);
        require(value.dtype == spec.dtype || (spec.bf16_source_allowed && value.dtype == "BF16")
                || (spec.int64_source_allowed && value.dtype == "I64"), "MP1 tensor dtype mismatch: " + key);
        if (key.rfind("mtp.", 0) == 0) ++auxiliary_; else required_.insert(key);
    }
    impl_->unchanged(); validated_ = true;
}

std::shared_ptr<backends::native::Tensor> Checkpoint::load(const std::string &key, core::Runtime &runtime) {
    require(validated_ && !failed_ && required_.count(key) && !loaded_.count(key), "unvalidated/duplicate/auxiliary checkpoint load: " + key);
    try {
        const auto &record = records_.at(key); auto data = impl_->read(record.offset, record.bytes);
        auto tensor = std::make_shared<backends::native::Tensor>(runtime, record.shape, dtype(record.dtype));
        tensor->upload(data.data(), data.size()); loaded_.insert(key); loaded_bytes_ += record.bytes; return tensor;
    } catch (...) { failed_ = true; throw; }
}
void Checkpoint::finish() {
    require(validated_ && !failed_ && required_ == loaded_, "incomplete/failed MP1 main-weight loading"); impl_->unchanged();
}
} // namespace llaisys::models::deepseek_v4

#pragma once

#include "src/backends/native/tensor.hpp"
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace native_fixture {
using Tensor = llaisys::backends::native::Tensor;
inline void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
inline void noPython() {
    check(!dlsym(RTLD_DEFAULT, "Py_IsInitialized"), "Python interpreter unexpectedly loaded");
    std::ifstream maps("/proc/self/maps"); check(maps.good(), "cannot inspect process mappings");
    std::string line;
    while (std::getline(maps, line)) check(line.find("libpython") == std::string::npos
        && line.find("libtorch_python") == std::string::npos, "Python runtime mapping unexpectedly loaded");
}
inline std::vector<unsigned char> bytes(const std::string &path, size_t count, uint64_t offset = 0) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    check(in.good() && offset <= static_cast<uint64_t>(in.tellg()) && count <= static_cast<uint64_t>(in.tellg()) - offset,
          "invalid tensor file/extent: " + path);
    in.seekg(offset); std::vector<unsigned char> value(count);
    if (count) in.read(reinterpret_cast<char *>(value.data()), count);
    check(in.good(), "truncated tensor file"); return value;
}
struct Value { std::shared_ptr<Tensor> tensor; std::string expected; };
inline std::pair<std::string, Value> read(std::istream &in, llaisys::core::Runtime &r) {
    std::string name, path, expected; unsigned code, bits, lanes; size_t rank, count; uint64_t offset;
    in >> name >> code >> bits >> lanes >> rank;
    check(in.good() && rank > 0 && rank <= 16 && code < 256 && bits < 256 && lanes < 65536, "invalid tensor descriptor");
    std::vector<int64_t> shape(rank); for (auto &dim : shape) in >> dim;
    in >> count >> offset >> std::quoted(path) >> std::quoted(expected);
    check(in.good() && count < (1ull << 31), "invalid tensor record");
    auto value = std::make_shared<Tensor>(r, shape, DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
    check(value->bytes() == count, "tensor shape/dtype byte mismatch");
    auto host = bytes(path, count, offset); value->upload(host.data(), host.size());
    return {name, {std::move(value), expected}};
}
inline void exact(Tensor &value, const std::string &path, const std::string &label) {
    auto wanted = bytes(path, value.bytes()); std::vector<unsigned char> actual(value.bytes());
    value.download(actual.data(), actual.size());
    if (wanted != actual) {
        size_t first = 0, changed = 0;
        for (size_t i = 0; i < actual.size(); ++i) if (wanted[i] != actual[i]) { if (!changed) first = i; ++changed; }
        throw std::runtime_error("non-exact " + label + ": changed_bytes=" + std::to_string(changed) + " first=" + std::to_string(first));
    }
}
} // namespace native_fixture

#pragma once
#include <map>
#include <mutex>
#include <string>
#include <cstdint>
namespace llaisys::ops {
inline std::mutex dispatch_mutex;
inline std::map<std::string, uint64_t> dispatch_counts;
inline void record_dispatch(const char *op, const char *impl) {
    std::lock_guard<std::mutex> lock(dispatch_mutex);
    ++dispatch_counts[std::string(op)+":"+impl];
}
}

#include <atomic>
#include <cstdint>
#ifdef ENABLE_NVIDIA_API
namespace llmops_integration {
extern std::atomic<uint64_t> dispatch_hits;
}
extern "C" __attribute__((visibility("default"))) uint64_t llmops_llaisys_dispatch_hits() {
    return llmops_integration::dispatch_hits.load();
}

#else
extern "C" __attribute__((visibility("default"))) uint64_t llmops_llaisys_dispatch_hits() { return 0; }
#endif
#include "../dispatch.hpp"
extern "C" __attribute__((visibility("default"))) const char *llaisysOperatorStats() {
    static thread_local std::string result;
    std::lock_guard<std::mutex> lock(llaisys::ops::dispatch_mutex);
    result="{";
    for(const auto &entry:llaisys::ops::dispatch_counts) {
        if(result.size()>1) result+=",";
        result+="\""+entry.first+"\":"+std::to_string(entry.second);
    }
    result+="}"; return result.c_str();
}

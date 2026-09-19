#pragma once
// Included by llaisys NVIDIA operator translation units; Tensor remains framework-owned.
#include "../dispatch.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <llmops/llmops.h>
#include <map>
#include <memory>
#include <stdexcept>
namespace llmops_integration {
inline std::atomic<uint64_t> dispatch_hits{0};
inline void check(llmopsStatus s) {
    if (s != LLMOPS_SUCCESS)
        throw std::runtime_error(std::string("llmops: ") + llmops_last_error());
}
inline void cuda_check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
inline int backend() {
    const char *s = std::getenv("LLAISYS_LLMOPS");
    if (!s || !std::strcmp(s, "native") || !std::strcmp(s, "auto"))
        return LLMOPS_NATIVE;
    if (!std::strcmp(s, "cublas"))
        return -1;
    if (!std::strcmp(s, "auto"))
        return LLMOPS_AUTO;
    if (!std::strcmp(s, "native"))
        return LLMOPS_NATIVE;
    if (!std::strcmp(s, "sm120"))
        return LLMOPS_SM120;
    if (!std::strcmp(s, "sm89"))
        return LLMOPS_SM89;
    throw std::runtime_error("LLAISYS_LLMOPS must be native (default), auto (native), sm89, sm120, or explicit cublas");
}
inline int dtype(llaisysDataType_t t) {
    switch (t) {
    case LLAISYS_DTYPE_F32:
        return LLMOPS_F32;
    case LLAISYS_DTYPE_F16:
        return LLMOPS_F16;
    case LLAISYS_DTYPE_BF16:
        return LLMOPS_BF16;
    default:
        return -1;
    }
}
struct Entry {
    llmopsPlan plan = nullptr;
    void *workspace = nullptr;
    size_t bytes = 0;
    ~Entry() {
        if (workspace)
            cudaFree(workspace);
        if (plan)
            llmops_plan_destroy(plan);
    }
};
struct State {
    llmopsContext context = nullptr;
    int device;
    std::map<std::array<int64_t, 12>, std::unique_ptr<Entry>> plans;
    explicit State(int d) : device(d) {
        check(llmops_context_create(d, cudaStreamPerThread, &context));
    }
    ~State() {
        int old = 0;
        cudaGetDevice(&old);
        cudaSetDevice(device);
        plans.clear();
        llmops_context_destroy(context);
        cudaSetDevice(old);
    }
};
inline State &state(int device) {
    static thread_local std::map<int, std::unique_ptr<State>> states;
    auto i = states.find(device);
    if (i == states.end())
        i = states.emplace(device, std::make_unique<State>(device)).first;
    return *i->second;
}
inline void same_device(llaisys::tensor_t a, int device) {
    if (a && (a->deviceType() != LLAISYS_DEVICE_NVIDIA || a->deviceId() != device))
        throw std::runtime_error("llmops adapter: tensors on different devices");
}
inline bool try_linear(llaisys::tensor_t out, llaisys::tensor_t x, llaisys::tensor_t w,
                       llaisys::tensor_t bias, llaisys::tensor_t residual = nullptr) {
    int mode = backend();
    if (mode < 0)
        return false;
    int dt = dtype(x->dtype()), od = dtype(out->dtype());
    if (dt < 0 || od < 0 || x->dtype() != w->dtype() || (od != dt && od != LLMOPS_F32) ||
        (bias && bias->dtype() != out->dtype()) || (residual && residual->dtype() != out->dtype()))
        throw std::runtime_error("llmops linear: unsupported dtype combination");
    if (x->ndim() != 2 || w->ndim() != 2 || out->ndim() != 2)
        throw std::runtime_error("llmops linear requires 2D tensors");
    if (x->strides()[1] != 1 || w->strides()[1] != 1 || out->strides()[1] != 1)
        throw std::runtime_error("llmops linear requires unit inner stride");
    int64_t m = x->shape()[0], k = x->shape()[1], n = w->shape()[0];
    if (w->shape()[1] != size_t(k) || out->shape()[0] != size_t(m) || out->shape()[1] != size_t(n))
        throw std::runtime_error("llmops linear shape mismatch");
    if (bias && (bias->ndim() != 1 || bias->shape()[0] != size_t(n) || bias->strides()[0] != 1))
        throw std::runtime_error("llmops bias must be contiguous [N]");
    if (residual && (residual->shape() != out->shape() || residual->strides() != out->strides()))
        throw std::runtime_error("llmops residual layout mismatch");
    int device = x->deviceId();
    for (auto t : {out, w, bias, residual})
        same_device(t, device);
    int current;
    cuda_check(cudaGetDevice(&current));
    if (current != device)
        throw std::runtime_error("llmops adapter device must be current");
    auto &s = state(device);
    std::array<int64_t, 12> key{
        m, n, k, x->strides()[0], w->strides()[0], out->strides()[0], dt, od, mode, 0, 0, 0};
    auto it = s.plans.find(key);
    if (it == s.plans.end()) {
        cudaStreamCaptureStatus capture;
        cuda_check(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
        if (capture != cudaStreamCaptureStatusNone)
            throw std::runtime_error("llmops: warm up this shape before graph capture");
        if (s.plans.size() >= 128)
            throw std::runtime_error("llmops plan cache capacity (128) exceeded");
        auto e = std::make_unique<Entry>();
        llmopsGemmDesc g{m,
                         n,
                         k,
                         key[3],
                         key[4],
                         key[5],
                         llmopsDtype(dt),
                         llmopsDtype(od),
                         LLMOPS_N,
                         LLMOPS_T,
                         LLMOPS_FP32,
                         llmopsBackend(mode)};
        check(llmops_gemm_plan(s.context, &g, &e->plan));
        e->bytes = llmops_workspace_size(e->plan);
        if (e->bytes)
            cuda_check(cudaMalloc(&e->workspace, e->bytes));
        it = s.plans.emplace(key, std::move(e)).first;
    }
    auto &e = *it->second;
    check(llmops_gemm_run(e.plan, x->data(), w->data(), nullptr, out->data(), 1, 0,
                          bias ? bias->data() : nullptr, residual ? residual->data() : nullptr,
                          e.workspace, e.bytes));
    ++dispatch_hits;
    llaisys::ops::record_dispatch("linear", llmops_plan_kernel(e.plan));
    return true;
}
} // namespace llmops_integration

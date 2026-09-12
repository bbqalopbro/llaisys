#include "native_kernel.hpp"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/c_backend_api.h>

#include <atomic>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <dlfcn.h>

namespace llaisys::backends::tilelang {
namespace {
std::atomic<uint64_t> destruction_errors{0};

struct CloseLibrary { void operator()(void *p) const { if (p) ::dlclose(p); } };
using Library = std::unique_ptr<void, CloseLibrary>;
Library runtimeSymbols(const void *symbol) {
    Dl_info info{};
    if (!::dladdr(symbol, &info) || !info.dli_fname)
        throw std::runtime_error("cannot resolve linked TileLang runtime library");
    // Python extension modules normally load RTLD_LOCAL. Exported upstream
    // kernels have undefined TVM helpers (no DT_NEEDED), unlike the standalone
    // executable where its dependencies are global. Promote ONLY the already
    // linked library identified by its real symbol; do not guess search paths
    // or require importing Python TileLang to populate global symbols.
    auto handle = ::dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    if (!handle) throw std::runtime_error(std::string("cannot expose linked TileLang runtime: ") + ::dlerror());
    return Library(handle);
}

class StreamScope {
public:
    explicit StreamScope(core::Runtime &runtime) : device_(runtime.deviceId()) {
        if (TVMFFIEnvSetStream(kDLCUDA, device_, reinterpret_cast<void *>(runtime.stream()), &previous_))
            throw std::runtime_error("cannot select the llaisys stream in TVM-FFI");
    }
    ~StreamScope() {
        if (TVMFFIEnvSetStream(kDLCUDA, device_, previous_, nullptr)) {
            ++destruction_errors;
            std::fputs("llaisys TileLang: cannot restore TVM-FFI stream\n", stderr);
        }
    }
private:
    int device_;
    TVMFFIStreamHandle previous_ = nullptr;
};
}

class NativeKernel::Impl {
public:
    // Retain runtime libraries until both the exported function and module
    // have been released. close() synchronizes outstanding launches first.
    Library tvm_symbols, ffi_symbols;
    tvm::ffi::Module module;
    tvm::ffi::Function function;
    Impl(const std::string &path, const std::string &entry)
        : tvm_symbols(runtimeSymbols(reinterpret_cast<const void *>(&TVMBackendGetFuncFromEnv))),
          ffi_symbols(runtimeSymbols(reinterpret_cast<const void *>(&TVMFFIFunctionCall))),
          module(tvm::ffi::Module::LoadFromFile(path)), function([&] {
            auto found = module->GetFunction(entry, true);
            if (!found.has_value()) throw std::invalid_argument("missing TileLang entry: " + entry);
            return *found;
        }()) {}
};

NativeKernel::NativeKernel(core::Runtime &runtime, const std::string &library,
                         native::KernelIdentity identity, const std::string &entry)
    : native::Kernel(runtime, std::move(identity)), thread_(std::this_thread::get_id()) {
    if (identity_.backend != "tilelang")
        throw std::invalid_argument("TileLang implementation must report tilelang backend identity");
    if (identity_.version != "0.1.8")
        throw std::invalid_argument("this native TileLang ABI/FP4 bridge is validated only for version 0.1.8");
    if (runtime.deviceType() != LLAISYS_DEVICE_NVIDIA || !runtime.isActive())
        throw std::invalid_argument("native TileLang kernel requires an active NVIDIA runtime");
    impl_ = std::make_unique<Impl>(library, entry);
}

void NativeKernel::call(const std::vector<NativeTensor *> &arguments) {
    if (std::this_thread::get_id() != thread_ || !runtime_.isActive() || !impl_)
        throw std::runtime_error("native kernel is closed or used outside its owning active runtime/thread");
    ++calls_;
    try {
        if (arguments.size() > static_cast<size_t>(INT32_MAX)) throw std::invalid_argument("too many kernel arguments");
        std::vector<DLTensor> descriptors;
        descriptors.reserve(arguments.size());
        std::vector<std::vector<int64_t>> packed_shapes(arguments.size()), packed_strides(arguments.size());
        for (auto *tensor : arguments) {
            if (!tensor || &tensor->runtime() != &runtime_) throw std::invalid_argument("foreign or null native tensor");
            descriptors.push_back(tensor->view());
            auto &view = descriptors.back();
            // Exported kernels may require byte_offset=0. A native contiguous
            // slice carries its offset separately; normalize only this borrowed
            // descriptor, as Torch's data_ptr-based launch adapter does.
            if (view.byte_offset) {
                view.data = static_cast<char *>(view.data) + view.byte_offset;
                view.byte_offset = 0;
            }
            if (view.dtype.bits < 8) {
                // TileLang 0.1.8 ArgBinder interprets sub-byte parameters as
                // byte-packed Torch storage, unlike logical DLPack shapes.
                // Adapt ONLY the call descriptor: FP4 data is not converted.
                if (view.dtype.code != kDLFloat4_e2m1fn || view.dtype.bits != 4
                    || view.dtype.lanes != 1 || view.shape[view.ndim - 1] % 2 || !tensor->isContiguous())
                    throw std::invalid_argument("TileLang packed bridge requires even-K scalar FP4");
                const size_t i = descriptors.size() - 1;
                auto &shape = packed_shapes[i];
                auto &strides = packed_strides[i];
                shape.assign(view.shape, view.shape + view.ndim);
                shape.back() /= 2;
                strides.resize(shape.size());
                int64_t stride = 1;
                for (size_t axis = shape.size(); axis-- > 0;) {
                    strides[axis] = stride;
                    stride *= shape[axis];
                }
                view.shape = shape.data();
                view.strides = strides.data();
                view.dtype = {kDLUInt, 8, 1};
            }
        }
        std::vector<tvm::ffi::AnyView> args;
        args.reserve(descriptors.size());
        // AnyView borrows TensorView's INTERNAL DLTensor, not the constructor
        // argument. A temporary TensorView would leave every arg dangling.
        std::vector<tvm::ffi::TensorView> views;
        views.reserve(descriptors.size());
        for (auto &tensor : descriptors) views.emplace_back(&tensor);
        for (const auto &view : views) args.emplace_back(view);
        StreamScope stream(runtime_);
        tvm::ffi::Any result;
        impl_->function.CallPacked(args.data(), static_cast<int32_t>(args.size()), &result);
    } catch (...) {
        ++failures_;
        throw;
    }
}

void NativeKernel::close() {
    if (!impl_) return;
    if (std::this_thread::get_id() != thread_ || !runtime_.isActive())
        throw std::runtime_error("close native kernel on its owning active runtime/thread");
    // A kernel library cannot be unloaded while queued launches still use it.
    runtime_.synchronize();
    impl_.reset();
}

NativeKernel::~NativeKernel() {
    try { close(); }
    catch (const std::exception &error) {
        ++destruction_errors;
        std::fprintf(stderr, "llaisys TileLang teardown failed: %s\n", error.what());
        (void)impl_.release(); // lost context/thread: do not unload in-flight code
    }
}

uint64_t NativeKernel::destructionErrors() { return destruction_errors.load(); }
} // namespace llaisys::backends::tilelang

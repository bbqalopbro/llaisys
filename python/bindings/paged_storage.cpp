#include "src/models/deepseek_v4/cache_layout.hpp"
#include "src/core/cache/paged_cache_storage.hpp"
#include "llaisys/runtime.h"

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef ENABLE_NVIDIA_API
#include <cuda_runtime_api.h>
#endif

#include <array>
#include <atomic>
#include <cstdio>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;
using llaisys::models::deepseek_v4::InterleavedLayerLayout;
using llaisys::models::deepseek_v4::checkedProduct;

namespace {
std::atomic<size_t> live_bytes{0}, live_layers{0}, destruction_errors{0};

// This guard does not switch llaisys's thread-local model context. Allocation
// and deletion may be initiated by different threads or DLPack consumers.
class DeviceGuard {
public:
    DeviceGuard(bool cuda, int device) : cuda_(cuda) {
#ifdef ENABLE_NVIDIA_API
        if (cuda_) {
            check(cudaGetDevice(&previous_));
            check(cudaSetDevice(device));
        }
#else
        (void)device;
        if (cuda_) throw std::runtime_error("CUDA storage is not compiled in this module");
#endif
    }
    ~DeviceGuard() {
#ifdef ENABLE_NVIDIA_API
        if (cuda_ && cudaSetDevice(previous_) != cudaSuccess) {
            ++destruction_errors;
            std::fputs("V4 cache: failed to restore CUDA device\n", stderr);
        }
#endif
    }
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
#ifdef ENABLE_NVIDIA_API
    static void check(cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
#endif
private:
    bool cuda_;
    int previous_ = 0;
};

struct LayerStorage {
    std::shared_ptr<const InterleavedLayerLayout> layout;
    std::unique_ptr<llaisys::core::PagedCacheStorage> storage;
    const LlaisysRuntimeAPI *api;
    bool cuda;
    int device;
    uintptr_t stream;
    size_t blocks, bytes = 0;

    LayerStorage(std::shared_ptr<const InterleavedLayerLayout> selected, size_t count,
                 bool use_cuda, int device_id, uintptr_t owner_stream)
        : layout(std::move(selected)), api(llaisysGetRuntimeAPI(use_cuda ? LLAISYS_DEVICE_NVIDIA : LLAISYS_DEVICE_CPU)),
          cuda(use_cuda), device(device_id), stream(owner_stream), blocks(count) {
        DeviceGuard guard(cuda, device);
#ifdef ENABLE_NVIDIA_API
        if (cuda) {
            unsigned int flags = 0;
            DeviceGuard::check(cudaStreamGetFlags(reinterpret_cast<cudaStream_t>(stream), &flags));
        }
#endif
        for (size_t component = 0; component < layout->numComponents(); ++component) {
            const size_t part = checkedProduct(count, layout->layerBytes(component, 0));
            if (bytes > static_cast<size_t>(INT64_MAX) - part) throw std::overflow_error("V4 cache bytes overflow");
            bytes += part;
        }
        storage = std::make_unique<llaisys::core::PagedCacheStorage>(count, layout, api);
        live_bytes += bytes;
        ++live_layers;
    }

    ~LayerStorage() {
        // No Python references or GIL dependence in the DLPack owner. At pool
        // teardown, synchronize the owning stream before releasing raw storage.
        // This is not a per-token synchronization or a cross-stream contract.
        try {
            DeviceGuard guard(cuda, device);
#ifdef ENABLE_NVIDIA_API
            if (cuda) DeviceGuard::check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
#endif
            storage->release();
            storage.reset();
            live_bytes -= bytes;
            --live_layers;
        } catch (const std::exception &error) {
            // A lost CUDA context cannot safely free the allocation. Do not
            // throw from a consumer's deleter or claim memory was recovered.
            ++destruction_errors;
            std::fprintf(stderr, "V4 cache destruction failed: %s\n", error.what());
            (void)storage.release();
        }
    }
};

// Each export has its own metadata holder. The allocation remains owned by
// shared_ptr<LayerStorage>; destroying/consuming a capsule cannot free live views.
template <typename Managed>
struct ExportedTensor {
    Managed managed{};
    std::shared_ptr<LayerStorage> owner;
    std::array<int64_t, 3> shape, strides;

    ExportedTensor(std::shared_ptr<LayerStorage> selected, size_t component) : owner(std::move(selected)) {
        shape = {1, static_cast<int64_t>(checkedProduct(owner->blocks, owner->layout->slots(component))),
                 static_cast<int64_t>(owner->layout->width(component))};
        strides = {static_cast<int64_t>(checkedProduct(shape[1], shape[2])), shape[2], 1};
        managed.dl_tensor = {owner->storage->componentPool(component),
                            {owner->cuda ? kDLCUDA : kDLCPU, owner->device}, 3,
                            {static_cast<uint8_t>(kDLBfloat), 16, 1}, shape.data(), strides.data(), 0};
        managed.manager_ctx = this;
        managed.deleter = [](Managed *tensor) { delete static_cast<ExportedTensor *>(tensor->manager_ctx); };
    }
};

template <typename Managed>
py::capsule makeCapsule(std::unique_ptr<ExportedTensor<Managed>> holder, const char *name) {
    py::capsule result(&holder->managed, name, [](PyObject *capsule) {
        if (PyCapsule_IsValid(capsule, "dltensor")) {
            auto *tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
            tensor->deleter(tensor);
        } else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
            auto *tensor = static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(capsule, "dltensor_versioned"));
            tensor->deleter(tensor);
        }
        // A consumed capsule is named used_dltensor[_versioned]; its consumer
        // owns the holder and will invoke the deleter exactly once.
    });
    holder.release();
    return result;
}

struct CacheView {
    std::shared_ptr<LayerStorage> owner;
    size_t component;

    py::tuple device() const { return py::make_tuple(static_cast<int>(owner->cuda ? kDLCUDA : kDLCPU), owner->device); }
    py::capsule exportTensor(py::object stream, py::object max_version, py::object dl_device, py::object copy) const {
        if (!copy.is_none() && !copy.is(py::bool_(false))) throw py::buffer_error("cache views never copy");
        if (!dl_device.is_none() && !dl_device.equal(device())) throw py::buffer_error("cross-device cache export is unsupported");
        if (owner->cuda) {
            // DLPack uses 1 for the legacy default stream (CUDA handle 0).
            const int64_t requested = stream.is_none() ? 1 : stream.cast<int64_t>();
            const uintptr_t normalized = requested == 1 ? 0 : static_cast<uintptr_t>(requested);
            if (requested <= 0 || normalized != owner->stream)
                throw py::buffer_error("cache view requires its owning CUDA stream");
        } else if (!stream.is_none()) {
            throw py::buffer_error("CPU cache export does not accept a stream");
        }
        if (!max_version.is_none()) {
            const auto version = max_version.cast<std::pair<int, int>>();
            if (version.first < 0 || version.second < 0) throw py::value_error("invalid DLPack max_version");
            if (version.first >= 1) {
                auto holder = std::make_unique<ExportedTensor<DLManagedTensorVersioned>>(owner, component);
                holder->managed.version = {1, 0};
                holder->managed.flags = 0; // Mutable view; not a copy.
                return makeCapsule(std::move(holder), "dltensor_versioned");
            }
        }
        return makeCapsule(std::make_unique<ExportedTensor<DLManagedTensor>>(owner, component), "dltensor");
    }
};

class PagedStorage {
public:
    PagedStorage(size_t blocks, size_t block_size, size_t latent_dim, size_t index_dim,
                 const std::vector<size_t> &ratios, const std::string &device_type, int device, uintptr_t stream) {
        if (device_type != "cpu" && device_type != "cuda") throw std::invalid_argument("cache storage supports CPU or CUDA");
        const bool cuda = device_type == "cuda";
        if (!blocks || blocks > INT32_MAX || ratios.empty() || device < 0 || (!cuda && (device != 0 || stream)))
            throw std::invalid_argument("invalid V4 cache blocks, layers, device or stream");
        // Validate every layer/size before performing any allocation.
        std::vector<std::shared_ptr<InterleavedLayerLayout>> layouts;
        for (auto ratio : ratios) {
            auto layout = std::make_shared<InterleavedLayerLayout>(block_size, latent_dim, index_dim, ratio);
            if (checkedProduct(blocks, layout->slots(0)) >= INT32_MAX)
                throw std::invalid_argument("V4 cache physical slots must fit int32");
            for (size_t c = 0; c < layout->numComponents(); ++c) checkedProduct(blocks, layout->layerBytes(c, 0));
            layouts.push_back(std::move(layout));
        }
        for (auto &layout : layouts) layers_.push_back(std::make_shared<LayerStorage>(layout, blocks, cuda, device, stream));
    }

    CacheView view(size_t layer, const std::string &component) const {
        auto owner = layers_.at(layer);
        const int index = owner->storage->componentIndex(component);
        if (index < 0) throw std::invalid_argument("cache component does not exist in this layer");
        return {std::move(owner), static_cast<size_t>(index)};
    }
    py::dict report() const {
        py::list layers;
        size_t bytes = 0;
        for (const auto &owner : layers_) {
            py::list components;
            bytes += owner->bytes;
            for (size_t c = 0; c < owner->layout->numComponents(); ++c) {
                components.append(py::dict("name"_a=owner->layout->componentName(c),
                    "data_ptr"_a=reinterpret_cast<uintptr_t>(owner->storage->componentPool(c)),
                    "block_stride_bytes"_a=owner->storage->componentBlockStride(c),
                    "slots_per_block"_a=owner->layout->slots(c), "width"_a=owner->layout->width(c)));
            }
            layers.append(py::dict("compression_ratio"_a=owner->layout->ratio(), "components"_a=components));
        }
        return py::dict("allocator"_a="existing-cpp-paged-cache-storage", "payload_bytes"_a=bytes,
                        "layers"_a=layers, "zero_copy"_a=true, "fallback"_a=false);
    }
private:
    std::vector<std::shared_ptr<LayerStorage>> layers_;
};
} // namespace

void bindPagedStorage(py::module_ &module) {
    using namespace pybind11::literals;
    py::class_<CacheView>(module, "V4CacheView")
        .def("__dlpack_device__", &CacheView::device)
        .def("__dlpack__", &CacheView::exportTensor, py::arg("stream")=py::none(),
             py::kw_only(), py::arg("max_version")=py::none(), py::arg("dl_device")=py::none(), py::arg("copy")=py::none());
    py::class_<PagedStorage>(module, "V4PagedStorage")
        .def(py::init<size_t, size_t, size_t, size_t, const std::vector<size_t> &, const std::string &, int, uintptr_t>(),
             py::arg("num_blocks"), py::arg("block_size"), py::arg("latent_dim"), py::arg("index_dim"),
             py::arg("compression_ratios"), py::arg("device_type")="cpu", py::arg("device_id")=0, py::arg("stream")=0)
        .def("view", &PagedStorage::view, py::arg("layer"), py::arg("component")="latent")
        .def("report", &PagedStorage::report);
    module.def("v4_storage_counters", [] {
        return py::dict("live_bytes"_a=live_bytes.load(), "live_layers"_a=live_layers.load(),
                        "destruction_errors"_a=destruction_errors.load());
    });
}

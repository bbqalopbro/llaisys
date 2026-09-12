// Native ownership/descriptor contract only. Model numerics and the real
// TileLang consumer are checked by the separate native fixture verifier.
// CUDA work must run inside a Slurm GPU allocation, never on the login node.
#include <cuda_runtime_api.h>

#include "src/backends/native/paged_storage.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/cache_layout.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using llaisys::backends::native::PagedStorage;
using llaisys::backends::native::Tensor;
using llaisys::core::Runtime;
using llaisys::models::deepseek_v4::InterleavedLayerLayout;
constexpr DLDataType bf16{kDLBfloat, 16, 1};
uint64_t checks = 0, negative_checks = 0, components_checked = 0;

void check(bool valid, const char *message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}

void cudaCheck(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename Exception, typename Function>
void rejects(Function function, const char *message) {
    try { function(); }
    catch (const Exception &) { ++checks; ++negative_checks; return; }
    throw std::runtime_error(message);
}

std::vector<int64_t> shape(const InterleavedLayerLayout &layout, size_t blocks, size_t component) {
    return {1, static_cast<int64_t>(blocks * layout.slots(component)),
            static_cast<int64_t>(layout.width(component))};
}

std::vector<uint16_t> download(const Tensor &tensor) {
    std::vector<uint16_t> output(tensor.bytes() / sizeof(uint16_t));
    tensor.download(output.data(), tensor.bytes());
    return output;
}

// Arbitrary byte layouts exercise generic preallocation checks without asking
// CUDA to reserve the enormous sizes encoded in the rejection cases.
class ByteLayout final : public llaisys::core::CacheLayout {
public:
    ByteLayout(std::vector<std::vector<size_t>> sizes, size_t layers)
        : sizes_(std::move(sizes)), layers_(layers) {
        for (size_t c = 0; c < sizes_.size(); ++c) names_.push_back("component" + std::to_string(c));
    }
    const char *name() const override { return "native-paged-rejection-fixture"; }
    size_t numLayers() const override { return layers_; }
    size_t blockSize() const override { return 128; }
    size_t numComponents() const override { return sizes_.size(); }
    const std::string &componentName(size_t component) const override { return names_.at(component); }
    size_t layerBytes(size_t component, size_t layer) const override { return sizes_.at(component).at(layer); }
private:
    std::vector<std::vector<size_t>> sizes_;
    size_t layers_;
    std::vector<std::string> names_;
};

void layoutAndAliases(Runtime &runtime) {
    for (size_t ratio : {size_t{0}, size_t{4}, size_t{128}}) {
        constexpr size_t blocks = 3;
        auto layout = std::make_shared<InterleavedLayerLayout>(128, 32, 16, ratio);
        auto pool = PagedStorage::create(runtime, blocks, layout);
        check(pool->numBlocks() == blocks && pool->layoutHandle() == layout && !pool->closed(),
              "native paged owner metadata mismatch");
        size_t total = 0;
        for (size_t c = 0; c < layout->numComponents(); ++c) total += blocks * layout->layerBytes(c, 0);
        check(PagedStorage::liveBytes() == total, "component payload accounting mismatch");
        for (size_t c = 0; c < layout->numComponents(); ++c) {
            ++components_checked;
            const int64_t rows = static_cast<int64_t>(layout->slots(c));
            const int64_t width = static_cast<int64_t>(layout->width(c));
            const int64_t block_elements = rows * width;
            const size_t component_bytes = blocks * layout->layerBytes(c, 0);
            check(pool->componentBytes(c) == component_bytes
                  && pool->componentBlockStride(c) == layout->layerBytes(c, 0)
                  && pool->componentLayerOffset(c, 0) == 0, "component layout byte/stride mismatch");
            auto first = pool->componentTensor(c, shape(*layout, blocks, c), bf16);
            auto second = pool->componentTensor(c, shape(*layout, blocks, c), bf16);
            const auto descriptor = first->view();
            check(descriptor.data && descriptor.byte_offset == 0 && descriptor.ndim == 3
                  && descriptor.device.device_type == kDLCUDA && descriptor.device.device_id == runtime.deviceId()
                  && descriptor.dtype.code == kDLBfloat && descriptor.dtype.bits == 16 && descriptor.dtype.lanes == 1
                  && descriptor.strides[0] == static_cast<int64_t>(blocks) * block_elements
                  && descriptor.strides[1] == width && descriptor.strides[2] == 1
                  && first->isContiguous() && first->bytes() == component_bytes,
                  "invalid full-component BF16 DLTensor descriptor");
            check(second->view().data == descriptor.data && first->overlaps(*second) && second->overlaps(*first)
                  && pool->activeExports() == 2, "independent exports must alias the same allocation");
            if (layout->numComponents() > 1) {
                const size_t other_component = 1 - c;
                auto other = pool->componentTensor(other_component, shape(*layout, blocks, other_component), bf16);
                check(!first->overlaps(*other) && pool->activeExports() == 3, "different components alias");
            }
            auto first_block = first->asStrided({rows, width}, {width, 1});
            auto next_block = first->asStrided({rows, width}, {width, 1}, block_elements);
            auto empty = first->asStrided({0, width}, {width, 1});
            check(first_block->overlaps(*first) && !first_block->overlaps(*next_block)
                  && !next_block->overlaps(*first_block) && !empty->overlaps(*first)
                  && pool->activeExports() == 2, "subviews must retain exports without inventing overlapping blocks");
            check(next_block->view().data == descriptor.data
                  && next_block->view().byte_offset == layout->layerBytes(c, 0), "block view offset mismatch");
            std::vector<uint16_t> expected(component_bytes / 2);
            for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<uint16_t>(0x3f00 + i % 97);
            first->upload(expected.data(), component_bytes);
            check(download(*second) == expected, "independent component export did not see uploaded bytes");
            rejects<std::runtime_error>([&] { pool->close(); }, "close accepted live exports");
            check(!pool->closed() && PagedStorage::liveBytes() == total, "rejected close changed storage state");
            std::vector<uint16_t> replacement(static_cast<size_t>(block_elements), static_cast<uint16_t>(0x4000 + c));
            next_block->upload(replacement.data(), next_block->bytes());
            std::copy(replacement.begin(), replacement.end(), expected.begin() + block_elements);
            check(download(*second) == expected, "storage was not usable or subview wrote wrong bytes after rejected close");
            empty.reset(); first.reset(); second.reset();
            check(pool->activeExports() == 1, "subviews did not retain the original export token");
            rejects<std::runtime_error>([&] { pool->close(); }, "close ignored live descendant views");
            check(download(*next_block) == replacement, "descendant view lost its storage owner");
            first_block.reset(); next_block.reset();
            check(pool->activeExports() == 0, "export token was not released after final descendant");
        }
        pool->close(); pool->close();
        check(pool->closed() && pool->activeExports() == 0 && PagedStorage::liveBytes() == 0,
              "explicit/repeated close did not recover all payload bytes");
        rejects<std::runtime_error>([&] { pool->componentTensor(0, shape(*layout, blocks, 0), bf16); },
                                    "closed storage accepted a new export");
        rejects<std::runtime_error>([&] { pool->componentBytes(0); }, "closed storage exposed component metadata");
    }
}

void retainedOwner(Runtime &runtime) {
    auto layout = std::make_shared<InterleavedLayerLayout>(128, 32, 16, 4);
    auto pool = PagedStorage::create(runtime, 1, layout);
    std::weak_ptr<PagedStorage> weak = pool;
    auto original = pool->componentTensor(0, shape(*layout, 1, 0), bf16);
    auto row = original->asStrided({32}, {1}, 32);
    pool.reset(); original.reset();
    check(!weak.expired() && PagedStorage::liveBytes() > 0, "owner was freed while a subview remained alive");
    std::vector<uint16_t> payload(32, 0x4040);
    row->upload(payload.data(), row->bytes());
    check(download(*row) == payload, "surviving subview cannot access its retained pool");
    row.reset();
    check(weak.expired() && PagedStorage::liveBytes() == 0, "final view did not release the retained pool");
}

void invalidArguments(Runtime &runtime) {
    auto layout = std::make_shared<InterleavedLayerLayout>(128, 32, 16, 4);
    rejects<std::invalid_argument>([&] { PagedStorage::create(runtime, 0, layout); }, "zero blocks accepted");
    rejects<std::invalid_argument>([&] { PagedStorage::create(runtime, 1, nullptr); }, "null layout accepted");
    rejects<std::invalid_argument>([&] {
        PagedStorage::create(runtime, 1, std::make_shared<ByteLayout>(std::vector<std::vector<size_t>>{}, 1));
    }, "empty component list accepted");
    rejects<std::invalid_argument>([&] {
        PagedStorage::create(runtime, 1, std::make_shared<ByteLayout>(std::vector<std::vector<size_t>>{{}}, 0));
    }, "empty layer list accepted");
    const size_t maximum = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    for (const auto &sizes : std::vector<std::vector<std::vector<size_t>>>{{{0}}, {{maximum, 1}}, {{maximum}, {1}}}) {
        rejects<std::overflow_error>([&] {
            PagedStorage::create(runtime, 1, std::make_shared<ByteLayout>(sizes, sizes.front().size()));
        }, "zero or overflowing component/layer/total bytes accepted");
        check(PagedStorage::liveBytes() == 0, "preallocation rejection leaked payload storage");
    }
    rejects<std::overflow_error>([&] {
        PagedStorage::create(runtime, 2,
            std::make_shared<ByteLayout>(std::vector<std::vector<size_t>>{{maximum / 2 + 1}}, 1));
    }, "block product overflow accepted");
    check(PagedStorage::liveBytes() == 0, "invalid layouts changed allocation accounting");
    auto pool = PagedStorage::create(runtime, 1, layout);
    const auto bytes = PagedStorage::liveBytes();
    const int64_t elements = static_cast<int64_t>(pool->componentBytes(0) / 2);
    for (const auto &invalid_shape : std::vector<std::vector<int64_t>>{
             {}, {-1}, {0}, {elements - 1}, {elements + 1}, {INT64_MAX, 2}, std::vector<int64_t>(17, 1)}) {
        rejects<std::invalid_argument>([&] { pool->componentTensor(0, invalid_shape, bf16); }, "invalid export shape accepted");
        check(pool->activeExports() == 0 && PagedStorage::liveBytes() == bytes, "failed export retained a token or storage");
    }
    rejects<std::invalid_argument>([&] { pool->componentTensor(0, {elements}, {kDLBfloat, 0, 1}); }, "invalid dtype accepted");
    rejects<std::out_of_range>([&] { pool->componentTensor(2, {elements}, bf16); }, "invalid component export accepted");
    rejects<std::out_of_range>([&] { pool->componentBytes(2); }, "invalid component bytes index accepted");
    rejects<std::out_of_range>([&] { pool->componentBlockStride(2); }, "invalid component stride index accepted");
    rejects<std::out_of_range>([&] { pool->componentLayerOffset(0, 1); }, "invalid layer offset accepted");
    auto view = pool->componentTensor(0, {elements}, bf16);
    rejects<std::invalid_argument>([&] { view->asStrided({elements + 1}, {1}); }, "out-of-bounds subview accepted");
    rejects<std::invalid_argument>([&] { view->asStrided({2}, {-1}); }, "negative subview stride accepted");
    rejects<std::invalid_argument>([&] { view->asStrided({1}, {1}, -1); }, "negative subview offset accepted");
    check(pool->activeExports() == 1 && PagedStorage::liveBytes() == bytes, "invalid view changed owner state");
    view.reset(); pool->close();
    check(PagedStorage::liveBytes() == 0, "rejection test did not recover payload storage");
}

void runtimeAndThreads(Runtime &runtime) {
    auto layout = std::make_shared<InterleavedLayerLayout>(128, 32, 16, 0);
    auto pool = PagedStorage::create(runtime, 1, layout);
    auto view = pool->componentTensor(0, shape(*layout, 1, 0), bf16);
    const auto bytes = PagedStorage::liveBytes();
    std::exception_ptr foreign_error;
    std::thread foreign([&] {
        try {
            rejects<std::runtime_error>([&] { PagedStorage::create(runtime, 1, layout); }, "foreign runtime accepted during construction");
            rejects<std::invalid_argument>([&] { Tensor invalid(runtime, {4}, bf16); }, "foreign runtime accepted by Tensor constructor");
            rejects<std::runtime_error>([&] { pool->componentTensor(0, shape(*layout, 1, 0), bf16); }, "foreign-thread export accepted");
            rejects<std::runtime_error>([&] { pool->componentBytes(0); }, "foreign-thread component access accepted");
            rejects<std::runtime_error>([&] { pool->close(); }, "foreign-thread close accepted");
            rejects<std::runtime_error>([&] { view->view(); }, "foreign-thread tensor access accepted");
            rejects<std::runtime_error>([&] { view->asStrided({4}, {1}); }, "foreign-thread subview creation accepted");
        } catch (...) { foreign_error = std::current_exception(); }
    });
    foreign.join();
    if (foreign_error) std::rethrow_exception(foreign_error);
    check(pool->activeExports() == 1 && PagedStorage::liveBytes() == bytes, "foreign-thread rejection changed owner state");
    auto &context = llaisys::core::context();
    context.setDevice(LLAISYS_DEVICE_CPU, 0);
    try {
        auto &cpu_runtime = context.runtime();
        check(!runtime.isActive() && cpu_runtime.deviceType() == LLAISYS_DEVICE_CPU, "CPU/inactive test did not change runtime");
        rejects<std::runtime_error>([&] { PagedStorage::create(cpu_runtime, 1, layout); }, "CPU runtime accepted by native paged storage");
        rejects<std::runtime_error>([&] { PagedStorage::create(runtime, 1, layout); }, "inactive runtime accepted during construction");
        rejects<std::invalid_argument>([&] { Tensor invalid(cpu_runtime, {4}, bf16); }, "CPU runtime accepted by Tensor constructor");
        rejects<std::runtime_error>([&] { pool->componentTensor(0, shape(*layout, 1, 0), bf16); }, "inactive runtime export accepted");
        rejects<std::runtime_error>([&] { pool->close(); }, "inactive runtime close accepted");
        rejects<std::runtime_error>([&] { view->view(); }, "inactive runtime tensor access accepted");
    } catch (...) {
        context.setDevice(LLAISYS_DEVICE_NVIDIA, runtime.deviceId());
        throw;
    }
    context.setDevice(LLAISYS_DEVICE_NVIDIA, runtime.deviceId());
    check(&context.runtime() == &runtime && runtime.isActive() && pool->activeExports() == 1,
          "original runtime/export did not survive rejected inactive operations");
    std::vector<uint16_t> payload(view->bytes() / 2, 0x4080);
    view->upload(payload.data(), view->bytes());
    check(download(*view) == payload, "runtime restoration did not preserve storage access");
    view.reset(); pool->close();
    check(PagedStorage::liveBytes() == 0, "thread/runtime test leaked payload storage");
}

struct AsyncResources {
    void *host = nullptr;
    cudaEvent_t event = nullptr;
    ~AsyncResources() {
        if (event) (void)cudaEventDestroy(event);
        if (host) (void)cudaFreeHost(host);
    }
    void close() {
        cudaCheck(cudaEventDestroy(event), "destroy async completion event"); event = nullptr;
        cudaCheck(cudaFreeHost(host), "free async pinned buffer"); host = nullptr;
    }
};

void queuedWork(Runtime &runtime) {
    for (bool explicit_close : {true, false}) {
        auto layout = std::make_shared<InterleavedLayerLayout>(128, 32, 16, 4);
        auto pool = PagedStorage::create(runtime, 16, layout);
        std::weak_ptr<PagedStorage> weak = pool;
        auto tensor = pool->componentTensor(0, shape(*layout, 16, 0), bf16);
        const auto descriptor = tensor->view();
        const size_t bytes = tensor->bytes();
        AsyncResources resources;
        cudaCheck(cudaMallocHost(&resources.host, bytes), "allocate pinned buffer");
        cudaCheck(cudaEventCreateWithFlags(&resources.event, cudaEventDisableTiming), "create completion event");
        const auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
        cudaCheck(cudaMemsetAsync(descriptor.data, 0x5a, bytes, stream), "queue paged payload fill");
        cudaCheck(cudaMemcpyAsync(resources.host, descriptor.data, bytes, cudaMemcpyDeviceToHost, stream), "queue paged payload read");
        cudaCheck(cudaEventRecord(resources.event, stream), "record paged payload completion");
        // No event wait, stream sync, Tensor download, or pageable copy occurs
        // here. Both explicit close and final owner destruction must wait for
        // all work that consumed the borrowed component pointer on this stream.
        if (explicit_close) { tensor.reset(); pool->close(); pool.reset(); }
        else { pool.reset(); tensor.reset(); }
        check(cudaEventQuery(resources.event) == cudaSuccess, "teardown returned before queued cache work completed");
        const auto *first = static_cast<const unsigned char *>(resources.host);
        check(std::all_of(first, first + bytes, [](unsigned char value) { return value == 0x5a; }),
              "queued cache read observed corrupted/freed storage");
        check(weak.expired() && PagedStorage::liveBytes() == 0, "async teardown retained the payload owner");
        resources.close();
    }
}
} // namespace

int main() {
    try {
        const char *job = std::getenv("SLURM_JOB_ID");
        check(job && *job, "native paged GPU contract test requires SLURM_JOB_ID");
        check(PagedStorage::liveBytes() == 0 && PagedStorage::destructionErrors() == 0, "unexpected initial paged counters");
        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &runtime = llaisys::core::context().runtime();
        check(runtime.deviceType() == LLAISYS_DEVICE_NVIDIA && runtime.isActive() && runtime.stream(),
              "test requires active NVIDIA runtime with its own stream");
        layoutAndAliases(runtime);
        retainedOwner(runtime);
        invalidArguments(runtime);
        runtimeAndThreads(runtime);
        queuedWork(runtime);
        runtime.synchronize();
        cudaCheck(cudaGetLastError(), "final CUDA error check");
        check(PagedStorage::liveBytes() == 0 && PagedStorage::destructionErrors() == 0, "native paged teardown counters are not clean");
        std::cout << "{\"all_passed\":true,\"checks\":" << checks << ",\"negative_checks\":" << negative_checks
                  << ",\"components_checked\":" << components_checked << ",\"async_teardown_cases\":2"
                  << ",\"live_bytes\":0,\"destruction_errors\":0,\"fallback\":false"
                  << ",\"foreign_final_destruction_tested\":false,\"multi_device_tested\":false}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "native paged storage contract failed: " << error.what() << '\n';
        return 2;
    }
}

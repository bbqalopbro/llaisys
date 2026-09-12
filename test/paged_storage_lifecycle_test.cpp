#include "src/models/deepseek_v4/cache_layout.hpp"
#include "src/core/cache/paged_cache_storage.hpp"

#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
std::set<void *> allocations;
int remaining_allocations = -1;
bool fail_next_free = false;

void require(bool valid) { if (!valid) throw std::runtime_error("cache storage lifecycle assertion failed"); }
void *allocate(size_t bytes) {
    if (remaining_allocations == 0) return nullptr;
    if (remaining_allocations > 0) --remaining_allocations;
    void *pointer = std::malloc(bytes);
    if (pointer) allocations.insert(pointer);
    return pointer;
}
void release(void *pointer) {
    if (fail_next_free) {
        fail_next_free = false;
        throw std::runtime_error("injected allocator release failure");
    }
    require(allocations.erase(pointer) == 1);
    std::free(pointer);
}
template <typename Exception, typename Function>
void expectFailure(Function function) {
    try { function(); } catch (const Exception &) { return; }
    throw std::runtime_error("expected cache storage failure was not raised");
}
} // namespace

int main() {
    using llaisys::core::PagedCacheStorage;
    using llaisys::models::deepseek_v4::InterleavedLayerLayout;
    auto layout = std::make_shared<InterleavedLayerLayout>(128, 512, 128, 4);
    require(layout->numLayers() == 1 && layout->numComponents() == 2);
    require(layout->slots(0) == 160 && layout->slots(1) == 32);
    require(layout->layerBytes(0, 0) == 160U * 512U * 2U);
    expectFailure<std::invalid_argument>([] { InterleavedLayerLayout(127, 512, 128, 4); });
    expectFailure<std::invalid_argument>([] { InterleavedLayerLayout(128, 512, 128, 8); });
    expectFailure<std::overflow_error>([] { InterleavedLayerLayout(128, INT64_MAX, 128, 0); });

    LlaisysRuntimeAPI api{};
    api.malloc_device = allocate;
    api.free_device = release;
    {
        PagedCacheStorage storage(3, layout, &api);
        require(allocations.size() == 2);
        require(storage.componentBlockStride(0) == 160U * 512U * 2U);
        require(storage.componentPtr(0, 2, 0) == static_cast<std::byte *>(storage.componentPool(0)) +
                                               2 * storage.componentBlockStride(0));
        expectFailure<std::out_of_range>([&] { storage.componentPtr(0, 3, 0); });
        fail_next_free = true;
        expectFailure<std::runtime_error>([&] { storage.release(); });
        require(allocations.size() == 1); // Other component was still released.
        expectFailure<std::logic_error>([&] { storage.componentPool(1); });
        storage.release();
        storage.release(); // Retry and repeated teardown do not double-free.
        require(allocations.empty());
        expectFailure<std::logic_error>([&] { storage.componentPtr(0, 0, 0); });
    }
    remaining_allocations = 1;
    expectFailure<std::bad_alloc>([&] { PagedCacheStorage storage(3, layout, &api); });
    require(allocations.empty()); // Constructor rolls back partial allocation.
    remaining_allocations = -1;
    {
        PagedCacheStorage storage(3, layout, &api);
        require(allocations.size() == 2);
    }
    require(allocations.empty());
    std::cout << "V4 native cache layout and storage lifecycle tests passed\n";
}

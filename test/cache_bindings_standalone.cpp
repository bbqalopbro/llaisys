// Isolated CPU test module: no installed _C.so, CUDA or model runtime linkage.
#include <pybind11/pybind11.h>

void bindCache(pybind11::module_ &module);

PYBIND11_MODULE(_cache_lease_test, module) {
    module.doc() = "Standalone real C++ cache ownership bindings for CPU tests";
    bindCache(module);
}

#include "src/core/cache/block_lease.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using llaisys::core::BlockMetadata;
using llaisys::core::CacheBlockPool;
using llaisys::core::CacheBlockLease;
using llaisys::core::CachePrefixIndex;

void bindCache(py::module_ &module) {
    // Keep the GIL for these short metadata operations. The underlying manager
    // is not a concurrent container; no Python scheduling policy moves here.
    py::class_<BlockMetadata>(module, "CacheBlockMetadata")
        .def_readonly("block_id", &BlockMetadata::block_id)
        .def_readonly("ref_count", &BlockMetadata::ref_count)
        .def_readonly("num_tokens", &BlockMetadata::num_tokens)
        .def_readonly("allocated", &BlockMetadata::allocated)
        .def_readonly("computed", &BlockMetadata::computed)
        .def_readonly("cached", &BlockMetadata::cached)
        .def_readonly("block_hash", &BlockMetadata::block_hash);
    py::class_<CacheBlockPool, std::shared_ptr<CacheBlockPool>>(module, "CacheBlockPool")
        .def(py::init<size_t>())
        .def("allocate", &CacheBlockPool::allocate, py::arg("count") = 1)
        .def("metadata", &CacheBlockPool::metadata)
        .def("uncache", &CacheBlockPool::uncache)
        .def_property_readonly("num_free", &CacheBlockPool::free)
        .def_property_readonly("num_total", &CacheBlockPool::total)
        .def_property_readonly("num_cached", &CacheBlockPool::cached);
    py::class_<CacheBlockLease, std::shared_ptr<CacheBlockLease>>(module, "CacheBlockLease")
        .def("close", &CacheBlockLease::close)
        .def("share", &CacheBlockLease::share)
        .def("prefix", &CacheBlockLease::prefix)
        .def("replace", &CacheBlockLease::replace)
        .def("append", &CacheBlockLease::append)
        .def("mark_computed", &CacheBlockLease::markComputed)
        .def("mark_computed_counts", &CacheBlockLease::markComputedCounts)
        .def_property_readonly("block_ids", &CacheBlockLease::ids)
        .def_property_readonly("closed", &CacheBlockLease::closed)
        .def_property_readonly("matched_tokens", &CacheBlockLease::matchedTokens)
        .def_property_readonly("terminal_hash", &CacheBlockLease::terminalHash)
        .def("__enter__", [](const std::shared_ptr<CacheBlockLease> &lease) { lease->requireOpen(); return lease; })
        .def("__exit__", [](CacheBlockLease &lease, py::object, py::object, py::object) { lease.close(); });
    py::class_<CachePrefixIndex>(module, "CachePrefixIndex")
        .def(py::init<std::shared_ptr<CacheBlockPool>, size_t, uint64_t>(),
             py::arg("pool"), py::arg("block_size"), py::arg("salt") = 0)
        .def("publish", &CachePrefixIndex::publish)
        .def("lookup", &CachePrefixIndex::lookup);
}

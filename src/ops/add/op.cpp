#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "../llmops.hpp"



#ifdef ENABLE_METAX_API
#include "metax/add_metax.hpp"
#endif

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    check_devices({c, a, b});
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        record_dispatch("add", "llmops.cpu");
        return llmops::cpu::add(c->data(), a->data(), b->data(), kernel_dtype(c->dtype()), c->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        record_dispatch("add", "llmops.cuda");
        return llmops::cuda::add(c->data(), a->data(), b->data(), kernel_dtype(c->dtype()), c->numel(), operator_stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops

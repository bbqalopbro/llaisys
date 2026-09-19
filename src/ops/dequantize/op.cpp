#include "op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_METAX_API
#include "metax/dequantize_metax.hpp"
#endif
namespace llaisys::ops {
void dequantize(tensor_t out, tensor_t weight, tensor_t scale) {
    check_devices({out, weight, scale});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::dequantize(view(out), view(weight), view(scale)); record_dispatch("dequantize", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::dequantize(view(out), view(weight), view(scale), operator_stream()); record_dispatch("dequantize", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) return metax::dequantize(out, weight, scale);
#endif
    throw std::runtime_error("dequantize: device backend not compiled or unsupported");
}
void dequantize_int4(tensor_t out, tensor_t weight, tensor_t scale, int group_size) {
    check_devices({out, weight, scale});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::dequantize_int4(view(out), view(weight), view(scale), group_size); record_dispatch("dequantize_int4", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::dequantize_int4(view(out), view(weight), view(scale), group_size, operator_stream()); record_dispatch("dequantize_int4", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) return metax::dequantize_int4(out, weight, scale, group_size);
#endif
    throw std::runtime_error("dequantize_int4: device backend not compiled or unsupported");
}
void dequantize_awq_int4(tensor_t out, tensor_t qweight, tensor_t qzeros, tensor_t scales, int group_size) {
    check_devices({out, qweight, qzeros, scales});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::dequantize_awq_int4(view(out), view(qweight), view(qzeros), view(scales), group_size); record_dispatch("dequantize_awq_int4", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::dequantize_awq_int4(view(out), view(qweight), view(qzeros), view(scales), group_size, operator_stream()); record_dispatch("dequantize_awq_int4", "llmops.cuda"); return; }
#endif

    throw std::runtime_error("dequantize_awq_int4: device backend not compiled or unsupported");
}
}

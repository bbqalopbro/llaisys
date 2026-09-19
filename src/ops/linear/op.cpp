#include "op.hpp"
#include "../dequantize/op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_metax.hpp"
#endif
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    check_devices({out, in, weight, bias});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::linear(view(out), view(in), view(weight), view(bias)); record_dispatch("linear", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) return nvidia::linear(out, in, weight, bias);
#endif
#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) return metax::linear(out, in, weight, bias);
#endif
    throw std::runtime_error("linear: device backend not compiled or unsupported");
}
void linear_add(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
                tensor_t residual) {
    check_devices({out, in, weight, bias, residual});
    if (out->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::linear_add(view(out), view(in), view(weight), view(bias), view(residual)); record_dispatch("linear_add", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) return nvidia::linear_add(out, in, weight, bias, residual);
#endif
    throw std::runtime_error("linear_add: device backend not compiled or unsupported");
}
void linear_int4(tensor_t out, tensor_t in, tensor_t weight, tensor_t scale,
                 tensor_t bias, int group_size, tensor_t residual) {
    check_devices({out,in,weight,scale,bias,residual});
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear_int4(out, in, weight, scale, bias, group_size, residual);
    }
#endif

    if(out->deviceType()!=LLAISYS_DEVICE_CPU) throw std::runtime_error("linear_int4: device unsupported");
    // CPU fallback: dequantize + linear (NOT fused, 仅用于测试)
    int64_t rows = weight->shape()[0];
    int64_t packed_cols = weight->shape()[1];
    int64_t cols = packed_cols * 2;
    auto dq = llaisys::Tensor::create({(size_t)rows, (size_t)cols}, LLAISYS_DTYPE_F32,
                                       LLAISYS_DEVICE_CPU, 0);
    dequantize_int4(dq, weight, scale, group_size);
    linear(out, in, dq, bias);
    if (residual) llmops::cpu::add(out->data(),out->data(),residual->data(),kernel_dtype(out->dtype()),out->numel());
}

}

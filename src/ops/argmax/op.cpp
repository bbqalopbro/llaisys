#include "op.hpp"
#include "../llmops.hpp"
#ifdef ENABLE_METAX_API
#include "metax/argmax_metax.hpp"
#endif
namespace llaisys::ops {
void argmax_rows(tensor_t indices, tensor_t maxima, tensor_t values) {
    check_devices({indices,maxima,values});
#ifdef ENABLE_NVIDIA_API
    if(values->deviceType()==LLAISYS_DEVICE_NVIDIA) {
        llmops::cuda::argmax_rows(view(indices),view(maxima),view(values),operator_stream());
        record_dispatch("argmax_rows","llmops.cuda"); return;
    }
#endif
    if(values->ndim()!=2 || indices->ndim()!=1 || maxima->ndim()!=1 ||
       indices->shape()[0]!=values->shape()[0] || maxima->shape()[0]!=values->shape()[0])
        throw std::runtime_error("argmax_rows shape mismatch");
    for(size_t i=0;i<values->shape()[0];++i)
        argmax(indices->slice(0,i,i+1),maxima->slice(0,i,i+1),values->slice(0,i,i+1));
}
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    check_devices({max_idx, max_val, vals});
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) { llmops::cpu::argmax(view(max_idx), view(max_val), view(vals)); record_dispatch("argmax", "llmops.cpu"); return; }
#ifdef ENABLE_NVIDIA_API
    if (max_idx->deviceType() == LLAISYS_DEVICE_NVIDIA) { llmops::cuda::argmax(view(max_idx), view(max_val), view(vals), operator_stream()); record_dispatch("argmax", "llmops.cuda"); return; }
#endif
#ifdef ENABLE_METAX_API
    if (max_idx->deviceType() == LLAISYS_DEVICE_METAX) return metax::argmax(max_idx, max_val, vals);
#endif
    throw std::runtime_error("argmax: device backend not compiled or unsupported");
}
}

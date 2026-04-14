#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/linear_metax.hpp"
#endif

namespace llaisys::ops {
template<typename T>
void linear_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) { // Y = XW^T + offset
    T* Y_ptr = reinterpret_cast<T*>(out -> data());
    const T* W_ptr = reinterpret_cast<const T*>(weight -> data());
    const T* X_ptr = reinterpret_cast<const T*>(in -> data());

    const T* offset_ptr = nullptr;
    if (bias && bias->data()) {
        offset_ptr = reinterpret_cast<const T*>(bias->data());
    }

    int64_t Y_row_dim = out -> shape()[0];
    int64_t Y_clow_dim = out -> shape()[1];

    int64_t X_row_dim = in -> shape()[0];
    int64_t X_clow_dim = in -> shape()[1];

    int64_t W_row_dim = weight -> shape()[0];
    int64_t W_clow_dim = weight -> shape()[1];

    if(X_clow_dim != W_clow_dim){
        throw std::runtime_error("X、W的维度不搭");
    }
    if(Y_row_dim != X_row_dim || Y_clow_dim != W_row_dim){
        throw std::runtime_error("Y和X、W的维度不搭");
    }

    ptrdiff_t Y_row_stride = out -> strides()[0];
    ptrdiff_t Y_clow_stride = out -> strides()[1];

    ptrdiff_t X_row_stride = in -> strides()[0];
    ptrdiff_t X_clow_stride = in -> strides()[1];

    ptrdiff_t W_row_stride = weight -> strides()[0];
    ptrdiff_t W_clow_stride = weight -> strides()[1];

    for(int i = 0; i < X_row_dim; i ++){
         for(int j = 0; j < W_row_dim; j ++){
            float sum = 0.0f;
            for(int l = 0; l < X_clow_dim; l++){
                int64_t X_idx = i * X_row_stride+ l * X_clow_stride;
                int64_t W_idx = j * W_row_stride+ l * W_clow_stride;
                sum += llaisys::utils::cast<float>(X_ptr[X_idx]) * llaisys::utils::cast<float>(W_ptr[W_idx]);
            }
            if (bias && bias->data()) {
                sum += llaisys::utils::cast<float>(offset_ptr[j]);
            }
            int64_t Y_idx = i * Y_row_stride + j * Y_clow_stride;
            Y_ptr[Y_idx] = llaisys::utils::cast<T>(sum);
         }
    }
}
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto w_dtype = weight->dtype();
    auto in_dtype = in->dtype();

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear(out, in, weight, bias);
    }
#endif

#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) {
        return metax::linear(out, in, weight, bias);
    }
#endif

    // Mixed precision path: FP16 weight + FP32 input → FP32 output
    if (w_dtype == llaisysDataType_t::LLAISYS_DTYPE_F16 && in_dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        // Use the FP16 kernel with cast<float> — it already accumulates in F32
        // But we need the output to be F32, so use a special mixed kernel
        const llaisys::fp16_t* W_ptr = reinterpret_cast<const llaisys::fp16_t*>(weight->data());
        const float* X_ptr = reinterpret_cast<const float*>(in->data());
        float* Y_ptr = reinterpret_cast<float*>(out->data());

        const float* offset_ptr = nullptr;
        if (bias && bias->data()) {
            offset_ptr = reinterpret_cast<const float*>(bias->data());
        }

        int64_t M = in->shape()[0], K = in->shape()[1], N = weight->shape()[0];
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int64_t l = 0; l < K; l++) {
                    sum += X_ptr[i * in->strides()[0] + l * in->strides()[1]]
                         * llaisys::utils::cast<float>(W_ptr[j * weight->strides()[0] + l * weight->strides()[1]]);
                }
                if (offset_ptr) sum += offset_ptr[j];
                Y_ptr[i * out->strides()[0] + j * out->strides()[1]] = sum;
            }
        }
        return;
    }

    if (w_dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        linear_cpu_kernel<float>(out, in, weight, bias);
    } 
    else if (w_dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) { 
        linear_cpu_kernel<llaisys::fp16_t>(out, in, weight, bias);
    } 
    else if (w_dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) { 
        linear_cpu_kernel<llaisys::bf16_t>(out, in, weight, bias);
    }
    else {
        throw std::runtime_error("数据类型不支持");
    }
}

void linear_add(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
                tensor_t residual) {
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear_add(out, in, weight, bias, residual);
    }
#endif

    // CPU fallback: separate linear + add
    linear(out, in, weight, bias);
    if (residual && residual->data()) {
        // reuse existing ops::add
        auto dt = out->dtype();
        int64_t n = out->numel();
        if (dt == LLAISYS_DTYPE_F32) {
            float *o = reinterpret_cast<float *>(out->data());
            const float *r = reinterpret_cast<const float *>(residual->data());
            for (int64_t i = 0; i < n; i++) o[i] += r[i];
        } else if (dt == LLAISYS_DTYPE_F16) {
            auto *o = reinterpret_cast<llaisys::fp16_t *>(out->data());
            const auto *r = reinterpret_cast<const llaisys::fp16_t *>(residual->data());
            for (int64_t i = 0; i < n; i++) {
                float v = llaisys::utils::cast<float>(o[i]) + llaisys::utils::cast<float>(r[i]);
                o[i] = llaisys::utils::cast<llaisys::fp16_t>(v);
            }
        }
    }
}

} // namespace llaisys::ops

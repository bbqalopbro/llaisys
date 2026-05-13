#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/self_attention_metax.hpp"
#endif

namespace llaisys::ops {
template<typename T>
void self_attention_cpu_kernel(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {//1、A = Q * K ^ T  2、softmax(A) 3、softmax(A) * V
    const T* q_ptr = reinterpret_cast<const T*>(q->data());
    const T* k_ptr = reinterpret_cast<const T*>(k->data());
    const T* v_ptr = reinterpret_cast<const T*>(v->data());
    T* out_ptr = reinterpret_cast<T*>(attn_val->data());

    int64_t seq_len = q->shape()[0];
    int64_t n_head  = q->shape()[1];
    int64_t head_dim = q->shape()[2]; // d

    // k, v: [total_len, nkvhead, head_dim/v_dim]
    // 注意：total_len 包含了历史缓存 + 当前输入
    int64_t total_len = k->shape()[0];
    int64_t n_kv_head = k->shape()[1];
    int64_t v_dim = v->shape()[2]; 

    ptrdiff_t q_s0 = q->strides()[0];
    ptrdiff_t q_s1 = q->strides()[1];
    ptrdiff_t q_s2 = q->strides()[2];

    ptrdiff_t k_s0 = k->strides()[0];
    ptrdiff_t k_s1 = k->strides()[1];
    ptrdiff_t k_s2 = k->strides()[2];

    ptrdiff_t v_s0 = v->strides()[0];
    ptrdiff_t v_s1 = v->strides()[1];
    ptrdiff_t v_s2 = v->strides()[2];

    ptrdiff_t o_s0 = attn_val->strides()[0];
    ptrdiff_t o_s1 = attn_val->strides()[1];
    ptrdiff_t o_s2 = attn_val->strides()[2];

    int64_t group_size = n_head / n_kv_head; //GQA的存在，要让Q头共享KV头

    std::vector<float> A(total_len);
    for(int64_t i = 0; i < seq_len; i++){ //遍历token
        int64_t current_pos = total_len - seq_len + i;
        for (int64_t h = 0; h < n_head; ++h){ //遍历Q头

            int64_t kv_h = h / group_size; //对应kv头
            float max_score = -std::numeric_limits<float>::infinity(); 
            for(int64_t j = 0; j < total_len; j++){
                if (j > current_pos) {
                    A[j] = -std::numeric_limits<float>::infinity(); //mask
                    continue;
                }
                double sum = 0.0;
                for(int64_t k = 0; k < head_dim; k++){ //点积
                    int64_t q_idx = i * q_s0 + h * q_s1 + k * q_s2;
                    int64_t k_idx = j * k_s0 + kv_h * k_s1 + k * k_s2;
                    double Q_val = static_cast<double>(llaisys::utils::cast<float>(q_ptr[q_idx]));
                    double K_val = static_cast<double>(llaisys::utils::cast<float>(k_ptr[k_idx]));  
                    sum += Q_val * K_val;
                }
                A[j] = static_cast<float>(sum * scale); //缩放
                if(max_score < A[j]) max_score = A[j]; //记录最大值
            }


            double sum = 0.0;
            for(int64_t j = 0; j < total_len; j++){
                if (A[j] == -std::numeric_limits<float>::infinity()) {
                    A[j] = 0.0f;
                } else {
                    A[j] = static_cast<float>(std::exp(static_cast<double>(A[j] - max_score)));
                }
                sum += A[j];
            }

            float inv_sum = 1.0f / (static_cast<float>(sum) + 1e-6f);
            for (int64_t j = 0; j < total_len; j ++) {
                A[j] *= inv_sum; 
            }

            for(int64_t v = 0; v < v_dim; v ++){
                double sum2 = 0.0;
                for (int64_t j = 0; j < total_len; j ++) {
                    float prob = A[j];
                    if (prob == 0.0f) continue;
                    int64_t v_idx = j * v_s0 + kv_h * v_s1 + v * v_s2;
                    float val_v = llaisys::utils::cast<float>(v_ptr[v_idx]);
                    sum2 += static_cast<double>(prob) * static_cast<double>(val_v);
                }
                int64_t out_idx = i * o_s0 + h * o_s1 + v * o_s2;
                out_ptr[out_idx] = llaisys::utils::cast<T>(static_cast<float>(sum2));
            }
        }
    }

}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    auto dtype = q->dtype();

#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::self_attention(attn_val, q, k, v, scale);
    }
#endif

#ifdef ENABLE_METAX_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_METAX) {
        return metax::self_attention(attn_val, q, k, v, scale);
    }
#endif

    if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        self_attention_cpu_kernel<float>(attn_val, q, k, v, scale);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) { 
        self_attention_cpu_kernel<llaisys::fp16_t>(attn_val, q, k, v, scale);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) { 
        self_attention_cpu_kernel<llaisys::bf16_t>(attn_val, q, k, v, scale);
    }
    else {
        throw std::runtime_error("数据类型不支持");
    }
}
} // namespace llaisys::ops

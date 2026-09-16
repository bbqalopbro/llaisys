#include "llaisys/ops.h"

#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/sample/op.hpp"
#include "../ops/dequantize/op.hpp"
#include "../ops/self_attention/paged_attention.hpp"

__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias ? bias->tensor : nullptr);
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        llaisys::ops::rearrange(out->tensor, in->tensor);
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }
    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    }
    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }
    void llaisysSample(llaisysTensor_t out_idx, llaisysTensor_t logits, float temperature, int top_k, float top_p, uint64_t seed) {
        llaisys::ops::sample(out_idx->tensor, logits->tensor, temperature, top_k, top_p, seed);
    }
    void llaisysDequantize(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale) {
        llaisys::ops::dequantize(out->tensor, weight->tensor, scale->tensor);
    }
    void llaisysDequantizeInt4(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale, int group_size) {
        llaisys::ops::dequantize_int4(out->tensor, weight->tensor, scale->tensor, group_size);
    }
    void llaisysLinearInt4(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight,
                           llaisysTensor_t scale, llaisysTensor_t bias,
                           int group_size, llaisysTensor_t residual) {
        llaisys::ops::linear_int4(out->tensor, in->tensor, weight->tensor, scale->tensor,
                                  bias ? bias->tensor : nullptr, group_size,
                                  residual ? residual->tensor : nullptr);
    }
    void llaisysPagedAttention(
        float *output, const float *query,
        void *k_pool, void *v_pool,
        int *block_tables, int *seq_lens,
        int batch_size, int num_heads, int num_kv_heads, int head_dim,
        int block_size, int max_blocks_per_seq,
        size_t pool_block_stride, size_t pool_layer_stride,
        int layer_idx, float scale,
        llaisysDeviceType_t device_type,
        int kv_quant) {
        llaisys::ops::paged_attention(output, query, k_pool, v_pool,
                                       block_tables, seq_lens,
                                       batch_size, num_heads, num_kv_heads, head_dim,
                                       block_size, max_blocks_per_seq,
                                       pool_block_stride, pool_layer_stride,
                                       layer_idx, scale, device_type,
                                       static_cast<llaisys::ops::KVQuantMode>(kv_quant));
    }
}

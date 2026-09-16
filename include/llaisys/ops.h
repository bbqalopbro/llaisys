#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H

#include "tensor.h"

__C {
    __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
    __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
    __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
    __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
    __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
    __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
    __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
    __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
    __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
    __export void llaisysSample(llaisysTensor_t out_idx, llaisysTensor_t logits, float temperature, int top_k, float top_p, uint64_t seed);
    __export void llaisysDequantize(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale);
    __export void llaisysDequantizeInt4(llaisysTensor_t out, llaisysTensor_t weight, llaisysTensor_t scale, int group_size);
    __export void llaisysLinearInt4(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight,
                                    llaisysTensor_t scale, llaisysTensor_t bias,
                                    int group_size, llaisysTensor_t residual);

    // Paged Attention: Q·K^T→softmax→·V with block-pooled KV-Cache
    // block_tables: [batch_size, max_blocks_per_seq] CPU-side int array
    // seq_lens: [batch_size] CPU-side int array
    // kv_quant: 0=FP32, 1=INT8, 2=INT4
    __export void llaisysPagedAttention(
        float *output, const float *query,
        void *k_pool, void *v_pool,
        int *block_tables, int *seq_lens,
        int batch_size, int num_heads, int num_kv_heads, int head_dim,
        int block_size, int max_blocks_per_seq,
        size_t pool_block_stride, size_t pool_layer_stride,
        int layer_idx, float scale,
        llaisysDeviceType_t device_type,
        int kv_quant);
}

#endif

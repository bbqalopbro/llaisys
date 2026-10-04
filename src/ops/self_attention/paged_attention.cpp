#include <llmops/ops/paged.hpp>
#include "paged_attention.hpp"
#include "../dispatch.hpp"
#include "../llmops.hpp"
#include "../../../src/core/kv_quant.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/paged_attention_nvidia.cuh"
#include "nvidia/flashinfer_adapter.cuh"
#endif

#include <cmath>
#include <cstdlib>
#include <string>
#include <limits>
#include <vector>
#include <stdexcept>

namespace llaisys::ops {

void paged_prefill(void *out,const void *q,const void *k,const void *v,const int *table,
    int nq,int nk,int h,int hkv,int dim,int bs,int max_pages,size_t block_stride,
    size_t layer_stride,int layer,float scale,llaisysDeviceType_t dev,llaisysDataType_t dtype,bool stable) {
#ifdef ENABLE_NVIDIA_API
    if(dev==LLAISYS_DEVICE_NVIDIA) {
        llmops::cuda::paged_prefill_device(out,q,k,v,table,nq,nk,h,hkv,dim,bs,max_pages,
            block_stride,layer_stride,layer,scale,kernel_dtype(dtype),operator_stream(),stable);
        record_dispatch("paged_prefill","llmops.cuda");return;
    }
#endif
    if(dev!=LLAISYS_DEVICE_CPU || dtype!=LLAISYS_DTYPE_F32)
        throw std::runtime_error("paged prefill: unsupported device/dtype");
    for(int row=0;row<nq;++row) {
        int length=nk-nq+row+1;
        llmops::cpu::paged_attention_fp32((float*)out+row*h*dim,(const float*)q+row*h*dim,
            k,v,table,&length,1,h,hkv,dim,bs,max_pages,block_stride,layer_stride,layer,scale);
    }
}

// ── Dispatch ──────────────────────────────────────────────────────

void paged_attention(
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDeviceType_t device_type,
    KVQuantMode kv_quant,
    llaisysDataType_t dtype)
{
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA && kv_quant == KVQuantMode::FP32) {
        int group_size = (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 0;
        bool flashinfer_dtype_supported = (dtype == LLAISYS_DTYPE_F16);
        bool flashinfer_head_dim_supported =
            (head_dim == 64 || head_dim == 128 || head_dim == 256);
        bool flashinfer_group_size_supported =
            (group_size == 1 || group_size == 2 || group_size == 4 || group_size == 8);
        const char *mode=std::getenv("LLAISYS_PAGED_ATTENTION");
        bool vendor=mode && std::string(mode)=="flashinfer";
        if(mode && std::string(mode)!="native" && !vendor) throw std::runtime_error("paged Attention mode must be native or flashinfer");
        if (vendor && flashinfer_dtype_supported &&
            flashinfer_head_dim_supported &&
            flashinfer_group_size_supported &&
            nvidia::flashinfer_available()) {
            nvidia::flashinfer_paged_attention(output, query, k_pool, v_pool,
                                               block_tables, seq_lens,
                                               batch_size, num_heads, num_kv_heads, head_dim,
                                               block_size, max_blocks_per_seq,
                                               pool_block_stride, pool_layer_stride,
                                               layer_idx, scale, dtype);
            record_dispatch("paged_attention", "framework.flashinfer");
            return;
        }
        if(vendor) throw std::runtime_error("FlashInfer unsupported dtype/shape or not compiled");
        nvidia::paged_attention(output, query, k_pool, v_pool,
                                block_tables, seq_lens,
                                batch_size, num_heads, num_kv_heads, head_dim,
                                block_size, max_blocks_per_seq,
                                pool_block_stride, pool_layer_stride,
                                layer_idx, scale, dtype);
        record_dispatch("paged_attention", "llmops.cuda");
        return;
    }
#endif

    if(device_type != LLAISYS_DEVICE_CPU) throw std::runtime_error("paged Attention: unsupported device or GPU KV quantization");
    if(dtype != LLAISYS_DTYPE_F32) throw std::runtime_error("CPU paged Attention requires FP32 query/output");
    // CPU fallback: 仍然使用 float* (CPU 推理不走 FP16)
    switch (kv_quant) {
        case KVQuantMode::INT8:
            llmops::cpu::paged_attention_int8((float*)output, (const float*)query, k_pool, v_pool,
                                     block_tables, seq_lens,
                                     batch_size, num_heads, num_kv_heads, head_dim,
                                     block_size, max_blocks_per_seq,
                                     pool_block_stride, pool_layer_stride,
                                     layer_idx, scale);
            break;
        case KVQuantMode::INT4:
            llmops::cpu::paged_attention_int4((float*)output, (const float*)query, k_pool, v_pool,
                                     block_tables, seq_lens,
                                     batch_size, num_heads, num_kv_heads, head_dim,
                                     block_size, max_blocks_per_seq,
                                     pool_block_stride, pool_layer_stride,
                                     layer_idx, scale);
            break;
        default:
            llmops::cpu::paged_attention_fp32((float*)output, (const float*)query, k_pool, v_pool,
                                     block_tables, seq_lens,
                                     batch_size, num_heads, num_kv_heads, head_dim,
                                     block_size, max_blocks_per_seq,
                                     pool_block_stride, pool_layer_stride,
                                     layer_idx, scale);
            break;
    }
}

void paged_attention_device(
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables_dev, const int *seq_lens_dev,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDeviceType_t device_type,
    llaisysDataType_t dtype, void *workspace, size_t workspace_bytes)
{
    const char *mode=std::getenv("LLAISYS_PAGED_ATTENTION");
    if(mode && std::string(mode)!="native") throw std::runtime_error("paged device/Graph API currently supports native only");
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        nvidia::paged_attention_device(output, query, k_pool, v_pool,
                                       block_tables_dev, seq_lens_dev,
                                       batch_size, num_heads, num_kv_heads, head_dim,
                                       block_size, max_blocks_per_seq,
                                       pool_block_stride, pool_layer_stride,
                                       layer_idx, scale, dtype, workspace, workspace_bytes);
        record_dispatch("paged_attention", "llmops.cuda");
        return;
    }
#endif
    throw std::runtime_error("paged_attention_device: only supported on NVIDIA GPU");
}

} // namespace llaisys::ops

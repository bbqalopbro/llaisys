/**
 * FlashInfer Adapter — wraps FlashInfer's BatchDecodeWithPagedKVCache API
 * into LLAISYS's paged attention interface.
 *
 * When ENABLE_FLASHINFER is defined (via xmake --flashinfer=y), this file
 * bridges between LLAISYS's flat block pool layout and FlashInfer's
 * BatchDecodeHandler. When not defined, the adapter stubs return false
 * and the caller falls back to the built-in CUDA kernel.
 *
 * Integration steps to enable:
 *   1. Install FlashInfer: pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
 *   2. Build with: xmake f --flashinfer=y --flashinfer-include=/path/to/flashinfer/include
 *   3. The paged_attention dispatch will automatically prefer FlashInfer
 */

#ifdef ENABLE_FLASHINFER

#include "flashinfer_adapter.cuh"
#include <flashinfer/decode.cuh>
#include <flashinfer/page.cuh>

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "[FlashInfer CUDA ERROR] %s at %s:%d\n",             \
                    cudaGetErrorString(err), __FILE__, __LINE__);                 \
            throw std::runtime_error(cudaGetErrorString(err));                    \
        }                                                                         \
    } while (0)

namespace llaisys::ops::nvidia {

bool flashinfer_available() { return true; }

void flashinfer_paged_attention(
    float *output, const float *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    // FlashInfer expects:
    //   Q:  [batch_size, num_heads, head_dim] on device
    //   KV paged: tables + pool pointers
    //
    // LLAISYS pool layout: [num_blocks, nlayer, block_size, nkvh, dh]
    // FlashInfer layout:   [num_blocks, 2, block_size, num_kv_heads, head_dim]
    //   where the "2" dimension is [K, V] interleaved
    //
    // We need to create a view that offsets into the correct layer.

    using namespace flashinfer;

    // Compute layer-offset base pointers
    const char *k_layer = static_cast<const char *>(k_pool) +
                          static_cast<size_t>(layer_idx) * pool_layer_stride;
    const char *v_layer = static_cast<const char *>(v_pool) +
                          static_cast<size_t>(layer_idx) * pool_layer_stride;

    // FlashInfer's paged KV cache descriptor
    // We use the batch decode handler with per-request page tables
    paged_kv_t<float, int32_t> paged_kv(
        num_kv_heads,
        block_size,
        head_dim,
        batch_size,
        reinterpret_cast<float *>(const_cast<char *>(k_layer)),
        reinterpret_cast<float *>(const_cast<char *>(v_layer)),
        const_cast<int *>(block_tables),
        const_cast<int *>(seq_lens),
        pool_block_stride,
        max_blocks_per_seq);

    // Create decode handler
    BatchDecodeHandler handler;
    handler.BeginForward<float, float, int32_t>(
        const_cast<int *>(seq_lens),
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        RotaryMode::kNone);

    // Run decode
    cudaError_t status = handler.Forward<float, float, int32_t>(
        const_cast<float *>(query),
        paged_kv,
        output,
        nullptr, // lse (log-sum-exp, optional)
        num_heads,
        RotaryMode::kNone,
        scale);

    if (status != cudaSuccess) {
        throw std::runtime_error("FlashInfer BatchDecode failed");
    }

    handler.EndForward();
}

} // namespace llaisys::ops::nvidia

#endif // ENABLE_FLASHINFER

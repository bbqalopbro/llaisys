/**
 * FlashInfer Adapter — bridges LLAISYS paged KV decode to FlashInfer's
 * public batch decode API.
 *
 * This adapter supports FP32/FP16/BF16 query + KV-cache. Middle accumulation
 * stays inside FlashInfer's kernel implementation.
 */

#ifdef ENABLE_FLASHINFER

#include "flashinfer_adapter.cuh"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#endif
#include <flashinfer/allocator.h>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

using flashinfer::DecodePlanInfo;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

struct FlashInferDecodeCache {
    // FlashInfer 的 decode kernel 需要临时 workspace。
    // 这里做成进程内静态 cache，避免每次 decode 都重新 malloc/free。
    void *float_workspace = nullptr;
    void *int_workspace = nullptr;
    void *page_locked_int_workspace = nullptr;
    size_t float_workspace_bytes = 0;
    size_t int_workspace_bytes = 0;

    // FlashInfer 使用 CSR 风格的分页索引：
    //
    //   indices:
    //     所有 batch 的物理 page/block id 连续拼在一起。
    //
    //   indptr:
    //     长度为 batch_size + 1。
    //     第 b 个 sequence 的 pages 在 indices[indptr[b] : indptr[b+1]]。
    //
    //   last_page_len:
    //     每个 sequence 最后一个 page 里实际有多少 token。
    //
    // 这些 buffer 在 GPU 上，FlashInfer kernel 直接读取。
    int32_t *d_indices = nullptr;
    int32_t *d_indptr = nullptr;
    int32_t *d_last_page_len = nullptr;
    size_t indices_capacity = 0;
    size_t indptr_capacity = 0;
    size_t last_page_capacity = 0;

    // plan_info 是 FlashInfer DecodePlan 的输出。
    // 它描述本次 batch decode 的调度方案、workspace 切分位置、
    // 是否 split_kv、padded_batch_size 等。
    DecodePlanInfo plan_info;

    // plan_key 用来判断当前 shape/page 结构是否和上次一致。
    // 如果一致，可以复用 plan；如果不一致，需要重新 DecodePlan。
    std::string plan_key;

    ~FlashInferDecodeCache() {
        if (float_workspace) cudaFree(float_workspace);
        if (int_workspace) cudaFree(int_workspace);
        if (page_locked_int_workspace) cudaFreeHost(page_locked_int_workspace);
        if (d_indices) cudaFree(d_indices);
        if (d_indptr) cudaFree(d_indptr);
        if (d_last_page_len) cudaFree(d_last_page_len);
    }
};

static FlashInferDecodeCache &cache() {
    static FlashInferDecodeCache c;
    return c;
}

static void ensure_workspace(size_t float_bytes, size_t int_bytes) {
    auto &c = cache();
    if (float_bytes > c.float_workspace_bytes) {
        if (c.float_workspace) CUDA_CHECK(cudaFree(c.float_workspace));
        CUDA_CHECK(cudaMalloc(&c.float_workspace, float_bytes));
        c.float_workspace_bytes = float_bytes;
        c.plan_key.clear();
    }
    if (int_bytes > c.int_workspace_bytes) {
        if (c.int_workspace) CUDA_CHECK(cudaFree(c.int_workspace));
        if (c.page_locked_int_workspace) CUDA_CHECK(cudaFreeHost(c.page_locked_int_workspace));
        CUDA_CHECK(cudaMalloc(&c.int_workspace, int_bytes));
        CUDA_CHECK(cudaMallocHost(&c.page_locked_int_workspace, int_bytes));
        c.int_workspace_bytes = int_bytes;
        c.plan_key.clear();
    }
}

static void ensure_index_buffers(size_t nnz_pages, size_t batch_size) {
    auto &c = cache();
    if (nnz_pages > c.indices_capacity) {
        if (c.d_indices) CUDA_CHECK(cudaFree(c.d_indices));
        CUDA_CHECK(cudaMalloc(&c.d_indices, nnz_pages * sizeof(int32_t)));
        c.indices_capacity = nnz_pages;
    }
    if (batch_size + 1 > c.indptr_capacity) {
        if (c.d_indptr) CUDA_CHECK(cudaFree(c.d_indptr));
        CUDA_CHECK(cudaMalloc(&c.d_indptr, (batch_size + 1) * sizeof(int32_t)));
        c.indptr_capacity = batch_size + 1;
    }
    if (batch_size > c.last_page_capacity) {
        if (c.d_last_page_len) CUDA_CHECK(cudaFree(c.d_last_page_len));
        CUDA_CHECK(cudaMalloc(&c.d_last_page_len, batch_size * sizeof(int32_t)));
        c.last_page_capacity = batch_size;
    }
}

static std::string make_plan_key(int batch_size, int num_heads, int num_kv_heads, int head_dim,
                                 int block_size, const std::vector<int32_t> &indptr) {
    // FlashInfer 的 plan 与 batch/page 结构有关。
    // 这里把会影响调度的字段编码成字符串：
    //   batch_size, num_heads, num_kv_heads, head_dim, block_size, indptr
    //
    // 注意这里只放了 indptr，没有放 indices。
    // 这表示“每个请求有多少 page”变化时重建 plan；
    // 具体 page id 变化时只更新 indices，不必重建 plan。
    std::string key = std::to_string(batch_size) + "|" + std::to_string(num_heads) + "|" +
                      std::to_string(num_kv_heads) + "|" + std::to_string(head_dim) + "|" +
                      std::to_string(block_size) + "|";
    for (int32_t v : indptr) {
        key += std::to_string(v);
        key.push_back(',');
    }
    return key;
}

template <typename T, uint32_t HEAD_DIM>
static void flashinfer_run_typed(
    T *output, const T *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    using Params = flashinfer::BatchDecodeParams<T, T, T, int32_t>;
    using AttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;

    // LLAISYS 传进来的 block_tables 是 dense 二维表：
    //
    //   block_tables[b * max_blocks_per_seq + bi]
    //      = 第 b 个 sequence 的第 bi 个逻辑 block 对应的物理 block_id
    //
    // FlashInfer 需要的是 CSR 风格分页表：
    //
    //   indices       = 所有有效物理 block_id 顺序拼接
    //   indptr        = 每个 batch 在 indices 里的起止位置
    //   last_page_len = 每个 batch 最后一页实际 token 数
    //
    // 例子：
    //   seq0 blocks: [5, 8, 2]
    //   seq1 blocks: [7]
    //   seq2 blocks: [4, 9]
    //
    // 转换后：
    //   indices = [5, 8, 2, 7, 4, 9]
    //   indptr  = [0, 3, 4, 6]
    std::vector<int32_t> indices;
    indices.reserve(static_cast<size_t>(batch_size) * max_blocks_per_seq);
    std::vector<int32_t> indptr(batch_size + 1, 0);
    std::vector<int32_t> last_page_len(batch_size, 0);

    for (int b = 0; b < batch_size; ++b) {
        int seq_len = seq_lens[b];
        int num_blocks = (seq_len + block_size - 1) / block_size;
        indptr[b + 1] = indptr[b] + num_blocks;

        // 最后一个 page 可能没满。
        // 例如 seq_len=37, block_size=16:
        //   num_blocks=3, last_page_len=5。
        last_page_len[b] = (seq_len > 0) ? ((seq_len - 1) % block_size + 1) : 0;
        for (int bi = 0; bi < num_blocks; ++bi) {
            indices.push_back(static_cast<int32_t>(block_tables[b * max_blocks_per_seq + bi]));
        }
    }

    ensure_workspace(64ULL * 1024 * 1024, 16ULL * 1024 * 1024);
    ensure_index_buffers(indices.size(), batch_size);

    auto &c = cache();
    std::string plan_key = make_plan_key(batch_size, num_heads, num_kv_heads, HEAD_DIM, block_size, indptr);

    if (c.plan_key != plan_key) {
        // FlashInfer 是 plan/run 两阶段：
        //
        //   DecodePlan:
        //     根据 batch/page 结构和 head 配置，规划 kernel 调度和 workspace 使用。
        //
        //   BatchDecodeWithPagedKVCacheDispatched:
        //     使用 plan_info 和 paged_kv_t 真正执行 attention。
        //
        // plan 只依赖 page 数量结构等信息，不依赖具体 K/V 数值。
        cudaStream_t stream = nullptr;
        DISPATCH_GQA_GROUP_SIZE(num_heads / num_kv_heads, GROUP_SIZE, {
            cudaError_t status = flashinfer::DecodePlan<HEAD_DIM, PosEncodingMode::kNone, AttentionVariant, Params>(
                c.float_workspace, c.float_workspace_bytes,
                c.int_workspace, c.page_locked_int_workspace,
                c.int_workspace_bytes, c.plan_info,
                indptr.data(), batch_size, num_heads, block_size,
                false, stream,
                flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                    GROUP_SIZE, HEAD_DIM, PosEncodingMode::kNone, AttentionVariant, Params>);
            CUDA_CHECK(status);
        });
        c.plan_key = plan_key;
    }

    CUDA_CHECK(cudaMemcpy(c.d_indices, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.d_indptr, indptr.data(), indptr.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.d_last_page_len, last_page_len.data(),
                          last_page_len.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    // LLAISYS 的物理 KV pool layout 是：
    //
    //   [num_blocks, nlayer, block_size, num_kv_heads, head_dim]
    //
    // FlashInfer 的 paged_kv_t 只需要“当前 layer 的 page pool”视角：
    //
    //   [num_blocks, block_size, num_kv_heads, head_dim]
    //
    // 因此这里先把 base pointer 从整个 pool 移到当前 layer。
    // 后续具体访问哪个 physical block/page，由 FlashInfer 通过 indices 完成。
    const char *k_layer = static_cast<const char *>(k_pool) + static_cast<size_t>(layer_idx) * pool_layer_stride;
    const char *v_layer = static_cast<const char *>(v_pool) + static_cast<size_t>(layer_idx) * pool_layer_stride;

    // FlashInfer 这里的 stride 单位是“元素个数”，不是字节数。
    //
    // 因为 k_layer/v_layer 会被 reinterpret_cast 成 T*，
    // 所以 pool_block_stride 需要除以 sizeof(T)。
    //
    // 这四个 stride 对应访问：
    //   K[page_id][token_offset][kv_head][dim]
    //
    // 地址展开后等价于：
    //   k_layer
    //   + page_id      * pool_block_stride
    //   + token_offset * num_kv_heads * HEAD_DIM * sizeof(T)
    //   + kv_head      * HEAD_DIM * sizeof(T)
    //   + dim          * sizeof(T)
    int64_t kv_strides[4] = {
        static_cast<int64_t>(pool_block_stride / sizeof(T)),
        static_cast<int64_t>(num_kv_heads * HEAD_DIM),
        static_cast<int64_t>(HEAD_DIM),
        1
    };

    // paged_kv_t 是 FlashInfer 眼里的“分页 KV cache 描述符”。
    // 它把以下信息打包到一起：
    //
    //   1. 当前 layer 的 K/V base pointer
    //   2. 每个 page/block 的 stride
    //   3. page 内 token/head/dim 的 layout
    //   4. CSR 页表：indices / indptr / last_page_len
    //
    // 因此 FlashInfer 并没有绕过本项目的 block pool；
    // 它只是用自己的高性能 kernel 读取同一份 paged KV cache。
    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        num_kv_heads, block_size, HEAD_DIM, batch_size, QKVLayout::kNHD,
        reinterpret_cast<T *>(const_cast<char *>(k_layer)),
        reinterpret_cast<T *>(const_cast<char *>(v_layer)),
        kv_strides, c.d_indices, c.d_indptr, c.d_last_page_len);

    Params params;
    // query/output layout:
    //   query: [batch_size, num_heads, HEAD_DIM]
    //   output: [batch_size, num_heads, HEAD_DIM]
    //
    // q_stride_n 表示相邻 batch 的 query 起点间隔多少元素；
    // q_stride_h 表示相邻 head 的 query 起点间隔多少元素。
    params.q = const_cast<T *>(query);
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = output;
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.padded_batch_size = static_cast<uint32_t>(c.plan_info.padded_batch_size);
    params.num_qo_heads = num_heads;
    params.q_stride_n = static_cast<int32_t>(num_heads * HEAD_DIM);
    params.q_stride_h = static_cast<int32_t>(HEAD_DIM);
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;
    params.request_indices = nullptr;
    params.kv_tile_indices = nullptr;
    params.o_indptr = nullptr;
    params.kv_chunk_size_ptr = nullptr;
    params.block_valid_mask = nullptr;
    params.partition_kv = false;

    T *tmp_v = nullptr;
    float *tmp_s = nullptr;
    params.request_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(c.int_workspace, c.plan_info.request_indices_offset);
    params.kv_tile_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(c.int_workspace, c.plan_info.kv_tile_indices_offset);
    params.o_indptr =
        flashinfer::GetPtrFromBaseOffset<int32_t>(c.int_workspace, c.plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        flashinfer::GetPtrFromBaseOffset<int32_t>(c.int_workspace, c.plan_info.kv_chunk_size_ptr_offset);

    if (c.plan_info.split_kv) {
        // split_kv 是 FlashInfer 针对长上下文/大 batch 的一种调度策略：
        // 把 KV 维度上的工作拆成多个 chunk，先写临时结果 tmp_v/tmp_s，
        // 再由内部流程合并，减少单个 CTA 的压力。
        tmp_v = flashinfer::GetPtrFromBaseOffset<T>(c.float_workspace, c.plan_info.v_offset);
        tmp_s = flashinfer::GetPtrFromBaseOffset<float>(c.float_workspace, c.plan_info.s_offset);
        if (c.plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                flashinfer::GetPtrFromBaseOffset<bool>(c.int_workspace, c.plan_info.block_valid_mask_offset);
        }
        params.padded_batch_size = static_cast<uint32_t>(c.plan_info.padded_batch_size);
        params.partition_kv = true;
    }

    DISPATCH_GQA_GROUP_SIZE(num_heads / num_kv_heads, GROUP_SIZE, {
        // 真正执行 FlashInfer paged decode attention：
        //
        //   对每个 batch/head：
        //     通过 paged_kv.indices/indptr 查 physical page
        //     从 k_layer/v_layer 读取历史 KV
        //     计算 softmax(QK^T) V
        //     写入 output
        cudaError_t status = flashinfer::BatchDecodeWithPagedKVCacheDispatched<
            HEAD_DIM, PosEncodingMode::kNone, AttentionVariant, Params>(params, tmp_v, tmp_s, false, nullptr);
        CUDA_CHECK(status);
    });
}

template <typename T>
static void dispatch_head_dim(
    T *output, const T *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    switch (head_dim) {
    case 64:
        flashinfer_run_typed<T, 64>(
            output, query, k_pool, v_pool, block_tables, seq_lens,
            batch_size, num_heads, num_kv_heads,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        break;
    case 128:
        flashinfer_run_typed<T, 128>(
            output, query, k_pool, v_pool, block_tables, seq_lens,
            batch_size, num_heads, num_kv_heads,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        break;
    case 256:
        flashinfer_run_typed<T, 256>(
            output, query, k_pool, v_pool, block_tables, seq_lens,
            batch_size, num_heads, num_kv_heads,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        break;
    default:
        throw std::runtime_error("FlashInfer adapter: unsupported head_dim");
    }
}

} // namespace

bool flashinfer_available() { return true; }

void flashinfer_paged_attention(
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    const int *block_tables, const int *seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale,
    llaisysDataType_t dtype)
{
    // 当前 adapter 只启用了 FP16。
    // 外层 dispatcher 已经保证：
    //   dtype == F16
    //   head_dim in {64, 128, 256}
    //   group_size in {1, 2, 4, 8}
    //
    // 这里仍保留 switch，作为防御式检查。
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        dispatch_head_dim<__half>(
            static_cast<__half *>(output), static_cast<const __half *>(query),
            k_pool, v_pool, block_tables, seq_lens,
            batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        break;
    default:
        throw std::runtime_error("FlashInfer adapter: unsupported dtype (expected F16)");
    }
}

} // namespace llaisys::ops::nvidia

#endif // ENABLE_FLASHINFER

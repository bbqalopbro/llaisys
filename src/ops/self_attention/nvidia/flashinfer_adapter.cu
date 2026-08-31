/**
 * FlashInfer Adapter — bridges LLAISYS paged KV prefill/decode to
 * FlashInfer's public batch APIs.
 *
 * Paged prefill supports FP16/BF16; paged decode currently enables FP16.
 * Intermediate accumulation stays inside FlashInfer's kernel implementation.
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
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
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
#include <cstdlib>
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
using flashinfer::MaskMode;
using flashinfer::PosEncodingMode;
using flashinfer::PrefillPlanInfo;
using flashinfer::QKVLayout;

constexpr size_t kPrefillFloatWorkspaceBytes = 64ULL * 1024 * 1024;
constexpr size_t kPrefillIntWorkspaceBytes = 16ULL * 1024 * 1024;

struct FlashInferPrefillWorkspace {
    void *float_workspace = nullptr;
    void *int_workspace = nullptr;
    void *page_locked_int_workspace = nullptr;

    int32_t *d_indices = nullptr;
    int32_t *d_kv_indptr = nullptr;
    int32_t *d_last_page_len = nullptr;
    int32_t *d_qo_indptr = nullptr;
    size_t indices_capacity = 0;
    size_t batch_capacity = 0;

    PrefillPlanInfo plan_info;
    int batch_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int block_size = 0;
    llaisysDataType_t dtype = LLAISYS_DTYPE_F16;
    bool prepared = false;
};

static void release_prefill_workspace(FlashInferPrefillWorkspace *workspace) {
    if (!workspace) return;
    if (workspace->float_workspace) cudaFree(workspace->float_workspace);
    if (workspace->int_workspace) cudaFree(workspace->int_workspace);
    if (workspace->page_locked_int_workspace)
        cudaFreeHost(workspace->page_locked_int_workspace);
    if (workspace->d_indices) cudaFree(workspace->d_indices);
    if (workspace->d_kv_indptr) cudaFree(workspace->d_kv_indptr);
    if (workspace->d_last_page_len) cudaFree(workspace->d_last_page_len);
    if (workspace->d_qo_indptr) cudaFree(workspace->d_qo_indptr);
}

static void ensure_prefill_workspace(FlashInferPrefillWorkspace &workspace) {
    if (!workspace.float_workspace)
        CUDA_CHECK(cudaMalloc(&workspace.float_workspace,
                              kPrefillFloatWorkspaceBytes));
    if (!workspace.int_workspace)
        CUDA_CHECK(cudaMalloc(&workspace.int_workspace,
                              kPrefillIntWorkspaceBytes));
    if (!workspace.page_locked_int_workspace)
        CUDA_CHECK(cudaMallocHost(&workspace.page_locked_int_workspace,
                                  kPrefillIntWorkspaceBytes));
}

static void ensure_prefill_metadata_buffers(
    FlashInferPrefillWorkspace &workspace, size_t num_pages, size_t batch_size) {
    if (num_pages > workspace.indices_capacity) {
        if (workspace.d_indices) CUDA_CHECK(cudaFree(workspace.d_indices));
        CUDA_CHECK(cudaMalloc(&workspace.d_indices,
                              num_pages * sizeof(int32_t)));
        workspace.indices_capacity = num_pages;
    }
    if (batch_size > workspace.batch_capacity) {
        if (workspace.d_kv_indptr) CUDA_CHECK(cudaFree(workspace.d_kv_indptr));
        if (workspace.d_last_page_len)
            CUDA_CHECK(cudaFree(workspace.d_last_page_len));
        if (workspace.d_qo_indptr) CUDA_CHECK(cudaFree(workspace.d_qo_indptr));
        CUDA_CHECK(cudaMalloc(&workspace.d_kv_indptr,
                              (batch_size + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&workspace.d_last_page_len,
                              batch_size * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&workspace.d_qo_indptr,
                              (batch_size + 1) * sizeof(int32_t)));
        workspace.batch_capacity = batch_size;
    }
}

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

template <typename T, uint32_t HEAD_DIM>
static void flashinfer_prefill_run_typed(
    FlashInferPrefillWorkspace &workspace,
    T *output, const T *query,
    const void *k_pool, const void *v_pool,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    using Params = flashinfer::BatchPrefillPagedParams<T, T, T, int32_t>;
    using AttentionVariant =
        flashinfer::DefaultAttention<false, false, false, false>;

    const char *k_layer = static_cast<const char *>(k_pool)
                        + static_cast<size_t>(layer_idx) * pool_layer_stride;
    const char *v_layer = static_cast<const char *>(v_pool)
                        + static_cast<size_t>(layer_idx) * pool_layer_stride;
    int64_t kv_strides[4] = {
        static_cast<int64_t>(pool_block_stride / sizeof(T)),
        static_cast<int64_t>(workspace.num_kv_heads * HEAD_DIM),
        static_cast<int64_t>(HEAD_DIM),
        1,
    };

    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        workspace.num_kv_heads, workspace.block_size, HEAD_DIM,
        workspace.batch_size, QKVLayout::kNHD,
        reinterpret_cast<T *>(const_cast<char *>(k_layer)),
        reinterpret_cast<T *>(const_cast<char *>(v_layer)),
        kv_strides, workspace.d_indices, workspace.d_kv_indptr,
        workspace.d_last_page_len);

    Params params;
    params.q = const_cast<T *>(query);
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = workspace.d_qo_indptr;
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = output;
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = flashinfer::uint_fastdiv(
        workspace.num_heads / workspace.num_kv_heads);
    params.num_qo_heads = workspace.num_heads;
    params.q_stride_n = workspace.num_heads * HEAD_DIM;
    params.q_stride_h = HEAD_DIM;
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        workspace.int_workspace, workspace.plan_info.request_indices_offset);
    params.qo_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        workspace.int_workspace, workspace.plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        workspace.int_workspace, workspace.plan_info.kv_tile_indices_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        workspace.int_workspace, workspace.plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        workspace.int_workspace, workspace.plan_info.kv_chunk_size_ptr_offset);
    params.merge_indptr = nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    params.max_total_num_rows =
        static_cast<uint32_t>(workspace.plan_info.total_num_rows);
    params.padded_batch_size =
        static_cast<uint32_t>(workspace.plan_info.padded_batch_size);
    params.partition_kv = workspace.plan_info.split_kv;

    T *tmp_v = nullptr;
    float *tmp_s = nullptr;
    if (workspace.plan_info.split_kv) {
        params.merge_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
            workspace.int_workspace, workspace.plan_info.merge_indptr_offset);
        tmp_v = flashinfer::GetPtrFromBaseOffset<T>(
            workspace.float_workspace, workspace.plan_info.v_offset);
        tmp_s = flashinfer::GetPtrFromBaseOffset<float>(
            workspace.float_workspace, workspace.plan_info.s_offset);
    }

    cudaError_t status = cudaSuccess;
    DISPATCH_CTA_TILE_Q(workspace.plan_info.cta_tile_q, CTA_TILE_Q, {
        status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM, HEAD_DIM, PosEncodingMode::kNone,
            false, MaskMode::kCausal, AttentionVariant, Params>(
                params, tmp_v, tmp_s, false, nullptr);
    });
    CUDA_CHECK(status);
}

template <typename T>
static void dispatch_prefill_head_dim(
    FlashInferPrefillWorkspace &workspace,
    T *output, const T *query,
    const void *k_pool, const void *v_pool,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale)
{
    switch (workspace.head_dim) {
    case 64:
        flashinfer_prefill_run_typed<T, 64>(
            workspace, output, query, k_pool, v_pool,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        return;
    case 128:
        flashinfer_prefill_run_typed<T, 128>(
            workspace, output, query, k_pool, v_pool,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        return;
    case 256:
        flashinfer_prefill_run_typed<T, 256>(
            workspace, output, query, k_pool, v_pool,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        return;
    default:
        throw std::runtime_error(
            "FlashInfer paged prefill: unsupported head_dim");
    }
}

} // namespace

bool flashinfer_available() { return true; }

void *flashinfer_paged_prefill_workspace_create() {
    return new FlashInferPrefillWorkspace();
}

void flashinfer_paged_prefill_workspace_destroy(void *workspace) {
    auto *prefill = static_cast<FlashInferPrefillWorkspace *>(workspace);
    release_prefill_workspace(prefill);
    delete prefill;
}

bool flashinfer_paged_prefill_supported(
    int num_heads, int num_kv_heads, int head_dim,
    llaisysDataType_t dtype)
{
    const char *disabled = std::getenv("LLAISYS_DISABLE_FLASHINFER_PREFILL");
    if (disabled && std::string(disabled) == "1") return false;
    if (num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0)
        return false;
    // Unlike the decode templates, FlashInfer's paged-prefill kernel carries
    // the GQA group size as a runtime fast-divisor.  Restricting this to the
    // decode specialization set (1/2/4/8) incorrectly rejects models such as
    // Qwen2-1.5B, whose local group size is 12 / 2 = 6.
    bool head_supported = head_dim == 64 || head_dim == 128 ||
                          head_dim == 256;
    bool dtype_supported = dtype == LLAISYS_DTYPE_F16 ||
                           dtype == LLAISYS_DTYPE_BF16;
    return head_supported && dtype_supported;
}

void flashinfer_paged_prefill_prepare(
    void *workspace,
    const int *block_tables, const int *seq_lens, const int *query_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    llaisysDataType_t dtype)
{
    auto *prefill = static_cast<FlashInferPrefillWorkspace *>(workspace);
    if (!prefill || !block_tables || !seq_lens || !query_lens)
        throw std::invalid_argument(
            "FlashInfer paged prefill: null workspace or metadata");
    if (batch_size <= 0 || block_size <= 0 || max_blocks_per_seq <= 0)
        throw std::invalid_argument(
            "FlashInfer paged prefill: invalid batch/page shape");
    if (!flashinfer_paged_prefill_supported(
            num_heads, num_kv_heads, head_dim, dtype))
        throw std::invalid_argument(
            "FlashInfer paged prefill: unsupported attention shape or dtype");

    prefill->prepared = false;
    std::vector<int32_t> indices;
    indices.reserve(static_cast<size_t>(batch_size) * max_blocks_per_seq);
    std::vector<int32_t> kv_indptr(batch_size + 1, 0);
    std::vector<int32_t> qo_indptr(batch_size + 1, 0);
    std::vector<int32_t> last_page_len(batch_size, 0);

    for (int batch = 0; batch < batch_size; ++batch) {
        int seq_len = seq_lens[batch];
        int query_len = query_lens[batch];
        if (seq_len <= 0 || query_len <= 0 || query_len > seq_len)
            throw std::invalid_argument(
                "FlashInfer paged prefill: invalid query/sequence length");
        int num_pages = (seq_len + block_size - 1) / block_size;
        if (num_pages > max_blocks_per_seq)
            throw std::invalid_argument(
                "FlashInfer paged prefill: page table is too narrow");
        kv_indptr[batch + 1] = kv_indptr[batch] + num_pages;
        qo_indptr[batch + 1] = qo_indptr[batch] + query_len;
        last_page_len[batch] = (seq_len - 1) % block_size + 1;
        for (int page = 0; page < num_pages; ++page) {
            int block_id = block_tables[batch * max_blocks_per_seq + page];
            if (block_id < 0)
                throw std::invalid_argument(
                    "FlashInfer paged prefill: negative block id");
            indices.push_back(static_cast<int32_t>(block_id));
        }
    }

    ensure_prefill_workspace(*prefill);
    ensure_prefill_metadata_buffers(*prefill, indices.size(), batch_size);
    CUDA_CHECK(cudaMemcpyAsync(
        prefill->d_indices, indices.data(),
        indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, nullptr));
    CUDA_CHECK(cudaMemcpyAsync(
        prefill->d_kv_indptr, kv_indptr.data(),
        kv_indptr.size() * sizeof(int32_t), cudaMemcpyHostToDevice, nullptr));
    CUDA_CHECK(cudaMemcpyAsync(
        prefill->d_last_page_len, last_page_len.data(),
        last_page_len.size() * sizeof(int32_t), cudaMemcpyHostToDevice, nullptr));
    CUDA_CHECK(cudaMemcpyAsync(
        prefill->d_qo_indptr, qo_indptr.data(),
        qo_indptr.size() * sizeof(int32_t), cudaMemcpyHostToDevice, nullptr));

    PrefillPlanInfo plan_info;
    CUDA_CHECK(flashinfer::PrefillPlan<int32_t>(
        prefill->float_workspace, kPrefillFloatWorkspaceBytes,
        prefill->int_workspace, prefill->page_locked_int_workspace,
        kPrefillIntWorkspaceBytes, plan_info,
        qo_indptr.data(), kv_indptr.data(),
        static_cast<uint32_t>(qo_indptr.back()),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(block_size),
        false, sizeof(__half), -1, -1, false, 0, nullptr));

    prefill->plan_info = plan_info;
    prefill->batch_size = batch_size;
    prefill->num_heads = num_heads;
    prefill->num_kv_heads = num_kv_heads;
    prefill->head_dim = head_dim;
    prefill->block_size = block_size;
    prefill->dtype = dtype;
    prefill->prepared = true;
}

void flashinfer_paged_prefill_run(
    void *workspace,
    void *output, const void *query,
    const void *k_pool, const void *v_pool,
    size_t pool_block_stride, size_t pool_layer_stride,
    int layer_idx, float scale, llaisysDataType_t dtype)
{
    auto *prefill = static_cast<FlashInferPrefillWorkspace *>(workspace);
    if (!prefill || !prefill->prepared)
        throw std::runtime_error(
            "FlashInfer paged prefill: workspace was not prepared");
    if (!output || !query || !k_pool || !v_pool || layer_idx < 0)
        throw std::invalid_argument(
            "FlashInfer paged prefill: invalid tensor pointer or layer");
    if (dtype != prefill->dtype)
        throw std::invalid_argument(
            "FlashInfer paged prefill: dtype differs from prepared plan");

    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        dispatch_prefill_head_dim<__half>(
            *prefill, static_cast<__half *>(output),
            static_cast<const __half *>(query), k_pool, v_pool,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        dispatch_prefill_head_dim<__nv_bfloat16>(
            *prefill, static_cast<__nv_bfloat16 *>(output),
            static_cast<const __nv_bfloat16 *>(query), k_pool, v_pool,
            pool_block_stride, pool_layer_stride, layer_idx, scale);
        return;
    default:
        throw std::runtime_error(
            "FlashInfer paged prefill: unsupported dtype");
    }
}

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

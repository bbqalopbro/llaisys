// ============================================================================
// qwen2.cpp — Qwen2 模型核心实现 (feat/vllm-paged-attention 分支)
// ============================================================================
// 包含:
//   1. 模型结构体 LlaisysQwen2Model (权重/KV-Cache/TP 配置)
//   2. 单请求推理: Infer (argmax) / InferSample (top-k/top-p 采样)
//   3. KV-Cache 快照: save/restore/truncate + 前缀树池
//   4. 权重加载: FP32/FP16/INT8/INT4/AWQ 多格式 + TP 切分
//   5. PagedAttention 批量推理:
//      - BlockAllocator: GPU 显存块分配器 (固定块大小)
//      - PageTable: 每个 slot 维护虚拟→物理块映射
//      - batch_prefill_impl: 逐 token prefill, KV 直接写入 block pool
//      - batch_decode_impl: 多 slot 批量 decode, B 个请求一次 paged_attention
//   6. Per-request sampling: 每个 slot 独立采样参数
//
// 内存布局 (PagedAttention):
//   block_pool_k/v: [num_blocks, nlayer, block_size, nkvh, dh]
//   每个 slot 持有 PageTable → vector<int> block_ids
//   token pos → block_id = block_ids[pos / block_size]
//   token pos → offset    = pos % block_size
// ============================================================================
#include "llaisys/models/qwen2.h"
#include "llaisys/distributed.h"
#include "../../ops/op.hpp"
#include "../../ops/dequantize/op.hpp"
#include "../../ops/self_attention/paged_attention.hpp"
#include "../../utils/types.hpp"
#include "../../distributed/comm.hpp"
#include "../../core/allocator/block_allocator.hpp"
#include "../../core/page_table.hpp"
#include "../../core/context/context.hpp"
#include "../../core/cuda_graph.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <random>
#include <unordered_map>

using namespace llaisys;

// ==========================================
// 1. 辅助工具函数
// ==========================================

static inline tensor_t TO_CPP_TENSOR(llaisysTensor_t t) {
    if (!t) return nullptr;
    return tensor_t(reinterpret_cast<Tensor*>(t), [](Tensor*){});
}

// ==========================================
// 2. 模型结构体定义
// ==========================================

// KV-Cache 快照 (CPU 侧存储)
struct LlaisysQwen2CacheSnapshot {
    int64_t pos;            // 保存时的 current_pos
    size_t nlayer;           // transformer 层数
    size_t pos_bytes;        // 每个位置每个 K/V 的字节数 = local_nkvh * dh * sizeof(float)
    // TP 元信息 (restore 时校验)
    int tp_size = 1;
    int tp_rank = 0;
    // buffers[layer * 2 + 0] = K, buffers[layer * 2 + 1] = V
    // 每个 buffer 大小 = pos * pos_bytes
    std::vector<std::vector<uint8_t>> buffers;
};

// 前缀树节点
struct TrieNode {
    std::unordered_map<int64_t, std::unique_ptr<TrieNode>> children;
    LlaisysQwen2CacheSnapshot* snapshot = nullptr;  // 可能为 null

    ~TrieNode() {
        if (snapshot) {
            delete snapshot;
            snapshot = nullptr;
        }
    }
};

// KV-Cache 前缀树池
struct LlaisysKVCachePool {
    std::unique_ptr<TrieNode> root;

    LlaisysKVCachePool() : root(std::make_unique<TrieNode>()) {}

    void clear() {
        root = std::make_unique<TrieNode>();
    }
};

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device_type;
    int device_id;

    // ── 计算精度 (FP32 / FP16) ──
    // GPU 默认 FP16: 激活/KV-Cache 用半精度, 中间计算保持 FP32
    // CPU 只支持 FP32
    llaisysDataType_t act_dtype;

    // ── TP (Tensor Parallel) 字段 ──
    int tp_size = 1;   // 总并行数
    int tp_rank = 0;   // 当前 rank
    // TP 后的本地维度 (tp_size=1 时等于 meta 原值)
    size_t local_nh;   // = meta.nh / tp_size
    size_t local_nkvh; // = meta.nkvh / tp_size
    size_t local_di;   // = meta.di / tp_size
    // TP 通信句柄 (nullptr 表示单设备 / 不通信)
    llaisys::distributed::comm_t comm = nullptr;
    
    std::vector<tensor_t> resources;
    std::vector<std::vector<tensor_t>> kv_caches;

    // INT8 量化标记
    bool has_quantized = false;

    // AWQ 原生模式标记 (I32 packed qweight 已加载)
    bool has_awq_native = false;

    // Dequantize 临时缓冲区 (按需分配)
    // key = (rows << 32) | cols, value = FP32 buffer
    std::unordered_map<uint64_t, tensor_t> dequant_cache;

    // 缓冲区声明
    tensor_t input_ids_buf;
    tensor_t pos_ids_buf;
    
    tensor_t hidden_states; 
    tensor_t residual;      
    tensor_t norm_out;      
    
    tensor_t q, k, v;       
    tensor_t attn_out;      
    
    tensor_t gate, up, mlp_act; 
    tensor_t logits;        
    tensor_t next_token;    
    tensor_t max_val;       

    int64_t current_pos = 0;
    uint64_t rng_seed = 42;

    // 设备感知内存操作辅助函数
    void memcpyH2D(tensor_t dst, const void* host_src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst->data(), host_src, bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_async(
                dst->data(), host_src, bytes, LLAISYS_MEMCPY_H2D, nullptr);
        }
    }

    void memcpyH2D(void* dev_dst, const void* host_src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dev_dst, host_src, bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_async(
                dev_dst, host_src, bytes, LLAISYS_MEMCPY_H2D, nullptr);
        }
    }

    void memcpyD2H(void* host_dst, tensor_t src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(host_dst, src->data(), bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_sync(
                host_dst, src->data(), bytes, LLAISYS_MEMCPY_D2H);
        }
    }

    void memcpyD2H(void* host_dst, const void* dev_src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(host_dst, dev_src, bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_sync(
                host_dst, dev_src, bytes, LLAISYS_MEMCPY_D2H);
        }
    }

    void memcpyOnDevice(void* dst, const void* src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst, src, bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_async(
                dst, src, bytes, LLAISYS_MEMCPY_D2D, nullptr);
        }
    }

    // TP: 若 tp_size>1 且有 comm, 对 tensor 执行 all-reduce sum
    // TP: 若 tp_size>1 且有 comm, 对 tensor 执行 all-reduce sum
    void allReduceIfTP(tensor_t t, size_t count) {
        if (tp_size > 1 && comm) {
            comm->allReduceSum((float*)t->data(), count);
        }
    }

    LlaisysQwen2Model(const LlaisysQwen2Meta* m, llaisysDeviceType_t dev, int dev_id,
                      int tp_sz = 1, int tp_rk = 0)
        : meta(*m), device_type(dev), device_id(dev_id >= 0 ? dev_id : 0),
          tp_size(tp_sz), tp_rank(tp_rk) {
        // GPU 默认 FP16 激活, CPU 只支持 FP32
        // TP 模式 (tp_size>1) 暂不支持 FP16 allReduce, 强制 FP32
        // 环境变量 LLAISYS_FORCE_FP32=1 可强制 FP32 (用于 benchmark 对比)
        const char* force_fp32 = std::getenv("LLAISYS_FORCE_FP32");
        if (force_fp32 && std::string(force_fp32) == "1") {
            act_dtype = LLAISYS_DTYPE_F32;
        } else if (dev != LLAISYS_DEVICE_CPU && tp_sz <= 1) {
            act_dtype = LLAISYS_DTYPE_F16;
        } else {
            act_dtype = LLAISYS_DTYPE_F32;
        }
        // 验证 TP 维度可整除
        if (tp_size < 1) tp_size = 1;
        if (tp_rank < 0 || tp_rank >= tp_size) tp_rank = 0;
        if (tp_size > 1) {
            if (meta.nh % tp_size != 0 || meta.nkvh % tp_size != 0 || meta.di % tp_size != 0) {
                std::cerr << "[qwen2] ERROR: nh=" << meta.nh << " nkvh=" << meta.nkvh
                          << " di=" << meta.di << " not divisible by tp_size=" << tp_size << std::endl;
                tp_size = 1;
                tp_rank = 0;
            }
        }
        local_nh = meta.nh / tp_size;
        local_nkvh = meta.nkvh / tp_size;
        local_di = meta.di / tp_size;
        weights.in_embed = nullptr;
        weights.out_embed = nullptr;
        weights.out_norm_w = nullptr;
        
        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer]();

        // INT8 量化 scale 数组
        weights.out_embed_scale = nullptr;
        weights.attn_q_w_scale = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_w_scale = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_w_scale = new llaisysTensor_t[meta.nlayer]();
        weights.attn_o_w_scale = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_gate_w_scale = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_up_w_scale = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_down_w_scale = new llaisysTensor_t[meta.nlayer]();

        // AWQ 零点数组
        weights.attn_q_w_qzeros = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_w_qzeros = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_w_qzeros = new llaisysTensor_t[meta.nlayer]();
        weights.attn_o_w_qzeros = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_gate_w_qzeros = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_up_w_qzeros = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_down_w_qzeros = new llaisysTensor_t[meta.nlayer]();

        init_cache();
        init_buffers();
    }

    ~LlaisysQwen2Model() {
        delete[] weights.attn_norm_w; delete[] weights.attn_q_w; delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;    delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;    delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;  delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;    delete[] weights.mlp_down_w;

        delete[] weights.attn_q_w_scale; delete[] weights.attn_k_w_scale;
        delete[] weights.attn_v_w_scale; delete[] weights.attn_o_w_scale;
        delete[] weights.mlp_gate_w_scale; delete[] weights.mlp_up_w_scale;
        delete[] weights.mlp_down_w_scale;
    }

    void init_cache() {
        // TP: 每个 rank 只存 local_nkvh 个 KV head
        // KV-Cache 使用 act_dtype (FP16 时显存减半)
        std::vector<size_t> shape = {meta.maxseq, local_nkvh, meta.dh};
        for (size_t i = 0; i < meta.nlayer; ++i) {
            auto k_c = Tensor::create(shape, act_dtype, device_type, device_id);
            auto v_c = Tensor::create(shape, act_dtype, device_type, device_id);
            kv_caches.push_back({k_c, v_c});
        }
    }

    void init_buffers() {
        input_ids_buf = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
        pos_ids_buf = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
        
        // 激活缓冲区使用 act_dtype (GPU=FP16, CPU=FP32)
        hidden_states = Tensor::create({1, meta.hs}, act_dtype, device_type, device_id);
        residual = Tensor::create({1, meta.hs}, act_dtype, device_type, device_id);
        norm_out = Tensor::create({1, meta.hs}, act_dtype, device_type, device_id);

        // TP: Q/K/V/attn_out 使用 local head 数
        size_t q_dim = local_nh * meta.dh;
        size_t kv_dim = local_nkvh * meta.dh;
        
        q = Tensor::create({1, q_dim}, act_dtype, device_type, device_id);
        k = Tensor::create({1, kv_dim}, act_dtype, device_type, device_id);
        v = Tensor::create({1, kv_dim}, act_dtype, device_type, device_id);
        
        attn_out = Tensor::create({1, local_nh, meta.dh}, act_dtype, device_type, device_id);
        
        // TP: MLP 使用 local_di
        gate = Tensor::create({1, local_di}, act_dtype, device_type, device_id);
        up = Tensor::create({1, local_di}, act_dtype, device_type, device_id);
        mlp_act = Tensor::create({1, local_di}, act_dtype, device_type, device_id);
        
        // logits 始终 FP32 (采样精度要求)
        logits = Tensor::create({1, meta.voc}, LLAISYS_DTYPE_F32, device_type, device_id);
        next_token = Tensor::create({1}, LLAISYS_DTYPE_I32, device_type, device_id);
        max_val = Tensor::create({1}, LLAISYS_DTYPE_F32, device_type, device_id);
    }

    // 获取或创建 dequantize 临时缓冲区 (lazy allocation, keyed by shape)
    tensor_t get_dequant_buf(size_t rows, size_t cols) {
        uint64_t key = ((uint64_t)rows << 32) | (uint64_t)cols;
        auto it = dequant_cache.find(key);
        if (it != dequant_cache.end()) return it->second;
        auto buf = Tensor::create({rows, cols}, LLAISYS_DTYPE_F32, device_type, device_id);
        dequant_cache[key] = buf;
        return buf;
    }

    // 量化感知的 linear 调用: 如果 weight 是 INT8/INT4/AWQ-I32 则先 dequantize
    void linear_maybe_dequant(tensor_t out, tensor_t in,
                              llaisysTensor_t w_handle, llaisysTensor_t scale_handle,
                              llaisysTensor_t bias_handle,
                              llaisysTensor_t qzeros_handle = nullptr) {
        auto w = TO_CPP_TENSOR(w_handle);
        if (!w) {
            fprintf(stderr, "[ERROR] linear_maybe_dequant: weight is NULL\n");
            return;
        }
        auto b = TO_CPP_TENSOR(bias_handle);

        if (w->dtype() == LLAISYS_DTYPE_I32 && scale_handle && qzeros_handle) {
            // AWQ 原生路径: int32 packed → dequantize_awq_int4 → FP32 → linear
            auto sc = TO_CPP_TENSOR(scale_handle);
            auto qz = TO_CPP_TENSOR(qzeros_handle);
            if (!sc || !qz) {
                fprintf(stderr, "[AWQ] ERROR: sc or qz null despite non-null handles\n");
                return;
            }
            // qweight: [in_features, out_packed], out is [out_features, in_features]
            size_t in_features = w->shape()[0];
            size_t out_packed  = w->shape()[1];
            size_t out_features = out_packed * 8;
            size_t num_groups = sc->shape()[0];
            int group_size = (int)(in_features / num_groups);
            auto dq_buf = get_dequant_buf(out_features, in_features);
            ops::dequantize_awq_int4(dq_buf, w, qz, sc, group_size);
            ops::linear(out, in, dq_buf, b);
        } else if (w->dtype() == LLAISYS_DTYPE_I8 && scale_handle) {
            // INT8 路径: dequantize → FP32 → linear
            auto sc = TO_CPP_TENSOR(scale_handle);
            size_t rows = w->shape()[0];
            size_t cols = w->shape()[1];
            auto dq_buf = get_dequant_buf(rows, cols);
            ops::dequantize(dq_buf, w, sc);
            ops::linear(out, in, dq_buf, b);
        } else if (w->dtype() == LLAISYS_DTYPE_U8 && scale_handle) {
            // INT4 packed 路径: dequantize_int4 → FP32 → linear
            auto sc = TO_CPP_TENSOR(scale_handle);
            size_t rows = w->shape()[0];
            size_t packed_cols = w->shape()[1];
            size_t cols = packed_cols * 2;       // 原始列数
            size_t num_groups = sc->shape()[1];  // scale 是 2D: [rows, num_groups]
            int group_size = (int)(cols / num_groups);
            auto dq_buf = get_dequant_buf(rows, cols);
            ops::dequantize_int4(dq_buf, w, sc, group_size);
            ops::linear(out, in, dq_buf, b);
        } else {
            // FP32 / FP16 / BF16 原始路径
            // 如果 weight 是 FP16 而 input 是 FP32, ops::linear 内部
            // 会自动走混合精度路径 (GPU: cuBLAS F16×F16→F32, CPU: cast 累加)
            ops::linear(out, in, w, b);
        }
    }
};

// ==========================================
// 2.5 TP 权重分片辅助
// ==========================================

enum class TpSlice { None, ColDim0, RowDim1 };

// 判断参数名对应的 TP 切分策略
static TpSlice classifyWeight(const std::string& suffix) {
    // Column Parallel (按 out_features / dim 0 切分)
    if (suffix == "self_attn.q_proj.weight" || suffix == "self_attn.q_proj.bias" ||
        suffix == "self_attn.q_proj.weight.scale" || suffix == "self_attn.q_proj.weight.qzeros" ||
        suffix == "self_attn.k_proj.weight" || suffix == "self_attn.k_proj.bias" ||
        suffix == "self_attn.k_proj.weight.scale" || suffix == "self_attn.k_proj.weight.qzeros" ||
        suffix == "self_attn.v_proj.weight" || suffix == "self_attn.v_proj.bias" ||
        suffix == "self_attn.v_proj.weight.scale" || suffix == "self_attn.v_proj.weight.qzeros" ||
        suffix == "mlp.gate_proj.weight" || suffix == "mlp.gate_proj.weight.scale" || suffix == "mlp.gate_proj.weight.qzeros" ||
        suffix == "mlp.up_proj.weight" || suffix == "mlp.up_proj.weight.scale" || suffix == "mlp.up_proj.weight.qzeros") {
        return TpSlice::ColDim0;
    }
    // Row Parallel (按 in_features / dim 1 切分)
    if (suffix == "self_attn.o_proj.weight" || suffix == "self_attn.o_proj.weight.qzeros" ||
        suffix == "mlp.down_proj.weight" || suffix == "mlp.down_proj.weight.qzeros") {
        return TpSlice::RowDim1;
    }
    // 不切分：embedding, norm, lm_head, row-parallel 的 scale
    return TpSlice::None;
}

// 在 CPU 上切分 2D 权重 [rows, cols]，返回切片数据和新 shape
static std::vector<uint8_t> slice2D(const void* data, size_t rows, size_t cols,
                                     size_t elem_size, TpSlice how,
                                     int tp_size, int tp_rank,
                                     size_t& out_rows, size_t& out_cols) {
    if (how == TpSlice::ColDim0) {
        // 按 dim 0 (行) 切分 — 数据连续
        out_rows = rows / tp_size;
        out_cols = cols;
        size_t offset = tp_rank * out_rows * cols * elem_size;
        size_t bytes = out_rows * cols * elem_size;
        std::vector<uint8_t> buf(bytes);
        std::memcpy(buf.data(), (const uint8_t*)data + offset, bytes);
        return buf;
    } else {
        // 按 dim 1 (列) 切分 — 需逐行拷贝
        out_rows = rows;
        out_cols = cols / tp_size;
        size_t row_bytes = out_cols * elem_size;
        size_t offset_cols = tp_rank * out_cols;
        size_t bytes = rows * row_bytes;
        std::vector<uint8_t> buf(bytes);
        for (size_t r = 0; r < rows; ++r) {
            const uint8_t* src = (const uint8_t*)data + (r * cols + offset_cols) * elem_size;
            uint8_t* dst = buf.data() + r * row_bytes;
            std::memcpy(dst, src, row_bytes);
        }
        return buf;
    }
}

// 切分 1D 张量 [size]
static std::vector<uint8_t> slice1D(const void* data, size_t size,
                                     size_t elem_size,
                                     int tp_size, int tp_rank,
                                     size_t& out_size) {
    out_size = size / tp_size;
    size_t offset = tp_rank * out_size * elem_size;
    size_t bytes = out_size * elem_size;
    std::vector<uint8_t> buf(bytes);
    std::memcpy(buf.data(), (const uint8_t*)data + offset, bytes);
    return buf;
}

// ==========================================
// 3. C 接口实现
// ==========================================

extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    if (!meta) return nullptr;
    int dev_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;
    return new LlaisysQwen2Model(meta, device, dev_id);
}

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreateTP(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
                                                             int device_id, int tp_size, int tp_rank) {
    if (!meta) return nullptr;
    return new LlaisysQwen2Model(meta, device, device_id >= 0 ? device_id : 0, tp_size, tp_rank);
}

__export int llaisysQwen2GetTpSize(struct LlaisysQwen2Model * model) {
    if (!model) return 1;
    return model->tp_size;
}

__export int llaisysQwen2GetTpRank(struct LlaisysQwen2Model * model) {
    if (!model) return 0;
    return model->tp_rank;
}

__export void llaisysQwen2SetComm(struct LlaisysQwen2Model * model, llaisysDistComm_t comm_handle) {
    if (!model) return;
    if (comm_handle) {
        // 从 C handle 获取内部 shared_ptr<Comm>
        auto* impl_ptr = (llaisys::distributed::comm_t*)llaisysDistCommGetImplPtr(comm_handle);
        if (impl_ptr) model->comm = *impl_ptr;
    } else {
        model->comm = nullptr;
    }
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    return &model->weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (!model || !token_ids || ntoken == 0) return -1;

    if (ntoken > 1) {
        model->current_pos = 0;
    }

    int64_t output_token = 0;

    for (size_t t = 0; t < ntoken; ++t) {
        int64_t token = token_ids[t];
        int64_t pos = model->current_pos;
        
        // H2D: 将 token 和 position 从 CPU 写入设备缓冲区
        model->memcpyH2D(model->input_ids_buf, &token, sizeof(int64_t));
        model->memcpyH2D(model->pos_ids_buf, &pos, sizeof(int64_t));

        // 1. Embedding
        ops::embedding(model->hidden_states, model->input_ids_buf, TO_CPP_TENSOR(model->weights.in_embed));

        // 2. Transformer Layers
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            // Save residual via pointer swap (zero-cost)
            std::swap(model->residual, model->hidden_states);

            // A. Pre-Norm (read from residual which now holds input)
            ops::rms_norm(model->norm_out, model->residual, TO_CPP_TENSOR(model->weights.attn_norm_w[i]), model->meta.epsilon);

            // B. QKV Linear
            model->linear_maybe_dequant(model->q, model->norm_out, model->weights.attn_q_w[i], model->weights.attn_q_w_scale[i], model->weights.attn_q_b[i], model->weights.attn_q_w_qzeros[i]);
            model->linear_maybe_dequant(model->k, model->norm_out, model->weights.attn_k_w[i], model->weights.attn_k_w_scale[i], model->weights.attn_k_b[i], model->weights.attn_k_w_qzeros[i]);
            model->linear_maybe_dequant(model->v, model->norm_out, model->weights.attn_v_w[i], model->weights.attn_v_w_scale[i], model->weights.attn_v_b[i], model->weights.attn_v_w_qzeros[i]);

            // C. RoPE
            auto q_3d = model->q->reshape({1, model->local_nh, model->meta.dh});
            auto k_3d = model->k->reshape({1, model->local_nkvh, model->meta.dh});
            auto v_3d = model->v->reshape({1, model->local_nkvh, model->meta.dh});

            ops::rope(q_3d, q_3d, model->pos_ids_buf, model->meta.theta);
            ops::rope(k_3d, k_3d, model->pos_ids_buf, model->meta.theta);

            // D. Update KV Cache (字节数根据 act_dtype 计算)
            if (model->current_pos < (int64_t)model->meta.maxseq) {
                size_t bytes = model->local_nkvh * model->meta.dh * llaisys::utils::dsize(model->act_dtype);
                char* k_dst = (char*)model->kv_caches[i][0]->data() + model->current_pos * bytes;
                char* v_dst = (char*)model->kv_caches[i][1]->data() + model->current_pos * bytes;
                model->memcpyOnDevice(k_dst, k_3d->data(), bytes);
                model->memcpyOnDevice(v_dst, v_3d->data(), bytes);
            }

            // E. Self Attention
            float scale = 1.0f / std::sqrt((float)model->meta.dh);
            auto k_slice = model->kv_caches[i][0]->slice(0, 0, model->current_pos + 1);
            auto v_slice = model->kv_caches[i][1]->slice(0, 0, model->current_pos + 1);
            
            ops::self_attention(model->attn_out, q_3d, k_slice, v_slice, scale);

            // F. Output Projection (TP: attn_out 是 local head 视图)
            auto attn_flat = model->attn_out->reshape({1, model->local_nh * model->meta.dh});
            model->linear_maybe_dequant(model->hidden_states, attn_flat, model->weights.attn_o_w[i], model->weights.attn_o_w_scale[i], nullptr, model->weights.attn_o_w_qzeros[i]);

            // TP: O proj 是 row parallel, 需要 all-reduce
            model->allReduceIfTP(model->hidden_states, model->meta.hs);

            // G. Residual Add 1
            ops::add(model->hidden_states, model->hidden_states, model->residual);

            // H. MLP Block
            std::swap(model->residual, model->hidden_states);
            ops::rms_norm(model->norm_out, model->residual, TO_CPP_TENSOR(model->weights.mlp_norm_w[i]), model->meta.epsilon);

            model->linear_maybe_dequant(model->gate, model->norm_out, model->weights.mlp_gate_w[i], model->weights.mlp_gate_w_scale[i], nullptr, model->weights.mlp_gate_w_qzeros[i]);
            model->linear_maybe_dequant(model->up, model->norm_out, model->weights.mlp_up_w[i], model->weights.mlp_up_w_scale[i], nullptr, model->weights.mlp_up_w_qzeros[i]);
            
            ops::swiglu(model->mlp_act, model->gate, model->up);
            
            model->linear_maybe_dequant(model->hidden_states, model->mlp_act, model->weights.mlp_down_w[i], model->weights.mlp_down_w_scale[i], nullptr, model->weights.mlp_down_w_qzeros[i]);

            // TP: down proj 是 row parallel, 需要 all-reduce
            model->allReduceIfTP(model->hidden_states, model->meta.hs);

            // I. Residual Add 2
            ops::add(model->hidden_states, model->hidden_states, model->residual);
        }

        // 4. Final Norm
        ops::rms_norm(model->hidden_states, model->hidden_states, TO_CPP_TENSOR(model->weights.out_norm_w), model->meta.epsilon);

        // 5. LM Head
        model->linear_maybe_dequant(model->logits, model->hidden_states, model->weights.out_embed, model->weights.out_embed_scale, nullptr);

        // 6. Argmax
        auto logits_2d = model->logits->reshape({1, model->meta.voc});
        ops::argmax(model->next_token, model->max_val, logits_2d);

        // D2H: 从设备读取 argmax 结果
        int32_t host_token;
        model->memcpyD2H(&host_token, model->next_token, sizeof(int32_t));
        output_token = host_token;
        
        model->current_pos++;
    }
    return output_token;
}

__export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken,
                                              float temperature, int top_k, float top_p) {
    if (!model || !token_ids || ntoken == 0) return -1;

    if (ntoken > 1) {
        model->current_pos = 0;
    }

    int64_t output_token = 0;
    bool use_greedy = (top_k == 1) || (temperature <= 0.0f);

    for (size_t t = 0; t < ntoken; ++t) {
        int64_t token = token_ids[t];
        int64_t pos = model->current_pos;

        model->memcpyH2D(model->input_ids_buf, &token, sizeof(int64_t));
        model->memcpyH2D(model->pos_ids_buf, &pos, sizeof(int64_t));

        // 1. Embedding
        ops::embedding(model->hidden_states, model->input_ids_buf, TO_CPP_TENSOR(model->weights.in_embed));

        // 2. Transformer Layers
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            std::swap(model->residual, model->hidden_states);
            ops::rms_norm(model->norm_out, model->residual, TO_CPP_TENSOR(model->weights.attn_norm_w[i]), model->meta.epsilon);

            model->linear_maybe_dequant(model->q, model->norm_out, model->weights.attn_q_w[i], model->weights.attn_q_w_scale[i], model->weights.attn_q_b[i], model->weights.attn_q_w_qzeros[i]);
            model->linear_maybe_dequant(model->k, model->norm_out, model->weights.attn_k_w[i], model->weights.attn_k_w_scale[i], model->weights.attn_k_b[i], model->weights.attn_k_w_qzeros[i]);
            model->linear_maybe_dequant(model->v, model->norm_out, model->weights.attn_v_w[i], model->weights.attn_v_w_scale[i], model->weights.attn_v_b[i], model->weights.attn_v_w_qzeros[i]);

            auto q_3d = model->q->reshape({1, model->local_nh, model->meta.dh});
            auto k_3d = model->k->reshape({1, model->local_nkvh, model->meta.dh});
            auto v_3d = model->v->reshape({1, model->local_nkvh, model->meta.dh});

            ops::rope(q_3d, q_3d, model->pos_ids_buf, model->meta.theta);
            ops::rope(k_3d, k_3d, model->pos_ids_buf, model->meta.theta);
            
            //kv cache (字节数根据 act_dtype 计算)
            if (model->current_pos < (int64_t)model->meta.maxseq) {
                size_t bytes = model->local_nkvh * model->meta.dh * llaisys::utils::dsize(model->act_dtype);
                char* k_dst = (char*)model->kv_caches[i][0]->data() + model->current_pos * bytes;
                char* v_dst = (char*)model->kv_caches[i][1]->data() + model->current_pos * bytes;
                model->memcpyOnDevice(k_dst, k_3d->data(), bytes);
                model->memcpyOnDevice(v_dst, v_3d->data(), bytes);
            }

            float scale = 1.0f / std::sqrt((float)model->meta.dh);
            auto k_slice = model->kv_caches[i][0]->slice(0, 0, model->current_pos + 1);
            auto v_slice = model->kv_caches[i][1]->slice(0, 0, model->current_pos + 1);

            ops::self_attention(model->attn_out, q_3d, k_slice, v_slice, scale);

            auto attn_flat = model->attn_out->reshape({1, model->local_nh * model->meta.dh});
            model->linear_maybe_dequant(model->hidden_states, attn_flat, model->weights.attn_o_w[i], model->weights.attn_o_w_scale[i], nullptr, model->weights.attn_o_w_qzeros[i]);
            // TP: O proj row parallel → all-reduce
            model->allReduceIfTP(model->hidden_states, model->meta.hs);
            ops::add(model->hidden_states, model->hidden_states, model->residual);

            std::swap(model->residual, model->hidden_states);
            ops::rms_norm(model->norm_out, model->residual, TO_CPP_TENSOR(model->weights.mlp_norm_w[i]), model->meta.epsilon);
            model->linear_maybe_dequant(model->gate, model->norm_out, model->weights.mlp_gate_w[i], model->weights.mlp_gate_w_scale[i], nullptr, model->weights.mlp_gate_w_qzeros[i]);
            model->linear_maybe_dequant(model->up, model->norm_out, model->weights.mlp_up_w[i], model->weights.mlp_up_w_scale[i], nullptr, model->weights.mlp_up_w_qzeros[i]);
            ops::swiglu(model->mlp_act, model->gate, model->up);
            model->linear_maybe_dequant(model->hidden_states, model->mlp_act, model->weights.mlp_down_w[i], model->weights.mlp_down_w_scale[i], nullptr, model->weights.mlp_down_w_qzeros[i]);
            // TP: down proj row parallel → all-reduce
            model->allReduceIfTP(model->hidden_states, model->meta.hs);
            ops::add(model->hidden_states, model->hidden_states, model->residual);
        }

        // 4. Final Norm
        ops::rms_norm(model->hidden_states, model->hidden_states, TO_CPP_TENSOR(model->weights.out_norm_w), model->meta.epsilon);

        // 5. LM Head
        model->linear_maybe_dequant(model->logits, model->hidden_states, model->weights.out_embed, model->weights.out_embed_scale, nullptr);

        // 6. Sampling or Argmax
        auto logits_2d = model->logits->reshape({1, model->meta.voc});
        if (use_greedy) {
            ops::argmax(model->next_token, model->max_val, logits_2d);
        } else {
            ops::sample(model->next_token, logits_2d, temperature, top_k, top_p, model->rng_seed++);
        }

        int32_t host_token;
        model->memcpyD2H(&host_token, model->next_token, sizeof(int32_t));
        output_token = host_token;

        model->current_pos++;
    }
    return output_token;
}

__export void llaisysQwen2ResetCache(struct LlaisysQwen2Model * model) {
    if (!model) return;
    model->current_pos = 0;
}

__export void llaisysQwen2LoadWeightByName(struct LlaisysQwen2Model* model, const char* name, void* data, int ndim, int64_t* shape, int dtype) {
    if (!model) return;

    std::vector<size_t> shape_vec;
    size_t numel = 1;
    for (int i = 0; i < ndim; ++i) {
        shape_vec.push_back((size_t)shape[i]);
        numel *= shape[i];
    }

    // 如果加载了 INT8 或 U8 (INT4 packed) 类型的权重，标记模型为量化模式
    if ((llaisysDataType_t)dtype == LLAISYS_DTYPE_I8 || (llaisysDataType_t)dtype == LLAISYS_DTYPE_U8) {
        model->has_quantized = true;
    }
    
    std::string key(name);
    size_t elem_size = llaisys::utils::dsize((llaisysDataType_t)dtype);

    // AWQ: qweight loaded as I32 with .weight suffix → mark as AWQ native quantized
    if ((llaisysDataType_t)dtype == LLAISYS_DTYPE_I32 &&
        key.size() > 7 && key.substr(key.size() - 7) == ".weight") {
        model->has_quantized = true;
        model->has_awq_native = true;
    }

    // ── 判断 TP 切分策略 ──
    TpSlice how = TpSlice::None;
    if (model->tp_size > 1) {
        // 分层权重: 提取 suffix 用于分类
        if (key.find("model.layers.") == 0) {
            size_t first_dot = 13;
            size_t second_dot = key.find('.', first_dot);
            std::string suffix = key.substr(second_dot + 1);
            how = classifyWeight(suffix);

            // AWQ native mode: tensor layouts are transposed vs standard convention.
            // qweight/qzeros: [in, out_packed], scales: [ngroups, out]
            // Standard:       [out, in],        scales: [out, ngroups]
            // → invert ColDim0 ↔ RowDim1 for all AWQ tensors (.weight, .qzeros, .scale)
            if (model->has_awq_native && how != TpSlice::None &&
                (suffix.find(".weight") != std::string::npos)) {
                how = (how == TpSlice::ColDim0) ? TpSlice::RowDim1 : TpSlice::ColDim0;
            }
            // AWQ row-parallel scales: non-AWQ leaves these as None since out_features
            // isn't split, but AWQ scales are [ngroups, out_features] and groups (along
            // in_features) must be split for row-parallel projections.
            if (model->has_awq_native && how == TpSlice::None &&
                (suffix == "self_attn.o_proj.weight.scale" ||
                 suffix == "mlp.down_proj.weight.scale")) {
                how = TpSlice::ColDim0;  // slice dim0 (ngroups)
            }
        }
        // 顶层权重 (embed, norm, lm_head) → TpSlice::None (不切分，全量复制)
    }

    // ── 根据 TP 策略创建 tensor (可能先在 CPU 切片) ──
    tensor_t tensor;
    if (how == TpSlice::None || model->tp_size <= 1) {
        // 无切分: 原始行为
        tensor = llaisys::Tensor::create(shape_vec, (llaisysDataType_t)dtype, model->device_type, model->device_id);
        tensor->load(data);
    } else if (ndim == 2) {
        // 2D 权重切分
        size_t rows = shape_vec[0], cols = shape_vec[1];
        size_t out_rows, out_cols;
        auto sliced = slice2D(data, rows, cols, elem_size, how,
                              model->tp_size, model->tp_rank, out_rows, out_cols);
        std::vector<size_t> new_shape = {out_rows, out_cols};
        tensor = llaisys::Tensor::create(new_shape, (llaisysDataType_t)dtype, model->device_type, model->device_id);
        tensor->load(sliced.data());
    } else if (ndim == 1 && how == TpSlice::ColDim0) {
        // 1D (bias / scale) + column parallel → 切分
        size_t out_size;
        auto sliced = slice1D(data, shape_vec[0], elem_size,
                              model->tp_size, model->tp_rank, out_size);
        std::vector<size_t> new_shape = {out_size};
        tensor = llaisys::Tensor::create(new_shape, (llaisysDataType_t)dtype, model->device_type, model->device_id);
        tensor->load(sliced.data());
    } else {
        // 1D + row parallel (scale 不切分) 或其他 → 不切分
        tensor = llaisys::Tensor::create(shape_vec, (llaisysDataType_t)dtype, model->device_type, model->device_id);
        tensor->load(data);
    }

    model->resources.push_back(tensor);
    llaisysTensor_t t_handle = reinterpret_cast<llaisysTensor_t>(tensor.get()); 

    // ── 处理 .scale 后缀 (量化 per-channel scale) ──
    if (key == "lm_head.weight.scale") { model->weights.out_embed_scale = t_handle; return; }

    // 顶层权重
    if (key == "model.embed_tokens.weight") { model->weights.in_embed = t_handle; return; }
    if (key == "model.norm.weight") { model->weights.out_norm_w = t_handle; return; }
    if (key == "lm_head.weight") { model->weights.out_embed = t_handle; return; }

    // 分层权重
    if (key.find("model.layers.") == 0) {
        size_t first_dot = 13;
        size_t second_dot = key.find('.', first_dot);
        int layer_idx = std::stoi(key.substr(first_dot, second_dot - first_dot));
        std::string suffix = key.substr(second_dot + 1);
        
        // 普通权重
        if (suffix == "input_layernorm.weight") model->weights.attn_norm_w[layer_idx] = t_handle;
        else if (suffix == "post_attention_layernorm.weight") model->weights.mlp_norm_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.q_proj.weight") model->weights.attn_q_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.k_proj.weight") model->weights.attn_k_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.v_proj.weight") model->weights.attn_v_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.o_proj.weight") model->weights.attn_o_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.q_proj.bias") model->weights.attn_q_b[layer_idx] = t_handle;
        else if (suffix == "self_attn.k_proj.bias") model->weights.attn_k_b[layer_idx] = t_handle;
        else if (suffix == "self_attn.v_proj.bias") model->weights.attn_v_b[layer_idx] = t_handle;
        else if (suffix == "mlp.gate_proj.weight") model->weights.mlp_gate_w[layer_idx] = t_handle;
        else if (suffix == "mlp.up_proj.weight") model->weights.mlp_up_w[layer_idx] = t_handle;
        else if (suffix == "mlp.down_proj.weight") model->weights.mlp_down_w[layer_idx] = t_handle;
        // 量化 scale
        else if (suffix == "self_attn.q_proj.weight.scale") model->weights.attn_q_w_scale[layer_idx] = t_handle;
        else if (suffix == "self_attn.k_proj.weight.scale") model->weights.attn_k_w_scale[layer_idx] = t_handle;
        else if (suffix == "self_attn.v_proj.weight.scale") model->weights.attn_v_w_scale[layer_idx] = t_handle;
        else if (suffix == "self_attn.o_proj.weight.scale") model->weights.attn_o_w_scale[layer_idx] = t_handle;
        else if (suffix == "mlp.gate_proj.weight.scale") model->weights.mlp_gate_w_scale[layer_idx] = t_handle;
        else if (suffix == "mlp.up_proj.weight.scale") model->weights.mlp_up_w_scale[layer_idx] = t_handle;
        else if (suffix == "mlp.down_proj.weight.scale") model->weights.mlp_down_w_scale[layer_idx] = t_handle;
        // AWQ qzeros
        else if (suffix == "self_attn.q_proj.weight.qzeros") model->weights.attn_q_w_qzeros[layer_idx] = t_handle;
        else if (suffix == "self_attn.k_proj.weight.qzeros") model->weights.attn_k_w_qzeros[layer_idx] = t_handle;
        else if (suffix == "self_attn.v_proj.weight.qzeros") model->weights.attn_v_w_qzeros[layer_idx] = t_handle;
        else if (suffix == "self_attn.o_proj.weight.qzeros") model->weights.attn_o_w_qzeros[layer_idx] = t_handle;
        else if (suffix == "mlp.gate_proj.weight.qzeros") model->weights.mlp_gate_w_qzeros[layer_idx] = t_handle;
        else if (suffix == "mlp.up_proj.weight.qzeros") model->weights.mlp_up_w_qzeros[layer_idx] = t_handle;
        else if (suffix == "mlp.down_proj.weight.qzeros") model->weights.mlp_down_w_qzeros[layer_idx] = t_handle;
    }
}

// ==========================================
// 4. KV-Cache 高级接口
// ==========================================

__export struct LlaisysQwen2CacheSnapshot *llaisysQwen2SaveCache(struct LlaisysQwen2Model * model) {
    if (!model || model->current_pos <= 0) return nullptr;

    auto* snap = new LlaisysQwen2CacheSnapshot();
    snap->pos = model->current_pos;
    snap->nlayer = model->meta.nlayer;
    snap->pos_bytes = model->local_nkvh * model->meta.dh * llaisys::utils::dsize(model->act_dtype);
    snap->tp_size = model->tp_size;
    snap->tp_rank = model->tp_rank;

    size_t total_bytes_per_buf = snap->pos * snap->pos_bytes;
    snap->buffers.resize(snap->nlayer * 2);

    for (size_t i = 0; i < snap->nlayer; ++i) {
        for (size_t kv = 0; kv < 2; ++kv) {
            size_t idx = i * 2 + kv;
            snap->buffers[idx].resize(total_bytes_per_buf);
            // 从设备拷贝到 CPU
            model->memcpyD2H(snap->buffers[idx].data(), model->kv_caches[i][kv], total_bytes_per_buf);
        }
    }

    return snap;
}

__export void llaisysQwen2RestoreCache(struct LlaisysQwen2Model * model, struct LlaisysQwen2CacheSnapshot * snapshot) {
    if (!model || !snapshot) return;
    if (snapshot->nlayer != model->meta.nlayer) return;
    // TP 兼容性校验: snapshot 必须与当前模型的 TP 配置匹配
    if (snapshot->tp_size != model->tp_size || snapshot->tp_rank != model->tp_rank) {
        std::cerr << "[qwen2] RestoreCache: TP mismatch (snapshot tp_size=" << snapshot->tp_size
                  << " tp_rank=" << snapshot->tp_rank << " vs model tp_size=" << model->tp_size
                  << " tp_rank=" << model->tp_rank << ")" << std::endl;
        return;
    }

    model->current_pos = snapshot->pos;
    size_t total_bytes = snapshot->pos * snapshot->pos_bytes;

    for (size_t i = 0; i < snapshot->nlayer; ++i) {
        for (size_t kv = 0; kv < 2; ++kv) {
            size_t idx = i * 2 + kv;
            // 从 CPU 拷贝回设备
            model->memcpyH2D(model->kv_caches[i][kv], snapshot->buffers[idx].data(), total_bytes);
        }
    }
}

__export void llaisysQwen2TruncateCache(struct LlaisysQwen2Model * model, int64_t pos) {
    if (!model) return;
    if (pos < 0) pos = 0;
    if (pos > model->current_pos) pos = model->current_pos;
    model->current_pos = pos;
}

__export int64_t llaisysQwen2GetCachePos(struct LlaisysQwen2Model * model) {
    if (!model) return 0;
    return model->current_pos;
}

__export void llaisysQwen2DestroyCacheSnapshot(struct LlaisysQwen2CacheSnapshot * snapshot) {
    if (snapshot) delete snapshot;
}

// ==========================================
// 5. 前缀树 KV-Cache 池
// ==========================================

__export struct LlaisysKVCachePool *llaisysKVCachePoolCreate(void) {
    return new LlaisysKVCachePool();
}

__export void llaisysKVCachePoolDestroy(struct LlaisysKVCachePool * pool) {
    if (pool) delete pool;
}

__export void llaisysKVCachePoolInsert(struct LlaisysKVCachePool * pool, int64_t * tokens, size_t len, struct LlaisysQwen2CacheSnapshot * snapshot) {
    if (!pool || !tokens || len == 0 || !snapshot) return;

    TrieNode* node = pool->root.get();
    for (size_t i = 0; i < len; ++i) {
        int64_t tok = tokens[i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) {
            node->children[tok] = std::make_unique<TrieNode>();
        }
        node = node->children[tok].get();
    }

    // 替换已有快照
    if (node->snapshot) {
        delete node->snapshot;
    }
    node->snapshot = snapshot;  // 转移所有权
}

__export struct LlaisysQwen2CacheSnapshot *llaisysKVCachePoolLookup(struct LlaisysKVCachePool * pool, int64_t * tokens, size_t len, size_t * match_len) {
    if (!pool || !tokens || len == 0) {
        if (match_len) *match_len = 0;
        return nullptr;
    }

    TrieNode* node = pool->root.get();
    LlaisysQwen2CacheSnapshot* best = nullptr;
    size_t best_len = 0;

    for (size_t i = 0; i < len; ++i) {
        int64_t tok = tokens[i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) break;

        node = it->second.get();
        if (node->snapshot) {
            best = node->snapshot;
            best_len = i + 1;
        }
    }

    if (match_len) *match_len = best_len;
    return best;
}

__export void llaisysKVCachePoolClear(struct LlaisysKVCachePool * pool) {
    if (!pool) return;
    pool->clear();
}

__export int llaisysQwen2IsQuantized(struct LlaisysQwen2Model * model) {
    if (!model) return 0;
    return model->has_quantized ? 1 : 0;
}

// ==========================================
// 6. PagedAttention 批量推理上下文
// ==========================================
// vLLM 风格的 Paged KV-Cache 管理:
//   - BlockAllocator: 共享 GPU 显存池, 按固定大小块分配
//   - PageTable: 每个 slot 维护自己的 block 映射 (虚拟 token 位置 → 物理块)
//   - BatchSlot: 单个请求的状态 (page_table + current_pos)
//   - BatchContext: 管理所有 slot + 共享 block_allocator + 批量缓冲区
// 
// 与 Phase 3 contiguous KV-Cache 的关键区别:
//   Phase 3: kv_caches[layer] = [maxseq, nkvh, dh] — 每个模型一个大连续数组
//   Paged:   block_pool = [num_blocks * block_size, nkvh, dh] — 共享池, 按需分配
//   优势:    多请求间共享显存 + 无需预留 maxseq 空间 + 可动态扩缩

// 单个 slot (请求) 的 KV-Cache 状态
struct BatchSlot {
    llaisys::core::PageTable page_table;  // 虚拟→物理块映射
    int64_t current_pos = 0;              // 该 slot 已写入的 token 数
    bool active = false;                  // 是否有活跃请求

    void init(int block_size) {
        page_table = llaisys::core::PageTable(block_size);
        current_pos = 0;
        active = false;
    }

    // 释放该 slot 占用的所有 block, 归还给 allocator
    void reset(llaisys::core::BlockAllocator &alloc) {
        page_table.release_all(alloc);
        current_pos = 0;
        active = false;
    }
};

// 批量推理上下文: 管理 max_batch_size 个 slot + 共享 block pool
struct LlaisysQwen2BatchContext {
    LlaisysQwen2Model* model;
    size_t max_batch_size;      // 最大并发请求数
    size_t max_seq_per_slot;    // 每个 slot 最大序列长度
    int block_size;             // 每个 block 包含的 token 数 (默认 16)
    std::vector<BatchSlot> slots;  // 所有 slot

    // 共享 block 分配器: 所有 slot 从同一个池分配/释放块
    // block_pool 是一整块 GPU 显存, 按 (block_id, layer, offset) 索引
    std::unique_ptr<llaisys::core::BlockAllocator> block_allocator;

    // CUDA Graph 加速 (可选)
    llaisys::core::CUDAGraphDecodeSession cuda_graph_session;

    // Batch 缓冲区: 按 max_batch_size 预分配, decode 时 slice 到实际 B
    tensor_t batch_input_ids;
    tensor_t batch_pos_ids;
    tensor_t batch_hidden;
    tensor_t batch_residual;
    tensor_t batch_norm_out;
    tensor_t batch_q;
    tensor_t batch_k;
    tensor_t batch_v;
    tensor_t batch_attn_out;
    tensor_t batch_gate;
    tensor_t batch_up;
    tensor_t batch_mlp_act;
    tensor_t batch_logits;

    // 单请求临时缓冲区
    tensor_t single_logits;
    tensor_t single_next_token;
    tensor_t single_max_val;

    LlaisysQwen2BatchContext(LlaisysQwen2Model* m, size_t max_bs, size_t max_seq = 0)
        : model(m), max_batch_size(max_bs), block_size(16)
    {
        auto& meta = model->meta;
        auto dev = model->device_type;
        auto dev_id = model->device_id;
        size_t q_dim = model->local_nh * meta.dh;
        size_t kv_dim = model->local_nkvh * meta.dh;

        max_seq_per_slot = (max_seq > 0 && max_seq <= meta.maxseq) ? max_seq : std::min((size_t)2048, meta.maxseq);

        // Block allocator: enough blocks for all slots at max capacity + headroom
        size_t max_total_tokens = max_bs * max_seq_per_slot;
        size_t num_blocks = (max_total_tokens + block_size - 1) / block_size + max_bs;

        // elem_size 使用 act_dtype 的字节数 (FP16=2, FP32=4)
        llaisys::core::BlockAllocatorConfig cfg = {
            num_blocks, (size_t)block_size, meta.nlayer,
            model->local_nkvh, meta.dh, llaisys::utils::dsize(model->act_dtype)
        };

        if (dev != LLAISYS_DEVICE_CPU) {
            llaisys::core::context().setDevice(dev, dev_id);
        }
        const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(dev);
        block_allocator = std::make_unique<llaisys::core::BlockAllocator>(cfg, api);

        slots.resize(max_bs);
        for (size_t i = 0; i < max_bs; ++i) {
            slots[i].init(block_size);
        }

        batch_input_ids = Tensor::create({max_bs}, LLAISYS_DTYPE_I64, dev, dev_id);
        batch_pos_ids   = Tensor::create({max_bs}, LLAISYS_DTYPE_I64, dev, dev_id);
        // 批量激活缓冲区使用 act_dtype (FP16/FP32)
        auto adtype = model->act_dtype;
        batch_hidden    = Tensor::create({max_bs, meta.hs}, adtype, dev, dev_id);
        batch_residual  = Tensor::create({max_bs, meta.hs}, adtype, dev, dev_id);
        batch_norm_out  = Tensor::create({max_bs, meta.hs}, adtype, dev, dev_id);
        batch_q         = Tensor::create({max_bs, q_dim}, adtype, dev, dev_id);
        batch_k         = Tensor::create({max_bs, kv_dim}, adtype, dev, dev_id);
        batch_v         = Tensor::create({max_bs, kv_dim}, adtype, dev, dev_id);
        batch_attn_out  = Tensor::create({max_bs, model->local_nh * meta.dh}, adtype, dev, dev_id);
        batch_gate      = Tensor::create({max_bs, model->local_di}, adtype, dev, dev_id);
        batch_up        = Tensor::create({max_bs, model->local_di}, adtype, dev, dev_id);
        batch_mlp_act   = Tensor::create({max_bs, model->local_di}, adtype, dev, dev_id);
        // logits 始终 FP32 (采样精度)
        batch_logits    = Tensor::create({max_bs, meta.voc}, LLAISYS_DTYPE_F32, dev, dev_id);

        single_logits     = Tensor::create({1, meta.voc}, LLAISYS_DTYPE_F32, dev, dev_id);
        single_next_token = Tensor::create({1}, LLAISYS_DTYPE_I32, dev, dev_id);
        single_max_val    = Tensor::create({1}, LLAISYS_DTYPE_F32, dev, dev_id);
    }

    void linear_maybe_dequant(tensor_t out, tensor_t in,
                              llaisysTensor_t w_handle, llaisysTensor_t scale_handle,
                              llaisysTensor_t bias_handle,
                              llaisysTensor_t qzeros_handle = nullptr) {
        model->linear_maybe_dequant(out, in, w_handle, scale_handle, bias_handle, qzeros_handle);
    }
};

// ── Prefill: 逐 token 处理 prompt, KV 直接写入 block pool ─────
// 注意: 当前实现是逐 token prefill (性能瓶颈!)
// 优化方向: batch prefill — 一次处理所有 prompt token 的 QKV 投影
//          只在最后一个 token 计算 logits

static int64_t batch_prefill_impl(LlaisysQwen2BatchContext* ctx, size_t slot_id,
                                   int64_t* token_ids, size_t ntoken,
                                   float temperature, int top_k, float top_p) {
    auto* model = ctx->model;
    auto& slot = ctx->slots[slot_id];
    auto& alloc = *ctx->block_allocator;
    auto& meta = model->meta;
    bool use_greedy = (top_k == 1) || (temperature <= 0.0f);

    size_t kv_dim = model->local_nkvh * meta.dh;
    size_t kv_bytes = kv_dim * llaisys::utils::dsize(model->act_dtype);
    float scale = 1.0f / std::sqrt((float)meta.dh);
    int64_t output_token = 0;

    // Pre-allocate enough blocks for the entire prompt
    size_t blocks_needed = (ntoken + alloc.block_size() - 1) / alloc.block_size();
    for (size_t bi = 0; bi < blocks_needed; ++bi) {
        if (slot.page_table.needs_new_block()) {
            int bid = alloc.alloc();
            if (bid < 0) {
                std::cerr << "[qwen2] chunked prefill: block pool exhausted" << std::endl;
                break;
            }
            slot.page_table.append_block(bid);
        }
        for (size_t tok = 0; tok < alloc.block_size(); ++tok)
            slot.page_table.inc_num_tokens();
    }
    slot.page_table.set_num_tokens(0);

    // Process tokens one at a time, writing KV directly to block pool
    for (size_t t = 0; t < ntoken; ++t) {
        int64_t token = token_ids[t];
        int64_t pos = static_cast<int64_t>(t);

        model->memcpyH2D(model->input_ids_buf, &token, sizeof(int64_t));
        model->memcpyH2D(model->pos_ids_buf, &pos, sizeof(int64_t));

        // Embedding
        ops::embedding(model->hidden_states, model->input_ids_buf,
                       TO_CPP_TENSOR(model->weights.in_embed));

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            std::swap(model->residual, model->hidden_states);

            // Pre-Norm
            ops::rms_norm(model->norm_out, model->residual,
                         TO_CPP_TENSOR(model->weights.attn_norm_w[layer]), meta.epsilon);

            // QKV Linear
            model->linear_maybe_dequant(model->q, model->norm_out,
                model->weights.attn_q_w[layer], model->weights.attn_q_w_scale[layer],
                model->weights.attn_q_b[layer], model->weights.attn_q_w_qzeros[layer]);
            model->linear_maybe_dequant(model->k, model->norm_out,
                model->weights.attn_k_w[layer], model->weights.attn_k_w_scale[layer],
                model->weights.attn_k_b[layer], model->weights.attn_k_w_qzeros[layer]);
            model->linear_maybe_dequant(model->v, model->norm_out,
                model->weights.attn_v_w[layer], model->weights.attn_v_w_scale[layer],
                model->weights.attn_v_b[layer], model->weights.attn_v_w_qzeros[layer]);

            // RoPE
            auto q_3d = model->q->reshape({1, model->local_nh, meta.dh});
            auto k_3d = model->k->reshape({1, model->local_nkvh, meta.dh});
            auto v_3d = model->v->reshape({1, model->local_nkvh, meta.dh});
            ops::rope(q_3d, q_3d, model->pos_ids_buf, meta.theta);
            ops::rope(k_3d, k_3d, model->pos_ids_buf, meta.theta);

            // Write KV directly to block pool (按字节寻址, 兼容 FP16/FP32)
            int bid = slot.page_table.get_block_for_token(static_cast<int>(t));
            int off = slot.page_table.get_offset_in_block(static_cast<int>(t));
            char* k_dst = (char*)alloc.get_k_ptr(bid, layer)
                           + (size_t)off * kv_bytes;
            char* v_dst = (char*)alloc.get_v_ptr(bid, layer)
                           + (size_t)off * kv_bytes;
            model->memcpyOnDevice(k_dst, k_3d->data(), kv_bytes);
            model->memcpyOnDevice(v_dst, v_3d->data(), kv_bytes);

            // Paged Attention over all tokens seen so far
            int seq_len_so_far = static_cast<int>(t + 1);
            int num_blocks_so_far = (seq_len_so_far + ctx->block_size - 1) / ctx->block_size;

            std::vector<int> bt(num_blocks_so_far);
            for (int j = 0; j < num_blocks_so_far; ++j)
                bt[j] = slot.page_table.block_ids()[j];

            llaisys::ops::paged_attention(
                model->attn_out->data(),
                q_3d->data(),
                alloc.pool_k_raw(), alloc.pool_v_raw(),
                bt.data(), &seq_len_so_far,
                1, static_cast<int>(model->local_nh),
                static_cast<int>(model->local_nkvh),
                static_cast<int>(meta.dh),
                ctx->block_size, num_blocks_so_far,
                alloc.block_stride(), alloc.layer_stride(),
                static_cast<int>(layer), scale,
                model->device_type,
                llaisys::ops::KVQuantMode::FP32,
                model->act_dtype);

            // O Projection
            auto attn_flat = model->attn_out->reshape({1, model->local_nh * meta.dh});
            model->linear_maybe_dequant(model->hidden_states, attn_flat,
                model->weights.attn_o_w[layer], model->weights.attn_o_w_scale[layer],
                nullptr, model->weights.attn_o_w_qzeros[layer]);
            model->allReduceIfTP(model->hidden_states, meta.hs);

            // Residual Add 1
            ops::add(model->hidden_states, model->hidden_states, model->residual);

            // MLP Block
            std::swap(model->residual, model->hidden_states);
            ops::rms_norm(model->norm_out, model->residual,
                         TO_CPP_TENSOR(model->weights.mlp_norm_w[layer]), meta.epsilon);
            model->linear_maybe_dequant(model->gate, model->norm_out,
                model->weights.mlp_gate_w[layer], model->weights.mlp_gate_w_scale[layer],
                nullptr, model->weights.mlp_gate_w_qzeros[layer]);
            model->linear_maybe_dequant(model->up, model->norm_out,
                model->weights.mlp_up_w[layer], model->weights.mlp_up_w_scale[layer],
                nullptr, model->weights.mlp_up_w_qzeros[layer]);
            ops::swiglu(model->mlp_act, model->gate, model->up);
            model->linear_maybe_dequant(model->hidden_states, model->mlp_act,
                model->weights.mlp_down_w[layer], model->weights.mlp_down_w_scale[layer],
                nullptr, model->weights.mlp_down_w_qzeros[layer]);
            model->allReduceIfTP(model->hidden_states, meta.hs);

            // Residual Add 2
            ops::add(model->hidden_states, model->hidden_states, model->residual);
        }

        // Only compute logits for the last token
        if (t == ntoken - 1) {
            ops::rms_norm(model->hidden_states, model->hidden_states,
                         TO_CPP_TENSOR(model->weights.out_norm_w), meta.epsilon);
            model->linear_maybe_dequant(model->logits, model->hidden_states,
                model->weights.out_embed, model->weights.out_embed_scale, nullptr);

            auto logits_2d = model->logits->reshape({1, meta.voc});
            if (use_greedy) {
                ops::argmax(model->next_token, model->max_val, logits_2d);
            } else {
                ops::sample(model->next_token, logits_2d,
                           temperature, top_k, top_p, model->rng_seed++);
            }

            int32_t host_token;
            model->memcpyD2H(&host_token, model->next_token, sizeof(int32_t));
            output_token = host_token;
        }

        slot.page_table.inc_num_tokens();
    }

    slot.current_pos = static_cast<int64_t>(ntoken);
    slot.active = true;
    return output_token;
}

// ── 批量 Decode: B 个 slot 并行处理一步 ───────────────────────────
// 流程:
//   1. 为每个 slot 分配新 block (如果当前 block 已满)
//   2. 构建 block_tables[B * max_blocks] + seq_lens[B] 传给 paged_attention
//   3. 整 batch 一次 Embedding → Transformer Layers → LM Head
//   4. 每层 attention 调用 paged_attention() 同时处理所有 B 个序列
//   5. 逐 slot argmax/sample, 更新 page_table + current_pos

static void batch_decode_impl(LlaisysQwen2BatchContext* ctx,
                               size_t* active_slots, size_t num_active,
                               int64_t* current_tokens,
                               float temperature, int top_k, float top_p,
                               int64_t* output_tokens) {
    if (num_active == 0) return;

    auto* model = ctx->model;
    auto& meta = model->meta;
    auto& alloc = *ctx->block_allocator;
    size_t B = num_active;
    size_t kv_dim = model->local_nkvh * meta.dh;
    size_t kv_bytes = kv_dim * llaisys::utils::dsize(model->act_dtype);
    bool use_greedy = (top_k == 1) || (temperature <= 0.0f);

    // ── 0. Allocate new blocks for current positions (once, before layer loop) ──
    for (size_t i = 0; i < B; ++i) {
        auto& slot = ctx->slots[active_slots[i]];
        if (slot.page_table.needs_new_block()) {
            int bid = alloc.alloc();
            if (bid < 0) {
                std::cerr << "[qwen2] decode: block pool exhausted for slot "
                          << active_slots[i] << std::endl;
                return;
            }
            slot.page_table.append_block(bid);
        }
    }

    // ── 1. Build batch block tables and seq_lens for paged attention ──
    int max_blocks_per_seq = 0;
    for (size_t i = 0; i < B; ++i) {
        auto& slot = ctx->slots[active_slots[i]];
        max_blocks_per_seq = std::max(max_blocks_per_seq, slot.page_table.num_blocks());
    }

    std::vector<int> block_tables(B * max_blocks_per_seq, 0);
    std::vector<int> seq_lens(B);
    for (size_t i = 0; i < B; ++i) {
        auto& slot = ctx->slots[active_slots[i]];
        seq_lens[i] = static_cast<int>(slot.current_pos + 1);
        for (int j = 0; j < slot.page_table.num_blocks(); ++j)
            block_tables[i * max_blocks_per_seq + j] = slot.page_table.block_ids()[j];
    }

    // ── 2. 准备输入 (H2D) ──
    auto b_input  = ctx->batch_input_ids->slice(0, 0, B);
    auto b_pos    = ctx->batch_pos_ids->slice(0, 0, B);
    auto b_hidden = ctx->batch_hidden->slice(0, 0, B);
    auto b_resid  = ctx->batch_residual->slice(0, 0, B);
    auto b_norm   = ctx->batch_norm_out->slice(0, 0, B);
    auto b_q      = ctx->batch_q->slice(0, 0, B);
    auto b_k      = ctx->batch_k->slice(0, 0, B);
    auto b_v      = ctx->batch_v->slice(0, 0, B);
    auto b_attn   = ctx->batch_attn_out->slice(0, 0, B);
    auto b_gate   = ctx->batch_gate->slice(0, 0, B);
    auto b_up     = ctx->batch_up->slice(0, 0, B);
    auto b_mlp    = ctx->batch_mlp_act->slice(0, 0, B);
    auto b_logits = ctx->batch_logits->slice(0, 0, B);

    std::vector<int64_t> host_tokens(B);
    std::vector<int64_t> host_positions(B);
    for (size_t i = 0; i < B; ++i) {
        host_tokens[i] = current_tokens[i];
        host_positions[i] = ctx->slots[active_slots[i]].current_pos;
    }
    model->memcpyH2D(b_input, host_tokens.data(), B * sizeof(int64_t));
    model->memcpyH2D(b_pos, host_positions.data(), B * sizeof(int64_t));

    // ── 3. 批量 Embedding ──
    ops::embedding(b_hidden, b_input, TO_CPP_TENSOR(model->weights.in_embed));

    // ── 4. Transformer Layers ──
    float scale = 1.0f / std::sqrt((float)meta.dh);

    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        std::swap(b_resid, b_hidden);

        // A. Pre-Norm
        ops::rms_norm(b_norm, b_resid,
                      TO_CPP_TENSOR(model->weights.attn_norm_w[layer]),
                      meta.epsilon);

        // B. QKV Linear
        ctx->linear_maybe_dequant(b_q, b_norm,
            model->weights.attn_q_w[layer], model->weights.attn_q_w_scale[layer],
            model->weights.attn_q_b[layer], model->weights.attn_q_w_qzeros[layer]);
        ctx->linear_maybe_dequant(b_k, b_norm,
            model->weights.attn_k_w[layer], model->weights.attn_k_w_scale[layer],
            model->weights.attn_k_b[layer], model->weights.attn_k_w_qzeros[layer]);
        ctx->linear_maybe_dequant(b_v, b_norm,
            model->weights.attn_v_w[layer], model->weights.attn_v_w_scale[layer],
            model->weights.attn_v_b[layer], model->weights.attn_v_w_qzeros[layer]);

        // C. RoPE
        auto q_3d = b_q->reshape({B, model->local_nh, meta.dh});
        auto k_3d = b_k->reshape({B, model->local_nkvh, meta.dh});
        ops::rope(q_3d, q_3d, b_pos, meta.theta);
        ops::rope(k_3d, k_3d, b_pos, meta.theta);

        // D. Write KV to block pool for current token (字节寻址, 兼容 FP16/FP32)
        auto v_3d = b_v->reshape({B, model->local_nkvh, meta.dh});
        for (size_t i = 0; i < B; ++i) {
            auto& slot = ctx->slots[active_slots[i]];
            int64_t pos = slot.current_pos;
            int bid = slot.page_table.get_block_for_token(static_cast<int>(pos));
            int off = slot.page_table.get_offset_in_block(static_cast<int>(pos));

            char* k_dst = (char*)alloc.get_k_ptr(bid, layer)
                           + (size_t)off * kv_bytes;
            char* v_dst = (char*)alloc.get_v_ptr(bid, layer)
                           + (size_t)off * kv_bytes;
            char* k_src = (char*)k_3d->data() + i * kv_bytes;
            char* v_src = (char*)v_3d->data() + i * kv_bytes;
            model->memcpyOnDevice(k_dst, k_src, kv_bytes);
            model->memcpyOnDevice(v_dst, v_src, kv_bytes);
        }

        // E. Paged Attention (all sequences in one call)
        llaisys::ops::paged_attention(
            b_attn->data(),
            q_3d->data(),
            alloc.pool_k_raw(), alloc.pool_v_raw(),
            block_tables.data(), seq_lens.data(),
            static_cast<int>(B),
            static_cast<int>(model->local_nh),
            static_cast<int>(model->local_nkvh),
            static_cast<int>(meta.dh),
            ctx->block_size,
            max_blocks_per_seq,
            alloc.block_stride(), alloc.layer_stride(),
            static_cast<int>(layer), scale,
            model->device_type,
            llaisys::ops::KVQuantMode::FP32,
            model->act_dtype);

        // F. O Projection
        ctx->linear_maybe_dequant(b_hidden, b_attn,
            model->weights.attn_o_w[layer], model->weights.attn_o_w_scale[layer],
            nullptr, model->weights.attn_o_w_qzeros[layer]);
        model->allReduceIfTP(b_hidden, B * meta.hs);

        // G. Residual Add 1
        ops::add(b_hidden, b_hidden, b_resid);

        // H. MLP Block
        std::swap(b_resid, b_hidden);
        ops::rms_norm(b_norm, b_resid,
                      TO_CPP_TENSOR(model->weights.mlp_norm_w[layer]),
                      meta.epsilon);

        ctx->linear_maybe_dequant(b_gate, b_norm,
            model->weights.mlp_gate_w[layer], model->weights.mlp_gate_w_scale[layer],
            nullptr, model->weights.mlp_gate_w_qzeros[layer]);
        ctx->linear_maybe_dequant(b_up, b_norm,
            model->weights.mlp_up_w[layer], model->weights.mlp_up_w_scale[layer],
            nullptr, model->weights.mlp_up_w_qzeros[layer]);
        ops::swiglu(b_mlp, b_gate, b_up);
        ctx->linear_maybe_dequant(b_hidden, b_mlp,
            model->weights.mlp_down_w[layer], model->weights.mlp_down_w_scale[layer],
            nullptr, model->weights.mlp_down_w_qzeros[layer]);
        model->allReduceIfTP(b_hidden, B * meta.hs);

        // I. Residual Add 2
        ops::add(b_hidden, b_hidden, b_resid);
    }

    // ── 5. Final Norm ──
    ops::rms_norm(b_hidden, b_hidden,
                  TO_CPP_TENSOR(model->weights.out_norm_w), meta.epsilon);

    // ── 6. LM Head ──
    ctx->linear_maybe_dequant(b_logits, b_hidden,
        model->weights.out_embed, model->weights.out_embed_scale, nullptr);

    // ── 7. Per-slot Argmax/Sample + position update ──
    for (size_t i = 0; i < B; ++i) {
        size_t sid = active_slots[i];

        size_t logits_bytes = meta.voc * sizeof(float);
        char* logits_src = (char*)b_logits->data() + i * logits_bytes;
        model->memcpyOnDevice(ctx->single_logits->data(), logits_src, logits_bytes);

        if (use_greedy) {
            ops::argmax(ctx->single_next_token, ctx->single_max_val, ctx->single_logits);
        } else {
            ops::sample(ctx->single_next_token, ctx->single_logits,
                        temperature, top_k, top_p, model->rng_seed++);
        }

        int32_t host_token;
        model->memcpyD2H(&host_token, ctx->single_next_token, sizeof(int32_t));
        output_tokens[i] = host_token;

        ctx->slots[sid].page_table.inc_num_tokens();
        ctx->slots[sid].current_pos++;
    }
}

// Per-request sampling variant: each slot uses its own temperature/top_k/top_p
static void batch_decode_per_request_impl(LlaisysQwen2BatchContext* ctx,
                                           size_t* active_slots, size_t num_active,
                                           int64_t* current_tokens,
                                           float* temperatures, int* top_ks, float* top_ps,
                                           int64_t* output_tokens) {
    if (num_active == 0) return;

    auto* model = ctx->model;
    auto& meta = model->meta;
    auto& alloc = *ctx->block_allocator;
    size_t B = num_active;
    size_t kv_dim = model->local_nkvh * meta.dh;
    size_t kv_bytes = kv_dim * llaisys::utils::dsize(model->act_dtype);

    // Block allocation
    for (size_t i = 0; i < B; ++i) {
        auto& slot = ctx->slots[active_slots[i]];
        if (slot.page_table.needs_new_block()) {
            int bid = alloc.alloc();
            if (bid < 0) {
                std::cerr << "[qwen2] decode: block pool exhausted for slot "
                          << active_slots[i] << std::endl;
                return;
            }
            slot.page_table.append_block(bid);
        }
    }

    int max_blocks_per_seq = 0;
    for (size_t i = 0; i < B; ++i) {
        auto& slot = ctx->slots[active_slots[i]];
        max_blocks_per_seq = std::max(max_blocks_per_seq, slot.page_table.num_blocks());
    }

    std::vector<int> block_tables(B * max_blocks_per_seq, 0);
    std::vector<int> seq_lens(B);
    for (size_t i = 0; i < B; ++i) {
        auto& slot = ctx->slots[active_slots[i]];
        seq_lens[i] = static_cast<int>(slot.current_pos + 1);
        for (int j = 0; j < slot.page_table.num_blocks(); ++j)
            block_tables[i * max_blocks_per_seq + j] = slot.page_table.block_ids()[j];
    }

    auto b_input  = ctx->batch_input_ids->slice(0, 0, B);
    auto b_pos    = ctx->batch_pos_ids->slice(0, 0, B);
    auto b_hidden = ctx->batch_hidden->slice(0, 0, B);
    auto b_resid  = ctx->batch_residual->slice(0, 0, B);
    auto b_norm   = ctx->batch_norm_out->slice(0, 0, B);
    auto b_q      = ctx->batch_q->slice(0, 0, B);
    auto b_k      = ctx->batch_k->slice(0, 0, B);
    auto b_v      = ctx->batch_v->slice(0, 0, B);
    auto b_attn   = ctx->batch_attn_out->slice(0, 0, B);
    auto b_gate   = ctx->batch_gate->slice(0, 0, B);
    auto b_up     = ctx->batch_up->slice(0, 0, B);
    auto b_mlp    = ctx->batch_mlp_act->slice(0, 0, B);
    auto b_logits = ctx->batch_logits->slice(0, 0, B);

    std::vector<int64_t> host_tokens(B);
    std::vector<int64_t> host_positions(B);
    for (size_t i = 0; i < B; ++i) {
        host_tokens[i] = current_tokens[i];
        host_positions[i] = ctx->slots[active_slots[i]].current_pos;
    }
    model->memcpyH2D(b_input, host_tokens.data(), B * sizeof(int64_t));
    model->memcpyH2D(b_pos, host_positions.data(), B * sizeof(int64_t));

    ops::embedding(b_hidden, b_input, TO_CPP_TENSOR(model->weights.in_embed));

    float scale = 1.0f / std::sqrt((float)meta.dh);

    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        std::swap(b_resid, b_hidden);
        ops::rms_norm(b_norm, b_resid,
                      TO_CPP_TENSOR(model->weights.attn_norm_w[layer]), meta.epsilon);
        ctx->linear_maybe_dequant(b_q, b_norm,
            model->weights.attn_q_w[layer], model->weights.attn_q_w_scale[layer],
            model->weights.attn_q_b[layer], model->weights.attn_q_w_qzeros[layer]);
        ctx->linear_maybe_dequant(b_k, b_norm,
            model->weights.attn_k_w[layer], model->weights.attn_k_w_scale[layer],
            model->weights.attn_k_b[layer], model->weights.attn_k_w_qzeros[layer]);
        ctx->linear_maybe_dequant(b_v, b_norm,
            model->weights.attn_v_w[layer], model->weights.attn_v_w_scale[layer],
            model->weights.attn_v_b[layer], model->weights.attn_v_w_qzeros[layer]);

        auto q_3d = b_q->reshape({B, model->local_nh, meta.dh});
        auto k_3d = b_k->reshape({B, model->local_nkvh, meta.dh});
        ops::rope(q_3d, q_3d, b_pos, meta.theta);
        ops::rope(k_3d, k_3d, b_pos, meta.theta);

        auto v_3d = b_v->reshape({B, model->local_nkvh, meta.dh});
        for (size_t i = 0; i < B; ++i) {
            auto& slot = ctx->slots[active_slots[i]];
            int64_t pos = slot.current_pos;
            int bid = slot.page_table.get_block_for_token(static_cast<int>(pos));
            int off = slot.page_table.get_offset_in_block(static_cast<int>(pos));

            char* k_dst = (char*)alloc.get_k_ptr(bid, layer)
                           + (size_t)off * kv_bytes;
            char* v_dst = (char*)alloc.get_v_ptr(bid, layer)
                           + (size_t)off * kv_bytes;
            char* k_src = (char*)k_3d->data() + i * kv_bytes;
            char* v_src = (char*)v_3d->data() + i * kv_bytes;
            model->memcpyOnDevice(k_dst, k_src, kv_bytes);
            model->memcpyOnDevice(v_dst, v_src, kv_bytes);
        }

        llaisys::ops::paged_attention(
            b_attn->data(), q_3d->data(),
            alloc.pool_k_raw(), alloc.pool_v_raw(),
            block_tables.data(), seq_lens.data(),
            static_cast<int>(B), static_cast<int>(model->local_nh),
            static_cast<int>(model->local_nkvh), static_cast<int>(meta.dh),
            ctx->block_size, max_blocks_per_seq,
            alloc.block_stride(), alloc.layer_stride(),
            static_cast<int>(layer), scale, model->device_type,
            llaisys::ops::KVQuantMode::FP32, model->act_dtype);

        ctx->linear_maybe_dequant(b_hidden, b_attn,
            model->weights.attn_o_w[layer], model->weights.attn_o_w_scale[layer],
            nullptr, model->weights.attn_o_w_qzeros[layer]);
        model->allReduceIfTP(b_hidden, B * meta.hs);
        ops::add(b_hidden, b_hidden, b_resid);

        std::swap(b_resid, b_hidden);
        ops::rms_norm(b_norm, b_resid,
                      TO_CPP_TENSOR(model->weights.mlp_norm_w[layer]), meta.epsilon);
        ctx->linear_maybe_dequant(b_gate, b_norm,
            model->weights.mlp_gate_w[layer], model->weights.mlp_gate_w_scale[layer],
            nullptr, model->weights.mlp_gate_w_qzeros[layer]);
        ctx->linear_maybe_dequant(b_up, b_norm,
            model->weights.mlp_up_w[layer], model->weights.mlp_up_w_scale[layer],
            nullptr, model->weights.mlp_up_w_qzeros[layer]);
        ops::swiglu(b_mlp, b_gate, b_up);
        ctx->linear_maybe_dequant(b_hidden, b_mlp,
            model->weights.mlp_down_w[layer], model->weights.mlp_down_w_scale[layer],
            nullptr, model->weights.mlp_down_w_qzeros[layer]);
        model->allReduceIfTP(b_hidden, B * meta.hs);
        ops::add(b_hidden, b_hidden, b_resid);
    }

    ops::rms_norm(b_hidden, b_hidden,
                  TO_CPP_TENSOR(model->weights.out_norm_w), meta.epsilon);
    ctx->linear_maybe_dequant(b_logits, b_hidden,
        model->weights.out_embed, model->weights.out_embed_scale, nullptr);

    // Per-request sampling
    for (size_t i = 0; i < B; ++i) {
        size_t sid = active_slots[i];
        float temp = temperatures[i];
        int topk = top_ks[i];
        float topp = top_ps[i];
        bool greedy = (topk == 1) || (temp <= 0.0f);

        size_t logits_bytes = meta.voc * sizeof(float);
        char* logits_src = (char*)b_logits->data() + i * logits_bytes;
        model->memcpyOnDevice(ctx->single_logits->data(), logits_src, logits_bytes);

        if (greedy) {
            ops::argmax(ctx->single_next_token, ctx->single_max_val, ctx->single_logits);
        } else {
            ops::sample(ctx->single_next_token, ctx->single_logits,
                        temp, topk, topp, model->rng_seed++);
        }

        int32_t host_token;
        model->memcpyD2H(&host_token, ctx->single_next_token, sizeof(int32_t));
        output_tokens[i] = host_token;

        ctx->slots[sid].page_table.inc_num_tokens();
        ctx->slots[sid].current_pos++;
    }
}

// ── Slot KV-Cache 保存/恢复 (Paged → 连续快照) ───────────────────
// 保存: 遍历 page_table, 从分散的 block pool 中按 token 顺序拷贝到连续 CPU buffer
// 恢复: 分配新 block, 将连续 CPU buffer 按 block_size 切分写回 block pool
// 用途: 会话切换, preemption (抢占时先保存, 重新调度时恢复)

static LlaisysQwen2CacheSnapshot* batch_slot_save_impl(
    LlaisysQwen2BatchContext* ctx, size_t slot_id)
{
    auto* model = ctx->model;
    auto& slot = ctx->slots[slot_id];
    auto& alloc = *ctx->block_allocator;
    if (slot.current_pos <= 0) return nullptr;

    auto* snap = new LlaisysQwen2CacheSnapshot();
    snap->pos = slot.current_pos;
    snap->nlayer = model->meta.nlayer;
    snap->pos_bytes = model->local_nkvh * model->meta.dh * llaisys::utils::dsize(model->act_dtype);
    snap->tp_size = model->tp_size;
    snap->tp_rank = model->tp_rank;
    snap->buffers.resize(snap->nlayer * 2);

    size_t total_bytes = snap->pos * snap->pos_bytes;
    size_t kv_row_bytes = snap->pos_bytes;

    for (size_t layer = 0; layer < snap->nlayer; ++layer) {
        for (size_t kv = 0; kv < 2; ++kv) {
            size_t idx = layer * 2 + kv;
            snap->buffers[idx].resize(total_bytes);
        }
    }

    // Linearize paged KV data into contiguous snapshot buffers
    for (int64_t t = 0; t < snap->pos; ++t) {
        int bid = slot.page_table.get_block_for_token(static_cast<int>(t));
        int off = slot.page_table.get_offset_in_block(static_cast<int>(t));
        for (size_t layer = 0; layer < snap->nlayer; ++layer) {
            char* k_src = (char*)alloc.get_k_ptr(bid, layer)
                           + (size_t)off * kv_row_bytes;
            char* v_src = (char*)alloc.get_v_ptr(bid, layer)
                           + (size_t)off * kv_row_bytes;
            void* k_dst = snap->buffers[layer * 2].data() + t * kv_row_bytes;
            void* v_dst = snap->buffers[layer * 2 + 1].data() + t * kv_row_bytes;
            model->memcpyD2H(k_dst, k_src, kv_row_bytes);
            model->memcpyD2H(v_dst, v_src, kv_row_bytes);
        }
    }
    return snap;
}

static void batch_slot_restore_impl(
    LlaisysQwen2BatchContext* ctx, size_t slot_id,
    LlaisysQwen2CacheSnapshot* snapshot)
{
    if (!snapshot) return;
    auto* model = ctx->model;
    auto& slot = ctx->slots[slot_id];
    auto& alloc = *ctx->block_allocator;
    if (snapshot->nlayer != model->meta.nlayer) return;
    if (snapshot->tp_size != model->tp_size || snapshot->tp_rank != model->tp_rank) {
        std::cerr << "[qwen2] BatchSlotRestore: TP mismatch" << std::endl;
        return;
    }

    // Release any existing blocks before restore
    slot.page_table.release_all(alloc);
    slot.current_pos = 0;

    size_t kv_row_bytes = snapshot->pos_bytes;

    // Re-populate page table and copy data from snapshot into block pool
    for (int64_t t = 0; t < snapshot->pos; ++t) {
        if (slot.page_table.needs_new_block()) {
            int bid = alloc.alloc();
            if (bid < 0) {
                std::cerr << "[qwen2] BatchSlotRestore: block pool exhausted" << std::endl;
                return;
            }
            slot.page_table.append_block(bid);
        }

        int bid = slot.page_table.get_block_for_token(static_cast<int>(t));
        int off = slot.page_table.get_offset_in_block(static_cast<int>(t));

        for (size_t layer = 0; layer < snapshot->nlayer; ++layer) {
            const void* k_src = snapshot->buffers[layer * 2].data() + t * kv_row_bytes;
            const void* v_src = snapshot->buffers[layer * 2 + 1].data() + t * kv_row_bytes;
            void* k_dst = (char*)alloc.get_k_ptr(bid, layer)
                          + (size_t)off * kv_row_bytes;
            void* v_dst = (char*)alloc.get_v_ptr(bid, layer)
                          + (size_t)off * kv_row_bytes;
            model->memcpyH2D(k_dst, k_src, kv_row_bytes);
            model->memcpyH2D(v_dst, v_src, kv_row_bytes);
        }
        slot.page_table.inc_num_tokens();
    }

    slot.current_pos = snapshot->pos;
    slot.active = true;
}

// ── C API for Batch Context ──────────────────────────────────────

__export struct LlaisysQwen2BatchContext *llaisysQwen2BatchContextCreate(
    struct LlaisysQwen2Model * model, size_t max_batch_size, size_t max_seq_per_slot)
{
    if (!model || max_batch_size == 0) return nullptr;
    return new LlaisysQwen2BatchContext(model, max_batch_size, max_seq_per_slot);
}

__export void llaisysQwen2BatchContextDestroy(struct LlaisysQwen2BatchContext * ctx) {
    if (ctx) delete ctx;
}

__export void llaisysQwen2BatchSlotReset(struct LlaisysQwen2BatchContext * ctx, size_t slot_id) {
    if (!ctx || slot_id >= ctx->max_batch_size) return;
    ctx->slots[slot_id].reset(*ctx->block_allocator);
}

__export int64_t llaisysQwen2BatchPrefill(
    struct LlaisysQwen2BatchContext * ctx,
    size_t slot_id,
    int64_t * token_ids, size_t ntoken,
    float temperature, int top_k, float top_p)
{
    if (!ctx || slot_id >= ctx->max_batch_size || !token_ids || ntoken == 0) return -1;
    ctx->slots[slot_id].reset(*ctx->block_allocator);
    return batch_prefill_impl(ctx, slot_id, token_ids, ntoken,
                              temperature, top_k, top_p);
}

__export void llaisysQwen2BatchDecode(
    struct LlaisysQwen2BatchContext * ctx,
    size_t * active_slots, size_t num_active,
    int64_t * current_tokens,
    float temperature, int top_k, float top_p,
    int64_t * output_tokens)
{
    if (!ctx || !active_slots || !current_tokens || !output_tokens || num_active == 0) return;
    // 验证 slot IDs
    for (size_t i = 0; i < num_active; ++i) {
        if (active_slots[i] >= ctx->max_batch_size) return;
    }
    batch_decode_impl(ctx, active_slots, num_active, current_tokens,
                      temperature, top_k, top_p, output_tokens);
}

__export int64_t llaisysQwen2BatchSlotGetPos(
    struct LlaisysQwen2BatchContext * ctx, size_t slot_id)
{
    if (!ctx || slot_id >= ctx->max_batch_size) return 0;
    return ctx->slots[slot_id].current_pos;
}

__export struct LlaisysQwen2CacheSnapshot *llaisysQwen2BatchSlotSave(
    struct LlaisysQwen2BatchContext * ctx, size_t slot_id)
{
    if (!ctx || slot_id >= ctx->max_batch_size) return nullptr;
    return batch_slot_save_impl(ctx, slot_id);
}

__export void llaisysQwen2BatchSlotRestore(
    struct LlaisysQwen2BatchContext * ctx, size_t slot_id,
    struct LlaisysQwen2CacheSnapshot * snapshot)
{
    if (!ctx || slot_id >= ctx->max_batch_size) return;
    batch_slot_restore_impl(ctx, slot_id, snapshot);
}

__export void llaisysQwen2BatchDecodePerRequest(
    struct LlaisysQwen2BatchContext * ctx,
    size_t * active_slots, size_t num_active,
    int64_t * current_tokens,
    float * temperatures, int * top_ks, float * top_ps,
    int64_t * output_tokens)
{
    if (!ctx || !active_slots || !current_tokens || !output_tokens || num_active == 0) return;
    if (!temperatures || !top_ks || !top_ps) return;
    for (size_t i = 0; i < num_active; ++i) {
        if (active_slots[i] >= ctx->max_batch_size) return;
    }
    batch_decode_per_request_impl(ctx, active_slots, num_active, current_tokens,
                                   temperatures, top_ks, top_ps, output_tokens);
}

__export size_t llaisysQwen2BatchGetFreeBlocks(
    struct LlaisysQwen2BatchContext * ctx)
{
    if (!ctx || !ctx->block_allocator) return 0;
    return ctx->block_allocator->num_free();
}

__export size_t llaisysQwen2BatchGetTotalBlocks(
    struct LlaisysQwen2BatchContext * ctx)
{
    if (!ctx || !ctx->block_allocator) return 0;
    return ctx->block_allocator->num_total();
}

__export int llaisysQwen2BatchGetBlockSize(
    struct LlaisysQwen2BatchContext * ctx)
{
    if (!ctx) return 0;
    return ctx->block_size;
}

} // extern "C"
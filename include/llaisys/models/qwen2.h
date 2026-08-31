#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h" // 确保包含基础类型定义 (如 llaisysDataType_t, __export, __C)
#include "../distributed.h" // llaisysDistComm_t

__C {
    // 模型元数据结构体
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    // 权重结构体 (用于内部管理或调试)
    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // model.norm.weight
        llaisysTensor_t *attn_norm_w; // input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;

        // INT8 量化 per-channel scale (shape [out_features], FP32)
        // 当权重为 INT8 时使用, 否则为 nullptr
        llaisysTensor_t out_embed_scale;
        llaisysTensor_t *attn_q_w_scale;
        llaisysTensor_t *attn_k_w_scale;
        llaisysTensor_t *attn_v_w_scale;
        llaisysTensor_t *attn_o_w_scale;
        llaisysTensor_t *mlp_gate_w_scale;
        llaisysTensor_t *mlp_up_w_scale;
        llaisysTensor_t *mlp_down_w_scale;

        // AWQ 原生零点 (int32 packed, 仅 AWQ 原生路径使用)
        llaisysTensor_t *attn_q_w_qzeros;
        llaisysTensor_t *attn_k_w_qzeros;
        llaisysTensor_t *attn_v_w_qzeros;
        llaisysTensor_t *attn_o_w_qzeros;
        llaisysTensor_t *mlp_gate_w_qzeros;
        llaisysTensor_t *mlp_up_w_qzeros;
        llaisysTensor_t *mlp_down_w_qzeros;
    };

    // 不透明的模型句柄
    struct LlaisysQwen2Model;

    // 不透明的 KV-Cache 快照句柄
    struct LlaisysQwen2CacheSnapshot;

    // 创建模型实例
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const struct LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    // 创建 TP (张量并行) 模型实例
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreateTP(const struct LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
                                                                 int device_id, int tp_size, int tp_rank);

    // 查询 TP 配置
    __export int llaisysQwen2GetTpSize(struct LlaisysQwen2Model * model);
    __export int llaisysQwen2GetTpRank(struct LlaisysQwen2Model * model);

    // 设置 TP 通信句柄 (传 NULL 清除)
    __export void llaisysQwen2SetComm(struct LlaisysQwen2Model * model, llaisysDistComm_t comm);

    // 销毁模型实例
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    // 获取权重结构体指针 (可选)
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    // 按名称加载权重 (Python 包装器使用此接口)
    __export void llaisysQwen2LoadWeightByName(struct LlaisysQwen2Model * model, const char * name, void * data, int ndim, int64_t * shape, int dtype);

    // 执行推理
    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    // 执行推理 (带采样参数)
    __export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken,
                                                  float temperature, int top_k, float top_p);

    // 重置 KV-Cache 位置 (不重新加载权重)
    __export void llaisysQwen2ResetCache(struct LlaisysQwen2Model * model);

    // ==========================================
    // Phase 4: KV-Cache 高级接口
    // ==========================================

    // 保存当前 KV-Cache 快照 (深拷贝到 CPU)
    __export struct LlaisysQwen2CacheSnapshot *llaisysQwen2SaveCache(struct LlaisysQwen2Model * model);

    // 从快照恢复 KV-Cache (从 CPU 拷贝回设备)
    __export void llaisysQwen2RestoreCache(struct LlaisysQwen2Model * model, struct LlaisysQwen2CacheSnapshot * snapshot);

    // 截断 KV-Cache 到指定位置 (pos 必须 <= current_pos)
    __export void llaisysQwen2TruncateCache(struct LlaisysQwen2Model * model, int64_t pos);

    // 获取当前 KV-Cache 位置
    __export int64_t llaisysQwen2GetCachePos(struct LlaisysQwen2Model * model);

    // 销毁 KV-Cache 快照
    __export void llaisysQwen2DestroyCacheSnapshot(struct LlaisysQwen2CacheSnapshot * snapshot);

    // ==========================================
    // 量化支持
    // ==========================================

    // 查询模型是否已加载量化权重
    __export int llaisysQwen2IsQuantized(struct LlaisysQwen2Model * model);

    // ==========================================
    // Phase 5 (项目#4): 批量推理 API
    // ==========================================

    // 不透明的批量推理上下文句柄
    struct LlaisysQwen2BatchContext;

    // 创建批量推理上下文 (预分配 max_batch_size 个 KV-Cache slot)
    // max_seq_per_slot: 每个 slot 的 KV-Cache 最大序列长度 (0 = 默认 2048)
    __export struct LlaisysQwen2BatchContext *llaisysQwen2BatchContextCreate(
        struct LlaisysQwen2Model * model, size_t max_batch_size, size_t max_seq_per_slot);

    // 销毁批量推理上下文
    __export void llaisysQwen2BatchContextDestroy(
        struct LlaisysQwen2BatchContext * ctx);

    // 重置指定 slot 的 KV-Cache (清空)
    __export void llaisysQwen2BatchSlotReset(
        struct LlaisysQwen2BatchContext * ctx, size_t slot_id);

    // Prefill: 在指定 slot 上对完整 prompt 执行 prefill, 返回首个 next token
    __export int64_t llaisysQwen2BatchPrefill(
        struct LlaisysQwen2BatchContext * ctx,
        size_t slot_id,
        int64_t * token_ids, size_t ntoken,
        float temperature, int top_k, float top_p);

    // Incremental prefill: append one prompt chunk without resetting the slot.
    // start_pos must equal the slot's current cache position. Intermediate
    // chunks return -1; the final chunk samples and returns the next token.
    __export int64_t llaisysQwen2BatchPrefillChunk(
        struct LlaisysQwen2BatchContext * ctx,
        size_t slot_id,
        const int64_t * token_ids, size_t ntoken,
        int64_t start_pos, int is_last_chunk,
        float temperature, int top_k, float top_p);

    // Block-granular prefix cache. Lookup resets the slot and attaches shared
    // complete blocks; publish indexes complete computed prompt blocks.
    __export size_t llaisysQwen2BatchPrefixLookup(
        struct LlaisysQwen2BatchContext * ctx, size_t slot_id,
        const int64_t * token_ids, size_t ntoken);
    __export int llaisysQwen2BatchPrefixPublish(
        struct LlaisysQwen2BatchContext * ctx, size_t slot_id,
        const int64_t * token_ids, size_t ntoken);

    // 批量 Decode: 对 num_active 个活跃 slot 执行一步 decode
    //   active_slots: [num_active] slot ID 数组
    //   current_tokens: [num_active] 各 slot 当前 token
    //   output_tokens: [num_active] 输出的 next token (由调用者分配)
    __export void llaisysQwen2BatchDecode(
        struct LlaisysQwen2BatchContext * ctx,
        size_t * active_slots, size_t num_active,
        int64_t * current_tokens,
        float temperature, int top_k, float top_p,
        int64_t * output_tokens);

    // 获取 slot 当前 KV-Cache 位置
    __export int64_t llaisysQwen2BatchSlotGetPos(
        struct LlaisysQwen2BatchContext * ctx, size_t slot_id);

    // 保存 slot 的 KV-Cache 快照
    __export struct LlaisysQwen2CacheSnapshot *llaisysQwen2BatchSlotSave(
        struct LlaisysQwen2BatchContext * ctx, size_t slot_id);

    // 从快照恢复 slot 的 KV-Cache
    __export void llaisysQwen2BatchSlotRestore(
        struct LlaisysQwen2BatchContext * ctx, size_t slot_id,
        struct LlaisysQwen2CacheSnapshot * snapshot);

    // Per-request sampling: each slot uses its own temperature/top_k/top_p
    __export void llaisysQwen2BatchDecodePerRequest(
        struct LlaisysQwen2BatchContext * ctx,
        size_t * active_slots, size_t num_active,
        int64_t * current_tokens,
        float * temperatures, int * top_ks, float * top_ps,
        int64_t * output_tokens);

    // Paged KV-Cache block allocator queries
    __export size_t llaisysQwen2BatchGetFreeBlocks(
        struct LlaisysQwen2BatchContext * ctx);
    __export size_t llaisysQwen2BatchGetTotalBlocks(
        struct LlaisysQwen2BatchContext * ctx);
    __export int llaisysQwen2BatchGetBlockSize(
        struct LlaisysQwen2BatchContext * ctx);

}
#endif // LLAISYS_MODELS_QWEN2_H

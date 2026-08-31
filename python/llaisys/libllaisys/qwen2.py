import ctypes
from . import LIB_LLAISYS

# 1. 定义 C 结构体 (对应 qwen2.h 中的 LlaisysQwen2Meta)
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

# 2. 配置函数签名
def _setup_functions():
    lib = LIB_LLAISYS
    
    # Create
    if hasattr(lib, 'llaisysQwen2ModelCreate'):
        lib.llaisysQwen2ModelCreate.argtypes = [
            ctypes.POINTER(LlaisysQwen2Meta), 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.c_int), 
            ctypes.c_int
        ]
        lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

    # Destroy
    if hasattr(lib, 'llaisysQwen2ModelDestroy'):
        lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2ModelDestroy.restype = None

    # Load Weight
    if hasattr(lib, 'llaisysQwen2LoadWeightByName'):
        lib.llaisysQwen2LoadWeightByName.argtypes = [
            ctypes.c_void_p,                # model
            ctypes.c_char_p,                # name
            ctypes.c_void_p,                # data
            ctypes.c_int,                   # ndim
            ctypes.POINTER(ctypes.c_int64), # shape
            ctypes.c_int                    # dtype
        ]
        lib.llaisysQwen2LoadWeightByName.restype = None

    # Infer
    if hasattr(lib, 'llaisysQwen2ModelInfer'):
        lib.llaisysQwen2ModelInfer.argtypes = [
            ctypes.c_void_p,                # model
            ctypes.POINTER(ctypes.c_int64), # token_ids
            ctypes.c_size_t                 # ntoken
        ]
        lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64

    # InferSample
    if hasattr(lib, 'llaisysQwen2ModelInferSample'):
        lib.llaisysQwen2ModelInferSample.argtypes = [
            ctypes.c_void_p,                # model
            ctypes.POINTER(ctypes.c_int64), # token_ids
            ctypes.c_size_t,                # ntoken
            ctypes.c_float,                 # temperature
            ctypes.c_int,                   # top_k
            ctypes.c_float,                 # top_p
        ]
        lib.llaisysQwen2ModelInferSample.restype = ctypes.c_int64

    # ResetCache
    if hasattr(lib, 'llaisysQwen2ResetCache'):
        lib.llaisysQwen2ResetCache.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2ResetCache.restype = None

    # ── Phase 4: KV-Cache 高级接口 ──

    # SaveCache
    if hasattr(lib, 'llaisysQwen2SaveCache'):
        lib.llaisysQwen2SaveCache.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2SaveCache.restype = ctypes.c_void_p

    # RestoreCache
    if hasattr(lib, 'llaisysQwen2RestoreCache'):
        lib.llaisysQwen2RestoreCache.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.llaisysQwen2RestoreCache.restype = None

    # TruncateCache
    if hasattr(lib, 'llaisysQwen2TruncateCache'):
        lib.llaisysQwen2TruncateCache.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        lib.llaisysQwen2TruncateCache.restype = None

    # GetCachePos
    if hasattr(lib, 'llaisysQwen2GetCachePos'):
        lib.llaisysQwen2GetCachePos.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2GetCachePos.restype = ctypes.c_int64

    # DestroyCacheSnapshot
    if hasattr(lib, 'llaisysQwen2DestroyCacheSnapshot'):
        lib.llaisysQwen2DestroyCacheSnapshot.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2DestroyCacheSnapshot.restype = None

    # ── Phase 5 (项目#4): 批量推理 API ──

    # BatchContextCreate
    if hasattr(lib, 'llaisysQwen2BatchContextCreate'):
        lib.llaisysQwen2BatchContextCreate.argtypes = [
            ctypes.c_void_p,  # model
            ctypes.c_size_t,  # max_batch_size
            ctypes.c_size_t,  # max_seq_per_slot
        ]
        lib.llaisysQwen2BatchContextCreate.restype = ctypes.c_void_p

    # BatchContextDestroy
    if hasattr(lib, 'llaisysQwen2BatchContextDestroy'):
        lib.llaisysQwen2BatchContextDestroy.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2BatchContextDestroy.restype = None

    # BatchSlotReset
    if hasattr(lib, 'llaisysQwen2BatchSlotReset'):
        lib.llaisysQwen2BatchSlotReset.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.llaisysQwen2BatchSlotReset.restype = None

    # BatchPrefill
    if hasattr(lib, 'llaisysQwen2BatchPrefill'):
        lib.llaisysQwen2BatchPrefill.argtypes = [
            ctypes.c_void_p,                # ctx
            ctypes.c_size_t,                # slot_id
            ctypes.POINTER(ctypes.c_int64), # token_ids
            ctypes.c_size_t,                # ntoken
            ctypes.c_float,                 # temperature
            ctypes.c_int,                   # top_k
            ctypes.c_float,                 # top_p
        ]
        lib.llaisysQwen2BatchPrefill.restype = ctypes.c_int64

    # Incremental BatchPrefillChunk
    if hasattr(lib, 'llaisysQwen2BatchPrefillChunk'):
        lib.llaisysQwen2BatchPrefillChunk.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_float,
        ]
        lib.llaisysQwen2BatchPrefillChunk.restype = ctypes.c_int64

    if hasattr(lib, 'llaisysQwen2BatchPrefixLookup'):
        lib.llaisysQwen2BatchPrefixLookup.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
        ]
        lib.llaisysQwen2BatchPrefixLookup.restype = ctypes.c_size_t
    if hasattr(lib, 'llaisysQwen2BatchPrefixPublish'):
        lib.llaisysQwen2BatchPrefixPublish.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
        ]
        lib.llaisysQwen2BatchPrefixPublish.restype = ctypes.c_int

    # BatchDecode
    if hasattr(lib, 'llaisysQwen2BatchDecode'):
        lib.llaisysQwen2BatchDecode.argtypes = [
            ctypes.c_void_p,                 # ctx
            ctypes.POINTER(ctypes.c_size_t), # active_slots
            ctypes.c_size_t,                 # num_active
            ctypes.POINTER(ctypes.c_int64),  # current_tokens
            ctypes.c_float,                  # temperature
            ctypes.c_int,                    # top_k
            ctypes.c_float,                  # top_p
            ctypes.POINTER(ctypes.c_int64),  # output_tokens
        ]
        lib.llaisysQwen2BatchDecode.restype = None

    # BatchSlotGetPos
    if hasattr(lib, 'llaisysQwen2BatchSlotGetPos'):
        lib.llaisysQwen2BatchSlotGetPos.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.llaisysQwen2BatchSlotGetPos.restype = ctypes.c_int64

    # BatchSlotSave
    if hasattr(lib, 'llaisysQwen2BatchSlotSave'):
        lib.llaisysQwen2BatchSlotSave.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.llaisysQwen2BatchSlotSave.restype = ctypes.c_void_p

    # BatchSlotRestore
    if hasattr(lib, 'llaisysQwen2BatchSlotRestore'):
        lib.llaisysQwen2BatchSlotRestore.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_size_t,  # slot_id
            ctypes.c_void_p,  # snapshot
        ]
        lib.llaisysQwen2BatchSlotRestore.restype = None

    # BatchDecodePerRequest
    if hasattr(lib, 'llaisysQwen2BatchDecodePerRequest'):
        lib.llaisysQwen2BatchDecodePerRequest.argtypes = [
            ctypes.c_void_p,                 # ctx
            ctypes.POINTER(ctypes.c_size_t), # active_slots
            ctypes.c_size_t,                 # num_active
            ctypes.POINTER(ctypes.c_int64),  # current_tokens
            ctypes.POINTER(ctypes.c_float),  # temperatures
            ctypes.POINTER(ctypes.c_int),    # top_ks
            ctypes.POINTER(ctypes.c_float),  # top_ps
            ctypes.POINTER(ctypes.c_int64),  # output_tokens
        ]
        lib.llaisysQwen2BatchDecodePerRequest.restype = None

    # BatchGetFreeBlocks
    if hasattr(lib, 'llaisysQwen2BatchGetFreeBlocks'):
        lib.llaisysQwen2BatchGetFreeBlocks.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2BatchGetFreeBlocks.restype = ctypes.c_size_t

    # BatchGetTotalBlocks
    if hasattr(lib, 'llaisysQwen2BatchGetTotalBlocks'):
        lib.llaisysQwen2BatchGetTotalBlocks.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2BatchGetTotalBlocks.restype = ctypes.c_size_t

    # BatchGetBlockSize
    if hasattr(lib, 'llaisysQwen2BatchGetBlockSize'):
        lib.llaisysQwen2BatchGetBlockSize.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2BatchGetBlockSize.restype = ctypes.c_int

    # ── Phase 5 (项目#5): 分布式推理 / 张量并行 TP API ──

    # CreateTP
    if hasattr(lib, 'llaisysQwen2ModelCreateTP'):
        lib.llaisysQwen2ModelCreateTP.argtypes = [
            ctypes.POINTER(LlaisysQwen2Meta),  # meta
            ctypes.c_int,                       # device
            ctypes.c_int,                       # device_id
            ctypes.c_int,                       # tp_size
            ctypes.c_int,                       # tp_rank
        ]
        lib.llaisysQwen2ModelCreateTP.restype = ctypes.c_void_p

    # GetTpSize
    if hasattr(lib, 'llaisysQwen2GetTpSize'):
        lib.llaisysQwen2GetTpSize.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2GetTpSize.restype = ctypes.c_int

    # GetTpRank
    if hasattr(lib, 'llaisysQwen2GetTpRank'):
        lib.llaisysQwen2GetTpRank.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2GetTpRank.restype = ctypes.c_int

    # SetComm
    if hasattr(lib, 'llaisysQwen2SetComm'):
        lib.llaisysQwen2SetComm.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.llaisysQwen2SetComm.restype = None

# 执行配置
_setup_functions()

# 3. 导出函数 (方便外部调用)
model_create = LIB_LLAISYS.llaisysQwen2ModelCreate
model_destroy = LIB_LLAISYS.llaisysQwen2ModelDestroy
load_weight = LIB_LLAISYS.llaisysQwen2LoadWeightByName
model_infer = LIB_LLAISYS.llaisysQwen2ModelInfer
model_infer_sample = LIB_LLAISYS.llaisysQwen2ModelInferSample
model_reset_cache = LIB_LLAISYS.llaisysQwen2ResetCache

# Phase 4 导出
cache_save = LIB_LLAISYS.llaisysQwen2SaveCache
cache_restore = LIB_LLAISYS.llaisysQwen2RestoreCache
cache_truncate = LIB_LLAISYS.llaisysQwen2TruncateCache
cache_get_pos = LIB_LLAISYS.llaisysQwen2GetCachePos
cache_snapshot_destroy = LIB_LLAISYS.llaisysQwen2DestroyCacheSnapshot

# Phase 5 (项目#4) 导出
batch_context_create = LIB_LLAISYS.llaisysQwen2BatchContextCreate
batch_context_destroy = LIB_LLAISYS.llaisysQwen2BatchContextDestroy
batch_slot_reset = LIB_LLAISYS.llaisysQwen2BatchSlotReset
batch_prefill = LIB_LLAISYS.llaisysQwen2BatchPrefill
batch_prefill_chunk = getattr(LIB_LLAISYS, 'llaisysQwen2BatchPrefillChunk', None)
batch_prefix_lookup = getattr(LIB_LLAISYS, 'llaisysQwen2BatchPrefixLookup', None)
batch_prefix_publish = getattr(LIB_LLAISYS, 'llaisysQwen2BatchPrefixPublish', None)
batch_decode = LIB_LLAISYS.llaisysQwen2BatchDecode
batch_slot_get_pos = LIB_LLAISYS.llaisysQwen2BatchSlotGetPos
batch_slot_save = LIB_LLAISYS.llaisysQwen2BatchSlotSave
batch_slot_restore = LIB_LLAISYS.llaisysQwen2BatchSlotRestore

batch_decode_per_request = LIB_LLAISYS.llaisysQwen2BatchDecodePerRequest

# Paged KV-Cache block allocator queries
batch_get_free_blocks = LIB_LLAISYS.llaisysQwen2BatchGetFreeBlocks
batch_get_total_blocks = LIB_LLAISYS.llaisysQwen2BatchGetTotalBlocks
batch_get_block_size = LIB_LLAISYS.llaisysQwen2BatchGetBlockSize

# Phase 5 (项目#5) 导出: 张量并行 TP
model_create_tp = LIB_LLAISYS.llaisysQwen2ModelCreateTP
model_get_tp_size = LIB_LLAISYS.llaisysQwen2GetTpSize
model_get_tp_rank = LIB_LLAISYS.llaisysQwen2GetTpRank
model_set_comm = LIB_LLAISYS.llaisysQwen2SetComm

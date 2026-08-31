"""
models/qwen2.py — Qwen2 模型高层 Python 封装
==============================================
核心职责:
  1. 读取 config.json → 填充 LlaisysQwen2Meta → 调用 C API 创建模型
  2. 加载权重: FP32 / INT8 / INT4 / GPTQ / AWQ 多格式支持
     - GPTQ/AWQ: CPU 端解包 → FP16 反量化, 结果缓存到 .llaisys_cache/
     - AWQ native: 原始 I32 数据直接传 GPU, 推理时 GPU 在线反量化
  3. generate / generate_stream: 调用 C 端 Infer/InferSample 做推理
  4. KV-Cache 管理: save/restore/truncate + block-level Prefix Cache
  5. BatchContext: 批量推理 (多 slot 并发 decode)
  6. 张量并行 (TP): model_create_tp + comm 绑定

调用链:
  Python Qwen2 → ctypes → C API (distributed.cc / qwen2.cc) → C++ Model
"""
import ctypes
import numpy as np
import torch
from typing import Sequence, Optional
from ..libllaisys import DeviceType
from .. import native as _native

# 引入底层接口定义
from ..libllaisys.qwen2 import (
    LlaisysQwen2Meta, 
    model_create, 
    model_destroy, 
    load_weight, 
    model_infer,
    model_infer_sample,
    model_reset_cache,
    # Phase 4: KV-Cache 高级接口
    cache_save,
    cache_restore,
    cache_truncate,
    cache_get_pos,
    cache_snapshot_destroy,
    # Phase 5 (项目#4): 批量推理 API
    batch_context_create,
    batch_context_destroy,
    batch_slot_reset,
    batch_prefill,
    batch_prefill_chunk,
    batch_prefix_lookup,
    batch_prefix_publish,
    batch_decode,
    batch_slot_get_pos,
    batch_slot_save,
    batch_slot_restore,
    # Per-request sampling
    batch_decode_per_request,
    # Paged KV-Cache block allocator queries
    batch_get_free_blocks,
    batch_get_total_blocks,
    batch_get_block_size,
    # Phase 5 (项目#5): 张量并行
    model_create_tp,
    model_get_tp_size,
    model_get_tp_rank,
    model_set_comm,
)

# 分布式通信
from ..libllaisys.distributed import (
    DistBackend,
    LlaisysDistConfig,
    comm_create,
    comm_destroy,
    comm_backend,
    comm_world_size,
    comm_rank,
    backend_available,
)

from pathlib import Path
import safetensors
import safetensors.torch
import os
import hashlib
import json as _json
import time

# =========================================================================
# GPTQ/AWQ format constants
# =========================================================================

# GPTQ quantized weight suffixes (without the .qweight etc.)
GPTQ_LINEAR_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# =========================================================================
# 强制覆盖函数签名 (确保指针类型正确)
# =========================================================================
model_create.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
model_create.restype = ctypes.c_void_p

model_destroy.argtypes = [ctypes.c_void_p]
model_destroy.restype = None

load_weight.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
load_weight.restype = None

# 重点：Infer 的签名
model_infer.argtypes = [
    ctypes.c_void_p,                # model
    ctypes.POINTER(ctypes.c_int64), # token_ids (int64_t*)
    ctypes.c_size_t                 # ntoken (size_t)
]
model_infer.restype = ctypes.c_int64

# InferSample 的签名
model_infer_sample.argtypes = [
    ctypes.c_void_p,                # model
    ctypes.POINTER(ctypes.c_int64), # token_ids (int64_t*)
    ctypes.c_size_t,                # ntoken (size_t)
    ctypes.c_float,                 # temperature
    ctypes.c_int,                   # top_k
    ctypes.c_float,                 # top_p
]
model_infer_sample.restype = ctypes.c_int64
# =========================================================================

class Qwen2:
    # =====================================================================
    # 默认架构参数（DeepSeek-R1-Distill-Qwen-1.5B）
    # 当 config.json 缺少某项时用作回退
    # =====================================================================
    _DEFAULT_CONFIG = {
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "eos_token_id": 151643,
        "max_position_embeddings": 131072,
        "sliding_window": 4096,
    }

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU,
                 max_seq_len: int | None = None,
                 tp_size: int = 1, tp_rank: int = 0,
                 dist_backend: DistBackend | None = None,
                 comm_handle=None):
        """创建 Qwen2 模型实例.

        Args:
            model_path: 模型权重目录路径.
            device: 设备类型 (CPU / NVIDIA).
            max_seq_len: 最大序列长度 (None 则从 config.json 推导).
            tp_size: 张量并行度 (默认 1 = 单卡).
            tp_rank: 当前进程的 TP rank (0-based).
            dist_backend: 分布式后端 (None 则在 tp_size>1 时自动选 NCCL, 否则不创建).
            comm_handle: 外部已创建的 comm 句柄 (覆盖自动创建).
        """
        model_path = Path(model_path)

        # ─── 从 config.json 读取架构参数 ───
        cfg = self._load_model_config(model_path)

        num_hidden_layers   = cfg.get("num_hidden_layers",   self._DEFAULT_CONFIG["num_hidden_layers"])
        hidden_size         = cfg.get("hidden_size",         self._DEFAULT_CONFIG["hidden_size"])
        num_attention_heads = cfg.get("num_attention_heads", self._DEFAULT_CONFIG["num_attention_heads"])
        num_key_value_heads = cfg.get("num_key_value_heads", self._DEFAULT_CONFIG["num_key_value_heads"])
        intermediate_size   = cfg.get("intermediate_size",   self._DEFAULT_CONFIG["intermediate_size"])
        vocab_size          = cfg.get("vocab_size",          self._DEFAULT_CONFIG["vocab_size"])
        rms_norm_eps        = cfg.get("rms_norm_eps",        self._DEFAULT_CONFIG["rms_norm_eps"])
        rope_theta          = cfg.get("rope_theta",          self._DEFAULT_CONFIG["rope_theta"])

        # eos_token_id 可能是 int 或 list
        eos_raw = cfg.get("eos_token_id", self._DEFAULT_CONFIG["eos_token_id"])
        end_token = eos_raw[0] if isinstance(eos_raw, list) else int(eos_raw)

        # maxseq：优先用户显式指定 > sliding_window (如果 use_sliding_window != false)
        #       > 上限 max_position_embeddings (截断到 32768 防爆显存)
        if max_seq_len is not None:
            maxseq = max_seq_len
        else:
            # 仅在 use_sliding_window 不为 false 时使用 sliding_window
            use_sw = cfg.get("use_sliding_window", True)  # 默认 True (旧模型没有此字段)
            sw = cfg.get("sliding_window")
            if use_sw and sw is not None and isinstance(sw, int) and sw > 0:
                maxseq = sw
            else:
                maxseq = min(cfg.get("max_position_embeddings",
                                     self._DEFAULT_CONFIG["max_position_embeddings"]),
                             32768)

        head_dim = hidden_size // num_attention_heads

        print(f"[Qwen2] Architecture from config.json:")
        print(f"  layers={num_hidden_layers}, hidden={hidden_size}, heads={num_attention_heads}, "
              f"kv_heads={num_key_value_heads}, head_dim={head_dim}")
        print(f"  intermediate={intermediate_size}, vocab={vocab_size}, maxseq={maxseq}")
        print(f"  rope_theta={rope_theta}, rms_eps={rms_norm_eps}, eos={end_token}")

        # ─── 填充 meta 结构体 ───
        meta = LlaisysQwen2Meta()
        meta.dtype     = 13          # FP32 计算精度
        meta.nlayer    = num_hidden_layers
        meta.hs        = hidden_size
        meta.nh        = num_attention_heads
        meta.nkvh      = num_key_value_heads
        meta.dh        = head_dim
        meta.di        = intermediate_size
        meta.maxseq    = maxseq
        meta.voc       = vocab_size
        meta.epsilon   = rms_norm_eps
        meta.theta     = rope_theta
        meta.end_token = end_token

        print("Creating Qwen2 Model instance...")
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._comm_handle = None   # 分布式通信句柄
        self._owns_comm = False    # 是否由本实例创建 (析构时需销毁)

        if tp_size > 1:
            # TP 模式: 使用 CreateTP
            device_id = tp_rank  # 默认每个 rank 对应一张卡
            self.model_handle = model_create_tp(
                ctypes.byref(meta), device.value, device_id, tp_size, tp_rank)

            # 创建或绑定通信句柄
            if comm_handle is not None:
                self._comm_handle = comm_handle
                self._owns_comm = False
            elif dist_backend is not None:
                cfg = LlaisysDistConfig()
                cfg.backend = int(dist_backend)
                cfg.world_size = tp_size
                cfg.rank = tp_rank
                cfg.local_device = device_id
                self._comm_handle = comm_create(cfg)
                self._owns_comm = True
            else:
                # tp_size>1 但没指定后端 → 自动用 NCCL (若可用), 否则 MOCK
                if backend_available(int(DistBackend.NCCL)):
                    auto_backend = DistBackend.NCCL
                else:
                    auto_backend = DistBackend.MOCK
                    print(f"[Qwen2] WARNING: NCCL not available, using MOCK backend for TP")
                cfg = LlaisysDistConfig()
                cfg.backend = int(auto_backend)
                cfg.world_size = tp_size
                cfg.rank = tp_rank
                cfg.local_device = device_id
                self._comm_handle = comm_create(cfg)
                self._owns_comm = True

            # 绑定到模型
            model_set_comm(self.model_handle, self._comm_handle)
            print(f"[Qwen2] TP mode: tp_size={tp_size}, tp_rank={tp_rank}")
        else:
            self.model_handle = model_create(ctypes.byref(meta), device.value, None, 0)

        # 保存配置供后续使用
        self._config = cfg
        self._end_token = end_token
        self._device_type = device

        # 加载权重
        print(f"Loading weights from {model_path}...")
        self._load_weights(model_path)
        print("Model loaded successfully.")

    @staticmethod
    def _load_model_config(model_path: Path) -> dict:
        """从 model_path/config.json 读取模型架构配置.

        Returns:
            dict: 解析后的配置字典, 若文件不存在则返回空 dict.
        """
        import json as _json
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = _json.load(f)
            print(f"[Qwen2] Loaded config from {config_path}")
            return cfg
        print(f"[Qwen2] WARNING: {config_path} not found, using default config")
        return {}

    # =====================================================================
    # Conversion cache helpers
    # =====================================================================

    @staticmethod
    def _cache_dir(model_path: Path) -> Path:
        return model_path / ".llaisys_cache"

    @staticmethod
    def _compute_model_fingerprint(model_path: Path) -> str:
        """Compute a fingerprint from original safetensors files (name + size)."""
        h = hashlib.sha256()
        for f in sorted(model_path.glob("*.safetensors")):
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
        # Also include config if present
        cfg_path = model_path / "config.json"
        if cfg_path.exists():
            h.update(cfg_path.read_bytes())
        return h.hexdigest()[:16]

    def _try_load_from_cache(self, model_path: Path) -> bool:
        """Try to load converted weights from cache. Returns True if successful."""
        cache_dir = self._cache_dir(model_path)
        meta_path = cache_dir / "cache_meta.json"
        if not meta_path.exists():
            return False

        with open(meta_path) as f:
            meta = _json.load(f)

        # Validate fingerprint
        current_fp = self._compute_model_fingerprint(model_path)
        if meta.get("fingerprint") != current_fp:
            print(f"[Cache] Fingerprint mismatch, cache invalidated")
            return False

        # Validate AWQ mode compatibility: native AWQ caches only non-quantized
        # tensors, so such a cache is incomplete for FP16 conversion mode.
        cached_awq_native = meta.get("awq_native", False)
        if cached_awq_native:
            print(f"[Cache] Cache was created in AWQ native mode (incomplete for FP16), skipping")
            return False

        cache_files = sorted(cache_dir.glob("weights_*.safetensors"))
        if not cache_files:
            return False

        print(f"[Cache] Loading {len(cache_files)} cached file(s) from {cache_dir}")
        t0 = time.time()

        dtype_map = meta.get("dtype_map", {})
        for cache_file in cache_files:
            with safetensors.safe_open(cache_file, framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    tensor = sf.get_tensor(name)
                    dtype_enum = dtype_map.get(name, 13)  # default FP32
                    self._call_load_weight(name, tensor, dtype_enum=dtype_enum)

        elapsed = time.time() - t0
        print(f"[Cache] Loaded in {elapsed:.1f}s (vs ~{meta.get('convert_time', '?')}s conversion)")
        return True

    @staticmethod
    def _save_to_cache(model_path: Path, tensors: dict, dtype_map: dict, convert_time: float):
        """Save converted tensors to cache directory.

        Args:
            model_path: Original model directory.
            tensors: dict of {name: tensor} to cache.
            dtype_map: dict of {name: dtype_enum} for each tensor.
            convert_time: Time spent on conversion (for display).
        """
        cache_dir = Qwen2._cache_dir(model_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Split into chunks of ~2GB to avoid single huge files
        MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024
        shard_idx = 0
        current_shard = {}
        current_bytes = 0

        sorted_names = sorted(tensors.keys())
        for name in sorted_names:
            t = tensors[name]
            nbytes = t.nelement() * t.element_size()
            if current_bytes > 0 and current_bytes + nbytes > MAX_SHARD_BYTES:
                # Flush current shard
                shard_path = cache_dir / f"weights_{shard_idx:04d}.safetensors"
                safetensors.torch.save_file(current_shard, str(shard_path))
                print(f"  [Cache] Saved shard {shard_path.name} ({current_bytes / 1024**3:.2f} GB)")
                shard_idx += 1
                current_shard = {}
                current_bytes = 0
            current_shard[name] = t
            current_bytes += nbytes

        if current_shard:
            shard_path = cache_dir / f"weights_{shard_idx:04d}.safetensors"
            safetensors.torch.save_file(current_shard, str(shard_path))
            print(f"  [Cache] Saved shard {shard_path.name} ({current_bytes / 1024**3:.2f} GB)")

        # Write metadata
        fingerprint = Qwen2._compute_model_fingerprint(model_path)
        meta = {
            "fingerprint": fingerprint,
            "convert_time": f"{convert_time:.1f}",
            "num_shards": shard_idx + 1,
            "num_tensors": len(tensors),
            "dtype_map": dtype_map,
        }
        meta_path = cache_dir / "cache_meta.json"
        with open(meta_path, "w") as f:
            _json.dump(meta, f, indent=2)
        print(f"  [Cache] Metadata saved to {meta_path}")

    def _get_dtype_enum(self, dtype_str):
        if "float32" in dtype_str: return 13
        if "float16" in dtype_str: return 12
        if "bfloat16" in dtype_str: return 19
        if "int64" in dtype_str: return 6
        if "int32" in dtype_str: return 5
        if "int8" in dtype_str: return 3
        return 0

    # =====================================================================
    # GPTQ/AWQ format helpers
    # =====================================================================

    @staticmethod
    def _unpack_int32_to_int4(packed_int32, bits=4):
        """Unpack int32-packed values to individual uint8 values.

        GPTQ packing: 8 × 4-bit values per int32 (for bits=4).
        qweight shape [rows_packed, cols] → [rows_packed * pack_factor, cols]
        qzeros  shape [num_groups, cols_packed] → [num_groups, cols_packed * pack_factor]
        """
        pack_factor = 32 // bits  # 8 for 4-bit
        mask = (1 << bits) - 1    # 0xF for 4-bit

        results = []
        for i in range(pack_factor):
            results.append(((packed_int32 >> (i * bits)) & mask).to(torch.int32))
        # Interleave along the packed axis
        return results, pack_factor

    @staticmethod
    def _convert_gptq_layer(qweight, qzeros, scales, bits=4, group_size=128):
        """Convert one GPTQ linear layer to FP16 (no double quantization).

        GPTQ packing: input dimension is packed.
            qweight: [in_features // pack_factor, out_features] int32
            qzeros:  [num_groups, out_features // pack_factor] int32
            scales:  [num_groups, out_features] float16/float32

        Returns:
            weight_f16:  [out_features, in_features] float16
        """
        pack_factor = 32 // bits  # 8 for 4-bit
        mask = (1 << bits) - 1    # 0xF

        in_packed, out_features = qweight.shape
        in_features = in_packed * pack_factor
        num_groups = scales.shape[0]

        # --- Step 1: Unpack qweight [in_packed, out] → [in, out] (vectorized) ---
        shifts = torch.arange(pack_factor, dtype=torch.int32) * bits  # [pack_factor]
        # qweight: [in_packed, out] → [in_packed, 1, out] broadcast with shifts [pack_factor]
        w_unpacked = ((qweight.unsqueeze(1) >> shifts.reshape(1, -1, 1)) & mask)  # [in_packed, pack_factor, out]
        w_unpacked = w_unpacked.reshape(in_features, out_features)  # [in, out]

        # --- Step 2: Unpack qzeros [ngroup, out_packed] → [ngroup, out] (vectorized) ---
        z_unpacked = ((qzeros.unsqueeze(-1) >> shifts) & mask)  # [ngroup, out_packed, pack_factor]
        z_unpacked = z_unpacked.reshape(num_groups, out_features)  # [ngroup, out]

        # --- Step 3: Dequantize to FP32 (vectorized) ---
        # AutoGPTQ stores zero_point - 1, so we add 1 back
        w_grouped = w_unpacked.reshape(num_groups, group_size, out_features).float()
        z_exp = (z_unpacked.float() + 1.0).unsqueeze(1)  # [ngroup, 1, out] (+1 GPTQ correction)
        s_exp = scales.float().unsqueeze(1)               # [ngroup, 1, out]
        w_float = ((w_grouped - z_exp) * s_exp).reshape(in_features, out_features)

        # --- Step 4: Transpose to our [out, in] layout & convert to FP16 ---
        return w_float.T.contiguous().to(torch.float16)

    @staticmethod
    def _convert_awq_layer(qweight, qzeros, scales, bits=4, group_size=128):
        """Convert one AWQ GEMM format linear layer to FP16 (no double quantization).

        AWQ GEMM packing: output dimension is packed with interleaved bit order.
            qweight: [in_features, out_features // pack_factor] int32
            qzeros:  [num_groups, out_features // pack_factor] int32
            scales:  [num_groups, out_features] float16/float32

        AWQ GEMM uses interleaved packing order [0,4,1,5,2,6,3,7] for the 8
        INT4 values within each int32, corresponding to bit shifts
        [0, 16, 4, 20, 8, 24, 12, 28].

        Returns:
            weight_f16:  [out_features, in_features] float16
        """
        pack_factor = 32 // bits  # 8 for 4-bit
        mask = (1 << bits) - 1    # 0xF

        in_features, out_packed = qweight.shape
        out_features = out_packed * pack_factor
        num_groups = scales.shape[0]

        # AWQ GEMM interleaved bit shifts: element order [0,4,1,5,2,6,3,7]
        shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)

        # --- Step 1: Unpack qweight [in, out_packed] → [in, out] (vectorized) ---
        w_unpacked = ((qweight.unsqueeze(-1) >> shifts) & mask)  # [in, out_packed, pack_factor]
        w_unpacked = w_unpacked.reshape(in_features, out_features)  # [in, out]

        # --- Step 2: Unpack qzeros [ngroup, out_packed] → [ngroup, out] (vectorized) ---
        z_unpacked = ((qzeros.unsqueeze(-1) >> shifts) & mask)  # [ngroup, out_packed, pack_factor]
        z_unpacked = z_unpacked.reshape(num_groups, out_features)  # [ngroup, out]

        # --- Step 3: Dequantize to FP32 (fully vectorized, no Python loop) ---
        w_grouped = w_unpacked.reshape(num_groups, group_size, out_features).float()
        z_exp = z_unpacked.float().unsqueeze(1)       # [ngroup, 1, out]
        s_exp = scales.float().unsqueeze(1)            # [ngroup, 1, out]
        w_float = ((w_grouped - z_exp) * s_exp).reshape(in_features, out_features)

        # --- Step 4: Transpose to our [out, in] layout & convert to FP16 ---
        return w_float.T.contiguous().to(torch.float16)

    @staticmethod
    def _requantize_to_int4(w_float, group_size):
        """Re-quantize FP32 weight [out, in] to our symmetric INT4 packed format.

        Returns:
            packed:  [out_features, in_features // 2] uint8
            scale:   [out_features, num_groups] float32
        """
        rows, cols = w_float.shape
        assert cols % group_size == 0, f"in_features {cols} not divisible by group_size {group_size}"
        ngroups = cols // group_size

        w_grouped = w_float.reshape(rows, ngroups, group_size)
        group_max = w_grouped.abs().amax(dim=-1).clamp(min=1e-10)
        our_scale = group_max / 7.0

        w_q = (w_grouped / our_scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
        w_q = w_q.reshape(rows, cols)

        # Pack two INT4 per byte
        w_even = (w_q[:, 0::2] + 8).to(torch.uint8)
        w_odd  = (w_q[:, 1::2] + 8).to(torch.uint8)
        packed = (w_odd << 4) | (w_even & 0x0F)

        return packed, our_scale.float()

    def _call_load_weight(self, name, tensor, dtype_enum):
        """Helper to call the C load_weight function with proper ctypes."""
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        c_name = name.encode('utf-8')
        data_ptr = tensor.data_ptr()
        ndim = len(tensor.shape)
        shape_array = (ctypes.c_int64 * ndim)(*tensor.shape)
        load_weight(self.model_handle, c_name, ctypes.c_void_p(data_ptr), ndim, shape_array, dtype_enum)

    def _load_weights(self, model_path):
        import json
        
        # ─── Detect quantization format ───
        quant_config_path = model_path / "quant_config.json"       # Our native format
        gptq_config_path = model_path / "quantize_config.json"     # GPTQ / AWQ standalone
        config_json_path = model_path / "config.json"              # HF config (may embed GPTQ config)

        # GPTQ/AWQ: standalone quantize_config.json OR embedded in config.json
        gcfg = None
        if gptq_config_path.exists():
            with open(gptq_config_path) as f:
                gcfg = json.load(f)
        elif config_json_path.exists():
            with open(config_json_path) as f:
                main_cfg = json.load(f)
            if "quantization_config" in main_cfg:
                qc = main_cfg["quantization_config"]
                if qc.get("quant_method") in ("gptq", "awq"):
                    gcfg = qc

        if gcfg is not None:
            quant_method = gcfg.get("quant_method", "gptq")
            bits = gcfg.get("bits", 4)
            group_size = gcfg.get("group_size", 128)
            desc_act = gcfg.get("desc_act", False)
            if desc_act:
                print("WARNING: desc_act=True 暂不支持, 视为 False (可能影响精度)")
            print(f"Detected {quant_method.upper()} model: {bits}-bit, group_size={group_size}")
            self._load_weights_gptq(model_path, bits, group_size, quant_method=quant_method)
            return

        # ─── Our native format (INT8 / INT4 / FP32) ───
        is_quantized = quant_config_path.exists()
        quant_bits = 0
        if is_quantized:
            with open(quant_config_path) as f:
                qcfg = json.load(f)
            quant_bits = qcfg.get("bits", 8)
            print(f"Detected quantized model: {qcfg.get('quant_method', 'unknown')} (bits={quant_bits})")
        
        files = sorted(list(model_path.glob("*.safetensors")))
        if not files:
            print(f"Warning: No .safetensors files found in {model_path}")
            
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                for name_ in data_.keys():
                    tensor = data_.get_tensor(name_)
                    
                    c_name = name_.encode('utf-8')
                    
                    # 确保内存连续
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    
                    # 根据 tensor dtype 决定如何传给 C++
                    if tensor.dtype == torch.int8:
                        # INT8 量化权重 — 原样传递
                        dtype_enum = 3  # LLAISYS_DTYPE_I8
                    elif tensor.dtype == torch.uint8:
                        # INT4 packed 量化权重 — 原样传递  (U8)
                        dtype_enum = 7  # LLAISYS_DTYPE_U8
                    elif name_.endswith(".scale"):
                        # scale 向量/矩阵 — 确保 FP32
                        if tensor.dtype != torch.float32:
                            tensor = tensor.to(torch.float32)
                        dtype_enum = 13  # LLAISYS_DTYPE_F32
                    else:
                        # 普通权重 — GPU 用 FP16, CPU 用 FP32
                        # BF16 原始权重直接转 FP16 (精度损失可忽略, 值域安全)
                        import os
                        use_fp16_weights = (self._device_type != DeviceType.CPU
                                           and os.environ.get("LLAISYS_FORCE_FP32", "0") != "1")
                        if use_fp16_weights:
                            tensor = tensor.to(torch.float16)
                            dtype_enum = 12  # LLAISYS_DTYPE_F16
                        else:
                            if tensor.dtype != torch.float32:
                                tensor = tensor.to(torch.float32)
                            dtype_enum = 13  # LLAISYS_DTYPE_F32
                        
                    data_ptr = tensor.data_ptr()
                    ndim = len(tensor.shape)
                    shape_array = (ctypes.c_int64 * ndim)(*tensor.shape)
                    
                    load_weight(self.model_handle, c_name, ctypes.c_void_p(data_ptr), ndim, shape_array, dtype_enum)

    def _load_weights_gptq(self, model_path, bits=4, group_size=128, quant_method="gptq"):
        """Load GPTQ/AWQ format model, converting to our symmetric INT4 at load time.

        Supports conversion result caching: first load converts and saves to
        .llaisys_cache/, subsequent loads read directly from cache.

        GPTQ packing (input packed):
          .qweight → [in_features // 8, out_features] int32
        AWQ GEMM packing (output packed):
          .qweight → [in_features, out_features // 8] int32
        Both share:
          .qzeros  → [num_groups, out_features // 8] int32
          .scales  → [num_groups, out_features] float16
          .g_idx   → [in_features] int32 (optional, ignored)

        Non-linear tensors (embed, norm, bias) are loaded as FP32.
        """
        is_awq = (quant_method == "awq")
        # Native AWQ kernel: pass raw I32 data to C++ GPU kernel instead of CPU→FP16 conversion.
        # Set LLAISYS_AWQ_NATIVE=0 to fall back to CPU FP16 conversion.
        import os
        use_native_awq = is_awq and os.environ.get("LLAISYS_AWQ_NATIVE", "1") != "0"

        # ─── Try loading from cache first (only for CPU conversion mode) ───
        if not use_native_awq and self._try_load_from_cache(model_path):
            print(f"  [Cache] Successfully loaded from cache, skipping conversion")
            return

        t_convert_start = time.time()
        convert_fn = self._convert_awq_layer if is_awq else self._convert_gptq_layer
        files = sorted(list(model_path.glob("*.safetensors")))
        if not files:
            print(f"Warning: No .safetensors files found in {model_path}")
            return

        # First pass: collect all tensors into dict
        all_tensors = {}
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    all_tensors[name] = f.get_tensor(name)

        processed = set()
        converted_count = 0
        cache_tensors = {}   # {name: tensor} for cache saving
        cache_dtypes = {}    # {name: dtype_enum}

        for name in sorted(all_tensors.keys()):
            if name in processed:
                continue
            tensor = all_tensors[name]

            if name.endswith(".qweight"):
                # ─── GPTQ quantized linear layer ───
                base = name[:-len(".qweight")]
                qweight = tensor
                qzeros = all_tensors.get(base + ".qzeros")
                scales = all_tensors.get(base + ".scales")

                if qzeros is None or scales is None:
                    print(f"  WARNING: Missing qzeros/scales for {base}, loading as FP32")
                    continue

                # Map to our weight naming convention
                weight_name = base + ".weight"

                if is_awq:
                    if use_native_awq:
                        # Native AWQ: pass raw packed I32 qweight + qzeros + FP32 scales to C++
                        # C++ kernel will dequantize on GPU at inference time.
                        # C++ LoadWeightByName handles TP slicing for AWQ layout
                        # (inverts ColDim0↔RowDim1 since AWQ is [in, out_packed]).
                        qw_i32 = qweight.to(torch.int32).contiguous()
                        qz_i32 = qzeros.to(torch.int32).contiguous()
                        sc_f32 = scales.to(torch.float32).contiguous()

                        self._call_load_weight(weight_name, qw_i32, dtype_enum=5)  # I32
                        self._call_load_weight(weight_name + ".qzeros", qz_i32, dtype_enum=5)  # I32
                        self._call_load_weight(weight_name + ".scale", sc_f32, dtype_enum=13)  # F32
                        vram_mb = (qw_i32.nelement() + qz_i32.nelement()) * 4 / (1024 * 1024)
                        tag = "AWQ-native"
                        print(f"  [{tag}] {weight_name}: qweight{list(qw_i32.shape)} + qzeros{list(qz_i32.shape)} + scales{list(sc_f32.shape)} ({vram_mb:.0f} MB)")
                    else:
                        # AWQ → FP16 dequantized weight (CPU conversion, no double quantization)
                        w_f16 = convert_fn(qweight, qzeros, scales, bits=bits, group_size=group_size)
                        self._call_load_weight(weight_name, w_f16, dtype_enum=12)  # FP16
                        cache_tensors[weight_name] = w_f16
                        cache_dtypes[weight_name] = 12
                        vram_mb = w_f16.nelement() * 2 / (1024 * 1024)
                        tag = "AWQ→FP16"
                        print(f"  [{tag}] {weight_name}: {list(w_f16.shape)} ({vram_mb:.0f} MB)")
                else:
                    # GPTQ → FP16 dequantized weight (no double quantization)
                    w_f16 = convert_fn(qweight, qzeros, scales, bits=bits, group_size=group_size)
                    self._call_load_weight(weight_name, w_f16, dtype_enum=12)  # FP16
                    cache_tensors[weight_name] = w_f16
                    cache_dtypes[weight_name] = 12
                    vram_mb = w_f16.nelement() * 2 / (1024 * 1024)
                    tag = "GPTQ→FP16"
                    print(f"  [{tag}] {weight_name}: {list(w_f16.shape)} ({vram_mb:.0f} MB)")

                processed.update([name, base + ".qzeros", base + ".scales"])
                if base + ".g_idx" in all_tensors:
                    processed.add(base + ".g_idx")
                if base + ".bias" in all_tensors:
                    # GPTQ linear may have bias — load as FP32
                    bias = all_tensors[base + ".bias"]
                    if bias.dtype != torch.float32:
                        bias = bias.to(torch.float32)
                    self._call_load_weight(base + ".bias", bias, dtype_enum=13)
                    cache_tensors[base + ".bias"] = bias
                    cache_dtypes[base + ".bias"] = 13
                    processed.add(base + ".bias")

                converted_count += 1

            elif name.endswith((".qzeros", ".scales", ".g_idx")):
                # Handled together with .qweight
                continue

            else:
                # ─── Regular tensor (embedding, norm, bias, lm_head) ───

                # lm_head.weight in GPTQ is sometimes still quantized
                if name == "lm_head.weight" and "lm_head.qweight" in all_tensors:
                    continue  # handled above

                # For large FP-only tensors (embedding, lm_head), keep as FP16
                # to save significant VRAM (each is [vocab, hidden] ≈ 1 GB saved)
                # Set _USE_FP16_LARGE = True to enable (requires mixed-precision kernel support)
                _USE_FP16_LARGE = True
                _FP16_ELIGIBLE = {"model.embed_tokens.weight", "lm_head.weight"}
                if _USE_FP16_LARGE and name in _FP16_ELIGIBLE:
                    if tensor.dtype != torch.float16:
                        tensor = tensor.to(torch.float16)
                    self._call_load_weight(name, tensor, dtype_enum=12)  # FP16
                    cache_tensors[name] = tensor
                    cache_dtypes[name] = 12
                    processed.add(name)
                    vram_mb = tensor.nelement() * 2 / (1024 * 1024)
                    print(f"  [FP16]  {name}: {list(tensor.shape)} → float16 ({vram_mb:.0f} MB)")
                else:
                    if tensor.dtype != torch.float32:
                        tensor = tensor.to(torch.float32)
                    self._call_load_weight(name, tensor, dtype_enum=13)  # FP32
                    cache_tensors[name] = tensor
                    cache_dtypes[name] = 13
                    processed.add(name)
                    print(f"  [KEEP]  {name}: {list(tensor.shape)} → float32")

        convert_time = time.time() - t_convert_start
        print(f"  Converted {converted_count} {quant_method.upper()} layers ({convert_time:.1f}s)")

        # ─── Save to cache for next time (only FP16 conversion mode) ───
        if not use_native_awq:
            try:
                self._save_to_cache(model_path, cache_tensors, cache_dtypes, convert_time)
                print(f"  [Cache] Conversion results cached — next load will be ~10× faster")
            except Exception as e:
                print(f"  [Cache] WARNING: Failed to save cache: {e}")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if not inputs:
            return []

        output_ids = list(inputs)
        max_tokens = max_new_tokens if max_new_tokens is not None else 20
        
        use_sampling = (top_k != 1) and (temperature > 0.0)
        
        # --- 1. Prefill ---
        input_len = len(inputs)
        input_np = np.array(inputs, dtype=np.int64)
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)
        input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        if use_sampling:
            next_token = model_infer_sample(
                self.model_handle, input_ptr, ctypes.c_size_t(input_len),
                ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
        else:
            next_token = model_infer(self.model_handle, input_ptr, ctypes.c_size_t(input_len))

        output_ids.append(int(next_token))
        current_token = next_token
        
        # --- 2. Decoding ---
        for _ in range(max_tokens - 1):
            token_np = np.array([current_token], dtype=np.int64)
            token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            
            if use_sampling:
                next_token = model_infer_sample(
                    self.model_handle, token_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
            else:
                next_token = model_infer(self.model_handle, token_ptr, ctypes.c_size_t(1))

            output_ids.append(int(next_token))
            current_token = next_token
            
            if next_token == self._end_token:  # EOS
                break

        return output_ids

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Generator that yields one token at a time."""
        if not inputs:
            return

        max_tokens = max_new_tokens if max_new_tokens is not None else 512
        use_sampling = (top_k != 1) and (temperature > 0.0)
        
        # Prefill
        input_len = len(inputs)
        input_np = np.array(inputs, dtype=np.int64)
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)
        input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        if use_sampling:
            next_token = model_infer_sample(
                self.model_handle, input_ptr, ctypes.c_size_t(input_len),
                ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
        else:
            next_token = model_infer(self.model_handle, input_ptr, ctypes.c_size_t(input_len))

        next_token = int(next_token)
        yield next_token
        current_token = next_token
        
        # Decode
        for _ in range(max_tokens - 1):
            if current_token == self._end_token:  # EOS
                break
            token_np = np.array([current_token], dtype=np.int64)
            token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            
            if use_sampling:
                next_token = model_infer_sample(
                    self.model_handle, token_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
            else:
                next_token = model_infer(self.model_handle, token_ptr, ctypes.c_size_t(1))

            next_token = int(next_token)
            yield next_token
            current_token = next_token

    def __del__(self):
        if hasattr(self, 'model_handle') and self.model_handle:
            model_destroy(self.model_handle)
            self.model_handle = None
        # 清理 comm (仅当由本实例创建时)
        if hasattr(self, '_comm_handle') and self._comm_handle and self._owns_comm:
            comm_destroy(self._comm_handle)
            self._comm_handle = None

    def reset_cache(self):
        """Reset the KV-cache position without reloading weights."""
        model_reset_cache(self.model_handle)

    # ==========================================
    # 张量并行 (TP) 属性
    # ==========================================

    @property
    def tp_size(self) -> int:
        """返回当前 TP 并行度."""
        return self._tp_size

    @property
    def tp_rank(self) -> int:
        """返回当前 TP rank."""
        return self._tp_rank

    @property
    def is_tp(self) -> bool:
        """是否处于 TP 模式 (tp_size > 1)."""
        return self._tp_size > 1

    # ==========================================
    # Phase 4: KV-Cache 高级接口
    # ==========================================

    def save_cache(self):
        """保存当前 KV-Cache 快照 (深拷贝到 CPU).
        
        Returns:
            int: 快照句柄 (C++ 指针), 如果 cache 为空则返回 None.
        """
        handle = cache_save(self.model_handle)
        if not handle:
            return None
        return handle

    def restore_cache(self, snapshot_handle):
        """从快照恢复 KV-Cache.
        
        Args:
            snapshot_handle: save_cache() 返回的快照句柄.
        """
        if snapshot_handle:
            cache_restore(self.model_handle, snapshot_handle)

    def truncate_cache(self, pos: int):
        """截断 KV-Cache 到指定位置.
        
        Args:
            pos: 目标位置 (0 = 清空, pos <= current_pos).
        """
        cache_truncate(self.model_handle, ctypes.c_int64(pos))

    def get_cache_pos(self) -> int:
        """获取当前 KV-Cache 位置 (已处理的 token 数)."""
        return int(cache_get_pos(self.model_handle))

    @staticmethod
    def destroy_snapshot(snapshot_handle):
        """释放快照内存.
        
        Args:
            snapshot_handle: save_cache() 返回的快照句柄.
        """
        if snapshot_handle:
            cache_snapshot_destroy(snapshot_handle)

    # ==========================================
    # Phase 5 (项目#4): 批量推理 API
    # ==========================================

    def create_batch_context(self, max_batch_size: int = 8, max_seq_per_slot: int = 2048):
        """创建批量推理上下文 (预分配 max_batch_size 个 KV-Cache slot).
        
        Args:
            max_batch_size: 最大并发请求数.
            max_seq_per_slot: 每个 slot 的 KV-Cache 最大序列长度.
        
        Returns:
            BatchContext 对象.
        """
        return BatchContext(self, max_batch_size, max_seq_per_slot)


class BatchContext:
    """批量推理上下文: 管理多个 KV-Cache slot, 支持批量 decode.
    
    Usage:
        ctx = model.create_batch_context(max_batch_size=8)
        
        # Prefill slot 0
        first_tok = ctx.prefill(slot_id=0, token_ids=[...], temperature=0.8)
        
        # Batch decode
        next_tokens = ctx.decode(
            active_slots=[0, 1, 2],
            current_tokens=[tok0, tok1, tok2],
            temperature=0.8
        )
        
        # Save/restore slot cache
        snapshot = ctx.slot_save(0)
        ctx.slot_restore(1, snapshot)
    """

    def __init__(self, model: Qwen2, max_batch_size: int = 8, max_seq_per_slot: int = 2048):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_per_slot = max_seq_per_slot
        self._native = None
        self._handle = None
        if _native.available:
            self._native = _native.Qwen2BatchRuntime(
                int(model.model_handle), max_batch_size, max_seq_per_slot
            )
        else:
            self._handle = batch_context_create(
                model.model_handle,
                ctypes.c_size_t(max_batch_size),
                ctypes.c_size_t(max_seq_per_slot),
            )
            if not self._handle:
                raise RuntimeError("Failed to create batch context")
        self._end_token = model._end_token

    def __del__(self):
        if hasattr(self, '_native') and self._native is not None:
            self._native = None
        if hasattr(self, '_handle') and self._handle:
            batch_context_destroy(self._handle)
            self._handle = None

    def slot_reset(self, slot_id: int):
        """重置指定 slot 的 KV-Cache."""
        if self._native is not None:
            self._native.reset(slot_id)
            return
        batch_slot_reset(self._handle, ctypes.c_size_t(slot_id))

    def prefill(
        self,
        slot_id: int,
        token_ids: Sequence[int],
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> int:
        """在指定 slot 上执行 prefill, 返回首个生成 token.
        
        Args:
            slot_id: slot 索引 (0 ~ max_batch_size-1).
            token_ids: 完整 prompt 的 token ID 序列.
            temperature: 采样温度.
            top_k: Top-K 采样.
            top_p: Top-P 采样.
            
        Returns:
            int: 首个生成的 token ID.
        """
        if self._native is not None:
            sampling = _native.SamplingParams()
            sampling.temperature = temperature
            sampling.top_k = top_k
            sampling.top_p = top_p
            return int(self._native.prefill(slot_id, list(token_ids), sampling))
        token_np = np.array(token_ids, dtype=np.int64)
        if not token_np.flags['C_CONTIGUOUS']:
            token_np = np.ascontiguousarray(token_np)
        token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        result = batch_prefill(
            self._handle,
            ctypes.c_size_t(slot_id),
            token_ptr,
            ctypes.c_size_t(len(token_ids)),
            ctypes.c_float(temperature),
            ctypes.c_int(top_k),
            ctypes.c_float(top_p),
        )
        return int(result)

    def prefill_chunk(
        self,
        slot_id: int,
        token_ids: Sequence[int],
        start_pos: int,
        is_last_chunk: bool = False,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> int | None:
        """Append a prompt chunk to an existing paged-cache slot.

        The first chunk uses ``start_pos=0`` after ``slot_reset``. Subsequent
        chunks must use the slot position produced by the preceding chunk.
        Only the final chunk returns a sampled token.
        """
        if batch_prefill_chunk is None:
            raise RuntimeError("native library does not support incremental prefill")
        if not 0 <= slot_id < self.max_batch_size:
            raise ValueError("slot_id is out of range")
        if start_pos < 0 or not token_ids:
            raise ValueError("start_pos must be non-negative and token_ids non-empty")
        if self._native is not None:
            sampling = _native.SamplingParams()
            sampling.temperature = temperature
            sampling.top_k = top_k
            sampling.top_p = top_p
            result = self._native.prefill_chunk(
                slot_id, list(token_ids), start_pos, is_last_chunk, sampling
            )
            return int(result) if is_last_chunk else None
        token_np = np.ascontiguousarray(token_ids, dtype=np.int64)
        result = batch_prefill_chunk(
            self._handle,
            ctypes.c_size_t(slot_id),
            token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(len(token_np)),
            ctypes.c_int64(start_pos),
            ctypes.c_int(bool(is_last_chunk)),
            ctypes.c_float(temperature),
            ctypes.c_int(top_k),
            ctypes.c_float(top_p),
        )
        if is_last_chunk and result < 0:
            raise RuntimeError("incremental prefill failed")
        return int(result) if is_last_chunk else None

    def prefix_lookup(self, slot_id: int, token_ids: Sequence[int]) -> int:
        """Attach reusable complete cache blocks and return matched tokens."""
        if self._native is not None:
            return int(self._native.prefix_lookup(slot_id, list(token_ids)))
        if batch_prefix_lookup is None:
            return 0
        token_np = np.ascontiguousarray(token_ids, dtype=np.int64)
        if token_np.size == 0:
            return 0
        return int(batch_prefix_lookup(
            self._handle, ctypes.c_size_t(slot_id),
            token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(token_np.size),
        ))

    def prefix_publish(self, slot_id: int, token_ids: Sequence[int]) -> bool:
        """Publish complete computed prompt blocks to the shared prefix cache."""
        if self._native is not None:
            return bool(self._native.prefix_publish(slot_id, list(token_ids)))
        if batch_prefix_publish is None:
            return False
        token_np = np.ascontiguousarray(token_ids, dtype=np.int64)
        if token_np.size == 0:
            return False
        return bool(batch_prefix_publish(
            self._handle, ctypes.c_size_t(slot_id),
            token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(token_np.size),
        ))

    def decode(
        self,
        active_slots: Sequence[int],
        current_tokens: Sequence[int],
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> list[int]:
        """对活跃 slots 执行一步批量 decode.
        
        Args:
            active_slots: 活跃 slot ID 列表.
            current_tokens: 各 slot 当前 token 列表 (与 active_slots 一一对应).
            temperature: 采样温度.
            top_k: Top-K 采样.
            top_p: Top-P 采样.
            
        Returns:
            list[int]: 各 slot 的 next token 列表.
        """
        num_active = len(active_slots)
        assert len(current_tokens) == num_active, "active_slots and current_tokens must have same length"
        
        slots_arr = (ctypes.c_size_t * num_active)(*active_slots)
        tokens_arr = (ctypes.c_int64 * num_active)(*current_tokens)
        output_arr = (ctypes.c_int64 * num_active)()
        
        batch_decode(
            self._handle,
            slots_arr,
            ctypes.c_size_t(num_active),
            tokens_arr,
            ctypes.c_float(temperature),
            ctypes.c_int(top_k),
            ctypes.c_float(top_p),
            output_arr,
        )
        
        return [int(output_arr[i]) for i in range(num_active)]

    def decode_per_request(
        self,
        active_slots: Sequence[int],
        current_tokens: Sequence[int],
        temperatures: Sequence[float],
        top_ks: Sequence[int],
        top_ps: Sequence[float],
    ) -> list[int]:
        """Per-request sampling: each slot uses its own sampling parameters.
        
        Args:
            active_slots: Active slot ID list.
            current_tokens: Current token per slot.
            temperatures: Temperature per slot.
            top_ks: Top-K per slot.
            top_ps: Top-P per slot.
            
        Returns:
            list[int]: Next token per slot.
        """
        num_active = len(active_slots)
        assert len(current_tokens) == num_active
        assert len(temperatures) == num_active
        assert len(top_ks) == num_active
        assert len(top_ps) == num_active

        if self._native is not None:
            return list(self._native.decode_per_request(
                list(active_slots), list(current_tokens), list(temperatures),
                list(top_ks), list(top_ps)
            ))

        slots_arr = (ctypes.c_size_t * num_active)(*active_slots)
        tokens_arr = (ctypes.c_int64 * num_active)(*current_tokens)
        temp_arr = (ctypes.c_float * num_active)(*temperatures)
        topk_arr = (ctypes.c_int * num_active)(*top_ks)
        topp_arr = (ctypes.c_float * num_active)(*top_ps)
        output_arr = (ctypes.c_int64 * num_active)()

        batch_decode_per_request(
            self._handle,
            slots_arr,
            ctypes.c_size_t(num_active),
            tokens_arr,
            temp_arr,
            topk_arr,
            topp_arr,
            output_arr,
        )

        return [int(output_arr[i]) for i in range(num_active)]

    def slot_get_pos(self, slot_id: int) -> int:
        """获取 slot 的当前 KV-Cache 位置."""
        if self._native is not None:
            return int(self._native.slot_position(slot_id))
        return int(batch_slot_get_pos(self._handle, ctypes.c_size_t(slot_id)))

    def slot_save(self, slot_id: int):
        """保存 slot 的 KV-Cache 快照.
        
        Returns:
            snapshot handle (C++ 指针), 如果 slot 为空则返回 None.
        """
        if self._native is not None:
            handle = self._native.slot_save(slot_id)
        else:
            handle = batch_slot_save(self._handle, ctypes.c_size_t(slot_id))
        if not handle:
            return None
        return handle

    def slot_restore(self, slot_id: int, snapshot_handle):
        """从快照恢复 slot 的 KV-Cache."""
        if snapshot_handle:
            if self._native is not None:
                self._native.slot_restore(slot_id, int(snapshot_handle))
                return
            batch_slot_restore(
                self._handle,
                ctypes.c_size_t(slot_id),
                snapshot_handle,
            )

    def get_free_blocks(self) -> int:
        """获取当前可用的 KV-Cache block 数量."""
        if self._native is not None:
            return int(self._native.free_blocks)
        return int(batch_get_free_blocks(self._handle))

    def get_total_blocks(self) -> int:
        """获取 KV-Cache block 总数."""
        if self._native is not None:
            return int(self._native.total_blocks)
        return int(batch_get_total_blocks(self._handle))

    def get_block_size(self) -> int:
        """获取每个 block 包含的 token 数."""
        if self._native is not None:
            return int(self._native.block_size)
        return int(batch_get_block_size(self._handle))

    def get_block_usage(self) -> dict:
        """获取 block 使用统计信息.
        
        Returns:
            dict with keys: total, free, used, utilization (0.0~1.0)
        """
        total = self.get_total_blocks()
        free = self.get_free_blocks()
        used = total - free
        return {
            "total": total,
            "free": free,
            "used": used,
            "utilization": used / total if total > 0 else 0.0,
        }

"""Header-first, strict MP1 loading. No unrecognized key is silently dropped."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import torch
from torch import nn
from safetensors import safe_open

from .layers import Linear, RMSNorm, parameter
from .model import Block


DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "I64": torch.int64,
          "I32": torch.int32, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E8M0": torch.float8_e8m0fnu, "F4": torch.float4_e2m1fn_x2}


def read_header(path):
    path = Path(path)
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError("truncated safetensors header")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > min(128 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError("invalid or oversized safetensors header")
        raw = handle.read(size)
    header = json.loads(raw)
    return {k: v for k, v in header.items() if k != "__metadata__"}, hashlib.sha256(raw).hexdigest()


def auxiliary_parameters(cfg):
    if cfg.mtp_stages == 0:
        return {}
    if not cfg.dspark_target_layer_ids:
        raise ValueError("DSpark checkpoint validation requires target layer IDs")
    auxiliary_cfg = replace(cfg, n_hash_layers=0, compress_ratios=(0,) * cfg.n_layers)
    with torch.device("meta"):
        block = Block(0, auxiliary_cfg, None, None)
        result = {f"mtp.{stage}.{key}": value for stage in range(cfg.mtp_stages)
                  for key, value in block.named_parameters()}
        main_proj = Linear(cfg, None, cfg.dim * len(cfg.dspark_target_layer_ids), cfg.dim)
        result.update({f"mtp.0.main_proj.{key}": value for key, value in main_proj.named_parameters()})
        result["mtp.0.main_norm.weight"] = RMSNorm(cfg.dim, cfg.norm_eps).weight
        prefix = f"mtp.{cfg.mtp_stages-1}."
        result.update({
            prefix + "norm.weight": parameter(cfg.dim, dtype=torch.float32),
            prefix + "markov_head.markov_w1.weight": parameter(cfg.vocab_size, cfg.dspark_markov_rank),
            prefix + "markov_head.markov_w2.weight": parameter(cfg.vocab_size, cfg.dspark_markov_rank, dtype=torch.float32),
            prefix + "confidence_head.proj.weight": parameter(1, cfg.dim + cfg.dspark_markov_rank, dtype=torch.float32),
            prefix + "hc_head_fn": parameter(cfg.hc_mult, cfg.dim * cfg.hc_mult, dtype=torch.float32),
            prefix + "hc_head_base": parameter(cfg.hc_mult, dtype=torch.float32),
            prefix + "hc_head_scale": parameter(1, dtype=torch.float32),
        })
    return result


def validate_header(model, header):
    expected = dict(model.named_parameters())
    auxiliary = auxiliary_parameters(model.config) if any(k.startswith("mtp.") for k in header) else {}
    validated = {**expected, **auxiliary}
    missing, extra = set(validated) - set(header), set(header) - set(validated)
    if extra:
        raise ValueError(f"unexpected checkpoint tensors: {sorted(extra)[:8]}")
    if missing:
        raise ValueError(f"missing checkpoint tensors ({len(missing)}): {sorted(missing)[:8]}")
    promotions = []
    for key, target in validated.items():
        entry = header[key]
        shape = tuple(entry["shape"])
        source_dtype = DTYPES.get(entry["dtype"])
        if source_dtype == torch.float4_e2m1fn_x2:
            if not shape or shape[-1] % 2:
                raise ValueError(f"invalid packed FP4 shape: {key}")
            shape = (*shape[:-1], shape[-1] // 2)
        if shape != tuple(target.shape):
            raise ValueError(f"shape mismatch for {key}: checkpoint {shape}, model {tuple(target.shape)}")
        if source_dtype != target.dtype:
            allowed = ((source_dtype == torch.bfloat16 and target.dtype == torch.float32)
                       or (source_dtype == torch.int64 and target.dtype == torch.int32 and key.endswith(".gate.tid2eid")))
            if not allowed:
                raise ValueError(f"dtype mismatch for {key}: {source_dtype} -> {target.dtype}")
            if key in expected:
                promotions.append({"tensor": key, "source": str(source_dtype), "destination": str(target.dtype)})
    return expected, sorted(auxiliary), promotions


@torch.inference_mode()
def load_converted_weights(model, path, *, device="cuda:0"):
    path = Path(path).resolve()
    header, digest = read_header(path)
    expected, ignored, promotions = validate_header(model, header)
    model._state_owner = object()
    model._load_failed = True
    loaded_bytes = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key, target in expected.items():
            value = handle.get_tensor(key)
            if key.endswith(".gate.tid2eid") and (value.min() < 0 or value.max() >= model.config.n_routed_experts):
                raise ValueError(f"expert ID outside configured range: {key}")
            value = value.to(device=device, dtype=target.dtype)
            owner_path, _, local_name = key.rpartition(".")
            owner = model.get_submodule(owner_path) if owner_path else model
            owner._parameters[local_name] = nn.Parameter(value, requires_grad=False)
            loaded_bytes += value.numel() * value.element_size()
    model._load_failed = False
    model.eval()
    return {
        "format": "converted-mp1", "checkpoint": str(path),
        "checkpoint_size_bytes": path.stat().st_size, "header_sha256": digest,
        "tensor_count": len(expected), "device_bytes": loaded_bytes,
        "dtype_counts": dict(Counter(str(p.dtype) for p in model.parameters())),
        "explicit_dtype_conversions": promotions,
        "ignored_mtp_tensors": len(ignored), "mtp_execution": False,
        "missing": [], "unexpected": [],
    }

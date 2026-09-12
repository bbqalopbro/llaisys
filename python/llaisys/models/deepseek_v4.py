"""Static DeepSeek-V4 checkpoint contracts.

This module intentionally has no torch or native-library dependency.  It is
used before model construction to validate the real checkpoint, describe the
compressed-attention/MoE layout, and map checkpoint names to model-owned
components.  Tensor payloads are never read by this module; safetensors header
inspection stops before the first tensor byte.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping


class DeepSeekV4ConfigError(ValueError):
    """Raised when a checkpoint does not satisfy the DeepSeek-V4 contract."""


@dataclass(frozen=True)
class DeepSeekV4Config:
    model_type: str
    architectures: tuple[str, ...]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    q_lora_rank: int
    qk_rope_head_dim: int
    o_groups: int
    o_lora_rank: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    compress_rope_theta: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    vocab_size: int
    max_position_embeddings: int
    torch_dtype: str
    quant_method: str
    quant_format: str
    quant_scale_format: str
    quant_weight_block_size: tuple[int, int]
    expert_dtype: str
    moe_intermediate_size: int
    num_routed_experts: int
    num_shared_experts: int
    experts_per_token: int
    num_hash_layers: int
    topk_method: str
    scoring_func: str
    routed_scaling_factor: float
    dspark_block_size: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    declared_nextn_predict_layers: int
    inference_mtp_stages: int | None

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "DeepSeekV4Config":
        model_dir = Path(model_dir)
        root = _read_json(model_dir / "config.json")
        inference_path = model_dir / "inference" / "config.json"
        inference = _read_json(inference_path) if inference_path.is_file() else None

        quant = _required_mapping(root, "quantization_config")
        block = tuple(_required_list(quant, "weight_block_size"))
        if len(block) != 2:
            raise DeepSeekV4ConfigError(
                "quantization_config.weight_block_size must have two dimensions"
            )

        config = cls(
            model_type=_required_str(root, "model_type"),
            architectures=tuple(_required_list(root, "architectures")),
            hidden_size=_required_int(root, "hidden_size"),
            num_hidden_layers=_required_int(root, "num_hidden_layers"),
            num_attention_heads=_required_int(root, "num_attention_heads"),
            num_key_value_heads=_required_int(root, "num_key_value_heads"),
            head_dim=_required_int(root, "head_dim"),
            q_lora_rank=_required_int(root, "q_lora_rank"),
            qk_rope_head_dim=_required_int(root, "qk_rope_head_dim"),
            o_groups=_required_int(root, "o_groups"),
            o_lora_rank=_required_int(root, "o_lora_rank"),
            sliding_window=_required_int(root, "sliding_window"),
            compress_ratios=tuple(_required_list(root, "compress_ratios")),
            compress_rope_theta=float(_required_number(root, "compress_rope_theta")),
            index_n_heads=_required_int(root, "index_n_heads"),
            index_head_dim=_required_int(root, "index_head_dim"),
            index_topk=_required_int(root, "index_topk"),
            vocab_size=_required_int(root, "vocab_size"),
            max_position_embeddings=_required_int(root, "max_position_embeddings"),
            torch_dtype=_required_str(root, "torch_dtype"),
            quant_method=_required_str(quant, "quant_method"),
            quant_format=_required_str(quant, "fmt"),
            quant_scale_format=_required_str(quant, "scale_fmt"),
            quant_weight_block_size=(int(block[0]), int(block[1])),
            expert_dtype=_required_str(root, "expert_dtype"),
            moe_intermediate_size=_required_int(root, "moe_intermediate_size"),
            num_routed_experts=_required_int(root, "n_routed_experts"),
            num_shared_experts=_required_int(root, "n_shared_experts"),
            experts_per_token=_required_int(root, "num_experts_per_tok"),
            num_hash_layers=_required_int(root, "num_hash_layers"),
            topk_method=_required_str(root, "topk_method"),
            scoring_func=_required_str(root, "scoring_func"),
            routed_scaling_factor=float(
                _required_number(root, "routed_scaling_factor")
            ),
            dspark_block_size=_required_int(root, "dspark_block_size"),
            dspark_target_layer_ids=tuple(
                _required_list(root, "dspark_target_layer_ids")
            ),
            dspark_markov_rank=_required_int(root, "dspark_markov_rank"),
            declared_nextn_predict_layers=_required_int(
                root, "num_nextn_predict_layers"
            ),
            inference_mtp_stages=(
                _required_int(inference, "n_mtp_layers") if inference else None
            ),
        )
        config._validate(inference)
        return config

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def main_compress_ratios(self) -> tuple[int, ...]:
        return self.compress_ratios[: self.num_hidden_layers]

    def _validate(self, inference: Mapping[str, Any] | None) -> None:
        if self.model_type != "deepseek_v4":
            raise DeepSeekV4ConfigError(
                f"expected model_type=deepseek_v4, got {self.model_type!r}"
            )
        if "DeepseekV4ForCausalLM" not in self.architectures:
            raise DeepSeekV4ConfigError(
                "architectures does not contain DeepseekV4ForCausalLM"
            )
        if self.head_dim <= self.qk_rope_head_dim:
            raise DeepSeekV4ConfigError("head_dim must be larger than rope head dim")
        if self.num_attention_heads % self.o_groups:
            raise DeepSeekV4ConfigError(
                "num_attention_heads must be divisible by o_groups"
            )
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise DeepSeekV4ConfigError(
                "compress_ratios does not cover every main transformer layer"
            )
        if any(r not in (0, 4, 128) for r in self.compress_ratios):
            raise DeepSeekV4ConfigError(
                "checkpoint uses an unknown compressed-attention ratio"
            )
        if self.num_hash_layers > self.num_hidden_layers:
            raise DeepSeekV4ConfigError("num_hash_layers exceeds main layer count")
        if not 0 < self.experts_per_token <= self.num_routed_experts:
            raise DeepSeekV4ConfigError(
                "experts_per_token must be within the routed expert set"
            )
        if inference:
            aliases = {
                "dim": self.hidden_size,
                "n_layers": self.num_hidden_layers,
                "n_heads": self.num_attention_heads,
                "n_routed_experts": self.num_routed_experts,
                "n_shared_experts": self.num_shared_experts,
                "n_activated_experts": self.experts_per_token,
                "q_lora_rank": self.q_lora_rank,
                "head_dim": self.head_dim,
                "rope_head_dim": self.qk_rope_head_dim,
                "window_size": self.sliding_window,
            }
            for name, expected in aliases.items():
                actual = inference.get(name)
                if actual != expected:
                    raise DeepSeekV4ConfigError(
                        f"inference/config.json {name}={actual!r}, expected {expected!r}"
                    )


@dataclass(frozen=True)
class DeepSeekV4WeightKey:
    name: str
    scope: str
    block_index: int | None
    role: str
    expert_index: int | None
    projection: str | None
    value_kind: str


_BLOCK_RE = re.compile(r"^(layers|mtp)\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
_SHARED_EXPERT_RE = re.compile(
    r"^ffn\.shared_experts\.(w[123])\.(weight|scale)$"
)


def parse_weight_key(name: str) -> DeepSeekV4WeightKey:
    """Map one published checkpoint key to a model-owned component."""
    match = _BLOCK_RE.match(name)
    if match:
        scope = "layer" if match.group(1) == "layers" else "mtp"
        block_index = int(match.group(2))
        relative = match.group(3)
    else:
        scope, block_index, relative = "root", None, name

    expert = _EXPERT_RE.match(relative)
    if expert:
        return DeepSeekV4WeightKey(
            name, scope, block_index, "routed_expert", int(expert.group(1)),
            expert.group(2), expert.group(3)
        )
    shared = _SHARED_EXPERT_RE.match(relative)
    if shared:
        return DeepSeekV4WeightKey(
            name, scope, block_index, "shared_expert", None,
            shared.group(1), shared.group(2)
        )

    role = _weight_role(relative)
    value_kind = relative.rsplit(".", 1)[-1]
    projection = relative.rsplit(".", 2)[-2] if "." in relative else None
    return DeepSeekV4WeightKey(
        name, scope, block_index, role, None, projection, value_kind
    )


def _weight_role(relative: str) -> str:
    if relative in {"embed.weight", "head.weight"}:
        return "token_io"
    if relative == "norm.weight" or relative.endswith("_norm.weight"):
        return "normalization"
    if relative.startswith("attn.indexer."):
        return "attention_indexer"
    if relative.startswith("attn.compressor."):
        return "attention_compressor"
    if relative.startswith("attn."):
        return "attention"
    if relative.startswith("ffn.gate."):
        return "moe_router"
    if relative.startswith("hc_"):
        return "hyper_connection"
    if relative.startswith(("main_proj.", "main_norm.")):
        return "dspark_input"
    if relative.startswith(("markov_head.", "confidence_head.")):
        return "dspark_head"
    raise DeepSeekV4ConfigError(f"unrecognized DeepSeek-V4 weight key: {relative}")


@dataclass(frozen=True)
class SafetensorMetadata:
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_offsets: tuple[int, int]


class DeepSeekV4WeightManifest:
    def __init__(
        self,
        model_dir: Path,
        declared_total_size: int,
        weight_map: Mapping[str, str],
    ) -> None:
        self.model_dir = model_dir
        self.declared_total_size = declared_total_size
        self.weight_map = dict(weight_map)

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "DeepSeekV4WeightManifest":
        model_dir = Path(model_dir)
        payload = _read_json(model_dir / "model.safetensors.index.json")
        metadata = _required_mapping(payload, "metadata")
        weight_map = _required_mapping(payload, "weight_map")
        manifest = cls(
            model_dir,
            _required_int(metadata, "total_size"),
            {str(k): str(v) for k, v in weight_map.items()},
        )
        manifest.validate_files()
        return manifest

    @property
    def tensor_count(self) -> int:
        return len(self.weight_map)

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    @property
    def mtp_stages(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    spec.block_index
                    for spec in self.iter_specs()
                    if spec.scope == "mtp" and spec.block_index is not None
                }
            )
        )

    def iter_specs(self) -> Iterable[DeepSeekV4WeightKey]:
        return (parse_weight_key(name) for name in self.weight_map)

    def role_counts(self) -> Counter[str]:
        return Counter(spec.role for spec in self.iter_specs())

    def validate_files(self) -> None:
        if not self.weight_map:
            raise DeepSeekV4ConfigError("safetensors weight_map is empty")
        missing = [name for name in self.shard_names if not (self.model_dir / name).is_file()]
        if missing:
            raise DeepSeekV4ConfigError(f"missing safetensors shards: {missing[:5]}")
        # Parsing every name here makes an unknown checkpoint namespace fail
        # before any tensor allocation starts.
        for name in self.weight_map:
            parse_weight_key(name)

    def read_tensor_metadata(self) -> dict[str, SafetensorMetadata]:
        result: dict[str, SafetensorMetadata] = {}
        for shard_name in self.shard_names:
            shard_path = self.model_dir / shard_name
            with shard_path.open("rb") as handle:
                header_size_raw = handle.read(8)
                if len(header_size_raw) != 8:
                    raise DeepSeekV4ConfigError(f"invalid safetensors file: {shard_path}")
                header_size = struct.unpack("<Q", header_size_raw)[0]
                header = json.loads(handle.read(header_size))
            for name, item in header.items():
                if name == "__metadata__":
                    continue
                offsets = item["data_offsets"]
                result[name] = SafetensorMetadata(
                    dtype=str(item["dtype"]),
                    shape=tuple(int(v) for v in item["shape"]),
                    shard=shard_name,
                    data_offsets=(int(offsets[0]), int(offsets[1])),
                )
        if set(result) != set(self.weight_map):
            missing = set(self.weight_map) - set(result)
            extra = set(result) - set(self.weight_map)
            raise DeepSeekV4ConfigError(
                f"index/header mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        return result

    def load_tensor_slice(
        self,
        name: str,
        selection: Any = ...,
        *,
        device: str = "cpu",
    ) -> Any:
        """Load one tensor or slice without materializing its complete shard.

        ``safetensors`` and torch are imported lazily so configuration and
        manifest inspection remain dependency-free.  Callers should request
        slices while bringing up the 155-GiB checkpoint.
        """
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"tensor is not present in checkpoint: {name}")
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise DeepSeekV4ConfigError(
                "loading tensor payloads requires the safetensors package"
            ) from error
        with safe_open(
            self.model_dir / shard, framework="pt", device=device
        ) as handle:
            tensor = handle.get_slice(name)
            return tensor[:] if selection is Ellipsis else tensor[selection]

    def validate_structure(self, config: DeepSeekV4Config) -> None:
        specs = tuple(self.iter_specs())
        layer_ids = {s.block_index for s in specs if s.scope == "layer"}
        expected_layers = set(range(config.num_hidden_layers))
        if layer_ids != expected_layers:
            raise DeepSeekV4ConfigError("checkpoint main layer ids do not match config")
        if self.mtp_stages != tuple(range(len(self.mtp_stages))):
            raise DeepSeekV4ConfigError("MTP/DSpark stage ids are not contiguous")
        if config.inference_mtp_stages is not None and len(self.mtp_stages) != config.inference_mtp_stages:
            raise DeepSeekV4ConfigError(
                "checkpoint MTP stage count disagrees with inference config"
            )

        routed: dict[tuple[str, int], set[int]] = {}
        for spec in specs:
            if spec.role != "routed_expert" or spec.block_index is None:
                continue
            routed.setdefault((spec.scope, spec.block_index), set()).add(
                int(spec.expert_index)
            )
        expected_experts = set(range(config.num_routed_experts))
        bad = [key for key, ids in routed.items() if ids != expected_experts]
        if bad:
            raise DeepSeekV4ConfigError(
                f"incomplete routed expert sets in blocks: {bad[:5]}"
            )

        hash_layers = {
            spec.block_index
            for spec in specs
            if spec.scope == "layer" and spec.name.endswith("ffn.gate.tid2eid")
        }
        if hash_layers != set(range(config.num_hash_layers)):
            raise DeepSeekV4ConfigError(
                "hash-routed layer ids disagree with num_hash_layers"
            )


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DeepSeekV4ConfigError(f"required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DeepSeekV4ConfigError(f"expected JSON object in {path}")
    return payload


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise DeepSeekV4ConfigError(f"{key} must be an object")
    return value


def _required_list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise DeepSeekV4ConfigError(f"{key} must be an array")
    return value


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise DeepSeekV4ConfigError(f"{key} must be a string")
    return value


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DeepSeekV4ConfigError(f"{key} must be an integer")
    return value


def _required_number(mapping: Mapping[str, Any], key: str) -> int | float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DeepSeekV4ConfigError(f"{key} must be numeric")
    return value

from dataclasses import dataclass
import json
from pathlib import Path

from ..deepseek_v4 import DeepSeekV4Config


@dataclass(frozen=True)
class InferenceConfig:
    dim: int
    vocab_size: int
    n_layers: int
    n_heads: int
    head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    moe_inter_dim: int
    n_routed_experts: int
    n_activated_experts: int
    compress_ratios: tuple[int, ...]
    rope_head_dim: int = 64
    n_hash_layers: int = 0
    n_shared_experts: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    indexer_tie_policy: str = "published"
    attention_metadata_policy: str = "published"
    window_size: int = 128
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    rope_theta: float = 10000
    compress_rope_theta: float = 160000
    original_seq_len: int = 65536
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    weight_dtype: str = "fp8"
    expert_dtype: str = "fp4"
    mtp_stages: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256

    def __post_init__(self):
        for name in ("dim", "vocab_size", "n_layers", "n_heads", "head_dim", "q_lora_rank",
                     "o_groups", "o_lora_rank", "moe_inter_dim", "n_routed_experts",
                     "n_activated_experts", "rope_head_dim", "index_n_heads", "index_head_dim",
                     "index_topk", "window_size", "max_seq_len", "hc_mult", "hc_sinkhorn_iters"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (len(self.compress_ratios) != self.n_layers
                or any(r not in (0, 4, 128) for r in self.compress_ratios)):
            raise ValueError("each main layer requires an explicit 0/4/128 compression ratio")
        if (self.rope_head_dim % 2 or self.rope_head_dim >= self.head_dim
                or self.rope_head_dim > self.index_head_dim or self.n_heads % self.o_groups):
            raise ValueError("invalid attention/RoPE dimensions")
        if not 0 <= self.n_hash_layers <= self.n_layers or self.n_activated_experts > self.n_routed_experts:
            raise ValueError("invalid routed/hash expert configuration")
        if self.n_shared_experts != 1:
            raise ValueError("this model contract requires exactly one shared expert")
        if self.weight_dtype not in ("bf16", "fp8") or self.expert_dtype not in ("bf16", "fp8", "fp4"):
            raise ValueError("unsupported weight dtype")
        if self.score_func not in ("softmax", "sigmoid", "sqrtsoftplus"):
            raise ValueError("unsupported routing score function")
        if self.indexer_tie_policy not in ("published", "index_ascending"):
            raise ValueError("unsupported Indexer tie policy")
        if self.attention_metadata_policy not in ("published", "fixed"):
            raise ValueError("unsupported attention metadata policy")

    @classmethod
    def from_directory(cls, path: str | Path, *, max_seq_len: int = 4096):
        path = Path(path)
        arch = DeepSeekV4Config.from_directory(path)
        raw = json.loads((path / "config.json").read_text())
        rope = raw["rope_scaling"]
        if max_seq_len > arch.max_position_embeddings:
            raise ValueError("requested cache exceeds model max_position_embeddings")
        return cls(
            dim=arch.hidden_size, vocab_size=arch.vocab_size, n_layers=arch.num_hidden_layers,
            n_heads=arch.num_attention_heads, head_dim=arch.head_dim, q_lora_rank=arch.q_lora_rank,
            o_groups=arch.o_groups, o_lora_rank=arch.o_lora_rank, moe_inter_dim=arch.moe_intermediate_size,
            n_routed_experts=arch.num_routed_experts, n_activated_experts=arch.experts_per_token,
            compress_ratios=arch.compress_ratios[:arch.num_hidden_layers],
            rope_head_dim=arch.qk_rope_head_dim, n_hash_layers=arch.num_hash_layers,
            n_shared_experts=arch.num_shared_experts, index_n_heads=arch.index_n_heads,
            index_head_dim=arch.index_head_dim, index_topk=arch.index_topk,
            window_size=arch.sliding_window, max_seq_len=max_seq_len,
            norm_eps=raw["rms_norm_eps"], hc_mult=raw["hc_mult"],
            hc_sinkhorn_iters=raw["hc_sinkhorn_iters"], hc_eps=raw["hc_eps"],
            score_func=arch.scoring_func, route_scale=arch.routed_scaling_factor,
            swiglu_limit=raw["swiglu_limit"], rope_theta=raw["rope_theta"],
            compress_rope_theta=arch.compress_rope_theta,
            original_seq_len=rope["original_max_position_embeddings"], rope_factor=rope["factor"],
            beta_fast=rope["beta_fast"], beta_slow=rope["beta_slow"],
            weight_dtype="fp8", expert_dtype=arch.expert_dtype,
            mtp_stages=arch.inference_mtp_stages or 0,
            dspark_target_layer_ids=arch.dspark_target_layer_ids,
            dspark_markov_rank=arch.dspark_markov_rank,
        )

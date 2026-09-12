"""Explicit, per-operator dispatch for the DeepSeek-V4 reference execution path.

This is an execution adapter, not a scheduler or a cache allocator. Backends
consume the same layouts and quantization contracts; a missing implementation
or a failed launch is an error, never an implicit backend change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
from types import MappingProxyType
from typing import Callable, Mapping


@dataclass(frozen=True)
class OperatorContract:
    revision: int
    inputs: str
    outputs: str
    mutation: str = "none"


CONTRACTS = MappingProxyType({
    "act_quant": OperatorContract(1, "BF16 [...,K]; block=128; FP32/UE8M0 scales",
                                  "FP8 [...,K], scale [...,K/128] or BF16 QDQ",
                                  "x only when inplace=True"),
    "fp4_act_quant": OperatorContract(1, "BF16 [...,K]; block=32; UE8M0 scales",
                                      "packed E2M1 [...,K/2], scale [...,K/32] or BF16 QDQ",
                                      "x only when inplace=True"),
    "fp8_gemm": OperatorContract(1, "FP8 A[...,K], B[N,K]; scales A[...,K/128], B[ceil(N/128),K/128]",
                                 "BF16 [...,N]; FP32 accumulation; no activation requantization"),
    "fp4_gemm": OperatorContract(1, "FP8 A[...,K], packed E2M1 B[N,K/2]; scales A[...,K/128], B[N,K/32]",
                                 "BF16 [...,N]; FP32 accumulation; W4A8, not W4A4"),
    "sparse_attn": OperatorContract(1, "BF16 Q[B,S,H,D], latent[B,L,D]; FP32 sink[H]; int32 indices[B,S,T], -1=invalid",
                                    "BF16 [B,S,H,D]; indexed attention with sink in softmax denominator"),
    "hc_split_sinkhorn": OperatorContract(1, "FP32 mixes[B,S,(2+HC)*HC], scale[3], base[(2+HC)*HC]",
                                          "FP32 pre[B,S,HC], post[B,S,HC], combine[B,S,HC,HC]"),
})


MODEL_CONTRACTS = MappingProxyType({
    **CONTRACTS,
    "dense_linear": OperatorContract(1, "BF16/FP32 X[...,K], matching W[N,K]; same device",
                                     "matching dtype [...,N]; no weight or activation quantization"),
    "grouped_linear": OperatorContract(1, "BF16/FP32 X[B,S,G,K], matching W[G,N,K]; same device",
                                       "matching dtype [B,S,G,N]; separate weight matrix per group"),
    "indexer_topk": OperatorContract(1, "BF16/FP32 scores[B,S,C], k<=C, explicit published/index_ascending tie policy",
                                    "integer indices[B,S,k], descending scores; tie policy must be honored"),
    "hc_sum": OperatorContract(1, "FP32 HC weighted terms, explicit reduction axis",
                               "FP32 sum with that axis removed; no dtype conversion"),
    "row_inv_rms": OperatorContract(1, "BF16/FP32 X[...,D], epsilon; square/mean in the input dtype",
                                    "same dtype [...,1], rsqrt(mean(square(X))+epsilon)"),
    "rms_norm": OperatorContract(1, "BF16/FP32 X[...,D], FP32 weight[D], epsilon",
                                 "X dtype [...,D]; FP32 normalization and weight multiplication"),
    "rotary_inplace": OperatorContract(1, "BF16/FP32 X[B,S,(H),R], complex64 frequencies[S,R/2], inverse flag",
                                       "same X storage; adjacent-pair FP32 complex rotation", "X, including strided RoPE slices"),
    "indexer_scores": OperatorContract(1, "BF16 Q[B,S,H,D], K[B,C,D], weights[B,S,H]",
                                      "BF16 [B,S,C]; BF16 dot/ReLU/product/reduction; no causal mask"),
    "router": OperatorContract(1, "FP32 logits[T,E]; score function, top-k, scale; FP32 bias[E] or hash IDs[T,K]",
                               "FP32 weights[T,K], integer expert IDs[T,K]; bias affects selection only"),
    "compressor_pool": OperatorContract(1, "FP32 projected values/scores[B,(G),R,D], scores may contain -inf",
                                        "FP32 [B,G,D] or decode [B,1,D]; softmax/reduction on R preserving input rank"),
    "expert_activation": OperatorContract(1, "BF16 gate/up[T,I], clamp limit; optional FP32 routing weights[T,1]",
                                          "BF16 [T,I]; FP32 SwiGLU, route scaling BEFORE BF16 cast and down projection"),
    "moe_dispatch": OperatorContract(1, "BF16 X[T,D], FP32 weights[T,K], distinct in-range integer expert IDs[T,K], E",
                                     "ExpertDispatch: owned packed hidden/weights/token IDs and device offsets; expert-ID-major stable order"),
    "moe_combine": OperatorContract(1, "BF16 expert outputs[R,D] in ExpertDispatch order; dispatch metadata",
                                    "FP32 [T,D]; accumulate in ascending expert ID, no route reweighting or shared expert"),
})


class BackendSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class _Implementation:
    backend: str
    version: str
    function: Callable
    contract: OperatorContract


class OperatorRegistry:
    def __init__(self, contracts=CONTRACTS):
        self._contracts = MappingProxyType(dict(contracts))
        if not self._contracts or any(not isinstance(c, OperatorContract) for c in self._contracts.values()):
            raise BackendSelectionError("a nonempty operator contract set is required")
        self._implementations = {}

    def register(self, backend: str, operator: str, function: Callable, *,
                 version: str, contract: OperatorContract):
        if not backend or not version or not callable(function):
            raise BackendSelectionError("backend, version and callable are required")
        if operator not in self._contracts or contract != self._contracts[operator]:
            raise BackendSelectionError(f"incompatible contract for {operator}")
        key = (backend, operator)
        if key in self._implementations:
            raise BackendSelectionError(f"implementation already registered: {key}")
        self._implementations[key] = _Implementation(backend, version, function, contract)

    def bind(self, selection: Mapping[str, str]) -> "BoundOperators":
        if set(selection) != set(self._contracts):
            raise BackendSelectionError(
                f"select every operator exactly once; missing={sorted(set(self._contracts) - set(selection))}, "
                f"unknown={sorted(set(selection) - set(self._contracts))}")
        resolved = {}
        for operator, backend in selection.items():
            key = (backend, operator)
            if key not in self._implementations:
                raise BackendSelectionError(f"backend {backend} does not implement {operator}")
            resolved[operator] = self._implementations[key]
        return BoundOperators(resolved)


class BoundOperators:
    def __init__(self, implementations):
        self._implementations = MappingProxyType(dict(implementations))
        self._counts = {name: {"calls": 0, "failures": 0} for name in implementations}
        self._functions = {name: self._wrap(name, impl.function)
                           for name, impl in implementations.items()}

    def _wrap(self, name, function):
        @wraps(function)
        def invoke(*args, **kwargs):
            self._counts[name]["calls"] += 1
            try:
                return function(*args, **kwargs)
            except Exception:
                self._counts[name]["failures"] += 1
                raise
        return invoke

    def function(self, name: str) -> Callable:
        return self._functions[name]

    def install_on(self, model_module):
        missing = [name for name in self._functions if not hasattr(model_module, name)]
        if missing:
            raise BackendSelectionError(f"model does not expose required operators: {missing}")
        for name, function in self._functions.items():
            setattr(model_module, name, function)

    def report(self):
        return {name: {"backend": impl.backend, "version": impl.version,
                       "contract": asdict(impl.contract), **self._counts[name],
                       "fallback": False}
                for name, impl in self._implementations.items()}


def register_tilelang(registry: OperatorRegistry, kernel_module, version: str):
    import torch

    def make_gemm(mode):
        def gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
            # The published Python wrappers allocate using a process-global
            # default dtype. Keep the identical kernel, but own its BF16 output
            # explicitly so two model/backend instances cannot change each other.
            m, n, k = _validate_gemm(mode, a, a_s, b, b_s, scale_dtype)
            out = torch.empty((*a.shape[:-1], n), dtype=torch.bfloat16, device=a.device)
            if m:
                tl_dtype = kernel_module.FE8M0 if scale_dtype == torch.float8_e8m0fnu else kernel_module.FP32
                with torch.cuda.device(a.device):
                    kernel = getattr(kernel_module, mode + "_gemm_kernel")(n, k, scale_dtype=tl_dtype)
                    kernel(a.reshape(m, k), b, out.reshape(m, n), a_s.reshape(m, k // 128), b_s)
            return out
        return gemm

    for operator, contract in CONTRACTS.items():
        function = make_gemm(operator[:3]) if operator in ("fp8_gemm", "fp4_gemm") else getattr(kernel_module, operator)
        registry.register("tilelang", operator, function,
                          version=version, contract=contract)


def _validate_gemm(mode, a, a_s, b, b_s, scale_dtype):
    import torch
    expected_b = torch.float4_e2m1fn_x2 if mode == "fp4" else torch.float8_e4m3fn
    if a.dtype != torch.float8_e4m3fn or b.dtype != expected_b:
        raise ValueError(f"{mode} GEMM requires FP8 activation and matching weight dtype")
    if a.ndim < 2 or b.ndim != 2 or a.shape[-1] <= 0:
        raise ValueError("invalid GEMM rank or K dimension")
    k, n = a.shape[-1], b.shape[0]
    if k % 128 or n <= 0 or b.shape[1] * (2 if mode == "fp4" else 1) != k:
        raise ValueError("GEMM weight shape or quantization block mismatch")
    if a_s.shape != (*a.shape[:-1], k // 128):
        raise ValueError("activation scale must use 1x128 blocks")
    expected_scale = (n, k // 32) if mode == "fp4" else ((n + 127) // 128, k // 128)
    if b_s.shape != expected_scale:
        raise ValueError(f"weight scale shape must be {expected_scale}")
    if scale_dtype not in (torch.float32, torch.float8_e8m0fnu):
        raise ValueError("only FP32 or UE8M0 scales are supported")
    if a_s.dtype != scale_dtype or b_s.dtype != scale_dtype:
        raise ValueError("scale_dtype does not match scale tensors")
    if a.device.type != "cuda" or any(
        t.device != a.device or not t.is_contiguous() for t in (a, a_s, b, b_s)
    ):
        raise ValueError("GEMM requires contiguous tensors on the same CUDA device")
    return a.numel() // k, n, k


def register_deepgemm(registry: OperatorRegistry, deep_gemm):
    import torch

    def make_gemm(mode):
        def gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
            m, n, k = _validate_gemm(mode, a, a_s, b, b_s, scale_dtype)
            out = torch.empty((*a.shape[:-1], n), dtype=torch.bfloat16, device=a.device)
            if m == 0:
                return out
            with torch.cuda.device(a.device):
                deep_gemm.fp8_fp4_gemm_nt(
                    (a.reshape(m, k), a_s.reshape(m, k // 128).float()),
                    (b.view(torch.int8) if mode == "fp4" else b, b_s.float()),
                    out.reshape(m, n), recipe_a=(1, 128),
                    recipe_b=(1, 32) if mode == "fp4" else (128, 128),
                )
            return out
        return gemm

    for mode in ("fp8", "fp4"):
        operator = mode + "_gemm"
        registry.register("deepgemm", operator, make_gemm(mode),
                          version=deep_gemm.__version__, contract=CONTRACTS[operator])


def register_torch_model(registry: OperatorRegistry, *, backend="torch", fixed_rows=0):
    """Explicit model-operator reference backend.

    Fixed-M tiling is an opt-in reference policy for studying shape-dependent
    rounding, not a fast GEMM claim. It still uses the operator library and does
    not alter input/weight dtypes, quantize twice, or change global math settings.
    """
    import torch
    from torch.nn import functional as F
    if type(fixed_rows) is not int or fixed_rows < 0:
        raise BackendSelectionError("fixed_rows must be a nonnegative integer")

    def validate(x, weight):
        if x.dtype not in (torch.bfloat16, torch.float32) or x.dtype != weight.dtype:
            raise ValueError("nonquantized projections require matching BF16 or FP32 dtype")
        if x.device != weight.device or x.shape[-1] != weight.shape[-1]:
            raise ValueError("nonquantized projection device or K mismatch")

    def linear(x, weight):
        if x.ndim < 2 or weight.ndim != 2:
            raise ValueError("dense linear requires X[...,K] and W[N,K]")
        validate(x, weight)
        if not fixed_rows:
            return F.linear(x, weight)
        rows = x.reshape(-1, x.shape[-1])
        output = x.new_empty(rows.shape[0], weight.shape[0])
        for start in range(0, rows.shape[0], fixed_rows):
            part = rows[start:start+fixed_rows]
            padded = F.pad(part, (0, 0, 0, fixed_rows-part.shape[0]))
            output[start:start+part.shape[0]] = F.linear(padded, weight)[:part.shape[0]]
        return output.reshape(*x.shape[:-1], weight.shape[0])

    def grouped(x, weight):
        if x.ndim != 4 or weight.ndim != 3 or x.shape[2] != weight.shape[0]:
            raise ValueError("grouped linear requires X[B,S,G,K] and W[G,N,K]")
        validate(x, weight)
        if not fixed_rows:
            return torch.einsum("bsgd,grd->bsgr", x, weight)
        batch, sequence, groups, k = x.shape
        rows = x.permute(2, 0, 1, 3).reshape(groups, batch * sequence, k)
        output = x.new_empty(groups, batch * sequence, weight.shape[1])
        for start in range(0, batch * sequence, fixed_rows):
            part = rows[:, start:start+fixed_rows]
            padded = F.pad(part, (0, 0, 0, fixed_rows-part.shape[1]))
            output[:, start:start+part.shape[1]] = torch.bmm(padded, weight.transpose(1, 2))[:, :part.shape[1]]
        return output.reshape(groups, batch, sequence, weight.shape[1]).permute(1, 2, 0, 3).contiguous()

    def topk(scores, k, tie_policy):
        if scores.ndim != 3 or type(k) is not int or not 0 <= k <= scores.shape[-1]:
            raise ValueError("Indexer top-k requires [B,S,C] scores and 0<=k<=C")
        if tie_policy == "published":
            return scores.topk(k, dim=-1)[1]
        if tie_policy == "index_ascending":
            return scores.argsort(dim=-1, descending=True, stable=True)[..., :k]
        raise ValueError(f"unknown Indexer tie policy: {tie_policy}")

    def hc_sum(value, dim):
        if value.dtype != torch.float32 or type(dim) is not int or not -value.ndim <= dim < value.ndim:
            raise ValueError("HC reduction requires FP32 terms and a valid axis")
        if not fixed_rows:
            return value.sum(dim=dim)
        if value.shape[dim] == 0:
            return value.sum(dim=dim)
        output = value.select(dim, 0).clone()
        for index in range(1, value.shape[dim]):
            output = output + value.select(dim, index)
        return output

    def row_inv_rms(value, eps):
        if value.ndim < 2 or value.shape[-1] == 0 or value.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("row inverse RMS requires nonempty BF16/FP32 feature rows")
        if not fixed_rows:
            return torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        rows = value.reshape(-1, value.shape[-1])
        output = value.new_empty(rows.shape[0], 1)
        for start in range(0, rows.shape[0], fixed_rows):
            part = rows[start:start+fixed_rows]
            padded = F.pad(part, (0, 0, 0, fixed_rows-part.shape[0]))
            output[start:start+part.shape[0]] = torch.rsqrt(padded.square().mean(-1, keepdim=True) + eps)[:part.shape[0]]
        return output.reshape(*value.shape[:-1], 1)

    for name, function in (("dense_linear", linear), ("grouped_linear", grouped),
                           ("indexer_topk", topk), ("hc_sum", hc_sum), ("row_inv_rms", row_inv_rms)):
        registry.register(backend, name, function, version=f"{torch.__version__};fixed_rows={fixed_rows}",
                          contract=MODEL_CONTRACTS[name])
    from .deepseek_v4_model_ops import register_torch_structure
    register_torch_structure(registry, backend=backend, contracts=MODEL_CONTRACTS,
                             version=f"{torch.__version__};fixed_rows={fixed_rows}", inv_rms=row_inv_rms)

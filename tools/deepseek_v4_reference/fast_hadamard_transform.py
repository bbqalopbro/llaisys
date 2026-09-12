"""Explicit adapter for the published model's Hadamard dependency."""

from __future__ import annotations

import torch


_backend = "cuda"
_extension = None
_calls = 0


def configure_backend(name: str) -> None:
    global _backend, _extension, _calls
    if name not in ("cuda", "torch-reference"):
        raise ValueError(f"unknown Hadamard backend: {name}")
    if name == "cuda":
        import fast_hadamard_transform_cuda
        _extension = fast_hadamard_transform_cuda
    _backend, _calls = name, 0


def backend_report() -> dict:
    return {"backend": _backend, "calls": _calls, "fallback": _backend == "torch-reference",
            "extension": getattr(_extension, "__file__", None) if _backend == "cuda" else None}


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    global _calls
    if _backend == "cuda":
        if _extension is None:
            configure_backend("cuda")
        result = _extension.fast_hadamard_transform(x, scale)
    else:
        result = torch_hadamard_transform(x, scale)
    _calls += 1
    return result


def torch_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    size = x.shape[-1]
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    dtype = x.dtype
    value = x.float()
    width = 1
    while width < size:
        grouped = value.reshape(*value.shape[:-1], -1, width * 2)
        left = grouped[..., :width]
        right = grouped[..., width:]
        value = torch.cat((left + right, left - right), dim=-1).flatten(-2)
        width *= 2
    return (value * scale).to(dtype)

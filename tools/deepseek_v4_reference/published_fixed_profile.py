"""Explicit numerical-policy oracle over the published V4 model control flow.

Only audited arithmetic call sites are transformed in memory. The shared source
file is never edited. This is NOT the unmodified published numerical baseline.
No llaisys model, cache, scheduler or model-operator implementation is imported.
"""

import ast
from collections import Counter
import hashlib
from pathlib import Path
import sys
import types

import torch
from torch.nn import functional as F


PROFILE = "fixed32-reference-v1"
EXPECTED = {"linear": 4, "inverse_rms": 4, "hc_sum": 3, "grouped": 1,
            "topk": 1, "window": 1, "attention": 2}


class PolicyTransform(ast.NodeTransformer):
    def __init__(self):
        self.scope = []
        self.counts = Counter()

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        return node

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        scope, call = ".".join(self.scope), ast.unparse(node.func)
        kind = None
        if call == "F.linear" and scope in ("linear", "Block.hc_pre", "Block.hc_head", "ParallelHead.forward"):
            if len(node.args) != 2 or node.keywords:
                raise ValueError("published F.linear contract changed")
            kind = "linear"
        elif call == "torch.rsqrt" and scope in ("Attention.forward", "Block.hc_pre", "Block.hc_head", "RMSNorm.forward"):
            variable, eps = ("q", "eps") if scope == "Attention.forward" else ("x", "norm_eps")
            if scope == "RMSNorm.forward":
                variable, eps = "x", "eps"
                expression = "var + self.eps"
            else:
                expression = f"{variable}.square().mean(-1, keepdim=True) + self.{eps}"
            expected = ast.parse(expression, mode="eval").body
            if len(node.args) != 1 or node.keywords or ast.dump(node.args[0]) != ast.dump(expected):
                raise ValueError("published inverse RMS expression changed")
            node.args = [ast.Name(variable, ast.Load()), ast.Attribute(ast.Name("self", ast.Load()), eps, ast.Load())]
            kind = "inverse_rms"
        elif call == "torch.sum" and scope in ("Block.hc_pre", "Block.hc_post", "Block.hc_head"):
            if len(node.args) != 1 or len(node.keywords) != 1 or ast.unparse(node.keywords[0].value) != "2":
                raise ValueError("published HC sum contract changed")
            kind = "hc_sum"
        elif scope == "Attention.forward" and call == "torch.einsum":
            if len(node.args) != 3 or ast.literal_eval(node.args[0]) != "bsgd,grd->bsgr":
                raise ValueError("published grouped projection changed")
            kind = "grouped"
        elif scope == "Indexer.forward" and call == "index_score.topk":
            node.args.insert(0, ast.Name("index_score", ast.Load()))
            kind = "topk"
        elif scope == "Attention.forward" and call == "get_window_topk_idxs":
            kind = "window"
        elif scope == "Attention.forward" and call == "sparse_attn":
            node.args.insert(0, ast.Name("self", ast.Load()))
            kind = "attention"
        if kind is not None:
            self.counts[kind] += 1
            node.func = ast.Name(f"_profile_{kind}", ast.Load())
        return node


def prepare_source(path):
    source = Path(path).read_text()
    transform = PolicyTransform()
    tree = ast.fix_missing_locations(transform.visit(ast.parse(source, filename=str(path))))
    if dict(transform.counts) != EXPECTED:
        raise ValueError(f"published profile call-site audit failed: {dict(transform.counts)} != {EXPECTED}")
    manifest = {"name": PROFILE, "fixed_rows": 32, "indexer_tie_policy": "index_ascending",
                "attention_metadata_policy": "fixed", "unmodified_published_baseline": False,
                "mtp_execution_supported": False, "transforms": dict(transform.counts),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "transformed_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return tree, manifest


def load_profiled_model(name, path):
    tree, manifest = prepare_source(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    counts = Counter()

    def fixed_linear(value, weight):
        flattened = value.flatten(0, -2)
        pieces = []
        for part in flattened.split(32, dim=0):
            workspace = value.new_zeros((32, value.shape[-1]))
            workspace[:part.shape[0]].copy_(part)
            pieces.append(F.linear(workspace, weight)[:part.shape[0]])
        if not pieces:
            return value.new_empty((*value.shape[:-1], weight.shape[0]))
        return torch.cat(pieces).reshape(*value.shape[:-1], weight.shape[0])

    def inverse_rms(value, eps):
        flattened = value.flatten(0, -2)
        pieces = []
        for part in flattened.split(32, dim=0):
            workspace = value.new_zeros((32, value.shape[-1]))
            workspace[:part.shape[0]].copy_(part)
            pieces.append(torch.rsqrt(workspace.square().mean(-1, keepdim=True) + eps)[:part.shape[0]])
        return torch.cat(pieces).reshape(*value.shape[:-1], 1)

    def hc_sum(value, dim):
        # Deliberately specified FP32 ascending-HC reduction, not FP64 or a
        # change in quantization boundary. Tensor shape cannot select a new tree.
        terms = value.unbind(dim=dim)
        result = terms[0]
        for term in terms[1:]:
            result = torch.add(result, term)
        return result

    def grouped(equation, value, weight):
        if equation != "bsgd,grd->bsgr":
            raise ValueError("unsupported reference grouped projection")
        batch, sequence, groups, width = value.shape
        rows = value.permute(2, 0, 1, 3).reshape(groups, -1, width)
        pieces = []
        for part in rows.split(32, dim=1):
            workspace = value.new_zeros((groups, 32, width))
            workspace[:, :part.shape[1]].copy_(part)
            pieces.append(torch.bmm(workspace, weight.transpose(1, 2))[:, :part.shape[1]])
        return torch.cat(pieces, dim=1).reshape(groups, batch, sequence, -1).permute(1, 2, 0, 3).contiguous()

    def topk(scores, k, dim=-1):
        indices = torch.argsort(scores, dim=dim, descending=True, stable=True)[..., :k]
        return scores.gather(dim, indices), indices

    def window(size, batch, sequence, start):
        indices = module.get_window_topk_idxs(size, batch, sequence, start)
        return F.pad(indices, (0, size-indices.shape[-1]), value=-1)

    def attention(layer, query, latent, sink, indices, scale):
        size = layer.window_size
        if layer.compress_ratio:
            capacity = layer.kv_cache.shape[1] - size
            size += min(capacity, layer.indexer.index_topk) if layer.compress_ratio == 4 else capacity
        indices = F.pad(indices, (0, size-indices.shape[-1]), value=-1)
        return module.sparse_attn(query, latent, sink, indices, scale)

    def counted(kind, function):
        def invoke(*args, **kwargs):
            counts[kind] += 1
            return function(*args, **kwargs)
        return invoke

    for kind, function in (("linear", fixed_linear), ("inverse_rms", inverse_rms), ("hc_sum", hc_sum),
                           ("grouped", grouped), ("topk", topk), ("window", window), ("attention", attention)):
        module.__dict__[f"_profile_{kind}"] = counted(kind, function)
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
        def unsupported_mtp(*args, **kwargs):
            raise RuntimeError("MTP/speculative execution is outside the fixed-profile oracle")
        module.Transformer.forward_spec = unsupported_mtp
    except Exception:
        sys.modules.pop(name, None)
        raise
    module._reference_profile = manifest
    module._reference_profile_counts = counts
    return module

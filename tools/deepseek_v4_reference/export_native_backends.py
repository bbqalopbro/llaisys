"""Export versioned native V4 kernel bundles without weights or golden files.

Run once inside the target Slurm GPU environment. The resulting local shared
libraries are executable code: only load bundles from trusted sources.
"""
import argparse
from dataclasses import asdict
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_kernels(cfg, kernel, export):
    """Shared shape-specialization list; export(name, operation, JITKernel)."""
    for name, width in (("quant_hidden", cfg.dim), ("quant_query_rank", cfg.q_lora_rank),
                        ("quant_output", cfg.o_groups * cfg.o_lora_rank), ("quant_intermediate", cfg.moe_inter_dim)):
        export(name, "act_quant", kernel.act_quant_kernel(width, 128, scale_dtype=kernel.FE8M0, round_scale=True))
    for name, n, k in (("gemm_query_a", cfg.q_lora_rank, cfg.dim), ("gemm_query_b", cfg.n_heads * cfg.head_dim, cfg.q_lora_rank),
                      ("gemm_latent", cfg.head_dim, cfg.dim), ("gemm_output", cfg.dim, cfg.o_groups * cfg.o_lora_rank),
                      ("indexer_gemm", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)):
        export(name, "fp8_gemm", kernel.fp8_gemm_kernel(n, k, scale_dtype=kernel.FE8M0))
    export("latent_qdq", "act_quant", kernel.act_quant_kernel(cfg.head_dim - cfg.rope_head_dim, 64,
        scale_dtype=kernel.FE8M0, round_scale=True, inplace=True))
    export("indexer_qdq", "fp4_act_quant", kernel.fp4_quant_kernel(cfg.index_head_dim, 32, inplace=True))
    export("sparse_attn", "sparse_attn", kernel.sparse_attn_kernel(cfg.n_heads, cfg.head_dim, cfg.head_dim ** -0.5))
    for mode in ("fp4", "fp8"):
        for projection, n, k in (("gate", cfg.moe_inter_dim, cfg.dim), ("down", cfg.dim, cfg.moe_inter_dim)):
            export(f"{mode}_{projection}", f"{mode}_gemm", getattr(kernel, f"{mode}_gemm_kernel")(n, k, scale_dtype=kernel.FE8M0))
    export("hc_split_sinkhorn", "hc_split_sinkhorn", kernel.hc_split_sinkhorn_kernel(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps))


def main(args):
    import torch
    import tilelang
    import tvm_ffi
    from llaisys.models.deepseek_v4_model.config import InferenceConfig
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(): raise RuntimeError("export requires Slurm GPU allocation")
    if (tilelang.__version__, tvm_ffi.__version__, torch.__version__) != ("0.1.8", "0.1.8.post2", "2.13.0+cu130"):
        raise RuntimeError("unsupported native reference compiler/ABI versions")
    source = args.source_model.resolve(); cfg = InferenceConfig.from_directory(source, max_seq_len=args.capacity)
    paths = [source / name for name in ("config.json", "inference/config.json", "inference/kernel.py")]
    hashes = {str(path): digest(path) for path in paths}
    spec = importlib.util.spec_from_file_location("llaisys_v4_export_kernel", source / "inference/kernel.py")
    kernel = importlib.util.module_from_spec(spec); sys.modules[spec.name] = kernel; spec.loader.exec_module(kernel)
    output = args.output_directory.resolve(); output.mkdir(parents=True, exist_ok=False)
    entries = {}
    def export(name, operation, compiled):
        path = output / (name + ".so"); cached = getattr(compiled.adapter, "libpath", None)
        if cached:
            shutil.copyfile(cached, path)
            if digest(cached) != digest(path): raise RuntimeError("cached kernel copy mismatch")
        else: compiled.adapter.executable.export_library(str(path))
        entries[name] = {"operation": operation, "library": path.name, "sha256": digest(path)}
    compile_kernels(cfg, kernel, export)
    if hashes != {str(path): digest(path) for path in paths}: raise RuntimeError("source changed during kernel export")
    data = {"format": "llaisys-v4-native-bundle-v1", "source_model": str(source), "configuration": asdict(cfg),
        "source_sha256": hashes, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "compute_capability": list(torch.cuda.get_device_capability(0)), "gpu": torch.cuda.get_device_name(0),
        "hadamard_version": digest(args.hadamard_source / "csrc/fast_hadamard_transform_cuda.cu"), "kernels": entries,
        "exporter_sha256": digest(__file__), "slurm_job_id": os.environ["SLURM_JOB_ID"], "weights_or_goldens_required": False}
    manifest = output / "bundle.json"; manifest.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "kernels": len(entries), "weights_loaded": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True); parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hadamard-source", type=Path, required=True); parser.add_argument("--capacity", type=int, default=4096)
    main(parser.parse_args())

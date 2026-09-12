"""Actual-weight native complete Transformer blocks against published Block.forward.

Covers ratio 0/4/128, hash/learned MoE and both HC branches with ALL block
weights. Python prepares goldens and exported kernels; native child has no
Python. Same-chunk comparison is separate from original full-prefill accuracy.
This is not 43-layer model, paged integration, serving or performance acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from run_independent import load_kernel, sha256
from llaisys.models.deepseek_v4_model.weights import read_header
from llaisys.models.deepseek_v4_model.config import InferenceConfig
from llaisys.models.deepseek_v4_model.model import Block as IncrementalBlock
from llaisys.models.deepseek_v4_model.state import CompressorState, LayerState
from llaisys.models.deepseek_v4_backends import CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model


def raw(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native Block verification requires Slurm GPU allocation")
    import tilelang
    import tvm_ffi
    import fast_hadamard_transform as hadamard
    from torch.utils.cpp_extension import CUDA_HOME
    if (tilelang.__version__, tvm_ffi.__version__, torch.__version__) != ("0.1.8", "0.1.8.post2", "2.13.0+cu130"):
        raise RuntimeError("native ABI/version mismatch")
    if not CUDA_HOME or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("compatible CUDA toolkit/CXX11 ABI required")
    hadamard.configure_backend("cuda")
    root = Path(__file__).resolve().parents[2]
    inference = args.source_model.resolve() / "inference"
    torch_root, ffi_root, tl_root = Path(torch.__file__).parent, Path(tvm_ffi.__file__).parent, Path(tilelang.__file__).parent
    cfg_json = json.loads((inference / "config.json").read_text())
    config = InferenceConfig.from_directory(args.source_model, max_seq_len=2112)
    if (config.dim, config.n_heads, config.head_dim, config.q_lora_rank, config.o_groups, config.o_lora_rank,
        config.rope_head_dim, config.window_size, config.compress_ratios[:4]) != (4096, 64, 512, 1024, 8, 1024, 64, 128, (0, 0, 4, 128)):
        raise RuntimeError("actual Flash-0731 Attention configuration required")
    archives = [args.build_dir.resolve() / f"libllaisys-{name}.a" for name in
                ("deepseek-v4-native", "tilelang-native", "aten-native", "native-tensor", "core", "device", "device-cpu", "utils")]
    sources = [Path(__file__), root / "test/block_native_standalone.cpp", root / "test/native_fixture_utils.hpp",
        inference / "model.py", inference / "kernel.py", inference / "config.json", args.source_model / "config.json",
        root / "xmake.lua", *archives, root / "python/llaisys/libllaisys/libllaisys.so",
        args.hadamard_source / "csrc/fast_hadamard_transform_cuda.cu", args.hadamard_object]
    for directory in ("src/backends/native", "src/backends/tilelang", "src/backends/aten", "src/backends/hadamard", "src/models/deepseek_v4"):
        sources.extend(sorted((root / directory).glob("*.[ch]pp")))
    sources.extend(root / "xmake" / name for name in ("native_model.lua", "native_tensor.lua", "tilelang.lua", "aten.lua"))
    sources.extend(root / "python/llaisys/models" / name for name in
        ("deepseek_v4_backends.py", "deepseek_v4_model_ops.py", "deepseek_v4_reference.py", "deepseek_v4.py"))
    sources.extend(root / "python/llaisys/models/deepseek_v4_model" / name for name in ("config.py", "layers.py", "model.py", "state.py", "weights.py"))
    sources.extend((Path(load_kernel.__code__.co_filename), ffi_root / "lib/libtvm_ffi.so", tl_root / "lib/libtilelang.so", tl_root / "lib/libtvm.so"))
    sources.extend(torch_root / "lib" / name for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so", "libc10_cuda.so"))
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    kernel = load_kernel(inference / "kernel.py"); sys.modules["kernel"] = kernel
    spec = importlib.util.spec_from_file_location("published_block_oracle", inference / "model.py")
    published = importlib.util.module_from_spec(spec); sys.modules[spec.name] = published; spec.loader.exec_module(published)
    cfg_json.update(max_batch_size=1, max_seq_len=2112)
    cfg = published.ModelArgs(**cfg_json)
    published.world_size, published.rank = 1, 0
    published.default_dtype = torch.float8_e4m3fn
    published.scale_fmt, published.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda"); torch.manual_seed(731)
    registry = OperatorRegistry(MODEL_CONTRACTS)
    register_tilelang(registry, kernel, tilelang.__version__); register_torch_model(registry)
    ops = registry.bind({name: "tilelang" if name in CONTRACTS else "torch" for name in MODEL_CONTRACTS})
    checkpoint = args.checkpoint.resolve(strict=True)
    stamp = checkpoint.stat(); checkpoint_stamp = (stamp.st_ino, stamp.st_size, stamp.st_mtime_ns)
    header, header_sha = read_header(checkpoint)
    with checkpoint.open("rb") as handle: data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native_block_", dir=args.output.parent)).resolve()
    codes = {torch.bfloat16: (4, 16, 1), torch.float32: (2, 32, 1), torch.complex64: (5, 64, 1),
             torch.int32: (0, 32, 1), torch.int64: (0, 64, 1), torch.float4_e2m1fn_x2: (17, 4, 1), torch.float8_e4m3fn: (10, 8, 1), torch.float8_e8m0fnu: (14, 8, 1)}
    def descriptor(name, value, path, *, offset=0, expected="-"):
        shape = list(value.shape)
        if value.dtype == torch.float4_e2m1fn_x2: shape[-1] *= 2
        return {"name": name, "dtype": codes[value.dtype], "shape": shape, "bytes": value.numel() * value.element_size(),
                "path": str(path), "offset": offset, "expected": str(expected)}
    def line(value):
        fields = [value["name"], *value["dtype"], len(value["shape"]), *value["shape"], value["bytes"], value["offset"]]
        return " ".join(map(str, fields)) + " " + json.dumps(value["path"]) + " " + json.dumps(value["expected"])
    def fixture(name, value, prefix):
        path = artifacts / f"{prefix}_{name}.bin"; content = raw(value); path.write_bytes(content)
        item = descriptor(name, value, path, expected=path); item["sha256"] = hashlib.sha256(content).hexdigest(); return item
    exports, origins, export_ops = {}, {}, {}
    def export(name, operation, compiled):
        path = artifacts / f"{name}.so"; cached = getattr(compiled.adapter, "libpath", None)
        if cached:
            shutil.copyfile(cached, path)
            if sha256(cached) != sha256(path): raise RuntimeError("cached TileLang artifact mismatch")
            origins[str(path)] = {"mode": "cached_library_copy", "source": str(cached)}
        else:
            compiled.adapter.executable.export_library(str(path)); origins[str(path)] = {"mode": "fresh_executable_export"}
        exports[name] = path; export_ops[name] = operation
    for name, width in (("quant_hidden", cfg.dim), ("quant_query_rank", cfg.q_lora_rank), ("quant_output", cfg.o_groups * cfg.o_lora_rank)):
        export(name, "act_quant", kernel.act_quant_kernel(width, 128, scale_dtype=kernel.FE8M0, round_scale=True))
    for name, n, k in (("gemm_query_a", cfg.q_lora_rank, cfg.dim),
                       ("gemm_query_b", cfg.n_heads * cfg.head_dim, cfg.q_lora_rank),
                       ("gemm_latent", cfg.head_dim, cfg.dim),
                       ("gemm_output", cfg.dim, cfg.o_groups * cfg.o_lora_rank),
                       ("indexer_gemm", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)):
        export(name, "fp8_gemm", kernel.fp8_gemm_kernel(n, k, scale_dtype=kernel.FE8M0))
    export("latent_qdq", "act_quant", kernel.act_quant_kernel(cfg.head_dim - cfg.rope_head_dim, 64,
           scale_dtype=kernel.FE8M0, round_scale=True, inplace=True))
    export("indexer_qdq", "fp4_act_quant", kernel.fp4_quant_kernel(cfg.index_head_dim, 32, inplace=True))
    export("sparse_attn", "sparse_attn", kernel.sparse_attn_kernel(cfg.n_heads, cfg.head_dim, cfg.head_dim ** -0.5))
    export("quant_intermediate", "act_quant", kernel.act_quant_kernel(cfg.moe_inter_dim, 128, scale_dtype=kernel.FE8M0, round_scale=True))
    for mode in ("fp4", "fp8"):
        for projection, n, k in (("gate", cfg.moe_inter_dim, cfg.dim), ("down", cfg.dim, cfg.moe_inter_dim)):
            export(f"{mode}_{projection}", f"{mode}_gemm", getattr(kernel, f"{mode}_gemm_kernel")(n, k, scale_dtype=kernel.FE8M0))
    export("hc_split_sinkhorn", "hc_split_sinkhorn", kernel.hc_split_sinkhorn_kernel(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps))
    components = []
    hadamard_version = sha256(args.hadamard_source / "csrc/fast_hadamard_transform_cuda.cu")
    lines = [f"LLAISYS_NATIVE_BLOCK_V1 {tilelang.__version__} {torch.__version__.split('+')[0]} {hadamard_version} 3"]
    with torch.inference_mode(), safe_open(checkpoint, framework="pt", device="cpu") as reader:
        for layer in (0, 2, 3):
            prefix = f"layers.{layer}"; ratio = config.compress_ratios[layer]
            official = published.Block(layer, cfg)
            incremental = IncrementalBlock(layer, config, ops, hadamard.hadamard_transform)
            names = set(dict(official.named_parameters()))
            actual = {key.removeprefix(prefix + ".") for key in reader.keys() if key.startswith(prefix + ".")}
            if names != actual or names != set(dict(incremental.named_parameters())) or len(names) != 8 + (6 * (cfg.n_routed_experts + cfg.n_shared_experts) + 2) + (23 if ratio == 4 else 16 if ratio else 12):
                raise RuntimeError(f"strict Block checkpoint mapping failed: {sorted(actual ^ names)}")
            records = []
            for name, param in official.named_parameters():
                value = reader.get_tensor(prefix + "." + name)
                converted = param.dtype == torch.float32 and value.dtype == torch.bfloat16
                hash_cast = name == "ffn.gate.tid2eid" and value.dtype == torch.int64 and param.dtype == torch.int32
                if hash_cast and (value.min().item() < 0 or value.max().item() >= cfg.n_routed_experts):
                    raise RuntimeError("hash expert ID out of range")
                if value.shape != param.shape or (value.dtype != param.dtype and not converted and not hash_cast):
                    raise RuntimeError(f"actual Block weight shape/dtype mismatch: {name}, {value.dtype}, {param.dtype}")
                record = descriptor(name, value, checkpoint, offset=data_start + header[prefix + "." + name]["data_offsets"][0])
                record.update(payload_sha256=hashlib.sha256(raw(value)).hexdigest(), checkpoint_key=prefix + "." + name,
                              reference_parameter_dtype=str(param.dtype)); records.append(record)
                param.copy_(value); dict(incremental.named_parameters())[name].copy_(value)
            for module in official.modules():
                if isinstance(module, published.Linear) and module.scale is not None and module.weight.scale is not module.scale:
                    raise RuntimeError("published weight.scale alias broken")
            frequencies = official.attn.freqs_cis
            freq_record = fixture("frequencies", frequencies, prefix)
            lines.append(" ".join(map(str, [prefix, cfg.dim, cfg.n_heads, cfg.head_dim, cfg.q_lora_rank, cfg.o_groups,
                         cfg.o_lora_rank, cfg.rope_head_dim, cfg.window_size, ratio, cfg.max_seq_len,
                         cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk, cfg.norm_eps, cfg.hc_mult, cfg.moe_inter_dim,
                         cfg.n_routed_experts, cfg.n_activated_experts, cfg.vocab_size, int(layer < cfg.n_hash_layers), cfg.route_scale,
                         cfg.swiglu_limit, cfg.score_func, cfg.hc_eps, cfg.hc_sinkhorn_iters, len(records) + 1])))
            lines.extend(line(item) for item in [*records, freq_record]); lines.append(str(len(exports)))
            lines.extend(f"{name} {export_ops[name]} {json.dumps(str(path))}" for name, path in exports.items())
            plans = [("published_short", [3, 1, 1], "published"), ("published_window", [127, 1, 1], "published"),
                     ("published_long", [2105 if ratio == 4 else 257, 1, 1], "published"),
                     ("chunk_non_aligned", [3, 2, 5, 127, 1], "incremental_python")]
            lines.append(str(len(plans))); results = []
            for name, sizes, oracle in plans:
                cache = torch.zeros(1, cfg.window_size + (cfg.max_seq_len // ratio if ratio else 0),
                                    cfg.head_dim, dtype=torch.bfloat16, device="cuda")
                index_cache = torch.zeros(1, cfg.max_seq_len // 4, cfg.index_head_dim, dtype=torch.bfloat16, device="cuda") if ratio == 4 else None
                state = LayerState(cache, frequencies, CompressorState.allocate(1, ratio, cfg.head_dim, "cuda") if ratio else None,
                                   index_cache, CompressorState.allocate(1, 4, cfg.index_head_dim, "cuda") if ratio == 4 else None)
                official.attn.kv_cache = cache
                if ratio:
                    official.attn.compressor.kv_cache = cache[:, cfg.window_size:]; official.attn.compressor.freqs_cis = frequencies
                    official.attn.compressor.kv_state.zero_(); official.attn.compressor.score_state.fill_(-torch.inf)
                if ratio == 4:
                    official.attn.indexer.kv_cache = index_cache; official.attn.indexer.freqs_cis = frequencies
                    official.attn.indexer.compressor.kv_cache = index_cache; official.attn.indexer.compressor.freqs_cis = frequencies
                    official.attn.indexer.compressor.kv_state.zero_(); official.attn.indexer.compressor.score_state.fill_(-torch.inf)
                model = official if oracle == "published" else incremental
                traces = {}; hooks = []
                for tag, module in (("attn_input", model.attn_norm), ("attn_output", model.attn), ("moe_input", model.ffn_norm), ("moe_output", model.ffn)):
                    hooks.append(module.register_forward_hook(lambda _m, _i, out, tag=tag: traces.__setitem__(tag, out.detach().clone())))
                lines.append(f"{name} {oracle} {len(sizes)}"); position = 0; steps = []
                try:
                    for index, tokens in enumerate(sizes):
                        traces.clear()
                        x = torch.randn(1, tokens, cfg.hc_mult, cfg.dim, dtype=torch.bfloat16, device="cuda")
                        ids = torch.randint(0, cfg.vocab_size, (1, tokens), device="cuda", dtype=torch.int64)
                        output = official(x, position, ids) if oracle == "published" else incremental(x, position, state, ids)
                        expected = {"input": x, "input_ids": ids, "output": output, "cache": cache, **traces}
                        if ratio:
                            expected.update(kv_state=official.attn.compressor.kv_state if oracle == "published" else state.compressor.kv,
                                            score_state=official.attn.compressor.score_state if oracle == "published" else state.compressor.scores)
                        if ratio == 4:
                            expected.update(index_cache=index_cache,
                                index_kv_state=official.attn.indexer.compressor.kv_state if oracle == "published" else state.index_compressor.kv,
                                index_score_state=official.attn.indexer.compressor.score_state if oracle == "published" else state.index_compressor.scores)
                        specs = [fixture(tag, value, f"{prefix}_{name}_{index}") for tag, value in expected.items()]
                        lines.append(f"{position} {tokens} {len(specs)}"); lines.extend(line(item) for item in specs)
                        steps.append({"position": position, "tokens": tokens, "tensors": specs}); position += tokens
                finally:
                    for hook in hooks: hook.remove()
                results.append({"name": name, "oracle": oracle, "tokens": sizes, "steps": steps})
                print(f"reference {prefix}/{name}: {sizes}", flush=True)
            components.append({"name": prefix, "ratio": ratio, "weights": records, "frequencies": freq_record, "plans": results})
    manifest = artifacts / "manifest.txt"; manifest.write_text("\n".join(lines) + "\n")
    executable = artifacts / "block-native-standalone"
    libdirs = [torch_root / "lib", ffi_root / "lib", tl_root / "lib", root / "python/llaisys/libllaisys", Path(CUDA_HOME) / "lib64"]
    command = [shutil.which("g++"), "-std=c++20", "-O2", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *["-I" + str(path) for path in (root, root / "include", ffi_root / "include", torch_root / "include",
            torch_root / "include/torch/csrc/api/include", Path(CUDA_HOME) / "include")], str(root / "test/block_native_standalone.cpp"),
        "-Wl,--start-group", *map(str, archives), "-Wl,--end-group", *["-L" + str(path) for path in libdirs], "-Wl,--no-as-needed",
        "-lllaisys", "-ltilelang", "-ltvm", "-ltvm_ffi", "-ltorch_cuda", "-ltorch_cpu", "-lc10_cuda", "-lc10", "-lcudart", "-ldl",
        *["-Wl,-rpath," + str(path) for path in libdirs], "-o", str(executable)]
    print("compile:", json.dumps(command), flush=True); subprocess.run(command, check=True)
    nvidia = torch_root.parent / "nvidia"
    child_libs = [torch_root / "lib", nvidia / "cu13/lib", *sorted(nvidia.glob("*/lib")), *libdirs[1:]]
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(map(str, child_libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    run = subprocess.run([str(executable), str(manifest)], env=env, capture_output=True, text=True)
    print(run.stdout, end="", flush=True); print(run.stderr, end="", file=sys.stderr, flush=True)
    result = json.loads(run.stdout.splitlines()[-1]) if run.returncode == 0 else {"all_passed": False}
    unchanged = hashes == {str(path.resolve()): sha256(path) for path in sources}
    after = checkpoint.stat(); payload_unchanged = checkpoint_stamp == (after.st_ino, after.st_size, after.st_mtime_ns)
    with checkpoint.open("rb") as handle:
        for component in components:
            for item in component["weights"]:
                handle.seek(item["offset"])
                if hashlib.sha256(handle.read(item["bytes"])).hexdigest() != item["payload_sha256"]: payload_unchanged = False
    policy_matches = result.get("allow_tf32_cublas") == torch.backends.cuda.matmul.allow_tf32
    report = {"scope": "actual-weight complete native Transformer Block; NOT 43-layer model, paged cache or performance", "result": result,
        "all_passed": run.returncode == 0 and result["all_passed"] and unchanged and payload_unchanged and policy_matches,
        "source_unchanged": unchanged, "selected_checkpoint_payload_unchanged": payload_unchanged,
        "reference_precision_policy_matches": policy_matches, "components": components,
        "configuration": {"hidden": cfg.dim, "heads": cfg.n_heads, "head_dim": cfg.head_dim, "query_rank": cfg.q_lora_rank,
                          "output_groups": cfg.o_groups, "output_rank": cfg.o_lora_rank, "rope_dimension": cfg.rope_head_dim,
                          "window": cfg.window_size, "capacity": cfg.max_seq_len, "index_topk": cfg.index_topk, "batch": 1,
                          "norm_epsilon": cfg.norm_eps, "hc_copies": cfg.hc_mult, "hc_epsilon": cfg.hc_eps, "hc_iterations": cfg.hc_sinkhorn_iters,
                          "moe_experts": cfg.n_routed_experts, "moe_topk": cfg.n_activated_experts, "shared_experts": cfg.n_shared_experts, "frequency_preparation": "published per-layer frequencies outside native execution",
                          "precision": "W8A8 dense/shared, W4A8 ALL routed experts, BF16 hidden/cache, FP32 norms/HC/compressor/router",
                          "indexer": "Hadamard/FP4 QDQ, published topk", "inputs": "deterministic random BF16 hidden, not text"},
        "reference_scope": {"published": "unchanged upstream Block.forward: full prefill and single-token decode",
                            "incremental_python": "existing project Block: SAME multi-token chunk schedule, not original full-prefill oracle"},
        "full_vs_chunk_numerical_gate": "not claimed by this same-schedule native migration test",
        "checkpoint": {"path": str(checkpoint), "size_bytes": after.st_size, "header_sha256": header_sha,
                       "whole_model_loaded": False, "whole_payload_hashed": False},
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "cuda": torch.version.cuda, "torch": torch.__version__, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__,
        "reference_operators": ops.report(), "reference_hadamard": hadamard.backend_report(),
        "native_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr, "export_origins": origins,
        "dependencies": subprocess.check_output(["ldd", str(executable)], env=env, text=True),
        "artifact_sha256": {str(path): sha256(path) for path in (executable, manifest, *exports.values(), *artifacts.glob("*.bin"))},
        "provenance": {"command": sys.argv, "build_command": command, "file_sha256": hashes,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(args.output)}), flush=True)
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("build/linux/x86_64/release"))
    parser.add_argument("--hadamard-source", type=Path, required=True)
    parser.add_argument("--hadamard-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try: main(args)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"all_passed": False, "scope": "native Block verification incomplete",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": sys.argv, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise

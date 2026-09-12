"""Native V4 compressor: published prefill/decode and separate chunk oracle.

Real ratio-4 latent, ratio-128 latent and Hadamard/FP4 Indexer compressors.
No Python executes in the native child. This is a model-component/state test,
not complete Attention, paged cache, 43-layer inference or a performance test.
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
from llaisys.models.deepseek_v4_model.layers import Compressor as IncrementalCompressor
from llaisys.models.deepseek_v4_model.state import CompressorState
from llaisys.models.deepseek_v4_backends import CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model


def raw(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native compressor verification requires Slurm GPU allocation")
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
    config = InferenceConfig.from_directory(args.source_model, max_seq_len=640)
    if (config.dim, config.head_dim, config.index_head_dim, config.rope_head_dim, config.compress_ratios[2:4]) != (4096, 512, 128, 64, (4, 128)):
        raise RuntimeError("actual Flash-0731 compressor configuration required")
    archives = [args.build_dir.resolve() / f"libllaisys-{name}.a" for name in
                ("deepseek-v4-native", "tilelang-native", "aten-native", "native-tensor", "core", "device", "device-cpu", "utils")]
    sources = [Path(__file__), root / "test/compressor_native_standalone.cpp", inference / "model.py", inference / "kernel.py",
        inference / "config.json", args.source_model / "config.json", root / "xmake.lua", *archives,
        root / "python/llaisys/libllaisys/libllaisys.so", args.hadamard_source / "csrc/fast_hadamard_transform_cuda.cu", args.hadamard_object]
    for directory in ("src/backends/native", "src/backends/tilelang", "src/backends/aten", "src/backends/hadamard", "src/models/deepseek_v4"):
        sources.extend(sorted((root / directory).glob("*.[ch]pp")))
    sources.extend(root / "xmake" / name for name in ("native_model.lua", "native_tensor.lua", "tilelang.lua", "aten.lua"))
    sources.extend(root / "python/llaisys/models" / name for name in
        ("deepseek_v4_backends.py", "deepseek_v4_model_ops.py", "deepseek_v4_reference.py", "deepseek_v4.py"))
    sources.extend(root / "python/llaisys/models/deepseek_v4_model" / name for name in ("config.py", "layers.py", "state.py", "weights.py"))
    sources.extend((Path(load_kernel.__code__.co_filename), ffi_root / "lib/libtvm_ffi.so", tl_root / "lib/libtilelang.so", tl_root / "lib/libtvm.so"))
    sources.extend(torch_root / "lib" / name for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so", "libc10_cuda.so"))
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    kernel = load_kernel(inference / "kernel.py")
    sys.modules["kernel"] = kernel
    spec = importlib.util.spec_from_file_location("published_compressor_oracle", inference / "model.py")
    published = importlib.util.module_from_spec(spec); sys.modules[spec.name] = published; spec.loader.exec_module(published)
    cfg_json.update(max_batch_size=1, max_seq_len=640)
    cfg = published.ModelArgs(**cfg_json)
    published.world_size, published.rank = 1, 0
    published.default_dtype = torch.float8_e4m3fn
    published.scale_fmt, published.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda"); torch.manual_seed(731)
    registry = OperatorRegistry(MODEL_CONTRACTS)
    register_tilelang(registry, kernel, tilelang.__version__); register_torch_model(registry)
    ops = registry.bind({name: "tilelang" if name in CONTRACTS else "torch" for name in MODEL_CONTRACTS})
    frequencies = published.precompute_freqs_cis(cfg.rope_head_dim, cfg.max_seq_len, cfg.original_seq_len,
        cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    checkpoint = args.checkpoint.resolve(strict=True)
    stamp = checkpoint.stat(); checkpoint_stamp = (stamp.st_ino, stamp.st_size, stamp.st_mtime_ns)
    header, header_sha = read_header(checkpoint)
    with checkpoint.open("rb") as handle: data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native_compressor_", dir=args.output.parent)).resolve()
    codes = {torch.bfloat16: (4, 16, 1), torch.float32: (2, 32, 1), torch.complex64: (5, 64, 1)}
    def descriptor(name, value, path, *, offset=0, expected="-"):
        return {"name": name, "dtype": codes[value.dtype], "shape": list(value.shape), "bytes": value.numel() * value.element_size(),
                "path": str(path), "offset": offset, "expected": str(expected)}
    def line(value):
        fields = [value["name"], *value["dtype"], len(value["shape"]), *value["shape"], value["bytes"], value["offset"]]
        return " ".join(map(str, fields)) + " " + json.dumps(value["path"]) + " " + json.dumps(value["expected"])
    def fixture(name, value, prefix):
        path = artifacts / f"{prefix}_{name}.bin"; content = raw(value); path.write_bytes(content)
        item = descriptor(name, value, path, expected=path); item["sha256"] = hashlib.sha256(content).hexdigest(); return item
    exported, origins = {}, {}
    def export(compiled):
        key = str(compiled.prim_func.script())
        if key not in exported:
            path = artifacts / f"quant{len(exported)}.so"; cached = getattr(compiled.adapter, "libpath", None)
            if cached:
                shutil.copyfile(cached, path)
                if sha256(cached) != sha256(path): raise RuntimeError("cached TileLang artifact copy mismatch")
                origins[str(path)] = {"mode": "cached_library_copy", "source": str(cached)}
            else:
                compiled.adapter.executable.export_library(str(path)); origins[str(path)] = {"mode": "fresh_executable_export"}
            exported[key] = path
        return exported[key]
    components = []
    hadamard_version = sha256(args.hadamard_source / "csrc/fast_hadamard_transform_cuda.cu")
    lines = [f"LLAISYS_NATIVE_COMPRESSOR_V1 {tilelang.__version__} {torch.__version__.split('+')[0]} {hadamard_version} 3"]
    with torch.inference_mode(), safe_open(checkpoint, framework="pt", device="cpu") as reader:
        for prefix, ratio, dimension, rotate in (("layers.2.attn.compressor", 4, 512, False),
            ("layers.3.attn.compressor", 128, 512, False), ("layers.2.attn.indexer.compressor", 4, 128, True)):
            official = published.Compressor(cfg, ratio, dimension, rotate)
            incremental = IncrementalCompressor(config, ops, ratio, dimension, hadamard.hadamard_transform, rotate)
            parameter_names = set(dict(official.named_parameters()))
            actual_names = {key.removeprefix(prefix + ".") for key in reader.keys() if key.startswith(prefix + ".")}
            if parameter_names != actual_names or parameter_names != set(dict(incremental.named_parameters())):
                raise RuntimeError("strict compressor checkpoint key mapping failed")
            records = []
            for name, param in official.named_parameters():
                value = reader.get_tensor(prefix + "." + name)
                if value.shape != param.shape or value.dtype not in (torch.bfloat16, torch.float32) or param.dtype != torch.float32:
                    raise RuntimeError("unexpected actual compressor weight shape/dtype")
                record = descriptor(name, value, checkpoint, offset=data_start + header[prefix + "." + name]["data_offsets"][0])
                record["payload_sha256"] = hashlib.sha256(raw(value)).hexdigest()
                record["checkpoint_key"] = prefix + "." + name
                record["reference_parameter_dtype"] = str(param.dtype); records.append(record)
                param.copy_(value); dict(incremental.named_parameters())[name].copy_(value)
            freq_record = fixture("frequencies", frequencies, prefix)
            compiled = kernel.fp4_quant_kernel(dimension, 32, inplace=True) if rotate else kernel.act_quant_kernel(
                dimension - cfg.rope_head_dim, 64, scale_dtype=kernel.FE8M0, round_scale=True, inplace=True)
            bundle = export(compiled)
            lines.append(" ".join(map(str, [prefix, cfg.dim, dimension, cfg.rope_head_dim, ratio, int(rotate), cfg.max_seq_len, cfg.norm_eps, 5])))
            lines.extend(line(item) for item in [*records, freq_record]); lines.append(json.dumps(str(bundle)))
            plans = [("published_long", [2 * ratio - 1, 1, 1], "published"),
                     ("published_short", [ratio - 1, 1, 1], "published"),
                     ("chunk_non_aligned", [3, 2, 5, 127, 1, 133], "incremental_python"),
                     ("chunk_large", [257, 257, 125 if ratio == 128 else 17, 1], "incremental_python")]
            lines.append(str(len(plans))); results = []
            for name, sizes, oracle in plans:
                state = CompressorState.allocate(1, ratio, dimension, "cuda")
                cache = torch.zeros(1, cfg.max_seq_len // ratio, dimension, dtype=torch.bfloat16, device="cuda")
                official.kv_state.zero_(); official.score_state.fill_(-torch.inf)
                official.kv_cache, official.freqs_cis = cache, frequencies
                model = official if oracle == "published" else incremental
                traces = {}; hooks = []
                for tag, module in (("projected_kv", model.wkv), ("projected_scores", model.wgate)):
                    hooks.append(module.register_forward_hook(lambda _m, _i, out, tag=tag: traces.__setitem__(tag, out.detach().clone())))
                hooks.append(model.norm.register_forward_pre_hook(lambda _m, inp: traces.__setitem__("pooled_bf16", inp[0].detach().clone())))
                lines.append(f"{name} {oracle} {len(sizes)}"); position = 0; steps = []
                try:
                    for index, tokens in enumerate(sizes):
                        traces.clear(); x = torch.randn(1, tokens, cfg.dim, dtype=torch.bfloat16, device="cuda")
                        output = official(x, position) if oracle == "published" else incremental(x, position, state, cache, frequencies)
                        emitted = (position % ratio + tokens) // ratio
                        if (output is None) != (emitted == 0): raise RuntimeError("reference emission count mismatch")
                        expected = {"input": x, "projected_kv": traces["projected_kv"], "projected_scores": traces["projected_scores"],
                            "kv_state": official.kv_state if oracle == "published" else state.kv,
                            "score_state": official.score_state if oracle == "published" else state.scores, "cache": cache}
                        if emitted: expected.update(output=output, pooled_bf16=traces["pooled_bf16"])
                        specs = [fixture(tag, value, f"{prefix}_{name}_{index}") for tag, value in expected.items()]
                        lines.append(f"{position} {tokens} {emitted} {len(specs)}"); lines.extend(line(item) for item in specs)
                        steps.append({"position": position, "tokens": tokens, "emitted": emitted, "tensors": specs})
                        position += tokens
                finally:
                    for hook in hooks: hook.remove()
                results.append({"name": name, "oracle": oracle, "tokens": sizes, "steps": steps})
                print(f"reference {prefix}/{name}: {sizes}", flush=True)
            components.append({"name": prefix, "ratio": ratio, "dimension": dimension, "rotate": rotate, "weights": records,
                "frequencies": freq_record, "quant_library": str(bundle), "plans": results})
    manifest = artifacts / "manifest.txt"; manifest.write_text("\n".join(lines) + "\n")
    executable = artifacts / "compressor-native-standalone"
    libdirs = [torch_root / "lib", ffi_root / "lib", tl_root / "lib", root / "python/llaisys/libllaisys", Path(CUDA_HOME) / "lib64"]
    command = [shutil.which("g++"), "-std=c++20", "-O2", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *["-I" + str(path) for path in (root, root / "include", ffi_root / "include", torch_root / "include",
            torch_root / "include/torch/csrc/api/include", Path(CUDA_HOME) / "include")], str(root / "test/compressor_native_standalone.cpp"),
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
    report = {"scope": "native ratio4/ratio128/rotated compressor components, not full Attention/model/paged cache", "result": result,
        "all_passed": run.returncode == 0 and result["all_passed"] and unchanged and payload_unchanged and policy_matches,
        "source_unchanged": unchanged, "selected_checkpoint_payload_unchanged": payload_unchanged,
        "reference_precision_policy_matches": policy_matches, "components": components,
        "configuration": {"hidden": cfg.dim, "rope_dimension": cfg.rope_head_dim, "capacity": cfg.max_seq_len,
                          "norm_epsilon": cfg.norm_eps, "rope_theta": cfg.compress_rope_theta,
                          "frequency_preparation": "published precompute_freqs_cis outside native execution",
                          "weight_storage": "checkpoint BF16/FP32, explicit load-time FP32 conversion",
                          "projection_and_pooling": "FP32", "latent_storage": "BF16",
                          "quantization": "non-RoPE FP8 QDQ block64 or Hadamard + FP4 QDQ block32"},
        "reference_scope": {"published": "unchanged upstream Compressor.forward: initial prefill and single-token decode",
                            "incremental_python": "existing project Compressor: matching multi-token chunk schedule, NOT original full-prefill oracle"},
        "full_vs_chunk_numerical_gate": "not claimed by this same-schedule native migration test",
        "checkpoint": {"path": str(checkpoint), "size_bytes": after.st_size, "header_sha256": header_sha,
                       "whole_model_loaded": False, "whole_payload_hashed": False},
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "cuda": torch.version.cuda, "torch": torch.__version__, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__,
        "reference_operators": ops.report(), "reference_hadamard": hadamard.backend_report(),
        "native_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr, "export_origins": origins,
        "dependencies": subprocess.check_output(["ldd", str(executable)], env=env, text=True),
        "artifact_sha256": {str(path): sha256(path) for path in (executable, manifest, *exported.values(), *artifacts.glob("*.bin"))},
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
        args.output.write_text(json.dumps({"all_passed": False, "scope": "native compressor verification incomplete",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": sys.argv, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise

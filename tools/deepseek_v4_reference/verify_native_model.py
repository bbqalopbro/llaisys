"""Full MP1 native model against identity-verified published full/decode logits.

Python only exports kernels, authenticates saved goldens and prepares input
fixtures. The child parses/loads all model weights and performs free greedy
generation without Python. This is correctness, not serving/performance or
original full/chunk equivalence acceptance.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
from transformers import AutoTokenizer
from run_independent import load_kernel, read_baselines, sha256
from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
from llaisys.models.deepseek_v4_model.config import InferenceConfig


def config_rejections(executable, source, checkpoint, capacity, artifacts, env):
    """Mutate only small private config copies; never modify shared weights."""
    hf = json.loads((source / "config.json").read_text())
    infer = json.loads((source / "inference/config.json").read_text())
    cases = []
    for field, value in (("hidden_size", True), ("hidden_size", 4096.5), ("hidden_size", -1),
                         ("architectures", ["WrongModel"]), ("n_shared_experts", 2),
                         ("expert_dtype", "fp8"), ("attention_bias", True),
                         ("qk_rope_head_dim", 65), ("compress_ratios", [0] * 46),
                         ("dspark_target_layer_ids", [40, 40, 42])):
        a = deepcopy(hf); a[field] = value; cases.append((field + "=" + str(value), a, infer))
    a = deepcopy(hf); del a["hidden_size"]; cases.append(("missing_hidden", a, infer))
    a = deepcopy(hf); a["quantization_config"]["scale_fmt"] = "fp32"; cases.append(("wrong_scale", a, infer))
    # Both configs agree, so only the actual checkpoint schema can reject this.
    a, b = deepcopy(hf), deepcopy(infer); a["hidden_size"] = b["dim"] = 4224
    cases.append(("valid_config_wrong_weight_shape", a, b))
    a, b = deepcopy(hf), deepcopy(infer); a["n_routed_experts"] = b["n_routed_experts"] = 255
    cases.append(("valid_config_wrong_weight_count", a, b))
    results = []
    for i, (name, a, b) in enumerate(cases):
        directory = artifacts / f"bad_config_{i}"; (directory / "inference").mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(a))
        (directory / "inference/config.json").write_text(json.dumps(b))
        run = subprocess.run([str(executable), "--validate", str(directory), str(checkpoint), str(capacity)],
                             env=env, capture_output=True, text=True)
        if run.returncode != 1 or not run.stderr.strip():
            raise RuntimeError(f"invalid config/schema not cleanly rejected: {name}: {run.returncode}")
        results.append({"case": name, "rejected": True, "error": run.stderr.strip()})
    return results


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native model verification requires Slurm GPU allocation")
    import tilelang
    import tvm_ffi
    from torch.utils.cpp_extension import CUDA_HOME
    if (tilelang.__version__, tvm_ffi.__version__, torch.__version__) != ("0.1.8", "0.1.8.post2", "2.13.0+cu130"):
        raise RuntimeError("native ABI/version mismatch")
    if not CUDA_HOME or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("compatible CUDA toolkit/CXX11 ABI required")
    root = Path(__file__).resolve().parents[2]
    source, checkpoint = args.source_model.resolve(), args.checkpoint.resolve(strict=True)
    inference = source / "inference"
    cfg = InferenceConfig.from_directory(source, max_seq_len=args.max_seq_len)
    torch_root, ffi_root, tl_root = Path(torch.__file__).parent, Path(tvm_ffi.__file__).parent, Path(tilelang.__file__).parent
    rapidjson = tl_root / "3rdparty/composable_kernel/include/rapidjson"
    archives = [args.build_dir.resolve() / f"libllaisys-{name}.a" for name in
                ("deepseek-v4-native", "tilelang-native", "aten-native", "native-tensor", "core", "device", "device-cpu", "utils")]
    sources = [Path(__file__), root / "test/model_native_standalone.cpp", root / "test/native_fixture_utils.hpp",
        inference / "model.py", inference / "kernel.py", inference / "config.json", source / "config.json",
        root / "xmake.lua", *archives, root / "python/llaisys/libllaisys/libllaisys.so",
        args.hadamard_source / "csrc/fast_hadamard_transform_cuda.cu", args.hadamard_object,
        Path(load_kernel.__code__.co_filename), Path(checkpoint_identity.__code__.co_filename),
        root / "python/llaisys/models/deepseek_v4_model/config.py", *sorted(rapidjson.rglob("*.h"))]
    for directory in ("src/backends/native", "src/backends/tilelang", "src/backends/aten", "src/backends/hadamard", "src/models/deepseek_v4"):
        sources.extend(sorted((root / directory).glob("*.[ch]pp")))
    sources.extend(root / "xmake" / name for name in ("native_model.lua", "native_tensor.lua", "tilelang.lua", "aten.lua"))
    sources.extend((ffi_root / "lib/libtvm_ffi.so", tl_root / "lib/libtilelang.so", tl_root / "lib/libtvm.so"))
    sources.extend(torch_root / "lib" / name for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so", "libc10_cuda.so"))
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native_model_", dir=args.output.parent)).resolve()
    executable = artifacts / "model-native-standalone"
    libdirs = [torch_root / "lib", ffi_root / "lib", tl_root / "lib", root / "python/llaisys/libllaisys", Path(CUDA_HOME) / "lib64"]
    command = [shutil.which("g++"), "-std=c++20", "-O2", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *["-I" + str(path) for path in (root, root / "include", ffi_root / "include", torch_root / "include",
            torch_root / "include/torch/csrc/api/include", Path(CUDA_HOME) / "include")], str(root / "test/model_native_standalone.cpp"),
        "-Wl,--start-group", *map(str, archives), "-Wl,--end-group", *["-L" + str(path) for path in libdirs], "-Wl,--no-as-needed",
        "-lllaisys", "-ltilelang", "-ltvm", "-ltvm_ffi", "-ltorch_cuda", "-ltorch_cpu", "-lc10_cuda", "-lc10", "-lcudart", "-ldl",
        *["-Wl,-rpath," + str(path) for path in libdirs], "-o", str(executable)]
    print("compile:", json.dumps(command), flush=True); subprocess.run(command, check=True)
    nvidia = torch_root.parent / "nvidia"
    child_libs = [torch_root / "lib", nvidia / "cu13/lib", *sorted(nvidia.glob("*/lib")), *libdirs[1:]]
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(map(str, child_libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    validate = subprocess.run([str(executable), "--validate", str(source), str(checkpoint), str(args.max_seq_len)],
                              env=env, capture_output=True, text=True)
    print(validate.stdout, end="", flush=True)
    if validate.returncode: raise RuntimeError("native schema validation failed: " + validate.stderr)
    schema = json.loads(validate.stdout)
    if schema["layers"] != cfg.n_layers or schema["required_weights"] != 67569 or schema["auxiliary_weights"] != 4702:
        raise RuntimeError("actual Flash-0731 whole checkpoint coverage differs")
    rejected = config_rejections(executable, source, checkpoint, args.max_seq_len, artifacts, env)
    print(f"native schema: complete; {len(rejected)} malformed configs/schema mismatches rejected", flush=True)
    print("hashing ENTIRE checkpoint payload before accepting published goldens", flush=True)
    identity = checkpoint_identity(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    cases = read_baselines(args.baseline, source, checkpoint, args.max_seq_len, tokenizer, cfg.vocab_size, identity)
    if not all(case["checkpoint_payload_verified"] for case in cases):
        raise RuntimeError("native full-model acceptance requires payload-verified published goldens")
    print("verified published goldens and full checkpoint identity", flush=True)
    kernel = load_kernel(inference / "kernel.py"); sys.modules["kernel"] = kernel
    spec = importlib.util.spec_from_file_location("published_native_frequency_oracle", inference / "model.py")
    published = importlib.util.module_from_spec(spec); sys.modules[spec.name] = published; spec.loader.exec_module(published)
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
    for name, width in (("quant_hidden", cfg.dim), ("quant_query_rank", cfg.q_lora_rank), ("quant_output", cfg.o_groups * cfg.o_lora_rank),
                        ("quant_intermediate", cfg.moe_inter_dim)):
        export(name, "act_quant", kernel.act_quant_kernel(width, 128, scale_dtype=kernel.FE8M0, round_scale=True))
    for name, n, k in (("gemm_query_a", cfg.q_lora_rank, cfg.dim), ("gemm_query_b", cfg.n_heads * cfg.head_dim, cfg.q_lora_rank),
                      ("gemm_latent", cfg.head_dim, cfg.dim), ("gemm_output", cfg.dim, cfg.o_groups * cfg.o_lora_rank),
                      ("indexer_gemm", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)):
        export(name, "fp8_gemm", kernel.fp8_gemm_kernel(n, k, scale_dtype=kernel.FE8M0))
    export("latent_qdq", "act_quant", kernel.act_quant_kernel(cfg.head_dim - cfg.rope_head_dim, 64, scale_dtype=kernel.FE8M0, round_scale=True, inplace=True))
    export("indexer_qdq", "fp4_act_quant", kernel.fp4_quant_kernel(cfg.index_head_dim, 32, inplace=True))
    export("sparse_attn", "sparse_attn", kernel.sparse_attn_kernel(cfg.n_heads, cfg.head_dim, cfg.head_dim ** -0.5))
    for mode in ("fp4", "fp8"):
        for projection, n, k in (("gate", cfg.moe_inter_dim, cfg.dim), ("down", cfg.dim, cfg.moe_inter_dim)):
            export(f"{mode}_{projection}", f"{mode}_gemm", getattr(kernel, f"{mode}_gemm_kernel")(n, k, scale_dtype=kernel.FE8M0))
    export("hc_split_sinkhorn", "hc_split_sinkhorn", kernel.hc_split_sinkhorn_kernel(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps))
    hadamard_version = sha256(args.hadamard_source / "csrc/fast_hadamard_transform_cuda.cu")
    lines = [f"LLAISYS_NATIVE_MODEL_V1 {tilelang.__version__} {torch.__version__.split('+')[0]} {hadamard_version}",
             f"{json.dumps(str(source))} {json.dumps(str(checkpoint))} {args.max_seq_len} {len(exports)}"]
    lines.extend(f"{key} {export_ops[key]} {json.dumps(str(path))}" for key, path in exports.items())
    def fixture(name, tensor):
        path = artifacts / f"{name}.bin"; path.write_bytes(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()); return path
    # Only frequency tables are generated here; no upstream model/weights are
    # loaded in the parent, leaving the whole GPU to the real native loader.
    with torch.device("cuda"), torch.inference_mode():
        for compressed in (False, True):
            freqs = published.precompute_freqs_cis(cfg.rope_head_dim, args.max_seq_len,
                cfg.original_seq_len if compressed else 0, cfg.compress_rope_theta if compressed else cfg.rope_theta,
                cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
            lines.append(json.dumps(str(fixture(f"frequencies_{compressed}", freqs))))
    lines.append(f"{len(cases)} {args.repeat}")
    for index, case in enumerate(cases):
        t = case["tensors"]; ids, output, logits = t["input_ids"], t["generated_ids"], t["logits"]
        lines.append(f"case_{index} {ids.shape[1]} {output.numel()} {json.dumps(str(fixture(f'case_{index}_ids', ids)))}")
        for step, token in enumerate(output.tolist()):
            lines.append(f"{token} {json.dumps(str(fixture(f'case_{index}_logits_{step}', logits[step])))}")
    manifest = artifacts / "manifest.txt"; manifest.write_text("\n".join(lines) + "\n")
    torch.cuda.synchronize(); published.precompute_freqs_cis.cache_clear(); del freqs; torch.cuda.empty_cache()
    print("launching native 43-layer model (no Python in child)", flush=True)
    # Stream child output to Slurm immediately; keep it for the audit report.
    with subprocess.Popen([str(executable), str(manifest)], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
        output_lines = []
        for row in child.stdout:
            output_lines.append(row); print(row, end="", flush=True)
        code = child.wait()
    stdout = "".join(output_lines)
    result = json.loads(output_lines[-1]) if code == 0 else {"all_passed": False}
    unchanged = hashes == {str(path.resolve()): sha256(path) for path in sources}
    verify_checkpoint_unchanged(checkpoint, identity)
    policy_matches = result.get("allow_tf32_cublas") == torch.backends.cuda.matmul.allow_tf32
    passed = code == 0 and result["all_passed"] and unchanged and policy_matches
    report = {"scope": "full native MP1 43-layer free greedy generation; continuous/ring cache; not pybind/serving/performance",
        "all_passed": passed, "result": result, "schema": schema, "config_rejections": rejected,
        "source_unchanged": unchanged, "checkpoint_unchanged": True, "checkpoint_identity": identity,
        "reference_precision_policy_matches": policy_matches, "full_vs_chunk_numerical_gate": "not claimed; original gate unchanged",
        "measurement": {"kind": "correctness_only", "batch": 1, "concurrency": 1, "warmup": 0, "repeats": args.repeat,
                        "cuda_graph": False, "prefix_cache": False, "paged_cache": False, "fallback": False},
        "configuration": {"capacity": args.max_seq_len, "layers": cfg.n_layers, "experts": cfg.n_routed_experts, "topk": cfg.n_activated_experts,
                          "precision": "W8A8 dense/shared; W4A8 routed; BF16 hidden/cache; FP32 norms/HC/compressor/router; no MTP execution"},
        "cases": [{k: v for k, v in case.items() if k != "tensors"} for case in cases],
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "cuda": torch.version.cuda, "torch": torch.__version__, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__,
        "native_exit_code": code, "stdout": stdout, "export_origins": origins,
        "dependencies": subprocess.check_output(["ldd", str(executable)], env=env, text=True),
        "artifact_sha256": {str(path): sha256(path) for path in (executable, manifest, *exports.values(), *artifacts.glob("*.bin"))},
        "provenance": {"command": sys.argv, "build_command": command, "file_sha256": hashes,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": passed, "output": str(args.output)}), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", nargs=2, action="append", required=True)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--build-dir", type=Path, default=Path("build/linux/x86_64/release"))
    parser.add_argument("--hadamard-source", type=Path, required=True)
    parser.add_argument("--hadamard-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try: main(args)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"all_passed": False, "scope": "native full-model verification incomplete",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": sys.argv, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise

"""Actual-weight native HC pre/post/head against unchanged published Block methods.

Python prepares weights, golden tensors and an exported TileLang kernel. The
native child executes no Python. This is HC component acceptance, not full
Transformer/model, task accuracy, paged storage or performance acceptance.
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
from types import SimpleNamespace

import torch
from safetensors import safe_open
from run_independent import load_kernel, sha256
from llaisys.models.deepseek_v4_model.weights import read_header


def raw(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native HC verification requires Slurm GPU allocation")
    import tilelang
    import tvm_ffi
    from torch.utils.cpp_extension import CUDA_HOME
    if (tilelang.__version__, tvm_ffi.__version__, torch.__version__) != ("0.1.8", "0.1.8.post2", "2.13.0+cu130"):
        raise RuntimeError("native ABI/version mismatch")
    if not CUDA_HOME or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("compatible CUDA toolkit/CXX11 ABI required")
    root = Path(__file__).resolve().parents[2]
    inference = args.source_model.resolve() / "inference"
    torch_root, ffi_root, tl_root = Path(torch.__file__).parent, Path(tvm_ffi.__file__).parent, Path(tilelang.__file__).parent
    cfg = json.loads((inference / "config.json").read_text())
    if (cfg["dim"], cfg["hc_mult"], cfg["hc_sinkhorn_iters"], cfg["n_layers"]) != (4096, 4, 20, 43):
        raise RuntimeError("actual Flash-0731 HC configuration required")
    # Resolve defaults from the actual published ModelArgs, not hard-coded model-family assumptions.
    archives = [args.build_dir.resolve() / f"libllaisys-{name}.a" for name in
                ("deepseek-v4-native", "tilelang-native", "aten-native", "native-tensor", "core", "device", "device-cpu", "utils")]
    sources = [Path(__file__), root / "test/hc_native_standalone.cpp", root / "test/native_fixture_utils.hpp",
        inference / "model.py", inference / "kernel.py", inference / "config.json", args.source_model / "config.json",
        root / "xmake.lua", *archives, root / "python/llaisys/libllaisys/libllaisys.so"]
    for directory in ("src/backends/native", "src/backends/tilelang", "src/backends/aten", "src/models/deepseek_v4"):
        sources.extend(sorted((root / directory).glob("*.[ch]pp")))
    sources.extend(root / "xmake" / name for name in ("native_model.lua", "native_tensor.lua", "tilelang.lua", "aten.lua"))
    sources.extend((Path(load_kernel.__code__.co_filename), Path(read_header.__code__.co_filename),
                    ffi_root / "lib/libtvm_ffi.so", tl_root / "lib/libtilelang.so", tl_root / "lib/libtvm.so"))
    sources.extend(torch_root / "lib" / name for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so", "libc10_cuda.so"))
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    kernel = load_kernel(inference / "kernel.py"); sys.modules["kernel"] = kernel
    spec = importlib.util.spec_from_file_location("published_hc_oracle", inference / "model.py")
    published = importlib.util.module_from_spec(spec); sys.modules[spec.name] = published; spec.loader.exec_module(published)
    config = published.ModelArgs(**cfg)
    oracle = SimpleNamespace(norm_eps=config.norm_eps, hc_eps=config.hc_eps,
                             hc_mult=config.hc_mult, hc_sinkhorn_iters=config.hc_sinkhorn_iters)
    torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda"); torch.manual_seed(731)
    checkpoint = args.checkpoint.resolve(strict=True)
    stamp = checkpoint.stat(); checkpoint_stamp = (stamp.st_ino, stamp.st_size, stamp.st_mtime_ns)
    header, header_sha = read_header(checkpoint)
    with checkpoint.open("rb") as handle: data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native_hc_", dir=args.output.parent)).resolve()
    codes = {torch.bfloat16: (4, 16, 1), torch.float32: (2, 32, 1)}
    def descriptor(name, value, path, *, offset=0, expected="-"):
        return {"name": name, "dtype": codes[value.dtype], "shape": list(value.shape), "bytes": value.numel() * value.element_size(),
                "path": str(path), "offset": offset, "expected": str(expected)}
    def line(value):
        fields = [value["name"], *value["dtype"], len(value["shape"]), *value["shape"], value["bytes"], value["offset"]]
        return " ".join(map(str, fields)) + " " + json.dumps(value["path"]) + " " + json.dumps(value["expected"])
    def fixture(name, value, prefix):
        path = artifacts / f"{prefix}_{name}.bin"; content = raw(value); path.write_bytes(content)
        item = descriptor(name, value, path, expected=path); item["sha256"] = hashlib.sha256(content).hexdigest(); return item
    compiled = kernel.hc_split_sinkhorn_kernel(config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps)
    bundle = artifacts / "hc_split.so"; cached = getattr(compiled.adapter, "libpath", None)
    if cached:
        shutil.copyfile(cached, bundle)
        if sha256(cached) != sha256(bundle): raise RuntimeError("cached TileLang artifact mismatch")
    else:
        compiled.adapter.executable.export_library(str(bundle))
    components = []
    prefixes = [f"layers.{layer}.hc_{part}" for layer in (0, 2, 3, 42) for part in ("attn", "ffn")] + ["hc_head"]
    lines = [f"LLAISYS_NATIVE_HC_V1 {tilelang.__version__} {torch.__version__.split('+')[0]} {config.dim} {config.hc_mult} "
             f"{config.hc_sinkhorn_iters} {config.norm_eps} {config.hc_eps} {json.dumps(str(bundle))} {len(prefixes)}"]
    with torch.inference_mode(), safe_open(checkpoint, framework="pt", device="cpu") as reader:
        for prefix in prefixes:
            head = prefix == "hc_head"; records, weights = [], []
            lines.append(f"{prefix} {int(head)}")
            for tag, suffix in (("projection", "fn"), ("scale", "scale"), ("base", "base")):
                key = f"{prefix}_{suffix}"; value = reader.get_tensor(key)
                if value.dtype != torch.float32: raise RuntimeError(f"actual HC weight must be FP32: {key}")
                item = descriptor(tag, value, checkpoint, offset=data_start + header[key]["data_offsets"][0])
                item.update(source_key=key, payload_sha256=hashlib.sha256(raw(value)).hexdigest())
                records.append(item); lines.append(line(item)); weights.append(value.cuda())
            fn, scale, base = weights; cases = []
            profiles = [(1, 1.0), (3, 1.0), (32, 0.0), (33, 16.0), (137, 1.0), (257, 1.0)]
            lines.append(str(len(profiles)))
            for tokens, amplitude in profiles:
                name = f"tokens{tokens}"
                x = torch.randn(1, tokens, config.hc_mult, config.dim, dtype=torch.bfloat16) * amplitude
                flat = x.flatten(2).float()
                inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.norm_eps)
                mixes = torch.nn.functional.linear(flat, fn) * inv
                if head:
                    reduced = published.Block.hc_head(oracle, x, fn, scale, base)
                    pre = torch.sigmoid(mixes * scale + base) + config.hc_eps
                    expected = {"input": x, "reduced": reduced, "mixes": mixes, "inverse_rms": inv, "pre_weights": pre}
                else:
                    reduced, post, comb = published.Block.hc_pre(oracle, x, fn, scale, base)
                    pre, _, _ = kernel.hc_split_sinkhorn(mixes, scale, base, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps)
                    branch = torch.randn(1, tokens, config.dim, dtype=torch.bfloat16)
                    expanded = published.Block.hc_post(oracle, branch, x, post, comb)
                    expected = {"input": x, "branch": branch, "reduced": reduced, "mixes": mixes, "inverse_rms": inv,
                                "pre_weights": pre, "post_weights": post, "combination": comb, "expanded": expanded}
                specs = [fixture(tag, value, f"{prefix}_{name}") for tag, value in expected.items()]
                lines.append(f"{name} {tokens} {len(specs)}"); lines.extend(line(item) for item in specs)
                cases.append({"name": name, "tokens": tokens, "amplitude": amplitude, "tensors": specs})
            components.append({"name": prefix, "head": head, "weights": records, "cases": cases})
            print(f"reference {prefix}: {[item[0] for item in profiles]}", flush=True)
    manifest = artifacts / "manifest.txt"; manifest.write_text("\n".join(lines) + "\n")
    executable = artifacts / "hc-native-standalone"
    libdirs = [torch_root / "lib", ffi_root / "lib", tl_root / "lib", root / "python/llaisys/libllaisys", Path(CUDA_HOME) / "lib64"]
    command = [shutil.which("g++"), "-std=c++20", "-O2", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *["-I" + str(path) for path in (root, root / "include", ffi_root / "include", torch_root / "include",
            torch_root / "include/torch/csrc/api/include", Path(CUDA_HOME) / "include")], str(root / "test/hc_native_standalone.cpp"),
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
    report = {"scope": "actual-weight native HC pre/post/head; NOT full Block/model, paged cache or performance", "result": result,
        "all_passed": run.returncode == 0 and result["all_passed"] and unchanged and payload_unchanged and policy_matches,
        "source_unchanged": unchanged, "selected_checkpoint_payload_unchanged": payload_unchanged,
        "reference_precision_policy_matches": policy_matches, "components": components,
        "configuration": {"hidden": config.dim, "copies": config.hc_mult, "sinkhorn_iterations": config.hc_sinkhorn_iters,
                          "norm_epsilon": config.norm_eps, "hc_epsilon": config.hc_eps, "batch": 1,
                          "precision": "FP32 weights/projection/reduction/mixing; BF16 hidden and branch",
                          "inputs": "deterministic random/zero/large-amplitude hidden, not text"},
        "reference_scope": "unchanged published Block.hc_pre/hc_post/hc_head; separately recorded FP32 intermediate expressions",
        "checkpoint": {"path": str(checkpoint), "size_bytes": after.st_size, "header_sha256": header_sha,
                       "whole_model_loaded": False, "whole_payload_hashed": False},
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "cuda": torch.version.cuda, "torch": torch.__version__, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__,
        "native_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
        "dependencies": subprocess.check_output(["ldd", str(executable)], env=env, text=True),
        "artifact_sha256": {str(path): sha256(path) for path in (executable, manifest, bundle, *artifacts.glob("*.bin"))},
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try: main(args)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"all_passed": False, "scope": "native HC verification incomplete",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": sys.argv, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise

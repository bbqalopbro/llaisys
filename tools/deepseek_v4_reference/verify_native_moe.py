"""Validate complete native V4 MoE layers against unchanged published MoE.forward.

Loads ALL experts of actual layers 0 (hash) and 3 (learned) from the MP1
checkpoint. Python only prepares reference fixtures/JIT bundles; the child
loads native tensors and executes the complete MoE with no Python runtime.
This is not 43-layer model/serving or TTFT/TPOT acceptance.
"""
import argparse
import gc
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


def raw(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native MoE verification requires a Slurm GPU allocation")
    import tilelang
    import tvm_ffi
    from torch.utils.cpp_extension import CUDA_HOME
    if (tilelang.__version__, tvm_ffi.__version__, torch.__version__) != ("0.1.8", "0.1.8.post2", "2.13.0+cu130"):
        raise RuntimeError("native reference ABI/version mismatch")
    if not CUDA_HOME or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("compatible CUDA toolkit/CXX11 ABI required")
    root = Path(__file__).resolve().parents[2]
    inference = args.source_model.resolve() / "inference"
    torch_root, ffi_root, tl_root = Path(torch.__file__).parent, Path(tvm_ffi.__file__).parent, Path(tilelang.__file__).parent
    config = json.loads((inference / "config.json").read_text())
    if (config["dim"], config["moe_inter_dim"], config["n_routed_experts"], config["n_activated_experts"],
        config["n_shared_experts"], config["n_hash_layers"], config["dtype"], config["expert_dtype"]) != (4096, 2048, 256, 6, 1, 3, "fp8", "fp4"):
        raise RuntimeError("fixture requires actual Flash-0731 MoE configuration, not an assumed model family")
    archives = [args.build_dir.resolve() / f"libllaisys-{name}.a" for name in
                ("deepseek-v4-native", "tilelang-native", "aten-native", "native-tensor", "core", "device", "device-cpu", "utils")]
    sources = [Path(__file__), root / "test/moe_native_standalone.cpp", inference / "model.py", inference / "kernel.py",
               inference / "config.json", root / "xmake.lua", *archives, root / "python/llaisys/libllaisys/libllaisys.so"]
    for directory in ("src/backends/native", "src/backends/tilelang", "src/backends/aten", "src/models/deepseek_v4"):
        sources.extend(sorted((root / directory).glob("*.[ch]pp")))
    sources.extend(root / "xmake" / name for name in ("native_model.lua", "native_tensor.lua", "tilelang.lua", "aten.lua"))
    sources.extend((Path(load_kernel.__code__.co_filename), Path(read_header.__code__.co_filename)))
    sources.extend(torch_root / "lib" / name for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so", "libc10_cuda.so"))
    sources.extend((ffi_root / "lib/libtvm_ffi.so", tl_root / "lib/libtilelang.so", tl_root / "lib/libtvm.so"))
    hashes = {str(path): sha256(path) for path in sources}
    checkpoint = args.checkpoint.resolve(strict=True)
    stat_before = checkpoint.stat()
    checkpoint_stamp = (stat_before.st_ino, stat_before.st_size, stat_before.st_mtime_ns)
    header, header_sha = read_header(checkpoint)
    with checkpoint.open("rb") as handle:
        data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
    kernel = load_kernel(inference / "kernel.py")
    sys.modules["kernel"] = kernel
    spec = importlib.util.spec_from_file_location("published_native_moe_oracle", inference / "model.py")
    published = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = published
    spec.loader.exec_module(published)
    cfg = published.ModelArgs(**config)
    # Same initialization as published Transformer, without allocating its
    # attention/cache/MTP tensors. Do not replace MoE.forward or its operators.
    published.world_size, published.rank = 1, 0
    published.default_dtype = torch.float8_e4m3fn
    published.scale_fmt, published.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(731)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native_moe_", dir=args.output.parent)).resolve()
    codes = {torch.bfloat16: (4, 16, 1), torch.float32: (2, 32, 1), torch.int32: (0, 32, 1),
             torch.int64: (0, 64, 1), torch.float8_e4m3fn: (10, 8, 1),
             torch.float8_e8m0fnu: (14, 8, 1), torch.float4_e2m1fn_x2: (17, 4, 1)}
    def descriptor(name, value, path, offset=0, expected="-"):
        shape = list(value.shape)
        if value.dtype == torch.float4_e2m1fn_x2:
            shape[-1] *= 2
        return {"name": name, "dtype": codes[value.dtype], "shape": shape, "bytes": value.numel() * value.element_size(),
                "path": str(path), "offset": offset, "expected": str(expected)}
    def line(value):
        fields = [value["name"], *value["dtype"], len(value["shape"]), *value["shape"], value["bytes"], value["offset"]]
        return " ".join(map(str, fields)) + " " + json.dumps(value["path"]) + " " + json.dumps(value["expected"])
    exports, origins = {}, {}
    def export(name, op, compiled):
        key = str(compiled.prim_func.script())
        if key not in exports:
            path = artifacts / f"kernel{len(exports)}.so"
            cached = getattr(compiled.adapter, "libpath", None)
            if cached:
                shutil.copyfile(cached, path)
                if sha256(cached) != sha256(path):
                    raise RuntimeError("cached TileLang bundle copy mismatch")
                origins[str(path)] = {"mode": "cached_library_copy", "source": str(cached)}
            else:
                compiled.adapter.executable.export_library(str(path))
                origins[str(path)] = {"mode": "fresh_executable_export"}
            exports[key] = path
        return {"name": name, "operation": op, "path": str(exports[key])}
    bundles = [export("quant_hidden", "act_quant", kernel.act_quant_kernel(cfg.dim, 128, scale_dtype=kernel.FE8M0, round_scale=True)),
               export("quant_intermediate", "act_quant", kernel.act_quant_kernel(cfg.moe_inter_dim, 128, scale_dtype=kernel.FE8M0, round_scale=True))]
    for mode in ("fp4", "fp8"):
        for name, n, k in (("gate", cfg.moe_inter_dim, cfg.dim), ("down", cfg.dim, cfg.moe_inter_dim)):
            bundles.append(export(f"{mode}_{name}", f"{mode}_gemm", getattr(kernel, f"{mode}_gemm_kernel")(n, k, scale_dtype=kernel.FE8M0)))
    layers = []
    manifest_lines = [f"LLAISYS_NATIVE_MOE_V1 {tilelang.__version__} {torch.__version__.split('+')[0]} 2"]
    with torch.inference_mode(), safe_open(checkpoint, framework="pt", device="cpu") as reader:
        for layer in (0, 3):
            print(f"prepare complete published MoE layer {layer}", flush=True)
            model = published.MoE(layer, cfg)
            records = []
            prefix = f"layers.{layer}.ffn."
            expected_keys = {key.removeprefix(prefix) for key in reader.keys() if key.startswith(prefix)}
            parameters = dict(model.named_parameters())
            if set(parameters) != expected_keys:
                raise RuntimeError(f"strict actual MoE weight mapping failed for layer {layer}")
            for name, param in parameters.items():
                key = prefix + name
                value = reader.get_tensor(key)
                hash_table_cast = name == "gate.tid2eid" and value.dtype == torch.int64 and param.dtype == torch.int32
                if value.shape != param.shape or (value.dtype != param.dtype and not hash_table_cast):
                    raise RuntimeError(f"actual weight shape/dtype mismatch: {key}, {value.dtype}, {param.dtype}")
                if hash_table_cast and (value.min().item() < 0 or value.max().item() >= cfg.n_routed_experts):
                    raise RuntimeError("hash table contains invalid expert IDs")
                payload = raw(value)
                record = descriptor(name, value, checkpoint, data_start + header[key]["data_offsets"][0])
                record["checkpoint_key"] = key
                record["payload_sha256"] = hashlib.sha256(payload).hexdigest()
                record["reference_parameter_dtype"] = str(param.dtype)
                records.append(record)
                param.copy_(value)
            del value, payload
            # copy_ preserves the published weight.scale alias; assignment of
            # new Parameters would sever it and silently invalidate the oracle.
            for module in model.modules():
                if isinstance(module, published.Linear) and module.scale is not None:
                    if module.weight.scale is not module.scale:
                        raise RuntimeError("published weight.scale ownership alias broken")
            manifest_lines.append(" ".join(map(str, [layer, cfg.dim, cfg.moe_inter_dim, cfg.n_routed_experts,
                cfg.n_activated_experts, cfg.vocab_size, int(layer < cfg.n_hash_layers), cfg.route_scale,
                cfg.swiglu_limit, cfg.score_func, len(records)])))
            manifest_lines.extend(line(record) for record in records)
            manifest_lines.append(str(len(bundles)))
            manifest_lines.extend(f"{bundle['name']} {bundle['operation']} {json.dumps(bundle['path'])}" for bundle in bundles)
            manifest_lines.append("3")
            cases = []
            for rows in (1, 33, 137):
                name = f"layer{layer}_m{rows}"
                x = torch.randn(rows, cfg.dim, dtype=torch.bfloat16, device="cuda")
                ids = torch.randint(0, cfg.vocab_size, (rows,), device="cuda", dtype=torch.int64)
                captured, expert_outputs = {}, {}
                def capture_gate(_module, _input, output):
                    captured["weights"], captured["ids"] = (part.detach().clone() for part in output)
                hooks = [model.gate.register_forward_hook(capture_gate)]
                for e, expert in enumerate(model.experts):
                    hooks.append(expert.register_forward_hook(lambda _m, _i, out, e=e: expert_outputs.__setitem__(e, out.detach().clone())))
                hooks.append(model.shared_experts.register_forward_hook(lambda _m, _i, out: captured.__setitem__("shared", out.detach().clone())))
                try:
                    # The actual, unchanged upstream implementation is the
                    # golden. Diagnostic reconstruction below must match it.
                    output = model(x, ids)
                finally:
                    for hook in hooks:
                        hook.remove()
                torch.cuda.synchronize()
                logits = torch.nn.functional.linear(x.float(), model.gate.weight.float())
                route_weights, route_ids = captured["weights"], captured["ids"]
                order = torch.argsort(route_ids.flatten().long(), stable=True)
                packed_rows = torch.div(order, cfg.n_activated_experts, rounding_mode="floor")
                counts = torch.bincount(route_ids.flatten().long(), minlength=cfg.n_routed_experts)
                offsets = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64), counts.cumsum(0)))
                packed_output = torch.empty(rows * cfg.n_activated_experts, cfg.dim, dtype=torch.bfloat16, device="cuda")
                routed = torch.zeros_like(x, dtype=torch.float32)
                host_offsets = offsets.tolist()
                for e, result in expert_outputs.items():
                    start, end = host_offsets[e:e + 2]
                    packed_output[start:end] = result
                    selected_rows, _ = torch.where(route_ids == e)
                    routed[selected_rows] += result
                reconstructed = (routed + captured["shared"]).to(torch.bfloat16)
                if raw(reconstructed) != raw(output):
                    raise RuntimeError("diagnostic reconstruction differs from original published MoE")
                expected = {"input": x, "tokens": ids, "output": output, "logits": logits,
                    "weights": route_weights, "ids": route_ids, "packed_hidden": x[packed_rows],
                    "packed_weights": route_weights.flatten()[order, None], "packed_rows": packed_rows,
                    "offsets": offsets, "packed_output": packed_output, "routed": routed, "shared": captured["shared"],
                    "hash_or_bias": model.gate.tid2eid[ids] if layer < cfg.n_hash_layers else model.gate.bias}
                fixtures = []
                for tag, value in expected.items():
                    path = artifacts / f"{name}_{tag}.bin"
                    value_bytes = raw(value); path.write_bytes(value_bytes)
                    record = descriptor(tag, value, path, expected=path)
                    record["sha256"] = hashlib.sha256(value_bytes).hexdigest()
                    fixtures.append(record)
                # Initialize the native output to a guard, not its golden.
                out_record = next(item for item in fixtures if item["name"] == "output")
                guard = artifacts / f"{name}_output_guard.bin"
                guard.write_bytes(bytes([0x5a]) * out_record["bytes"])
                out_record["path"] = str(guard)
                manifest_lines.append(f"{name} {rows} {len(fixtures)}")
                manifest_lines.extend(line(record) for record in fixtures)
                cases.append({"name": name, "rows": rows, "routed_experts_executed": len(expert_outputs),
                              "empty_experts": cfg.n_routed_experts - len(expert_outputs), "tensors": fixtures})
                print(f"reference {name}: experts={len(expert_outputs)}, exact reconstructed output", flush=True)
            layers.append({"layer": layer, "hash_routing": layer < cfg.n_hash_layers, "weight_count": len(records),
                           "weight_bytes": sum(record["bytes"] for record in records), "weights": records, "cases": cases})
            del model, parameters, expected, expert_outputs, captured
            gc.collect(); torch.cuda.empty_cache()
    manifest = artifacts / "manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n")
    executable = artifacts / "moe-native-standalone"
    libdirs = [torch_root / "lib", ffi_root / "lib", tl_root / "lib", root / "python/llaisys/libllaisys", Path(CUDA_HOME) / "lib64"]
    command = [shutil.which("g++"), "-std=c++20", "-O2", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *["-I" + str(path) for path in (root, root / "include", ffi_root / "include", torch_root / "include",
                                       torch_root / "include/torch/csrc/api/include", Path(CUDA_HOME) / "include")],
        str(root / "test/moe_native_standalone.cpp"), "-Wl,--start-group", *map(str, archives), "-Wl,--end-group",
        *["-L" + str(path) for path in libdirs], "-Wl,--no-as-needed", "-lllaisys", "-ltilelang", "-ltvm", "-ltvm_ffi",
        "-ltorch_cuda", "-ltorch_cpu", "-lc10_cuda", "-lc10", "-lcudart", "-ldl",
        *["-Wl,-rpath," + str(path) for path in libdirs], "-o", str(executable)]
    print("compile:", json.dumps(command), flush=True)
    subprocess.run(command, check=True)
    nvidia = torch_root.parent / "nvidia"
    child_libs = [torch_root / "lib", nvidia / "cu13/lib", *sorted(nvidia.glob("*/lib")), *libdirs[1:]]
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(map(str, child_libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    run = subprocess.run([str(executable), str(manifest)], env=env, capture_output=True, text=True)
    print(run.stdout, end="", flush=True); print(run.stderr, end="", file=sys.stderr, flush=True)
    result = json.loads(run.stdout.splitlines()[-1]) if run.returncode == 0 else {"all_passed": False}
    unchanged = hashes == {str(path): sha256(path) for path in sources}
    stat_after = checkpoint.stat()
    checkpoint_unchanged = checkpoint_stamp == (stat_after.st_ino, stat_after.st_size, stat_after.st_mtime_ns)
    # Re-hash every selected tensor extent after native execution, not only the
    # safetensors header, while avoiding reading the other 41 huge layers.
    with checkpoint.open("rb") as handle:
        for layer in layers:
            for record in layer["weights"]:
                handle.seek(record["offset"])
                if hashlib.sha256(handle.read(record["bytes"])).hexdigest() != record["payload_sha256"]:
                    checkpoint_unchanged = False
    policy_matches = result.get("allow_tf32_cublas") == torch.backends.cuda.matmul.allow_tf32
    artifacts_sha = {str(path): sha256(path) for path in (executable, manifest, *exports.values(), *artifacts.glob("*.bin"))}
    report = {"scope": "complete native single-rank MoE layers 0 and 3; not a complete 43-layer model or EP",
        "all_passed": run.returncode == 0 and result["all_passed"] and unchanged and checkpoint_unchanged and policy_matches,
        "result": result, "native_exit_code": run.returncode, "source_unchanged": unchanged,
        "selected_checkpoint_payload_unchanged": checkpoint_unchanged, "reference_precision_policy_matches": policy_matches,
        "reference": "unchanged published model.py MoE.forward, all 256 routed experts and one shared expert loaded",
        "input_kind": "deterministic random BF16 hidden activations and real-vocabulary token IDs; not a text-level test",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "cuda": torch.version.cuda, "torch": torch.__version__, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__,
        "checkpoint": {"path": str(checkpoint), "header_sha256": header_sha, "size_bytes": stat_after.st_size,
                       "whole_model_loaded": False, "whole_payload_hashed": False},
        "configuration": config, "layers": layers, "bundles": bundles, "export_origins": origins,
        "stdout": run.stdout, "stderr": run.stderr, "dependencies": subprocess.check_output(["ldd", str(executable)], env=env, text=True),
        "artifact_sha256": artifacts_sha,
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
    try:
        main(args)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"all_passed": False, "scope": "native MoE verification incomplete",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": sys.argv,
            "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise

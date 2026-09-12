"""Compare thirteen ATen model ops plus native composition utilities with Torch.

Python prepares raw fixtures. The child is a C++ executable linked to ATen,
without libtorch_python, a Python interpreter, or per-op Python callbacks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from llaisys.models.deepseek_v4_backends import CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_torch_model


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def raw(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native ATen verification requires a Slurm GPU allocation")
    root = Path(__file__).resolve().parents[2]
    torch_root = Path(torch.__file__).resolve().parent
    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError("this reference validation is pinned to PyTorch 2.13.0+cu130")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError("native ATen build requires the detected CUDA toolkit")
    import tvm_ffi
    ffi_root = Path(tvm_ffi.__file__).resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native_structure_", dir=args.output.parent)).resolve()
    archives = [root / "build/linux/x86_64/release" / f"libllaisys-{name}.a"
                for name in ("core", "device", "device-cpu", "utils")]
    if args.native_library:
        archives.append(args.native_library.resolve().parent / "libllaisys-native-tensor.a")
    sources = [Path(__file__), root / "src/backends/native/tensor.hpp", root / "src/backends/native/tensor.cpp",
               root / "src/backends/native/kernel.hpp", root / "src/backends/aten/structural_kernel.hpp",
               root / "src/backends/aten/structural_kernel.cpp", root / "test/aten_native_standalone.cpp",
               root / "python/llaisys/models/deepseek_v4_backends.py", root / "python/llaisys/models/deepseek_v4_model_ops.py",
               root / "python/llaisys/models/deepseek_v4_reference.py", *archives,
               root / "python/llaisys/libllaisys/libllaisys.so"]
    hadamard_object = args.hadamard_object.resolve(strict=True)
    hadamard_source = args.hadamard_source.resolve(strict=True)
    sources.extend((hadamard_object, root / "src/backends/hadamard/native_kernel.hpp", root / "src/backends/hadamard/native_kernel.cpp"))
    sources.extend(sorted((hadamard_source / "csrc").glob("*.h")))
    sources.append(hadamard_source / "csrc/fast_hadamard_transform_cuda.cu")
    sources.extend((root / "xmake.lua", root / "xmake/aten.lua", root / "xmake/native_tensor.lua"))
    sources.extend(torch_root / "lib" / name for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so", "libc10_cuda.so"))
    if args.native_library:
        sources.append(args.native_library.resolve(strict=True))
    cubins = subprocess.check_output([str(Path(CUDA_HOME) / "bin/cuobjdump"), "--list-elf", str(hadamard_object)], text=True)
    capability = torch.cuda.get_device_capability(0)
    if f"sm_{capability[0]}{capability[1]}" not in cubins:
        raise RuntimeError("Hadamard object has no cubin for the allocated GPU architecture")
    hashes = {str(path): sha(path) for path in sources}
    contracts = {name: contract for name, contract in MODEL_CONTRACTS.items() if name not in CONTRACTS}
    registry = OperatorRegistry(contracts)
    register_torch_model(registry)
    # The six TileLang contracts are not bound/executed in this structural test.
    ops = registry.bind({name: "torch" for name in contracts})
    reference = {name: ops.function(name) for name in contracts}
    torch.manual_seed(731)
    cases = []
    codes = {torch.bfloat16: (4, 16, 1), torch.float32: (2, 32, 1),
             torch.int32: (0, 32, 1), torch.int64: (0, 64, 1), torch.complex64: (5, 64, 1)}

    def add(name, operation, tensors, invoke=None, *, options=None, reject=False):
        owners = [item[0] if isinstance(item, tuple) else item for item in tensors]
        views = [item[1] if isinstance(item, tuple) else item for item in tensors]
        if any(not owner.is_contiguous() or owner.storage_offset() for owner in owners):
            raise ValueError("fixtures must provide complete contiguous storage owners")
        before = [raw(value) for value in owners]
        if not reject:
            invoke(*views)
        torch.cuda.synchronize()
        after = [raw(value) for value in owners]
        specs = []
        for i, (owner, view, initial, expected) in enumerate(zip(owners, views, before, after)):
            inp, out = artifacts / f"case{len(cases)}_arg{i}_in.bin", artifacts / f"case{len(cases)}_arg{i}_out.bin"
            inp.write_bytes(initial); out.write_bytes(expected)
            specs.append({"dtype": codes[owner.dtype], "storage_shape": list(owner.shape), "bytes": len(initial),
                          "shape": list(view.shape), "strides": list(view.stride()), "offset": view.storage_offset(),
                          "input": str(inp), "output": str(out), "input_sha256": hashlib.sha256(initial).hexdigest(),
                          "output_sha256": hashlib.sha256(expected).hexdigest()})
        cases.append({"name": name, "operation": operation, "options": options or {}, "tensors": specs, "expect_rejection": reject})

    def normal(name, operation, inputs, shape, dtype, *scalars, options=None):
        out = torch.zeros(shape, device="cuda", dtype=dtype)
        add(name, operation, [*inputs, out], lambda *values: values[-1].copy_(reference[operation](*values[:-1], *scalars)), options=options)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device="cuda")

    for rows in (1, 33, 137):
        for dtype in (torch.bfloat16, torch.float32):
            suffix = f"m{rows}_{str(dtype).split('.')[-1]}"
            k = 16384 if dtype == torch.float32 else 512
            normal("dense_" + suffix, "dense_linear", [rand((1, rows, k), dtype), rand((24, k), dtype)], (1, rows, 24), dtype)
            normal("grouped_" + suffix, "grouped_linear", [rand((1, rows, 8, 512), dtype), rand((8, 128, 512), dtype)], (1, rows, 8, 128), dtype)
            normal("norm_" + suffix, "rms_norm", [rand((1, rows, 4096), dtype), rand((4096,), torch.float32)],
                   (1, rows, 4096), dtype, 1e-6)
            normal("inverse_rms_" + suffix, "row_inv_rms", [rand((1, rows, 16384 if dtype == torch.float32 else 512), dtype)],
                   (1, rows, 1), dtype, 1e-6)
            for inverse in (False, True):
                base = rand((1, rows, 8, 512), dtype)
                freqs = torch.polar(torch.ones(rows, 32, device="cuda"), rand((rows, 32), torch.float32))
                add(f"rope_{suffix}_inverse{int(inverse)}", "rotary_inplace", [(base, base[..., -64:]), freqs],
                    lambda x, f, inverse=inverse: reference["rotary_inplace"](x, f, inverse), options={"inverse_rotary": inverse})
        normal(f"hc_m{rows}", "hc_sum", [rand((1, rows, 4, 4096), torch.float32)], (1, rows, 4096), torch.float32, 2)
        normal(f"index_scores_m{rows}", "indexer_scores", [rand((1, rows, 64, 128)), rand((1, 65, 128)), rand((1, rows, 64))],
               (1, rows, 65), torch.bfloat16)
        scores = torch.randint(-4, 9, (1, rows, 576), device="cuda").to(torch.bfloat16)
        scores[..., -13:] = -torch.inf
        for stable in (False, True):
            normal(f"top512_m{rows}_stable{int(stable)}", "indexer_topk", [scores.clone()], (1, rows, 512), torch.int64,
                   512, "index_ascending" if stable else "published", options={"stable_indexer_ties": stable})
        for function in ("softmax", "sigmoid", "sqrtsoftplus"):
            for hashed in (False, True):
                logits = rand((rows, 256), torch.float32)
                selector = (torch.rand(rows, 256, device="cuda").topk(6, -1)[1].int() if hashed else rand((256,), torch.float32))
                out_w = torch.zeros(rows, 6, device="cuda")
                out_i = torch.zeros(rows, 6, dtype=selector.dtype if hashed else torch.int64, device="cuda")
                def route(logits, selector, out_w, out_i, function=function, hashed=hashed):
                    weight, ids = reference["router"](logits, None if hashed else selector, selector if hashed else None, 6, function, 1.5)
                    out_w.copy_(weight); out_i.copy_(ids)
                add(f"router_{function}_hash{int(hashed)}_m{rows}", "router", [logits, selector, out_w, out_i], route,
                    options={"score_function": function, "hash_routing": hashed, "route_scale": 1.5})
        for decode in (False, True):
            shape = (1, 8, 512) if decode else (1, rows, 8, 512)
            values, scores = rand(shape, torch.float32), rand(shape, torch.float32)
            scores[..., :4, :] = -torch.inf
            normal(f"pool_decode{int(decode)}_m{rows}", "compressor_pool", [values, scores],
                   (1, 1, 512) if decode else (1, rows, 512), torch.float32)
        for routed in (False, True):
            gate, up = rand((rows, 2048)), rand((rows, 2048))
            inputs = [gate, up] + ([torch.rand(rows, 1, device="cuda")] if routed else [])
            out = torch.zeros_like(gate)
            def activation(*values):
                values[-1].copy_(reference["expert_activation"](values[0], values[1], 10.0, values[2] if len(values) == 4 else None))
            add(f"activation_routed{int(routed)}_m{rows}", "expert_activation", [*inputs, out], activation,
                options={"activation_limit": 10.0})
        x, weights = rand((rows, 512)), torch.rand(rows, 6, device="cuda")
        ids = torch.rand(rows, 16, device="cuda").topk(6, -1)[1]
        packed = reference["moe_dispatch"](x, weights, ids, 256)
        packed_out = [torch.zeros_like(v) for v in (packed.hidden, packed.route_weights, packed.token_indices, packed.expert_offsets)]
        def dispatch(x, w, ids, hidden, rw, indices, offsets):
            result = reference["moe_dispatch"](x, w, ids, 256)
            for out, value in zip((hidden, rw, indices, offsets), (result.hidden, result.route_weights, result.token_indices, result.expert_offsets)):
                out.copy_(value)
        add(f"dispatch_m{rows}_empty_experts", "moe_dispatch", [x, weights, ids, *packed_out], dispatch)
        expert_outputs = rand(packed.hidden.shape)
        out = torch.zeros(rows, 512, device="cuda")
        add(f"combine_m{rows}", "moe_combine", [expert_outputs, packed.token_indices, packed.expert_offsets, out],
            lambda value, ids, offsets, out, packed=packed: out.copy_(reference["moe_combine"](value, packed)))

    normal("empty_index_scores", "indexer_scores", [rand((1, 3, 64, 128)), rand((1, 0, 128)), rand((1, 3, 64))], (1, 3, 0), torch.bfloat16)
    normal("empty_index_topk", "indexer_topk", [rand((1, 3, 0))], (1, 3, 0), torch.int64, 0, "published")
    normal("empty_linear", "dense_linear", [rand((0, 512)), rand((24, 512))], (0, 24), torch.bfloat16)
    add("reject_norm_weight", "rms_norm", [rand((1, 33, 512)), rand((511,), torch.float32), rand((1, 33, 512))], reject=True)
    add("reject_topk_count", "indexer_topk", [rand((1, 3, 4)), torch.zeros(1, 3, 5, dtype=torch.int64, device="cuda")], reject=True)
    add("reject_hash_ids", "router", [rand((2, 8), torch.float32), torch.full((2, 2), 8, dtype=torch.int32, device="cuda"),
                                      torch.zeros(2, 2, device="cuda"), torch.zeros(2, 2, dtype=torch.int32, device="cuda")],
        options={"hash_routing": True}, reject=True)
    add("reject_combine_offsets", "moe_combine", [rand((4, 16)), torch.tensor([0, 1, 0, 1], device="cuda"),
        torch.tensor([0, 3, 2, 4], device="cuda"), torch.zeros(2, 16, device="cuda")], reject=True)

    # Native model composition utilities have explicit contracts independent
    # of the original thirteen Python model contracts.
    for rows in (0, 1, 33, 137):
        for source_dtype, target_dtype in ((torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)):
            value = rand((rows, 4096), source_dtype)
            add(f"cast_m{rows}_{source_dtype}", "tensor_cast", [value, torch.zeros_like(value, dtype=target_dtype)],
                lambda value, out: out.copy_(value.to(out.dtype)))
        table = torch.randint(0, 256, (129, 6), device="cuda", dtype=torch.int32)
        indices = torch.randint(0, 129, (rows,), device="cuda", dtype=torch.int64)
        add(f"gather_m{rows}", "row_gather", [table, indices, torch.zeros(rows, 6, device="cuda", dtype=torch.int32)],
            lambda table, indices, out: out.copy_(table[indices]))
        hash_table = table.long()
        add(f"hash_table_cast_m{rows}", "tensor_cast", [hash_table, torch.zeros_like(table)],
            lambda value, out: out.copy_(value.int()))
        routed, shared = rand((rows, 4096), torch.float32), rand((rows, 4096))
        add(f"finalize_m{rows}", "moe_finalize", [routed, shared, torch.zeros_like(shared)],
            lambda routed, shared, out: out.copy_((routed + shared).bfloat16()))
    add("reject_hash_duplicate", "router", [rand((2, 8), torch.float32), torch.zeros(2, 2, dtype=torch.int32, device="cuda"),
        torch.zeros(2, 2, dtype=torch.float32, device="cuda"), torch.zeros(2, 2, dtype=torch.int32, device="cuda")],
        options={"hash_routing": True}, reject=True)
    add("reject_gather_range", "row_gather", [rand((4, 16)), torch.tensor([-1, 4], device="cuda"), rand((2, 16))], reject=True)
    add("reject_cast_integer", "tensor_cast", [torch.zeros(2, 8, dtype=torch.int32, device="cuda"), rand((2, 8))], reject=True)
    add("reject_cast_integer_overflow", "tensor_cast", [torch.full((2, 8), 2**32, dtype=torch.int64, device="cuda"),
        torch.zeros(2, 8, dtype=torch.int32, device="cuda")], reject=True)
    add("reject_finalize_dtype", "moe_finalize", [rand((2, 16)), rand((2, 16)), rand((2, 16))], reject=True)

    import fast_hadamard_transform as hadamard
    hadamard.configure_backend("cuda")
    for rows in (1, 33, 137):
        for dtype in (torch.bfloat16, torch.float32):
            for dim in (128, 512):
                value = rand((1, rows, 8, dim), dtype)
                scale = dim ** -0.5
                add(f"hadamard_m{rows}_d{dim}_{str(dtype).split('.')[-1]}", "hadamard", [value, torch.zeros_like(value)],
                    lambda value, output, scale=scale: output.copy_(hadamard.hadamard_transform(value, scale)),
                    options={"hadamard_scale": scale})
    add("reject_hadamard_nonpower2", "hadamard", [rand((1, 96)), torch.zeros(1, 96, dtype=torch.bfloat16, device="cuda")], reject=True)

    manifest = artifacts / "manifest.txt"
    version = torch.__version__.split("+")[0]
    hadamard_version = sha(hadamard_source / "csrc/fast_hadamard_transform_cuda.cu")
    lines = [f"LLAISYS_ATEN_NATIVE_TEST_V1 {len(cases)} {version} {hadamard_version}"]
    for case in cases:
        opt = case["options"]
        fields = [case["name"], case["operation"], opt.get("epsilon", 1e-6), opt.get("activation_limit", 0),
                  opt.get("route_scale", 1), opt.get("reduction_axis", 2), int(opt.get("inverse_rotary", False)),
                  int(opt.get("stable_indexer_ties", False)), int(opt.get("hash_routing", False)),
                  opt.get("score_function", "sqrtsoftplus"), opt.get("hadamard_scale", 1), len(case["tensors"]), int(case["expect_rejection"])]
        lines.append(" ".join(map(str, fields)))
        for tensor in case["tensors"]:
            fields = [*tensor["dtype"], len(tensor["storage_shape"]), *tensor["storage_shape"], tensor["bytes"]]
            lines.append(" ".join(map(str, fields)) + " " + json.dumps(tensor["input"]) + " " + json.dumps(tensor["output"]))
            lines.append(" ".join(map(str, [len(tensor["shape"]), *tensor["shape"], *tensor["strides"], tensor["offset"]])))
    manifest.write_text("\n".join(lines) + "\n")
    executable = artifacts / "aten-native-standalone"
    backend_inputs = ([str(args.native_library.resolve())] if args.native_library else
                      [str(root / path) for path in ("src/backends/native/tensor.cpp", "src/backends/aten/structural_kernel.cpp", "src/backends/hadamard/native_kernel.cpp")]
                      + [str(hadamard_object)])
    command = [shutil.which("g++"), "-std=c++20", "-O2", "-pthread", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               *["-I" + str(path) for path in (root, root / "include", ffi_root / "include", torch_root / "include",
                                               torch_root / "include/torch/csrc/api/include", Path(CUDA_HOME) / "include")],
               "-I" + str(hadamard_source / "csrc"),
               str(root / "test/aten_native_standalone.cpp"), *backend_inputs,
               "-Wl,--start-group", *map(str, archives), "-Wl,--end-group", "-L" + str(root / "python/llaisys/libllaisys"),
               "-L" + str(torch_root / "lib"), "-L" + str(Path(CUDA_HOME) / "lib64"), "-Wl,--no-as-needed", "-lllaisys", "-ltorch_cuda", "-ltorch_cpu", "-lc10_cuda", "-lc10", "-lcudart", "-ldl",
               "-Wl,-rpath," + str(torch_root / "lib"), "-Wl,-rpath," + str(root / "python/llaisys/libllaisys"), "-o", str(executable)]
    print("compile:", json.dumps(command), flush=True)
    subprocess.run(command, check=True)
    nvidia_root = torch_root.parent / "nvidia"
    libdirs = [torch_root / "lib", nvidia_root / "cu13/lib", *sorted(nvidia_root.glob("*/lib")), root / "python/llaisys/libllaisys"]
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(map(str, libdirs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    run = subprocess.run([str(executable), str(manifest)], capture_output=True, text=True, env=env)
    print(run.stdout, end="", flush=True); print(run.stderr, end="", file=sys.stderr, flush=True)
    result = json.loads(run.stdout.splitlines()[-1]) if run.returncode == 0 else {"all_passed": False}
    unchanged = hashes == {str(path): sha(path) for path in sources}
    policy_matches = result.get("allow_tf32_cublas") == torch.backends.cuda.matmul.allow_tf32
    coverage = all(entry["calls"] > 0 and not entry["failures"] for entry in ops.report().values())
    report = {"scope": "thirteen ATen structural ops, three composition utilities and CUDA Hadamard; not full C++ model", "result": result,
              "all_passed": run.returncode == 0 and result["all_passed"] and unchanged and policy_matches and coverage,
              "source_unchanged": unchanged, "reference_precision_policy_matches": policy_matches,
              "all_thirteen_reference_operators_exercised": coverage,
              "composition_utilities": ["tensor_cast", "row_gather", "moe_finalize"],
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0),
              "compute_capability": torch.cuda.get_device_capability(0), "torch": torch.__version__, "cuda": torch.version.cuda,
              "reference_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cases": cases,
              "reference_operators": ops.report(),
              "reference_hadamard": hadamard.backend_report(), "hadamard_source": str(hadamard_source), "hadamard_object": str(hadamard_object),
              "hadamard_cubins": cubins, "xmake_native_library": str(args.native_library.resolve()) if args.native_library else None,
              "native_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
              "dependencies": subprocess.check_output(["ldd", str(executable)], env=env, text=True),
              "artifact_sha256": {str(path): sha(path) for path in (executable, manifest)},
              "provenance": {"command": sys.argv, "build_command": command, "file_sha256": hashes,
                             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                             "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(args.output)}), flush=True)
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hadamard-source", type=Path, required=True)
    parser.add_argument("--hadamard-object", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, help="actual xmake-built native ATen/Hadamard archive")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"all_passed": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                          "error": f"{type(error).__name__}: {error}", "command": sys.argv}, indent=2) + "\n")
        raise

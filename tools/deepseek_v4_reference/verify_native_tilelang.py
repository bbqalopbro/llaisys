"""Export unchanged upstream TileLang kernels, then test in a native process.

Python is used to prepare test vectors, JIT and export. The C++ child executable
uses existing llaisys Runtime/Storage and TVM-FFI, with no Python/Torch runtime.
This is native-backend acceptance, not complete model or serving acceptance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import torch

from run_independent import load_kernel, sha256


def raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="actual MP1 checkpoint; read only selected Linear weights")
    parser.add_argument("--native-library", type=Path,
                        help="link the native backend archive built by xmake instead of compiling it inline")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native GPU kernel verification requires a Slurm allocation")
    import tilelang
    import tvm_ffi
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError("native storage verification requires a detected CUDA toolkit")
    cuda_root = Path(CUDA_HOME)
    if tilelang.__version__ != "0.1.8" or tvm_ffi.__version__ != "0.1.8.post2":
        raise RuntimeError("native ABI validation requires TileLang 0.1.8 and TVM-FFI 0.1.8.post2")
    root = Path(__file__).resolve().parents[2]
    ffi_root, tl_root = Path(tvm_ffi.__file__).parent, Path(tilelang.__file__).parent
    # Runtime/Storage are internal C++ symbols, not part of libllaisys's public
    # C ABI. Reuse their existing build archives; do not duplicate/reimplement
    # allocators or widen the shared library's symbol visibility for a test.
    archives = [root / "build/linux/x86_64/release" / f"libllaisys-{name}.a"
                for name in ("core", "device", "device-cpu", "utils")]
    if args.native_library:
        archives.append(args.native_library.resolve().parent / "libllaisys-native-tensor.a")
        archives.append(args.native_library.resolve().parent / "libllaisys-deepseek-v4-native.a")
    source = args.source_model.resolve() / "inference/kernel.py"
    kernel = load_kernel(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix="native_tilelang_", dir=args.output.parent)).resolve()
    paths = [Path(__file__), source, root / "src/backends/tilelang/native_kernel.hpp",
             root / "src/backends/tilelang/native_kernel.cpp", root / "test/tilelang_native_standalone.cpp",
             root / "python/llaisys/libllaisys/libllaisys.so", ffi_root / "lib/libtvm_ffi.so", tl_root / "lib/libtvm.so",
             tl_root / "lib/libtilelang.so",
             *archives]
    paths.extend(sorted((root / "src/backends/native").glob("*.[ch]pp")))
    paths.extend(root / "src/models/deepseek_v4" / filename for filename in ("linear.hpp", "linear.cpp"))
    paths.append(root / "src/models/deepseek_v4/cache_layout.hpp")
    paths.extend(sorted((root / "src/core/cache").glob("*.[ch]pp")))
    paths.extend((root / "xmake.lua", root / "xmake/tilelang.lua", root / "xmake/native_tensor.lua", root / "xmake/native_model.lua", Path(load_kernel.__code__.co_filename),
                  root / "python/llaisys/models/deepseek_v4_model/weights.py"))
    if args.native_library:
        paths.append(args.native_library.resolve(strict=True))
    hashes = {str(path): sha256(path) for path in paths}
    torch.manual_seed(731)
    entries, linears, exports, export_origins = [], [], {}, {}
    # Tensor inputs to every compiled primitive are preallocated explicitly,
    # including outputs. Sub-byte shapes use the primitive's LOGICAL elements.
    def export_kernel(compiled):
        key = str(compiled.prim_func.script())
        if key not in exports:
            library = artifact_dir / f"kernel_{len(exports)}.so"
            # Disk-cache hits are already exported runtime.Module libraries;
            # unlike freshly compiled Executables they cannot SaveToBytes.
            # Preserve the exact cached artifact, never recompile/fallback.
            cached = getattr(compiled.adapter, "libpath", None)
            if cached:
                cached = Path(cached).resolve(strict=True)
                shutil.copyfile(cached, library)
                if sha256(cached) != sha256(library):
                    raise RuntimeError("cached TileLang artifact copy changed bytes")
                export_origins[str(library)] = {"mode": "cached_library_copy", "source": str(cached)}
            else:
                compiled.adapter.executable.export_library(str(library))
                export_origins[str(library)] = {"mode": "fresh_executable_export"}
            exports[key] = library
        return exports[key]

    def tensor_spec(prefix, i, value, dtype, initial, expected):
        shape = list(value.shape)
        code, bits, lanes = dtype
        if bits < 8:
            shape[-1] = shape[-1] * value.element_size() * 8 // (bits * lanes)
        input_path = artifact_dir / f"{prefix}_arg{i}_in.bin"
        output_path = artifact_dir / f"{prefix}_arg{i}_out.bin"
        input_path.write_bytes(initial)
        output_path.write_bytes(expected)
        return {"dtype": list(dtype), "shape": shape, "bytes": len(initial),
                "input": str(input_path), "output": str(output_path),
                "input_sha256": hashlib.sha256(initial).hexdigest(),
                "output_sha256": hashlib.sha256(expected).hexdigest()}

    def add(name, compiled, values):
        index = len(entries)
        library = export_kernel(compiled)
        before = [raw(value) for value in values]
        compiled(*values)
        torch.cuda.synchronize()
        after = [raw(value) for value in values]
        prim = compiled.prim_func
        specs = []
        for i, (param, value, initial, expected) in enumerate(zip(prim.params, values, before, after)):
            buf = prim.buffer_map[param]
            dtype = buf.dtype
            specs.append(tensor_spec(f"case{index}", i, value, (dtype.type_code, dtype.bits, dtype.lanes), initial, expected))
        entries.append({"name": name, "library": str(library), "tensors": specs})

    for rows in (1, 33, 137):
        x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        for qdq in (False, True):
            quant = kernel.act_quant_kernel(512, 128, out_dtype=kernel.BF16 if qdq else kernel.FP8,
                                           scale_dtype=kernel.FE8M0, round_scale=True, inplace=qdq)
            add(f"act_quant_m{rows}_qdq{int(qdq)}", quant,
                [x.clone(), torch.zeros_like(x) if qdq else torch.zeros_like(x, dtype=torch.float8_e4m3fn),
                 torch.zeros(rows, 4, dtype=torch.float8_e8m0fnu, device="cuda")])
            quant4 = kernel.fp4_quant_kernel(512, 32, inplace=qdq)
            add(f"fp4_quant_m{rows}_qdq{int(qdq)}", quant4,
                [x.clone(), torch.zeros_like(x) if qdq else torch.zeros(rows, 256, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2),
                 torch.zeros(rows, 16, dtype=torch.float8_e8m0fnu, device="cuda")])
        for mode in ("fp8", "fp4"):
            a = torch.randn(rows, 512, device="cuda").to(torch.float8_e4m3fn)
            b = (torch.randn(256, 512, device="cuda").to(torch.float8_e4m3fn) if mode == "fp8" else
                 torch.randint(0, 256, (256, 256), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2))
            scale_a = torch.ones(rows, 4, device="cuda").to(torch.float8_e8m0fnu)
            scale_b = torch.ones(2, 4, device="cuda").to(torch.float8_e8m0fnu) if mode == "fp8" else torch.ones(256, 16, device="cuda").to(torch.float8_e8m0fnu)
            add(f"{mode}_gemm_m{rows}", getattr(kernel, mode + "_gemm_kernel")(256, 512, scale_dtype=kernel.FE8M0),
                [a, b, torch.zeros(rows, 256, dtype=torch.bfloat16, device="cuda"), scale_a, scale_b])
        hc = 4
        add(f"hc_sinkhorn_m{rows}", kernel.hc_split_sinkhorn_kernel(hc, 20, 1e-6),
            [torch.randn(rows, 24, device="cuda"), torch.randn(3, device="cuda"), torch.randn(24, device="cuda"),
             torch.zeros(rows, 4, device="cuda"), torch.zeros(rows, 4, device="cuda"), torch.zeros(rows, 4, 4, device="cuda")])
    for seq, candidates in ((1, 64), (3, 576)):
        q = torch.randn(1, seq, 64, 512, device="cuda", dtype=torch.bfloat16)
        latent = torch.randn(1, 640, 512, device="cuda", dtype=torch.bfloat16)
        indices = torch.randint(0, 640, (1, seq, candidates), dtype=torch.int32, device="cuda")
        indices[:, :, -13:] = -1
        add(f"sparse_attention_s{seq}_k{candidates}", kernel.sparse_attn_kernel(64, 512, 512 ** -0.5),
            [q, latent, torch.zeros_like(q), torch.randn(64, device="cuda"), indices])
        # The C++ process uploads these fixtures directly into the existing
        # core PagedCacheStorage's interleaved component. Same kernel/indices,
        # including masked -1 and nonmonotonic physical slots; no gather buffer.
        # This validates storage/ABI, not logical page-table mapping or a model.
        for ratio in (0, 4, 128):
            slots = 128 + (128 // ratio if ratio else 0)
            length = 4 * slots
            paged_latent = torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16)
            paged_indices = torch.randint(0, length, (1, seq, candidates), dtype=torch.int32, device="cuda")
            paged_indices[:, :, -13:] = -1
            add(f"sparse_attention_paged_r{ratio}_s{seq}_k{candidates}", kernel.sparse_attn_kernel(64, 512, 512 ** -0.5),
                [q.clone(), paged_latent, torch.zeros_like(q), torch.randn(64, device="cuda"), paged_indices])

    from safetensors import safe_open
    from llaisys.models.deepseek_v4_model.weights import read_header
    header, header_hash = read_header(args.checkpoint)
    selected = []
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        for prefix in ("layers.0.attn.wq_a", "layers.0.ffn.experts.0.w1", "layers.0.ffn.experts.0.w2"):
            weight_key, scale_key = prefix + ".weight", prefix + ".scale"
            weight = checkpoint.get_tensor(weight_key).cuda()
            weight_scale = checkpoint.get_tensor(scale_key).cuda()
            packed = weight.dtype == torch.float4_e2m1fn_x2
            if weight.dtype not in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
                raise ValueError(f"unexpected actual weight dtype: {weight_key}")
            n, k = weight.shape[0], weight.shape[1] * (2 if packed else 1)
            selected.append({"weight_key": weight_key, "scale_key": scale_key,
                             "weight_header": header[weight_key], "scale_header": header[scale_key],
                             "weight_payload_sha256": hashlib.sha256(raw(weight)).hexdigest(),
                             "scale_payload_sha256": hashlib.sha256(raw(weight_scale)).hexdigest()})
            quant = kernel.act_quant_kernel(k, 128, scale_dtype=kernel.FE8M0, round_scale=True)
            gemm = getattr(kernel, "fp4_gemm_kernel" if packed else "fp8_gemm_kernel")(n, k, scale_dtype=kernel.FE8M0)
            quant_lib, gemm_lib = export_kernel(quant), export_kernel(gemm)
            for rows in (1, 33, 137):
                x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
                a = torch.zeros(rows, k, dtype=torch.float8_e4m3fn, device="cuda")
                a_s = torch.zeros(rows, k // 128, dtype=torch.float8_e8m0fnu, device="cuda")
                out = torch.zeros(rows, n, dtype=torch.bfloat16, device="cuda")
                values = [x, weight, weight_scale, out, a, a_s]
                before = [raw(value) for value in values]
                quant(x, a, a_s)
                gemm(a, weight, out, a_s, weight_scale)
                torch.cuda.synchronize()
                after = [raw(value) for value in values]
                dtypes = [(4, 16, 1), (17, 4, 1) if packed else (10, 8, 1), (14, 8, 1),
                          (4, 16, 1), (10, 8, 1), (14, 8, 1)]
                specs = [tensor_spec(f"linear{len(linears)}", i, value, dtype, initial, expected)
                         for i, (value, dtype, initial, expected) in enumerate(zip(values, dtypes, before, after))]
                linears.append({"name": f"{prefix}_m{rows}", "quant_library": str(quant_lib),
                                "gemm_library": str(gemm_lib), "tensors": specs})
    manifest = artifact_dir / "manifest.txt"
    lines = [f"LLAISYS_TILELANG_NATIVE_TEST_V2 {len(entries)} {tilelang.__version__}"]
    def append_tensors(tensors):
        for tensor in tensors:
            numbers = [*tensor["dtype"], len(tensor["shape"]), *tensor["shape"], tensor["bytes"]]
            lines.append(" ".join(map(str, numbers)) + " " + json.dumps(tensor["input"]) + " " + json.dumps(tensor["output"]))
    for entry in entries:
        lines.append(f"{entry['name']} {json.dumps(entry['library'])} {len(entry['tensors'])}")
        append_tensors(entry["tensors"])
    lines.append(f"LINEARS {len(linears)}")
    for entry in linears:
        lines.append(f"{entry['name']} {json.dumps(entry['quant_library'])} {json.dumps(entry['gemm_library'])}")
        append_tensors(entry["tensors"])
    manifest.write_text("\n".join(lines) + "\n")
    executable = artifact_dir / "tilelang-native-standalone"
    native_lib = root / "python/llaisys/libllaisys"
    backend_inputs = ([str(args.native_library.resolve())] if args.native_library else
                      [str(root / name) for name in ("src/backends/native/tensor.cpp", "src/backends/native/paged_storage.cpp", "src/backends/tilelang/native_kernel.cpp",
                                                    "src/models/deepseek_v4/linear.cpp")])
    command = [shutil.which("g++"), "-std=c++17", "-O2", "-pthread", "-I" + str(root), "-I" + str(root / "include"),
               "-I" + str(ffi_root / "include"), "-I" + str(cuda_root / "include"),
               str(root / "test/tilelang_native_standalone.cpp"), "-L" + str(native_lib), "-L" + str(ffi_root / "lib"),
               "-L" + str(tl_root / "lib"), *backend_inputs,
               "-Wl,--start-group", *map(str, archives), "-Wl,--end-group",
               "-Wl,--no-as-needed", "-lllaisys", "-ltilelang", "-ltvm", "-ltvm_ffi", "-ldl", "-L" + str(cuda_root / "lib64"), "-lcudart",
               "-Wl,-rpath," + str(native_lib), "-Wl,-rpath," + str(ffi_root / "lib"), "-Wl,-rpath," + str(tl_root / "lib"),
               "-o", str(executable)]
    print("compile:", shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([str(ffi_root / "lib"), str(tl_root / "lib"), str(native_lib), env.get("LD_LIBRARY_PATH", "")])
    run = subprocess.run([str(executable), str(manifest)], env=env, capture_output=True, text=True)
    print(run.stdout, end="", flush=True)
    print(run.stderr, end="", file=sys.stderr, flush=True)
    result = json.loads(run.stdout.splitlines()[-1]) if run.returncode == 0 else {"all_passed": False}
    artifacts = {str(path): sha256(path) for path in (executable, manifest, *exports.values())}
    unchanged = hashes == {str(path): sha256(path) for path in paths}
    report = {"scope": "pure C++ TileLang kernel backend, not model runtime completion", "result": result,
              "all_passed": run.returncode == 0 and result["all_passed"] and unchanged,
              "source_unchanged": unchanged, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
              "visible_gpu_count": torch.cuda.device_count(),
              "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
              "host_compiler": subprocess.check_output([command[0], "--version"], text=True).splitlines()[0],
              "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__, "torch": torch.__version__,
              "cuda": torch.version.cuda, "kernel_specializations": len(exports), "cases": entries,
              "linear_cases": linears,
              "checkpoint": {"path": str(args.checkpoint.resolve()), "header_sha256": header_hash,
                             "size_bytes": args.checkpoint.stat().st_size, "selected_tensors": selected,
                             "whole_model_loaded": False, "whole_payload_hashed": False},
              "export_origins": export_origins,
              "native_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
              "xmake_native_library": str(args.native_library.resolve()) if args.native_library else None,
              "dependencies": subprocess.check_output(["ldd", str(executable)], env=env, text=True),
              "artifact_sha256": artifacts,
              "provenance": {"command": sys.argv, "build_command": command, "file_sha256": hashes,
                             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                             "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(args.output)}), flush=True)
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as error:
        # Failed preparation/compilation must not leave an old green report at
        # the selected output path. Native execution failures already produce
        # the detailed report above and exit nonzero via SystemExit.
        failure = {"scope": "native TileLang verification failed before a complete result",
                   "all_passed": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "command": sys.argv, "error": f"{type(error).__name__}: {error}"}
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, indent=2) + "\n")
        except OSError as report_error:
            print(f"cannot write failure report: {report_error}", file=sys.stderr)
        raise

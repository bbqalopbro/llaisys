"""Check the standalone W4A8 host ABI and optionally collect nvcc resource evidence.

Runs with Python's standard library, without Torch or a GPU allocation. Every
launch test supplies a contract-invalid argument rejected before cudaGetDevice;
this checks host validation, not device correctness or GPU capability rejection.

Reproduce compilation and all host checks (choose an unused output library)::

    python3 tools/deepseek_v4_cuda/test_w4a8_abi.py --compile \
      --library /tmp/w4a8_resource_check.so \
      --output benchmark_results/deepseek_v4_cuda/w4a8_decode_sm103_abi.json \
      --resource-log benchmark_results/deepseek_v4_cuda/w4a8_decode_sm103_compile.log

The JSON records the exact nvcc argv, compiler version, source/library/log SHA256,
and per-specialization registers, stack and spill bytes. The text log preserves
the original ptxas output. A successful host check is not model acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource_entries(log):
    entries = []
    current = None
    for line in log.splitlines():
        kernel = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line)
        if kernel:
            current = dict(symbol=kernel[1], architecture=kernel[2])
            shape = re.search(r"decode(_split_k)?ILi(\d+)ELi(\d+)E", kernel[1])
            if shape:
                current.update(m=int(shape[2]), warps=int(shape[3]), split_k=bool(shape[1]))
            entries.append(current)
        if current is None:
            continue
        memory = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if memory:
            current.update(stack_bytes=int(memory[1]), spill_store_bytes=int(memory[2]),
                           spill_load_bytes=int(memory[3]))
        registers = re.search(r"Used (\d+) registers, used (\d+) barriers", line)
        if registers:
            current.update(registers=int(registers[1]), barriers=int(registers[2]))
    return entries


def abi_checks(library):
    loaded = ctypes.CDLL(str(library))
    launch = loaded.llaisys_w4a8_decode_sm103
    launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
    launch.restype = ctypes.c_int
    version = loaded.llaisys_w4a8_decode_sm103_version
    version.argtypes = []
    version.restype = ctypes.c_char_p
    identity = version().decode("ascii")
    if identity != "llaisys-w4a8-sm103-experimental-v2":
        raise RuntimeError(f"unexpected ABI version: {identity}")
    # Sentinel addresses are never dereferenced: each case fails the public
    # argument contract before the CUDA device query or launch.
    baseline = [0x10000, 0x20000, 0x30000, 0x40000, 0x50000,
                1, 1, 128, 128, 1, None, 0]
    cases = [
        (0, None, "null_activation"),
        (5, 0, "zero_rows"), (5, 9, "rows_above_decode_limit"),
        (6, 0, "zero_output_columns"), (6, 65537, "columns_above_supported_limit"),
        (7, 96, "incomplete_quantization_group"), (7, 32896, "k_above_supported_limit"),
        (8, 129, "unaligned_activation_row_stride"),
        (8, 112, "overlapping_activation_rows"),
        (9, 0, "short_scale_row_stride"), (9, -1, "negative_scale_row_stride"),
        (11, 4, "unknown_variant"), (11, -1, "negative_variant"),
        (4, 0x10000, "output_aliases_activation"),
        (4, 0x20000, "output_aliases_activation_scale"),
        (4, 0x30000, "output_aliases_weight"),
        (4, 0x40000, "output_aliases_weight_scale"),
        (8, 2**62, "activation_stride_address_overflow"),
        (9, 2**62, "scale_stride_address_overflow"),
        (0, 0x10001, "unaligned_activation_address"),
        (2, 0x30001, "unaligned_weight_address"),
        (4, 0x50001, "unaligned_output_address"),
    ]
    results = []
    for argument, value, name in cases:
        args = baseline.copy()
        args[argument] = value
        actual = launch(*args)
        results.append(dict(name=name, cuda_error=actual, expected_cuda_error=1, passed=actual == 1))
    return identity, results


def product_exactness():
    """Exhaust the finite E4M3 x E2M1 domain without CUDA or Torch."""
    weights = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
    count = 0
    failures = []
    for bits in range(256):
        exponent, mantissa = (bits >> 3) & 15, bits & 7
        if exponent == 15 and mantissa == 7:  # E4M3FN's two signed NaN encodings.
            continue
        activation = ((1 + mantissa / 8) * 2.0 ** (exponent - 7)
                      if exponent else mantissa * 2.0 ** -9)
        if bits & 128:
            activation = -activation
        for code, weight in enumerate(weights):
            exact = activation * weight
            rounded_to_half = struct.unpack("e", struct.pack("e", exact))[0]
            count += 1
            if rounded_to_half != exact:
                failures.append(dict(activation_bits=bits, weight_code=code))
    return dict(finite_input_pairs=count, passed=count == 4064 and not failures,
                failures=failures, scope="finite unscaled products only; all accumulation remains FP32")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--resource-log", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    library = args.library.resolve()
    source = root / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.cu"
    header = source.with_suffix(".h")
    report = dict(all_passed=False, scope="host ABI argument rejection and optional compiler resource audit",
                  gpu_execution=False, positive_shape_launches=0,
                  source_sha256={str(p): sha256(p) for p in (source, header, Path(__file__).resolve())},
                  command=sys.argv, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.compile:
            if not args.nvcc:
                raise RuntimeError("nvcc was not found")
            resource_log = (args.resource_log or args.output.with_suffix(".compile.log")).resolve()
            resource_log.parent.mkdir(parents=True, exist_ok=True)
            library.parent.mkdir(parents=True, exist_ok=True)
            command = [args.nvcc, "-O3", "-std=c++17", "-arch=sm_103a", "-DLLAISYS_B300_STANDALONE", "--shared",
                       "-Xcompiler=-fPIC", "-lineinfo", "--ptxas-options=-v", str(source), "-o", str(library)]
            before = time.perf_counter()
            result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            resource_log.write_text(result.stdout)
            entries = resource_entries(result.stdout)
            expected = {(m, warps, False) for m in range(1, 9) for warps in (4, 8)}
            expected.update((m, 4, True) for m in range(1, 9))
            observed = {(e.get("m"), e.get("warps"), e.get("split_k")) for e in entries}
            no_spills = (len(entries) == 24 and observed == expected and all(
                e.get("stack_bytes") == e.get("spill_store_bytes") == e.get("spill_load_bytes") == 0
                for e in entries))
            report["compilation"] = dict(command=command, exit_code=result.returncode,
                elapsed_seconds=time.perf_counter() - before,
                nvcc_version=subprocess.check_output([args.nvcc, "--version"], text=True),
                resource_log=str(resource_log), resource_log_sha256=sha256(resource_log),
                kernels=entries, all_24_specializations_zero_stack_and_spills=no_spills)
            if result.returncode:
                raise RuntimeError(f"nvcc failed with exit code {result.returncode}; see {resource_log}")
            if not no_spills:
                raise RuntimeError("expected 24 complete zero-stack, zero-spill ptxas resource records")
        report["library"] = str(library)
        report["library_sha256"] = sha256(library)
        report["version"], report["abi_cases"] = abi_checks(library)
        report["packed_half_product_exactness"] = product_exactness()
        report["all_passed"] = (all(case["passed"] for case in report["abi_cases"])
                                and report["packed_half_product_exactness"]["passed"])
        if report["source_sha256"] != {p: sha256(p) for p in report["source_sha256"]}:
            raise RuntimeError("source changed during host ABI/compiler audit")
    except Exception as error:
        report["all_passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(all_passed=report["all_passed"], output=str(args.output),
                         abi_cases=len(report.get("abi_cases", [])))))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

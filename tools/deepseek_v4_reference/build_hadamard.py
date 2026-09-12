"""Build the upstream Hadamard extension for the allocated GPU, without sudo."""

import argparse
import json
from pathlib import Path
import subprocess

import torch
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--python-include", action="append", default=[])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this build inside a GPU Slurm allocation")
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    args.build_dir.mkdir(parents=True, exist_ok=True)
    module = load(
        name="fast_hadamard_transform_cuda",
        sources=[str(args.source / "csrc" / name) for name in
                 ("fast_hadamard_transform.cpp", "fast_hadamard_transform_cuda.cu")],
        extra_include_paths=[str(args.source), *args.python_include],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
            "-lineinfo", f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ],
        build_directory=str(args.build_dir), verbose=True,
    )
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(), "cuda_arch": f"sm_{arch}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "extension": module.__file__,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2))


if __name__ == "__main__":
    main()

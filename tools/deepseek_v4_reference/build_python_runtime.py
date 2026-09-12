"""Build the existing pybind runtime with the allocated GPU and current Python.

No model inference is executed. This restores the existing execution boundary;
it does not claim a DeepSeek C++ runner is implemented.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmake", default=shutil.which("xmake"))
    parser.add_argument("--python-include", type=Path, default=Path(sysconfig.get_path("include")))
    parser.add_argument("--python-system-include", type=Path,
                        help="additional multiarch include root, for unpacked Python development headers")
    parser.add_argument("--pybind11-include", type=Path, default=Path(torch.__file__).parent / "include")
    parser.add_argument("--dlpack-include", type=Path,
                        help="enable native V4 cache views using this DLPack include root")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("this build must run inside an allocated Slurm GPU job")
    if not args.xmake or not Path(args.xmake).is_file() or args.jobs <= 0:
        raise ValueError("an existing xmake executable and positive build job count are required")
    if not (args.python_include / "Python.h").is_file():
        raise ValueError("Python.h is missing from the selected include directory")
    if not (args.pybind11_include / "pybind11/pybind11.h").is_file():
        raise ValueError("pybind11 headers are missing from the selected include directory")
    if args.dlpack_include and not (args.dlpack_include / "dlpack/dlpack.h").is_file():
        raise ValueError("dlpack/dlpack.h is missing from the selected include directory")
    root = Path(__file__).resolve().parents[2]
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    build_env = dict(os.environ)
    extra_include = None
    if args.python_system_include:
        extra_include = str(args.python_system_include.resolve())
        build_env["CPLUS_INCLUDE_PATH"] = os.pathsep.join(filter(None, (
            extra_include, build_env.get("CPLUS_INCLUDE_PATH", ""))))
    commands = [
        [args.xmake, "f", "-y", "--nv-gpu=y", "--flashinfer=y", f"--cuda-arch={architecture}",
         "--python-bindings=y", f"--python-include={args.python_include.resolve()}",
         f"--pybind11-include={args.pybind11_include.resolve()}"],
        [args.xmake, "build", "-j", str(args.jobs), "llaisys-python"],
        [args.xmake, "install", "-y", "llaisys"],
    ]
    if args.dlpack_include:
        commands[0].append(f"--dlpack-include={args.dlpack_include.resolve()}")
        commands.extend(([args.xmake, "build", "llaisys-cache-storage-test"],
                         [args.xmake, "run", "llaisys-cache-storage-test"]))
    report = {"scope": "existing_pybind_build_and_schedule_contract_smoke_no_model_inference",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0),
              "compute_capability": capability, "cuda_arch": architecture, "torch": torch.__version__,
              "cuda": torch.version.cuda, "python": sys.version, "python_executable": sys.executable,
              "extra_python_system_include": extra_include, "commands": commands, "success": False}
    try:
        for command in commands:
            print(json.dumps(command), flush=True)
            subprocess.run(command, cwd=root, env=build_env, check=True)
        sys.path.insert(0, str(root / "python"))
        native = importlib.import_module("llaisys._C")
        plan, prefill, decode = native.SchedulePlan(), native.PrefillItem(), native.DecodeItem()
        plan.step_id, plan.reset_slots = 17, [0]
        prefill.request_id, prefill.slot_id, prefill.token_ids = 101, 0, [1, 2, 3]
        prefill.start_pos, prefill.is_last_chunk = 0, True
        decode.request_id, decode.slot_id, decode.token_id = 102, 1, 4
        plan.prefills, plan.decodes = [prefill], [decode]
        assert plan.step_id == 17 and plan.reset_slots == [0]
        assert plan.prefills[0].token_ids == [1, 2, 3] and plan.decodes[0].token_id == 4
        assert plan.prefills[0].request_id == 101 and plan.decodes[0].request_id == 102
        if args.dlpack_include:
            assert native.v4_native_storage_available
            storage = native.V4PagedStorage(2, 128, 128, 128, [0, 4, 128])
            tensor = torch.utils.dlpack.from_dlpack(storage.view(1))
            assert tuple(tensor.shape) == (1, 320, 128) and tensor.dtype == torch.bfloat16
            tensor.fill_(1)
            del tensor, storage
        module_path = Path(native.__file__).resolve()
        report.update(success=True, module=str(module_path),
                      module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
                      schedule_plan_roundtrip=True, model_inference_tested=False)
        report["native_v4_storage_available"] = native.v4_native_storage_available
        report["native_cache_lifecycle_test_passed"] = bool(args.dlpack_include)
        library = root / "python/llaisys/libllaisys/libllaisys.so"
        report["runtime_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__).resolve(), *sorted((root / "python/bindings").glob("*.cpp")),
            *sorted((root / "src/core/cache").glob("*.[ch]pp")), root / "xmake.lua", root / "xmake/nvidia.lua")}
        for path in sorted((root / "src/models/deepseek_v4").glob("*.[ch]pp")):
            report["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        test_source = root / "test/paged_storage_lifecycle_test.cpp"
        report["source_sha256"][str(test_source)] = hashlib.sha256(test_source.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

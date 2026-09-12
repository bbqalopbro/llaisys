"""Run the V4 regression with source identity and honest method/subtest counts."""

import argparse
from contextlib import redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import torch


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_deepseek_v4*.py")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--include-scheduler", action="store_true", help="also run existing model-free scheduler tests")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise RuntimeError("CUDA regression requires an allocated Slurm GPU")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), *sorted((root / "test").glob(args.pattern)),
             *sorted((root / "tools/deepseek_v4_reference").glob("*.py")),
             *sorted((root / "python/llaisys/models").glob("deepseek_v4*.py")),
             *sorted((root / "python/llaisys/models/deepseek_v4_model").glob("*.py")),
             *sorted((root / "python/bindings").glob("*.cpp")),
             *sorted((root / "src/core/cache").glob("*.[ch]pp")), root / "python/llaisys/_C.so"]
    paths += sorted((root / "src/models/deepseek_v4").glob("*.[ch]pp"))
    paths.append(root / "python/llaisys/libllaisys/libllaisys.so")
    paths.append(root / "python/server/engine.py")
    if args.include_scheduler:
        paths.append(root / "test/test_scheduler.py")

    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    before = hashes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    suite = unittest.TestLoader().discover(str(root / "test"), pattern=args.pattern)
    if args.include_scheduler:
        suite.addTests(unittest.TestLoader().discover(str(root / "test"), pattern="test_scheduler.py"))
    started = time.perf_counter()
    with log_path.open("w") as log, redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(suite)
    failures = [{"test": test.id(), "method": getattr(test, "test_case", test).id(), "traceback": traceback}
                for test, traceback in result.failures]
    errors = [{"test": test.id(), "method": getattr(test, "test_case", test).id(), "traceback": traceback}
              for test, traceback in result.errors]
    unchanged = hashes() == before
    report = {
        "scope": "V4 unit/regression tests, not full-checkpoint or serving acceptance",
        "device": args.device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "compute_capability": torch.cuda.get_device_capability(0) if args.device == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version,
        "tests_run": result.testsRun, "passed_methods": len(result.passed),
        "failed_methods": len({entry["method"] for entry in failures}), "failure_entries": failures,
        "error_entries": errors, "skipped": [(test.id(), why) for test, why in result.skipped],
        "passed_test_ids": result.passed, "seconds": time.perf_counter() - started,
        "all_passed": result.wasSuccessful() and unchanged, "log": str(log_path.resolve()),
        "source_unchanged_during_tests": unchanged,
        "provenance": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                       "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
                       "command": sys.argv, "file_sha256": before}}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("tests_run", "passed_methods", "failed_methods", "all_passed", "log")}))
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()

"""Run real isolated cache bindings plus native CPU ownership regressions.

The temporary llaisys package only redirects the existing test file's import
to the freshly compiled extension. Its CacheBlock* objects and methods are real
C++ bindings, not Python doubles. Run this driver in a fresh Python process.
No installed library is rebuilt, loaded, copied or replaced by this driver.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_binary(path, *, environment=None, json_output=False):
    command = [str(path.resolve())]
    overrides = environment or {}
    result = subprocess.run(command, env=dict(os.environ, **overrides), text=True,
                            capture_output=True, timeout=60)
    report = dict(command=command, environment_overrides=overrides,
                  returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    report["all_passed"] = result.returncode == 0 and not result.stderr
    if json_output:
        try:
            report["result"] = json.loads(result.stdout)
            report["all_passed"] &= (report["result"].get("all_passed") is True
                                     and report["result"].get("checks", 0) > 0)
        except (ValueError, TypeError):
            report["all_passed"] = False
    else:
        report["all_passed"] &= result.stdout.strip() == "cache core tests passed"
    return report


def binding_regression(module_path, test_path):
    if any(name in sys.modules for name in ("llaisys", "llaisys._C", "_cache_lease_test")):
        raise RuntimeError("isolated bindings require a fresh process without llaisys modules")
    spec = importlib.util.spec_from_file_location("_cache_lease_test", module_path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    if Path(native.__file__).resolve() != module_path.resolve():
        raise RuntimeError("test did not load the requested native module")
    package = types.ModuleType("llaisys")
    package.__path__ = []
    package._C = native
    sys.modules["llaisys"] = package
    sys.modules["llaisys._C"] = native
    test_spec = importlib.util.spec_from_file_location("llaisys_isolated_cache_binding_tests", test_path)
    tests = importlib.util.module_from_spec(test_spec)
    sys.modules[test_spec.name] = tests
    try:
        test_spec.loader.exec_module(tests)
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        expected = suite.countTestCases()
        if expected < 12:
            raise RuntimeError("existing cache binding regression suite was unexpectedly reduced")
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        return dict(all_passed=result.wasSuccessful() and result.testsRun == expected and not result.skipped,
                    tests_run=result.testsRun, tests_expected=expected,
                    failures=[dict(test=str(test), traceback=trace) for test, trace in result.failures],
                    errors=[dict(test=str(test), traceback=trace) for test, trace in result.errors],
                    skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
                    output=log.getvalue(), module=str(module_path.resolve()),
                    real_cpp_bindings=True, python_doubles=False, installed_module_loaded=False)
    finally:
        for name in (test_spec.name, "llaisys._C", "llaisys", spec.name):
            sys.modules.pop(name, None)


def main(args):
    root = Path(__file__).resolve().parents[1]
    relative_sources = [
        "src/core/cache/block_lease.hpp", "src/core/cache/block_lease.cpp",
        "src/core/cache/block_manager.hpp", "src/core/cache/block_manager.cpp",
        "src/core/cache/block_prefix_cache.hpp", "src/core/cache/block_prefix_cache.cpp",
        "src/core/cache/cache_layout.hpp", "src/core/cache/cache_layout.cpp",
        "src/engine/schedule_plan.hpp", "python/bindings/cache.cpp",
        "test/block_lease_test.cpp", "test/cache_core_test.cpp",
        "test/cache_bindings_standalone.cpp", "test/run_cache_bindings_standalone.py",
        "test/test_deepseek_v4_cache_bindings.py",
    ]
    sources = [root / name for name in relative_sources]
    installed = [root / "python/llaisys/_C.so", root / "python/llaisys/libllaisys/libllaisys.so"]
    binaries = [args.module, args.lease_test, args.core_test, args.sanitized_test]
    if args.module.resolve() in {path.resolve() for path in installed}:
        raise ValueError("use a separately built test module, never the installed baseline")
    source_hashes = {str(path): digest(path) for path in sources}
    installed_hashes = {str(path): digest(path) for path in installed}
    artifact_hashes = {str(path.resolve()): digest(path) for path in binaries}
    report = dict(all_passed=False, scope="real CPU cache ownership and pybind contracts; not GPU/model correctness",
                  generated_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  python=sys.version, file_sha256=source_hashes, artifact_sha256=artifact_hashes,
                  installed_baseline_sha256=installed_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["native_lease"] = run_binary(args.lease_test, json_output=True)
        report["existing_core"] = run_binary(args.core_test)
        report["asan_ubsan"] = run_binary(args.sanitized_test, json_output=True,
            environment={"ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
                         "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"})
        report["asan_ubsan"]["leak_sanitizer_enabled"] = False
        report["asan_ubsan"]["leak_scan_not_claimed"] = True
        report["pybind"] = binding_regression(args.module.resolve(), root / "test/test_deepseek_v4_cache_bindings.py")
        report["source_unchanged"] = source_hashes == {str(path): digest(path) for path in sources}
        report["artifacts_unchanged"] = artifact_hashes == {str(path.resolve()): digest(path) for path in binaries}
        report["installed_baseline_unchanged"] = installed_hashes == {str(path): digest(path) for path in installed}
        report["all_passed"] = all(report[key]["all_passed"] for key in ("native_lease", "existing_core", "asan_ubsan", "pybind")) \
            and all(report[key] for key in ("source_unchanged", "artifacts_unchanged", "installed_baseline_unchanged"))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(all_passed=report["all_passed"], output=str(args.output.resolve()),
                         native_checks=report["native_lease"].get("result", {}).get("checks"),
                         binding_tests=report["pybind"]["tests_run"], leak_scan=False)))
    if not report["all_passed"]:
        raise RuntimeError("standalone CPU cache regression failed; see detailed report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--lease-test", type=Path, required=True)
    parser.add_argument("--core-test", type=Path, required=True)
    parser.add_argument("--sanitized-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

"""Real-weight pybind SchedulePlan integration against published text goldens.

Each pybind call executes whole native model steps, not Python operator loops.
Different requests share immutable weights and own separate continuous caches.
Serial slots are explicit; this is not fused batching or performance evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading

import torch
from transformers import AutoTokenizer
from run_independent import read_baselines, sha256
from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
from llaisys.models.deepseek_v4_model.config import InferenceConfig


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native Session verification requires Slurm GPU allocation")
    root = Path(__file__).resolve().parents[2]
    source, checkpoint, module_path = args.source_model.resolve(), args.checkpoint.resolve(), args.module.resolve()
    prior = json.loads(args.kernel_report.read_text())
    if not prior["all_passed"]: raise ValueError("kernel bundles must come from a passed native model run")
    for relative in ("config.json", "inference/config.json", "inference/kernel.py", "inference/model.py"):
        path = source / relative
        if prior["provenance"]["file_sha256"].get(str(path)) != sha256(path): raise ValueError("bundle model/source identity changed")
    manifests = [Path(p) for p in prior["artifact_sha256"] if Path(p).name == "manifest.txt"]
    if len(manifests) != 1: raise ValueError("ambiguous native bundle manifest")
    manifest = manifests[0]
    if sha256(manifest) != prior["artifact_sha256"][str(manifest)]: raise ValueError("bundle manifest changed")
    lines = manifest.read_text().splitlines(); header = shlex.split(lines[0]); location = shlex.split(lines[1])
    if header[0] != "LLAISYS_NATIVE_MODEL_V1" or Path(location[0]) != source or int(location[2]) != args.max_seq_len:
        raise ValueError("bundle source/config mismatch")
    bundles = {}
    for line in lines[2:2 + int(location[3])]:
        key, operation, path = shlex.split(line)
        if key in bundles or sha256(path) != prior["artifact_sha256"].get(path): raise ValueError("changed/duplicate kernel artifact")
        bundles[key] = (operation, path)
    paths = [Path(__file__), args.kernel_report, module_path, manifest, *[Path(v[1]) for v in bundles.values()],
             root / "python/bindings/v4_native.cpp", root / "src/backends/v4_reference.cpp", root / "src/backends/v4_reference.hpp",
             root / "xmake/native_model.lua", root / "xmake/aten.lua", root / "xmake/tilelang.lua",
             root / "python/llaisys/_C.so", root / "python/llaisys/libllaisys/libllaisys.so",
             Path(read_baselines.__code__.co_filename), Path(checkpoint_identity.__code__.co_filename)]
    for directory in ("src/models/deepseek_v4", "src/backends/native", "src/backends/aten", "src/backends/hadamard", "src/backends/tilelang"):
        paths.extend(sorted((root / directory).glob("*.[ch]pp")))
    paths.extend(source / relative for relative in ("config.json", "inference/config.json", "inference/kernel.py", "inference/model.py"))
    paths.extend(root / "python/llaisys/models/deepseek_v4_model" / name for name in ("config.py", "chat.py"))
    hashes = {str(path.resolve()): sha256(path) for path in paths}
    cfg = InferenceConfig.from_directory(source, max_seq_len=args.max_seq_len)
    print("verifying complete checkpoint identity", flush=True); identity = checkpoint_identity(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    cases = read_baselines(args.baseline, source, checkpoint, args.max_seq_len, tokenizer, cfg.vocab_size, identity)
    if not all(c["checkpoint_payload_verified"] for c in cases): raise ValueError("all goldens must bind complete weight identity")
    spec = importlib.util.spec_from_file_location("_v4_native", module_path)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    counter, stop = [0], threading.Event()
    def heartbeat():
        while not stop.wait(0.005): counter[0] += 1
    thread = threading.Thread(target=heartbeat); thread.start()
    session = None; steps = comparisons = negative = 0; results = []
    try:
        print("loading pybind native Session", flush=True)
        before_load = counter[0]
        session = native.Session(str(source), str(checkpoint), args.max_seq_len, 2, bundles, header[1], header[3])
        gil_load = counter[0] - before_load
        info_loaded = session.info(); print(json.dumps(info_loaded), flush=True)
        if info_loaded["loaded_weights"] != 67569 or info_loaded["layers"] != 43 or gil_load == 0:
            raise RuntimeError("incomplete native load or GIL not released")
        def reject(plan):
            nonlocal negative
            try: session.execute(plan)
            except (ValueError, RuntimeError, TypeError, OverflowError): negative += 1
            else: raise AssertionError("invalid native SchedulePlan accepted")
        for plan in ({"reset_slots": [True]}, {"reset_slots": [2]}, {"reset_slots": [0, 0]}, {"bad": 1},
                     {"decodes": [{"slot_id": 0, "token_id": 1}]},
                     {"prefills": [{"slot_id": 0, "token_ids": []}]},
                     {"prefills": [{"slot_id": 0, "token_ids": [True]}]},
                     {"prefills": [{"slot_id": 0, "token_ids": [-1]}]},
                     {"prefills": [{"slot_id": 0, "token_ids": [cfg.vocab_size]}]},
                     {"prefills": [{"slot_id": 0, "token_ids": [1], "start_pos": 1}]},
                     {"prefills": [{"slot_id": 0, "token_ids": [1], "is_last_chunk": False}]},
                     {"prefills": [{"slot_id": 0, "token_ids": [1], "sampling": {"temperature": 1, "top_k": 2}}]}):
            reject(plan)
        if session.info()["active_slots"] != 0: raise AssertionError("invalid plan mutated empty slots")
        with ThreadPoolExecutor(max_workers=2) as callers:
            # Each wave has up to two DIFFERENT real prompts. A single execute
            # contains both slots; shorter completed requests are reset while
            # the other keeps decoding, without sharing mutable model state.
            for start in range(0, len(cases), 2):
                wave = cases[start:start + 2]; generated = [[] for _ in wave]
                plan = {"step_id": steps, "reset_slots": list(range(len(wave))), "prefills": [
                    {"request_id": start + slot, "slot_id": slot, "token_ids": c["tensors"]["input_ids"][0].tolist()}
                    for slot, c in enumerate(wave)]}
                while plan.get("prefills") or plan.get("decodes"):
                    before = counter[0]
                    actual = callers.submit(session.execute, plan, True).result()
                    if counter[0] <= before: raise AssertionError("native execute held Python GIL")
                    if actual["step_id"] != steps: raise AssertionError("wrong SchedulePlan step ID")
                    resets, decodes = [], []
                    for row in actual["outputs"]:
                        slot = row["slot_id"]; case = wave[slot]; step = len(generated[slot])
                        if row["request_id"] != start + slot: raise AssertionError("request/slot mapping changed")
                        logits = torch.tensor(row["logits"], dtype=torch.float32, device="cpu").reshape(1, -1)
                        expected = case["tensors"]["logits"][step]
                        if not torch.equal(logits.view(torch.uint8), expected.view(torch.uint8)):
                            raise AssertionError(f"non-exact native pybind logits case={start + slot}, step={step}")
                        if row["token_id"] != int(case["tensors"]["generated_ids"][step]): raise AssertionError("wrong generated token")
                        comparisons += 1; generated[slot].append(row["token_id"])
                        if len(generated[slot]) < case["tensors"]["generated_ids"].numel():
                            decodes.append({"request_id": start + slot, "slot_id": slot, "token_id": row["token_id"]})
                        else: resets.append(slot)
                    steps += 1
                    plan = {"step_id": steps, "reset_slots": resets, "decodes": decodes}
                    if not decodes: session.execute(plan)
                for slot, case in enumerate(wave):
                    results.append({"case_id": case["case_id"], "input_tokens": case["tensors"]["input_ids"].shape[1],
                        "output_tokens": len(generated[slot]), "generated_ids": generated[slot], "exact_logits": True,
                        "golden_file": case["golden_file"], "golden_sha256": case["golden_sha256"]})
                print(f"PASS cases {start}..{start + len(wave) - 1}, total {comparisons} exact logits", flush=True)
                if session.info()["active_slots"] != 0: raise AssertionError("finished slots not released")
            info_final = session.info()
            callers.submit(session.close).result() # teardown from another caller thread
            session.close() # idempotent
            try: session.info()
            except (ValueError, RuntimeError): negative += 1
            else: raise AssertionError("closed Session accepted command")
    finally:
        if session is not None: session.close()
        stop.set(); thread.join()
    verify_checkpoint_unchanged(checkpoint, identity)
    unchanged = hashes == {str(path.resolve()): sha256(path) for path in paths}
    report = {"all_passed": unchanged, "scope": "whole native model through pybind SchedulePlan; serial slots, not paged/fused/HTTP/performance",
        "source_unchanged": unchanged, "checkpoint_identity": identity, "checkpoint_unchanged": True,
        "cases": results, "logits_byte_comparisons": comparisons, "plan_steps": steps, "negative_checks": negative,
        "loaded": info_loaded, "final": info_final, "gil_heartbeat_during_load": gil_load,
        "free_generation": True, "python_operator_callbacks": False, "fallback": False,
        "measurement": {"kind": "correctness_only", "slots": 2, "warmup": 0, "repeat": 1, "cuda_graph": False, "prefix_cache": False},
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__,
        "provenance": {"command": sys.argv, "file_sha256": hashes,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "dirty": True}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": unchanged, "output": str(args.output)}), flush=True)
    raise SystemExit(0 if unchanged else 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-model", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--module", type=Path, default=Path("build/linux/x86_64/release/_v4_native.so"))
    p.add_argument("--kernel-report", type=Path, required=True); p.add_argument("--baseline", nargs=2, action="append", required=True)
    p.add_argument("--max-seq-len", type=int, default=4096); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    try: main(args)
    except Exception as error:
        args.output.write_text(json.dumps({"all_passed": False, "scope": "native pybind verification incomplete",
            "command": sys.argv, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise

"""Tokenize chat and execute the full C++ V4 model; no golden files required.

Single-request greedy CLI, not HTTP or a serving performance benchmark.
Python formats/encodes text and submits model steps; every operator runs in C++.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoTokenizer
from llaisys.models.deepseek_v4_model.chat import ChatCodec
from llaisys.models.deepseek_v4_model.native_session import load_native_session, _digest


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(): raise RuntimeError("native chat requires Slurm GPU allocation")
    if args.max_new_tokens <= 0: raise ValueError("max-new-tokens must be positive")
    source, checkpoint = args.source_model.resolve(), args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    codec = ChatCodec(source, tokenizer)
    encoded = codec.encode([{"role": "user", "content": args.prompt}], thinking_mode=args.thinking_mode)
    before = {str(path.resolve()): _digest(path) for path in (Path(__file__), args.module, args.bundle, codec.path,
        Path(sys.modules[load_native_session.__module__].__file__))}
    generated, reason = [], "length"
    with load_native_session(source, checkpoint, args.bundle, args.module) as session:
        info = session.info()
        if len(tokenizer) > info["vocabulary"] or len(encoded["input_ids"]) + args.max_new_tokens - 1 > info["capacity"]:
            raise ValueError("tokenizer or generation budget exceeds model capacity")
        plan = {"prefills": [{"slot_id": 0, "token_ids": encoded["input_ids"]}]}
        for step in range(args.max_new_tokens):
            plan["step_id"] = step; row = session.execute(plan)["outputs"][0]; token = row["token_id"]
            generated.append(token)
            if token == codec.eos_id: reason = "stop"; break
            plan = {"decodes": [{"slot_id": 0, "token_id": token}]}
        session.execute({"reset_slots": [0]}); final = session.info()
    completion = codec.decode_completion(generated, thinking_mode=args.thinking_mode, finish_reason=reason)
    unchanged = before == {path: _digest(path) for path in before}
    report = {"executor": "native-cpp-pybind", "model": str(source), "checkpoint": str(checkpoint),
        "input_ids": encoded["input_ids"], "generated_ids": generated, "input_tokens": len(encoded["input_ids"]),
        "output_tokens": len(generated), "completion": completion, "codec": codec.report(), "model_runtime": info,
        "final_runtime": final, "golden_required": False, "teacher_forcing": False, "source_unchanged": unchanged,
        "measurement": {"kind": "functional_smoke_not_ttft_tpot", "batch": 1, "concurrency": 1, "warmup": 0, "repeat": 1,
                        "cuda_graph": False, "prefix_cache": False, "fallback": False},
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "provenance": {"command": sys.argv, "file_sha256": before,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
        "execution_passed": unchanged and final["active_slots"] == 0 and completion["parse_error"] is None}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(completion, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["execution_passed"] else 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-model", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True); p.add_argument("--module", type=Path, default=Path("build/linux/x86_64/release/_v4_native.so"))
    p.add_argument("--prompt", required=True); p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat"); p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())

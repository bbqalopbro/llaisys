"""Full-checkpoint acceptance through the EXISTING threaded InferenceEngine.

The selected baseline supplies inputs and comparison data only; the engine feeds
back its own generated tokens. Step observers make this a correctness run, NOT
a performance benchmark. Model execution is explicitly the Python reference
backend, with C++ storage and upstream TileLang operators, not C++ fused batches.
"""

import argparse
import asyncio
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from llaisys import _C
from llaisys.models.deepseek_v4_backends import CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model
from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
from llaisys.models.deepseek_v4_model import InferenceConfig, DeepSeekV4Model, load_converted_weights
from llaisys.models.deepseek_v4_model.batch import DeepSeekV4ServingModel
from llaisys.models.deepseek_v4_model.chat import ChatCodec, check_chat_answer
from server.engine import InferenceEngine, SamplingParams
from run_independent import load_kernel, read_baselines, compare, sha256


async def exercise(model, codec, case, args):
    inputs = case["tensors"]["input_ids"][0].tolist()
    golden_ids = case["tensors"]["generated_ids"].tolist()
    observed = []

    def observe(stage, slot, position, tokens, logits):
        if logits is None:
            return
        index = 0 if stage != "decode" else position - len(inputs) + 1
        if not 0 <= index < len(golden_ids):
            raise RuntimeError("execution step is outside the selected golden history")
        observed.append({"stage": stage, "slot": slot, "position": position, "step": index,
                         **compare(logits.detach().cpu(), case["tensors"]["logits"][index])})

    facade = DeepSeekV4ServingModel(model, eos_id=codec.eos_id, num_blocks=args.num_blocks,
                                   enable_prefix_cache=args.prefix_cache,
                                   allow_experimental_chunking=args.allow_experimental_chunking, observer=observe)
    engine = InferenceEngine(facade, max_batch_size=args.concurrency, max_seq_per_slot=args.max_seq_len,
                             block_watermark=0, prefill_chunk_size=args.chunk_size or args.max_seq_len,
                             enable_block_prefix_cache=args.prefix_cache)
    native_before = _C.v4_storage_counters()
    requests = [engine.submit(inputs, SamplingParams(temperature=0, top_k=1, top_p=1,
                                                    max_tokens=len(golden_ids)), str(index), stream=True)
                for index in range(args.concurrency)]
    started = time.perf_counter()
    engine.start()
    try:
        results = await asyncio.wait_for(asyncio.gather(*(request.future for request in requests)), timeout=600)
        streams = [[token async for token in request.stream_tokens()] for request in requests]
    finally:
        await asyncio.to_thread(engine.stop, 60)
    elapsed = time.perf_counter() - started
    gc.collect()
    native_after = _C.v4_storage_counters()
    completions = []
    for ids in results:
        if case["chat"] is None:
            completions.append({"raw_text": codec.tokenizer.decode(ids, skip_special_tokens=False)})
        else:
            completion = codec.decode_completion(ids, thinking_mode=case["chat"]["thinking_mode"],
                                                 finish_reason="stop" if ids[-1] == codec.eos_id else "length")
            completions.append({"completion": completion, "expected_match": check_chat_answer(completion, case)})
    token_match = all(ids == golden_ids for ids in results) and streams == results
    logits_exact = len(observed) == len(golden_ids) * args.concurrency and all(step["exact_equal"] for step in observed)
    lifecycle = native_before == native_after and facade.last_report["pool"]["num_free"] == facade.last_report["pool"]["num_blocks"]
    return {"case_id": case["case_id"], "input_tokens": len(inputs), "output_budget": len(golden_ids),
            "generated_ids": results, "streamed_ids": streams, "completions": completions,
            "all_generated_ids_match_golden": token_match, "all_logits_exact": logits_exact,
            "steps": observed, "engine_stats": engine.get_stats(), "runtime": facade.last_report,
            "native_lifecycle": {"before": native_before, "after": native_after, "passed": lifecycle},
            "unwarmed_correctness_wall_seconds": elapsed,
            "baseline_report": case["baseline_report"], "baseline_report_sha256": case["baseline_report_sha256"],
            "golden_file": case["golden_file"], "golden_sha256": case["golden_sha256"],
            "all_passed": token_match and logits_exact and lifecycle}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", nargs=2, action="append", required=True)
    parser.add_argument("--case-index", type=int, action="append")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--allow-experimental-chunking", action="store_true")
    parser.add_argument("--reference-profile", choices=("published", "fixed32-reference-v1"), default="published")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("full-weight GPU acceptance requires a Slurm allocation")
    if args.concurrency < 1 or args.chunk_size < 0:
        raise ValueError("invalid concurrency/chunk size")
    source, checkpoint = args.source_model.resolve(), args.checkpoint.resolve()
    cfg = InferenceConfig.from_directory(source, max_seq_len=args.max_seq_len)
    fixed = args.reference_profile != "published"
    if fixed:
        cfg = replace(cfg, indexer_tie_policy="index_ascending", attention_metadata_policy="fixed")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    codec = ChatCodec(source, tokenizer)
    import tilelang
    import tvm_ffi
    import fast_hadamard_transform as hadamard
    hadamard.configure_backend("cuda")
    registry = OperatorRegistry(MODEL_CONTRACTS)
    register_tilelang(registry, load_kernel(source / "inference/kernel.py"), tilelang.__version__)
    register_torch_model(registry, backend="torch-fixed32" if fixed else "torch", fixed_rows=32 if fixed else 0)
    ops = registry.bind({key: "tilelang" if key in CONTRACTS else "torch-fixed32" if fixed else "torch"
                         for key in MODEL_CONTRACTS})
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "tools/deepseek_v4_reference").glob("*.py")),
             *sorted((root / "python/llaisys/models").glob("deepseek_v4*.py")),
             *sorted((root / "python/llaisys/models/deepseek_v4_model").glob("*.py")),
             root / "python/server/engine.py", Path(_C.__file__), root / "python/llaisys/libllaisys/libllaisys.so",
             codec.path, source / "inference/kernel.py", source / "inference/model.py",
             source / "inference/config.json", source / "config.json", Path(hadamard.__file__),
             Path(hadamard.backend_report()["extension"])]
    hashes = {str(path): sha256(path) for path in paths}
    print("verifying complete checkpoint payload", flush=True)
    identity = checkpoint_identity(checkpoint)
    cases = read_baselines(args.baseline, source, checkpoint, args.max_seq_len, tokenizer, cfg.vocab_size,
                           identity, args.reference_profile)
    if args.case_index is not None:
        cases = [cases[index] for index in args.case_index]
    model = DeepSeekV4Model(cfg, ops, hadamard.hadamard_transform)
    weights = load_converted_weights(model, checkpoint)
    print(f"strict load: {weights['tensor_count']} main tensors", flush=True)
    results = []
    for case in cases:
        result = asyncio.run(exercise(model, codec, case, args))
        results.append(result)
        print(json.dumps({key: result[key] for key in ("case_id", "all_passed", "all_logits_exact", "all_generated_ids_match_golden")}), flush=True)
    verify_checkpoint_unchanged(checkpoint, identity)
    unchanged = hashes == {str(path): sha256(path) for path in paths}
    report = {"scope": "existing InferenceEngine + Python V4 reference model + C++ paged storage",
              "cpp_model_runtime_complete": False, "published_model_imported": False, "teacher_forcing": False,
              "reference_profile": args.reference_profile, "unmodified_published_baseline_tested": not fixed,
              "model_config": asdict(cfg), "weights": weights, "checkpoint_identity": identity,
              "gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(),
              "compute_capability": torch.cuda.get_device_capability(0), "torch": torch.__version__,
              "cuda": torch.version.cuda, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__,
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "operator_backends": ops.report(),
              "hadamard": hadamard.backend_report(), "codec": codec.report(),
              "measurement": {"kind": "observed_correctness_not_performance", "warmup": 0, "repeats": 1,
                              "concurrency": args.concurrency, "gpu_batch_size": 1, "chunk_size": args.chunk_size,
                              "prefix_cache": args.prefix_cache, "cuda_graph": False, "fallback": False, "mtp": False},
              "cases": results, "source_unchanged": unchanged,
              "all_passed": unchanged and all(case["all_passed"] for case in results),
              "provenance": {"command": sys.argv, "file_sha256": hashes,
                             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                             "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(args.output)}), flush=True)
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    main()

"""Run real V4 conversations with the independent single-GPU executor.

This CLI does not import the upstream model or require saved golden logits.
It is not yet the production C++ batch/serving path. Tools are never executed.
"""

import argparse
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
from llaisys.models.deepseek_v4_backends import (
    CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_torch_model, register_deepgemm,
)
from llaisys.models.deepseek_v4_model import InferenceConfig, DeepSeekV4Model, PagedCachePool, load_converted_weights
from llaisys.models.deepseek_v4_model.chat import ChatCodec, check_chat_answer
from llaisys.models.deepseek_v4_model.generation import greedy_generate
from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
from run_independent import load_kernel, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases-file", required=True, type=Path,
                        help="JSON array of {messages, thinking_mode?, expected_exact?} conversations")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    parser.add_argument("--prefix-cache", action="store_true",
                        help="explicit experimental V4 block/compressor prefix sharing")
    parser.add_argument("--gemm-backend", choices=("tilelang", "deepgemm"), default="tilelang")
    parser.add_argument("--model-linear-backend", choices=("torch", "torch-fixed32"), default="torch")
    parser.add_argument("--indexer-tie-policy", choices=("published", "index_ascending"), default="published")
    parser.add_argument("--attention-metadata-policy", choices=("published", "fixed"), default="published")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("this GPU runner requires a Slurm allocation")
    if args.max_new_tokens <= 0 or args.prefill_chunk_size < 0:
        raise ValueError("invalid generation budget/chunk size")
    source, checkpoint = args.source_model.resolve(), args.checkpoint.resolve()
    cfg = replace(InferenceConfig.from_directory(source, max_seq_len=args.max_seq_len),
                  indexer_tie_policy=args.indexer_tie_policy, attention_metadata_policy=args.attention_metadata_policy)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    if len(tokenizer) > cfg.vocab_size:
        raise ValueError("tokenizer exceeds the model vocabulary")
    codec = ChatCodec(source, tokenizer)
    cases = json.loads(args.cases_file.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases-file must contain a nonempty JSON array")
    prepared = []
    for case in cases:
        options = {key: case.get(key, default) for key, default in
                   (("thinking_mode", "chat"), ("drop_thinking", True), ("reasoning_effort", "low"))}
        encoded = codec.encode(case["messages"], **options)
        if len(encoded["input_ids"]) >= cfg.max_seq_len:
            raise ValueError("conversation leaves no generation capacity")
        prepared.append((case, options, encoded))
    import tilelang
    import tvm_ffi
    import fast_hadamard_transform as hadamard
    hadamard.configure_backend("cuda")
    registry = OperatorRegistry(MODEL_CONTRACTS)
    register_tilelang(registry, load_kernel(source / "inference/kernel.py"), tilelang.__version__)
    register_torch_model(registry, backend=args.model_linear_backend,
                         fixed_rows=32 if args.model_linear_backend == "torch-fixed32" else 0)
    selection = {key: "tilelang" if key in CONTRACTS else args.model_linear_backend for key in MODEL_CONTRACTS}
    if args.gemm_backend == "deepgemm":
        import deep_gemm
        register_deepgemm(registry, deep_gemm)
        selection.update(fp8_gemm="deepgemm", fp4_gemm="deepgemm")
    ops = registry.bind(selection)
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(__file__).with_name("run_independent.py"), args.cases_file.resolve(),
             codec.path, source / "inference/kernel.py", source / "inference/config.json", source / "config.json",
             Path(_C.__file__), root / "python/llaisys/libllaisys/libllaisys.so",
             Path(hadamard.__file__), Path(hadamard.backend_report()["extension"]),
             *sorted((root / "python/llaisys/models").glob("deepseek_v4*.py")),
             *sorted((root / "python/llaisys/models/deepseek_v4_model").glob("*.py"))]
    before = {str(path): sha256(path) for path in paths}
    identity = checkpoint_identity(checkpoint)
    model = DeepSeekV4Model(cfg, ops, hadamard.hadamard_transform)
    weights = load_converted_weights(model, checkpoint)
    native_before = _C.v4_storage_counters()
    pool = PagedCachePool(model, num_blocks=(cfg.max_seq_len + 127) // 128, block_size=128,
                         storage_backend="cpp", enable_prefix_cache=args.prefix_cache)
    results = []
    for case, options, encoded in prepared:
        started = time.perf_counter()
        result = greedy_generate(model, encoded["input_ids"], max_new_tokens=args.max_new_tokens,
                                 eos_id=codec.eos_id, chunk_size=args.prefill_chunk_size, cache_pool=pool,
                                 reuse_prefix=args.prefix_cache)
        torch.cuda.synchronize()
        result["wall_seconds"] = time.perf_counter() - started
        result["completion"] = codec.decode_completion(result["generated_ids"], thinking_mode=options["thinking_mode"],
                                                       finish_reason=result["finish_reason"])
        result["case_id"], result["encoding_options"] = case.get("id", str(len(results))), options
        result["input_ids"] = encoded["input_ids"]
        result["expected_match"] = check_chat_answer(result["completion"], case)
        results.append(result)
        print(json.dumps({"id": result["case_id"], "completion": result["completion"],
                          "expected_match": result["expected_match"]}, ensure_ascii=False), flush=True)
    pool_report = pool.report()
    pool = None
    gc.collect()
    native_after = _C.v4_storage_counters()
    verify_checkpoint_unchanged(checkpoint, identity)
    unchanged = before == {str(path): sha256(path) for path in paths}
    report = {"executor": "llaisys-independent-python-reference", "teacher_forcing": False,
              "published_model_imported": False, "model_config": asdict(cfg), "weights": weights,
              "checkpoint_identity": identity, "codec": codec.report(), "operator_backends": ops.report(),
              "hadamard": hadamard.backend_report(), "gpu": torch.cuda.get_device_name(0),
              "compute_capability": torch.cuda.get_device_capability(0), "torch": torch.__version__,
              "cuda": torch.version.cuda, "tilelang": tilelang.__version__, "tvm_ffi": tvm_ffi.__version__,
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "paged_pool": pool_report,
              "native_storage_lifecycle": {"before": native_before, "after": native_after},
              "measurement": {"kind": "unwarmed_single_request_smoke_not_serving_ttft_tpot", "warmup": 0,
                              "repeats": 1, "batch": 1, "concurrency": 1, "prefix_cache": args.prefix_cache,
                              "cuda_graph": False, "fallback": False, "mtp": False},
              "cases": results, "source_unchanged": unchanged,
              "all_conversations_completed": all(c["completion"]["complete"] for c in results),
              "provenance": {"file_sha256": before, "command": sys.argv,
                             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                             "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())},
              "all_passed": unchanged and native_after == native_before and pool_report["num_free"] == pool_report["num_blocks"]
                            and all(c["expected_match"] is not False and c["completion"]["parse_error"] is None for c in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "report": str(args.output)}), flush=True)
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    main()

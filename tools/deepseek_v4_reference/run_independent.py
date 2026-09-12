"""Validate the independent llaisys V4 executor against saved published logits.

Only the explicitly selected kernel backend is imported from the model package;
its model.py, global model settings, cache buffers and sampling are not used.
This remains a correctness executor, not a serving performance benchmark.
"""

import argparse
import gc
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from llaisys.models.deepseek_v4_backends import (
    CONTRACTS, MODEL_CONTRACTS, OperatorRegistry, register_tilelang, register_deepgemm, register_torch_model,
)
from llaisys.models.deepseek_v4_model import DeepSeekV4Model, InferenceConfig, load_converted_weights
from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged, require_same_checkpoint
from llaisys.models.deepseek_v4_model.chat import ChatCodec, check_chat_answer
from llaisys.models.deepseek_v4_model.generation import greedy_generate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_kernel(path):
    spec = importlib.util.spec_from_file_location("llaisys_v4_tilelang_kernels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_baselines(pairs, source, checkpoint, max_seq_len, tokenizer, model_vocab_size, weight_identity=None,
                   reference_profile="published"):
    if len(tokenizer) > model_vocab_size:
        raise ValueError("tokenizer (including added tokens) exceeds the model vocabulary")
    cases = []
    for report_path, golden_directory in pairs:
        report_path, golden_directory = Path(report_path).resolve(), Path(golden_directory).resolve()
        report = json.loads(report_path.read_text())
        if reference_profile not in ("published", "fixed32-reference-v1"):
            raise ValueError("unknown baseline numerical profile")
        expected_backend = "published-tilelang" + ("-fixed32-reference-v1" if reference_profile != "published" else "")
        if (report["backend"] != expected_backend or report["measurement"]["fallback"]
                or report.get("reference_profile", "published") != reference_profile):
            raise ValueError("baseline must be the explicit published TileLang path without fallback")
        if reference_profile != "published":
            path = Path(__file__).with_name("published_fixed_profile.py")
            spec = importlib.util.spec_from_file_location("llaisys_profile_manifest", path)
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            if report.get("profile_adapter") != adapter.prepare_source(source / "inference/model.py")[1]:
                raise ValueError("baseline numerical-profile adapter or audited source changed")
        if report.get("checkpoint_identity") is not None:
            if weight_identity is None:
                raise ValueError("baseline requires checkpoint payload identity verification")
            require_same_checkpoint(weight_identity, report["checkpoint_identity"])
        if (Path(report["source_model"]).resolve() != source
                or Path(report["converted_model"]).resolve() != checkpoint.parent):
            raise ValueError("baseline model/checkpoint path differs from this run")
        command = report["provenance"]["command"]
        if int(command[command.index("--max-seq-len") + 1]) != max_seq_len:
            raise ValueError("baseline cache/RoPE maximum differs from this run")
        hashes = report["provenance"]["file_sha256"]
        for relative in ("inference/kernel.py", "inference/config.json", "inference/model.py"):
            path = source / relative
            if hashes.get(str(path)) != sha256(path):
                raise ValueError(f"baseline source changed: {path}")
        for index, case in enumerate(report["cases"]):
            path = golden_directory / f"case_{index}.safetensors"
            if report.get("golden_file_sha256") is not None:
                if report["golden_file_sha256"].get(path.name) != sha256(path):
                    raise ValueError(f"golden artifact hash differs from baseline: {path}")
            tensors = load_file(str(path))
            ids, generated, logits = (tensors[key] for key in ("input_ids", "generated_ids", "logits"))
            chat = case.get("chat")
            if chat is not None:
                codec = ChatCodec(source, tokenizer)
                if case.get("input_format") != "chat" or chat["codec"] != codec.report():
                    raise ValueError("baseline chat encoding/tokenizer identity differs")
                encoded_chat = codec.encode(chat["messages"], **{key: chat[key] for key in
                                            ("thinking_mode", "drop_thinking", "reasoning_effort")})
                if encoded_chat["text"] != case["prompt"]:
                    raise ValueError("baseline chat messages and rendered prompt disagree")
                encoded = torch.tensor([encoded_chat["input_ids"]], dtype=torch.int64, device="cpu")
                if not case.get("stop_on_eos") or case.get("finish_reason") not in ("stop", "length"):
                    raise ValueError("chat baseline must record EOS/length termination")
                completion = codec.decode_completion(generated.tolist(), thinking_mode=chat["thinking_mode"],
                                                     finish_reason=case["finish_reason"])
                if completion != case["completion"]:
                    raise ValueError("baseline chat completion differs from recorded output")
            else:
                if case.get("input_format", "raw-completion") != "raw-completion":
                    raise ValueError("chat baseline is missing its encoding metadata")
                encoded = tokenizer.encode(case["prompt"], return_tensors="pt")
            if (not torch.equal(ids, encoded) or generated.tolist() != case["generated_ids"]
                    or ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64
                    or logits.shape != (generated.numel(), 1, model_vocab_size)
                    or logits.dtype != torch.float32 or not torch.isfinite(logits).all()
                    or not torch.equal(logits.argmax(-1).flatten(), generated)
                    or generated.numel() < 1 or ids.shape[1] + generated.numel() > max_seq_len):
                raise ValueError(f"invalid or mismatched golden tensors: {path}")
            cases.append({"prompt": case["prompt"], "expected_contains": case.get("expected_contains"),
                          "expected_exact": case.get("expected_exact"), "chat": chat,
                          "case_id": case.get("case_id", str(index)),
                          "max_new_tokens": case.get("max_new_tokens", generated.numel()),
                          "tensors": tensors, "golden_file": str(path), "golden_sha256": sha256(path),
                          "checkpoint_payload_verified": report.get("checkpoint_identity") is not None,
                          "reference_profile": reference_profile,
                          "baseline_report": str(report_path), "baseline_report_sha256": sha256(report_path)})
    if not cases:
        raise ValueError("at least one baseline case is required")
    return cases


def compare(actual, expected):
    difference = actual - expected
    finite = bool(torch.isfinite(actual).all())
    return {"finite": finite, "argmax_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
            "max_abs_error": difference.abs().max().item() if finite else None,
            "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-8)).item() if finite else None,
            "cosine_similarity": torch.nn.functional.cosine_similarity(actual, expected, dim=-1).item() if finite else None,
            "exact_equal": bool(torch.equal(actual, expected))}


@torch.inference_mode()
def evaluate(model, tokenizer, case, warmup, dump_path, chunk_size=0, cache_pool=None,
             intermediate_logits=False, prefix_mode="off"):
    if prefix_mode not in ("off", "publish", "reuse"):
        raise ValueError("invalid prefix validation mode")
    golden = case["tensors"]
    inputs = golden["input_ids"].to(model.embed.weight.device)
    def prefill(state, capture=False):
        size = chunk_size or inputs.shape[1]
        for position in range(state.position, inputs.shape[1], size):
            final = position + size >= inputs.shape[1]
            output = model(inputs[:, position:position+size], state,
                           capture_layers=capture and final, emit_logits=intermediate_logits or final)
        return output
    for _ in range(warmup):
        state = model.new_request(cache_pool=cache_pool)
        try:
            if prefix_mode == "reuse":
                state.attach_prefix(golden["input_ids"][0].tolist())
            prefill(state)
            model(golden["generated_ids"][:1].reshape(1, 1).to(inputs.device), state)
        finally:
            state.close()
    state = model.new_request(cache_pool=cache_pool)
    steps, saved, predicted, hiddens = [], [], [], {}
    hit_tokens, published = 0, False
    try:
        if prefix_mode == "reuse":
            hit_tokens = state.attach_prefix(golden["input_ids"][0].tolist())
        for index, expected in enumerate(golden["logits"]):
            current = (inputs if index == 0 else
                       golden["generated_ids"][index-1:index].reshape(1, 1).to(inputs.device))
            position = state.position
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = (prefill(state, dump_path is not None) if index == 0 else
                      model(current, state, capture_layers=dump_path is not None))
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            if index == 0 and prefix_mode == "publish":
                published = state.publish_prefix()
            logits = output.logits.cpu()
            saved.append(logits)
            predicted.append(logits.argmax(-1).item())
            steps.append({"step": index, "phase": "prefill" if index == 0 else "decode",
                          "position": position, "input_tokens": current.shape[1] - (hit_tokens if index == 0 else 0),
                          "latency_ms": elapsed, **compare(logits, expected)})
            for layer, value in enumerate(output.layer_hiddens):
                hiddens[f"step_{index}.layer_{layer}"] = value.cpu()
    finally:
        state.close()
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({**hiddens, "input_ids": golden["input_ids"], "teacher_forced_ids": golden["generated_ids"],
                   "logits": torch.stack(saved)}, str(dump_path))
    text = tokenizer.decode(predicted)
    expected_text = case["expected_contains"]
    passed = all(s["finite"] and s["argmax_equal"] and s["relative_l2"] <= 0.01
                 and s["cosine_similarity"] >= 0.999 for s in steps)
    return {key: value for key, value in case.items() if key != "tensors"} | {
        "input_tokens": inputs.shape[1], "output_tokens": len(predicted), "generated_ids": predicted,
        "prefix_mode": prefix_mode, "prefix_hit_tokens": hit_tokens, "prefix_published": published,
        "prefill_tokens_executed": inputs.shape[1] - hit_tokens,
        "intermediate_prefill_logits": intermediate_logits,
        "skipped_prefill_heads": (0 if intermediate_logits or not chunk_size else
                                  (inputs.shape[1] - hit_tokens - 1) // chunk_size),
        "generated_text": text, "teacher_forced_ids": golden["generated_ids"].tolist(),
        "expected_match": expected_text in text if expected_text else None,
        "steps": steps, "numerical_gate_passed": passed,
        "all_logits_exact": all(s["exact_equal"] for s in steps), "_logits": torch.stack(saved)}


def evaluate_free_generation(model, tokenizer, case, source, chunk_size, cache_pool):
    result = greedy_generate(model, case["tensors"]["input_ids"][0].tolist(),
                             max_new_tokens=case["max_new_tokens"], eos_id=tokenizer.eos_token_id,
                             chunk_size=chunk_size, cache_pool=cache_pool,
                             reuse_prefix=cache_pool is not None and cache_pool.prefix is not None)
    result["teacher_forcing"] = False
    result["matches_golden_ids"] = result["generated_ids"] == case["tensors"]["generated_ids"].tolist()
    result["expected_match"] = None
    if case["chat"] is not None:
        codec = ChatCodec(source, tokenizer)
        result["completion"] = codec.decode_completion(result["generated_ids"],
            thinking_mode=case["chat"]["thinking_mode"], finish_reason=result["finish_reason"])
        result["expected_match"] = check_chat_answer(result["completion"], case)
    else:
        result["text"] = tokenizer.decode(result["generated_ids"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", nargs=2, action="append", required=True, metavar=("REPORT", "LOGITS_DIR"))
    parser.add_argument("--reference-profile", choices=("published", "fixed32-reference-v1"), default="published")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--gemm-backend", choices=("tilelang", "deepgemm"), default="tilelang")
    parser.add_argument("--model-linear-backend", choices=("torch", "torch-fixed32"), default="torch")
    parser.add_argument("--indexer-tie-policy", choices=("published", "index_ascending"), default="published")
    parser.add_argument("--attention-metadata-policy", choices=("published", "fixed"), default="published")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=0,
                        help="explicit experimental multi-token chunk path; 0 uses full prefill")
    parser.add_argument("--intermediate-prefill-logits", action="store_true",
                        help="diagnostic mode: also compute unused logits on intermediate chunks")
    parser.add_argument("--compare-full-in-process", action="store_true",
                        help="also compare chunk/full with identical selected numerical policies")
    parser.add_argument("--free-generation", action="store_true",
                        help="also generate without golden-token feedback, stopping on real EOS")
    parser.add_argument("--prefix-reuse", action="store_true",
                        help="seed complete blocks and compressor states, then validate actual skipped prefill")
    parser.add_argument("--cache-layout", choices=("contiguous", "paged"), default="contiguous")
    parser.add_argument("--cache-block-size", type=int, default=128)
    parser.add_argument("--cache-num-blocks", type=int, default=0,
                        help="0 allocates enough physical blocks for max-seq-len")
    parser.add_argument("--paged-storage", choices=("cpp", "torch"), default="cpp")
    parser.add_argument("--compare-paged-storage-in-process", action="store_true",
                        help="compare native payload ownership against explicit Torch paged storage")
    parser.add_argument("--compare-contiguous-in-process", action="store_true",
                        help="compare paged with contiguous storage using the SAME selected chunk schedule")
    parser.add_argument("--dump-layer-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.prefill_chunk_size < 0 or not torch.cuda.is_available():
        raise ValueError("nonnegative warmup and an allocated CUDA device are required")
    if args.cache_block_size <= 0 or args.cache_num_blocks < 0:
        raise ValueError("positive cache block size and nonnegative block count are required")
    if args.compare_contiguous_in_process and args.cache_layout != "paged":
        raise ValueError("contiguous comparison requires explicitly selected paged storage")
    if args.prefix_reuse and args.cache_layout != "paged":
        raise ValueError("prefix reuse requires explicit paged storage")
    if args.compare_paged_storage_in_process and (args.cache_layout != "paged" or args.paged_storage != "cpp"):
        raise ValueError("storage comparison requires explicitly selected native paged storage")
    if args.reference_profile != "published" and (
            args.model_linear_backend != "torch-fixed32" or args.indexer_tie_policy != "index_ascending"
            or args.attention_metadata_policy != "fixed" or args.gemm_backend != "tilelang"):
        raise ValueError("fixed-profile baseline requires matching explicitly selected model arithmetic policies")
    source, checkpoint = args.source_model.resolve(), args.checkpoint.resolve()
    cfg = InferenceConfig.from_directory(source, max_seq_len=args.max_seq_len)
    cfg = replace(cfg, indexer_tie_policy=args.indexer_tie_policy, attention_metadata_policy=args.attention_metadata_policy)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    weight_identity = None
    if any(json.loads(Path(pair[0]).read_text()).get("checkpoint_identity") is not None for pair in args.baseline):
        print("verifying entire checkpoint payload identity", flush=True)
        weight_identity = checkpoint_identity(checkpoint)
    cases = read_baselines(args.baseline, source, checkpoint, args.max_seq_len, tokenizer, cfg.vocab_size,
                           weight_identity, args.reference_profile)
    import tilelang
    import tvm_ffi
    registry = OperatorRegistry(MODEL_CONTRACTS)
    kernel = load_kernel(source / "inference/kernel.py")
    register_tilelang(registry, kernel, tilelang.__version__)
    register_torch_model(registry, backend=args.model_linear_backend,
                         fixed_rows=32 if args.model_linear_backend == "torch-fixed32" else 0)
    selection = {name: "tilelang" for name in CONTRACTS}
    selection.update({name: args.model_linear_backend for name in set(MODEL_CONTRACTS)-set(CONTRACTS)})
    if args.gemm_backend == "deepgemm":
        import deep_gemm
        register_deepgemm(registry, deep_gemm)
        selection.update(fp8_gemm="deepgemm", fp4_gemm="deepgemm")
    ops = registry.bind(selection)
    import fast_hadamard_transform as hadamard
    hadamard.configure_backend("cuda")
    initial_dtype = torch.get_default_dtype()
    model = DeepSeekV4Model(cfg, ops, hadamard.hadamard_transform)
    started = time.perf_counter()
    weights = load_converted_weights(model, checkpoint)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"strict load: {weights['tensor_count']} main tensors, {load_seconds:.3f}s", flush=True)
    cache_pool = storage_reference = None
    native_before = None
    if args.cache_layout == "paged":
        from llaisys.models.deepseek_v4_model.paged import PagedCachePool
        if args.paged_storage == "cpp":
            from llaisys import _C
            native_before = _C.v4_storage_counters()
        count = args.cache_num_blocks or (cfg.max_seq_len + args.cache_block_size - 1) // args.cache_block_size
        cache_pool = PagedCachePool(model, num_blocks=count, block_size=args.cache_block_size,
                                   storage_backend=args.paged_storage, enable_prefix_cache=args.prefix_reuse)
        if args.compare_paged_storage_in_process:
            storage_reference = PagedCachePool(model, num_blocks=count, block_size=args.cache_block_size,
                                              storage_backend="torch")
    results = []
    for index, case in enumerate(cases):
        dump = args.dump_layer_dir / f"case_{index}.safetensors" if args.dump_layer_dir else None
        full = evaluate(model, tokenizer, case, args.warmup, None, 0) if args.compare_full_in_process else None
        contiguous = (evaluate(model, tokenizer, case, args.warmup, None, args.prefill_chunk_size,
                               intermediate_logits=args.intermediate_prefill_logits)
                      if args.compare_contiguous_in_process else None)
        seed = None
        if args.prefix_reuse:
            cache_pool.prefix.clear()
            seed = evaluate(model, tokenizer, case, args.warmup, None, args.prefill_chunk_size, cache_pool,
                            intermediate_logits=args.intermediate_prefill_logits, prefix_mode="publish")
        result = evaluate(model, tokenizer, case, args.warmup, dump, args.prefill_chunk_size, cache_pool,
                          intermediate_logits=args.intermediate_prefill_logits,
                          prefix_mode="reuse" if args.prefix_reuse else "off")
        if args.free_generation:
            result["free_generation"] = evaluate_free_generation(
                model, tokenizer, case, source, args.prefill_chunk_size, cache_pool)
        actual_logits = result.pop("_logits")
        if seed is not None:
            comparison = [compare(actual, expected) for actual, expected in zip(actual_logits, seed.pop("_logits"))]
            expected_hit = (result["input_tokens"] - 1) // cache_pool.block_size * cache_pool.block_size
            result["prefix_reuse_comparison"] = {
                "steps": comparison, "expected_hit_tokens": expected_hit,
                "passed": result["prefix_hit_tokens"] == expected_hit and all(row["exact_equal"] for row in comparison),
                "all_logits_exact": all(row["exact_equal"] for row in comparison), "seed": seed}
        if storage_reference is not None:
            reference = evaluate(model, tokenizer, case, args.warmup, None, args.prefill_chunk_size, storage_reference,
                                 intermediate_logits=args.intermediate_prefill_logits)
            comparison = [compare(actual, expected) for actual, expected in zip(actual_logits, reference.pop("_logits"))]
            result["same_schedule_torch_storage"] = {
                "steps": comparison, "passed": all(row["exact_equal"] for row in comparison),
                "all_logits_exact": all(row["exact_equal"] for row in comparison),
                "torch_storage_vs_saved_baseline": reference["steps"]}
        if full is not None:
            expected_logits = full.pop("_logits")
            comparison = [compare(actual, expected) for actual, expected in zip(actual_logits, expected_logits)]
            passed = all(row["finite"] and row["argmax_equal"] and row["relative_l2"] <= 0.01
                         and row["cosine_similarity"] >= 0.999 for row in comparison)
            result["same_profile_full"] = {"steps": comparison, "passed": passed,
                                           "all_logits_exact": all(row["exact_equal"] for row in comparison),
                                           "full_vs_saved_published_baseline": full["steps"]}
        if contiguous is not None:
            comparison = [compare(actual, expected) for actual, expected in zip(actual_logits, contiguous.pop("_logits"))]
            passed = all(row["finite"] and row["argmax_equal"] and row["relative_l2"] <= 0.01
                         and row["cosine_similarity"] >= 0.999 for row in comparison)
            result["same_schedule_contiguous"] = {
                "steps": comparison, "passed": passed,
                "all_logits_exact": all(row["exact_equal"] for row in comparison),
                "contiguous_vs_saved_published_baseline": contiguous["steps"]}
        results.append(result)
        print(f"case {index}: input={result['input_tokens']} exact={result['all_logits_exact']} "
              f"passed={result['numerical_gate_passed']} "
              f"same_profile_full={result.get('same_profile_full', {}).get('passed')}", flush=True)
    root = Path(__file__).resolve().parents[2]
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
    paths = [Path(__file__), source / "config.json", source / "inference/config.json", source / "inference/kernel.py",
             root / "python/llaisys/models/deepseek_v4.py", root / "python/llaisys/models/deepseek_v4_backends.py",
             root / "python/llaisys/models/deepseek_v4_model_ops.py",
             root / "python/llaisys/models/deepseek_v4_evidence.py",
             root / "python/llaisys/models/deepseek_v4_reference.py", Path(hadamard.__file__),
             Path(hadamard.backend_report()["extension"]),
             *sorted((root / "python/llaisys/models/deepseek_v4_model").glob("*.py"))]
    if cache_pool is not None:
        from llaisys import _C
        paths.extend((Path(_C.__file__), root / "python/bindings/cache.cpp"))
        if cache_pool.native_storage is not None:
            paths.extend((root / "python/bindings/paged_storage.cpp", root / "src/models/deepseek_v4/cache_layout.hpp",
                          root / "src/core/cache/paged_cache_storage.cpp", root / "src/core/cache/paged_cache_storage.hpp",
                          root / "python/llaisys/libllaisys/libllaisys.so"))
    if args.reference_profile != "published":
        paths.append(Path(__file__).with_name("published_fixed_profile.py"))
    unchanged_dtype = torch.get_default_dtype() == initial_dtype
    if weight_identity is not None:
        verify_checkpoint_unchanged(checkpoint, weight_identity)
    result = {
        "executor": "llaisys-independent-python-reference", "published_model_imported": False,
        "free_generation_tested": args.free_generation,
        "reference_profile": args.reference_profile,
        "unmodified_published_baseline_tested": args.reference_profile == "published",
        "model_config": asdict(cfg), "weights": weights, "weight_load_seconds": load_seconds,
        "checkpoint_identity": weight_identity,
        "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__, "cuda": torch.version.cuda, "tilelang": tilelang.__version__,
        "tvm_ffi": tvm_ffi.__version__, "global_default_dtype": str(initial_dtype),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "global_default_dtype_unchanged": unchanged_dtype, "operator_backends": ops.report(),
        "hadamard": hadamard.backend_report(), "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_scope": "Torch allocator only; excludes explicitly reported native cache allocation",
        "native_cache_allocation_bytes": (cache_pool.report()["payload_bytes"]
                                          if cache_pool is not None and cache_pool.native_storage is not None else 0),
        "cache_layout": args.cache_layout, "paged_pool": cache_pool.report() if cache_pool is not None else None,
        "measurement": {"kind": "correctness_step_latency_not_serving_ttft_tpot", "batch_size": 1,
                        "concurrency": 1, "warmup_per_case": args.warmup,
                        "warmup_kind": "selected cache layout and prefill schedule, followed by one decode",
                        "full_comparison_warmup": ("contiguous full prefill and one decode"
                                                   if args.compare_full_in_process else None),
                        "contiguous_comparison_warmup": ("contiguous storage, selected prefill schedule and one decode"
                                                         if args.compare_contiguous_in_process else None),
                        "repeats": 1, "cuda_graph": False, "prefix_cache": args.prefix_reuse, "paged_cache": cache_pool is not None,
                        "prefix_lookup_publish_in_step_timing": False,
                        "fallback": False, "layer_capture_in_timing": bool(args.dump_layer_dir),
                        "additional_same_profile_full_run": args.compare_full_in_process,
                        "additional_same_schedule_contiguous_run": args.compare_contiguous_in_process},
        "torch_storage_comparison": storage_reference.report() if storage_reference is not None else None,
        "prefill_chunk_size": args.prefill_chunk_size,
        "intermediate_prefill_logits": args.intermediate_prefill_logits,
        "baseline_identity_limit": (None if all(case["checkpoint_payload_verified"] for case in cases) else
                                    "Some legacy baselines do not bind checkpoint payload hashes; "
                                    "see checkpoint_payload_verified for each case."),
        "provenance": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                       "worktree_dirty": bool(git("status", "--porcelain")), "command": sys.argv,
                       "file_sha256": {str(path): sha256(path) for path in paths}},
        "numerical_gate": {"min_cosine": 0.999, "max_relative_l2": 0.01, "require_same_argmax": True},
        "cases": results, "all_passed": unchanged_dtype and all(
            c["numerical_gate_passed"] and c["expected_match"] is not False
            and c.get("same_profile_full", {}).get("passed", True)
            and c.get("same_schedule_torch_storage", {}).get("passed", True)
            and c.get("prefix_reuse_comparison", {}).get("passed", True)
            and c.get("free_generation", {}).get("matches_golden_ids", True)
            and c.get("free_generation", {}).get("expected_match") is not False
            and c.get("same_schedule_contiguous", {}).get("passed", True) for c in results),
        "all_same_profile_comparisons_pass": (all(c["same_profile_full"]["passed"] for c in results)
                                             if args.compare_full_in_process else None),
        "all_logits_exact": all(c["all_logits_exact"] for c in results),
    }
    if native_before is not None:
        before_release = _C.v4_storage_counters()
        cache_pool = None
        gc.collect()
        after_release = _C.v4_storage_counters()
        returned = after_release == native_before
        result["native_storage_lifecycle"] = {
            "before_pool": native_before, "before_release": before_release,
            "after_release": after_release, "returned_to_baseline": returned}
        result["all_passed"] = result["all_passed"] and returned
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": result["all_passed"], "all_logits_exact": result["all_logits_exact"],
                      "report": str(args.output)}), flush=True)
    if not result["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

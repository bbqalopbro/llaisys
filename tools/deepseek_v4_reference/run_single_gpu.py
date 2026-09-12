"""Run the published DeepSeek-V4 model on one GPU and emit accuracy evidence.

This is a bring-up/reference runner, not the production llaisys serving path.
It validates converted real weights, full-model prefill/decode, and deterministic
greedy output while the native DeepSeek backend is being implemented.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import types

import torch
from safetensors.torch import load_file, load_model, save_file
from transformers import AutoTokenizer

from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
from llaisys.models.deepseek_v4_model.chat import ChatCodec, check_chat_answer


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reset_runtime_state(model: torch.nn.Module) -> None:
    """Reset only mutable KV/compressor state; keep RoPE and parameters intact."""
    for name, buffer in model.named_buffers():
        if name.endswith("score_state"):
            buffer.fill_(float("-inf"))
        elif name.endswith(("kv_cache", "kv_state")):
            buffer.zero_()


@torch.inference_mode()
def _evaluate_case(model, tokenizer, prompt: str, max_new_tokens: int,
                   max_seq_len: int, layer_diagnostics: bool = False,
                   dump_logits: Path | None = None, golden_logits: Path | None = None,
                   token_replay: bool = True, encoded_ids=None, stop_on_eos=False) -> dict:
    input_ids = (tokenizer.encode(prompt, return_tensors="pt") if encoded_ids is None else
                 torch.tensor([encoded_ids], dtype=torch.int64, device="cpu")).cuda()
    if input_ids.shape[1] + max_new_tokens > max_seq_len:
        raise ValueError(f"prompt exceeds configured sequence length: {prompt!r}")
    golden = load_file(str(golden_logits)) if golden_logits is not None else None
    if golden is not None and (not torch.equal(golden["input_ids"], input_ids.cpu())
                               or not 0 < golden["logits"].shape[0] <= max_new_tokens
                               or (not stop_on_eos and golden["logits"].shape[0] != max_new_tokens)):
        raise ValueError("golden inputs or output length do not match this run")

    captures: dict[str, dict[int, torch.Tensor]] = {"full": {}, "replay": {}}
    capture_mode = {"value": "full"}
    hooks = []
    if layer_diagnostics:
        for layer_id, layer in enumerate(model.layers):
            def capture(_module, _inputs, output, *, index=layer_id):
                captures[capture_mode["value"]][index] = (
                    output[:, -1].detach().float().clone()
                )
            hooks.append(layer.register_forward_hook(capture))

    _reset_runtime_state(model)
    _, full_logits, _ = model(input_ids, 0)
    full_logits = full_logits.float().clone()
    replay_logits = None
    if token_replay:
        _reset_runtime_state(model)
        capture_mode["value"] = "replay"
        for position in range(input_ids.shape[1]):
            _, replay_logits, _ = model(input_ids[:, position : position + 1], position)
        replay_logits = replay_logits.float()
        difference = (full_logits - replay_logits).abs()
    for hook in hooks:
        hook.remove()

    diagnostic_results = []
    if layer_diagnostics:
        for layer_id in range(len(model.layers)):
            full_hidden = captures["full"][layer_id].flatten()
            replay_hidden = captures["replay"][layer_id].flatten()
            hidden_difference = (full_hidden - replay_hidden).abs()
            diagnostic_results.append({
                "layer": layer_id,
                "max_abs_error": float(hidden_difference.max().item()),
                "mean_abs_error": float(hidden_difference.mean().item()),
                "cosine_similarity": float(
                    torch.nn.functional.cosine_similarity(
                        full_hidden, replay_hidden, dim=0
                    ).item()
                ),
            })

    _reset_runtime_state(model)
    tokens = input_ids
    steps = []
    saved_logits = []
    predicted_ids = []
    start_pos = 0
    steps_to_run = golden["logits"].shape[0] if golden is not None else max_new_tokens
    for step_index in range(steps_to_run):
        current = tokens[:, start_pos:]
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        next_token, logits, _ = model(current, start_pos)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        selected = logits.argmax(dim=-1)
        if not torch.equal(next_token, selected):
            raise AssertionError("temperature=0 output disagrees with logits argmax")
        steps.append({
            "start_pos": start_pos,
            "input_tokens": current.shape[1],
            "next_token": int(next_token.item()),
            "max_logit": float(logits.max().item()),
            "latency_ms": elapsed * 1000,
        })
        predicted_ids.append(int(next_token.item()))
        if dump_logits is not None or golden is not None:
            saved_logits.append(logits.detach().float().cpu())
        fed_token = (golden["generated_ids"][step_index].to(device=tokens.device).reshape(1)
                     if golden is not None else next_token)
        tokens = torch.cat((tokens, fed_token[:, None]), dim=1)
        start_pos = tokens.shape[1] - 1
        if stop_on_eos and golden is None and predicted_ids[-1] == tokenizer.eos_token_id:
            break

    golden_comparison = []
    if golden is not None:
        for step, (actual, expected) in enumerate(zip(saved_logits, golden["logits"])):
            error = actual - expected
            golden_comparison.append({
                "step": step, "phase": "prefill" if step == 0 else "decode",
                "argmax_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
                "max_abs_error": error.abs().max().item(),
                "relative_l2": (error.norm() / expected.norm().clamp_min(1e-8)).item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(actual, expected, dim=-1).item(),
            })
    if dump_logits is not None:
        dump_logits.parent.mkdir(parents=True, exist_ok=True)
        save_file({"input_ids": input_ids.cpu(),
                   "generated_ids": torch.tensor(predicted_ids, dtype=torch.int64, device="cpu"),
                   "logits": torch.stack(saved_logits)}, str(dump_logits))

    full_argmax = int(full_logits.argmax(dim=-1).item())
    replay_argmax = int(replay_logits.argmax(dim=-1).item()) if token_replay else None
    return {
        "prompt": prompt,
        "prompt_tokens": input_ids.shape[1],
        "token_replay_tested": token_replay,
        "accuracy": {
            "full_prefill_argmax": full_argmax,
            "token_replay_argmax": replay_argmax,
            "argmax_equal": full_argmax == replay_argmax,
            "logits_max_abs_error": float(difference.max().item()),
            "logits_mean_abs_error": float(difference.mean().item()),
            "logits_cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    full_logits, replay_logits, dim=-1
                ).item()
            ),
        } if token_replay else None,
        "layer_diagnostics": diagnostic_results if layer_diagnostics else None,
        "generated_ids": predicted_ids,
        "max_new_tokens": max_new_tokens,
        "stop_on_eos": stop_on_eos,
        "finish_reason": ("stop" if stop_on_eos and predicted_ids[-1] == tokenizer.eos_token_id else "length"),
        "generated_text": tokenizer.decode(predicted_ids),
        "teacher_forced_ids": tokens[0, input_ids.shape[1] :].tolist() if golden is not None else None,
        "golden_comparison": golden_comparison if golden is not None else None,
        "steps": steps,
    }


@torch.inference_mode()
def _compare_component(model, tokenizer, published, original_forward,
                       native_forward, prompt, generated_ids, component="Compressor"):
    """Compare one changed component with identical weights, inputs and GEMM.

    Decode is teacher-forced using the native run's token IDs, so a change in
    sampling cannot obscure which logits are being compared.
    """
    tokens = tokenizer.encode(prompt, return_tensors="pt").cuda()
    traces = {}
    try:
        for name, forward in (("published", original_forward), ("native", native_forward)):
            getattr(published, component).forward = forward
            _reset_runtime_state(model)
            position = 0
            current = tokens
            outputs = []
            for index in range(max(1, len(generated_ids))):
                _, logits, _ = model(current, position)
                outputs.append(logits.float().clone())
                position += current.shape[1]
                if index < len(generated_ids):
                    current = torch.tensor([[generated_ids[index]]], dtype=tokens.dtype, device=tokens.device)
            traces[name] = outputs
    finally:
        getattr(published, component).forward = native_forward
    rows = []
    for index, (expected, actual) in enumerate(zip(traces["published"], traces["native"])):
        difference = (actual - expected).abs()
        rows.append({
            "phase": "prefill" if index == 0 else "decode",
            "step": index,
            "argmax_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(actual, expected, dim=-1).item()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--converted-model", type=Path, required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument(
        "--cases-file", type=Path,
        help="optional JSON array of {prompt, expected_contains} smoke cases",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gemm-backend", choices=("tilelang", "deepgemm"))
    parser.add_argument("--hadamard-backend", choices=("cuda", "torch-reference"), default="cuda")
    parser.add_argument("--dump-logits-dir", type=Path)
    parser.add_argument("--compare-logits-dir", type=Path)
    parser.add_argument("--reference-mode", choices=("full-and-replay", "full-only"), default="full-and-replay",
                        help="full-only captures full-prefill/decode golden; does NOT test token replay")
    parser.add_argument("--hash-checkpoint", action="store_true",
                        help="stream the entire converted checkpoint through SHA256 before loading")
    parser.add_argument("--reference-profile", choices=("published", "fixed32-reference-v1"), default="published",
                        help="explicit alternative arithmetic policy; never labeled as the unmodified published baseline")
    parser.add_argument(
        "--layer-diagnostics", action="store_true",
        help="record final-token full-prefill versus replay error after every layer",
    )
    parser.add_argument(
        "--native-structural-library", "--native-sparse-attention-library",
        dest="native_structural_library", type=Path,
        help="explicitly use llaisys BF16 sparse attention and FP32 HC operations",
    )
    parser.add_argument(
        "--torch-reference-structural-ops",
        action="store_true",
        help="explicitly replace sparse attention and HC with PyTorch oracles",
    )
    parser.add_argument(
        "--dequantized-pytorch-linear",
        action="store_true",
        help="explicitly dequantize FP8/FP4 weights and use PyTorch linear",
    )
    parser.add_argument(
        "--native-quantized-linear", action="store_true",
        help="use the explicit fused correctness GEMM for FP8/FP4 weights",
    )
    parser.add_argument(
        "--operator-library-linear", action="store_true",
        help="decode quantized weights and execute GEMM through cuBLAS",
    )
    parser.add_argument(
        "--native-compressor", action="store_true",
        help="explicitly use CUDA projected pooling and streaming compressor state",
    )
    parser.add_argument(
        "--compare-compressor", action="store_true",
        help="compare published and native compressor with otherwise identical execution",
    )
    parser.add_argument(
        "--native-indexer", action="store_true",
        help="use cuBLAS BF16 Indexer scores and CUB stable causal Top-K",
    )
    parser.add_argument(
        "--compare-indexer", action="store_true",
        help="compare published and native Indexer with otherwise identical execution",
    )
    args = parser.parse_args()

    if args.reference_mode == "full-only" and args.layer_diagnostics:
        raise ValueError("full/replay layer diagnostics require full-and-replay mode")
    if args.reference_profile != "published" and (
            args.gemm_backend not in (None, "tilelang") or args.hadamard_backend != "cuda"
            or args.torch_reference_structural_ops or args.dequantized_pytorch_linear
            or args.native_structural_library or args.native_compressor or args.native_indexer):
        raise ValueError("fixed-profile oracle requires TileLang kernels and CUDA Hadamard without legacy replacements")
    checkpoint_path = args.converted_model / "model0-mp1.safetensors"
    weight_identity = None
    if args.hash_checkpoint:
        print("hashing entire checkpoint payload", file=sys.stderr, flush=True)
        weight_identity = checkpoint_identity(checkpoint_path)
        print(f"checkpoint SHA256: {weight_identity['sha256']}", file=sys.stderr, flush=True)

    if args.native_structural_library and not args.torch_reference_structural_ops:
        raise ValueError(
            "native sparse attention requires --torch-reference-structural-ops "
            "for the remaining unsupported structural operations"
        )
    if (args.native_quantized_linear or args.operator_library_linear) and not args.native_structural_library:
        raise ValueError(
            "native linear backends require --native-structural-library"
        )
    if args.native_quantized_linear and args.operator_library_linear:
        raise ValueError("select only one native linear backend")
    if args.native_compressor and not args.native_structural_library:
        raise ValueError("--native-compressor requires --native-structural-library")
    if args.compare_compressor and not args.native_compressor:
        raise ValueError("--compare-compressor requires --native-compressor")
    if args.native_indexer and not args.native_structural_library:
        raise ValueError("--native-indexer requires --native-structural-library")
    if args.compare_indexer and not args.native_indexer:
        raise ValueError("--compare-indexer requires --native-indexer")

    if args.gemm_backend and args.torch_reference_structural_ops:
        raise ValueError("--gemm-backend selects the published structural path; do not combine with legacy reference flags")
    if args.dump_logits_dir and args.compare_logits_dir:
        raise ValueError("dump a baseline or compare with it, not both")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    hadamard = _load_module("fast_hadamard_transform", Path(__file__).with_name("fast_hadamard_transform.py"))
    hadamard.configure_backend(args.hadamard_backend)
    inference = args.source_model / "inference"
    sys.path.insert(0, str(inference))
    if args.dequantized_pytorch_linear and args.torch_reference_structural_ops:
        kernel = types.ModuleType("kernel")
        for name in (
            "act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
            "sparse_attn", "hc_split_sinkhorn",
        ):
            setattr(kernel, name, lambda *unused, **unused_kw: None)
        sys.modules["kernel"] = kernel
    else:
        kernel = _load_module("kernel", inference / "kernel.py")
    profile_adapter = None
    if args.reference_profile == "published":
        model_source = _load_module("deepseek_v4_published_model", inference / "model.py")
    else:
        profile_adapter = _load_module("deepseek_v4_published_policy", Path(__file__).with_name("published_fixed_profile.py"))
        model_source = profile_adapter.load_profiled_model("deepseek_v4_published_model", inference / "model.py")
    backend = "published-tilelang" + ("-fixed32-reference-v1" if profile_adapter else "")
    operator_dispatch = None
    if not args.torch_reference_structural_ops:
        import tilelang
        dispatch_module = _load_module("deepseek_v4_dispatch", Path(__file__).parents[2]
                                      / "python/llaisys/models/deepseek_v4_backends.py")
        registry = dispatch_module.OperatorRegistry()
        dispatch_module.register_tilelang(registry, kernel, tilelang.__version__)
        selection = {name: "tilelang" for name in dispatch_module.CONTRACTS}
        if args.gemm_backend == "deepgemm":
            import deep_gemm
            dispatch_module.register_deepgemm(registry, deep_gemm)
            selection.update(fp8_gemm="deepgemm", fp4_gemm="deepgemm")
            backend = "published-tilelang+deepgemm-w8a8-w4a8"
        operator_dispatch = registry.bind(selection)
        operator_dispatch.install_on(model_source)
    native_ops = None
    if args.torch_reference_structural_ops:
        reference_path = (
            Path(__file__).parents[2]
            / "python"
            / "llaisys"
            / "models"
            / "deepseek_v4_reference.py"
        )
        reference = _load_module("llaisys_deepseek_v4_e2e_reference", reference_path)
        if args.native_structural_library:
            native_path = (
                Path(__file__).parents[2]
                / "python" / "llaisys" / "models" / "deepseek_v4_native.py"
            )
            native_module = _load_module(
                "llaisys_deepseek_v4_e2e_native", native_path
            )
            native_ops = native_module.DeepSeekV4NativeReferenceOps(
                args.native_structural_library
            )
            model_source.sparse_attn = native_ops.sparse_attention
        else:
            model_source.sparse_attn = reference.sparse_latent_attention
        if args.native_structural_library:
            model_source.hc_split_sinkhorn = native_ops.hyperconnection_split

            published_gate_forward = model_source.Gate.forward

            def native_gate_forward(self, x, input_ids=None):
                if self.hash:
                    return published_gate_forward(self, x, input_ids)
                if self.score_func != "sqrtsoftplus":
                    raise RuntimeError(
                        "native router only supports explicit sqrtsoftplus routing"
                    )
                logits = model_source.linear(
                    x.float(), self.weight.float()
                ).contiguous()
                return native_ops.route_sqrt_softplus(
                    logits, self.bias, self.topk, self.route_scale
                )

            model_source.Gate.forward = native_gate_forward
        else:
            model_source.hc_split_sinkhorn = (
                lambda mixes, scale, base, hc, iterations, eps:
                reference.hyperconnection_split(
                    mixes, scale, base, hc, iterations=iterations, eps=eps
                )
            )
        backend = (
            "tilelang-gemm+native-bf16-sparse-attention+native-hc-router"
            if args.native_structural_library
            else "tilelang-gemm+torch-sparse-attention-hc-reference"
        )

    with (inference / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["max_batch_size"] = 1
    config["max_seq_len"] = args.max_seq_len
    config["temperature"] = 0.0
    model_args = model_source.ModelArgs(**config)

    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.device("cuda"):
        model = model_source.Transformer(model_args)
    missing_weights, unexpected_weights = load_model(
        model,
        str(args.converted_model / "model0-mp1.safetensors"),
        strict=False,
    )
    if missing_weights or unexpected_weights:
        raise ValueError(f"published model weight mismatch: missing={sorted(missing_weights)[:8]}, "
                         f"unexpected={sorted(unexpected_weights)[:8]}")
    if args.dequantized_pytorch_linear:
        if not args.torch_reference_structural_ops:
            raise ValueError(
                "--dequantized-pytorch-linear requires the explicit structural reference flag"
            )
        dense_cache: dict[int, torch.Tensor] = {}

        def reference_linear(x, weight, bias=None):
            if bias is not None:
                raise ValueError("published DeepSeek-V4 linear weights are bias-free")
            if weight.dtype == torch.float4_e2m1fn_x2:
                quantized_x = x.clone()
                if args.native_structural_library:
                    native_ops.quantize_activation_(
                        quantized_x, 128, mode="fp8", power_of_two_scale=True
                    )
                else:
                    quantized_x = reference.simulate_fp8_activation_quant(
                        quantized_x, 128, power_of_two_scale=True
                    )
                if args.native_quantized_linear or args.operator_library_linear:
                    return native_ops.quantized_linear(
                        quantized_x, weight, weight.scale, mode="fp4",
                        backend="cublas" if args.operator_library_linear else "reference",
                    )
                decoded = reference.dequantize_fp4_e2m1(
                    weight.view(torch.uint8), weight.scale
                ).to(torch.bfloat16)
            elif weight.dtype == torch.float8_e4m3fn:
                quantized_x = x.clone()
                if args.native_structural_library:
                    native_ops.quantize_activation_(
                        quantized_x, 128, mode="fp8", power_of_two_scale=True
                    )
                else:
                    quantized_x = reference.simulate_fp8_activation_quant(
                        quantized_x, 128, power_of_two_scale=True
                    )
                if args.native_quantized_linear or args.operator_library_linear:
                    return native_ops.quantized_linear(
                        quantized_x, weight, weight.scale, mode="fp8",
                        backend="cublas" if args.operator_library_linear else "reference",
                    )
                key = id(weight)
                decoded = dense_cache.get(key)
                if decoded is None:
                    decoded = reference.dequantize_fp8_blocks(
                        weight, weight.scale
                    ).to(torch.bfloat16)
                    dense_cache[key] = decoded
            else:
                decoded = weight
                quantized_x = x
            return torch.nn.functional.linear(
                quantized_x.to(decoded.dtype), decoded
            )

        def reference_act_quant(x, block_size=128, scale_fmt=None,
                                scale_dtype=torch.float32, inplace=False):
            del scale_dtype
            if not inplace:
                raise RuntimeError(
                    "non-inplace act_quant must not be reached by the dequantized backend"
                )
            if args.native_structural_library:
                return native_ops.quantize_activation_(
                    x, block_size, mode="fp8",
                    power_of_two_scale=scale_fmt is not None,
                )
            value = reference.simulate_fp8_activation_quant(
                x, block_size, power_of_two_scale=scale_fmt is not None
            )
            x.copy_(value)
            return x

        def reference_fp4_quant(x, block_size=32, inplace=False):
            if not inplace:
                raise RuntimeError(
                    "non-inplace fp4 quant must not be reached by the dequantized backend"
                )
            if args.native_structural_library:
                return native_ops.quantize_activation_(
                    x, block_size, mode="fp4", power_of_two_scale=True
                )
            value = reference.simulate_fp4_activation_quant(x, block_size)
            x.copy_(value)
            return x

        model_source.linear = reference_linear
        model_source.act_quant = reference_act_quant
        model_source.fp4_act_quant = reference_fp4_quant
        backend = (
            "cublas-fp32-gemm+native-quant-decode-attention-hc-router-actquant"
            if args.operator_library_linear
            else "native-fp8-fp4-linear+native-bf16-attention-hc-router-actquant"
            if args.native_quantized_linear
            else "dequantized-pytorch-linear+native-bf16-attention-hc-router-actquant"
            if args.native_structural_library
            else "dequantized-pytorch-reference"
        )
    if args.native_compressor:
        published_compressor_forward = model_source.Compressor.forward
        model_source.Compressor.forward = native_module.make_native_compressor_forward(
            model_source, native_ops
        )
        backend += "+native-streaming-compressor"
    if args.native_indexer:
        published_indexer_forward = model_source.Indexer.forward
        model_source.Indexer.forward = native_module.make_native_indexer_forward(model_source, native_ops)
        backend += "+cublas-bf16-indexer+cub-stable-topk"
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    torch.set_default_device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.converted_model)
    if args.cases_file:
        cases = json.loads(args.cases_file.read_text(encoding="utf-8"))
    else:
        cases = [{"prompt": args.prompt}]
    case_results = []
    chat_codec = None
    for case_index, case in enumerate(cases):
        chat = None
        if "messages" in case:
            if "prompt" in case:
                raise ValueError("a case must contain messages or a raw prompt, not both")
            if chat_codec is None:
                chat_codec = ChatCodec(args.source_model, tokenizer)
            options = {name: case.get(name, default) for name, default in
                       (("thinking_mode", "chat"), ("drop_thinking", True), ("reasoning_effort", "low"))}
            encoded = chat_codec.encode(case["messages"], **options)
            prompt, encoded_ids = encoded["text"], encoded["input_ids"]
            chat = {"messages": case["messages"], **options, "codec": chat_codec.report()}
        else:
            prompt, encoded_ids = case["prompt"], None
        case_result = _evaluate_case(
            model, tokenizer, prompt, args.max_new_tokens,
            args.max_seq_len, args.layer_diagnostics,
            args.dump_logits_dir / f"case_{case_index}.safetensors" if args.dump_logits_dir else None,
            args.compare_logits_dir / f"case_{case_index}.safetensors" if args.compare_logits_dir else None,
            token_replay=args.reference_mode == "full-and-replay",
            encoded_ids=encoded_ids, stop_on_eos=chat is not None,
        )
        case_result["case_id"] = case.get("id", str(case_index))
        case_result["input_format"] = "chat" if chat else "raw-completion"
        case_result["chat"] = chat
        expected = case.get("expected_contains")
        case_result["expected_contains"] = expected
        case_result["expected_match"] = (
            expected in case_result["generated_text"] if expected else None
        )
        if chat is not None:
            case_result["completion"] = chat_codec.decode_completion(
                case_result["generated_ids"], thinking_mode=chat["thinking_mode"],
                finish_reason=case_result["finish_reason"])
            case_result["expected_exact"] = case.get("expected_exact")
            case_result["expected_match"] = check_chat_answer(case_result["completion"], case)
        if args.compare_compressor:
            case_result["compressor_comparison"] = _compare_component(
                model, tokenizer, model_source, published_compressor_forward,
                model_source.Compressor.forward, prompt,
                case_result["generated_ids"],
            )
        if args.compare_indexer:
            case_result["indexer_comparison"] = _compare_component(
                model, tokenizer, model_source, published_indexer_forward,
                model_source.Indexer.forward, prompt,
                case_result["generated_ids"], component="Indexer",
            )
        case_results.append(case_result)
        print(f"completed case: {case_result['prompt_tokens']} tokens", file=sys.stderr, flush=True)

    root = Path(__file__).resolve().parents[2]
    def git_output(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()

    evidence_paths = [Path(__file__), inference / "model.py", inference / "kernel.py",
                      root / "python/llaisys/models/deepseek_v4_evidence.py",
                      inference / "config.json", Path(hadamard.__file__)]
    if args.cases_file:
        evidence_paths.append(args.cases_file)
    if chat_codec is not None:
        evidence_paths.extend((chat_codec.path, root / "python/llaisys/models/deepseek_v4_model/chat.py"))
    if operator_dispatch is not None:
        evidence_paths.append(Path(dispatch_module.__file__))
    if profile_adapter is not None:
        evidence_paths.append(Path(profile_adapter.__file__))
    hadamard_info = hadamard.backend_report()
    if hadamard_info["extension"]:
        evidence_paths.append(Path(hadamard_info["extension"]))
    if args.native_structural_library:
        evidence_paths.extend((args.native_structural_library, native_path))
    if weight_identity is not None:
        verify_checkpoint_unchanged(checkpoint_path, weight_identity)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "backend": backend,
        "reference_mode": args.reference_mode,
        "reference_profile": args.reference_profile,
        "profile_adapter": getattr(model_source, "_reference_profile", None),
        "profile_operation_counts": dict(getattr(model_source, "_reference_profile_counts", {})),
        "checkpoint_identity": weight_identity,
        "weight_key_validation": {"missing": sorted(missing_weights), "unexpected": sorted(unexpected_weights)},
        "golden_file_sha256": ({f"case_{index}.safetensors": hashlib.sha256(
            (args.dump_logits_dir / f"case_{index}.safetensors").read_bytes()).hexdigest()
            for index in range(len(case_results))} if args.dump_logits_dir else None),
        "operator_backends": operator_dispatch.report() if operator_dispatch else None,
        "hadamard": hadamard_info,
        "tilelang": getattr(sys.modules.get("tilelang"), "__version__", None),
        "tvm_ffi": getattr(sys.modules.get("tvm_ffi"), "__version__", None),
        "indexer_tie_policy": ("score descending, candidate index ascending"
                               if args.native_indexer or profile_adapter else "published torch.topk"),
        "provenance": {
            "commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "worktree_dirty": bool(git_output("status", "--porcelain")),
            "file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence_paths},
            "command": sys.argv,
        },
        "measurement": {
            "kind": "correctness_runner_step_latency_not_serving_ttft_tpot",
            "batch_size": 1, "concurrency": 1, "cuda_graph": False,
            "prefix_cache": False,
            "warmup": ("one full prefill and one token replay per case" if args.reference_mode == "full-and-replay"
                       else "one full prefill per case; no token replay"),
            "timed_repeats": 1, "fallback": hadamard_info["fallback"],
            "weights": "FP8 E4M3 dense, FP4 E2M1 experts, E8M0 scales",
            "activation_dtype": "BF16 with explicit QAT boundaries",
        },
        "source_model": str(args.source_model),
        "converted_model": str(args.converted_model),
        "weight_load_seconds": load_seconds,
        "cases": case_results,
        "all_argmax_equal": all(
            case["accuracy"]["argmax_equal"] for case in case_results
        ) if args.reference_mode == "full-and-replay" else None,
        "all_expected_match": all(
            case["expected_match"] is not False for case in case_results
        ),
        "all_golden_comparisons_pass": (
            all(row["argmax_equal"] and row["cosine_similarity"] >= 0.999 and row["relative_l2"] <= 0.01
                for case in case_results for row in case["golden_comparison"])
            if args.compare_logits_dir else None
        ),
        "all_compressor_comparisons_pass": (
            all(row["argmax_equal"] and row["cosine_similarity"] >= 0.999
                for case in case_results for row in case["compressor_comparison"])
            if args.compare_compressor else None
        ),
        "all_indexer_comparisons_pass": (
            all(row["argmax_equal"] and row["cosine_similarity"] >= 0.999
                for case in case_results for row in case["indexer_comparison"])
            if args.compare_indexer else None
        ),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "native_operation_counts": (
            dict(native_ops.operation_counts) if native_ops is not None else None
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if (result["all_argmax_equal"] is False or not result["all_expected_match"]
            or result["all_compressor_comparisons_pass"] is False
            or result["all_indexer_comparisons_pass"] is False
            or result["all_golden_comparisons_pass"] is False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Real-checkpoint native model through the existing threaded InferenceEngine.

Golden logits are observed only, never fed back. Different prompts occupy two
serial native slots. Observer-enabled runs are correctness, not TTFT/TPOT data.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoTokenizer

from llaisys.models.deepseek_v4_evidence import checkpoint_identity, verify_checkpoint_unchanged
from llaisys.models.deepseek_v4_model.chat import ChatCodec
from llaisys.models.deepseek_v4_model.config import InferenceConfig
from llaisys.models.deepseek_v4_model.native_session import load_native_session
from llaisys.models.deepseek_v4_model.native_batch import DeepSeekV4NativeServingModel
from server.engine import InferenceEngine, SamplingParams, RequestStatus
from run_independent import read_baselines, sha256


def make_engine(facade, capacity):
    return InferenceEngine(facade, max_batch_size=2, max_seq_per_slot=capacity,
                           prefill_chunk_size=capacity, enable_block_prefix_cache=False, block_watermark=0)


async def wave(session, cases, eos, capacity):
    observed = [0] * len(cases)
    def observe(stage, slot, position, tokens, values):
        case, step = cases[slot], observed[slot]
        inputs = case["tensors"]["input_ids"][0].tolist()
        expected = case["tensors"]["logits"][step]
        actual = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(1, -1)
        if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            raise AssertionError(f"native serving logits mismatch: {case['case_id']}/{step}")
        if position != (0 if step == 0 else len(inputs) + step - 1):
            raise AssertionError("native serving position mismatch")
        if tokens != (inputs if step == 0 else [int(case["tensors"]["generated_ids"][step - 1])]):
            raise AssertionError("native serving actual input history diverged")
        observed[slot] += 1
    facade = DeepSeekV4NativeServingModel(session, eos_id=eos, observer=observe)
    engine = make_engine(facade, capacity)
    requests = [engine.submit(case["tensors"]["input_ids"][0].tolist(),
                 SamplingParams(temperature=0, top_k=1, top_p=1, max_tokens=case["tensors"]["generated_ids"].numel()),
                 case["case_id"], stream=True) for case in cases]
    streams = [[] for _ in requests]
    async def consume(slot, req):
        async for token in req.stream_tokens():
            streams[slot].append(token)
    consumers = [asyncio.create_task(consume(i, req)) for i, req in enumerate(requests)]
    try:
        engine.start()
        outputs = await asyncio.wait_for(asyncio.gather(*(r.future for r in requests)), 900)
        await asyncio.wait_for(asyncio.gather(*consumers), 10)
    finally:
        await asyncio.to_thread(engine.stop, 120)
        for consumer in consumers:
            if not consumer.done(): consumer.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)
    expected_ids = [c["tensors"]["generated_ids"].tolist() for c in cases]
    if outputs != expected_ids or streams != outputs or observed != [len(v) for v in expected_ids]:
        raise AssertionError("native serving token/stream/step mismatch")
    if any(r.status != RequestStatus.DONE for r in requests) or session.info()["active_slots"]:
        raise AssertionError("native serving completion leaked state")
    if engine.queue.active_count or engine.queue.waiting_count:
        raise AssertionError("native serving queue not drained")
    return dict(cases=[dict(case_id=c["case_id"], input_tokens=c["tensors"]["input_ids"].shape[1],
                generated_ids=outputs[i], output_tokens=len(outputs[i]), logits_byte_comparisons=observed[i],
                golden_file=c["golden_file"], golden_sha256=c["golden_sha256"],
                baseline_report=c["baseline_report"], baseline_report_sha256=c["baseline_report_sha256"])
                for i, c in enumerate(cases)], runtime=facade.last_report, engine=engine.get_stats(),
                streamed_ids_equal=True, logits_exact=True, all_passed=True)


async def cancellation_and_recovery(session, case, eos, capacity):
    mode, holder = ["cancel"], {}
    def observe(stage, slot, position, tokens, logits):
        if mode[0] == "cancel":
            if not holder["engine"].cancel(holder["request"].request_id):
                raise AssertionError("active cancellation was not accepted")
        elif mode[0] == "fail":
            raise RuntimeError("injected diagnostic failure after native execution")
    facade = DeepSeekV4NativeServingModel(session, eos_id=eos, observer=observe)
    engine = holder["engine"] = make_engine(facade, capacity)
    inputs, expected = case["tensors"]["input_ids"][0].tolist(), case["tensors"]["generated_ids"].tolist()
    def submit():
        request = engine.submit(inputs, SamplingParams(temperature=0, top_k=1, top_p=1, max_tokens=len(expected)),
                                "native-lifecycle", stream=True)
        holder["request"] = request
        return request
    cancelled = submit()
    try:
        engine.start()
        try: await asyncio.wait_for(cancelled.future, 120)
        except asyncio.CancelledError: pass
        else: raise AssertionError("cancelled native request completed normally")
        if [t async for t in cancelled.stream_tokens()] or session.info()["active_slots"]:
            raise AssertionError("cancelled native request leaked token/state")
        mode[0] = "fail"; failed = submit()
        try: await asyncio.wait_for(failed.future, 120)
        except RuntimeError as error:
            if "injected diagnostic" not in str(error): raise
        else: raise AssertionError("observer exception did not fail native request")
        if [t async for t in failed.stream_tokens()] or session.info()["active_slots"]:
            raise AssertionError("failed native request leaked token/state")
        mode[0] = "recover"; recovered = submit()
        actual = await asyncio.wait_for(recovered.future, 120)
        if actual != expected or [t async for t in recovered.stream_tokens()] != expected:
            raise AssertionError("native request did not recover after cancellation/error")
    finally:
        await asyncio.to_thread(engine.stop, 120)
    return dict(active_cancel=True, observer_failure=True, subsequent_request_matches_golden=True,
                cancelled_status=cancelled.status.value, failed_status=failed.status.value,
                recovered_ids=actual, final_active_slots=session.info()["active_slots"], runtime=facade.last_report)


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("native serving acceptance requires Slurm GPU allocation")
    root = Path(__file__).resolve().parents[2]
    source, checkpoint = args.source_model.resolve(), args.checkpoint.resolve()
    bundle = json.loads(args.bundle.read_text())
    capacity = bundle["configuration"]["max_seq_len"]
    cfg = InferenceConfig.from_directory(source, max_seq_len=capacity)
    paths = [Path(__file__), args.module.resolve(), args.bundle.resolve(), root / "python/server/engine.py",
             root / "python/bindings/v4_native.cpp", root / "src/backends/v4_reference.cpp",
             root / "src/backends/v4_reference.hpp", root / "xmake/native_model.lua",
             Path(checkpoint_identity.__code__.co_filename), Path(read_baselines.__code__.co_filename),
             root / "python/llaisys/_C.so", root / "python/llaisys/libllaisys/libllaisys.so"]
    paths.extend(sorted((root / "python/llaisys/models/deepseek_v4_model").glob("*.py")))
    for directory in ("src/models/deepseek_v4", "src/backends/native", "src/backends/aten", "src/backends/tilelang", "src/backends/hadamard"):
        paths.extend(sorted((root / directory).glob("*.[ch]pp")))
    paths.extend(sorted((root / "build/linux/x86_64/release").glob("libllaisys-*.a")))
    paths.extend(args.bundle.resolve().parent / entry["library"] for entry in bundle["kernels"].values())
    paths.extend(source / rel for rel in ("config.json", "inference/config.json", "inference/kernel.py", "inference/model.py"))
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    print("verifying entire checkpoint payload", flush=True)
    identity = checkpoint_identity(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent, local_files_only=True, trust_remote_code=False)
    codec = ChatCodec(source, tokenizer)
    cases = read_baselines(args.baseline, source, checkpoint, capacity, tokenizer, cfg.vocab_size, identity)
    if not cases or not all(c["checkpoint_payload_verified"] for c in cases):
        raise ValueError("native serving goldens must bind complete checkpoint identity")
    print("loading native serving Session", flush=True)
    session = load_native_session(source, checkpoint, args.bundle, args.module, slots=2)
    results = []
    try:
        before = session.info()
        if before["loaded_weights"] != 67569 or before["layers"] != 43:
            raise AssertionError("incomplete native model load")
        for start in range(0, len(cases), 2):
            result = asyncio.run(wave(session, cases[start:start + 2], codec.eos_id, capacity))
            results.append(result)
            print(json.dumps({"passed_cases": [c["case_id"] for c in result["cases"]],
                              "logits": sum(c["logits_byte_comparisons"] for c in result["cases"])}), flush=True)
        lifecycle = asyncio.run(cancellation_and_recovery(session, cases[0], codec.eos_id, capacity))
        after = session.info()
        calls = {op["operation"]: op["calls"] for op in after["operators"]}
        expected_calls = sum(c["tensors"]["generated_ids"].numel() for c in cases) + 2 + cases[0]["tensors"]["generated_ids"].numel()
        if (after["model_calls"] != expected_calls or calls["token_embedding"] != expected_calls
                or calls["sparse_attn"] != expected_calls * cfg.n_layers
                or after["model_failures"] or any(op["failures"] or op["fallback"] for op in after["operators"])):
            raise AssertionError("native operator accounting is inconsistent or execution failed")
    finally:
        session.close()
    verify_checkpoint_unchanged(checkpoint, identity)
    unchanged = hashes == {str(p.resolve()): sha256(p) for p in paths}
    report = dict(all_passed=unchanged, source_unchanged=unchanged, checkpoint_identity=identity,
        scope="existing InferenceEngine + native C++ 43-layer Session; serial slots, continuous/ring cache",
        teacher_forcing=False, python_operator_callbacks=False, fallback=False, waves=results, lifecycle=lifecycle,
        logits_byte_comparisons=sum(c["logits_byte_comparisons"] for w in results for c in w["cases"]),
        loaded=before, final=after, codec=codec.report(), slurm_job_id=os.environ["SLURM_JOB_ID"],
        hardware=dict(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
                      compute_capability=list(torch.cuda.get_device_capability(0))),
        software=dict(torch=torch.__version__, cuda=torch.version.cuda, tilelang=bundle["tilelang"], tvm_ffi=bundle["tvm_ffi"]),
        measurement=dict(kind="observed_correctness_not_performance", concurrency=2, gpu_batch_size=1,
                         warmup=0, repeats=1, prefix_cache=False, cuda_graph=False, experimental_chunking=False),
        provenance=dict(command=sys.argv, file_sha256=hashes,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": unchanged, "output": str(args.output)}), flush=True)
    if not unchanged: raise RuntimeError("native serving sources changed while running")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True); parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--baseline", nargs=2, action="append", required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    try: main(args)
    except Exception as error:
        args.output.write_text(json.dumps(dict(all_passed=False, scope="native serving verification incomplete",
            error=f"{type(error).__name__}: {error}", command=sys.argv, slurm_job_id=os.environ.get("SLURM_JOB_ID")), indent=2) + "\n")
        raise

"""B300 cuBLASLt MXFP8 candidate versus the checkpoint's TileLang FP8 GEMM.

Native code requires CUDA 13 + cuBLASLt only. This harness additionally requires
Torch with E8M0, TileLang 0.1.8 and compatible TVM-FFI (project: 0.1.8.post2).
No model weights are loaded; random inputs use actual Flash-0731 projection
dimensions. This is an operator benchmark, not a model-logits or serving test.
Run in a Slurm GPU allocation. Build the independent DSO using the command in
src/ops/deepseek_v4/nvidia/fp8_prefill_sm103.cu, then provide --library.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Native:
    def __init__(self, library, m, n, k, stream, workspace_mb):
        self.dll = ctypes.CDLL(str(library.resolve()))
        self.plan = ctypes.c_void_p()
        self.stream = stream
        v = ctypes.c_void_p
        self.dll.llaisys_v4_fp8_last_error.restype = ctypes.c_char_p
        self.dll.cublasLtGetVersion.restype = ctypes.c_size_t
        self.dll.cublasLtGetVersion.argtypes = []
        signatures = {
            "create": [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t,
                       ctypes.c_size_t, ctypes.POINTER(v)],
            "prepare_b": [v, v], "prepare_a": [v, v],
            "gemm": [v, v, v, v], "run": [v, v, v, v, v],
            "select_algorithm": [v, ctypes.c_int], "algorithm_count": [v],
            "destroy": [v],
        }
        for name, args in signatures.items():
            function = getattr(self.dll, "llaisys_v4_fp8_" + name)
            function.argtypes = args
            function.restype = ctypes.c_int
        self.check(self.dll.llaisys_v4_fp8_create(m, n, k, stream.cuda_stream,
                   workspace_mb * 1024**2, ctypes.byref(self.plan)))

    def check(self, status):
        if status:
            raise RuntimeError(self.dll.llaisys_v4_fp8_last_error().decode())

    def call(self, name, *tensors):
        self.check(getattr(self.dll, "llaisys_v4_fp8_" + name)(self.plan,
                   *(tensor.data_ptr() for tensor in tensors)))

    def close(self):
        if self.plan:
            status=self.dll.llaisys_v4_fp8_destroy(self.plan)
            if status in (0,2):
                self.plan = ctypes.c_void_p()
            self.check(status)


def metric(output, reference):
    x, y = output.float().flatten(), reference.float().flatten()
    delta = x - y
    return dict(relative_l2=float(delta.norm() / y.norm().clamp_min(1e-30)),
                max_absolute=float(delta.abs().max()),
                cosine=float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
                finite=bool(torch.isfinite(x).all()))


def timed_graph(call, stream, *, nodes=20, replays=30, output=None, expected=None):
    """Return graph-amortized device time; allocation/JIT/init are excluded."""
    for _ in range(5):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(nodes):
            call()
    if output is not None:
        # Capture/warmup may already have produced the right answer. Poison the
        # destination afterwards so an empty/non-executing graph cannot pass.
        output.fill_(float("nan"))
        graph.replay()
        stream.synchronize()
        if not torch.equal(output,expected):
            raise AssertionError("graph replay did not replace poisoned output with the expected result")
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(5):
        start.record(stream)
        for _ in range(replays):
            graph.replay()
        end.record(stream)
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (nodes * replays))
    del graph
    return dict(median_us=statistics.median(values), samples_us=values,
                graph_nodes=nodes, replays_per_sample=replays, samples=len(values),
                poisoned_output_replay_checked=output is not None)


def run_case(args, kernel, m, n, k, stream, weights=None):
    # Deterministic E4M3 payload and exact UE8M0 bytes. These are synthetics,
    # not values sampled from checkpoint tensors or a task-quality workload.
    a = (torch.randn(m, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    b = ((torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
         if weights is None else weights[0])
    sa = torch.randint(120, 130, (m, k // 128), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sb = (torch.randint(120, 130, (n // 128, k // 128), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
          if weights is None else weights[1])
    output = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    baseline = torch.empty_like(output)
    # Invoke the compiled kernel directly so neither path times output allocation.
    tile = kernel.fp8_gemm_kernel(n, k, scale_dtype=kernel.FE8M0)
    tile_call = lambda: tile(a, b, baseline, sa, sb)
    native = Native(args.library, m, n, k, stream, args.workspace_mb)
    try:
        native.call("prepare_b", sb)
        native.call("prepare_a", sa)
        run = lambda: native.call("run", a, sa, b, output)
        gemm = lambda: native.call("gemm", a, b, output)
        # Initialization-only algorithm selection; benchmark its device execution,
        # then freeze one candidate before graph capture and final measurement.
        choices = []
        count = native.dll.llaisys_v4_fp8_algorithm_count(native.plan)
        if count < 1:
            raise RuntimeError("no candidate algorithms")
        for index in range(min(count, args.tune_candidates)):
            native.check(native.dll.llaisys_v4_fp8_select_algorithm(native.plan, index))
            try:
                result = timed_graph(gemm, stream, nodes=5, replays=5)
            except Exception as error:
                # Do not continue after a CUDA launch/capture error: it may poison
                # the stream. The case fails with its actual diagnostic intact.
                raise RuntimeError(f"algorithm {index} failed: {error}") from error
            choices.append((result["median_us"], index))
        selected = min(choices)[1]
        native.check(native.dll.llaisys_v4_fp8_select_algorithm(native.plan, selected))
        run(); tile_call(); stream.synchronize()
        # Independent full-FP32 dequantized reference, TF32 explicitly disabled.
        # This allocation and matmul are correctness-only, outside timed regions.
        expanded_a = a.float() * sa.float().repeat_interleave(128, dim=1)
        expanded_b = b.float() * sb.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        reference = expanded_a @ expanded_b.T
        versus_tile = metric(output, baseline)
        versus_fp32 = metric(output, reference)
        tile_vs_fp32 = metric(baseline, reference)
        passed = (versus_tile["finite"] and versus_fp32["finite"] and tile_vs_fp32["finite"] and
                  versus_tile["relative_l2"] <= args.relative_l2 and
                  versus_fp32["relative_l2"] <= args.relative_l2 and
                  tile_vs_fp32["relative_l2"] <= args.relative_l2)
        graph_output = output.clone()
        graph_baseline = baseline.clone()
        # Non-equal but overlapping pointers must be rejected before launch.
        rejected={}
        for label,ptr in (("partial_output_alias",a.data_ptr()+16),("misaligned_output",output.data_ptr()+1)):
            status=native.dll.llaisys_v4_fp8_gemm(native.plan,a.data_ptr(),b.data_ptr(),ptr)
            rejected[label]=status!=0
            if not rejected[label]:raise AssertionError(f"native accepted {label}")
        # Ordinary Torch allocations are 256B-aligned and can hide a heuristic
        # alignment mismatch. Test the advertised 16B pointer alignment as well.
        a_storage=torch.empty(a.numel()+16,device="cuda",dtype=torch.uint8)
        b_storage=torch.empty(b.numel()+16,device="cuda",dtype=torch.uint8)
        o_storage=torch.empty(output.numel()+8,device="cuda",dtype=torch.bfloat16)
        a16=a_storage[16:].view(a.dtype).reshape_as(a);a16.copy_(a)
        b16=b_storage[16:].view(b.dtype).reshape_as(b);b16.copy_(b)
        out16=o_storage[8:].reshape_as(output)
        native.call("run",a16,sa,b16,out16);stream.synchronize()
        aligned_view_correct=torch.equal(out16,graph_output)
        if not aligned_view_correct:raise AssertionError("16B-aligned matrix views changed output")
        del a16,b16,out16,a_storage,b_storage,o_storage
        changing_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(changing_graph,stream=stream):
            run()
        sa.view(torch.uint8).add_(1)  # Same pointer, new exponent => exactly twice each scale.
        output.fill_(float("nan"))
        changing_graph.replay();tile_call();stream.synchronize()
        dynamic_native=metric(output,reference*2)
        dynamic_tile=metric(baseline,reference*2)
        dynamic_passed=(dynamic_native["finite"] and dynamic_tile["finite"] and
            dynamic_native["relative_l2"]<=args.relative_l2 and dynamic_tile["relative_l2"]<=args.relative_l2)
        del changing_graph
        sa.view(torch.uint8).sub_(1)
        run();tile_call();stream.synchronize()
        inclusive = timed_graph(run, stream, nodes=args.graph_nodes, replays=args.replays,
                                output=output,expected=graph_output)
        replay_correct = torch.equal(output, graph_output)
        gemm_only = timed_graph(gemm, stream, nodes=args.graph_nodes, replays=args.replays,
                               output=output,expected=graph_output)
        scale_only = timed_graph(lambda: native.call("prepare_a", sa), stream,
                                 nodes=args.graph_nodes, replays=args.replays)
        tile_time = timed_graph(tile_call, stream, nodes=args.graph_nodes, replays=args.replays,
                               output=baseline,expected=graph_baseline)
        tile_replay_correct=torch.equal(baseline,graph_baseline)
        result = dict(m=m, n=n, k=k, passed=passed and replay_correct and tile_replay_correct and dynamic_passed,
            data=("synthetic E4M3 A/B; random UE8M0 scales" if weights is None else
                  "real MP1 E4M3 B and UE8M0 B scales; synthetic E4M3 A and UE8M0 A scales"),
            versus_tilelang=versus_tile, versus_fp32=versus_fp32,
            tilelang_versus_fp32=tile_vs_fp32, graph_replay_equal=replay_correct,
            tilelang_graph_replay_equal=tile_replay_correct,
            changed_a_scales_graph_passed=dynamic_passed,
            changed_a_scales_native=dynamic_native,changed_a_scales_tilelang=dynamic_tile,
            selected_weights=None if weights is None else weights[2],
            rejected_invalid_arguments=rejected,aligned_16b_views_equal=aligned_view_correct,
            cublaslt_version=native.dll.cublasLtGetVersion(),
            algorithm_candidates=count, selected_algorithm_index=selected,
            tuning_us=[dict(index=index, microseconds=value) for value,index in choices],
            native_with_a_scale_preparation=inclusive, native_gemm=gemm_only,
            a_scale_preparation=scale_only, tilelang=tile_time,
            speedup_including_a_scale=tile_time["median_us"]/inclusive["median_us"],
            effective_tflops=2*m*n*k/(inclusive["median_us"]*1e6),
            b_scale_preparation="once before measurement; FP8 B payload unchanged",
            m_padding=((m+15)//16)*16-m, workspace_bytes=args.workspace_mb*1024**2)
        return result
    finally:
        native.close()


def load_weights(path,layer,name,n,k):
    """Read only the two selected MP1 payloads, preserving raw FP8/E8M0 bytes.

    Identity is selected-payload + full-header hashes and file stat; this does
    not claim to hash/validate every weight in the complete checkpoint.
    """
    suffix={"query_a":"wq_a","query_b":"wq_b","latent":"wkv",
            "output":"wo_b","indexer":"indexer.wq_b"}[name]
    stem=f"layers.{layer}.attn.{suffix}"
    keys=[stem+".weight",stem+".scale"]
    expected=[("F8_E4M3",[n,k],torch.float8_e4m3fn),
              ("F8_E8M0",[n//128,k//128],torch.float8_e8m0fnu)]
    tensors=[];payloads={}
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno())
        header_size=struct.unpack("<Q",stream.read(8))[0]
        if header_size>128*1024**2 or header_size+8>before.st_size:
            raise ValueError("invalid safetensors header size")
        header_bytes=stream.read(header_size)
        def unique(items):
            result={}
            for key,value in items:
                if key in result:raise ValueError("duplicate safetensors header key")
                result[key]=value
            return result
        header=json.loads(header_bytes,object_pairs_hook=unique)
        ranges=[]
        for key,(dtype,shape,torch_dtype) in zip(keys,expected):
            item=header[key];start,end=item["data_offsets"]
            elements=shape[0]*shape[1]
            if (item["dtype"]!=dtype or item["shape"]!=shape or start<0 or end-start!=elements or
                    8+header_size+end>before.st_size):
                raise ValueError(f"selected tensor contract mismatch: {key}")
            ranges.append((start,end))
            stream.seek(8+header_size+start);raw=bytearray(stream.read(elements))
            if len(raw)!=elements:raise IOError("short selected payload read")
            payloads[key]=dict(dtype=dtype,shape=shape,bytes=elements,sha256=hashlib.sha256(raw).hexdigest())
            tensors.append(torch.frombuffer(raw,dtype=torch.uint8).clone().view(torch_dtype).reshape(shape).cuda())
        if ranges[0][0]<ranges[1][1] and ranges[1][0]<ranges[0][1]:
            raise ValueError("selected tensor payloads overlap")
        after=os.fstat(stream.fileno())
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise RuntimeError("checkpoint changed while reading selected payloads")
    info=dict(path=str(path),scope="selected weight/scale payloads plus full header; not entire checkpoint payload",
              header_sha256=hashlib.sha256(header_bytes).hexdigest(),file_size=before.st_size,
              tensors=payloads)
    return (*tensors,info)


def main(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run in a Slurm GPU allocation")
    if args.graph_nodes<1 or args.replays<1 or args.tune_candidates<1:
        raise ValueError("positive graph/tuning counts required")
    torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_dtype(torch.bfloat16)
    config = json.loads((args.source_model / "inference/config.json").read_text())
    shapes = {"query_a": (config["q_lora_rank"], config["dim"]),
              "query_b": (config["n_heads"]*config["head_dim"], config["q_lora_rank"]),
              "latent": (config["head_dim"], config["dim"]),
              "output": (config["dim"], config["o_groups"]*config["o_lora_rank"]),
              "indexer": (config["index_n_heads"]*config["index_head_dim"], config["q_lora_rank"])}
    source = args.source_model / "inference/kernel.py"
    spec = importlib.util.spec_from_file_location("v4_published_fp8_benchmark", source)
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel
    spec.loader.exec_module(kernel)
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), args.library, source, args.source_model/"inference/config.json",
             root/"src/ops/deepseek_v4/nvidia/fp8_prefill_sm103.cu",
             root/"src/ops/deepseek_v4/nvidia/fp8_prefill_sm103.h"]
    hashes = {str(p):digest(p) for p in files}
    checkpoint_stat=None if args.checkpoint is None else args.checkpoint.stat()
    args.report = dict(all_passed=False, scope="B300 FP8 GEMM operator benchmark; synthetic activations",
        slurm_job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
        cuda=torch.version.cuda, provenance=hashes, cases=[],
        precision="native MXFP8 Tensor Core FP32 accumulation, BF16 output; UE8M0 replication only",
        threshold_relative_l2=args.relative_l2, cublaslt_fallback=False,
        whole_model_validated=False, measurement="CUDA event time over captured repeated calls; 5 samples",
        dependencies="native DSO: CUDA13 and cuBLASLt; harness: Torch, TileLang0.1.8, TVM-FFI0.1.8.post2")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in args.shapes:
            n,k=shapes[name]
            weights=None if args.checkpoint is None else load_weights(args.checkpoint,args.layer,name,n,k)
            for m in args.m:
                row=run_case(args,kernel,m,n,k,stream,weights)
                row["projection"]=name
                args.report["cases"].append(row)
                args.output.write_text(json.dumps(args.report,indent=2)+"\n")
                print(json.dumps({k:row[k] for k in ("projection","m","n","k","passed","speedup_including_a_scale")}),flush=True)
    unchanged=hashes=={str(p):digest(p) for p in files}
    if args.checkpoint is not None:
        final_stat=args.checkpoint.stat()
        checkpoint_unchanged=all(getattr(checkpoint_stat,k)==getattr(final_stat,k) for k in
                                 ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns"))
        args.report["checkpoint_unchanged"]=checkpoint_unchanged
        unchanged=unchanged and checkpoint_unchanged
    args.report.update(source_unchanged=unchanged,
        all_passed=unchanged and all(r["passed"] for r in args.report["cases"]))
    args.output.write_text(json.dumps(args.report,indent=2)+"\n")
    if not args.report["all_passed"]:
        raise RuntimeError("operator correctness/provenance gate failed")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--source-model",type=Path,default=Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,help="optional converted MP1 safetensors; reads selected real B/scales only")
    parser.add_argument("--layer",type=int,default=2,help="selected model layer; layer 2 includes the learned indexer")
    parser.add_argument("--m",type=int,nargs="+",default=[32,256,2105])
    parser.add_argument("--shapes",nargs="+",choices=["query_a","query_b","latent","output","indexer"],default=["query_a","query_b","latent","output","indexer"])
    parser.add_argument("--workspace-mb",type=int,default=32)
    parser.add_argument("--tune-candidates",type=int,default=4)
    parser.add_argument("--graph-nodes",type=int,default=20)
    parser.add_argument("--replays",type=int,default=30)
    parser.add_argument("--relative-l2",type=float,default=.01)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:main(args)
    except Exception as error:
        report=getattr(args,"report",{})
        report.update(all_passed=False,error=f"{type(error).__name__}: {error}")
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        raise

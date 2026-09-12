"""GPU contract/edge tests for the standalone B300 W4A8 decode operator."""
import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import sys

from bench_w4a8_decode import load_module, sha256, write_report


def main(args):
    import torch
    torch.manual_seed(91226)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "python/llaisys/models/deepseek_v4_w4a8_cuda.py"
    native = load_module("w4a8_edge_native", wrapper).W4A8DecodeSm103(args.library)
    sources = [Path(__file__), wrapper, args.library,
        root / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.cu",
        root / "src/ops/deepseek_v4/nvidia/w4a8_decode_sm103.h"]
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    report = dict(all_passed=False, native_version=native.version, source_sha256=hashes,
                  gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  cases=[], graph_input_mutation=False, concurrent_streams=False)
    write_report(args.output, report)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         device="cuda", dtype=torch.float32)

    def inputs(m, n, k):
        # All sixteen FP4 codes and nonuniform scale groups. Scale byte zero
        # exercises UE8M0's 2^-127 value; most groups remain normally scaled.
        a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 20).to(torch.float8_e4m3fn)
        sa = torch.randint(119, 129, (m, k // 128), device="cuda", dtype=torch.uint8)
        sb = torch.randint(120, 128, (n, k // 32), device="cuda", dtype=torch.uint8)
        sb[:, 0] = 0
        b = (torch.arange(n * k // 2, device="cuda", dtype=torch.int64) % 256).byte().reshape(n, k // 2)
        return a, sa.view(torch.float8_e8m0fnu), b, sb.view(torch.float8_e8m0fnu)

    def oracle(a, sa, b, sb):
        m, k = a.shape; n = b.shape[0]
        codes = torch.stack((b & 15, b >> 4), dim=-1).reshape(n, k).long()
        qa = (a.float().reshape(m, k // 128, 128) * sa.float().unsqueeze(-1)).reshape(m, k)
        qb = (table[codes].reshape(n, k // 32, 32) * sb.float().unsqueeze(-1)).reshape(n, k)
        return (qa @ qb.T).bfloat16()

    def check(actual, expected):
        delta = actual.float() - expected.float()
        relative = float(delta.norm() / expected.float().norm().clamp_min(1e-20))
        if not bool(torch.isfinite(actual).all()) or relative > .005:
            raise AssertionError(f"numeric failure: relative L2={relative}")
        return relative

    try:
        for m in range(1, 9):
            for n, k in ((1, 128), (3, 256), (13, 4096), (129, 128)):
                a, sa, b, sb = inputs(m, n, k)
                expected = oracle(a, sa, b, sb)
                for variant in (0, 1, 2, 3):
                    # Keep eight BF16 guard values on each side. The output
                    # offset is sixteen bytes, preserving C ABI alignment.
                    storage = torch.full((m * n + 16,), 57., dtype=torch.bfloat16, device="cuda")
                    output = storage[8:-8].view(m, n)
                    native.gemm(a, sa, b, sb, out=output, variant=variant)
                    error = check(output, expected)
                    assert bool((storage[:8] == 57).all() and (storage[-8:] == 57).all())
                    report["cases"].append(dict(m=m, n=n, k=k, variant=variant, relative_l2=error, guards_intact=True))
        a, sa, b, sb = inputs(5, 13, 256)
        output = torch.empty(5, 13, dtype=torch.bfloat16, device="cuda")
        native.gemm(a, sa, b, sb, out=output, variant=3)
        torch.cuda.synchronize()
        before = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            native.gemm(a, sa, b, sb, out=output, variant=3)
        # Change payload after capture; replay must consume the new bytes.
        a.copy_(torch.full(a.shape, 3., device="cuda", dtype=torch.float32).to(a.dtype))
        expected = oracle(a, sa, b, sb)
        graph.replay()
        torch.cuda.synchronize()
        check(output, expected)
        assert not torch.equal(output, before)
        report["graph_input_mutation"] = True
        del graph

        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        results = []
        for index, stream in enumerate(streams):
            a, sa, b, sb = inputs(3 + index, 129, 256)
            expected = oracle(a, sa, b, sb)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                output = native.gemm(a, sa, b, sb, variant=1 + index)
            results.append((output, expected))
            # Deliberately release all external input owners before waiting.
            # The wrapper's record_stream keeps allocator reuse safe.
            del a, sa, b, sb
            gc.collect()
            pressure = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
            pressure.fill_(173)
        for stream in streams:
            stream.synchronize()
        for output, expected in results:
            check(output, expected)
        report["concurrent_streams"] = True
        report["source_unchanged"] = all(sha256(p) == h for p, h in hashes.items())
        report["all_passed"] = report["source_unchanged"] and len(report["cases"]) == 128
        if not report["all_passed"]:
            raise AssertionError("incomplete tests or changed source")
        print(json.dumps(dict(all_passed=True, numerical_cases=len(report["cases"]),
                              graph_input_mutation=True, concurrent_streams=True)), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_report(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

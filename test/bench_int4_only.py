#!/usr/bin/env python3
"""INT4 量化模型专用基准测试"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import llaisys
from test_utils import llaisys_device

MODEL_PATH = "quantized_model_int4"


def get_gpu_mem_mb():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        return int(r.stdout.strip().split('\n')[0])
    except Exception:
        return -1


def bench(model, input_ids, max_new=50, warmup=2, repeat=3):
    for _ in range(warmup):
        model.generate(input_ids, max_new_tokens=max_new, top_k=1, top_p=1.0, temperature=1.0)
    times, outs = [], []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=max_new, top_k=1, top_p=1.0, temperature=1.0)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        outs.append(len(out) - len(input_ids))
    avg_t = sum(times) / len(times)
    avg_o = sum(outs) / len(outs)
    return avg_t, avg_o, avg_o / avg_t if avg_t > 0 else 0, min(times), max(times)


def main():
    device = llaisys_device("nvidia")
    mem_before = get_gpu_mem_mb()
    print(f"GPU 显存 (加载前): {mem_before} MB", flush=True)

    print(f"加载 INT4 量化模型: {MODEL_PATH}", flush=True)
    model = llaisys.models.Qwen2(MODEL_PATH, device)
    mem_after = get_gpu_mem_mb()
    print(f"INT4 模型显存: {mem_after} MB (增量 {mem_after - mem_before} MB)", flush=True)

    print(f"\n{'InputLen':<10} {'AvgOut':<8} {'AvgTime(ms)':<14} {'Tok/s':<10} {'Min(ms)':<10} {'Max(ms)':<10}", flush=True)
    print("-" * 62, flush=True)

    for input_len in [16, 64, 128, 256, 512]:
        prompt = [(i % 1000) + 1 for i in range(input_len)]
        try:
            avg_t, avg_o, tps, mn, mx = bench(model, prompt, max_new=50, warmup=2, repeat=3)
            print(f"{input_len:<10} {avg_o:<8.1f} {avg_t*1000:<14.1f} {tps:<10.1f} {mn*1000:<10.1f} {mx*1000:<10.1f}", flush=True)
        except Exception as e:
            print(f"{input_len:<10} ERROR: {e}", flush=True)
            break

    print(f"\nGPU 显存 (测试后): {get_gpu_mem_mb()} MB", flush=True)


if __name__ == "__main__":
    main()

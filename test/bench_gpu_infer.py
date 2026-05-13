#!/usr/bin/env python3
"""
LLAISYS GPU 推理性能基准测试

测试项:
  1. FP16 vs FP32 推理速度对比 (端到端 generate)
  2. 不同输入长度下的 prefill 吞吐量
  3. decode 吞吐量 (tokens/s)
  4. INT8 量化模型 vs FP16 对比
  5. GPU 显存使用情况

用法:
  python test/bench_gpu_infer.py --device nvidia
  LLAISYS_FORCE_FP32=1 python test/bench_gpu_infer.py --device nvidia  # 强制 FP32
"""
import sys, os, io, time, argparse, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import llaisys
from test_utils import llaisys_device

# ── 配置 ──

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "DeepSeek-R1-Distill-Qwen-1.5B")
QUANT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "quantized_model")

# 不同长度的测试 prompt (重复 token 即可)
def make_prompt(length: int) -> list:
    """生成指定长度的 token_ids 输入"""
    # 使用重复 token: [1, 2, 3, ...] 循环
    return [(i % 1000) + 1 for i in range(length)]


def get_gpu_mem_mb():
    """获取当前 GPU 显存使用量 (MB)"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip().split('\n')[0])
    except Exception:
        return -1


def bench_generate(model, input_ids, max_new_tokens=50, warmup=2, repeat=5,
                   top_k=1, temperature=1.0, top_p=1.0):
    """测量 generate 的延迟和吞吐量"""
    # warmup
    for _ in range(warmup):
        model.generate(input_ids, max_new_tokens=max_new_tokens,
                       top_k=top_k, top_p=top_p, temperature=temperature)

    times = []
    output_lens = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             top_k=top_k, top_p=top_p, temperature=temperature)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # output 包含 input + generated tokens
        new_tokens = len(out) - len(input_ids)
        output_lens.append(new_tokens)

    avg_time = sum(times) / len(times)
    avg_output = sum(output_lens) / len(output_lens)
    prefill_input_len = len(input_ids)

    return {
        "input_len": prefill_input_len,
        "max_new_tokens": max_new_tokens,
        "avg_output_tokens": avg_output,
        "avg_time_s": avg_time,
        "tokens_per_sec": avg_output / avg_time if avg_time > 0 else 0,
        "times": times,
    }


def print_table(headers, rows, title=""):
    """打印对齐的表格"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2
                  for i, h in enumerate(headers)]

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))

    for row in rows:
        line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(line)


def main():
    parser = argparse.ArgumentParser(description="LLAISYS GPU Inference Benchmark")
    parser.add_argument("--device", type=str, default="nvidia", choices=["cpu", "nvidia"])
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-quant", action="store_true", help="跳过量化模型测试")
    parser.add_argument("--skip-fp32", action="store_true", help="跳过 FP32 测试")
    args = parser.parse_args()

    device = llaisys_device(args.device)
    input_lengths = [16, 64, 128, 256, 512]

    print(f"设备: {args.device}")
    print(f"GPU 显存 (测试前): {get_gpu_mem_mb()} MB")

    # ────────────────────────────────────────────────────
    # Test 1: FP16 推理基准
    # ────────────────────────────────────────────────────
    print("\n加载 FP16 模型...")
    mem_before = get_gpu_mem_mb()
    model_fp16 = llaisys.models.Qwen2(MODEL_PATH, device)
    mem_after = get_gpu_mem_mb()
    print(f"模型加载完成，显存增量: {mem_after - mem_before} MB")

    rows_fp16 = []
    for input_len in input_lengths:
        prompt = make_prompt(input_len)
        result = bench_generate(model_fp16, prompt,
                                max_new_tokens=args.max_new_tokens,
                                warmup=args.warmup, repeat=args.repeat)
        rows_fp16.append([
            input_len,
            f"{result['avg_output_tokens']:.1f}",
            f"{result['avg_time_s']*1000:.1f}",
            f"{result['tokens_per_sec']:.1f}",
            f"{min(result['times'])*1000:.1f}",
            f"{max(result['times'])*1000:.1f}",
        ])

    print_table(
        ["InputLen", "AvgOut", "AvgTime(ms)", "Tok/s", "Min(ms)", "Max(ms)"],
        rows_fp16,
        title="FP16 推理基准 (DeepSeek-R1-Distill-Qwen-1.5B)"
    )

    # ────────────────────────────────────────────────────
    # Test 2: FP32 推理对比 (可选)
    # ────────────────────────────────────────────────────
    if not args.skip_fp32:
        # 释放 FP16 模型
        del model_fp16
        gc.collect()

        print("\n加载 FP32 模型 (LLAISYS_FORCE_FP32=1)...")
        os.environ["LLAISYS_FORCE_FP32"] = "1"
        mem_before = get_gpu_mem_mb()
        model_fp32 = llaisys.models.Qwen2(MODEL_PATH, device)
        mem_after = get_gpu_mem_mb()
        print(f"FP32 模型加载完成，显存增量: {mem_after - mem_before} MB")

        rows_fp32 = []
        for input_len in input_lengths:
            prompt = make_prompt(input_len)
            result = bench_generate(model_fp32, prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    warmup=args.warmup, repeat=args.repeat)
            rows_fp32.append([
                input_len,
                f"{result['avg_output_tokens']:.1f}",
                f"{result['avg_time_s']*1000:.1f}",
                f"{result['tokens_per_sec']:.1f}",
            ])

        print_table(
            ["InputLen", "AvgOut", "AvgTime(ms)", "Tok/s"],
            rows_fp32,
            title="FP32 推理对比"
        )

        # FP16 vs FP32 加速比
        print("\n  FP16 vs FP32 加速比:")
        for i, input_len in enumerate(input_lengths):
            fp16_tps = float(rows_fp16[i][3])
            fp32_tps = float(rows_fp32[i][3])
            speedup = fp16_tps / fp32_tps if fp32_tps > 0 else 0
            print(f"    InputLen={input_len:>4}: FP16 {fp16_tps:.1f} tok/s vs FP32 {fp32_tps:.1f} tok/s = {speedup:.2f}x")

        del model_fp32
        os.environ.pop("LLAISYS_FORCE_FP32", None)
        gc.collect()

    # ────────────────────────────────────────────────────
    # Test 3: INT8 量化模型对比 (可选)
    # ────────────────────────────────────────────────────
    if not args.skip_quant and os.path.isdir(QUANT_MODEL_PATH):
        # 确保 FP16 模型已释放 (跳过 FP32 时模型仍在内存中)
        try:
            del model_fp16
        except NameError:
            pass
        gc.collect()
        time.sleep(1)  # 等待 GPU 显存释放

        print("\n加载 INT8 量化模型...")
        mem_before = get_gpu_mem_mb()
        try:
            model_int8 = llaisys.models.Qwen2(QUANT_MODEL_PATH, device)
            mem_after = get_gpu_mem_mb()
            print(f"INT8 模型加载完成，显存增量: {mem_after - mem_before} MB")

            rows_int8 = []
            for input_len in [16, 64, 128, 256]:
                prompt = make_prompt(input_len)
                result = bench_generate(model_int8, prompt,
                                        max_new_tokens=args.max_new_tokens,
                                        warmup=args.warmup, repeat=args.repeat)
                rows_int8.append([
                    input_len,
                    f"{result['avg_output_tokens']:.1f}",
                    f"{result['avg_time_s']*1000:.1f}",
                    f"{result['tokens_per_sec']:.1f}",
                ])

            print_table(
                ["InputLen", "AvgOut", "AvgTime(ms)", "Tok/s"],
                rows_int8,
                title="INT8 量化推理对比"
            )

            del model_int8
            gc.collect()
        except Exception as e:
            print(f"INT8 模型加载失败: {e}")

    # ────────────────────────────────────────────────────
    # Test 4: GPU Paged Attention Benchmark (C++)
    # ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  运行 C++ Paged Attention GPU Benchmark")
    print("=" * 70)
    bench_bin = os.path.join(os.path.dirname(__file__), "..",
                             "build", "linux", "x86_64", "release",
                             "llaisys-bench-paged-attention")
    if os.path.isfile(bench_bin):
        import subprocess
        result = subprocess.run([bench_bin], capture_output=True, text=True, timeout=120)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"[STDERR] {result.stderr[-500:]}")
    else:
        print(f"  跳过: {bench_bin} 不存在")

    print(f"\n测试结束. GPU 显存 (测试后): {get_gpu_mem_mb()} MB")


if __name__ == "__main__":
    main()

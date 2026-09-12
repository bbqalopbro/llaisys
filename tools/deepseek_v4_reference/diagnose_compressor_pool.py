"""Capture the first projected pool whose BF16 rounding differs from PyTorch.

Run with run_single_gpu.py arguments. The model executes one prefill only;
the first mismatch is saved to /tmp/llaisys-compressor-pool.pt for reduction
diagnostics. This is not an accuracy or performance benchmark.
"""
import torch
import run_single_gpu as runner


load_module = runner._load_module
calls = 0


def diagnostic_load(name, path):
    module = load_module(name, path)
    if name != "llaisys_deepseek_v4_e2e_native":
        return module
    original = module.DeepSeekV4NativeReferenceOps.compress_projected

    def inspect(self, kv, score, ape, ks, ss, ratio, start_pos):
        global calls
        actual = original(self, kv, score, ape, ks, ss, ratio, start_pos)
        if start_pos or actual.shape[1] == 0:
            return actual
        calls += 1
        batch, _, width = kv.shape
        groups = actual.shape[1]
        dimension = actual.shape[2]
        values = kv[:, :groups*ratio].reshape(batch, groups, ratio, width)
        scores = score[:, :groups*ratio].reshape(batch, groups, ratio, width) + ape
        if ratio == 4:
            values = torch.cat((torch.cat((torch.zeros_like(values[:, :1, :, :dimension]), values[:, :-1, :, :dimension]), 1), values[..., dimension:]), 2)
            scores = torch.cat((torch.cat((torch.full_like(scores[:, :1, :, :dimension], -torch.inf), scores[:, :-1, :, :dimension]), 1), scores[..., dimension:]), 2)
        expected = (values * scores.softmax(2)).sum(2)
        mismatch = actual.bfloat16() != expected.bfloat16()
        print({"compressor_call": calls, "ratio": ratio, "shape": list(kv.shape),
               "max_fp32_error": float((actual-expected).abs().max()),
               "bf16_mismatches": int(mismatch.sum())}, flush=True)
        if mismatch.any():
            torch.save({"kv": kv.cpu(), "score": score.cpu(), "ape": ape.cpu(),
                        "actual": actual.cpu(), "expected": expected.cpu(),
                        "ratio": ratio, "call": calls}, "/tmp/llaisys-compressor-pool.pt")
            raise RuntimeError("captured first BF16 pool mismatch in /tmp/llaisys-compressor-pool.pt")
        return actual

    module.DeepSeekV4NativeReferenceOps.compress_projected = inspect
    return module


@torch.inference_mode()
def prefill_only(model, tokenizer, prompt, *args):
    tokens = tokenizer.encode(prompt, return_tensors="pt").cuda()
    runner._reset_runtime_state(model)
    model(tokens, 0)
    print("prefill completed without BF16 pool mismatch", flush=True)
    raise SystemExit(0)


runner._load_module = diagnostic_load
runner._evaluate_case = prefill_only
if __name__ == "__main__":
    runner.main()

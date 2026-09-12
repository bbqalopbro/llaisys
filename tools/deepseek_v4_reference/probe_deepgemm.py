"""Check DeepGEMM FP8/FP4 on real checkpoint slices before model integration."""
import argparse
import importlib.util
import json
from pathlib import Path

import deep_gemm
import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(908)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("quant_oracle", root / "python/llaisys/models/deepseek_v4_reference.py")
    reference = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    rows = []
    for stem, mode in (("layers.0.attn.wq_a", "fp8"), ("layers.0.ffn.experts.0.w1", "fp4")):
        with safe_open(args.weights, framework="pt", device="cuda") as file:
            weight, scale = file.get_tensor(stem + ".weight"), file.get_tensor(stem + ".scale")
        n, packed_k = weight.shape
        k = packed_k * (2 if mode == "fp4" else 1)
        decoded = (reference.dequantize_fp4_e2m1(weight.view(torch.uint8), scale) if mode == "fp4"
                   else reference.dequantize_fp8_blocks(weight, scale)).float()
        b = (weight.view(torch.int8) if mode == "fp4" else weight, scale.float())
        for m in (1, 5, 32, 129, 259):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            a = deep_gemm.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128)
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            deep_gemm.fp8_fp4_gemm_nt(a, b, y, recipe_a=(1, 128), recipe_b=(1, 32) if mode == "fp4" else (128, 128))
            qx = (a[0].float().reshape(m, k // 128, 128) * a[1].unsqueeze(-1)).reshape(m, k)
            expected = (qx @ decoded.T).bfloat16()
            error = y.float() - expected.float()
            cosine = torch.nn.functional.cosine_similarity(y.float().flatten(), expected.float().flatten(), dim=0).item()
            relative_l2 = (error.norm() / expected.float().norm().clamp_min(1e-8)).item()
            row = {"weight": stem, "mode": mode, "m": m, "n": n, "k": k,
                   "cosine": cosine, "relative_l2": relative_l2, "max_abs_error": error.abs().max().item(),
                   "different_elements": error.count_nonzero().item()}
            rows.append(row)
            print(row, flush=True)
            if cosine < .99999 or relative_l2 > .005:
                raise RuntimeError("DeepGEMM slice accuracy gate failed")
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                                      "deep_gemm": deep_gemm.__version__, "torch": torch.__version__, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

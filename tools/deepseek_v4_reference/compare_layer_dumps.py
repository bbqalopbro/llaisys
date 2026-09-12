"""Compare saved same-history layer captures without loading weights or using a GPU."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = load_file(str(args.reference)), load_file(str(args.candidate))
    if set(reference) != set(candidate):
        raise ValueError("capture key sets differ")
    for key in ("input_ids", "teacher_forced_ids"):
        if not torch.equal(reference[key], candidate[key]):
            raise ValueError(f"different input history: {key}")
    rows = []
    for key in sorted(k for k in reference if k.startswith("step_")):
        expected, actual = reference[key].float(), candidate[key].float()
        if expected.shape != actual.shape or not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise ValueError(f"nonfinite or differently shaped capture: {key}")
        step, layer = (int(part.split("_")[1]) for part in key.split("."))
        error = actual - expected
        rows.append({"step": step, "layer": layer, "max_abs_error": error.abs().max().item(),
                     "relative_l2": (error.norm() / expected.norm().clamp_min(1e-8)).item(),
                     "cosine": torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
                     "exact_equal": torch.equal(actual, expected)})
    rows.sort(key=lambda row: (row["step"], row["layer"]))
    result = {"reference": str(args.reference), "candidate": str(args.candidate),
              "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (args.reference, args.candidate)},
              "scope": "same-token-history layer differences; not an automatic accuracy acceptance decision",
              "layers": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for step in sorted({row["step"] for row in rows}):
        different = [row for row in rows if row["step"] == step and not row["exact_equal"]]
        print(json.dumps({"step": step, "first_difference": different[0] if different else None}))


if __name__ == "__main__":
    main()

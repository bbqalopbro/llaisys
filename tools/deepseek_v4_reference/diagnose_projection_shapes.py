"""Isolate projection shape effects using identical real-model input rows.

Accepts run_single_gpu.py arguments plus --probe-output and --probe-layers.
Executes only the first selected blocks of one prefill. For each projection,
compares its last output row against a separate one-row call with the same
input. Also checks HC preprocessing before attention/cache access. This is
diagnostic evidence, not a model accuracy or latency benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import torch
import run_single_gpu as runner


def metrics(full, single):
    full, single = full.float().flatten(), single.float().flatten()
    difference = (full - single).abs()
    return {
        "elements": full.numel(),
        "different_elements": int(torch.count_nonzero(difference).item()),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(full, single, dim=0).item()),
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--probe-layers", type=int, default=4)
    options, remaining = parser.parse_known_args()
    if options.probe_layers < 1:
        raise ValueError("probe-layers must be positive")
    command = sys.argv.copy()
    sys.argv = [sys.argv[0], *remaining]

    @torch.inference_mode()
    def inspect(model, tokenizer, prompt, max_new_tokens, max_seq_len, layer_diagnostics=False):
        del max_new_tokens, layer_diagnostics
        published = sys.modules["deepseek_v4_published_model"]
        if not hasattr(published.sparse_attn, "__self__"):
            raise ValueError("projection diagnosis requires the explicit native structural library")
        tokens = tokenizer.encode(prompt, return_tensors="pt").cuda()
        if tokens.shape[1] > max_seq_len or tokens.shape[1] < 2:
            raise ValueError("projection diagnosis requires 2..max-seq-len input tokens")
        rows, hooks, originals = [], [], []
        layers = list(model.layers[:options.probe_layers])
        active_layer = {"index": 0}
        original_einsum = torch.einsum

        def inspect_einsum(equation, *operands, **kwargs):
            output = original_einsum(equation, *operands, **kwargs)
            if isinstance(equation, str) and equation.replace(" ", "") == "bsgd,grd->bsgr":
                single = original_einsum(equation, operands[0][:, -1:].clone(), operands[1], **kwargs)
                rows.append({"name": f"layers.{active_layer['index']}.attn.wo_a",
                             "kind": "bf16_grouped_projection_same_input_row",
                             "input_rows": operands[0].shape[1],
                             "weight_dtype": str(operands[1].dtype),
                             **metrics(output[:, -1:], single)})
            return output

        def projection_hook(name):
            def hook(module, inputs, output):
                x = inputs[0]
                if x.numel() // x.shape[-1] < 2:
                    return
                single_input = x.reshape(-1, x.shape[-1])[-1:].clone()
                # Direct forward avoids recursively invoking this diagnostic hook.
                single_output = module.forward(single_input)
                rows.append({"name": name, "kind": "linear_same_input_row",
                             "input_rows": x.numel() // x.shape[-1],
                             "weight_dtype": str(module.weight.dtype),
                             **metrics(output.reshape(-1, output.shape[-1])[-1], single_output)})
            return hook

        for index, layer in enumerate(layers):
            for name, module in layer.named_modules():
                # Capture attention, compressor, Indexer and shared expert
                # projections; routed expert batches are outside this probe.
                if isinstance(module, published.Linear) and not name.startswith("ffn.experts."):
                    hooks.append(module.register_forward_hook(projection_hook(f"layers.{index}.{name}")))
            original = layer.hc_pre
            originals.append((layer, original))

            def hc_pre(self, x, weight, scale, base, *, original=original, index=index):
                full = original(x, weight, scale, base)
                single = original(x[:, -1:].clone(), weight, scale, base)
                kind = "attn" if weight is self.hc_attn_fn else "ffn"
                rows.append({"name": f"layers.{index}.hc_{kind}", "kind": "hc_same_input_row",
                             "input_rows": x.shape[1],
                             "output": metrics(full[0][:, -1:], single[0]),
                             "post": metrics(full[1][:, -1:], single[1]),
                             "combination": metrics(full[2][:, -1:], single[2])})
                return full

            layer.hc_pre = types.MethodType(hc_pre, layer)
        try:
            torch.einsum = inspect_einsum
            runner._reset_runtime_state(model)
            hidden = model.embed(tokens).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
            for index, layer in enumerate(layers):
                active_layer["index"] = index
                hidden = layer(hidden, 0, tokens)
                print(f"projection probe completed layer {index}", file=sys.stderr, flush=True)
        finally:
            torch.einsum = original_einsum
            for hook in hooks:
                hook.remove()
            for layer, original in originals:
                layer.hc_pre = original

        library = Path(published.sparse_attn.__self__._library._name)
        native_module = sys.modules["llaisys_deepseek_v4_e2e_native"]
        sources = (Path(__file__), Path(runner.__file__), Path(published.__file__),
                   Path(native_module.__file__), library)
        result = {
            "kind": "same_input_projection_shape_diagnostic_not_model_accuracy",
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "prompt_tokens": tokens.shape[1], "layers": len(layers),
            "command": command,
            "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "comparisons": rows,
        }
        options.probe_output.parent.mkdir(parents=True, exist_ok=True)
        options.probe_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        raise SystemExit(0)

    runner._evaluate_case = inspect
    runner.main()


if __name__ == "__main__":
    main()

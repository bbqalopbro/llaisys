"""Standalone script for profiling llaisys inference only (no HuggingFace)."""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from test_utils import *
import llaisys
from transformers import AutoTokenizer
import time

model_path = "/home/bbq/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

device_name = "nvidia"
model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)

input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": "Who are you?"}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer.encode(input_content)

start = time.time()
outputs = model.generate(inputs, max_new_tokens=10, top_k=1, top_p=1.0, temperature=1.0)
elapsed = time.time() - start

print(f"\nTokens: {outputs[:20]}")
print(f"Time: {elapsed:.3f}s")

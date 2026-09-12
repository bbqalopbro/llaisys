"""Explicit adapter for the encoding shipped with the selected V4 checkpoint.

The caller supplies a trusted local model directory. No remote code is fetched,
no model.py is imported, and tool calls are parsed, never executed.
"""

import copy
import hashlib
from pathlib import Path
import types


class ChatCodec:
    def __init__(self, source_directory, tokenizer):
        self.path = Path(source_directory).resolve() / "encoding/encoding_dsv4.py"
        payload = self.path.read_bytes()
        self.encoding_sha256 = hashlib.sha256(payload).hexdigest()
        self.module = types.ModuleType("llaisys_v4_checkpoint_encoding")
        # Execute exactly the bytes whose identity is recorded, not a stale pyc.
        exec(compile(payload, str(self.path), "exec"), self.module.__dict__)
        self.tokenizer = tokenizer
        self.eos_id, self.bos_id = tokenizer.eos_token_id, tokenizer.bos_token_id
        for token, token_id in ((self.module.bos_token, self.bos_id),
                                (self.module.eos_token, self.eos_id)):
            if type(token_id) is not int or tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError("checkpoint encoding and tokenizer BOS/EOS disagree")
        self.tokenizer_sha256 = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()

    @staticmethod
    def validate_options(thinking_mode, drop_thinking=True, reasoning_effort="low"):
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError("thinking_mode must be chat or thinking")
        if type(drop_thinking) is not bool or reasoning_effort not in ("low", "high", "max"):
            raise ValueError("invalid drop_thinking or reasoning_effort")

    def encode(self, messages, *, thinking_mode="chat", drop_thinking=True, reasoning_effort="low"):
        self.validate_options(thinking_mode, drop_thinking, reasoning_effort)
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in (
                    "system", "user", "assistant", "tool", "latest_reminder", "developer"):
                raise ValueError("unsupported or missing message role")
            if message.get("content") is not None and not isinstance(message["content"], str):
                raise ValueError("this text model requires string message content")
            # Upstream preprocessing ignores user-supplied content_blocks. Do not
            # silently accept images or an already-preprocessed conversation.
            if "content_blocks" in message:
                raise ValueError("pass text/tool messages, not preprocessed content_blocks")
        text = self.module.encode_messages(copy.deepcopy(messages), thinking_mode=thinking_mode,
                                          drop_thinking=drop_thinking, reasoning_effort=reasoning_effort)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[0] != self.bos_id:
            raise ValueError("encoded conversation must start with the checkpoint BOS")
        return {"text": text, "input_ids": ids}

    def decode_completion(self, token_ids, *, thinking_mode="chat", finish_reason):
        self.validate_options(thinking_mode)
        ids = list(token_ids)
        if finish_reason not in ("stop", "length", "cancelled"):
            raise ValueError("invalid finish_reason")
        if finish_reason == "stop":
            if not ids or ids[-1] != self.eos_id or self.eos_id in ids[:-1]:
                raise ValueError("stop requires exactly one terminal EOS")
        elif self.eos_id in ids:
            raise ValueError("EOS output cannot be labelled length/cancelled")
        text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        message, error = None, None
        if finish_reason == "stop":
            try:
                message = self.module.parse_message_from_completion_text(text, thinking_mode=thinking_mode)
            except (AssertionError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        return {"raw_text": text, "finish_reason": finish_reason,
                "message": message, "parse_error": error,
                "complete": finish_reason == "stop" and message is not None}

    def report(self):
        return {"format": "deepseek-v4-checkpoint-encoding", "encoding_path": str(self.path),
                "encoding_sha256": self.encoding_sha256, "tokenizer_backend_sha256": self.tokenizer_sha256,
                "bos_id": self.bos_id, "eos_id": self.eos_id, "add_special_tokens": False,
                "tool_execution": False, "fallback": False}


def check_chat_answer(completion, case):
    """Deterministic fixture check, not a general model-quality benchmark."""
    if case.get("expected_exact") is None and case.get("expected_contains") is None:
        return None
    if not completion["complete"]:
        return False
    content = completion["message"]["content"]
    return ((case.get("expected_exact") is None or content.strip() == case["expected_exact"])
            and (case.get("expected_contains") is None or case["expected_contains"] in content))

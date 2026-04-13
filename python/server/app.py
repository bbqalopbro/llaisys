"""
LLAISYS Chat Server — OpenAI-compatible Chat Completion API

整体架构:
  用户 HTTP 请求  →  FastAPI 路由 (app.py)
                           │
                    ┌──────┴──────┐
                    │SessionManager│  管理多会话 + KV-Cache 快照
                    └──────┬──────┘
                           │ submit()
                    ┌──────┴──────┐
                    │InferenceEngine│  请求队列 + 后台 worker 线程
                    └──────┬──────┘
                           │ ctypes
                    ┌──────┴──────┐
                    │  C++ Model  │  (Qwen2 + PagedAttention)
                    └─────────────┘

主要路由:
  POST /v1/chat/completions — OpenAI 兼容接口 (支持 stream=true SSE 流式)
  POST /v1/sessions         — 创建会话
  POST /v1/sessions/{id}/switch — 切换会话 (保存/恢复 KV-Cache)
  PUT  /v1/sessions/{id}/edit   — 编辑历史消息 (截断 KV-Cache 重新生成)
  GET  /health              — 健康检查
  GET  /stats               — 引擎统计 (排队数/并发数/总 token 数)

Usage:
    python -m server.app --model /path/to/model [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import os
import time
import uuid
from typing import AsyncGenerator, Optional

# Ensure the llaisys package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessageResponse,
    ChoiceDelta,
    UsageInfo,
    # Phase 4 models
    CreateSessionRequest,
    EditMessageRequest,
    RegenerateRequest,
    SessionInfo,
    SessionHistoryResponse,
    ChatMessage,
)
from server.session import SessionManager
from server.engine import InferenceEngine, SamplingParams

import llaisys
from llaisys.libllaisys import DeviceType

# ── Globals (initialised in startup) ─────────────────────────────────

app = FastAPI(title="LLAISYS Chat Server", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: llaisys.models.Qwen2 | None = None
TOKENIZER = None
MODEL_PATH: str = ""
DEVICE: DeviceType = DeviceType.CPU
SESSION_MGR: SessionManager | None = None
ENGINE: InferenceEngine | None = None

# ── Model Management ─────────────────────────────────────────────────

def load_model(model_path: str, device: str = "cpu",
               tp_size: int = 1, tp_rank: int = 0) -> None:
    """Load model and tokenizer (called once at startup)."""
    global MODEL, TOKENIZER, MODEL_PATH, DEVICE, SESSION_MGR, ENGINE

    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    # Resolve model path
    if os.path.isdir(model_path):
        resolved_path = model_path
    else:
        print(f"Downloading model {model_path} ...")
        resolved_path = snapshot_download(model_path)

    MODEL_PATH = resolved_path
    if device == "nvidia":
        DEVICE = DeviceType.NVIDIA
    elif device == "metax":
        DEVICE = DeviceType.METAX
    else:
        DEVICE = DeviceType.CPU

    print(f"Loading tokenizer from {resolved_path} ...")
    TOKENIZER = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=True,
    )

    print(f"Loading LLAISYS model from {resolved_path} (device={device}, tp_size={tp_size}, tp_rank={tp_rank}) ...")
    MODEL = llaisys.models.Qwen2(resolved_path, DEVICE,
                                  tp_size=tp_size, tp_rank=tp_rank)

    # Phase 4: 初始化 Session Manager
    SESSION_MGR = SessionManager(MODEL)

    # Phase 5 (项目#4): 初始化 InferenceEngine
    ENGINE = InferenceEngine(MODEL, TOKENIZER)
    ENGINE.start()

    print("Model ready. Session manager initialized. Inference engine started.")


def _reset_model() -> None:
    """Reset KV-cache position (no weight reload needed)."""
    MODEL.reset_cache()


# ── Helpers ──────────────────────────────────────────────────────────

def _encode_messages(messages: list) -> list[int]:
    """Apply chat template and tokenize."""
    conversation = [{"role": m.role, "content": m.content} for m in messages]
    text = TOKENIZER.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    return TOKENIZER.encode(text)


def _ensure_session(session_id: Optional[str]) -> str:
    """确保 session 存在且已激活, 返回 session_id."""
    session = SESSION_MGR.activate_or_create(session_id)
    return session.session_id


# ── Engine-based Non-streaming endpoint ──────────────────────────────

async def _generate_full_engine(request: ChatCompletionRequest, session_id: str) -> ChatCompletionResponse:
    """通过 InferenceEngine 生成完整回复 (非阻塞)."""
    input_ids = _encode_messages(request.messages)
    prompt_tokens = len(input_ids)

    params = SamplingParams(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    loop = asyncio.get_running_loop()
    req = ENGINE.submit(input_ids, params, session_id, stream=False, loop=loop)

    # 非阻塞等待: worker 线程处理完成后 future 会被 set_result
    generated_tokens = await req.future

    # 过滤 EOS
    eos = getattr(MODEL, '_end_token', 151643)
    if generated_tokens and generated_tokens[-1] == eos:
        generated_tokens = generated_tokens[:-1]

    text = TOKENIZER.decode(generated_tokens, skip_special_tokens=True)
    completion_tokens = len(generated_tokens)

    # Session: 记录消息
    for msg in request.messages:
        SESSION_MGR.add_message(session_id, msg.role, msg.content)
    SESSION_MGR.add_message(
        session_id, "assistant", text,
        cache_start_pos=prompt_tokens,
        cache_end_pos=prompt_tokens + completion_tokens,
    )

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessageResponse(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        session_id=session_id,
    )


# ── Engine-based Streaming endpoint ──────────────────────────────────

async def _generate_stream_engine(request: ChatCompletionRequest, session_id: str) -> AsyncGenerator[str, None]:
    """通过 InferenceEngine 流式生成 (非阻塞, 异步读取 token)."""
    input_ids = _encode_messages(request.messages)
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    params = SamplingParams(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    loop = asyncio.get_running_loop()
    req = ENGINE.submit(input_ids, params, session_id, stream=True, loop=loop)

    eos = getattr(MODEL, '_end_token', 151643)

    # Send initial role chunk
    initial_chunk = ChatCompletionStreamResponse(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChoiceDelta(role="assistant"),
                finish_reason=None,
            )
        ],
        session_id=session_id,
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream tokens from engine's output queue
    generated_tokens = []
    async for token_id in req.stream_tokens():
        if token_id == eos:
            break

        generated_tokens.append(token_id)
        text = TOKENIZER.decode([token_id], skip_special_tokens=True)
        if not text:
            continue

        chunk = ChatCompletionStreamResponse(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChoiceDelta(content=text),
                    finish_reason=None,
                )
            ],
            session_id=session_id,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk
    final_chunk = ChatCompletionStreamResponse(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop",
            )
        ],
        session_id=session_id,
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    # Session: 记录消息
    full_text = TOKENIZER.decode(generated_tokens, skip_special_tokens=True)
    prompt_tokens = len(input_ids)

    for msg in request.messages:
        SESSION_MGR.add_message(session_id, msg.role, msg.content)
    SESSION_MGR.add_message(
        session_id, "assistant", full_text,
        cache_start_pos=prompt_tokens,
        cache_end_pos=prompt_tokens + len(generated_tokens),
    )


# ── Legacy direct-call helpers (used by regenerate/edit) ─────────────

def _generate_full_direct(request: ChatCompletionRequest, session_id: str) -> ChatCompletionResponse:
    """直接调用模型生成 (阻塞, 用于 regenerate/edit 等内部场景)."""
    _reset_model()

    input_ids = _encode_messages(request.messages)
    prompt_tokens = len(input_ids)

    output_ids = MODEL.generate(
        input_ids,
        max_new_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
    )

    new_tokens = output_ids[prompt_tokens:]
    text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    cache_pos_after_gen = MODEL.get_cache_pos()

    for msg in request.messages:
        SESSION_MGR.add_message(session_id, msg.role, msg.content)
    SESSION_MGR.add_message(
        session_id, "assistant", text,
        cache_start_pos=prompt_tokens,
        cache_end_pos=cache_pos_after_gen,
    )

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessageResponse(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        session_id=session_id,
    )


# ── Regeneration helpers ─────────────────────────────────────────────

def _regenerate_full(session_id: str, request: RegenerateRequest) -> ChatCompletionResponse:
    """重新生成: 删除最后一条 assistant 消息, 从 truncated cache 重新推理."""
    truncate_pos, removed = SESSION_MGR.prepare_regenerate(session_id)

    # 获取会话中剩余的历史消息
    history = SESSION_MGR.get_history(session_id)
    if not history:
        raise HTTPException(status_code=400, detail="No messages to regenerate from")

    # 用剩余历史重新编码并生成
    input_ids = _encode_messages([ChatMessage(**m) for m in history])

    _reset_model()
    output_ids = MODEL.generate(
        input_ids,
        max_new_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
    )

    new_tokens = output_ids[len(input_ids):]
    text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)

    cache_pos = MODEL.get_cache_pos()
    SESSION_MGR.add_message(
        session_id, "assistant", text,
        cache_start_pos=len(input_ids),
        cache_end_pos=cache_pos,
    )

    return ChatCompletionResponse(
        model="deepseek-r1-distill-qwen-1.5b",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessageResponse(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(input_ids),
            completion_tokens=len(new_tokens),
            total_tokens=len(input_ids) + len(new_tokens),
        ),
        session_id=session_id,
    )


async def _regenerate_stream(session_id: str, request: RegenerateRequest) -> AsyncGenerator[str, None]:
    """流式重新生成."""
    truncate_pos, removed = SESSION_MGR.prepare_regenerate(session_id)

    history = SESSION_MGR.get_history(session_id)
    if not history:
        yield 'data: {"error": "No messages to regenerate from"}\n\n'
        return

    input_ids = _encode_messages([ChatMessage(**m) for m in history])

    _reset_model()
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    initial_chunk = ChatCompletionStreamResponse(
        id=chat_id, created=created,
        model="deepseek-r1-distill-qwen-1.5b",
        choices=[ChatCompletionStreamChoice(index=0, delta=ChoiceDelta(role="assistant"), finish_reason=None)],
        session_id=session_id,
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    generated_tokens = []
    for token_id in MODEL.generate_stream(
        input_ids,
        max_new_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
    ):
        if token_id == 151643:
            break
        generated_tokens.append(token_id)
        text = TOKENIZER.decode([token_id], skip_special_tokens=True)
        if not text:
            continue
        chunk = ChatCompletionStreamResponse(
            id=chat_id, created=created,
            model="deepseek-r1-distill-qwen-1.5b",
            choices=[ChatCompletionStreamChoice(index=0, delta=ChoiceDelta(content=text), finish_reason=None)],
            session_id=session_id,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    final_chunk = ChatCompletionStreamResponse(
        id=chat_id, created=created,
        model="deepseek-r1-distill-qwen-1.5b",
        choices=[ChatCompletionStreamChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        session_id=session_id,
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    full_text = TOKENIZER.decode(generated_tokens, skip_special_tokens=True)
    cache_pos = MODEL.get_cache_pos()
    SESSION_MGR.add_message(
        session_id, "assistant", full_text,
        cache_start_pos=len(input_ids),
        cache_end_pos=cache_pos,
    )


# ── Routes ───────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    session_id = _ensure_session(request.session_id)

    # Phase 5: 使用 InferenceEngine (非阻塞, 支持并发)
    if ENGINE is not None:
        if request.stream:
            return StreamingResponse(
                _generate_stream_engine(request, session_id),
                media_type="text/event-stream",
            )
        return await _generate_full_engine(request, session_id)

    # Fallback: 直接调用模型 (阻塞)
    return _generate_full_direct(request, session_id)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-r1-distill-qwen-1.5b",
                "object": "model",
                "owned_by": "llaisys",
            }
        ],
    }


@app.get("/health")
async def health():
    engine_stats = ENGINE.get_stats() if ENGINE else None
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "engine": engine_stats,
    }


# ── Engine stats endpoint ────────────────────────────────────────────

@app.get("/v1/engine/stats")
async def engine_stats():
    """获取推理引擎状态 (队列大小, 活跃请求数等)."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return ENGINE.get_stats()


# ── Phase 4: Session Management Routes ───────────────────────────────

@app.post("/v1/sessions")
async def create_session(request: CreateSessionRequest = None):
    """创建新会话."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    req = request or CreateSessionRequest()
    session = SESSION_MGR.create_session(req.session_id)
    return SessionInfo(
        session_id=session.session_id,
        message_count=0,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_active=False,
    )


@app.get("/v1/sessions")
async def list_sessions():
    """列出所有会话."""
    if SESSION_MGR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    sessions = SESSION_MGR.list_sessions()
    return {"sessions": [SessionInfo(**s) for s in sessions]}


@app.get("/v1/sessions/{session_id}")
async def get_session_history(session_id: str):
    """获取指定会话的对话历史."""
    if SESSION_MGR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    session = SESSION_MGR.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    history = SESSION_MGR.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(**m) for m in history],
    )


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话."""
    if SESSION_MGR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    ok = SESSION_MGR.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"status": "deleted", "session_id": session_id}


@app.post("/v1/sessions/{session_id}/switch")
async def switch_session(session_id: str):
    """切换到指定会话 (保存当前会话的 KV-Cache, 恢复目标会话)."""
    if SESSION_MGR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    ok = SESSION_MGR.switch_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"status": "switched", "session_id": session_id}


# ── Phase 4: Edit / Regenerate Routes ────────────────────────────────

@app.post("/v1/edit")
async def edit_message(request: EditMessageRequest):
    """编辑指定位置的消息, 截断后续对话, 可选重新生成."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    session_id = request.session_id
    session = SESSION_MGR.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        truncate_pos, removed = SESSION_MGR.edit_message(
            session_id, request.message_index, request.new_content
        )
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 添加编辑后的消息
    SESSION_MGR.add_message(session_id, "user", request.new_content)

    if not request.regenerate:
        return {
            "status": "edited",
            "session_id": session_id,
            "truncated_to": truncate_pos,
            "removed_messages": removed,
        }

    # 编辑后重新生成
    regen_req = RegenerateRequest(
        session_id=session_id,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    if request.stream:
        return StreamingResponse(
            _regenerate_stream(session_id, regen_req),
            media_type="text/event-stream",
        )

    return _regenerate_full(session_id, regen_req)


@app.post("/v1/regenerate")
async def regenerate(request: RegenerateRequest):
    """重新生成最后一条 assistant 回复."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    session = SESSION_MGR.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

    SESSION_MGR.switch_session(request.session_id)

    if request.stream:
        return StreamingResponse(
            _regenerate_stream(request.session_id, request),
            media_type="text/event-stream",
        )

    return _regenerate_full(request.session_id, request)


# ── Static files (Web UI) ───────────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chat Server")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Model path or HuggingFace repo id")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "nvidia", "metax"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallelism degree (default: 1 = single device)")
    parser.add_argument("--tp-rank", type=int, default=0,
                        help="Current TP rank (0-based, for manual launch)")
    args = parser.parse_args()

    load_model(args.model, args.device, tp_size=args.tp_size, tp_rank=args.tp_rank)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

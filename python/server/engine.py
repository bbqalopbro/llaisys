"""
LLAISYS Inference Engine — 请求队列 + 异步推理服务 (Continuous Batching)

核心架构:
  ┌──────────┐     submit()    ┌──────────────┐
  │ FastAPI  │ ───────────────→│ RequestQueue │  (线程安全, Condition 通知)
  │ (app.py) │                 └──────┬───────┘
  └──────────┘                        │ get_pending()
                                      ↓
                               ┌──────────────┐
                               │InferenceEngine│  (后台 worker 线程)
                               │  _worker_loop │
                               └──────┬───────┘
                                      │ ctypes 调用
                                      ↓
                               ┌──────────────┐
                               │ C++ BatchCtx │  (PagedAttention KV-Cache)
                               └──────────────┘

worker 循环的 4 个阶段:
  1. Decode: 优先对已完成 prompt 的 slot 执行一步 Graph decode
  2. Admit/Prefill: 分配 slot，按剩余 token 预算推进 prompt 分块
  3. Finish: 完成的请求移出 batch, KV-Cache 存入前缀树池
  4. Wait:   空闲时阻塞等待新请求 (避免 busy loop)

关键设计:
  - 跨线程通信: worker 线程 → asyncio 主线程 via loop.call_soon_threadsafe
  - Preemption: block 不足时驱逐已生成最多 token 的请求 (保存快照 → 重入队列)
  - 前缀匹配: 新请求优先从前缀树池复用 KV-Cache, 跳过已 prefill 的部分
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Sequence

logger = logging.getLogger("llaisys.engine")


# ── 请求状态 ──────────────────────────────────────────────────────

class RequestStatus(str, Enum):
    WAITING = "waiting"        # 在队列中等待
    PREFILLING = "prefilling"  # 正在 prefill
    DECODING = "decoding"      # 正在 decode (逐 token 生成)
    DONE = "done"              # 完成
    CANCELLED = "cancelled"    # 已取消
    ERROR = "error"            # 出错


# ── 采样参数 ─────────────────────────────────────────────────────

@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 512


# ── 推理请求 ─────────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    """单个推理请求."""
    request_id: str
    input_ids: List[int]           # tokenize 后的完整输入序列
    params: SamplingParams
    session_id: str
    stream: bool = False

    # 异步通信
    future: asyncio.Future = field(default=None, repr=False)
    output_queue: asyncio.Queue = field(default=None, repr=False)
    loop: asyncio.AbstractEventLoop = field(default=None, repr=False)

    # 生成状态
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: List[int] = field(default_factory=list)
    last_token: int = 0
    num_computed_tokens: int = 0  # tokens actually present in KV, excludes last sampled token
    kv_cache_snapshot: object = None  # C++ KV-Cache 快照句柄

    # 时间戳
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0

    # 错误信息
    error_message: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"req-{uuid.uuid4().hex[:12]}"

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.DONE, RequestStatus.CANCELLED, RequestStatus.ERROR)

    async def stream_tokens(self) -> AsyncGenerator[int, None]:
        """异步生成器, 逐个 yield token_id."""
        if self.output_queue is None:
            return
        while True:
            token = await self.output_queue.get()
            if token is None:  # 结束信号
                break
            yield token


# ── 请求队列 ─────────────────────────────────────────────────────

class RequestQueue:
    """线程安全的请求池."""

    def __init__(self, max_size: int = 1024):
        self._lock = threading.Lock()
        self._waiting: List[InferenceRequest] = []
        self._active: Dict[str, InferenceRequest] = {}  # request_id → request
        self._max_size = max_size
        self._not_empty = threading.Condition(self._lock)

    def submit(self, request: InferenceRequest) -> bool:
        """提交请求到等待队列. 返回是否成功."""
        with self._lock:
            if len(self._waiting) >= self._max_size:
                return False
            self._waiting.append(request)
            self._not_empty.notify()
            return True

    def get_pending(self, max_count: int = 1) -> List[InferenceRequest]:
        """从等待队列取出至多 max_count 个请求."""
        with self._lock:
            count = min(max_count, len(self._waiting))
            if count == 0:
                return []
            batch = self._waiting[:count]
            self._waiting = self._waiting[count:]
            for req in batch:
                self._active[req.request_id] = req
            return batch

    def transition(self, request, status):
        """Do not overwrite a cancellation arriving during GPU work or snapshot I/O."""
        with self._lock:
            if request.status == RequestStatus.CANCELLED:
                return False
            request.status = status
            return True

    def pop_cancelled(self):
        with self._lock:
            cancelled = [r for r in self._waiting if r.status == RequestStatus.CANCELLED]
            self._waiting = [r for r in self._waiting if r.status != RequestStatus.CANCELLED]
            return cancelled

    def requeue(self, request: InferenceRequest):
        """An already admitted request keeps its place even if new submissions fill the queue."""
        with self._lock:
            self._active.pop(request.request_id, None)
            self._waiting.append(request)
            self._not_empty.notify()

    def wait_for_requests(self, timeout: float = 0.1) -> bool:
        """等待直到有请求可用或超时. 返回是否有请求."""
        with self._not_empty:
            if not self._waiting:
                self._not_empty.wait(timeout)
            return len(self._waiting) > 0

    def mark_done(self, request_id: str):
        """标记请求完成, 从活跃集合中移除."""
        with self._lock:
            self._active.pop(request_id, None)

    def cancel(self, request_id: str) -> bool:
        """取消请求. 返回是否成功."""
        with self._lock:
            # 从等待队列中移除
            for i, req in enumerate(self._waiting):
                if req.request_id == request_id:
                    req.status = RequestStatus.CANCELLED
                    self._not_empty.notify()
                    return True
            # 标记活跃请求为取消
            req = self._active.get(request_id)
            if req:
                req.status = RequestStatus.CANCELLED
                return True
            return False

    @property
    def waiting_count(self) -> int:
        with self._lock:
            return len(self._waiting)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._waiting) == 0 and len(self._active) == 0


# ── 推理引擎 ─────────────────────────────────────────────────────

class InferenceEngine:
    """
    后台推理引擎 — 连续批处理 (Continuous Batching) with PagedAttention.
    
    每轮迭代:
      1. Decode 优先：已就绪 slot 执行一步批量 decode（独立采样参数）
      2. 动态 Admission 与 chunked prefill：共享每轮 token 预算
      3. 完成的请求移出 batch, 释放 slot
      4. Preemption: block 不足时 evict 低优先级请求
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        max_batch_size: int = 4,
        max_seq_per_slot: int = 2048,
        max_queue_size: int = 1024,
        block_watermark: float = 0.1,
        capture_sizes: Optional[Sequence[int]] = None,
        prefill_chunk_size: int = 256,
        max_num_batched_tokens: int = 512,
        scheduler_trace: bool = False,
    ):
        if max_batch_size < 1 or max_seq_per_slot < 1 or prefill_chunk_size < 1:
            raise ValueError("batch, sequence and chunk sizes must be positive")
        if max_num_batched_tokens < max_batch_size:
            raise ValueError("token budget must cover one decode token per slot")
        self.prefill_chunk_size = prefill_chunk_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self._trace_enabled = scheduler_trace
        self._scheduler_trace = deque(maxlen=512)
        self._prefill_chunks = 0
        self._prefill_tokens = 0
        self._prefix_hits = 0
        self._prefix_tokens = 0
        self._max_scheduled_tokens = 0
        self._iteration = 0
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_per_slot = max_seq_per_slot
        self.block_watermark = block_watermark
        self.capture_sizes = None if capture_sizes is None else list(capture_sizes)

        self.queue = RequestQueue(max_size=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._ready = threading.Event()
        self._startup_error = None
        self._graph_stats = {}
        self._batch_size_histogram = {}
        self._eos_token_id = getattr(model, '_end_token', 151643)

        # KV-Cache 前缀树池
        self._cache_pool = None
        try:
            self._cache_pool = model.create_cache_pool()
            logger.info("KV-Cache prefix pool created")
        except Exception:
            logger.warning("KV-Cache prefix pool not available")

        # 统计
        self._total_requests = 0
        self._total_tokens = 0
        self._total_preemptions = 0

    def start(self):
        """启动后台 worker 线程."""
        if self._running:
            return
        self._ready.clear()
        self._startup_error = None
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="inference-engine",
            daemon=True,
        )
        self._worker_thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self._worker_thread.join()
            raise RuntimeError("Inference engine initialization failed") from self._startup_error
        logger.info(f"Inference engine started (max_batch_size={self.max_batch_size})")

    def stop(self):
        """停止后台 worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join()
            self._worker_thread = None
        logger.info("Inference engine stopped")

    def submit(
        self,
        input_ids: List[int],
        params: SamplingParams,
        session_id: str,
        stream: bool = False,
        loop: asyncio.AbstractEventLoop = None,
    ) -> InferenceRequest:
        """提交推理请求 (非阻塞).
        
        Returns:
            InferenceRequest 对象, 可通过 .future 等待结果,
            或通过 .stream_tokens() 流式读取.
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

        request = InferenceRequest(
            request_id=f"req-{uuid.uuid4().hex[:12]}",
            input_ids=list(input_ids),
            params=params,
            session_id=session_id,
            stream=stream,
            future=loop.create_future(),
            output_queue=asyncio.Queue() if stream else None,
            loop=loop,
        )

        if (not input_ids or params.max_tokens < 1 or
                len(input_ids) + params.max_tokens - 1 > self.max_seq_per_slot):
            self._finish_request(request, ValueError("empty prompt or request exceeds slot sequence capacity"))
            return request
        if not self._running:
            self._finish_request(request, RuntimeError("inference engine is not running"))
            return request
        request.future.add_done_callback(
            lambda future: self.queue.cancel(request.request_id) if future.cancelled() else None)
        if not self.queue.submit(request):
            self._finish_request(request, RuntimeError("Request queue is full"))
        else:
            logger.debug(f"Request {request.request_id} submitted (queue={self.queue.waiting_count})")

        return request

    # ── 连续批处理 Worker 循环 ───────────────────────────────────

    def _estimate_blocks_needed(self, prompt_len: int, max_tokens: int, block_size: int) -> int:
        """Estimate blocks needed for a new request."""
        total_tokens = prompt_len + max_tokens
        return (total_tokens + block_size - 1) // block_size

    def _can_admit(self, batch_ctx, prompt_len: int, max_tokens: int) -> bool:
        """Check if there are enough free blocks to admit a new request."""
        block_size = batch_ctx.get_block_size()
        if block_size <= 0:
            return True
        needed = self._estimate_blocks_needed(prompt_len, max_tokens, block_size)
        free = batch_ctx.get_free_blocks()
        total = batch_ctx.get_total_blocks()
        watermark = int(total * self.block_watermark)
        return free >= needed + watermark

    def _try_preempt(self, batch_ctx, running: Dict[int, InferenceRequest],
                     free_slots: List[int], needed_blocks: int, exclude_slot=None) -> bool:
        candidates = [sid for sid in running if sid != exclude_slot
                      and running[sid].status != RequestStatus.CANCELLED
                      and (running[sid].num_computed_tokens or running[sid].generated_tokens)]
        if not candidates:
            return False
        victim_slot = max(candidates, key=lambda sid: len(running[sid].generated_tokens))
        req = running[victim_slot]
        try:
            snapshot = batch_ctx.slot_save(victim_slot)
            if not snapshot:
                return False
        except Exception:
            logger.exception("Could not snapshot request for preemption")
            return False
        req.kv_cache_snapshot = snapshot
        batch_ctx.slot_reset(victim_slot)
        del running[victim_slot]
        free_slots.append(victim_slot)
        self.queue.transition(req, RequestStatus.WAITING)
        self.queue.requeue(req)
        self._total_preemptions += 1
        self._trace("preempt", request=req.request_id)
        return True

    def _trace(self, kind, **fields):
        if getattr(self, "_trace_enabled", False):
            self._scheduler_trace.append(dict(iteration=self._iteration, kind=kind,
                                              time=time.perf_counter(), **fields))

    def _destroy_snapshot(self, req):
        if req.kv_cache_snapshot is not None:
            self.model.destroy_snapshot(req.kv_cache_snapshot)
            req.kv_cache_snapshot = None

    def _worker_loop(self):
        """Decode first; spend the remaining per-iteration token budget on prompt chunks.

        Prefill runs eagerly between decode graph replays. A PREFILLING slot is
        never submitted to the decode graph and publishes no intermediate tokens.
        """
        batch_ctx = None
        running: Dict[int, InferenceRequest] = {}
        free_slots = list(range(self.max_batch_size))
        cursor = 0

        def retire(sid, error=None):
            req = running.pop(sid)
            if error is None and req.status != RequestStatus.CANCELLED and self._cache_pool:
                # The last sampled output has not been fed through the model yet.
                tokens = req.input_ids + req.generated_tokens[:-1]
                snapshot = None
                try:
                    if tokens and batch_ctx.slot_get_pos(sid) == len(tokens):
                        snapshot = batch_ctx.slot_save(sid)
                        if snapshot:
                            self.model.cache_pool_insert(self._cache_pool, tokens, snapshot)
                            snapshot = None  # ownership transferred to the pool
                except Exception:
                    logger.exception("Prefix cache insertion failed")
                finally:
                    if snapshot:
                        self.model.destroy_snapshot(snapshot)
            batch_ctx.slot_reset(sid)
            free_slots.append(sid)
            self._finish_request(req, error)
            self.queue.mark_done(req.request_id)

        def publish(sid, token):
            req = running[sid]
            if not self.queue.transition(req, RequestStatus.DECODING):
                retire(sid)
                return
            req.generated_tokens.append(int(token))
            req.last_token = int(token)
            self._send_token_async(req, int(token))
            if token == self._eos_token_id or len(req.generated_tokens) >= req.params.max_tokens:
                retire(sid)

        try:
            batch_ctx = self.model.create_batch_context(self.max_batch_size, self.max_seq_per_slot)
            if self.capture_sizes is not None:
                batch_ctx.set_capture_sizes(self.capture_sizes)
            if hasattr(batch_ctx, "prepare_graphs"):
                batch_ctx.prepare_graphs()
                self._graph_stats = batch_ctx.graph_stats()
            self._ready.set()
            while self._running:
                self._iteration += 1
                budget = self.max_num_batched_tokens
                for req in self.queue.pop_cancelled():
                    self._destroy_snapshot(req)
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)
                for sid in list(running):
                    if running[sid].status == RequestStatus.CANCELLED:
                        retire(sid)

                # Existing decoders get a step before any new prompt computation.
                decoding = [sid for sid, req in running.items() if req.status == RequestStatus.DECODING]
                if decoding:
                    try:
                        bs = batch_ctx.get_block_size()
                        needed = sum(batch_ctx.slot_get_pos(sid) % bs == 0 for sid in decoding)
                        while needed > batch_ctx.get_free_blocks():
                            if not self._try_preempt(batch_ctx, running, free_slots, needed):
                                raise RuntimeError("insufficient KV blocks for decode")
                            decoding = [sid for sid in decoding if sid in running]
                            needed = sum(batch_ctx.slot_get_pos(sid) % bs == 0 for sid in decoding)
                        if decoding:
                            values = batch_ctx.decode_per_request(
                                decoding, [running[s].last_token for s in decoding],
                                [running[s].params.temperature for s in decoding],
                                [running[s].params.top_k for s in decoding],
                                [running[s].params.top_p for s in decoding])
                            budget -= len(decoding)
                            self._batch_size_histogram[len(decoding)] = self._batch_size_histogram.get(len(decoding), 0) + 1
                            self._trace("decode", requests=[running[s].request_id for s in decoding], tokens=len(decoding))
                            for sid, token in zip(decoding, values):
                                running[sid].num_computed_tokens += 1
                                publish(sid, int(token))
                    except Exception as exc:
                        logger.exception("Batch decode failed")
                        for sid in decoding:
                            if sid in running:
                                retire(sid, exc)

                # Admission assigns a slot, but does not run an unbounded prompt.
                # Snapshot restoration is exceptional and never consumes a sampled token.
                pending = self.queue.waiting_count
                while free_slots and pending > 0:
                    pending -= 1
                    items = self.queue.get_pending(1)
                    if not items:
                        break
                    req = items[0]
                    if req.status == RequestStatus.CANCELLED:
                        self._destroy_snapshot(req)
                        self._finish_request(req)
                        self.queue.mark_done(req.request_id)
                        continue
                    bs = batch_ctx.get_block_size()
                    restore_blocks = ((req.num_computed_tokens + bs - 1) // bs
                                      if req.kv_cache_snapshot is not None else 1)
                    watermark = int(batch_ctx.get_total_blocks() * self.block_watermark) if running else 0
                    if restore_blocks + watermark > batch_ctx.get_free_blocks():
                        self.queue.requeue(req)
                        break
                    sid = free_slots.pop(0)
                    running[sid] = req
                    req.started_at = req.started_at or time.time()
                    try:
                        if req.kv_cache_snapshot is not None:
                            batch_ctx.slot_restore(sid, req.kv_cache_snapshot)
                            self._destroy_snapshot(req)
                            if batch_ctx.slot_get_pos(sid) != req.num_computed_tokens:
                                raise RuntimeError("preemption snapshot position mismatch")
                        else:
                            batch_ctx.slot_reset(sid)
                            if self._cache_pool:
                                snapshot, matched = self.model.cache_pool_lookup(self._cache_pool, req.input_ids)
                                if (snapshot and matched > 0 and
                                        (matched + bs - 1) // bs <= batch_ctx.get_free_blocks()):
                                    batch_ctx.slot_restore(sid, snapshot)
                                    # Full prefix hit still needs logits: recompute the last prompt token.
                                    matched = min(matched, len(req.input_ids) - 1)
                                    batch_ctx.slot_truncate(sid, matched)
                                    req.num_computed_tokens = matched
                                    self._prefix_hits += 1
                                    self._prefix_tokens += matched
                                    self._trace("prefix", request=req.request_id, tokens=matched)
                        status = RequestStatus.DECODING if req.generated_tokens else RequestStatus.PREFILLING
                        if not self.queue.transition(req, status):
                            retire(sid)
                            continue
                        self._trace("admit", request=req.request_id, position=req.num_computed_tokens)
                    except Exception as exc:
                        self._destroy_snapshot(req)
                        retire(sid, exc)

                # Rotate the starting slot so several long prompts share the budget fairly.
                prefilling = [sid for sid, req in running.items() if req.status == RequestStatus.PREFILLING]
                if prefilling:
                    offset = cursor % len(prefilling)
                    prefilling = prefilling[offset:] + prefilling[:offset]
                    cursor += 1
                for sid in prefilling:
                    if budget <= 0 or not self._running:
                        break
                    if sid not in running:
                        continue
                    req = running[sid]
                    if req.status == RequestStatus.CANCELLED:
                        retire(sid)
                        continue
                    start = req.num_computed_tokens
                    count = min(self.prefill_chunk_size, budget, len(req.input_ids) - start)
                    final = start + count == len(req.input_ids)
                    try:
                        bs = batch_ctx.get_block_size()
                        needed = (start + count + bs - 1) // bs - (start + bs - 1) // bs
                        while needed > batch_ctx.get_free_blocks():
                            if not self._try_preempt(batch_ctx, running, free_slots, needed, exclude_slot=sid):
                                raise RuntimeError("insufficient KV blocks for prefill chunk")
                        value = batch_ctx.prefill_chunk(sid, req.input_ids[start:start + count], final=final,
                            temperature=req.params.temperature, top_k=req.params.top_k, top_p=req.params.top_p)
                        req.num_computed_tokens += count
                        budget -= count
                        self._prefill_chunks += 1
                        self._prefill_tokens += count
                        self._trace("prefill", request=req.request_id, start=start, tokens=count, final=final)
                        if req.status == RequestStatus.CANCELLED:
                            retire(sid)
                        elif final:
                            if value is None:
                                raise RuntimeError("final prefill chunk did not return a token")
                            publish(sid, int(value))
                        elif value is not None:
                            raise RuntimeError("intermediate prefill chunk unexpectedly sampled")
                    except Exception as exc:
                        logger.exception("Prefill chunk failed")
                        retire(sid, exc)
                self._max_scheduled_tokens = max(self._max_scheduled_tokens,
                                                  self.max_num_batched_tokens - budget)
                if hasattr(batch_ctx, "graph_stats") and self._iteration % 32 == 0:
                    self._graph_stats = batch_ctx.graph_stats()
                if not running and not self.queue.waiting_count:
                    self.queue.wait_for_requests(timeout=0.05)
        except Exception as exc:
            if not self._ready.is_set():
                self._startup_error = exc
            logger.exception("Inference worker failed")
            for sid in list(running):
                retire(sid, exc)
        finally:
            self._running = False
            self._ready.set()
            for sid in list(running):
                running[sid].status = RequestStatus.CANCELLED
                retire(sid)
            while self.queue.waiting_count:
                for req in self.queue.get_pending(1):
                    req.status = RequestStatus.CANCELLED
                    self._destroy_snapshot(req)
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)
            if batch_ctx is not None:
                if hasattr(batch_ctx, "graph_stats"):
                    self._graph_stats = batch_ctx.graph_stats()
                del batch_ctx
            logger.info("Worker loop exited")

    def _send_token_async(self, req: InferenceRequest, token_id: int):
        """线程安全地向 asyncio 队列发送 token."""
        if req.loop and req.output_queue:
            req.loop.call_soon_threadsafe(req.output_queue.put_nowait, token_id)

    def _finish_request(self, req: InferenceRequest, error: Exception = None):
        """标记请求完成, 设置 future 结果."""
        req.finished_at = time.time()

        if error:
            req.status = RequestStatus.ERROR
            req.error_message = str(error)
        elif req.status != RequestStatus.CANCELLED:
            req.status = RequestStatus.DONE

        # Check future state on its owning loop, not before enqueueing the callback.
        def settle():
            if req.future is not None and not req.future.done():
                if error:
                    req.future.set_exception(error)
                elif req.status == RequestStatus.CANCELLED:
                    req.future.cancel()
                else:
                    req.future.set_result(req.generated_tokens)
            if req.output_queue is not None:
                req.output_queue.put_nowait(None)
        if req.loop:
            req.loop.call_soon_threadsafe(settle)

        elapsed = req.finished_at - req.started_at if req.started_at > 0 else 0
        n_tokens = len(req.generated_tokens)
        tps = n_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Request {req.request_id} finished: "
            f"{n_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s) "
            f"status={req.status.value}"
        )

        self._total_requests += 1
        self._total_tokens += n_tokens

    # ── 状态查询 ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """返回引擎状态统计."""
        return {
            "running": self._running,
            "waiting_requests": self.queue.waiting_count,
            "active_requests": self.queue.active_count,
            "max_batch_size": self.max_batch_size,
            "total_requests_served": self._total_requests,
            "total_tokens_generated": self._total_tokens,
            "decode_batch_size_histogram": dict(self._batch_size_histogram),
            "batch_graph": self._graph_stats,
            "prefill_chunk_size": self.prefill_chunk_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "prefill_chunks": self._prefill_chunks,
            "prefill_tokens": self._prefill_tokens,
            "prefix_hits": self._prefix_hits,
            "prefix_tokens_reused": self._prefix_tokens,
            "max_scheduled_tokens": self._max_scheduled_tokens,
            "scheduler_trace": list(self._scheduler_trace),
            "total_preemptions": self._total_preemptions,
        }

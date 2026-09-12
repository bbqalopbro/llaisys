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
                                      │ SchedulePlan / batch runtime
                                      ↓
                               ┌──────────────┐
                               │ C++ BatchCtx │  (PagedAttention KV-Cache)
                               └──────────────┘

worker 循环的 4 个阶段:
  1. Admit:  从队列取新请求, 检查 block 容量, 必要时 preempt 低优先级请求
  2. Decode: 对所有活跃 slot 执行一步批量 decode (per-request 采样参数)
  3. Finish: 完成的请求移出 batch，释放 request block 引用
  4. Wait:   空闲时阻塞等待新请求 (避免 busy loop)

关键设计:
  - 跨线程通信: worker 线程 → asyncio 主线程 via loop.call_soon_threadsafe
  - Preemption: block 不足时驱逐已生成最多 token 的请求 (保存快照 → 重入队列)
  - 前缀匹配: 以完整 cache block 为粒度，hash 命中后直接共享物理 block
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Sequence

logger = logging.getLogger("llaisys.engine")


class _RequestCancelled(Exception):
    pass


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
    kv_cache_snapshot: object = None  # C++ KV-Cache 快照句柄

    # 时间戳
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0

    # 错误信息
    error_message: str = ""
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    completion_scheduled: bool = field(default=False, repr=False)

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
            self._active.pop(request.request_id, None)
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
            # Let the worker perform completion and snapshot cleanup even for
            # waiting requests; removing them here strands futures/streams.
            for req in self._waiting:
                if req.request_id == request_id:
                    req.cancelled.set()
                    req.status = RequestStatus.CANCELLED
                    self._not_empty.notify()
                    return True
            # 标记活跃请求为取消
            req = self._active.get(request_id)
            if req:
                req.cancelled.set()
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

    def take_cancelled(self):
        with self._lock:
            cancelled = [req for req in self._waiting if req.cancelled.is_set()]
            self._waiting = [req for req in self._waiting if not req.cancelled.is_set()]
            return cancelled

    def drain(self):
        with self._lock:
            requests = {req.request_id: req for req in self._waiting}
            requests.update(self._active)
            self._waiting.clear()
            self._active.clear()
            return list(requests.values())


# ── 推理引擎 ─────────────────────────────────────────────────────

class InferenceEngine:
    """
    后台推理引擎 — 连续批处理 (Continuous Batching) with PagedAttention.
    
    每轮迭代:
      1. 动态 Admission: 检查 block 容量后从队列取新请求 → 分配 slot → prefill
      2. Per-request 采样: 对所有活跃 slot 执行一步批量 decode（独立采样参数）
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
        prefill_chunk_size: int = 256,
        enable_block_prefix_cache: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_per_slot = max_seq_per_slot
        self.block_watermark = block_watermark
        self.prefill_chunk_size = max(1, prefill_chunk_size)
        self.enable_block_prefix_cache = enable_block_prefix_cache

        # Optional model capability check. Unsupported policies must fail before
        # admission, not silently become a different cache/execution strategy.
        validate_engine = getattr(model, "validate_engine_config", None)
        if validate_engine is not None:
            validate_engine(max_batch_size=max_batch_size, max_seq_per_slot=max_seq_per_slot,
                            prefill_chunk_size=self.prefill_chunk_size,
                            enable_block_prefix_cache=enable_block_prefix_cache)

        self.queue = RequestQueue(max_size=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._ready = threading.Event()
        self._worker_error = None
        self._eos_token_id = getattr(model, '_end_token', 151643)

        # 统计
        self._total_requests = 0
        self._total_tokens = 0
        self._total_preemptions = 0

    def start(self):
        """启动后台 worker 线程."""
        if self._running:
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            raise RuntimeError("previous inference worker has not exited")
        self._ready.clear()
        self._worker_error = None
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="inference-engine",
            daemon=True,
        )
        self._worker_thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError("inference worker initialization is still running")
        if self._worker_error is not None:
            raise RuntimeError("inference worker initialization failed") from self._worker_error
        logger.info(f"Inference engine started (max_batch_size={self.max_batch_size})")

    def stop(self, timeout: float = 30):
        """停止后台 worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                raise TimeoutError("inference worker is still finishing an in-flight execution")
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
            params=replace(params),
            session_id=session_id,
            stream=stream,
            future=loop.create_future(),
            output_queue=asyncio.Queue() if stream else None,
            loop=loop,
        )

        try:
            if self._worker_error is not None:
                raise RuntimeError("inference worker is unavailable") from self._worker_error
            if not request.input_ids or type(params.max_tokens) is not int or params.max_tokens <= 0:
                raise ValueError("nonempty input_ids and positive max_tokens are required")
            if len(request.input_ids) + params.max_tokens - 1 > self.max_seq_per_slot:
                raise ValueError("request exceeds its slot sequence capacity")
            validator = getattr(self.model, "validate_request", None)
            if validator is not None:
                validator(request.input_ids, request.params, self.max_seq_per_slot)
            if not self.queue.submit(request):
                raise RuntimeError("Request queue is full")
        except Exception as error:
            self._finish_request(request, error=error)
        else:
            logger.debug(f"Request {request.request_id} submitted (queue={self.queue.waiting_count})")
            request.future.add_done_callback(
                lambda future: self.cancel(request.request_id) if future.cancelled() else None)

        return request

    def cancel(self, request_id: str) -> bool:
        return self.queue.cancel(request_id)

    # ── 连续批处理 Worker 循环 ───────────────────────────────────

    def _estimate_blocks_needed(self, prompt_len: int, max_tokens: int, block_size: int) -> int:
        """Estimate blocks needed for a new request."""
        # The last sampled token is returned, not fed through another decode.
        total_tokens = prompt_len + max_tokens - 1
        return (total_tokens + block_size - 1) // block_size

    def _can_admit(self, batch_ctx, prompt_len: int, max_tokens: int, reserved_blocks: int = 0) -> bool:
        """Check if there are enough free blocks to admit a new request."""
        block_size = batch_ctx.get_block_size()
        if block_size <= 0:
            return True
        needed = self._estimate_blocks_needed(prompt_len, max_tokens, block_size)
        free = batch_ctx.get_free_blocks()
        total = batch_ctx.get_total_blocks()
        watermark = int(total * self.block_watermark)
        return free >= needed + watermark + reserved_blocks

    def _reserved_growth(self, batch_ctx, running):
        position = getattr(batch_ctx, "slot_get_pos", None)
        size = batch_ctx.get_block_size()
        if position is None or size <= 0:
            return 0
        return sum(max(0, self._estimate_blocks_needed(len(req.input_ids), req.params.max_tokens, size)
                       - (position(slot_id) + size - 1) // size) for slot_id, req in running.items())

    def _requeue(self, req):
        req.status = RequestStatus.WAITING
        if not self.queue.submit(req):
            self._finish_request(req, error=RuntimeError("request requeue is full"))
            self.queue.mark_done(req.request_id)

    def _prefill_prompt(self, batch_ctx, slot_id: int,
                        req: InferenceRequest, start_pos: int = 0) -> int:
        """Run scheduler-owned chunked prefill and return the first output token."""
        all_tokens = req.input_ids
        if not all_tokens or start_pos < 0 or start_pos >= len(all_tokens):
            raise ValueError("input_ids must not be empty")
        tokens = all_tokens[start_pos:]
        if req.cancelled.is_set():
            raise _RequestCancelled()
        chunk_fn = getattr(batch_ctx, "prefill_chunk", None)
        if start_pos == 0 and (chunk_fn is None or
                               len(tokens) <= self.prefill_chunk_size):
            return int(batch_ctx.prefill(
                slot_id=slot_id,
                token_ids=tokens,
                temperature=req.params.temperature,
                top_k=req.params.top_k,
                top_p=req.params.top_p,
            ))

        if chunk_fn is None:
            raise RuntimeError("prefix reuse requires incremental prefill support")
        if start_pos == 0:
            batch_ctx.slot_reset(slot_id)
        cursor = start_pos
        result = None
        while cursor < len(all_tokens):
            if req.cancelled.is_set() or (hasattr(self, "_running") and not self._running):
                raise _RequestCancelled()
            end_pos = min(cursor + self.prefill_chunk_size, len(all_tokens))
            result = chunk_fn(
                slot_id=slot_id,
                token_ids=all_tokens[cursor:end_pos],
                start_pos=cursor,
                is_last_chunk=end_pos == len(all_tokens),
                temperature=req.params.temperature,
                top_k=req.params.top_k,
                top_p=req.params.top_p,
            )
            cursor = end_pos
        if result is None:
            raise RuntimeError("final prefill chunk did not produce a token")
        return int(result)

    def _try_preempt(self, batch_ctx, running: Dict[int, InferenceRequest],
                     free_slots: List[int], needed_blocks: int) -> bool:
        """Evict the request with most generated tokens to free blocks.
        
        Returns True if a request was successfully preempted.
        """
        if not running:
            return False

        # Evict the request with most generated tokens (least priority)
        victim_slot = max(running.keys(),
                          key=lambda s: len(running[s].generated_tokens))
        victim_req = running[victim_slot]

        logger.info(
            f"Preempting request {victim_req.request_id} "
            f"(slot={victim_slot}, {len(victim_req.generated_tokens)} tokens generated)"
        )

        # Save KV-Cache snapshot before eviction
        try:
            snapshot = batch_ctx.slot_save(victim_slot)
            victim_req.kv_cache_snapshot = snapshot
        except Exception as e:
            logger.error(f"Failed to save snapshot for preemption: {e}")
            return False
        if snapshot is None:
            return False

        # Release the slot
        batch_ctx.slot_reset(victim_slot)
        running.pop(victim_slot)
        free_slots.append(victim_slot)

        # Re-queue the preempted request
        self._requeue(victim_req)
        self._total_preemptions += 1

        return True

    def _worker_loop(self):
        """后台 worker 主循环 — 连续批处理 with dynamic admission + preemption."""
        logger.info("Continuous batching worker loop started (paged mode)")

        batch_ctx = None
        running: Dict[int, InferenceRequest] = {}
        free_slots: List[int] = list(range(self.max_batch_size))
        try:
            batch_ctx = self.model.create_batch_context(self.max_batch_size, self.max_seq_per_slot)
            self._ready.set()
            self._run_batches(batch_ctx, running, free_slots)
        except Exception as error:
            self._worker_error = error
            logger.exception("Inference worker failed")
        finally:
            self._running = False
            self._ready.set()
            if batch_ctx is not None:
                for slot_id in range(self.max_batch_size):
                    try:
                        batch_ctx.slot_reset(slot_id)
                    except Exception as error:
                        self._worker_error = error
                        logger.exception("Failed to reset slot during worker shutdown")
            for req in self.queue.drain():
                self._finish_request(req, error=self._worker_error or RuntimeError("inference engine stopped"))
            close = getattr(batch_ctx, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as error:
                    self._worker_error = error
                    logger.exception("Failed to close batch context")

    def _run_batches(self, batch_ctx, running, free_slots):

        while self._running:
            for req in self.queue.take_cancelled():
                self._finish_request(req)
            for slot_id, req in list(running.items()):
                if req.cancelled.is_set():
                    batch_ctx.slot_reset(slot_id)
                    running.pop(slot_id)
                    free_slots.append(slot_id)
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)
            # ── 1. Admit: dynamic admission based on block availability ──
            while self._running and free_slots and self.queue.waiting_count > 0:
                # Peek at next request to check block budget
                new_requests = self.queue.get_pending(max_count=1)
                if not new_requests:
                    break
                req = new_requests[0]

                if req.cancelled.is_set() or req.status == RequestStatus.CANCELLED:
                    req.cancelled.set()
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)
                    continue

                block_size = batch_ctx.get_block_size()
                if block_size > 0 and self._estimate_blocks_needed(len(req.input_ids), req.params.max_tokens, block_size) > (
                        batch_ctx.get_total_blocks() - int(batch_ctx.get_total_blocks() * self.block_watermark)):
                    self._finish_request(req, error=ValueError("request budget exceeds total cache capacity"))
                    self.queue.mark_done(req.request_id)
                    continue

                # Dynamic admission: check block capacity
                preempted = False
                if not self._can_admit(batch_ctx, len(req.input_ids), req.params.max_tokens,
                                       self._reserved_growth(batch_ctx, running)):
                    # Try preemption to free blocks
                    needed = self._estimate_blocks_needed(
                        len(req.input_ids), req.params.max_tokens,
                        batch_ctx.get_block_size()
                    )
                    if not self._try_preempt(batch_ctx, running, free_slots, needed):
                        # Cannot admit even after preemption — put back
                        self._requeue(req)
                        break
                    preempted = True

                    # Re-check after preemption
                    if not self._can_admit(batch_ctx, len(req.input_ids), req.params.max_tokens,
                                           self._reserved_growth(batch_ctx, running)):
                        self._requeue(req)
                        break

                slot_id = free_slots.pop(0)
                if req.started_at == 0:
                    req.started_at = time.time()
                req.status = RequestStatus.PREFILLING

                try:
                    # Check if this request was preempted and has a snapshot
                    if req.kv_cache_snapshot is not None:
                        batch_ctx.slot_restore(slot_id, req.kv_cache_snapshot)
                        self._release_snapshot(req)
                        req.kv_cache_snapshot = None
                        # Already prefilled, just resume decoding
                        req.status = RequestStatus.DECODING
                        running[slot_id] = req
                        logger.debug(
                            f"Request {req.request_id}: resumed from preemption snapshot"
                        )
                        if preempted:
                            break
                        continue

                    # Block-granular prefix matching. The runtime attaches
                    # retained physical blocks directly to this slot.
                    match_len = 0
                    prefix_lookup = getattr(batch_ctx, "prefix_lookup", None)
                    if self.enable_block_prefix_cache and prefix_lookup:
                        match_len = int(prefix_lookup(slot_id, req.input_ids))

                    if match_len == 0:
                        batch_ctx.slot_reset(slot_id)
                    first_token = self._prefill_prompt(
                        batch_ctx, slot_id, req, start_pos=match_len
                    )
                    prefix_publish = getattr(batch_ctx, "prefix_publish", None)
                    if self.enable_block_prefix_cache and prefix_publish:
                        prefix_publish(slot_id, req.input_ids)

                    first_token = int(first_token)
                    if req.cancelled.is_set():
                        raise _RequestCancelled()

                except Exception as e:
                    if not isinstance(e, _RequestCancelled):
                        logger.error(f"Prefill error for {req.request_id}: {e}", exc_info=True)
                    else:
                        req.cancelled.set()
                    batch_ctx.slot_reset(slot_id)
                    free_slots.append(slot_id)
                    self._finish_request(req, error=None if isinstance(e, _RequestCancelled) else e)
                    self.queue.mark_done(req.request_id)
                    continue

                req.generated_tokens.append(first_token)
                req.last_token = first_token
                req.status = RequestStatus.DECODING

                if req.stream and req.output_queue:
                    self._send_token_async(req, first_token)

                if (first_token == self._eos_token_id or
                        len(req.generated_tokens) >= req.params.max_tokens):
                    free_slots.append(slot_id)
                    batch_ctx.slot_reset(slot_id)
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)
                else:
                    running[slot_id] = req
                # Allow execution to make progress before considering the victim
                # again; otherwise admission can endlessly swap two requests.
                if preempted:
                    break

            # ── 2. Decode with per-request sampling parameters ──
            if not self._running:
                break
            if running:
                active_slots = list(running.keys())
                current_tokens = [running[s].last_token for s in active_slots]
                temperatures = [running[s].params.temperature for s in active_slots]
                top_ks = [running[s].params.top_k for s in active_slots]
                top_ps = [running[s].params.top_p for s in active_slots]

                try:
                    next_tokens = batch_ctx.decode_per_request(
                        active_slots=active_slots,
                        current_tokens=current_tokens,
                        temperatures=temperatures,
                        top_ks=top_ks,
                        top_ps=top_ps,
                    )
                    if len(next_tokens) != len(active_slots):
                        raise RuntimeError("batch runtime returned an incorrect number of tokens")
                except Exception as e:
                    logger.error(f"Batch decode error: {e}", exc_info=True)
                    for sid in list(running.keys()):
                        req = running.pop(sid)
                        free_slots.append(sid)
                        batch_ctx.slot_reset(sid)
                        self._finish_request(req, error=e)
                        self.queue.mark_done(req.request_id)
                    continue

                # ── 3. 处理结果, 移除已完成的请求 ──
                finished_slots: List[int] = []
                for slot_id, next_tok in zip(active_slots, next_tokens):
                    req = running[slot_id]
                    if req.cancelled.is_set():
                        finished_slots.append(slot_id)
                        continue
                    next_tok = int(next_tok)
                    req.generated_tokens.append(next_tok)
                    req.last_token = next_tok

                    if req.stream and req.output_queue:
                        self._send_token_async(req, next_tok)

                    if (next_tok == self._eos_token_id or
                            len(req.generated_tokens) >= req.params.max_tokens or
                            req.status == RequestStatus.CANCELLED):
                        finished_slots.append(slot_id)

                for slot_id in finished_slots:
                    req = running.pop(slot_id)

                    batch_ctx.slot_reset(slot_id)
                    free_slots.append(slot_id)
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)

            # ── 4. 空闲等待 ──
            if not running and self.queue.waiting_count == 0:
                self.queue.wait_for_requests(timeout=0.05)

        logger.info("Worker loop exited")

    def _send_token_async(self, req: InferenceRequest, token_id: int):
        """线程安全地向 asyncio 队列发送 token."""
        if req.loop and req.output_queue:
            req.loop.call_soon_threadsafe(req.output_queue.put_nowait, token_id)

    def _finish_request(self, req: InferenceRequest, error: Exception = None):
        """标记请求完成, 设置 future 结果."""
        if req.completion_scheduled:
            return
        req.completion_scheduled = True
        req.finished_at = time.time()
        try:
            self._release_snapshot(req)
        except Exception as cleanup_error:
            error = error or cleanup_error
            logger.exception("Failed to release request snapshot")
        if req.cancelled.is_set():
            req.status = RequestStatus.CANCELLED
        elif error:
            req.status = RequestStatus.ERROR
            req.error_message = str(error)
        else:
            req.status = RequestStatus.DONE

        # Check done() on the event-loop thread, not before scheduling it: a
        # client may cancel the future between those two operations.
        def deliver():
            if req.future is not None and not req.future.done():
                if req.status == RequestStatus.CANCELLED:
                    req.future.cancel()
                elif error is not None:
                    req.future.set_exception(error)
                else:
                    req.future.set_result(list(req.generated_tokens))
            if req.stream and req.output_queue is not None:
                req.output_queue.put_nowait(None)
        if req.loop is not None and not req.loop.is_closed():
            req.loop.call_soon_threadsafe(deliver)

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

    def _release_snapshot(self, req):
        snapshot, req.kv_cache_snapshot = req.kv_cache_snapshot, None
        if snapshot is None:
            return
        close = getattr(snapshot, "close", None)
        if close is not None:
            close()
        else:
            destroy = getattr(getattr(self, "model", None), "destroy_snapshot", None)
            if destroy is not None:
                destroy(snapshot)

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
            "total_preemptions": self._total_preemptions,
        }

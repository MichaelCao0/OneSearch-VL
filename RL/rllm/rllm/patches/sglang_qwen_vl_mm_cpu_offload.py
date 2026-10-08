"""Run SGLang Qwen-VL multimodal preprocessing outside its asyncio loop.

SGLang 0.5.3 labels ``QwenVLImageProcessor.process_mm_data_async`` async, but
the implementation performs PIL resize, video decode/resize and the complete
HF processor call synchronously.  Under a rollout burst those calls serialize
on TokenizerManager's event loop and later requests hit the 10-second MM
deadline even when all media is local.

This opt-in patch executes the complete coroutine in a bounded worker pool.
Each worker owns an asyncio loop for the few awaits inside the upstream
coroutine, while all blocking CPU work stays off TokenizerManager's loop.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any


_PATCH_MARKER = "_rllm_cpu_offload_original"
_STATS_LOCK = threading.Lock()
_VIDEO_GATE_INIT_LOCK = threading.Lock()
_STATS = {
    "calls": 0,
    "completed": 0,
    "active": 0,
    "max_active": 0,
    "queue_total_s": 0.0,
    "queue_max_s": 0.0,
    "service_total_s": 0.0,
    "service_max_s": 0.0,
    "cancelled": 0,
    "video_calls": 0,
    "video_completed": 0,
    "video_active": 0,
    "video_max_active": 0,
    "video_waiters": 0,
    "video_max_waiters": 0,
    "video_gate_wait_total_s": 0.0,
    "video_gate_wait_max_s": 0.0,
}


def stats_snapshot() -> dict[str, int | float]:
    """Return one process-local cumulative snapshot for periodic reporting."""
    with _STATS_LOCK:
        return dict(_STATS)


def _thread_count() -> int:
    raw = os.environ.get("SGLANG_MM_CPU_THREADS", "32")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"SGLANG_MM_CPU_THREADS must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"SGLANG_MM_CPU_THREADS must be positive, got {value}")
    return value


def _slow_log_threshold_s() -> float:
    raw = os.environ.get("SGLANG_RLLM_MM_CPU_SLOW_LOG_S", "5")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"SGLANG_RLLM_MM_CPU_SLOW_LOG_S must be numeric, got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"SGLANG_RLLM_MM_CPU_SLOW_LOG_S must be non-negative, got {value}"
        )
    return value


def _video_concurrency() -> int:
    """Maximum simultaneous video MM preprocess calls per SGLang replica.

    A value of zero preserves the upstream/global MM admission behavior. Video
    tensors are much larger than image tensors, so a separate positive limit
    prevents several requests from moving full videos to one GPU at once while
    leaving image-only calls governed by SGLang's existing concurrency limit.
    """
    raw = os.environ.get("SGLANG_MM_VIDEO_CONCURRENCY", "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_MM_VIDEO_CONCURRENCY must be an integer, "
            f"got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            "SGLANG_MM_VIDEO_CONCURRENCY must be non-negative, "
            f"got {value}"
        )
    return value


def _request_obj(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    request_obj = kwargs.get("request_obj")
    if request_obj is not None:
        return request_obj
    # Qwen-VL currently receives request_obj as the third positional argument.
    return next(
        (
            value
            for value in reversed(args)
            if hasattr(value, "rid")
            or hasattr(value, "video_data")
        ),
        None,
    )


def _has_payload(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_has_payload(item) for item in value)
    return value is not None


def _is_video_request(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return _has_payload(getattr(_request_obj(args, kwargs), "video_data", None))


def _get_video_gate(processor: Any, concurrency: int):
    if concurrency <= 0:
        return None
    gate = getattr(processor, "_rllm_mm_video_gate", None)
    if gate is not None:
        return gate
    with _VIDEO_GATE_INIT_LOCK:
        gate = getattr(processor, "_rllm_mm_video_gate", None)
        if gate is None:
            gate = threading.BoundedSemaphore(concurrency)
            processor._rllm_mm_video_gate = gate
    return gate


def _request_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    request_obj = _request_obj(args, kwargs)
    return str(getattr(request_obj, "rid", "anonymous_rid"))


def apply(processor_cls: type | None = None) -> bool:
    """Patch QwenVLImageProcessor; return whether a new patch was installed."""
    if os.environ.get("SGLANG_MM_CPU_OFFLOAD", "0") != "1":
        return False

    if processor_cls is None:
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        processor_cls = QwenVLImageProcessor

    original = processor_cls.process_mm_data_async
    if hasattr(original, _PATCH_MARKER):
        return False

    threads = _thread_count()
    slow_log_s = _slow_log_threshold_s()
    video_concurrency = _video_concurrency()

    async def process_mm_data_async(self: Any, *args: Any, **kwargs: Any):
        executor = getattr(self, "_rllm_mm_cpu_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=threads,
                thread_name_prefix="sglang-mm-cpu",
            )
            self._rllm_mm_cpu_executor = executor

        submitted_at = time.perf_counter()
        rid = _request_id(args, kwargs)
        is_video = _is_video_request(args, kwargs)

        # Construct and consume the upstream coroutine in the worker thread so
        # none of its nominally-async but blocking sections run on this loop.
        # The counters live inside the worker rather than around
        # run_in_executor: they therefore measure running work, not queued
        # futures.
        def run_original():
            video_gate = _get_video_gate(self, video_concurrency) if is_video else None
            video_gate_wait_s = 0.0
            video_admitted = False
            if is_video:
                with _STATS_LOCK:
                    _STATS["video_calls"] += 1
                    _STATS["video_waiters"] += 1
                    _STATS["video_max_waiters"] = max(
                        _STATS["video_max_waiters"], _STATS["video_waiters"]
                    )
                gate_wait_started = time.perf_counter()
                if video_gate is not None:
                    video_gate.acquire()
                video_gate_wait_s = time.perf_counter() - gate_wait_started
                with _STATS_LOCK:
                    _STATS["video_waiters"] -= 1
                    _STATS["video_active"] += 1
                    _STATS["video_max_active"] = max(
                        _STATS["video_max_active"], _STATS["video_active"]
                    )
                video_admitted = True
            started_at = time.perf_counter()
            queue_s = started_at - submitted_at
            with _STATS_LOCK:
                _STATS["calls"] += 1
                call_no = _STATS["calls"]
                _STATS["active"] += 1
                _STATS["max_active"] = max(
                    _STATS["max_active"], _STATS["active"]
                )
                active_at_start = _STATS["active"]
                max_active = _STATS["max_active"]
            try:
                return asyncio.run(original(self, *args, **kwargs))
            finally:
                finished_at = time.perf_counter()
                service_s = finished_at - started_at
                with _STATS_LOCK:
                    _STATS["active"] -= 1
                    _STATS["completed"] += 1
                    _STATS["queue_total_s"] += queue_s
                    _STATS["queue_max_s"] = max(
                        _STATS["queue_max_s"], queue_s
                    )
                    _STATS["service_total_s"] += service_s
                    _STATS["service_max_s"] = max(
                        _STATS["service_max_s"], service_s
                    )
                    active_after = _STATS["active"]
                    if video_admitted:
                        _STATS["video_active"] -= 1
                        _STATS["video_completed"] += 1
                        _STATS["video_gate_wait_total_s"] += video_gate_wait_s
                        _STATS["video_gate_wait_max_s"] = max(
                            _STATS["video_gate_wait_max_s"], video_gate_wait_s
                        )
                if video_gate is not None and video_admitted:
                    video_gate.release()
                if service_s >= slow_log_s:
                    print(
                        "[mm-cpu-offload-stats] "
                        f"call={call_no} rid={rid} "
                        f"executor_queue_s={started_at - submitted_at:.6f} "
                        f"service_s={service_s:.6f} "
                        f"active_at_start={active_at_start} "
                        f"active_after={active_after} "
                        f"max_active={max_active}",
                        flush=True,
                    )

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(executor, run_original)
        try:
            # wait_for() cancels this coroutine at SGLang's MM deadline, but a
            # running ThreadPoolExecutor future cannot be cancelled. Shield it
            # so the cancellation path below can retain the upstream admission
            # slot until the real CPU work has drained. Otherwise the semaphore
            # is released while the timed-out thread remains active and later
            # calls can silently exceed mm_max_concurrent_calls.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled_at = time.perf_counter()
            with _STATS_LOCK:
                _STATS["cancelled"] += 1
            try:
                await future
            finally:
                print(
                    "[mm-cpu-offload-timeout-drain] "
                    f"rid={rid} deadline_elapsed_s="
                    f"{cancelled_at - submitted_at:.6f} "
                    f"drain_s={time.perf_counter() - cancelled_at:.6f} "
                    f"total_s={time.perf_counter() - submitted_at:.6f}",
                    flush=True,
                )
            raise

    setattr(process_mm_data_async, _PATCH_MARKER, original)
    processor_cls.process_mm_data_async = process_mm_data_async
    print(
        f"[mm-cpu-offload] patched {processor_cls.__module__}."
        f"{processor_cls.__name__}.process_mm_data_async threads={threads} "
        f"slow_log_s={slow_log_s} "
        f"video_concurrency={video_concurrency or 'upstream'}",
        flush=True,
    )
    return True


def is_applied(processor_cls: type | None = None) -> bool:
    """Return whether the active processor method carries this patch."""
    if processor_cls is None:
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        processor_cls = QwenVLImageProcessor
    return hasattr(processor_cls.process_mm_data_async, _PATCH_MARKER)

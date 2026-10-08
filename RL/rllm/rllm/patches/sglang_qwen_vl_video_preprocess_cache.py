"""Cache SGLang Qwen-VL video decode/resize results inside each server actor.

SGLang's Qwen processor opens and preprocesses every ``video_data`` item before
the request reaches the GPU scheduler.  Multi-turn RL sends the same video path
on every turn, so the default path repeatedly performs Decord ``get_batch`` and
torchvision resize for identical media.  This opt-in patch keeps the resulting
CPU tensor and metadata in a bounded, process-local LRU.

The cache intentionally stops before the Hugging Face processor.  Prompt text,
expanded token ids and mRoPE positions still get recomputed for every turn, so
the patch cannot reuse prompt-dependent state or change video-token semantics.
Concurrent misses for one source are coalesced through a thread-safe Future.
"""

from __future__ import annotations

import asyncio
import copy
import contextvars
import hashlib
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from typing import Any


_METHOD_PATCH_MARKER = "_rllm_video_preprocess_cache_original"
_PREPROCESS_PATCH_MARKER = "_rllm_video_preprocess_cache_original"
_REQUEST_STATE: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("rllm_sglang_video_cache_request", default=None)
)
_LOCK = threading.RLock()
_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, int]] = OrderedDict()
_PENDING: dict[tuple[Any, ...], Future] = {}
_CACHE_BYTES = 0
_CONFIG: dict[str, Any] = {}
_STATS = {
    "calls": 0,
    "hits": 0,
    "misses": 0,
    "waits": 0,
    "evictions": 0,
    "oversize": 0,
    "bytes_saved": 0,
}


def _positive_int_env(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _request_obj(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = kwargs.get("request_obj")
    if value is not None:
        return value
    return next(
        (candidate for candidate in reversed(args) if hasattr(candidate, "video_data")),
        None,
    )


def _source_key(source: Any, signature: tuple[Any, ...]) -> tuple[Any, ...] | None:
    if isinstance(source, str):
        path = Path(source)
        try:
            stat = path.stat()
        except OSError:
            # URLs and base64 strings are still stable within one rollout.  Do
            # not print them: presigned URLs may contain credentials.
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            return ("str", digest, signature)
        return (
            "path",
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            signature,
        )
    if isinstance(source, bytes):
        return ("bytes", hashlib.sha256(source).hexdigest(), len(source), signature)
    # SGLang dictionaries represent already-precomputed multimodal data.  They
    # must bypass this raw-video cache.
    return None


def _value_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return int(value.numel()) * int(value.element_size())
        except (TypeError, ValueError, RuntimeError):
            return 0
    if isinstance(value, dict):
        return sum(_value_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_bytes(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 0


def _copy_result(value: Any) -> Any:
    # Metadata may be augmented downstream.  The large CPU tensor is treated
    # as immutable and deliberately shared; cloning it would erase the speedup.
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[1], dict)
    ):
        return value[0], copy.deepcopy(value[1])
    return value


def _snapshot_locked() -> dict[str, int]:
    return {
        **_STATS,
        "entries": len(_CACHE),
        "bytes": _CACHE_BYTES,
        "pending": len(_PENDING),
    }


def snapshot() -> dict[str, int]:
    """Return process-local counters for diagnostics and unit tests."""
    with _LOCK:
        return _snapshot_locked()


def _log(event: str, elapsed_s: float) -> None:
    with _LOCK:
        stats = _snapshot_locked()
        should_log = (
            stats["calls"] <= 4
            or stats["calls"] % _CONFIG["log_every"] == 0
            or (event == "miss" and elapsed_s >= _CONFIG["slow_log_s"])
        )
    if not should_log:
        return
    print(
        "[mm-video-preprocess-cache] "
        f"event={event} elapsed_s={elapsed_s:.6f} "
        f"calls={stats['calls']} hits={stats['hits']} "
        f"misses={stats['misses']} waits={stats['waits']} "
        f"entries={stats['entries']} bytes={stats['bytes']} "
        f"pending={stats['pending']} evictions={stats['evictions']} "
        f"oversize={stats['oversize']} bytes_saved={stats['bytes_saved']}",
        flush=True,
    )


def _lookup_or_claim(key: tuple[Any, ...]):
    with _LOCK:
        _STATS["calls"] += 1
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            _STATS["hits"] += 1
            _STATS["bytes_saved"] += cached[1]
            return "hit", cached[0]
        pending = _PENDING.get(key)
        if pending is not None:
            _STATS["waits"] += 1
            return "wait", pending
        pending = Future()
        _PENDING[key] = pending
        _STATS["misses"] += 1
        return "miss", pending


def _publish(key: tuple[Any, ...], pending: Future, value: Any, size: int) -> None:
    global _CACHE_BYTES
    with _LOCK:
        if size <= _CONFIG["max_bytes"]:
            old = _CACHE.pop(key, None)
            if old is not None:
                _CACHE_BYTES -= old[1]
            while _CACHE and (
                _CACHE_BYTES + size > _CONFIG["max_bytes"]
                or len(_CACHE) >= _CONFIG["max_entries"]
            ):
                _, (_, evicted_size) = _CACHE.popitem(last=False)
                _CACHE_BYTES -= evicted_size
                _STATS["evictions"] += 1
            _CACHE[key] = (value, size)
            _CACHE_BYTES += size
        else:
            _STATS["oversize"] += 1
        _PENDING.pop(key, None)
        pending.set_result(value)


def _publish_error(key: tuple[Any, ...], pending: Future, exc: BaseException) -> None:
    with _LOCK:
        _PENDING.pop(key, None)
        pending.set_exception(exc)


def _processor_signature(qwen_module: Any) -> tuple[Any, ...]:
    names = (
        "FRAME_FACTOR",
        "IMAGE_FACTOR",
        "VIDEO_MIN_PIXELS",
        "VIDEO_MAX_PIXELS",
        "VIDEO_TOTAL_PIXELS",
        "FPS",
        "FPS_MIN_FRAMES",
        "FPS_MAX_FRAMES",
    )
    return tuple((name, getattr(qwen_module, name, None)) for name in names)


def apply(processor_cls: type | None = None, qwen_module: Any | None = None) -> bool:
    """Install the opt-in request context and preprocessing LRU wrappers."""
    max_bytes = _positive_int_env(
        "SGLANG_RLLM_VIDEO_PREPROCESS_CACHE_BYTES", 0, allow_zero=True
    )
    if max_bytes == 0:
        return False

    if qwen_module is None:
        from sglang.srt.multimodal.processors import qwen_vl as qwen_module
    if processor_cls is None:
        processor_cls = qwen_module.QwenVLImageProcessor

    original_method = processor_cls.process_mm_data_async
    original_preprocess = qwen_module.preprocess_video
    if hasattr(original_method, _METHOD_PATCH_MARKER) or hasattr(
        original_preprocess, _PREPROCESS_PATCH_MARKER
    ):
        return False

    _CONFIG.update(
        max_bytes=max_bytes,
        max_entries=_positive_int_env(
            "SGLANG_RLLM_VIDEO_PREPROCESS_CACHE_ENTRIES", 64
        ),
        log_every=_positive_int_env("SGLANG_RLLM_VIDEO_PREPROCESS_LOG_EVERY", 50),
        slow_log_s=float(
            os.environ.get("SGLANG_RLLM_VIDEO_PREPROCESS_SLOW_LOG_S", "2")
        ),
        signature=_processor_signature(qwen_module),
    )
    if _CONFIG["slow_log_s"] < 0:
        raise ValueError("SGLANG_RLLM_VIDEO_PREPROCESS_SLOW_LOG_S must be non-negative")

    async def cached_preprocess_video(video: Any, *args: Any, **kwargs: Any):
        state = _REQUEST_STATE.get()
        if state is None or state["index"] >= len(state["keys"]):
            return await original_preprocess(video, *args, **kwargs)
        key = state["keys"][state["index"]]
        state["index"] += 1
        if key is None:
            return await original_preprocess(video, *args, **kwargs)

        started_at = time.perf_counter()
        disposition, payload = _lookup_or_claim(key)
        if disposition == "hit":
            elapsed_s = time.perf_counter() - started_at
            _log("hit", elapsed_s)
            return _copy_result(payload)
        if disposition == "wait":
            value = await asyncio.shield(asyncio.wrap_future(payload))
            elapsed_s = time.perf_counter() - started_at
            _log("wait", elapsed_s)
            return _copy_result(value)

        pending = payload
        try:
            value = await original_preprocess(video, *args, **kwargs)
        except BaseException as exc:
            _publish_error(key, pending, exc)
            raise
        elapsed_s = time.perf_counter() - started_at
        _publish(key, pending, value, _value_bytes(value))
        _log("miss", elapsed_s)
        return _copy_result(value)

    async def process_mm_data_async(self: Any, *args: Any, **kwargs: Any):
        request = _request_obj(args, kwargs)
        raw_videos = getattr(request, "video_data", None)
        if raw_videos is None:
            keys: list[tuple[Any, ...] | None] = []
        else:
            if not isinstance(raw_videos, (list, tuple)):
                raw_videos = [raw_videos]
            keys = [
                _source_key(source, _CONFIG["signature"]) for source in raw_videos
            ]
        token = _REQUEST_STATE.set({"keys": keys, "index": 0})
        try:
            return await original_method(self, *args, **kwargs)
        finally:
            _REQUEST_STATE.reset(token)

    setattr(cached_preprocess_video, _PREPROCESS_PATCH_MARKER, original_preprocess)
    setattr(process_mm_data_async, _METHOD_PATCH_MARKER, original_method)
    qwen_module.preprocess_video = cached_preprocess_video
    processor_cls.process_mm_data_async = process_mm_data_async
    print(
        "[mm-video-preprocess-cache] enabled "
        f"max_bytes={_CONFIG['max_bytes']} "
        f"max_entries={_CONFIG['max_entries']} "
        f"log_every={_CONFIG['log_every']} "
        f"slow_log_s={_CONFIG['slow_log_s']}",
        flush=True,
    )
    return True


def is_applied(processor_cls: type | None = None) -> bool:
    if processor_cls is None:
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        processor_cls = QwenVLImageProcessor
    return hasattr(processor_cls.process_mm_data_async, _METHOD_PATCH_MARKER)


def _reset_for_tests() -> None:
    """Clear process-local state; tests restore patched callables separately."""
    global _CACHE_BYTES
    with _LOCK:
        _CACHE.clear()
        _PENDING.clear()
        _CACHE_BYTES = 0
        _CONFIG.clear()
        for key in _STATS:
            _STATS[key] = 0

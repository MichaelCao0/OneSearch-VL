"""Add cumulative observability to SGLang's GPU VLM embedding cache.

SGLang 0.5.3 keeps vision-encoder outputs in a bounded ``MultiModalCache``
but exposes neither hits nor evictions.  This opt-in wrapper preserves the
cache implementation and return values while emitting cumulative counters.
It is installed before SGLang forks scheduler processes, so every TP1 replica
owns independent counters and reports its real cold-cache behavior.
"""

from __future__ import annotations

import os
import threading
from typing import Any


_PATCH_MARKER = "_rllm_vlm_cache_stats_original"
_STATS_LOCK = threading.Lock()
_STATS = {
    "gets": 0,
    "hits": 0,
    "misses": 0,
    "puts": 0,
    "put_failures": 0,
    "evictions": 0,
    "max_entry_bytes": 0,
}


def _log_every() -> int:
    raw = os.environ.get("SGLANG_RLLM_VLM_CACHE_STATS_LOG_EVERY", "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_RLLM_VLM_CACHE_STATS_LOG_EVERY must be an integer, "
            f"got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            "SGLANG_RLLM_VLM_CACHE_STATS_LOG_EVERY must be non-negative, "
            f"got {value}"
        )
    return value


def _emit(cache: Any, event: str, last_entry_bytes: int, log_every: int) -> None:
    with _STATS_LOCK:
        snapshot = dict(_STATS)
    ops = int(snapshot["gets"]) + int(snapshot["puts"])
    if log_every <= 0 or ops == 0 or ops % log_every:
        return
    print(
        "[vlm-embedding-cache] "
        f"replica_tag={os.environ.get('SGLANG_RLLM_REPLICA_TAG', 'unknown')} "
        f"event={event} ops={ops} gets={snapshot['gets']} "
        f"hits={snapshot['hits']} misses={snapshot['misses']} "
        f"puts={snapshot['puts']} put_failures={snapshot['put_failures']} "
        f"evictions={snapshot['evictions']} entries={len(cache)} "
        f"bytes={int(cache.current_size)} "
        f"capacity_bytes={int(cache.max_size)} "
        f"last_entry_bytes={last_entry_bytes} "
        f"max_entry_bytes={snapshot['max_entry_bytes']}",
        flush=True,
    )


def apply(cache_cls: type | None = None) -> bool:
    """Patch ``MultiModalCache``; return whether a new patch was installed."""
    log_every = _log_every()
    if log_every <= 0:
        return False

    if cache_cls is None:
        from sglang.srt.mem_cache.multimodal_cache import MultiModalCache

        cache_cls = MultiModalCache

    original_get = cache_cls.get
    if hasattr(original_get, _PATCH_MARKER):
        return False
    original_put = cache_cls.put

    def get(self: Any, mm_hash: int):
        value = original_get(self, mm_hash)
        with _STATS_LOCK:
            _STATS["gets"] += 1
            _STATS["hits" if value is not None else "misses"] += 1
        _emit(self, "get", 0, log_every)
        return value

    def put(self: Any, mm_hash: int, embedding: Any) -> bool:
        before_entries = len(self)
        already_present = mm_hash in self.mm_cache
        entry_bytes = int(self._get_tensor_size(embedding))
        inserted = original_put(self, mm_hash, embedding)
        with _STATS_LOCK:
            _STATS["puts"] += 1
            _STATS["put_failures"] += int(not inserted)
            if inserted and not already_present:
                _STATS["evictions"] += max(0, before_entries + 1 - len(self))
            _STATS["max_entry_bytes"] = max(
                _STATS["max_entry_bytes"], entry_bytes
            )
        _emit(self, "put", entry_bytes, log_every)
        return inserted

    setattr(get, _PATCH_MARKER, original_get)
    setattr(put, _PATCH_MARKER, original_put)
    cache_cls.get = get
    cache_cls.put = put
    print(
        "[vlm-embedding-cache] patched "
        f"{cache_cls.__module__}.{cache_cls.__name__} log_every={log_every}",
        flush=True,
    )
    return True


def is_applied(cache_cls: type | None = None) -> bool:
    if cache_cls is None:
        from sglang.srt.mem_cache.multimodal_cache import MultiModalCache

        cache_cls = MultiModalCache
    return hasattr(cache_cls.get, _PATCH_MARKER)


def run_scheduler_process_with_vlm_cache_stats(*args: Any, **kwargs: Any) -> Any:
    """Install counters inside SGLang's spawned scheduler, then run it.

    SGLang forces the ``spawn`` multiprocessing start method.  Patches applied
    in the HTTP/Ray actor therefore do not survive in the scheduler process
    that owns ``embedding_cache`` and calls ``MultiModalCache.get/put``.  This
    top-level entrypoint is deliberately importable by multiprocessing: the
    actor swaps SGLang's process target to this function before launch.
    """
    applied_now = apply()
    print(
        "[vlm-embedding-cache] scheduler verification "
        f"applied_now={applied_now} verified={is_applied()} "
        f"log_every={_log_every()} "
        f"replica_tag={os.environ.get('SGLANG_RLLM_REPLICA_TAG', 'unknown')}",
        flush=True,
    )
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)

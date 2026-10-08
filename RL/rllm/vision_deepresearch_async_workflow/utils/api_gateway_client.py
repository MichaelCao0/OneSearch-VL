"""
LLM Judge Client — thin OpenAI-compatible Chat Completions wrapper.

Used by the query-utility reward in `deepresearch_workflow._judge_query_utility`
to call an external judge model (e.g. GPT-4o) and score search-trajectory
quality on a 0.0–1.0 scale.

The client is optional: if no API key is configured, the reward falls back
to 0.0 and training proceeds normally.

Environment variables
---------------------
JUDGE_API_BASE_URL   OpenAI-compatible endpoint (default: ``https://api.openai.com/v1``).
JUDGE_API_KEY        Bearer token for the endpoint.
JUDGE_MODEL          Default model name when the caller does not pass one.

Usage::

    from vision_deepresearch_async_workflow.utils.api_gateway_client import (
        api_gateway_chat,
        is_api_gateway_configured,
    )

    if is_api_gateway_configured():
        result = api_gateway_chat(messages, model_marker="gpt-4o")
        text = result["choices"][0]["message"]["content"]
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import random
import threading
import time
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass


_BASE_URL = (os.getenv("JUDGE_BASE_URL") or os.getenv("JUDGE_API_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
_API_KEY = os.getenv("JUDGE_API_KEY", "")
_DEFAULT_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
_JUDGE_MAX_CONCURRENCY = max(1, int(os.getenv("JUDGE_MAX_CONCURRENCY", "64")))
_JUDGE_QUEUE_TIMEOUT = float(os.getenv("JUDGE_QUEUE_TIMEOUT", "300"))
_JUDGE_RETRY_ATTEMPTS = max(1, int(os.getenv("JUDGE_RETRY_ATTEMPTS", "3")))
_JUDGE_RETRY_DELAY = max(0.0, float(os.getenv("JUDGE_RETRY_DELAY", "1")))
_JUDGE_SEMAPHORE = threading.BoundedSemaphore(_JUDGE_MAX_CONCURRENCY)
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@contextmanager
def judge_request_slot():
    """Process-wide bulkhead shared by accuracy and query-utility judges."""

    acquired = _JUDGE_SEMAPHORE.acquire(timeout=_JUDGE_QUEUE_TIMEOUT)
    if not acquired:
        raise TimeoutError(
            "judge request queue timed out after "
            f"{_JUDGE_QUEUE_TIMEOUT}s at concurrency {_JUDGE_MAX_CONCURRENCY}"
        )
    try:
        yield
    finally:
        _JUDGE_SEMAPHORE.release()


def judge_status_is_transient(status_code: int) -> bool:
    return int(status_code) in _TRANSIENT_STATUS


def judge_retry_wait_s(headers: Any, attempt: int) -> float:
    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        explicit = float(retry_after) if retry_after is not None else 0.0
    except (TypeError, ValueError):
        explicit = 0.0
    exponential = _JUDGE_RETRY_DELAY * float(2 ** max(attempt - 1, 0))
    return max(explicit, exponential) + random.uniform(0.0, 0.25)


def is_api_gateway_configured() -> bool:
    """Return True when a judge API key is available."""
    return bool(_API_KEY)


def api_gateway_chat(
    messages: list[dict],
    model_marker: str | None = None,
    max_tokens: int | None = None,
    timeout: int = 300,
    temperature: float | None = None,
) -> dict:
    """Call the OpenAI-compatible Chat Completions endpoint.

    Parameters
    ----------
    messages
        OpenAI-format list of ``{"role": ..., "content": ...}`` dicts.
    model_marker
        Model name. Falls back to ``$JUDGE_MODEL`` / ``"gpt-4o-mini"``.
    max_tokens, temperature
        Standard sampling params; passed through if provided.
    timeout
        Per-request timeout in seconds.

    Returns
    -------
    dict
        The raw OpenAI response JSON. The caller reads
        ``result["choices"][0]["message"]["content"]``.
    """

    if not is_api_gateway_configured():
        raise RuntimeError(
            "Judge API not configured. Set JUDGE_API_KEY "
            "(and optionally JUDGE_API_BASE_URL / JUDGE_MODEL)."
        )

    model = model_marker or _DEFAULT_MODEL

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, _JUDGE_RETRY_ATTEMPTS + 1):
        try:
            with judge_request_slot():
                resp = requests.post(
                    f"{_BASE_URL}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
            if resp.status_code == 200:
                return resp.json()
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_error = exc
            status = int(getattr(exc.response, "status_code", 0))
            if not judge_status_is_transient(status):
                raise
            response_headers = getattr(exc.response, "headers", {})
        except Exception as exc:
            last_error = exc
            response_headers = {}
        if attempt < _JUDGE_RETRY_ATTEMPTS:
            time.sleep(judge_retry_wait_s(response_headers, attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("judge request exhausted without response or exception")

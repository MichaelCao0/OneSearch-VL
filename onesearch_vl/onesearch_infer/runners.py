"""Inference runners for the supported model families.

Runners share a tiny interface:

* :py:meth:`BaseRunner.load`  — perform any one-time setup (model load).
* :py:meth:`BaseRunner.infer` — accept Gemini-style ``contents`` plus a
  system prompt and return a Gemini-shaped response dict.

The Gemini-shaped response keeps the calling code in :mod:`pipeline`
agnostic to the underlying provider. The minimal contract is::

    {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "..."}]},
                "finishReason": "STOP" | "ERROR",
            }
        ],
        "usageMetadata": {...},
        "modelVersion": "...",
    }
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

from . import config, messages
from .config import ModelSpec


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Seed model API key rotation
# ---------------------------------------------------------------------------
_SEED_KEY_LOCK = threading.Lock()
_SEED_KEY_CYCLE: Optional["itertools.cycle"] = None
_SEED_KEY_LIST: List[str] = []


def _init_seed_keys() -> List[str]:
    raw = os.environ.get("SEED_API_KEYS", "").strip()
    if not raw:
        return []
    flat: List[str] = []
    for token in re.split(r"[,\s]+", raw):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            key, w = token.rsplit(":", 1)
            try:
                weight = max(1, int(w))
            except ValueError:
                key, weight = token, 1
        else:
            key, weight = token, 1
        flat.extend([key] * weight)
    return flat


def _pick_seed_key(fallback: str) -> str:
    """Round-robin one key from SEED_API_KEYS; thread-safe. Returns ``fallback``
    (the single SEED_API_KEY) when the pool is unset."""
    global _SEED_KEY_CYCLE, _SEED_KEY_LIST
    with _SEED_KEY_LOCK:
        if _SEED_KEY_CYCLE is None:
            _SEED_KEY_LIST = _init_seed_keys()
            if _SEED_KEY_LIST:
                _SEED_KEY_CYCLE = itertools.cycle(_SEED_KEY_LIST)
                logger.warning(
                    "Seed API key rotation enabled (%d slots): %s",
                    len(_SEED_KEY_LIST),
                    [f"***{k[-4:]}" for k in dict.fromkeys(_SEED_KEY_LIST)],
                )
            else:
                _SEED_KEY_CYCLE = itertools.cycle([""])  # sentinel
        if _SEED_KEY_LIST:
            return next(_SEED_KEY_CYCLE)
    return fallback


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Tunables shared across runners."""

    temperature: float = 0.0
    max_tokens: int = 32768
    enable_thinking: bool = False
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseRunner:
    spec: ModelSpec

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    def load(self) -> None:  # pragma: no cover - default no-op
        """Hook for one-time model preparation."""

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


def _empty_response(model_version: str, error: str) -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": f"Error: {error}"}]},
                "finishReason": "ERROR",
            }
        ],
        "usageMetadata": {},
        "modelVersion": model_version,
        "raw_response": {"error": error},
    }


def _wrap_gemini(
    text: str,
    finish: str,
    usage_meta: Optional[Dict[str, int]],
    model_version: str,
    raw_response: Any = None,
) -> Dict[str, Any]:
    """Wrap a plain completion into the Gemini-shaped response the pipeline
    expects. Shared by the OpenAI-compatible vLLM runners (HTTP + in-process).
    """
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": finish,
            }
        ],
        "usageMetadata": usage_meta or {},
        "modelVersion": model_version,
        "raw_response": raw_response,
    }


def _seed_error_non_retryable(body: str) -> bool:
    """Return True when repeating the same Seed request cannot succeed."""

    low = (body or "").lower()
    if "dimensions are too small" in low:
        return True
    if "minimum allowed dimension" in low:
        return True
    if "error while connecting" in low:
        return True
    if "failed to fetch" in low and "url" in low:
        return True
    if "only base64, http or https urls are supported" in low:
        return True
    if "invalidparameter.unsupportedinput" in low:
        return True
    return False


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudeAnthropicRunner(BaseRunner):
    """Drives Claude through the plain Anthropic ``/v1/messages`` API.

    Uses the public Anthropic schema: image blocks via ``{"type":"image","source":
    {"type":"url"|"base64",...}}`` and a text/tool-use response in
    ``content[]``. The agent's tool protocol is still the text-based
    ``<tool_call>`` convention, so we only need to surface the assistant's
    text back into the pipeline's ``candidates`` structure.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.model = (
            os.environ.get("ANTHROPIC_MODEL")
            or spec.default_checkpoint
            or config.ANTHROPIC_MODEL
        )
        self.base_url = config.ANTHROPIC_BASE_URL.rstrip("/")
        self.token = config.ANTHROPIC_API_KEY
        self.version = config.ANTHROPIC_VERSION
        # Inline resized frames as base64 to bound request size. Cache by URL
        # since the agent loop re-sends the
        # initial frames on every turn.
        self._img_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _url_to_image_block(self, url: str) -> Optional[Dict[str, Any]]:
        if url in self._img_cache:
            return self._img_cache[url]
        block: Optional[Dict[str, Any]] = None
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            import base64 as _b64
            raw = resp.content
            media = "image/png"
            data: str
            try:
                import io as _io
                from PIL import Image as _Img
                le = int(os.environ.get("CLAUDE_FRAME_LONG_EDGE", "512"))
                q = int(os.environ.get("CLAUDE_FRAME_JPEG_Q", "85"))
                im = _Img.open(_io.BytesIO(raw))
                w, h = im.size
                s = le / max(w, h)
                if s < 1:
                    im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
                buf = _io.BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=q)
                data = _b64.b64encode(buf.getvalue()).decode()
                media = "image/jpeg"
            except Exception:  # PIL unavailable / decode fail -> raw bytes
                data = _b64.b64encode(raw).decode()
                ct = resp.headers.get("Content-Type", "image/png").split(";")[0]
                media = ct if ct.startswith("image/") else "image/png"
            block = {"type": "image",
                     "source": {"type": "base64", "media_type": media, "data": data}}
        except Exception as exc:  # noqa: BLE001
            logger.warning("frame fetch failed (%s): %s", url[:80], exc)
        self._img_cache[url] = block
        return block

    def _to_anthropic(self, contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reuse to_claude_messages, then normalise image parts to the
        official Anthropic image-block schema. URL images are fetched and
        inlined as base64 to control frame resolution."""
        msgs = messages.to_claude_messages(contents)
        out: List[Dict[str, Any]] = []
        for m in msgs:
            blocks: List[Dict[str, Any]] = []
            for b in m.get("content", []):
                t = b.get("type")
                if t == "image_url":  # {"type":"image_url","value":<url>}
                    blk = self._url_to_image_block(b.get("value", ""))
                    if blk:
                        blocks.append(blk)
                elif t == "image":  # already {"source":{"base64",...}}
                    blocks.append(b)
                elif t == "text":
                    blocks.append({"type": "text", "text": b.get("text", "")})
            if blocks:
                out.append({"role": m.get("role", "user"), "content": blocks})
        return out

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        if not self.token:
            return _empty_response(self.display_name, "ANTHROPIC_API_KEY not set")
        cfg = cfg or InferenceConfig()
        if not self.model:
            return _empty_response(self.display_name, "Set ANTHROPIC_MODEL to your model ID")
        url = self.base_url + ("/messages" if self.base_url.endswith("/v1") else "/v1/messages")
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages": self._to_anthropic(contents),
        }
        if system_instruction:
            payload["system"] = system_instruction
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.token,
            "anthropic-version": self.version,
        }
        last_err = ""
        for attempt in range(1, 6):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=180)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
                    time.sleep(min(30, 2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(30, 2 ** attempt))
        else:
            return _empty_response(self.display_name, last_err or "request failed")

        text_parts: List[Dict[str, str]] = []
        for item in data.get("content", []) or []:
            if item.get("type") == "text" and item.get("text"):
                text_parts.append({"text": item["text"]})
        usage = data.get("usage", {}) or {}
        usage_metadata = {
            "promptTokenCount": usage.get("input_tokens", 0),
            "candidatesTokenCount": usage.get("output_tokens", 0),
            "totalTokenCount": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": text_parts or [{"text": ""}]},
                    "finishReason": data.get("stop_reason", "STOP") or "STOP",
                }
            ],
            "usageMetadata": usage_metadata,
            "modelVersion": self.display_name,
            "raw_response": data,
        }


# ---------------------------------------------------------------------------
# Optional Seed model via a user-configured OpenAI-compatible endpoint
# ---------------------------------------------------------------------------


class SeedRunner(BaseRunner):
    """Drive the agent loop with Doubao-Seed-2.0-pro (or any OpenAI-compatible
    chat model exposed at ``SEED_API_BASE``).

    This is a *plain HTTP runner*: no GPU, no local model load. Each turn
    converts Gemini-style ``contents`` to OpenAI Chat Completions messages
    and POSTs them to the gateway. The response text is wrapped back into a
    Gemini-shape ``candidates[*].content.parts`` so the rest of the pipeline
    (tool parsing, image_paths_dict, trajectory writeout) works unchanged.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.api_base: str = ""
        self.api_key: str = ""
        self.model: str = ""
        self.timeout: int = 120
        self.max_retry: int = 4
        self.retry_sleep: float = 1.0
        # cache: (abs_path, mtime, size) -> presigned https URL
        # Avoids re-uploading the same source video across agent turns.
        self._video_upload_cache: Dict[Tuple[str, float, int], str] = {}

    def load(self) -> None:
        self.api_base = os.environ.get(
            "SEED_API_BASE", ""
        ).rstrip("/")
        self.api_key = os.environ.get("SEED_API_KEY", "")
        # Require a single API key or a key rotation pool from the environment.
        if not self.api_key and not _init_seed_keys():
            raise RuntimeError(
                "Set SEED_API_KEY or SEED_API_KEYS in your environment."
            )
        # default model can come from the registry's default_checkpoint slot,
        # which we'll populate to "Doubao-Seed-2.0-pro".
        self.model = os.environ.get("SEED_MODEL", self.spec.default_checkpoint or "Doubao-Seed-2.0-pro")
        self.timeout = int(os.environ.get("SEED_TIMEOUT", "120"))
        self.max_retry = int(os.environ.get("SEED_MAX_RETRY", "1000"))
        self.retry_sleep = float(os.environ.get("SEED_RETRY_SLEEP_SEC", "1"))
        if not self.api_key and not _init_seed_keys():
            raise RuntimeError("Neither SEED_API_KEY nor SEED_API_KEYS is set.")
        logger.info(
            "SeedRunner ready: model=%s endpoint=%s max_retry=%d retry_sleep=%.1fs",
            self.model, self.api_base, self.max_retry, self.retry_sleep,
        )

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        # ``self.model`` is only set by load(); use it as the loaded flag so
        # the SEED_API_KEYS-only mode (empty self.api_key) still loads once.
        if not self.model:
            self.load()
        cfg = cfg or InferenceConfig()

        oai_messages = messages.to_oai_messages(contents)
        self._rewrite_local_videos_in_place(oai_messages)
        body_messages: List[Dict[str, Any]] = []
        if system_instruction:
            body_messages.append({"role": "system", "content": system_instruction})
        body_messages.extend(oai_messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": body_messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.request_id:
            payload["rid"] = cfg.request_id
        url = self.api_base + "/chat/completions"

        last_error: Optional[Exception] = None
        attempt = 0
        while attempt < self.max_retry:
            # Pick the key per attempt so a 429 retry rotates to another key
            # instead of hammering the same rate-limited appkey. Falls back to
            # the single self.api_key when SEED_API_KEYS is unset.
            headers = {
                "Authorization": f"Bearer {_pick_seed_key(self.api_key)}",
                "Content-Type": "application/json",
            }
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    # (connect_timeout, read_timeout). Without an explicit
                    # connect timeout, requests inherits the OS default and
                    # an unresponsive keep-alive connection can stall the
                    # whole agent loop for self.timeout seconds before raising.
                    timeout=(10.0, self.timeout),
                )
                if resp.status_code == 429:
                    attempt += 1
                    # Record the 429 so an exhausted retry budget surfaces the
                    # real cause ("rate-limited") instead of "Error: None".
                    last_error = RuntimeError(
                        f"HTTP 429 rate-limited (retries exhausted at {attempt})"
                    )
                    # Flat retry_sleep (default 1s) instead of exponential
                    # backoff: with SEED_MAX_RETRY=1000 the key-rotation pool
                    # keeps retrying until a key frees up, and a long backoff
                    # would stall a single case for minutes. Keep pressure light
                    # by rotating keys, not by sleeping longer.
                    logger.info(
                        "Seed 429 rate-limit (retry %d/%d), sleep %.1fs",
                        attempt, self.max_retry, self.retry_sleep,
                    )
                    time.sleep(self.retry_sleep)
                    continue
                if resp.status_code != 200:
                    body_snip = resp.text[:500]
                    last_error = RuntimeError(
                        f"HTTP {resp.status_code}: {body_snip[:200]}"
                    )
                    logger.warning(
                        "Seed gateway HTTP %s body=%s",
                        resp.status_code, body_snip[:200],
                    )
                    if resp.status_code == 450:
                        logger.warning(
                            "Seed gateway 450 (content moderation) — unrecoverable, skipping case"
                        )
                        return _empty_response(self.display_name, str(last_error))
                    if resp.status_code == 400 and _seed_error_non_retryable(
                        body_snip
                    ):
                        logger.warning(
                            "Seed gateway 400 (non-retryable client error) — aborting turn"
                        )
                        return _empty_response(self.display_name, str(last_error))
                    attempt += 1
                    time.sleep(self.retry_sleep)
                    continue
                data = resp.json()
                break
            except Exception as exc:
                last_error = exc
                attempt += 1
                logger.warning(
                    "Seed gateway request failed (attempt %d/%d): %s",
                    attempt, self.max_retry, exc,
                )
                time.sleep(self.retry_sleep)
        else:
            return _empty_response(self.display_name, str(last_error))

        choices = data.get("choices") or []
        text = ""
        finish = "STOP"
        if choices:
            msg = choices[0].get("message", {}) or {}
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            # Doubao-Seed-2.0-pro is an o1-style reasoning model: it
            # frequently splits a single ReAct turn into:
            #   - ``reasoning_content``: the <think>...</think> chain
            #   - ``content``: the <tool_call>...</tool_call> block
            # The agent loop on the next turn needs to see BOTH so its
            # reasoning remains available as context. Wrap reasoning_content
            # in a <think> block when content does not already contain one.
            has_think_in_content = "<think>" in content
            if reasoning and not has_think_in_content:
                rc = reasoning.strip()
                if content:
                    text = f"<think>\n{rc}\n</think>\n{content}"
                else:
                    text = rc
            else:
                text = content or reasoning
            finish = (choices[0].get("finish_reason") or "STOP").upper()

        usage = data.get("usage") or {}
        usage_meta: Dict[str, int] = {}
        if usage:
            usage_meta = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }

        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish,
                }
            ],
            "usageMetadata": usage_meta,
            "modelVersion": self.display_name,
            "raw_response": data,
        }


    def _rewrite_local_videos_in_place(self, oai_messages: List[Dict[str, Any]]) -> None:
        """Walk ``oai_messages`` and replace any local-path ``video_url``
        parts with a public asset URL of the prepped clip.

        Idempotent: already-HTTPS URLs are left alone (already remote).
        Cached by (abs_path, mtime, size) so repeated agent turns on the
        same source video upload only once.
        """

        # Lazy import: ffmpeg-based prep lives in ``preprocessing/`` since
        # the deepresearch runtime never touches raw video. We still allow
        # callers who hand SeedRunner a local-path video to fall back to
        # the offline prepper.
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(here)
        for extra in (repo, os.path.join(repo, "preprocessing")):
            if extra not in sys.path:
                sys.path.insert(0, extra)
        try:
            from video_prep import prep_video_for_seed  # type: ignore
            from upload import upload_cos  # type: ignore
        except ImportError as exc:
            logger.warning(
                "Seed video prep unavailable (%s); passing video_url parts through unchanged.",
                exc,
            )
            return

        for msg in oai_messages:
            parts = msg.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict) or part.get("type") != "video_url":
                    continue
                vu = part.get("video_url")
                if not isinstance(vu, dict):
                    continue
                url = vu.get("url", "")
                local = self._video_url_to_local_path(url)
                if local is None:
                    continue  # already an https/data URI -- pass through
                try:
                    new_url = self._prep_and_upload_video(local, prep_video_for_seed, upload_cos)
                except Exception as exc:
                    logger.warning(
                        "Seed video prep failed for %s: %s; sending original URL",
                        local, exc,
                    )
                    continue
                vu["url"] = new_url
                # We did client-side sampling; do not let any leftover
                # ``fps`` override pull more frames than the repacked clip has.
                vu.pop("fps", None)

    @staticmethod
    def _video_url_to_local_path(url: str) -> Optional[str]:
        """Return the abs filesystem path if ``url`` points to a local file.

        Handles ``file://``, bare absolute paths, and bare relative paths
        that exist on disk. Returns ``None`` for http(s) and data URIs.
        """
        if not url:
            return None
        if url.startswith("file://"):
            return url[len("file://"):]
        if url.startswith(("http://", "https://", "data:")):
            return None
        if os.path.isabs(url) and os.path.exists(url):
            return url
        if os.path.exists(url):
            return os.path.abspath(url)
        return None

    def _prep_and_upload_video(
        self,
        local_path: str,
        prep_fn,
        upload_fn,
    ) -> str:
        """Prep + upload + cache. Returns presigned HTTPS URL.

        Refresh cached URLs when their remaining lifetime falls below
        ``SEED_VIDEO_URL_REFRESH_THRESHOLD`` seconds (default 600).
        """
        st = os.stat(local_path)
        cache_key = (os.path.abspath(local_path), st.st_mtime, st.st_size)
        cached = self._video_upload_cache.get(cache_key)
        if cached and self._url_still_fresh(cached):
            return cached

        prepped = prep_fn(local_path)
        _, signed_url = upload_fn(
            local_path=prepped.output_path,
            filename=os.path.basename(prepped.output_path),
            date_str=time.strftime("%Y-%m-%d"),
            mode="seed_runner",
            userid="seed-runner",
        )
        if not signed_url:
            raise RuntimeError(f"upload_cos returned no URL for {prepped.output_path}")
        logger.info(
            "Seed prepped video %s -> %d frames @ long_edge=%d, est_tokens=%d",
            os.path.basename(local_path),
            prepped.target_frames,
            prepped.long_edge,
            prepped.expected_prompt_tokens,
        )
        self._video_upload_cache[cache_key] = signed_url
        return signed_url

    @staticmethod
    def _url_still_fresh(url: str) -> bool:
        """Return True if the presigned URL has more than the refresh
        threshold left on its ``Expires=<epoch_seconds>`` query param.

        Defaults to 600 s (10 min) margin — long enough that a single
        ReAct turn won't race the expiry.
        """
        try:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(url).query)
            expires = int(qs.get("Expires", ["0"])[0])
        except Exception:
            return False
        if not expires:
            return False
        threshold = int(os.environ.get("SEED_VIDEO_URL_REFRESH_THRESHOLD", "600"))
        return (expires - time.time()) > threshold


# ---------------------------------------------------------------------------
# Local vLLM (OpenAI-compatible server hosting a Qwen3-VL checkpoint)
# ---------------------------------------------------------------------------


class VLLMRunner(BaseRunner):
    """Drive the agent loop against a local vLLM OpenAI-compatible server.

    This is the local-GPU sibling of :class:`SeedRunner`: same plain-HTTP
    contract (convert Gemini ``contents`` to OpenAI Chat Completions and POST
    to ``/chat/completions``), but pointed at a self-hosted vLLM server
    (``VLLM_API_BASE``, default ``http://localhost:8001/v1``) instead of the
    Seed gateway. No GPU work happens here — the server owns the model.

    Differences from SeedRunner:
      * No additional video upload / ffmpeg preparation. The video pipeline feeds a presigned
        https mp4 URL from the grids manifest, which the vLLM server fetches and
        samples natively. Frame sampling (fps / max_pixels / max_frames) is
        configured on the server via ``--mm-processor-kwargs``, so we drop the
        per-request rewrite. ``file://`` URLs also work when the server is
        launched with ``--allowed-local-media-path``.
      * Auth is optional: local vLLM defaults to no key (``VLLM_API_KEY`` empty
        sends ``Bearer EMPTY`` to satisfy the OpenAI client contract).
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.api_base: str = ""
        self.api_key: str = ""
        self.model: str = ""
        self.timeout: int = 300
        self.max_retry: int = 5
        self.retry_sleep: float = 2.0

    def load(self) -> None:
        self.api_base = os.environ.get(
            "VLLM_API_BASE", "http://localhost:8001/v1"
        ).rstrip("/")
        # Local vLLM ships with no auth by default; send a placeholder so the
        # Authorization header is always well-formed.
        self.api_key = os.environ.get("VLLM_API_KEY", "") or "EMPTY"
        self.model = os.environ.get(
            "VLLM_MODEL", self.spec.default_checkpoint or "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.timeout = int(os.environ.get("VLLM_TIMEOUT", "300"))
        self.max_retry = int(os.environ.get("VLLM_MAX_RETRY", "5"))
        self.retry_sleep = float(os.environ.get("VLLM_RETRY_SLEEP_SEC", "2"))
        logger.info(
            "VLLMRunner ready: model=%s endpoint=%s max_retry=%d",
            self.model, self.api_base, self.max_retry,
        )

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        if not self.model:
            self.load()
        cfg = cfg or InferenceConfig()

        oai_messages = messages.to_oai_messages(contents)
        body_messages: List[Dict[str, Any]] = []
        if system_instruction:
            body_messages.append({"role": "system", "content": system_instruction})
        body_messages.extend(oai_messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": body_messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.request_id:
            payload["rid"] = cfg.request_id
        url = self.api_base + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        attempt = 0
        data: Optional[Dict[str, Any]] = None
        while attempt < self.max_retry:
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(10.0, self.timeout),
                )
                if resp.status_code != 200:
                    body_snip = resp.text[:500]
                    last_error = RuntimeError(
                        f"HTTP {resp.status_code}: {body_snip[:200]}"
                    )
                    logger.warning(
                        "vLLM server HTTP %s body=%s",
                        resp.status_code, body_snip[:200],
                    )
                    attempt += 1
                    time.sleep(self.retry_sleep)
                    continue
                data = resp.json()
                break
            except Exception as exc:
                last_error = exc
                attempt += 1
                logger.warning(
                    "vLLM server request failed (attempt %d/%d): %s",
                    attempt, self.max_retry, exc,
                )
                time.sleep(self.retry_sleep)
        else:
            return _empty_response(self.display_name, str(last_error))

        choices = data.get("choices") or []
        text = ""
        finish = "STOP"
        if choices:
            msg = choices[0].get("message", {}) or {}
            content = msg.get("content", "") or ""
            # Qwen3-VL served under vLLM may emit a separate reasoning stream
            # (``reasoning_content``) when a thinking template is enabled. Mirror
            # SeedRunner: recover it as a <think> block so the next agent turn
            # sees its own chain-of-thought.
            reasoning = msg.get("reasoning_content", "") or ""
            has_think_in_content = "<think>" in content
            if reasoning and not has_think_in_content:
                rc = reasoning.strip()
                text = f"<think>\n{rc}\n</think>\n{content}" if content else rc
            else:
                text = content or reasoning
            finish = (choices[0].get("finish_reason") or "STOP").upper()

        usage = data.get("usage") or {}
        usage_meta: Dict[str, int] = {}
        if usage:
            usage_meta = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }

        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish,
                }
            ],
            "usageMetadata": usage_meta,
            "modelVersion": self.display_name,
            "raw_response": data,
        }


# ---------------------------------------------------------------------------
# In-process vLLM (vllm.LLM instantiated directly, no HTTP server)
# ---------------------------------------------------------------------------


class VLLMInProcessRunner(BaseRunner):
    """Drive the agent loop with an in-process ``vllm.LLM`` engine.

    Same OpenAI message format as :class:`VLLMRunner` (reuses
    :func:`messages.to_oai_messages`), but instead of POSTing to a separate
    server we hold the engine in this process and call ``llm.chat(...)``.

    Trade-offs vs the HTTP runner:
      * No server to stand up — one process does everything. Good for
        single-process offline batch inference / alignment / debugging.
      * Images never leave the process as URLs; vLLM decodes the data: URI
        locally, so there is no download hop that can silently fetch the
        wrong picture.
      * Weights are resident in THIS process, so N concurrent workers each
        pay a full copy — prefer the HTTP runner for high-concurrency
        production.
    """

    def __init__(
        self,
        spec: ModelSpec,
        checkpoint: str = "",
        gpus: str = "0",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint
        self.gpus = gpus
        self.dtype_name = dtype
        self._llm = None
        self._sampling_cls = None

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.checkpoint:
            raise RuntimeError(
                f"Checkpoint not configured for {self.spec.name!r}. "
                "Provide --checkpoint or set QWEN3VL_8B_PATH."
            )

        # CRITICAL: do NOT touch torch.cuda before constructing ``LLM``.
        # vLLM V1 spawns an EngineCore subprocess; if CUDA is already
        # initialized in this parent (e.g. via _resolve_devices ->
        # torch.cuda.is_available()), the child dies with
        # "Cannot re-initialize CUDA in forked subprocess". So we parse the
        # GPU list purely as a string and force the spawn start method.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        gpu_ids = [p.strip() for p in (self.gpus or "0").split(",") if p.strip() != ""]
        if not gpu_ids:
            gpu_ids = ["0"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        tp_size = len(gpu_ids)

        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "vllm is required for the in-process backend. "
                "Install it or use --model vllm (HTTP server)."
            ) from exc
        self._sampling_cls = SamplingParams

        # Engine knobs mirror serve_qwen3vl_vllm.sh so HTTP and in-process
        # produce identical tokenisation.
        limit_images = int(os.environ.get("VLLM_LIMIT_IMAGES", "32"))
        limit_videos = int(os.environ.get("VLLM_LIMIT_VIDEOS", "2"))
        media_io_kwargs = {"video": {"num_frames": -1}}
        mm_processor_kwargs = {
            "fps": config.VIDEO_FPS,
        }
        enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "").lower() in (
            "1", "true", "yes",
        )
        self._llm = LLM(
            model=self.checkpoint,
            tensor_parallel_size=tp_size,
            dtype=self.dtype_name,
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM", "0.85")),
            max_model_len=int(os.environ.get("VLLM_MAX_LEN", "32768")),
            limit_mm_per_prompt={"image": limit_images, "video": limit_videos},
            media_io_kwargs=media_io_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        logger.info(
            "VLLMInProcessRunner ready: model=%s tp=%d dtype=%s",
            self.checkpoint, tp_size, self.dtype_name,
        )

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        if self._llm is None:
            self.load()
        cfg = cfg or InferenceConfig()

        oai_messages = messages.to_oai_messages(contents)
        body_messages: List[Dict[str, Any]] = []
        if system_instruction:
            body_messages.append({"role": "system", "content": system_instruction})
        body_messages.extend(oai_messages)

        req_mm_kwargs: Optional[Dict[str, Any]] = None
        max_pixels = int(config.VIDEO_MAX_PIXELS)
        n_frames = 0
        for msg in body_messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "video_url":
                    if part.get("max_pixels"):
                        max_pixels = int(part.pop("max_pixels"))
                    if part.get("n_frames"):
                        n_frames = int(part.pop("n_frames"))
                    part.pop("min_pixels", None)
        if n_frames > 0:
            t_bar = max(2, round(n_frames / 2) * 2)
            longest = max_pixels * t_bar
            req_mm_kwargs = {
                "fps": config.VIDEO_FPS,
                "size": {"shortest_edge": 4096, "longest_edge": longest},
            }

        sp_kwargs: Dict[str, Any] = {
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.temperature <= 0:
            sp_kwargs["top_p"] = 1.0
            sp_kwargs["top_k"] = -1
        sp = self._sampling_cls(**sp_kwargs)
        try:
            chat_kwargs: Dict[str, Any] = {"sampling_params": sp, "use_tqdm": False}
            if req_mm_kwargs is not None:
                chat_kwargs["mm_processor_kwargs"] = req_mm_kwargs
            outputs = self._llm.chat(body_messages, **chat_kwargs)
        except Exception as exc:
            logger.error("In-process vLLM chat failed: %s", exc, exc_info=True)
            return _empty_response(self.display_name, str(exc))

        if not outputs:
            return _empty_response(self.display_name, "empty vLLM output")
        req = outputs[0]
        comp = req.outputs[0] if req.outputs else None
        text = comp.text if comp is not None else ""
        finish = (getattr(comp, "finish_reason", None) or "STOP").upper()

        prompt_len = len(getattr(req, "prompt_token_ids", None) or [])
        gen_len = len(getattr(comp, "token_ids", None) or []) if comp else 0
        usage_meta = {
            "promptTokenCount": prompt_len,
            "candidatesTokenCount": gen_len,
            "totalTokenCount": prompt_len + gen_len,
        }
        return _wrap_gemini(text, finish, usage_meta, self.display_name)


# ---------------------------------------------------------------------------
# Qwen3-VL (dense and MoE)
# ---------------------------------------------------------------------------


def _patch_moe_scatter_dtype() -> None:
    """Realign router weights to the input dtype for Qwen3-VL MoE.

    Older Transformers releases scatter the router weights with the
    softmax dtype which does not match ``hidden_states`` when the model
    runs in bfloat16 / float16. This patch mirrors the upstream fix so
    the 30B-A3B checkpoint can be served immediately.
    """

    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextSparseMoeBlock,
        )
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on transformers
        logger.warning("MoE scatter patch skipped: %s", exc)
        return

    def _patched_forward(self, hidden_states):  # type: ignore[no-untyped-def]
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weights = torch.zeros(
            router_logits.shape,
            dtype=routing_weights.dtype,
            device=router_logits.device,
        ).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out, router_logits

    Qwen3VLMoeTextSparseMoeBlock.forward = _patched_forward


def _resolve_devices(gpus: str) -> Tuple[List[int], object, str]:
    """Return ``(valid_ids, device_map, device_str)``.

    The device map follows Hugging Face conventions:

    * ``"auto"`` for multi-GPU model parallel
    * ``{"": "cuda:N"}`` for single-GPU placement
    * ``"cpu"`` when CUDA is not available
    """

    import torch

    if not torch.cuda.is_available():
        return [], "cpu", "cpu"

    requested = [
        int(part.strip()) for part in (gpus or "0").split(",") if part.strip() != ""
    ] or [0]
    available = list(range(torch.cuda.device_count()))
    valid = [g for g in requested if g in available]
    if not valid:
        valid = available[:1]
    if len(valid) >= 2:
        return valid, "auto", f"cuda:{valid[0]}"
    return valid, {"": f"cuda:{valid[0]}"}, f"cuda:{valid[0]}"


class Qwen3VLRunner(BaseRunner):
    """Local inference for Qwen3-VL dense (8B/32B) and MoE (30B-A3B)."""

    def __init__(
        self,
        spec: ModelSpec,
        checkpoint: str,
        gpus: str = "0",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint
        self.gpus = gpus
        self.dtype_name = dtype
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._processor = None
        self._device: Optional[object] = None
        self._uses_old_api = False
        self._process_vision_info = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.checkpoint:
            raise RuntimeError(
                f"Checkpoint not configured for {self.spec.name!r}. "
                "Provide --checkpoint or set the matching env variable."
            )

        if self.spec.needs_moe_patch:
            _patch_moe_scatter_dtype()

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qwen_vl_utils is required. Install with: pip install qwen-vl-utils"
            ) from exc
        self._process_vision_info = process_vision_info


        if self.spec.family == "qwen3_vl_moe":
            try:
                from transformers import (  # type: ignore
                    Qwen3VLMoeForConditionalGeneration as ModelClass,
                )
                self._uses_old_api = False
            except ImportError:
                from transformers import (  # type: ignore
                    Qwen3VLForConditionalGeneration as ModelClass,
                )
                self._uses_old_api = True
        else:
            from transformers import (  # type: ignore
                Qwen3VLForConditionalGeneration as ModelClass,
            )
            self._uses_old_api = True

        valid_ids, device_map, device_str = _resolve_devices(self.gpus)
        self._device = device_str

        import torch

        if os.environ.get("HF_DISABLE_CUDNN", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            torch.backends.cudnn.enabled = False
            logger.warning(
                "HF_DISABLE_CUDNN=1: using native CUDA fallback for vision convolutions"
            )

        torch_dtype = getattr(torch, self.dtype_name, torch.bfloat16)

        self._validate_checkpoint(self.checkpoint)

        load_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.spec.family == "qwen3_vl_moe" and not self._uses_old_api:
            load_kwargs["dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype
        attn_implementation = os.environ.get("HF_ATTN_IMPLEMENTATION", "").strip()
        if attn_implementation:
            if attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
                raise ValueError(
                    "HF_ATTN_IMPLEMENTATION must be eager, sdpa or flash_attention_2"
                )
            load_kwargs["attn_implementation"] = attn_implementation

        logger.info(
            "Loading %s from %s (device_map=%s, dtype=%s, attn=%s)",
            self.spec.display_name,
            self.checkpoint,
            device_map,
            torch_dtype,
            attn_implementation or "checkpoint_default",
        )
        self._model = ModelClass.from_pretrained(self.checkpoint, **load_kwargs)
        self._model.eval()

        if self.spec.family == "qwen3_vl_moe":
            self._enforce_dtype(torch_dtype)

        self._processor = self._load_processor()

        if isinstance(device_map, dict):
            logger.info("Single-GPU deployment: %s", device_str)
        elif device_map == "auto":
            logger.info(
                "Model parallel deployment across GPUs %s",
                ", ".join(map(str, valid_ids)),
            )
        else:
            logger.info("CPU deployment")

    def _validate_checkpoint(self, path: str) -> None:
        if "/" not in path or os.path.exists(path):
            if os.path.exists(path) and os.path.isdir(path):
                if not os.path.exists(os.path.join(path, "config.json")):
                    raise FileNotFoundError(
                        f"Missing config.json under checkpoint dir: {path}"
                    )
                has_weights = (
                    bool(list(Path(path).glob("*.safetensors")))
                    or bool(list(Path(path).glob("*.bin")))
                    or os.path.exists(
                        os.path.join(path, "model.safetensors.index.json")
                    )
                )
                if not has_weights:
                    logger.warning(
                        "Checkpoint dir %s has no .safetensors/.bin weights",
                        path,
                    )

    def _enforce_dtype(self, target_dtype) -> None:
        """Recursively cast every floating parameter to ``target_dtype``."""


        def _cast(module):
            for _, param in module.named_parameters(recurse=False):
                if param is not None and param.dtype.is_floating_point:
                    param.data = param.data.to(dtype=target_dtype)
            for _, buffer in module.named_buffers(recurse=False):
                if buffer is not None and buffer.dtype.is_floating_point:
                    buffer.data = buffer.data.to(dtype=target_dtype)
            for child in module.children():
                _cast(child)

        _cast(self._model)
        self._model = self._model.to(dtype=target_dtype)

    # Match the pixel caps used during SFT.
    # Image caps are applied on the processor (used by image resize paths).
    # Video caps MUST also be set on each message video element — qwen_vl_utils
    # process_vision_info reads ele["max_pixels"], not processor.video_max_pixels.
    _SFT_IMAGE_MAX_PIXELS: int = 262144   # 512×512
    _SFT_IMAGE_MIN_PIXELS: int = 1024     # LLaMA-Factory ProcessorArguments default
    _SFT_VIDEO_MAX_PIXELS: int = 100352
    _SFT_VIDEO_MAX_FRAMES: int = 128
    _SFT_VIDEO_FPS: float = 2.0
    _SFT_VIDEO_MIN_PIXELS: int = 256      # LLaMA-Factory ProcessorArguments default

    def _load_processor(self):
        from transformers import AutoProcessor  # type: ignore

        if os.path.exists(self.checkpoint) and os.path.isdir(self.checkpoint):
            preprocessor_config = os.path.join(
                self.checkpoint, "preprocessor_config.json"
            )
            if os.path.exists(preprocessor_config):
                proc = AutoProcessor.from_pretrained(
                    self.checkpoint, trust_remote_code=self.trust_remote_code
                )
                self._apply_pixel_caps(proc)
                return proc
        # Try the bundled checkpoint first; otherwise look for the matching HF id.
        try:
            proc = AutoProcessor.from_pretrained(
                self.checkpoint, trust_remote_code=self.trust_remote_code
            )
            self._apply_pixel_caps(proc)
            return proc
        except Exception as exc:  # pragma: no cover - environment-bound
            fallback = self.spec.default_checkpoint
            if fallback and fallback != self.checkpoint:
                logger.warning(
                    "Processor for %s not found, falling back to %s (%s)",
                    self.checkpoint,
                    fallback,
                    exc,
                )
                proc = AutoProcessor.from_pretrained(
                    fallback, trust_remote_code=self.trust_remote_code
                )
                self._apply_pixel_caps(proc)
                return proc
            raise

    def _apply_pixel_caps(self, proc) -> None:
        """Pin image pixel caps on the processor (SFT-aligned).

        ``video_max_pixels`` is also set for parity with LLaMA-Factory's
        patcher, but infer-time video resize is driven by message-level
        ``max_pixels`` inside ``qwen_vl_utils.fetch_video`` — see
        ``_ensure_video_sft_caps``.
        """
        ip = getattr(proc, "image_processor", None)
        if ip is None:
            return
        setattr(proc, "image_max_pixels", self._SFT_IMAGE_MAX_PIXELS)
        setattr(proc, "image_min_pixels", self._SFT_IMAGE_MIN_PIXELS)
        setattr(proc, "video_max_pixels", self._SFT_VIDEO_MAX_PIXELS)
        if getattr(ip, "max_pixels", None) is None or ip.max_pixels > self._SFT_IMAGE_MAX_PIXELS:
            ip.max_pixels = self._SFT_IMAGE_MAX_PIXELS
        ip.min_pixels = self._SFT_IMAGE_MIN_PIXELS
        logger.info(
            "Processor pixel caps applied: image_max_pixels=%d, video_max_pixels=%d "
            "(video resize still requires message-level max_pixels)",
            self._SFT_IMAGE_MAX_PIXELS,
            self._SFT_VIDEO_MAX_PIXELS,
        )

    @classmethod
    def _ensure_video_sft_caps(cls, qwen_messages: List[Dict[str, Any]]) -> None:
        """Fill missing video sampling knobs so process_vision_info matches SFT.

        Mutates ``qwen_messages`` in place. Safe no-op when knobs are already
        present (e.g. after messages._resolve_video_part).
        """
        defaults = {
            "max_pixels": cls._SFT_VIDEO_MAX_PIXELS,
            "max_frames": cls._SFT_VIDEO_MAX_FRAMES,
            "fps": cls._SFT_VIDEO_FPS,
            "min_pixels": cls._SFT_VIDEO_MIN_PIXELS,
        }
        for msg in qwen_messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "video" and "video" not in part:
                    continue
                for key, value in defaults.items():
                    part.setdefault(key, value)

    # --------------------------------------------------------------- infer

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        if self._model is None:
            self.load()
        cfg = cfg or InferenceConfig()

        try:
            qwen_messages = messages.to_qwen3vl_messages(contents)
            # Use a real ``system`` role (SFT qwen3_vl template), not text
            # prepended into the first user turn.
            qwen_messages = self._inject_system(qwen_messages, system_instruction)
            inputs = self._build_inputs(qwen_messages)
            output_text, prompt_len, gen_len = self._generate(inputs, cfg)
        except Exception as exc:
            logger.error("Qwen3-VL inference failed: %s", exc, exc_info=True)
            return _empty_response(self.display_name, str(exc))

        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": output_text}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_len,
                "candidatesTokenCount": gen_len,
                "totalTokenCount": prompt_len + gen_len,
            },
            "modelVersion": self.display_name,
            "raw_response": {"text": output_text},
        }

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _inject_system(
        msgs: List[Dict[str, Any]], system_instruction: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Prepend an OpenAI-style ``system`` message (matches SFT template).

        The prompt occupies an independent ``<|im_start|>system`` block,
        matching the Qwen chat template.
        """

        if not system_instruction:
            return msgs
        if msgs and msgs[0].get("role") == "system":
            return msgs
        return [
            {"role": "system", "content": system_instruction},
            *msgs,
        ]

    # Back-compat alias for tests / callers that still import the old name.
    _prepend_system = _inject_system

    @staticmethod
    def _unpack_video_inputs(
        video_inputs: Optional[List[Any]],
    ) -> tuple[Optional[List[Any]], Optional[List[Any]]]:
        """Split ``process_vision_info`` video list into tensors + metadata.

        When ``return_video_metadata=True``, each entry is
        ``(tensor, metadata_dict)``. Metadata must be forwarded to the
        Qwen3-VL processor so it can inject ``<{t:.1f} seconds>`` markers
        using the same formula as LLaMA-Factory SFT.
        """

        if not video_inputs:
            return video_inputs, None
        if isinstance(video_inputs[0], tuple) and len(video_inputs[0]) == 2:
            tensors = [v[0] for v in video_inputs]
            metadata = [v[1] for v in video_inputs]
            return tensors, metadata
        return video_inputs, None

    @classmethod
    def _sft_video_timestamps(
        cls,
        metadata: Dict[str, Any],
        *,
        merge_size: int = 2,
        calculate_fn: Optional[Callable[..., List[float]]] = None,
    ) -> List[float]:
        """Mirror ``Qwen3VLProcessor._calculate_timestamps`` for tests/logging."""

        # qwen_vl_utils may return ``frames_indices`` as a multi-element
        # torch.Tensor. Tensor truth-value checks (``value or []``) raise
        # "Boolean value of Tensor ... is ambiguous".
        raw_indices = metadata.get("frames_indices")
        indices = [] if raw_indices is None else list(raw_indices)
        fps = float(metadata.get("fps") or 0.0) or 1.0
        if calculate_fn is not None:
            return list(calculate_fn(indices, fps, merge_size))
        if len(indices) % merge_size != 0:
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
        raw = [idx / fps for idx in indices]
        return [
            (raw[i] + raw[i + merge_size - 1]) / 2
            for i in range(0, len(raw), merge_size)
        ]

    def _build_inputs(self, qwen_messages: List[Dict[str, Any]]):
        # Ensure every video element carries SFT-aligned sampling knobs
        # before qwen_vl_utils.process_vision_info resizes frames.
        self._ensure_video_sft_caps(qwen_messages)
        text = self._processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )

        # Request video metadata so the processor generates temporal markers
        # from the clip's sampling information, matching LLaMA-Factory SFT.
        video_kwargs: dict = {}
        image_inputs = None
        video_inputs = None
        if self._uses_old_api:
            try:
                image_inputs, video_inputs, video_kwargs = self._process_vision_info(
                    qwen_messages,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            except TypeError:
                try:
                    image_inputs, video_inputs, video_kwargs = self._process_vision_info(
                        qwen_messages, return_video_kwargs=True
                    )
                except TypeError:
                    image_inputs, video_inputs = self._process_vision_info(qwen_messages)
        else:
            try:
                image_inputs, video_inputs, video_kwargs = self._process_vision_info(
                    qwen_messages,
                    image_patch_size=self._processor.image_processor.patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            except TypeError:
                image_inputs, video_inputs = self._process_vision_info(qwen_messages)

        video_inputs, video_metadata = self._unpack_video_inputs(video_inputs)
        if video_metadata:
            for meta in video_metadata:
                fps = meta.get("fps") if isinstance(meta, dict) else getattr(meta, "fps", None)
                frame_indices = (
                    meta.get("frames_indices")
                    if isinstance(meta, dict)
                    else getattr(meta, "frames_indices", None)
                )
                n_idx = 0 if frame_indices is None else len(frame_indices)
                logger.info(
                    "Qwen3-VL video_metadata for timestamps: fps=%s n_frames_indices=%d "
                    "(processor will inject <t seconds> like SFT)",
                    fps,
                    n_idx,
                )
                # Spot-check expected first/last stamp for grid.mp4 @ 1fps.
                if isinstance(meta, dict) and meta.get("fps") and n_idx > 0:
                    stamps = self._sft_video_timestamps(meta)
                    if stamps:
                        logger.info(
                            "Expected SFT-style timestamps: n=%d first=%.1f last=%.1f",
                            len(stamps),
                            stamps[0],
                            stamps[-1],
                        )

        if video_inputs:
            extra: Dict[str, Any] = {}
            if video_metadata is not None:
                extra["video_metadata"] = video_metadata
            return self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
                **extra,
            )
        if image_inputs is not None and len(image_inputs) > 0:
            return self._processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        return self._processor(
            text=[text], padding=True, return_tensors="pt"
        )


    def _resolve_io_device(self):
        import torch

        device_map = getattr(self._model, "hf_device_map", None)
        if device_map:
            for value in device_map.values():
                if isinstance(value, torch.device) and value.type == "cuda":
                    return value
                if isinstance(value, str) and "cuda" in value:
                    return torch.device(value)
        return next(self._model.parameters()).device

    def _generate(self, inputs, cfg: InferenceConfig):
        import torch

        device = self._resolve_io_device()
        if hasattr(self._model, "lm_head") and hasattr(self._model.lm_head, "weight"):
            model_dtype = self._model.lm_head.weight.dtype
        else:
            model_dtype = next(self._model.parameters()).dtype

        for key in list(inputs.keys()):
            value = inputs[key]
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    inputs[key] = value.to(device=device, dtype=model_dtype)
                else:
                    inputs[key] = value.to(device=device)
            elif isinstance(value, (list, tuple)) and value and isinstance(
                value[0], torch.Tensor
            ):
                cast = []
                for tensor in value:
                    if tensor.dtype.is_floating_point:
                        cast.append(tensor.to(device=device, dtype=model_dtype))
                    else:
                        cast.append(tensor.to(device=device))
                inputs[key] = type(value)(cast)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "do_sample": cfg.temperature > 0,
        }
        if cfg.temperature > 0:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = 0.8
            gen_kwargs["top_k"] = 20

        with torch.no_grad():
            if model_dtype == torch.bfloat16 and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated = self._model.generate(**inputs, **gen_kwargs)
            else:
                generated = self._model.generate(**inputs, **gen_kwargs)

        prompt_lens = [seq.shape[0] for seq in inputs.input_ids]
        trimmed = [
            out[in_len:] for out, in_len in zip(generated, prompt_lens)
        ]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = decoded[0] if isinstance(decoded, list) else str(decoded)
        return text, int(prompt_lens[0]), int(trimmed[0].shape[0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_runner(
    model_name: str,
    checkpoint: Optional[str] = None,
    gpus: str = "0",
    dtype: str = "bfloat16",
) -> BaseRunner:
    if model_name not in config.MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}. Choose one of: {config.list_model_names()}"
        )
    spec = config.MODEL_REGISTRY[model_name]
    if spec.family == "claude":
        return ClaudeAnthropicRunner(spec)
    if spec.family == "claude_anthropic":
        return ClaudeAnthropicRunner(spec)
    if spec.family == "seed":
        return SeedRunner(spec)
    if spec.family == "vllm":
        return VLLMRunner(spec)
    if spec.family == "vllm_inproc":
        return VLLMInProcessRunner(
            spec,
            checkpoint=config.resolve_checkpoint(spec, checkpoint),
            gpus=gpus,
            dtype=dtype,
        )
    return Qwen3VLRunner(
        spec,
        checkpoint=config.resolve_checkpoint(spec, checkpoint),
        gpus=gpus,
        dtype=dtype,
    )

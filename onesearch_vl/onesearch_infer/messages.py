"""Conversion helpers between Gemini-style content and runner-specific formats."""

from __future__ import annotations

import base64
import io
import logging
import os
import urllib.parse
from typing import Any, Dict, Iterable, List

from PIL import Image

from . import config
from . import image_io


logger = logging.getLogger(__name__)

_SFT_IMAGE_MIN_PIXELS = 1024
_SFT_IMAGE_MAX_PIXELS = 262144


def _qwen_image_part(image: Any) -> Dict[str, Any]:
    return {
        "type": "image",
        "image": image,
        "min_pixels": _SFT_IMAGE_MIN_PIXELS,
        "max_pixels": _SFT_IMAGE_MAX_PIXELS,
    }


def to_claude_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to Claude content blocks.

    Claude has no native video input through this gateway, so video parts
    are converted to a short text placeholder noting which clip the agent
    is supposedly looking at; the agent must rely on ``select_frame``
    derived images instead.
    """

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        for part in parts:
            if "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if url:
                    block.append({"type": "image_url", "value": url})
            elif "video_url" in part or "video" in part:
                ref = (
                    part.get("video_url", {}).get("url", "")
                    if isinstance(part.get("video_url"), dict)
                    else part.get("video", "")
                )
                hint = (
                    f"[Video input not supported by this backend. "
                    f"Reference: {str(ref)[:160]}. "
                    f"Use select_frame / select_timespan to obtain still frames.]"
                )
                block.append({"type": "text", "text": hint})
            elif "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                mime = data.get("mime_type", "") or ""
                if mime.startswith("video/"):
                    block.append(
                        {
                            "type": "text",
                            "text": (
                                "[Inline video stripped: backend lacks video "
                                "input. Use select_frame to grab still frames.]"
                            ),
                        }
                    )
                    continue
                if not mime:
                    mime = image_io.detect_image_format(payload)
                block.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": payload,
                        },
                    }
                )
            elif "text" in part:
                block.append({"type": "text", "text": part["text"]})
        if block:
            claude_role = "assistant" if role == "model" else role
            messages.append({"role": claude_role, "content": block})
    return messages


def _resolve_image_url_to_pil(url: str) -> Image.Image | None:
    """Try the network first, then fall back to ``FVQA_IMAGE_DIR``."""

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        local_path = urllib.parse.unquote(parsed.path)
        try:
            with Image.open(local_path) as image:
                return image.copy()
        except Exception as exc:
            logger.warning("Failed to read file URI image %s: %s", local_path, exc)
            return None

    fetch_url = image_io.cos_url_to_internal(url)
    data = image_io.download_image_bytes(fetch_url)
    if data:
        try:
            return Image.open(io.BytesIO(data))
        except Exception as exc:
            logger.warning("Failed to decode downloaded image: %s", exc)

    if config.FVQA_IMAGE_DIR and os.path.isdir(config.FVQA_IMAGE_DIR):
        candidate = os.path.basename(parsed.path)
        if candidate:
            local = os.path.join(config.FVQA_IMAGE_DIR, candidate)
            if os.path.exists(local):
                try:
                    return Image.open(local)
                except Exception as exc:
                    logger.warning("Failed to read local image %s: %s", local, exc)
    return None


# Sampling knobs that qwen_vl_utils.fetch_video reads from the message
# element (not from processor.video_max_pixels).
_VIDEO_SAMPLING_KEYS = ("fps", "min_pixels", "max_pixels", "max_frames", "total_pixels")


def _sft_video_defaults() -> Dict[str, Any]:
    """Defaults matching SFT joint yaml / qwen_vl_utils VIDEO_FRAME_MIN."""
    return {
        "fps": config.VIDEO_FPS,
        "max_pixels": config.VIDEO_MAX_PIXELS,
        "max_frames": config.VIDEO_MAX_FRAMES,
        "min_pixels": config.VIDEO_MIN_PIXELS,
    }


def _resolve_video_part(part: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert a Gemini-style video part to a Qwen3-VL ``video`` part.

    Accepts ``{"video": <local_path|url>}`` or
    ``{"video_url": {"url": ..., "fps": ..., "max_pixels": ...}}`` or
    ``{"file_path": ..., "fps": ...}``.

    Sampling knobs may live on the part top-level *or* nested inside
    ``video_url`` (pipeline bootstrap uses the nested form). Top-level
    wins on conflict. Missing knobs fall back to SFT-aligned defaults so
    ``process_vision_info`` does not silently use 602112 px/frame.
    """

    nested: Dict[str, Any] = {}
    video_value: Any = None
    if "video_url" in part:
        v = part["video_url"]
        if isinstance(v, dict):
            nested = v
            video_value = v.get("url", "")
        else:
            video_value = str(v)
    elif "video" in part:
        video_value = part["video"]
    if not video_value:
        return None

    # Strip data: URI wrapper to a temp file Qwen can decode.
    if isinstance(video_value, str) and video_value.startswith("data:video/"):
        try:
            mime_and_b64 = video_value.split(",", 1)
            payload = (
                mime_and_b64[1] if len(mime_and_b64) == 2 else video_value.split(";base64,", 1)[1]
            )
            data = base64.b64decode(payload)
        except Exception as exc:
            logger.warning("Failed to decode video data URI: %s", exc)
            return None
        import tempfile

        fh = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        fh.write(data)
        fh.close()
        video_value = fh.name

    qwen_part: Dict[str, Any] = {"type": "video", "video": video_value}
    for key in _VIDEO_SAMPLING_KEYS:
        if key in part:
            qwen_part[key] = part[key]
        elif key in nested:
            qwen_part[key] = nested[key]
    for key, default in _sft_video_defaults().items():
        qwen_part.setdefault(key, default)
    return qwen_part


def to_qwen3vl_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to Qwen3-VL chat messages."""

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        for part in parts:
            if "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                mime = data.get("mime_type", "") or ""
                if "base64," in payload:
                    payload = payload.split("base64,", 1)[1]
                # Inline video: dump to a temp file Qwen can decode.
                if mime.startswith("video/"):
                    try:
                        import tempfile

                        decoded = base64.b64decode(payload)
                        suffix = ".mp4" if "mp4" in mime else ".bin"
                        fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                        fh.write(decoded)
                        fh.close()
                        block.append({"type": "video", "video": fh.name})
                    except Exception as exc:
                        logger.warning("Failed to materialise inline video: %s", exc)
                    continue
                try:
                    pil_image = Image.open(io.BytesIO(base64.b64decode(payload)))
                    block.append(_qwen_image_part(pil_image))
                except Exception as exc:
                    logger.warning("Failed to decode inline base64 image: %s", exc)
                    block.append(_qwen_image_part(payload))
            elif "video_url" in part or "video" in part:
                qwen_video = _resolve_video_part(part)
                if qwen_video is not None:
                    block.append(qwen_video)
            elif "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if not url:
                    continue
                pil_image = _resolve_image_url_to_pil(url)
                if pil_image is not None:
                    block.append(_qwen_image_part(pil_image))
            elif "text" in part:
                block.append({"type": "text", "text": part["text"]})
        if block:
            qwen_role = "assistant" if role == "model" else role
            messages.append({"role": qwen_role, "content": block})
    return messages


def to_oai_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to OpenAI Chat Completions messages.

    Used by the Seed 2.0 (Doubao) backend, which exposes an OpenAI-compatible
    endpoint. Images are passed as ``image_url`` parts; remote URLs are kept
    as-is, inline base64 is wrapped in a ``data:<mime>;base64,...`` URI.

    Video parts are converted to ``video_url`` (data: URI) when the backend
    is known to support video input. For backends that do not, the caller
    should drop the video part before invoking this converter; we keep a
    best-effort passthrough so OpenAI-compatible video-in models still work.
    """

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        oai_parts: List[Dict[str, Any]] = []
        for part in parts:
            if "text" in part and part.get("text"):
                oai_parts.append({"type": "text", "text": part["text"]})
            elif "video_url" in part or "video" in part:
                v = part.get("video_url")
                if isinstance(v, dict):
                    url = v.get("url", "")
                else:
                    url = str(part.get("video", ""))
                if url:
                    vpart: Dict[str, Any] = {"type": "video_url",
                                             "video_url": {"url": url}}
                    # Preserve per-frame budget + frame count so the in-process
                    # vLLM runner can compute the video processor's t_bar total
                    # (longest_edge = max_pixels * t_bar) to match HF/SFT.
                    if isinstance(v, dict):
                        for k in ("max_pixels", "min_pixels", "n_frames"):
                            if v.get(k) is not None:
                                vpart[k] = v[k]
                    oai_parts.append(vpart)
            elif "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if url:
                    oai_parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
            elif "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "") or ""
                if "base64," in payload:
                    payload = payload.split("base64,", 1)[1]
                mime = data.get("mime_type", "") or "image/png"
                oai_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{payload}"},
                    }
                )
        if not oai_parts:
            continue
        oai_role = "assistant" if role == "model" else role
        # If only one text part, collapse to a string for compatibility with
        # endpoints that don't accept a list-of-parts content for plain text.
        if len(oai_parts) == 1 and oai_parts[0]["type"] == "text":
            messages.append({"role": oai_role, "content": oai_parts[0]["text"]})
        else:
            messages.append({"role": oai_role, "content": oai_parts})
    return messages

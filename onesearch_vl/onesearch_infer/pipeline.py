"""Per-case multimodal inference pipeline with a multi-turn agent loop.

Video inputs can use a local MP4 or pre-sampled frames. Bootstrap images,
video clips, and tool outputs share ``image_paths_dict`` so subsequent
turns can reference their registered handles.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from . import config, image_io, messages as messages_mod, tools, video_io
from .runners import BaseRunner, InferenceConfig
from .sft_chat_format import observation_user_message
from .prompts import get_system_prompt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory serialisation
# ---------------------------------------------------------------------------


def _strip_base64_payloads(obj: Any, image_urls: Dict[str, str]) -> Any:
    """Replace large base64 image/video blobs with their URLs / placeholders."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, bytes):
        return f"<bytes: {len(obj)} bytes>"
    if isinstance(obj, str):
        # Bare data: URIs (video or image) embedded as plain strings.
        if obj.startswith("data:video/") and ";base64," in obj:
            return f"<base64_video_data: {len(obj)} chars>"
        if obj.startswith("data:image/") and ";base64," in obj:
            return f"<base64_image_data: {len(obj)} chars>"
        return obj
    if isinstance(obj, list):
        return [_strip_base64_payloads(item, image_urls) for item in obj]
    if isinstance(obj, dict):
        if "source" in obj and isinstance(obj["source"], dict):
            data = obj["source"].get("data", "")
            if isinstance(data, str) and len(data) > 100:
                return {
                    "type": "image",
                    "source": {"type": "base64", "data": f"<base64_image_data: {len(data)} chars>"},
                }
        if "inline_data" in obj and isinstance(obj["inline_data"], dict):
            data = obj["inline_data"].get("data", "")
            mime = obj["inline_data"].get("mime_type", "") or ""
            if isinstance(data, str) and len(data) > 100:
                if mime.startswith("video/"):
                    return {
                        "type": "video_url",
                        "video_url": {"url": "<inline_video_stripped>"},
                    }
                return {
                    "inline_data": {
                        "mime_type": mime or "image/*",
                        "data": f"<base64_image_data: {len(data)} chars>",
                    }
                }
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, str) and len(v) > 1000:
                if any(key in obj for key in ("type", "media_type", "mime_type")):
                    mime = obj.get("media_type") or obj.get("mime_type") or ""
                    if mime.startswith("video/"):
                        out[k] = f"<base64_video_data: {len(v)} chars>"
                    else:
                        out[k] = f"<base64_image_data: {len(v)} chars>"
                    continue
            out[k] = _strip_base64_payloads(v, image_urls)
        return out
    return obj


# ---------------------------------------------------------------------------
# Image bootstrap helpers
# ---------------------------------------------------------------------------


def _next_img_id(image_paths_dict: Dict[str, Any]) -> str:
    """Compute the next ``img_N`` handle name using a prefix-aware counter.

    Counting ``len(dict)`` is wrong as soon as the dict carries non-image
    handles (``video_1`` / ``clip_2`` / ...) — the same number gets
    handed out twice. We always count only ``img_*`` keys.
    """

    return f"img_{sum(1 for k in image_paths_dict if k.startswith('img_')) + 1}"


def _add_image_url(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    url: str,
) -> None:
    image_id = _next_img_id(image_paths_dict)
    image_paths_dict[image_id] = url
    initial_parts.append({"image_url": {"url": url}})


def _add_inline_image(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    payload: bytes | str,
    raw_storage: Any,
    *,
    downscale: bool = True,
) -> None:
    if isinstance(payload, bytes):
        max_side = (
            int(getattr(config, "BOOTSTRAP_FRAME_MAX_SIDE", 0) or 0)
            if downscale
            else 0
        )
        if max_side > 0:
            try:
                img = Image.open(io.BytesIO(payload))
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / float(max(w, h))
                    img = img.convert("RGB").resize(
                        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        Image.LANCZOS,
                    )
                    buf = io.BytesIO()
                    img.save(buf, format="PNG", optimize=True)
                    payload = buf.getvalue()
            except Exception as exc:
                logger.warning("inline downscale failed: %s", exc)
        encoded = image_io.image_to_base64(payload)
    else:
        encoded = payload
    if not encoded:
        return
    image_id = _next_img_id(image_paths_dict)
    image_paths_dict[image_id] = raw_storage
    initial_parts.append(
        {
            "inline_data": {
                "mime_type": image_io.detect_image_format(encoded),
                "data": encoded,
            }
        }
    )


def _read_image_bytes_capped(path: str, max_side: int) -> bytes:
    """Return PNG bytes of ``path``, downscaled so the longer edge <= max_side.

    Returns the original bytes when ``max_side <= 0`` or the image is
    already small enough. Used for in-context inline payloads to keep
    requests under upstream gateway size limits.
    """

    with open(path, "rb") as fh:
        raw = fh.read()
    if max_side <= 0:
        return raw
    try:
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= max_side:
            return raw
        scale = max_side / float(long_edge)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = img.convert("RGB").resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("inline downscale failed for %s: %s", path, exc)
        return raw


def _bootstrap_video(
    manifest_record: Dict[str, Any],
    supports_video: bool = True,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], Optional[int]]:
    """Build the initial (image_paths_dict, parts) pair for a video case.

    ``manifest_record`` is one line from ``grids_manifest.jsonl`` (produced
    by ``preprocessing/preprocess_grids.py``). It carries the presigned S3
    URLs for the 1-fps composed mp4 (``video_url``) and the N evenly-spaced
    frame PNGs (``object_urls`` / ``frame_urls``). The runtime never touches
    raw video; all heavy lifting was done offline.

    When ``supports_video`` is False (e.g. Claude, which has no native video
    input), we cannot send the mp4. Instead we seed the prompt with a small
    set of evenly-spaced frames as ``image_url`` parts so the model gets a
    visual overview, then it uses ``select_frame`` to pull exact frames.
    ``VIDEO_INIT_FRAMES`` controls how many overview frames to send.
    """

    grid = video_io.build_grid_from_manifest(manifest_record)
    image_paths_dict: Dict[str, Any] = {"video_1": grid}
    n_frames = grid["n_frames"]

    if supports_video:
        initial_parts: List[Dict[str, Any]] = [
            {
                "video_url": {
                    "url": grid["video_url"],
                    "fps": config.VIDEO_FPS,
                    "max_pixels": config.VIDEO_MAX_PIXELS,
                    "max_frames": config.VIDEO_MAX_FRAMES,
                    "min_pixels": config.VIDEO_MIN_PIXELS,
                    # n_frames lets the in-process vLLM runner size the video
                    # processor's t_bar total budget so per-frame resolution
                    # matches HF/SFT (longest_edge = max_pixels * t_bar).
                    "n_frames": n_frames,
                }
            }
        ]
        return image_paths_dict, initial_parts, n_frames

    # No native video: seed with evenly-spaced overview frames.
    # Default sends ALL frames (VIDEO_INIT_FRAMES large); set a smaller value
    # to subsample if the context/token budget is tight.
    frame_urls = grid.get("frame_urls") or []
    n_init = int(os.environ.get("VIDEO_INIT_FRAMES", "128"))
    n_init = max(1, min(n_init, len(frame_urls))) if frame_urls else 0
    if n_init and n_init >= len(frame_urls):
        idxs = list(range(len(frame_urls)))
    elif n_init > 1:
        step = (len(frame_urls) - 1) / (n_init - 1)
        idxs = sorted({int(round(i * step)) for i in range(n_init)})
    else:
        idxs = [0] if frame_urls else []
    initial_parts = [
        {
            "text": (
                f"[video_1: {n_frames} evenly-spaced frames indexed 0..{n_frames - 1}. "
                f"Below are {len(idxs)} overview frames at indices {idxs}. "
                f"Use select_frame / select_timespan to inspect exact frames.]"
            )
        }
    ]
    for i in idxs:
        initial_parts.append({"image_url": {"url": frame_urls[i]}})
    return image_paths_dict, initial_parts, n_frames


def _extract_final_response_text(turns: List[Dict[str, Any]]) -> str:
    """Return the answer text to expose to evaluators.

    Prefer the last explicit ``<response>`` block. If the model ended with a
    plain answer, keep that text after removing internal ReAct tags. Avoid
    concatenating historical ``<think>`` / ``<tool_call>`` blocks as answers.
    """

    for turn in reversed(turns):
        text = turn.get("response_text", "") or ""
        match = re.search(r"<response>(.*?)</response>", text, flags=re.DOTALL)
        if match and match.group(1).strip():
            return match.group(0).strip()
    for turn in reversed(turns):
        text = turn.get("response_text", "") or ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL).strip()
        if cleaned:
            return cleaned
    # If the episode exhausts its turn budget while still emitting a tool
    # call, expose the final model round verbatim to the judge.  This matches
    # RL val-only postprocessing and avoids an extra prompt/model call solely
    # to manufacture a final answer.
    for turn in reversed(turns):
        text = str(turn.get("response_text", "") or "").strip()
        if text:
            return text
    return ""


def _bootstrap_images(
    row: Any,
    case_id: str,
    case_idx: int,
    filename_prefix: str,
    image_urls_dict: Optional[Dict[str, List[Any]]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the initial (image_paths_dict, parts) pair from row data.

    Resolution order:

    1. URLs supplied via ``image_urls_dict`` (cheapest, no network).
    2. The configured ``FVQA_IMAGE_DIR`` (loads bytes from disk).
    3. The ``images`` column on the row (parquet payloads).
    4. The singular ``image`` column used by VDR-Bench.
    """

    image_paths_dict: Dict[str, Any] = {}
    initial_parts: List[Dict[str, Any]] = []

    if image_urls_dict and case_id in image_urls_dict:
        for entry in image_urls_dict[case_id]:
            if isinstance(entry, dict):
                url = entry.get("cos_url") or entry.get("image_url") or entry.get("url")
            else:
                url = str(entry)
            if url:
                _add_image_url(image_paths_dict, initial_parts, url)

    if not initial_parts and config.FVQA_IMAGE_DIR:
        local = image_io._try_local_image(case_id)
        if local:
            try:
                with open(local, "rb") as fh:
                    payload = fh.read()
                _add_inline_image(image_paths_dict, initial_parts, payload, local)
            except OSError as exc:
                logger.warning("Failed to read local image %s: %s", local, exc)

    if not initial_parts:
        images_raw = row.get("images", []) if hasattr(row, "get") else []
        if images_raw is None:
            images: List[Any] = []
        elif isinstance(images_raw, (list, tuple)):
            images = list(images_raw)
        elif hasattr(images_raw, "tolist"):
            converted = images_raw.tolist()
            images = converted if isinstance(converted, list) else [converted]
        else:
            images = [images_raw]
        # VDR-Bench stores the input JPEG as one raw-binary ``image`` value
        # instead of an ``images`` list.  Treat it as a one-element list so it
        # follows the exact same inline-image path as FVQA-style payloads.
        if not images and hasattr(row, "get"):
            single_image = row.get("image", None)
            if single_image is not None:
                images = [single_image]
        # Vision-DeepResearch-Eval parquets store image bytes in a sibling
        # ``image_packed`` column as a JSON string of {filename, data(b64)}.
        # ``images`` only carries filenames, so we resolve bytes from there.
        packed_lookup: Dict[str, bytes] = {}
        packed_raw = row.get("image_packed", None) if hasattr(row, "get") else None
        if isinstance(packed_raw, str) and packed_raw:
            try:
                import base64 as _b64
                import json as _json

                for item in _json.loads(packed_raw):
                    if isinstance(item, dict) and "data" in item:
                        try:
                            packed_lookup[item.get("filename", "")] = _b64.b64decode(
                                item["data"]
                            )
                        except Exception as exc:
                            logger.warning(
                                "image_packed decode failed for %s: %s",
                                item.get("filename"),
                                exc,
                            )
            except Exception as exc:
                logger.warning("image_packed parse failed: %s", exc)

        for entry in images:
            if entry is None:
                continue
            url: Optional[str] = None
            payload: Optional[bytes] = None
            if isinstance(entry, dict):
                url = entry.get("url") or entry.get("image_url") or entry.get("cos_url")
                payload = entry.get("bytes")
            elif isinstance(entry, str):
                if entry.startswith(("http://", "https://")):
                    url = entry
                elif os.path.isfile(entry):
                    try:
                        with open(entry, "rb") as fh:
                            payload = fh.read()
                    except OSError as exc:
                        logger.warning("Failed to read local image %s: %s", entry, exc)
                elif entry in packed_lookup:
                    payload = packed_lookup[entry]
                elif packed_lookup:
                    # filename mismatch — fall back to the first packed entry.
                    payload = next(iter(packed_lookup.values()))
            elif isinstance(entry, bytes):
                payload = entry

            if url:
                _add_image_url(image_paths_dict, initial_parts, url)
                continue
            if payload:
                pil_image = Image.open(io.BytesIO(payload))
                from . import cos_upload

                # Clean evaluation can send the exact original bytes directly
                # to the local SGLang server. This avoids making initial model
                # input depend on an external COS upload; tool-produced images
                # still use COS as before. The default remains unchanged for RL.
                inline_original = os.environ.get(
                    "INLINE_ORIGINAL_IMAGE_BYTES", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                cos_url = None
                if not inline_original:
                    cos_url = cos_upload.upload_pil_image(
                        pil_image,
                        filename_prefix,
                        case_idx,
                        0,
                        "original_image",
                    )
                if cos_url:
                    _add_image_url(image_paths_dict, initial_parts, cos_url)
                else:
                    _add_inline_image(
                        image_paths_dict,
                        initial_parts,
                        payload,
                        payload,
                        downscale=not inline_original,
                    )
    return image_paths_dict, initial_parts


# ---------------------------------------------------------------------------
# Shared ReAct agent loop (video / FVQA / MI)
# ---------------------------------------------------------------------------


def run_agent_loop(
    *,
    runner: BaseRunner,
    gemini_contents: List[Dict[str, Any]],
    image_paths_dict: Dict[str, Any],
    trajectory: Dict[str, Any],
    output_dir: str,
    intermediate_dir: str,
    case_id: str,
    case_idx: int,
    filename_prefix: str,
    runner_supports_video: bool,
    visual_lookup: Optional[Callable[..., object]] = None,
    inference_cfg: Optional[InferenceConfig] = None,
    system_instruction: Optional[str] = None,
    modality: str = "video",
    search_backend: Any = None,
) -> Dict[str, Any]:
    """Multi-turn tool loop shared by video, single-image, and multi-image pipelines."""

    cfg = inference_cfg or InferenceConfig()
    sys_prompt = system_instruction if system_instruction is not None else get_system_prompt(modality)

    def _obs(observation_text: str, media_parts: Optional[List[Dict[str, Any]]] = None):
        gemini_contents.append(
            observation_user_message(
                observation_text,
                media_parts,
            )
        )

    def _chat_completions_snapshot(
        input_contents: List[Dict[str, Any]],
        assistant_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Snapshot the messages fed to the model this turn, in RL-episode
        ``chat_completions`` shape: ``[{role, content}, ...]`` = system + the
        accumulated OpenAI-format conversation + (optionally) this turn's
        assistant output. Base64 blobs are stripped later by the whole-
        trajectory ``_strip_base64_payloads`` pass, matching ``response``.
        """
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        msgs.extend(messages_mod.to_oai_messages(input_contents))
        if assistant_text is not None:
            msgs.append({"role": "assistant", "content": assistant_text})
        return msgs

    # Use per-modality turn limits, with explicit overrides for evaluation.
    # Preserve the configured budget so tool calls can finish before the answer.
    sft_max_turns = {
        "single_image": int(os.environ.get("SINGLE_IMAGE_MAX_TURNS", "8")),
        "multi_image": int(os.environ.get("MULTI_IMAGE_MAX_TURNS", "12")),
        "video": int(os.environ.get("VIDEO_AGENT_MAX_TURNS", "12")),
        "audio_video": int(os.environ.get("AV_AGENT_MAX_TURNS", "8")),
    }
    for turn_num in range(min(config.MAX_TURNS, sft_max_turns.get(modality, 12))):
        try:
            response = runner.infer(
                contents=gemini_contents,
                system_instruction=sys_prompt,
                cfg=cfg,
            )
        except Exception as exc:
            logger.error("Inference failed on turn %d: %s", turn_num, exc, exc_info=True)
            trajectory["turns"].append({
                "turn": turn_num,
                "chat_completions": _chat_completions_snapshot(gemini_contents),
                "error": str(exc),
            })
            trajectory["status"] = "error"
            trajectory["error"] = str(exc)
            break

        response_text = ""
        for cand in response.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    response_text += part["text"]

        turn_record: Dict[str, Any] = {
            "turn": turn_num,
            "chat_completions": _chat_completions_snapshot(
                gemini_contents, response_text
            ),
            "response": response,
            "response_text": response_text,
        }
        trajectory["turns"].append(turn_record)

        candidates = response.get("candidates", []) or []
        # A backbone-level failure (retries exhausted, gateway error) comes back
        # as finishReason=ERROR with text like "Error: ...". Do NOT treat that
        # placeholder as the model's final answer — mark the case failed so it
        # is filtered/re-run instead of poisoning the coldstart data.
        if candidates and candidates[0].get("finishReason") == "ERROR":
            err = response_text or "backbone ERROR"
            logger.warning("Turn %d backbone ERROR, marking case failed: %s",
                           turn_num, err[:200])
            trajectory["status"] = "error"
            trajectory["error"] = err
            break

        if candidates:
            gemini_contents.append(
                {
                    "role": "model",
                    "parts": candidates[0].get("content", {}).get("parts", []),
                }
            )

        if tools.has_response_tag(response_text):
            break

        tool_call_json = tools.extract_tool_call(response_text)
        if not tool_call_json:
            # A malformed tagged tool call is not a legitimate final answer.
            # Return a compact protocol error so the model can repair JSON on
            # the next turn, just like a normal tool execution error.
            if "<tool_call>" in response_text:
                observation_text = (
                    "<observation>\n"
                    "Tool execution error:\n"
                    "Malformed tool_call JSON. Emit exactly one valid call as "
                    '{"name":"tool_name","arguments":{...}} or provide a '
                    "final answer inside <response>...</response>.\n"
                    "</observation>"
                )
                logger.warning(
                    "Turn %d contained malformed tool_call JSON; requesting repair.",
                    turn_num,
                )
                turn_record["tool_output"] = observation_text
                if modality == "audio_video":
                    turn_record["protocol_repair"] = True
                _obs(observation_text)
                continue
            if modality == "audio_video":
                observation_text = (
                    "<observation>\nProtocol error: the previous turn contained no complete action or final answer. "
                    "Continue with exactly one executable <tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call> "
                    "or a final answer inside <response>...</response>.\n</observation>"
                )
                turn_record["tool_output"] = observation_text
                turn_record["protocol_repair"] = True
                _obs(observation_text)
                continue
            logger.info("Turn %d ended without tool call or response tag.", turn_num)
            break

        os.makedirs(intermediate_dir, exist_ok=True)
        backend_kwargs = {"search_backend": search_backend} if search_backend is not None else {}
        tool_message, new_images = tools.execute_tool(
            tool_call_json,
            image_paths_dict,
            case_id,
            case_idx,
            turn_num,
            intermediate_dir,
            filename_prefix=filename_prefix,
            visual_lookup=visual_lookup,
            modality=modality,
            **backend_kwargs,
        )
        observation_text = f"<observation>\n{tool_message}\n</observation>"
        turn_record["tool_output"] = observation_text

        if new_images:
            for new_id, payload in new_images.items():
                image_paths_dict[new_id] = payload
                if new_id.startswith("clip_") and isinstance(payload, dict):
                    clip_video_url = payload.get("video_url")
                    if payload.get("kind") == "native_av" and runner_supports_video and clip_video_url:
                        _obs(observation_text, [_native_av_part(payload)])
                        continue
                    if runner_supports_video and clip_video_url:
                        # Media before text — matches SFT ``<video><observation>...``.
                        _obs(
                            observation_text,
                            [
                                {
                                    "video_url": {
                                        "url": clip_video_url,
                                        "fps": config.VIDEO_FPS,
                                        "max_pixels": config.VIDEO_MAX_PIXELS,
                                        "max_frames": config.VIDEO_MAX_FRAMES,
                                        "min_pixels": config.VIDEO_MIN_PIXELS,
                                        # Clip frame count (falls back to the
                                        # max_frames cap) so the in-process vLLM
                                        # runner can size t_bar per-request.
                                        "n_frames": (
                                            payload.get("n_frames")
                                            or config.VIDEO_MAX_FRAMES
                                        ),
                                    }
                                }
                            ],
                        )
                    else:
                        _obs(observation_text)
                    continue
                if isinstance(payload, str) and payload.startswith(
                    ("http://", "https://")
                ):
                    # Media before text — matches SFT ``<image><observation>...``.
                    _obs(
                        observation_text,
                        [{"image_url": {"url": payload}}],
                    )
                elif isinstance(payload, str) and os.path.exists(payload):
                    uploaded_url: Optional[str] = None
                    try:
                        from . import cos_upload as _cos_upload

                        if _cos_upload.upload_available():
                            pil_image = Image.open(payload)
                            uploaded_url = _cos_upload.upload_pil_image(
                                pil_image,
                                filename_prefix,
                                case_idx,
                                turn_num,
                                f"observation_{new_id}",
                            )
                    except Exception as exc:
                        logger.warning("observation upload failed for %s: %s", payload, exc)
                    if uploaded_url:
                        image_paths_dict[new_id] = uploaded_url
                        _obs(
                            observation_text,
                            [{"image_url": {"url": uploaded_url}}],
                        )
                        continue
                    try:
                        max_side = int(getattr(config, "BOOTSTRAP_FRAME_MAX_SIDE", 0) or 0)
                        data = _read_image_bytes_capped(payload, max_side)
                        encoded = image_io.image_to_base64(data) or ""
                        _obs(
                            observation_text,
                            [
                                {
                                    "inline_data": {
                                        "mime_type": image_io.detect_image_format(
                                            encoded
                                        ),
                                        "data": encoded,
                                    }
                                }
                            ],
                        )
                    except OSError as exc:
                        logger.warning("Cannot read intermediate image %s: %s", payload, exc)
                        _obs(observation_text)
                elif isinstance(payload, bytes):
                    max_side = int(getattr(config, "BOOTSTRAP_FRAME_MAX_SIDE", 0) or 0)
                    if max_side > 0:
                        try:
                            img = Image.open(io.BytesIO(payload))
                            w, h = img.size
                            if max(w, h) > max_side:
                                scale = max_side / float(max(w, h))
                                img = img.convert("RGB").resize(
                                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                                    Image.LANCZOS,
                                )
                                buf = io.BytesIO()
                                img.save(buf, format="PNG", optimize=True)
                                payload = buf.getvalue()
                        except Exception as exc:
                            logger.warning("inline downscale failed: %s", exc)
                    encoded = image_io.image_to_base64(payload) or ""
                    _obs(
                        observation_text,
                        [
                            {
                                "inline_data": {
                                    "mime_type": image_io.detect_image_format(
                                        encoded
                                    ),
                                    "data": encoded,
                                }
                            }
                        ],
                    )
                else:
                    _obs(observation_text)
        else:
            _obs(observation_text)

    trajectory["final_response_text"] = _extract_final_response_text(trajectory["turns"])
    # Mark completion status so resume + caller can distinguish a genuinely
    # finished case from one aborted by an inference error (see the except
    # branch above, which sets status="error"). Default to "ok" here.
    trajectory.setdefault("status", "ok")

    image_urls = {
        img_id: data
        for img_id, data in image_paths_dict.items()
        if isinstance(data, str) and data.startswith(("http://", "https://"))
    }
    serialised = _strip_base64_payloads(trajectory, image_urls)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}_trajectory.json")
    # Atomic write: dump to a temp file in the same dir then os.replace, so a
    # preemption mid-write never leaves a truncated file that resume would
    # mistake for a completed case.
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(serialised, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)
    logger.info("Trajectory saved to %s", out_path)
    return trajectory


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_single_case(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
    dataset_type: str = "train",
    visual_lookup: Optional[Callable[..., object]] = None,
    inference_cfg: Optional[InferenceConfig] = None,
    *,
    manifest_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Drive one **video** deep-research case through the agent.

    Requires ``manifest_record`` (one line of ``grids_manifest.jsonl`` from
    ``preprocessing/preprocess_grids.py``): S3 URLs for the composed 1-fps mp4
    and evenly-spaced frame PNGs. Bootstrap registers ``video_1`` for
    ``select_timespan`` / ``select_frame``.

    For **single-image** use ``pipeline_single.process_single_case_single`` /
    ``run_infer_single.py``. For **multi-image** use
    ``pipeline_mi.process_single_case_mi`` / ``run_infer_mi.py``.
    """

    if manifest_record is None:
        raise ValueError(
            "process_single_case requires manifest_record (video DR). "
            "Use pipeline_single.process_single_case_single for image-only cases."
        )

    filename_prefix = "video_dr"

    if hasattr(row, "get"):
        case_id = row.get("data_id", f"case_{case_idx}")
        category = row.get("category", "unknown")
        data_source = row.get("data_source", "unknown")
        prompt_list = row.get("prompt", [])
    else:
        case_id = f"case_{case_idx}"
        category = "unknown"
        data_source = "unknown"
        prompt_list = []

    logger.info(
        "Processing case %d (%s, category=%s, source=%s)",
        case_idx + 1,
        case_id,
        category,
        data_source,
    )

    if isinstance(prompt_list, list) and prompt_list:
        first = prompt_list[0]
        prompt_text = (
            first.get("content", "") if isinstance(first, dict) else str(first)
        )
    else:
        prompt_text = str(prompt_list) if prompt_list else ""

    if not prompt_text and hasattr(row, "get"):
        q = row.get("question", "")
        if isinstance(q, str) and q:
            # BC-VL questions are prefixed with "image_id: <fname> Question: ",
            # which is dataset bookkeeping (image already attached separately)
            # and should not leak into the user-facing prompt.
            import re as _re
            q_clean = _re.sub(
                r"^\s*image_id:\s*[^\s]+\s+Question:\s*", "", q, flags=_re.IGNORECASE
            )
            prompt_text = q_clean.strip()

    intermediate_dir = os.path.join(output_dir, "intermediate")

    runner_supports_video = bool(getattr(getattr(runner, "spec", None), "supports_video", False))

    os.makedirs(intermediate_dir, exist_ok=True)
    image_paths_dict, initial_parts, video_n_frames = _bootstrap_video(
        manifest_record=manifest_record,
        supports_video=runner_supports_video,
    )
    if not initial_parts:
        logger.warning(
            "Video case %s has no resolvable visual input; agent will rely on text only.",
            case_id,
        )

    initial_parts.append({"text": prompt_text})
    sys_prompt = get_system_prompt("video")

    gemini_contents: List[Dict[str, Any]] = [
        {"role": "user", "parts": initial_parts}
    ]

    trajectory: Dict[str, Any] = {
        "case_id": case_id,
        "case_idx": case_idx,
        "category": category,
        "data_source": data_source,
        "modality": "video",
        "prompt": prompt_list,
        "turns": [],
        "timestamp": datetime.now().isoformat(),
    }

    cfg = inference_cfg or InferenceConfig()

    return run_agent_loop(
        runner=runner,
        gemini_contents=gemini_contents,
        image_paths_dict=image_paths_dict,
        trajectory=trajectory,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        case_id=case_id,
        case_idx=case_idx,
        filename_prefix=filename_prefix,
        runner_supports_video=runner_supports_video,
        visual_lookup=visual_lookup,
        inference_cfg=cfg,
        system_instruction=sys_prompt,
        modality="video",
    )


def _native_av_part(handle: Dict[str, Any]) -> Dict[str, Any]:
    """Retain original AV media in the existing runner/message protocol."""
    return {"video_url": {
        "url": handle["video_url"],
        "has_audio": handle["has_audio"],
        "native_av": True,
        "source_start_s": handle["source_start_s"],
        "source_end_s": handle["source_end_s"],
        "clip_duration_s": handle["duration"],
    }}


def process_native_av_case(
    *, question: str, source_path: str, runner: BaseRunner,
    output_dir: str, case_id: str, case_idx: int = 0,
    search_backend: Any = None,
    inference_cfg: Optional[InferenceConfig] = None,
    system_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Bootstrap real AV, then use the original OneSearch multi-turn loop.

    This is an AV entrypoint rather than a second agent loop. source_path is
    private media binding; only the question and actual media enter the actor.
    """
    from .av_io import make_native_av_handle

    if not question.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
        raise ValueError("Nonempty question and safe case ID required")
    if not getattr(runner, "supports_audio_video", False):
        raise ValueError("Native AV requires an audio-video runner")
    handle = make_native_av_handle(source_path)
    media_note = (
        f"video_1: {handle['duration']:.3f} seconds; "
        f"audio track {'present' if handle['has_audio'] else 'absent'}. "
        "inspect_av uses seconds relative to the selected handle."
    )
    contents = [{"role": "user", "parts": [
        _native_av_part(handle), {"text": media_note + "\nQuestion: " + question}
    ]}]
    trajectory = {
        "case_id": case_id, "case_idx": case_idx, "modality": "audio_video",
        "question": question, "input_mode": "native_audio_video",
        "input_media": {"path": handle["video_url"], "duration_s": handle["duration"],
                        "has_audio": handle["has_audio"]},
        "turns": [], "timestamp": datetime.now().isoformat(),
    }
    return run_agent_loop(
        runner=runner, gemini_contents=contents,
        image_paths_dict={"video_1": handle}, trajectory=trajectory,
        output_dir=output_dir, intermediate_dir=os.path.join(output_dir, "intermediate"),
        case_id=case_id, case_idx=case_idx, filename_prefix="native_av",
        runner_supports_video=True, inference_cfg=inference_cfg,
        system_instruction=system_instruction or get_system_prompt("audio_video"),
        modality="audio_video", search_backend=search_backend,
    )

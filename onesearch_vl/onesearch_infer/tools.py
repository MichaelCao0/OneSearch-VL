"""Agent tool definitions, parsing helpers and the dispatcher."""

from __future__ import annotations

import io
import ipaddress
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image

from . import config
from . import cos_upload
from . import image_io
from . import search
from . import video_io
from .prompts import get_tool_specs, tool_names, retrieval_disabled, is_retrieval_tool
from .image_engines import ImageEnhancementEngine, ImageToolEngine


logger = logging.getLogger(__name__)

# Modalities that bootstrap ``img_*`` only (no ``video_1`` / frame grid).
IMAGE_MODALITIES = frozenset({"single_image", "multi_image"})
VIDEO_ONLY_TOOL_NAMES = frozenset({"select_timespan", "select_frame"})


# ---------------------------------------------------------------------------
# JSON tool schema
# ---------------------------------------------------------------------------


def get_tools_definition(modality: str = "video") -> str:
    """Return exactly the tool definitions in the shared modality prompt."""
    return json.dumps(get_tool_specs(modality), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_SEED_FUNCTION_CALL_RE = re.compile(
    r"<\|FunctionCallBegin\|>\s*(\[.*?\]|\{.*?\})\s*<\|FunctionCallEnd\|>",
    re.DOTALL,
)
_FALLBACK_TAGS: Dict[str, Tuple[re.Pattern, str]] = {
    "web_search": (
        re.compile(r"<web_search>\s*(\{.*?\})\s*</web_search>", re.DOTALL),
        "web_search",
    ),
    "image_search": (
        re.compile(
            r"<(?:image_search|local_image_search|lens_scan)>\s*(\{.*?\})"
            r"\s*</(?:image_search|local_image_search|lens_scan)>",
            re.DOTALL,
        ),
        "image_search",
    ),
    "text_search": (
        re.compile(
            r"<(?:text_search|local_search)>\s*(\{.*?\})"
            r"\s*</(?:text_search|local_search)>",
            re.DOTALL,
        ),
        "text_search",
    ),
    "crop": (re.compile(r"<crop>\s*(\{.*?\})\s*</crop>", re.DOTALL), "crop"),
    "layout_parsing": (
        re.compile(r"<(?:layout_parsing|ocr)>\s*(\{.*?\})\s*</(?:layout_parsing|ocr)>", re.DOTALL),
        "layout_parsing",
    ),
    "perspective_correct": (
        re.compile(r"<perspective_correct>\s*(\{.*?\})\s*</perspective_correct>", re.DOTALL),
        "perspective_correct",
    ),
    "super_resolution": (
        re.compile(r"<super_resolution>\s*(\{.*?\})\s*</super_resolution>", re.DOTALL),
        "super_resolution",
    ),
    "sharpen": (
        re.compile(r"<sharpen>\s*(\{.*?\})\s*</sharpen>", re.DOTALL),
        "sharpen",
    ),
    "select_timespan": (
        re.compile(r"<select_timespan>\s*(\{.*?\})\s*</select_timespan>", re.DOTALL),
        "select_timespan",
    ),
    "select_frame": (
        re.compile(r"<select_frame>\s*(\{.*?\})\s*</select_frame>", re.DOTALL),
        "select_frame",
    ),
}


def _normalize_search_aliases(name: str, params: dict) -> dict:
    """Convert ``query``/``lang`` aliases to canonical ``q``/``hl``."""

    if name in {"text_search", "local_search", "web_search"}:
        if "query" in params and "q" not in params:
            params["q"] = params.pop("query")
        if "lang" in params and "hl" not in params:
            params["hl"] = params.pop("lang")
    return params


def _parse_tool_call_payload(payload: Any) -> Optional[str]:
    """Normalise a parsed tool-call dict/list into canonical JSON string."""

    if isinstance(payload, list):
        if not payload:
            return None
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    name = payload.get("name", "")
    if "arguments" in payload or "parameters" in payload:
        params = payload.get("arguments", payload.get("parameters", {})) or {}
    else:
        # Accept valid JSON with arguments flattened next to ``name``, e.g.
        # ``{"name":"text_search","q":"...","hl":"en"}``.
        # The tool name and every argument remain explicit, so this form is
        # unambiguous.  Recover it without guessing missing values.  Keep an
        # explicitly supplied arguments/parameters container authoritative.
        params = {key: value for key, value in payload.items() if key != "name"}
    params = _normalize_search_aliases(name, dict(params))
    if not name:
        return None
    return json.dumps({"name": name, "parameters": params}, ensure_ascii=False)


_MISSING_ARGUMENTS_KEY_RE = re.compile(
    r'^\{\s*"name"\s*:\s*"(?P<name>[^"\\]+)"\s*,\s*'
    r'(?P<params>\{.*\})\s*\}$',
    re.DOTALL,
)
_IMAGE_SEARCH_FREEFORM_RE = re.compile(
    r'^\{\s*"name"\s*:\s*"image_search"\s*,\s*'
    r'(?P<hint>[^{}]+?)\s*\}$',
    re.DOTALL,
)
_AT_IMAGE_KEY_RE = re.compile(r'(?P<prefix>\{|,)\s*@image\s*:')


def _repair_missing_arguments_key(raw: str) -> Optional[str]:
    """Recover two unambiguous tool-call shorthands.

    Accept a missing ``arguments`` key when the call contains either
    ``{"name":"image_search",{"url":"img_1"}}`` or the
    ``{@image:"img_1"}`` handle notation.  Both carry a single explicit tool
    name and parameter object, so canonicalising them is safer than feeding a
    deterministic protocol error back to the model.  Other malformed JSON is
    still rejected.
    """

    stripped = raw.strip()
    # A flattened call can retain the two closing braces from the canonical
    # nested form, e.g. ``{"name":"text_search","q":"..."}}``.  Remove
    # exactly one trailing brace only when the remaining text parses into an
    # explicit named call with at least one flattened argument.  Canonical
    # nested calls also end in ``}}`` but fail this guarded parse after one
    # brace is removed, so they are never rewritten here.
    if stripped.endswith("}}"):
        try:
            flattened = json.loads(stripped[:-1])
        except (json.JSONDecodeError, TypeError):
            flattened = None
        if (
            isinstance(flattened, dict)
            and flattened.get("name")
            and "arguments" not in flattened
            and "parameters" not in flattened
            and len(flattened) > 1
        ):
            return _parse_tool_call_payload(flattened)

    match = _MISSING_ARGUMENTS_KEY_RE.fullmatch(stripped)
    if not match:
        freeform = _IMAGE_SEARCH_FREEFORM_RE.fullmatch(stripped)
        if freeform and freeform.group("hint").strip():
            return _parse_tool_call_payload(
                {"name": "image_search", "arguments": {"image": "img_1"}}
            )
        return None
    params_raw = _AT_IMAGE_KEY_RE.sub(
        lambda item: f'{item.group("prefix")}"image":',
        match.group("params"),
    )
    try:
        params = json.loads(params_raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(params, dict):
        return None
    return _parse_tool_call_payload(
        {"name": match.group("name"), "arguments": params}
    )


def extract_tool_call(text: str) -> Optional[str]:
    """Pull the next tool invocation out of an agent message.

    Returns a JSON string ``{"name": str, "parameters": {...}}`` on success
    and ``None`` when no recognisable tool call is present.
    """

    match = _TOOL_CALL_RE.search(text)
    if match:
        raw = match.group(1).strip()
        try:
            return _parse_tool_call_payload(json.loads(raw))
        except Exception:
            repaired = _repair_missing_arguments_key(raw)
            if repaired is not None:
                return repaired

    seed_match = _SEED_FUNCTION_CALL_RE.search(text)
    if seed_match:
        try:
            return _parse_tool_call_payload(json.loads(seed_match.group(1).strip()))
        except Exception:
            pass

    for pattern, canonical in _FALLBACK_TAGS.values():
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            params = json.loads(raw)
            if not isinstance(params, dict):
                params = {"value": params}
        except Exception:
            params = {"q": raw} if canonical in {"text_search", "web_search"} else {"url": raw}
        params = _normalize_search_aliases(canonical, dict(params))
        return json.dumps(
            {"name": canonical, "parameters": params}, ensure_ascii=False
        )

    return None


def has_response_tag(text: str) -> bool:
    """Return ``True`` once the agent has produced a final response block."""

    return "<response>" in text and ("</response>" in text or "boxed{" in text)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


ToolResult = Tuple[str, Dict[str, str]]


def _is_private_http_url(value: str) -> bool:
    """Whether an HTTP URL is unreachable from external search providers."""
    try:
        host = urllib.parse.urlparse(value).hostname or ""
        if host.lower() == "localhost":
            return True
        ip = ipaddress.ip_address(host)
        return bool(
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        )
    except ValueError:
        return False


def _upload_private_search_image(
    image_url: str,
    case_idx: int,
    turn_num: int,
    filename_prefix: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Materialize a private frame URL and publish it for reverse search."""
    data = image_io.download_image_bytes(image_url)
    if not data:
        return None, f"Failed to download private image URL {image_url!r}."
    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img.load()
    except Exception as exc:
        return None, f"Failed to decode private image for image_search: {exc}"
    url = cos_upload.upload_pil_image(
        pil_img, filename_prefix, case_idx, turn_num, "image_search"
    )
    if not url:
        return None, "Failed to upload private image to COS for image_search."
    return url, None


def _resolve_image_for_search(
    image_paths_dict: dict,
    image_ref: str,
    image_url_param: str,
    case_idx: int,
    turn_num: int,
    filename_prefix: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(resolved_url, error_message)`` for ``image_search``."""

    target_ref: Optional[str] = None
    image_url: Optional[str] = None

    if image_url_param:
        if image_url_param in image_paths_dict:
            target_ref = image_url_param
        elif isinstance(image_url_param, str) and image_url_param.startswith(
            ("http://", "https://")
        ):
            image_url = image_url_param
        else:
            available = [
                k for k in image_paths_dict
                if k.startswith(("img_", "video_", "clip_"))
            ]
            return None, (
                f"image_search url={image_url_param!r} is not a valid http(s) "
                f"URL and not a registered handle. Call select_frame first to "
                f"produce an img_n handle. Available handles: {available}"
            )
    elif image_ref:
        target_ref = image_ref

    if target_ref:
        if target_ref not in image_paths_dict:
            return (
                None,
                f"Image reference '{target_ref}' not found. "
                f"Available: {list(image_paths_dict.keys())}",
            )
        image_data = image_paths_dict[target_ref]
        if isinstance(image_data, str) and image_data.startswith(
            ("http://", "https://")
        ):
            if _is_private_http_url(image_data):
                return _upload_private_search_image(
                    image_data, case_idx, turn_num, filename_prefix
                )
            return image_data, None
        if (
            isinstance(image_data, str)
            and len(image_data) < 500
            and os.path.exists(image_data)
        ):
            with open(image_data, "rb") as fh:
                pil_img = Image.open(io.BytesIO(fh.read()))
            url = cos_upload.upload_pil_image(
                pil_img, filename_prefix, case_idx, turn_num, "image_search"
            )
            return (url, None) if url else (
                None,
                "Failed to upload local image to COS for image_search.",
            )
        if isinstance(image_data, bytes):
            pil_img = Image.open(io.BytesIO(image_data))
            url = cos_upload.upload_pil_image(
                pil_img, filename_prefix, case_idx, turn_num, "image_search"
            )
            return (url, None) if url else (
                None,
                "Failed to upload image bytes to COS for image_search.",
            )
        return None, f"Unsupported image data type for {target_ref!r}"

    if image_url:
        if _is_private_http_url(image_url):
            return _upload_private_search_image(
                image_url, case_idx, turn_num, filename_prefix
            )
        return image_url, None
    return None, "image_search requires either a 'url' or an image reference."


def _apply_image_op(
    operation: Callable[[ImageEnhancementEngine], object],
    image_data,
    return_engine: ImageEnhancementEngine,
) -> Optional[Image.Image]:
    return_engine.load_image(image_data)
    operation(return_engine)
    return return_engine.to_pil()


def _persist_new_image(
    pil_image: Image.Image,
    intermediate_dir: str,
    filename_prefix: str,
    case_id: str,
    case_idx: int,
    turn_num: int,
    tool_name: str,
    image_paths_dict: dict,
) -> Tuple[str, str]:
    """Persist a freshly produced image and return ``(image_id, url_or_path)``."""

    os.makedirs(intermediate_dir, exist_ok=True)
    # Count only image handles (img_n) so video/clip handles in the same
    # dict don't shift the numbering and create duplicates.
    new_id = f"img_{sum(1 for k in image_paths_dict if k.startswith('img_')) + 1}"
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id))
    save_path = os.path.join(
        intermediate_dir,
        f"{filename_prefix}_{safe_case_id}_{case_idx}_turn{turn_num}_{tool_name}_{new_id}.png",
    )
    orig_size = pil_image.size
    pil_image = image_io.ensure_min_image_side(pil_image)
    if pil_image.size != orig_size:
        logger.info(
            "Upscaled %s output %s -> %s for Seed min side=%d",
            tool_name,
            orig_size,
            pil_image.size,
            int(getattr(config, "SEED_MIN_IMAGE_SIDE", 14) or 14),
        )
    pil_image.save(save_path)

    cos_url = cos_upload.upload_pil_image(
        pil_image,
        filename_prefix,
        case_idx,
        turn_num,
        f"{tool_name}_{new_id}",
    )
    return new_id, cos_url or save_path


def execute_tool(
    tool_call_json: str | dict,
    image_paths_dict: dict,
    case_id: str,
    case_idx: int,
    turn_num: int,
    intermediate_dir: str,
    filename_prefix: str = "fvqa_train",
    visual_lookup: Optional[Callable[..., object]] = None,
    modality: str = "video",
) -> ToolResult:
    """Dispatch a parsed tool call. Returns ``(message, new_images)``."""

    try:
        call = (
            json.loads(tool_call_json)
            if isinstance(tool_call_json, str)
            else tool_call_json
        )
    except Exception as exc:
        return f"Tool execution error:\nInvalid tool call payload: {exc}", {}

    name = call.get("name", "")
    if retrieval_disabled() and is_retrieval_tool(name):
        return (
            f"Tool execution error:\nRetrieval disabled (ONESEARCH_DISABLE_RETRIEVAL=1): {name}.",
            {},
        )
    params = call.get("parameters", {}) or {}

    if name == "visit_webpage":
        return (
            "Tool execution error:\nvisit_webpage is not available. "
            "Use text_search to fetch and summarise web evidence.",
            {},
        )

    if modality in IMAGE_MODALITIES and name in VIDEO_ONLY_TOOL_NAMES:
        return (
            f"Tool execution error:\n{name} is not available in {modality} mode. "
            "Operate directly on the provided img_* handles; there is no video_1 "
            "or frame grid.",
            {},
        )

    if name not in tool_names(modality):
        return f"Tool execution error:\n{name!r} is not available in {modality} mode.", {}

    if name in {"text_search", "local_search", "web_search"}:
        query = params.get("q") or params.get("query") or ""
        if not query:
            return "Tool execution error:\n'q' is required for text_search.", {}
        return search.text_search(
            query=query,
            lang=params.get("hl") or params.get("lang") or "en",
            top_k=search.normalize_top_k(params.get("top_k", 5)),
        ), {}

    if name == "select_timespan":
        video_ref = params.get("video") or params.get("clip") or ""
        if not video_ref:
            return (
                "Tool execution error:\n'video' (or 'clip') is required for "
                "select_timespan.",
                {},
            )
        entry = image_paths_dict.get(video_ref)
        if not isinstance(entry, dict) or not entry.get("frame_urls"):
            return (
                f"Tool execution error:\nselect_timespan expects a frame-grid "
                f"handle, got {video_ref!r} = {type(entry).__name__}.",
                {},
            )
        si = params.get("start_index", params.get("start_idx"))
        ei = params.get("end_index", params.get("end_idx"))
        if si is None or ei is None:
            return (
                "Tool execution error:\nselect_timespan requires integer "
                f"start_index / end_index (0 <= start < end <= {entry.get('n_frames')}).",
                {},
            )
        try:
            si_i, ei_i = video_io.parse_grid_index(si), video_io.parse_grid_index(ei)
        except (TypeError, ValueError):
            return (
                "Tool execution error:\nstart_index / end_index must be integers.",
                {},
            )
        clip_id = f"clip_{sum(1 for k in image_paths_dict if k.startswith('clip_')) + 1}"
        sub_grid = video_io.grid_slice(entry, si_i, ei_i)
        if not sub_grid:
            return (
                f"Tool execution error:\nFailed to slice frame range "
                f"[{si_i}, {ei_i}) from {video_ref!r}.",
                {},
            )

        # The 1-fps grid has one playback second per slot. Publish the exact
        # subclip before publishing a successful clip handle.
        compose = os.environ.get("SELECT_TIMESPAN_COMPOSE_SUBCLIP", "1") == "1"
        parent_url = entry.get("video_url") or ""
        warn_suffix = ""
        if not compose:
            return "Tool execution error: select_timespan requires subclip composition.", {}
        elif not parent_url:
            return "Tool execution error: grid video_url is missing.", {}
        if compose and parent_url:
            cache_dir = os.path.join(intermediate_dir, "grid_cache")
            parent_local = video_io.cache_parent_grid_mp4(parent_url, cache_dir)
            if parent_local is None:
                warn_suffix = " [warn: parent mp4 download failed]"
            else:
                safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id))
                subclip_path = os.path.join(
                    intermediate_dir,
                    f"{filename_prefix}_{safe_case_id}_{case_idx}_turn{turn_num}_{clip_id}_subclip.mp4",
                )
                ok = video_io.cut_subclip(parent_local, int(entry.get("url_offset", 0) or 0) + si_i, int(entry.get("url_offset", 0) or 0) + ei_i, subclip_path)
                if not ok:
                    warn_suffix = " [warn: ffmpeg cut failed]"
                else:
                    sub_url = cos_upload.upload_local_mp4(
                        subclip_path,
                        filename_prefix,
                        case_idx,
                        turn_num,
                        f"select_timespan_{clip_id}",
                    )
                    if sub_url:
                        sub_grid["video_url"] = sub_url
                        sub_grid["subclip_path"] = subclip_path
                        sub_grid["url_offset"] = 0
                    else:
                        warn_suffix = " [warn: sub-clip upload failed]"

        if warn_suffix:
            return "Tool execution error: requested subclip could not be composed." + warn_suffix, {}

        msg = (
            f"select_timespan succeeded. New clip handle: {clip_id} "
            f"(frames=[{si_i}, {ei_i}) of {video_ref!r}; this clip is "
            f"now shown to you as {sub_grid['n_frames']} frames indexed "
            f"0..{sub_grid['n_frames'] - 1}).{warn_suffix}"
        )
        return msg, {clip_id: sub_grid}

    if name == "select_frame":
        video_ref = params.get("video") or params.get("clip") or ""
        if not video_ref:
            return (
                "Tool execution error:\n'video' (or 'clip') is required for "
                "select_frame.",
                {},
            )
        entry = image_paths_dict.get(video_ref)
        if not isinstance(entry, dict) or not entry.get("frame_urls"):
            return (
                f"Tool execution error:\nselect_frame expects a frame-grid "
                f"handle, got {video_ref!r} = {type(entry).__name__}.",
                {},
            )
        fi = params.get("frame_index", params.get("idx"))
        if fi is None:
            return (
                "Tool execution error:\nselect_frame requires integer "
                f"frame_index (0 <= frame_index < {entry.get('n_frames')}).",
                {},
            )
        try:
            fi_i = video_io.parse_grid_index(fi)
        except (TypeError, ValueError):
            return (
                "Tool execution error:\nselect_frame requires integer "
                f"frame_index (0 <= frame_index < {entry.get('n_frames')}).",
                {},
            )
        frame_url = video_io.grid_frame_path(entry, fi_i)
        if not frame_url:
            return (
                f"Tool execution error:\nframe_index {fi_i} not available "
                f"for {video_ref!r}.",
                {},
            )
        new_id = f"img_{sum(1 for k in image_paths_dict if k.startswith('img_')) + 1}"
        n_frames = int(entry.get("n_frames", 0))
        idx_label = f"frame_index={fi_i}/{n_frames - 1}"
        msg = (
            f"select_frame succeeded. New image handle: {new_id} "
            f"({idx_label} from {video_ref!r})."
        )
        return msg, {new_id: frame_url}

    if name in {"image_search", "local_image_search", "lens_scan"}:
        url, err = _resolve_image_for_search(
            image_paths_dict,
            image_ref=params.get("image", ""),
            image_url_param=params.get("url", ""),
            case_idx=case_idx,
            turn_num=turn_num,
            filename_prefix=filename_prefix,
        )
        if err:
            return f"Tool execution error:\n{err}", {}
        return search.image_search(image_url=url, visual_lookup=visual_lookup), {}

    if name == "crop":
        image_ref = params.get("image", "")
        if not image_ref:
            return "Tool execution error:\n'image' is required for crop.", {}

        # ``video_*``/``clip_*`` entries are frame-grid metadata dictionaries,
        # not image payloads.  A model can occasionally pass one directly to
        # crop instead of selecting a frame first.  Treat that as an ordinary
        # recoverable tool error so the agent can self-correct; otherwise the
        # dict reaches PIL and aborts the whole sample with TypeError.
        image_entry = image_paths_dict.get(image_ref)
        if isinstance(image_entry, dict):
            return (
                "Tool execution error:\n"
                f"crop expects an img_* image handle, got frame-grid handle "
                f"{image_ref!r}. Call select_frame first, then crop the returned "
                "img_* handle.",
                {},
            )

        image_data, _ = image_io.ensure_image_local(
            image_ref,
            image_paths_dict,
            intermediate_dir,
            case_idx,
            turn_num,
            tool_name="crop",
            case_id=case_id,
            filename_prefix=filename_prefix,
        )
        if image_data is None:
            return f"Tool execution error:\nFailed to load image {image_ref!r}.", {}
        engine = ImageToolEngine()
        engine.load_image(image_data)
        if engine.current_image is None:
            return f"Tool execution error:\nCould not decode {image_ref!r}.", {}
        W, H = engine.current_image.size

        bbox = params.get("bbox_1000") or params.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return (
                    "Tool execution error:\nbbox_1000 must be a 4-element list "
                    "[x1, y1, x2, y2] in [0, 1000].",
                    {},
                )
            try:
                x1n, y1n, x2n, y2n = [int(float(v)) for v in bbox]
            except (TypeError, ValueError):
                return "Tool execution error:\nbbox_1000 entries must be integers.", {}
            # Clamp normalized bounding-box coordinates to [0, 1000].
            x1n = max(0, min(1000, x1n))
            y1n = max(0, min(1000, y1n))
            x2n = max(0, min(1000, x2n))
            y2n = max(0, min(1000, y2n))
            if x2n <= x1n or y2n <= y1n:
                return (
                    "Tool execution error:\nbbox_1000 must satisfy x1<x2 and y1<y2.",
                    {},
                )
            x = int(round(x1n / 1000.0 * W))
            y = int(round(y1n / 1000.0 * H))
            width = int(round((x2n - x1n) / 1000.0 * W))
            height = int(round((y2n - y1n) / 1000.0 * H))
        else:
            try:
                x, y = int(params.get("x", 0)), int(params.get("y", 0))
                width = int(params.get("width", 0))
                height = int(params.get("height", 0))
            except (TypeError, ValueError):
                return "Tool execution error:\nx/y/width/height must be integers.", {}
            # Pixel mode is the legacy path; clamp + reject obvious nonsense to
            # avoid silently cropping empty regions when Seed hallucinated coords
            # outside the actual frame size.
            x = max(0, min(W, x))
            y = max(0, min(H, y))
            width = max(0, min(W - x, width))
            height = max(0, min(H - y, height))

        if width <= 0 or height <= 0:
            return (
                "Tool execution error:\nresulting crop has zero area "
                f"(image is {W}x{H}). Pass a tighter bbox_1000 within [0,1000].",
                {},
            )

        cropped = engine.crop(x, y, width, height)
        new_id, location = _persist_new_image(
            cropped,
            intermediate_dir,
            filename_prefix,
            case_id,
            case_idx,
            turn_num,
            "crop",
            image_paths_dict,
        )
        msg = (
            f"Image cropped successfully. New image ID: {new_id}."
        )
        return msg, {new_id: location}

    if name == "layout_parsing":
        file_path = params.get("file_path", "")
        image_ref = params.get("image", "")
        if not file_path and image_ref:
            image_data, local_path = image_io.ensure_image_local(
                image_ref,
                image_paths_dict,
                intermediate_dir,
                case_idx,
                turn_num,
                tool_name="layout_parsing",
                case_id=case_id,
                filename_prefix=filename_prefix,
            )
            if local_path:
                file_path = local_path
            elif isinstance(image_data, bytes):
                os.makedirs(intermediate_dir, exist_ok=True)
                file_path = os.path.join(
                    intermediate_dir,
                    f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_layout_source.png",
                )
                with open(file_path, "wb") as fh:
                    fh.write(image_data)
                image_paths_dict[image_ref] = file_path
        if not file_path:
            return (
                "Tool execution error:\nlayout_parsing needs 'file_path' "
                "or a resolvable 'image' reference.",
                {},
            )

        result = search.layout_parsing(
            file_path,
            use_chart_recognition=bool(params.get("use_chart_recognition", False)),
            use_doc_orientation_classify=bool(
                params.get("use_doc_orientation_classify", False)
            ),
            use_doc_unwarping=bool(params.get("use_doc_unwarping", False)),
        )
        if result.get("error"):
            return f"Tool execution error:\n{result['error']}", {}
        body = result.get("formatted_text", "") or "(no text detected)"
        report = (
            "Tool execution result:\n"
            "Layout Parsing SUCCESS.\n\n"
            "ALL RECOGNIZED TEXT:\n"
            f"{body}"
        )
        return report, {}

    if name in {"perspective_correct", "super_resolution", "sharpen"}:
        image_ref = params.get("image", "")
        if not image_ref:
            return f"Tool execution error:\n'image' is required for {name}.", {}
        image_data, _ = image_io.ensure_image_local(
            image_ref,
            image_paths_dict,
            intermediate_dir,
            case_idx,
            turn_num,
            tool_name=name,
            case_id=case_id,
            filename_prefix=filename_prefix,
        )
        if image_data is None:
            return f"Tool execution error:\nFailed to load image {image_ref!r}.", {}
        engine = ImageEnhancementEngine()
        if not ImageEnhancementEngine.available():
            return (
                "Tool execution error:\nOpenCV is required for image "
                "enhancement tools. Install with: pip install opencv-python.",
                {},
            )
        if name == "perspective_correct":
            op = lambda e: e.auto_correct_perspective()
        elif name == "super_resolution":
            scale = int(params.get("scale", 4))
            model_path = params.get("model_path") or os.environ.get(
                "SR_MODEL_PATH", "EDSR_x4.pb"
            )
            op = lambda e: e.apply_super_resolution(model_path=model_path, scale=scale)
        else:
            amount = float(params.get("amount", 1.5))
            op = lambda e: e.enhance_sharpness(amount=amount)
        pil_result = _apply_image_op(op, image_data, engine)
        if pil_result is None:
            return f"Tool execution error:\n{name} produced no image.", {}
        new_id, location = _persist_new_image(
            pil_result,
            intermediate_dir,
            filename_prefix,
            case_id,
            case_idx,
            turn_num,
            name,
            image_paths_dict,
        )
        msg = (
            f"{name} succeeded. New image ID: {new_id}."
        )
        return msg, {new_id: location}

    return f"Tool execution error:\nUnknown tool: {name!r}", {}

"""
Visual Investigation Tools – ported from qwen_search_fvqa.py.

Provides: crop, layout_parsing, text_search, image_search,
          web_search, perspective_correct, super_resolution, sharpen.

All tools follow the async DeepResearchTool interface.
"""

import asyncio
import base64
import hashlib
import inspect
import io
import json
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image

from vision_deepresearch_async_workflow.tools.shared import (
    ToolCallTimeout,
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_tool_event,
    next_handle_id,
    run_blocking,
    run_blocking_bounded,
    run_with_retries_async,
    set_cache_async,
)

# ---------------------------------------------------------------------------
# Optional dependency: OpenCV (for perspective_correct, super_resolution, sharpen)
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency: COS upload
# ---------------------------------------------------------------------------
UPLOAD_AVAILABLE = False
upload_cos = None

# Search for an optional ``upload.py`` module (COS uploader). By default we look
# at the project root AND the onesearch_vl inference dir (which ships the portable
# asset publisher). The RL repo root has no upload.py, so without the
# onesearch_vl default UPLOAD_AVAILABLE stays False and image_search silently
# fails for every frame ("COS upload available: False"). Deriving it from
# __file__ makes the fix work without requiring COS_UPLOAD_PATHS in .env (which
# is gitignored). Additional roots can still be injected via COS_UPLOAD_PATHS.
_here = os.path.dirname(os.path.abspath(__file__))
_upload_paths = [
    os.path.join(_here, "..", "..", "..", ".."),
    # tools -> vision_deepresearch_async_workflow -> rllm -> RL -> OneSearch-VL,
    # then into onesearch_vl/ where the inference pipeline keeps upload.py.
    os.path.join(_here, "..", "..", "..", "..", "onesearch_vl"),
]
_extra_paths = os.getenv("COS_UPLOAD_PATHS", "")
if _extra_paths:
    _upload_paths.extend(p for p in _extra_paths.split(":") if p)

for _p in _upload_paths:
    _ufile = os.path.join(_p, "upload.py")
    if not os.path.exists(_ufile):
        continue
    try:
        import importlib.util

        _spec = importlib.util.spec_from_file_location("upload", _ufile)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            upload_cos = _mod.upload_cos
            if callable(upload_cos):
                UPLOAD_AVAILABLE = True
                print(f"[visual_tools] COS upload loaded from {_ufile}")
                break
    except Exception as _e:
        print(f"[visual_tools] Failed to load upload.py from {_ufile}: {_e}")
        continue

if not UPLOAD_AVAILABLE:
    print("[visual_tools] WARNING: COS upload not available. image_search will not work for local images.")
    def upload_cos(*args, **kwargs):
        return None, None


from .onesearch_engine import search as _search

def _onesearch_timeout(env_name: str, default: float) -> float:
    try:
        return float(os.getenv(env_name, str(default)))
    except (TypeError, ValueError):
        return default

_LOCAL_VISUAL_SEM = None

def _get_local_visual_sem() -> asyncio.Semaphore:
    global _LOCAL_VISUAL_SEM
    if _LOCAL_VISUAL_SEM is None:
        n = int(os.getenv("LOCAL_VISUAL_MAX_CONCURRENCY", "16"))
        _LOCAL_VISUAL_SEM = asyncio.Semaphore(n if n > 0 else 2**31)
    return _LOCAL_VISUAL_SEM

async def _run_local_visual(
    tool: DeepResearchTool,
    func,
    *,
    label: str,
    timing: dict[str, float] | None,
):
    timeout = _onesearch_timeout("LOCAL_VISUAL_TIMEOUT", 60.0)
    phase_timing: dict[str, float] = {}
    try:
        return await asyncio.wait_for(
            run_blocking_bounded(
                func,
                executor=tool.executor,
                semaphore=_get_local_visual_sem(),
                timeout=timeout,
                queue_timeout=timeout,
                label=label,
                timing=phase_timing,
            ),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, ToolCallTimeout):
        return (
            "Tool execution error:\n"
            f"{label} exceeded its total {timeout:.0f}s deadline."
        )
    finally:
        if isinstance(timing, dict):
            for name, seconds in phase_timing.items():
                metric = f"local_{name}"
                timing[metric] = timing.get(metric, 0.0) + seconds

def _upload_pil_to_cos(
    pil_img: Image.Image,
    tool_name: str,
    userid: str = os.getenv("COS_USERID", "onesearch-vl"),
) -> Optional[str]:
    if not UPLOAD_AVAILABLE:
        print(f"[visual_tools] _upload_pil_to_cos: COS upload not available (UPLOAD_AVAILABLE=False)")
        return None
    tmp_path = None
    try:
        tool_tmp_dir = os.getenv("RLLM_TOOL_TMP_DIR") or None
        if tool_tmp_dir:
            os.makedirs(tool_tmp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False, dir=tool_tmp_dir
        ) as tmp:
            pil_img.save(tmp.name, format="PNG")
            tmp_path = tmp.name
        date_str = datetime.now().strftime("%Y-%m-%d")
        ts = int(time.time() * 1000)
        filename = f"vdr_{tool_name}_{ts}.png"
        cos_key, cos_url = upload_cos(
            tmp_path, filename, date_str,
            f"vdr_{date_str}", userid, use_direct_url=True,
        )
        if cos_url:
            print(f"[visual_tools] Uploaded to COS: {cos_url}")
        else:
            print(f"[visual_tools] _upload_pil_to_cos: upload_cos returned None for {filename}")
        return cos_url or None
    except Exception as exc:
        print(f"[visual_tools] _upload_pil_to_cos failed: {type(exc).__name__}: {exc}")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

def _download_image_bytes(url: str, timeout: int = 60, retries: int = 3) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=(timeout, timeout), stream=True)
            resp.raise_for_status()
            return resp.content
        except Exception:
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
    return None

def _resolve_image_ref(
    image_ref: str,
    image_paths: dict,
    intermediate_dir: str,
    tool_name: str,
) -> Optional[str]:
    """Return a local file path for *image_ref* (downloading if needed).

    Handles: local path str, URL str, bytes, PIL.Image, HuggingFace dict
    ({"bytes": ..., "path": ...}), and base64 data URIs.
    """
    if image_ref not in image_paths:
        return None
    data = image_paths[image_ref]

    # Already a local file path
    if isinstance(data, str) and len(data) < 500 and os.path.exists(data):
        return data

    # URL → download
    if isinstance(data, str) and data.startswith(("http://", "https://")):
        img_bytes = _download_image_bytes(data)
        if img_bytes is None:
            return None
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        with open(local, "wb") as f:
            f.write(img_bytes)
        image_paths[image_ref] = local
        return local

    # Base64 data URI → decode
    if isinstance(data, str) and data.startswith("data:image/"):
        try:
            _, encoded = data.split(",", 1)
            raw = base64.b64decode(encoded)
            os.makedirs(intermediate_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
            Image.open(io.BytesIO(raw)).save(local)
            image_paths[image_ref] = local
            return local
        except Exception:
            pass

    # Raw bytes
    if isinstance(data, bytes):
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        with open(local, "wb") as f:
            f.write(data)
        image_paths[image_ref] = local
        return local

    # PIL Image
    if isinstance(data, Image.Image):
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        data.save(local)
        image_paths[image_ref] = local
        return local

    # HuggingFace dict format: {"bytes": b"...", "path": "..."}
    if isinstance(data, dict):
        pil = None
        if "bytes" in data and data["bytes"] is not None:
            try:
                pil = Image.open(io.BytesIO(data["bytes"])).convert("RGB")
            except Exception:
                pass
        if pil is None and "path" in data and isinstance(data["path"], str):
            try:
                pil = Image.open(data["path"]).convert("RGB")
            except Exception:
                pass
        if pil is None:
            d_str = data.get("data") or data.get("url") or ""
            if isinstance(d_str, str) and d_str.startswith("data:image/"):
                try:
                    _, encoded = d_str.split(",", 1)
                    pil = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                except Exception:
                    pass
        if pil is not None:
            os.makedirs(intermediate_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
            pil.save(local)
            image_paths[image_ref] = local
            return local

    print(f"[visual_tools] _resolve_image_ref: unhandled data type {type(data).__name__} for {image_ref}")
    return None

def _ensure_image_url(
    image_ref: str,
    image_paths: dict,
    intermediate_dir: str,
    tool_name: str,
) -> Optional[str]:
    """
    Resolve *image_ref* to a public URL suitable for APIs that require one.

    Priority:
      1. Already a URL string in image_paths → return directly
      2. Local file / PIL / bytes → upload to COS → return URL
      3. image_ref itself looks like a URL → return as-is

    On success the URL is cached back into image_paths[image_ref] so
    subsequent calls skip the upload.
    """
    if image_ref in image_paths:
        data = image_paths[image_ref]
        # Already a URL
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            return data

        # Resolve to a local file first
        local = _resolve_image_ref(image_ref, image_paths, intermediate_dir, tool_name)
        if local:
            try:
                pil = Image.open(local)
                cos_url = _upload_pil_to_cos(pil, tool_name)
                if cos_url:
                    image_paths[image_ref] = cos_url
                    return cos_url
            except Exception as exc:
                print(f"[visual_tools] _ensure_image_url: failed to upload {image_ref}: {exc}")
        return None

    if isinstance(image_ref, str) and image_ref.startswith(("http://", "https://")):
        return image_ref
    return None

class ImageEnhancementEngine:
    """OpenCV-based image enhancement."""

    def __init__(self):
        self.current_image = None

    def load_from_path(self, path: str):
        self.current_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        return self

    def load_from_pil(self, pil_img: Image.Image):
        self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return self

    # -- perspective correct --
    def auto_correct_perspective(self):
        if self.current_image is None:
            return None
        if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 4:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screen_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is None:
            return self.current_image
        pts = screen_cnt.reshape(4, 2).astype("float32")
        s = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        maxW = max(int(wA), int(wB))
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        maxH = max(int(hA), int(hB))
        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        self.current_image = cv2.warpPerspective(self.current_image, M, (maxW, maxH))
        return self.current_image

    # -- super resolution --
    def apply_super_resolution(self, model_path: str = "EDSR_x4.pb", scale: int = 4):
        if self.current_image is None:
            return None
        if not os.path.exists(model_path):
            return self.current_image
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel("edsr", scale)
            self.current_image = sr.upsample(self.current_image)
        except Exception:
            pass
        return self.current_image

    # -- sharpen --
    def enhance_sharpness(self, amount: float = 1.5):
        if self.current_image is None:
            return None
        blurred = cv2.GaussianBlur(self.current_image, (0, 0), 3)
        self.current_image = cv2.addWeighted(self.current_image, 1.0 + amount, blurred, -amount, 0)
        return self.current_image

    def to_pil(self) -> Optional[Image.Image]:
        if self.current_image is None:
            return None
        return Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))

class CropTool(DeepResearchTool):
    """Crop a specific region from an image.

    Accepts EITHER normalized ``bbox_1000`` coordinates ``[x1, y1, x2, y2]`` in
    ``[0, 1000]`` (preferred — no need to know the frame's pixel size; the tool
    converts to pixels) OR legacy pixel ``x/y/width/height``. Mirrors the
    onesearch_vl ``crop`` tool (onesearch_infer/tools.py) so video-chain crops
    behave identically in RL rollouts and inference.
    """

    def __init__(self):
        super().__init__(
            name="crop",
            # Runtime schema; the actor reads the shared complete modality prompt.
            description=(
                "Crop a rectangular region from a frame before image_search. "
                "Use this for logos, faces, signs, jersey numbers, scoreboard "
                "regions, and other small visual cues.\n\n"
                "PREFER bbox_1000 (normalized coordinates) over pixel x/y/w/h: "
                "you do not need to know the frame's actual pixel size — the "
                "runner converts [0,1000] coordinates to pixels for you."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference (img_n)."},
                    "bbox_1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": (
                            "Normalized box [x1, y1, x2, y2] in [0, 1000]. "
                            "x1<x2, y1<y2. 0 = left/top edge of the frame, "
                            "1000 = right/bottom edge."
                        ),
                    },
                    "x": {"type": "integer", "description": "(Legacy, prefer bbox_1000) crop top-left pixel X."},
                    "y": {"type": "integer", "description": "(Legacy, prefer bbox_1000) crop top-left pixel Y."},
                    "width": {"type": "integer", "description": "(Legacy, prefer bbox_1000) crop pixel width."},
                    "height": {"type": "integer", "description": "(Legacy, prefer bbox_1000) crop pixel height."},
                },
                "required": ["image"],
            },
        )

    async def call(
        self,
        image: str = "",
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
        bbox_1000=None,
        bbox=None,
        **ctx,
    ) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get(
            "intermediate_dir",
            "/tmp/vdr_tools",
        )
        if not image or image not in image_paths:
            return (
                f"Error: Image reference '{image}' not found. "
                f"Available: {list(image_paths.keys())}"
            )

        def execute() -> str:
            local = _resolve_image_ref(
                image,
                image_paths,
                intermediate_dir,
                "crop",
            )
            if not local:
                return f"Error: Failed to resolve image for {image}"
            try:
                with Image.open(local) as source:
                    pil_img = source.convert("RGB")
                image_width, image_height = pil_img.size
                crop_x, crop_y = x, y
                crop_width, crop_height = width, height
                box = bbox_1000 if bbox_1000 is not None else bbox
                if box is not None:
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        return (
                            "Error: bbox_1000 must be a 4-element list "
                            "[x1, y1, x2, y2] in [0, 1000]"
                        )
                    try:
                        x1n, y1n, x2n, y2n = [
                            int(float(value)) for value in box
                        ]
                    except (TypeError, ValueError):
                        return "Error: bbox_1000 entries must be integers"
                    x1n = max(0, min(1000, x1n))
                    y1n = max(0, min(1000, y1n))
                    x2n = max(0, min(1000, x2n))
                    y2n = max(0, min(1000, y2n))
                    if x2n <= x1n or y2n <= y1n:
                        return (
                            "Error: bbox_1000 must satisfy x1<x2 and y1<y2"
                        )
                    crop_x = int(round(x1n / 1000.0 * image_width))
                    crop_y = int(round(y1n / 1000.0 * image_height))
                    crop_width = int(
                        round((x2n - x1n) / 1000.0 * image_width)
                    )
                    crop_height = int(
                        round((y2n - y1n) / 1000.0 * image_height)
                    )
                if crop_width <= 0 or crop_height <= 0:
                    return (
                        "Error: provide bbox_1000 [x1,y1,x2,y2] in "
                        "[0,1000], or positive pixel width and height"
                    )

                cropped = pil_img.crop(
                    (
                        crop_x,
                        crop_y,
                        crop_x + crop_width,
                        crop_y + crop_height,
                    )
                )
                new_id = next_handle_id(image_paths, "img_")
                cos_url = _upload_pil_to_cos(cropped, "crop")
                if cos_url:
                    image_paths[new_id] = cos_url
                else:
                    os.makedirs(intermediate_dir, exist_ok=True)
                    save = os.path.join(
                        intermediate_dir,
                        f"vdr_crop_{int(time.time() * 1000)}.png",
                    )
                    cropped.save(save)
                    image_paths[new_id] = save
                return (
                    "Image cropped successfully. "
                    f"New image ID: {new_id}."
                )
            except Exception as exc:
                return f"Error executing crop: {exc}"

        return await _run_local_visual(
            self,
            execute,
            label="crop",
            timing=ctx.get("_tool_timing"),
        )

def _enhance_image_sync(
    image: str,
    image_paths: dict,
    intermediate_dir: str,
    *,
    tool_name: str,
    file_prefix: str,
    transform,
) -> str:
    local = _resolve_image_ref(
        image,
        image_paths,
        intermediate_dir,
        tool_name,
    )
    if not local:
        return f"Error: Failed to resolve image for {image}"
    try:
        engine = ImageEnhancementEngine()
        engine.load_from_path(local)
        transform(engine)
        pil = engine.to_pil()
        if not pil:
            return f"Error: {tool_name} failed"
        new_id = next_handle_id(image_paths, "img_")
        cos_url = _upload_pil_to_cos(pil, tool_name)
        if cos_url:
            image_paths[new_id] = cos_url
        else:
            os.makedirs(intermediate_dir, exist_ok=True)
            save = os.path.join(
                intermediate_dir,
                f"{file_prefix}_{int(time.time() * 1000)}.png",
            )
            pil.save(save)
            image_paths[new_id] = save
        return f"{tool_name} succeeded. New image ID: {new_id}."
    except Exception as exc:
        return f"Error: {exc}"

class PerspectiveCorrectTool(DeepResearchTool):
    """Auto perspective correction via OpenCV."""

    def __init__(self):
        super().__init__(
            name="perspective_correct",
            description=(
                "Auto-correct perspective distortion in a frame so a "
                "skewed sign / document becomes rectilinear. Chain "
                "before layout_parsing when text is photographed at "
                "an angle."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get(
            "intermediate_dir",
            "/tmp/vdr_tools",
        )
        if not image or image not in image_paths:
            return (
                f"Error: Image '{image}' not found. "
                f"Available: {list(image_paths.keys())}"
            )
        return await _run_local_visual(
            self,
            lambda: _enhance_image_sync(
                image,
                image_paths,
                intermediate_dir,
                tool_name="perspective_correct",
                file_prefix="vdr_persp",
                transform=lambda engine: engine.auto_correct_perspective(),
            ),
            label="perspective_correct",
            timing=ctx.get("_tool_timing"),
        )

class SuperResolutionTool(DeepResearchTool):
    """Super resolution via OpenCV DNN."""

    def __init__(self):
        super().__init__(
            name="super_resolution",
            description=(
                "Up-scale a low-resolution / pixelated image so small "
                "text or distant logos become legible. Use before "
                "layout_parsing or image_search when the source crop "
                "is < ~300px wide."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "scale": {"type": "integer", "default": 4},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", scale: int = 4, **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get(
            "intermediate_dir",
            "/tmp/vdr_tools",
        )
        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found."
        return await _run_local_visual(
            self,
            lambda: _enhance_image_sync(
                image,
                image_paths,
                intermediate_dir,
                tool_name="super_resolution",
                file_prefix="vdr_sr",
                transform=lambda engine: engine.apply_super_resolution(
                    scale=scale
                ),
            ),
            label="super_resolution",
            timing=ctx.get("_tool_timing"),
        )

class SharpenTool(DeepResearchTool):
    """Sharpen / unsharp-mask via OpenCV."""

    def __init__(self):
        super().__init__(
            name="sharpen",
            description=(
                "Sharpen an image to reduce motion blur / soft focus. "
                "Use before layout_parsing when the text is legible "
                "in shape but the strokes are soft."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "amount": {"type": "number", "default": 1.5},
                },
                "required": ["image"],
            },
        )

    async def call(
        self,
        image: str = "",
        amount: float = 1.5,
        **ctx,
    ) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get(
            "intermediate_dir",
            "/tmp/vdr_tools",
        )
        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found."
        return await _run_local_visual(
            self,
            lambda: _enhance_image_sync(
                image,
                image_paths,
                intermediate_dir,
                tool_name="sharpen",
                file_prefix="vdr_sharp",
                transform=lambda engine: engine.enhance_sharpness(
                    amount=amount
                ),
            ),
            label="sharpen",
            timing=ctx.get("_tool_timing"),
        )

async def _run_search(tool, operation, label, ctx):
    timeout = _onesearch_timeout(label.upper() + "_BACKEND_TIMEOUT", 120.0)
    timing = {}
    control = _search.SearchControl(deadline=time.monotonic() + timeout, timing=timing, kind=label)
    # Keep per-instance gates on the loop that owns this tool.
    if not hasattr(tool, "_api_semaphore"):
        limit = max(1, int(os.getenv(label.upper() + "_MAX_CONCURRENCY", "16")))
        tool._api_semaphore = asyncio.Semaphore(limit)
    try:
        return await run_blocking_bounded(
            lambda: operation(control), executor=tool.executor,
            semaphore=tool._api_semaphore, timeout=timeout, queue_timeout=timeout,
            cancel=control.cancel_event.set, label=label, timing=timing,
        )
    except (asyncio.TimeoutError, ToolCallTimeout, _search.SearchCancelled):
        return f"Tool execution error:\n{label} deadline exceeded."
    except _search.ProviderError as exc:
        return f"Tool execution error:\n{label} failed: {exc}"
    finally:
        target = ctx.get("_tool_timing")
        if isinstance(target, dict):
            for key, value in timing.items():
                target[key] = target.get(key, 0.0) + value


class TextSearchTool(DeepResearchTool):
    def __init__(self):
        super().__init__(
            name="text_search",
            description=(
                "Run a Bing/Serper web search and return Qwen-summarised "
                "snippets for the top-k hits. Use this to verify "
                "named entities and fetch facts not visible in pixels."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "query": {"type": "string", "description": "Alias for 'q'."},
                    "hl": {"type": "string", "default": "en"},
                    "lang": {
                        "type": "string",
                        "description": "Alias for 'hl'.",
                        "default": "en",
                    },
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": [],
            },
        )

    async def call(self, q="", query="", hl="", top_k=5, lang="", **ctx):
        return await _run_search(self, lambda control: _search.text_search(
            q or query, lang=hl or lang or "en", top_k=top_k, control=control), "text_search", ctx)


class WebSearchTool(DeepResearchTool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Perform a web search and return snippet results.",
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query keywords"},
                    "hl": {"type": "string", "default": "en"},
                },
                "required": ["q"],
            },
        )

    async def call(self, q="", hl="en", **ctx):
        if not q:
            return "Tool execution error:\nWeb search requires 'q'."
        return await _run_search(self, lambda control: _search.web_search(
            q, lang=hl, top_k=10, control=control), "text_search", ctx)


class ImageSearchTool(DeepResearchTool):
    def __init__(self):
        super().__init__(
            name="image_search",
            description=(
                "Reverse-image search for a frame (or cropped region) "
                "to surface candidate sources / page titles. ALWAYS "
                "follow this up with text_search to "
                "verify the candidate identification."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Image reference (img_n) or a direct URL.",
                    },
                },
                "required": ["url"],
            },
        )

    async def call(self, url="", **ctx):
        if not url:
            return "Tool execution error:\nImage search requires 'url'."
        def execute(control):
            resolved = _ensure_image_url(url, ctx.get("image_paths", {}),
                                         ctx.get("intermediate_dir", "/tmp/vdr_tools"), "image_search")
            control.check()
            if not resolved:
                return "Tool execution error:\nCannot resolve or publish image; configure ASSET_DIR and ASSET_BASE_URL."
            return _search.image_search(resolved, visual_lookup=ctx.get("visual_lookup"), control=control)
        return await _run_search(self, execute, "image_search", ctx)


class LayoutParsingTool(DeepResearchTool):
    def __init__(self):
        super().__init__(
            name="layout_parsing",
            description=(
                "OCR a frame or cropped region using Doubao-Seed-2.0-pro. "
                "Returns the verbatim visible text plus a candidate entity "
                "name. Use for logos, signs, banners, title cards, "
                "scoreboards, jerseys, and chyrons."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": (
                            "Image reference (img_n). The crop tool's output "
                            "is the ideal input here."
                        ),
                    }
                },
                "required": ["image"],
            },
        )

    async def call(self, image="", file_path="", use_chart_recognition=False,
                   use_doc_orientation_classify=False, use_doc_unwarping=False, **ctx):
        def execute(control):
            local = file_path or _resolve_image_ref(image, ctx.get("image_paths", {}),
                        ctx.get("intermediate_dir", "/tmp/vdr_tools"), "layout_parsing")
            control.check()
            if not local:
                return "Tool execution error:\nlayout_parsing needs 'file_path' or a resolvable 'image' reference."
            result = _search.layout_parsing(local, use_chart_recognition,
                use_doc_orientation_classify, use_doc_unwarping, control=control)
            if result.get("error"):
                return "Tool execution error:\n" + result["error"]
            return "Tool execution result:\nLayout Parsing SUCCESS.\n\nALL RECOGNIZED TEXT:\n" + (result["formatted_text"] or "(no text detected)")
        return await _run_search(self, execute, "layout_parsing", ctx)



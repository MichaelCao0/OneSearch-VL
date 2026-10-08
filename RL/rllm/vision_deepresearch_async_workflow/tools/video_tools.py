"""
Video Deep Research tools – ported from onesearch_vl/onesearch_infer.

Provides the two entry-point tools of the video tool chain:

    select_timespan -> select_frame -> crop -> image_search -> text_search

Both operate against a *frame-grid handle* stored in the shared
``image_paths`` dict under a ``video_n`` / ``clip_n`` key. A grid handle is a
dict (NOT a path/URL string like the image handles)::

    {
        "video_url":  str,        # presigned URL of the 1-fps composed mp4
        "frame_urls": List[str],  # presigned URLs of frame_NNN.png, in order
        "n_frames":   int,
        "duration":   float,
    }

``select_frame`` resolves one frame URL out of the grid and registers it as a
new ``img_n`` handle (a plain URL string), which the agent's
``_load_pil_from_payload`` already knows how to fetch and attach to the next
turn. ``select_timespan`` slices a contiguous frame range into a new
``clip_n`` grid handle, optionally composing + uploading a real sub-clip mp4.

The grid-handle plumbing mirrors the inference pipeline in
``onesearch_vl/onesearch_infer/tools.py::execute_tool`` (the select_timespan /
select_frame branches), adapted to the async ``DeepResearchTool`` interface and
the shared ``image_paths`` side-channel used by the RL rollout.
"""

import os
import re
import time
from typing import Any, Dict, Optional

from vision_deepresearch_async_workflow.tools import video_io
from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    log_tool_event,
    next_handle_id,
)

# Reuse the COS uploader bootstrap from visual_tools so both image and video
# tools publish through the same (optional) ``upload.py``.
try:
    from vision_deepresearch_async_workflow.tools.visual_tools import (
        UPLOAD_AVAILABLE,
        _upload_pil_to_cos,  # noqa: F401  (re-exported for symmetry/tests)
        upload_cos,
    )
except Exception:  # noqa: BLE001  pragma: no cover - visual_tools always present
    UPLOAD_AVAILABLE = False

    def upload_cos(*args, **kwargs):
        return None, None


def _grid_handle(image_paths: dict, ref: str) -> Optional[Dict[str, Any]]:
    """Return the grid dict stored under *ref*, or ``None`` if not a grid."""

    entry = image_paths.get(ref)
    if isinstance(entry, dict) and entry.get("frame_urls"):
        return entry
    return None


def _upload_local_mp4(local_path: str, tag: str) -> Optional[str]:
    """Upload a local mp4 to COS and return its public URL (or ``None``).

    Mirrors ``cos_upload.upload_local_mp4`` but goes through the same
    ``upload.py`` loaded by ``visual_tools`` so the RL package needs no
    extra uploader dependency.
    """

    if not UPLOAD_AVAILABLE:
        return None
    if not os.path.exists(local_path):
        return None
    try:
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")
        ts = int(time.time() * 1000)
        filename = f"vdr_{tag}_{ts}.mp4"
        userid = os.getenv("COS_USERID", "onesearch-vl")
        cos_key, cos_url = upload_cos(
            local_path, filename, date_str, f"vdr_{date_str}", userid,
            use_direct_url=True,
        )
        return cos_url or None
    except Exception as exc:  # noqa: BLE001
        log_tool_event(
            "VideoTool", "UploadMp4Error", f"{type(exc).__name__}: {exc}",
            level="WARNING",
        )
        return None


class SelectTimespanTool(DeepResearchTool):
    """Slice a contiguous frame range out of a video/clip grid handle."""

    def __init__(self):
        super().__init__(
            name="select_timespan",
            description=(
                "Cut a sub-clip from a video handle and return a new "
                "``clip_n`` reference. Use this to focus the rest of "
                "the toolchain on a contiguous frame range instead of "
                "the whole video. The agent always sees a fixed grid "
                "of N evenly-sampled frames per video/clip (N <= 128); "
                "this tool takes the indices in that grid, not seconds."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "video": {
                        "type": "string",
                        "description": (
                            "Handle of the source (e.g., 'video_1', "
                            "'clip_2')."
                        ),
                    },
                    "start_index": {
                        "type": "integer",
                        "description": (
                            "Start frame index (0-based, inclusive) "
                            "within the source's N-frame grid."
                        ),
                    },
                    "end_index": {
                        "type": "integer",
                        "description": (
                            "End frame index (exclusive, must be > "
                            "start_index, <= N)."
                        ),
                    },
                },
                "required": ["video", "start_index", "end_index"],
            },
        )

    async def call(
        self,
        video: str = "",
        clip: str = "",
        start_index: Any = None,
        end_index: Any = None,
        start_idx: Any = None,
        end_idx: Any = None,
        **ctx,
    ) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        video_ref = video or clip
        if not video_ref:
            return (
                "Tool execution error:\n'video' (or 'clip') is required for "
                "select_timespan."
            )
        entry = _grid_handle(image_paths, video_ref)
        if entry is None:
            return (
                f"Tool execution error:\nselect_timespan expects a frame-grid "
                f"handle, got {video_ref!r} = "
                f"{type(image_paths.get(video_ref)).__name__}. "
                f"Available: {list(image_paths.keys())}"
            )

        si = start_index if start_index is not None else start_idx
        ei = end_index if end_index is not None else end_idx
        if si is None or ei is None:
            return (
                "Tool execution error:\nselect_timespan requires integer "
                f"start_index / end_index (0 <= start < end <= {entry.get('n_frames')})."
            )
        try:
            si_i, ei_i = video_io.parse_grid_index(si), video_io.parse_grid_index(ei)
        except (TypeError, ValueError):
            return "Tool execution error:\nstart_index / end_index must be integers."

        clip_id = next_handle_id(image_paths, "clip_")
        sub_grid = video_io.grid_slice(entry, si_i, ei_i)
        if not sub_grid:
            return (
                f"Tool execution error:\nFailed to slice frame range "
                f"[{si_i}, {ei_i}) from {video_ref!r}."
            )

        # Publish a clip only when its video and frame sequence describe the
        # same range. A failed composition must not expose the parent video.
        compose = os.environ.get("SELECT_TIMESPAN_COMPOSE_SUBCLIP", "1") == "1"
        parent_url = entry.get("video_url") or ""
        if not compose or not parent_url:
            return "Tool execution error: select_timespan requires subclip composition and video_url."
        # url_offset = seconds of frame 0 of THIS source inside parent_url.
        parent_offset = int(entry.get("url_offset", 0) or 0)
        cut_s, cut_e = sub_grid.get("parent_range", [si_i, ei_i])
        warn_suffix = ""
        if compose and parent_url:
            try:
                cache_dir = os.path.join(intermediate_dir, "grid_cache")
                parent_local = await self._run_blocking(
                    lambda: video_io.cache_parent_grid_mp4(parent_url, cache_dir)
                )
                if parent_local is None:
                    warn_suffix = (
                        " [warn: parent mp4 download failed]"
                    )
                else:
                    os.makedirs(intermediate_dir, exist_ok=True)
                    ts = int(time.time() * 1000)
                    subclip_path = os.path.join(
                        intermediate_dir, f"vdr_select_timespan_{clip_id}_{ts}.mp4"
                    )
                    ok = await self._run_blocking(
                        lambda: video_io.cut_subclip(
                            parent_local, cut_s, cut_e, subclip_path,
                            start_offset=parent_offset,
                        )
                    )
                    if not ok:
                        warn_suffix = (
                            " [warn: ffmpeg cut failed]"
                        )
                    else:
                        sub_url = await self._run_blocking(
                            lambda: _upload_local_mp4(subclip_path, f"select_timespan_{clip_id}")
                        )
                        if sub_url:
                            # The uploaded mp4 is freshly re-indexed from second
                            # 0, so the new clip's frame 0 sits at offset 0 of
                            # its own video_url — reset url_offset.
                            sub_grid["video_url"] = sub_url
                            sub_grid["subclip_path"] = subclip_path
                            sub_grid["url_offset"] = 0
                        else:
                            warn_suffix = (
                                " [warn: sub-clip upload failed]"
                            )
            except Exception as exc:  # noqa: BLE001
                log_tool_event(
                    "VideoTool", "ComposeError",
                    f"select_timespan compose failed: {type(exc).__name__}: {exc}",
                    level="WARNING",
                )
                warn_suffix = (
                    " [warn: sub-clip composition errored]"
                )

        if warn_suffix:
            return "Tool execution error: requested subclip could not be composed." + warn_suffix
        image_paths[clip_id] = sub_grid
        return (
            f"select_timespan succeeded. New clip handle: {clip_id} "
            f"(frames=[{si_i}, {ei_i}) of {video_ref!r}; this clip is now shown "
            f"to you as {sub_grid['n_frames']} frames indexed "
            f"0..{sub_grid['n_frames'] - 1}).{warn_suffix}"
        )


class SelectFrameTool(DeepResearchTool):
    """Extract a single still frame from a video/clip grid handle."""

    def __init__(self):
        super().__init__(
            name="select_frame",
            description=(
                "Extract a single still frame from a video/clip handle "
                "by index into the agent-visible N-frame grid "
                "(N <= 128, evenly spaced over the source's full "
                "duration). Returns a new ``img_n`` reference suitable "
                "for crop / image_search."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "video": {
                        "type": "string",
                        "description": "Source handle (video_n / clip_n).",
                    },
                    "frame_index": {
                        "type": "integer",
                        "description": (
                            "0-based frame index within the source's "
                            "N-frame grid (0 <= frame_index < N)."
                        ),
                    },
                },
                "required": ["video", "frame_index"],
            },
        )

    async def call(
        self,
        video: str = "",
        clip: str = "",
        frame_index: Any = None,
        idx: Any = None,
        **ctx,
    ) -> str:
        image_paths: dict = ctx.get("image_paths", {})

        video_ref = video or clip
        if not video_ref:
            return (
                "Tool execution error:\n'video' (or 'clip') is required for "
                "select_frame."
            )
        entry = _grid_handle(image_paths, video_ref)
        if entry is None:
            return (
                f"Tool execution error:\nselect_frame expects a frame-grid "
                f"handle, got {video_ref!r} = "
                f"{type(image_paths.get(video_ref)).__name__}. "
                f"Available: {list(image_paths.keys())}"
            )

        fi = frame_index if frame_index is not None else idx
        if fi is None:
            return (
                "Tool execution error:\nselect_frame requires integer "
                f"frame_index (0 <= frame_index < {entry.get('n_frames')})."
            )
        try:
            fi_i = video_io.parse_grid_index(fi)
        except (TypeError, ValueError):
            return (
                "Tool execution error:\nselect_frame requires integer "
                f"frame_index (0 <= frame_index < {entry.get('n_frames')})."
            )

        frame_url = video_io.grid_frame_path(entry, fi_i)
        if not frame_url:
            return (
                f"Tool execution error:\nframe_index {fi_i} not available "
                f"for {video_ref!r}."
            )

        new_id = next_handle_id(image_paths, "img_")
        image_paths[new_id] = frame_url
        n_frames = int(entry.get("n_frames", 0))
        idx_label = f"frame_index={fi_i}/{n_frames - 1}"
        return (
            f"select_frame succeeded. New image handle: {new_id} "
            f"({idx_label} from {video_ref!r})."
        )

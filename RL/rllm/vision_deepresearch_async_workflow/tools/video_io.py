"""Grid handle helpers for local or remote pre-materialised media.

All raw-video processing (ffmpeg transcode, decord frame extraction, mp4
composition, S3 upload) has moved offline to ``preprocessing/preprocess_grids.py``.
At runtime the deepresearch pipeline only ever sees pre-materialised grid
records loaded from ``grids_manifest.jsonl``. A grid is::

    {
        "video_url":  str,       # presigned URL of the 1-fps composed mp4
        "frame_urls": List[str], # presigned URLs of frame_NNN.png, in order
        "n_frames":   int,
        "duration":   float,
    }

This module exposes the two ops the agent loop needs against such a grid:
``grid_frame_path`` (single frame URL by index) and ``grid_slice`` (sub-range
URL list as a new clip handle).

``select_timespan`` additionally composes a real sub-clip mp4 (since the
parent grid.mp4 is encoded at 1fps with long_edge=512, ``frame_index``
equals seconds in that mp4). ``cache_parent_grid_mp4`` + ``cut_subclip``
do the download-once + ffmpeg cut.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from fractions import Fraction
import logging
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)



def parse_grid_index(value):
    """Accept integers or their string form; never truncate floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Frame indices must be integers")
    if isinstance(value, str) and not value.lstrip("+-").isdigit():
        raise ValueError("Frame indices must be integers")
    return int(value)


@lru_cache(maxsize=8192)
def _probe_local_grid(path, mtime_ns, size):
    del mtime_ns, size  # File identity participates in the cache key.
    result = subprocess.check_output(
        ["ffprobe", "-v", "error", "-analyzeduration", "0", "-probesize", "32768",
         "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate,nb_frames", "-of", "json", path], timeout=30)
    stream = json.loads(result)["streams"][0]
    return float(Fraction(stream["avg_frame_rate"])), int(stream["nb_frames"])


def validate_grid_contract(record):
    """Reject inconsistent inventories and local full videos posing as grids."""
    urls = record.get("frame_urls") or record.get("object_urls") or []
    n = parse_grid_index(record.get("n_frames", len(urls)))
    if n <= 0 or len(urls) != n:
        raise ValueError("Grid n_frames must exactly match frame_urls")
    if len(set(urls)) != n:
        raise ValueError("Grid frame URLs must be unique and ordered")
    indices = record.get("frame_indices")
    if indices is not None and list(indices) != list(range(n)):
        raise ValueError("Root grid frame_indices must be contiguous zero-based slots")
    for key in ("source_frame_indices", "source_timestamps_s"):
        if key in record and len(record[key]) != n:
            raise ValueError(f"Grid {key} length must match n_frames")
    if "grid_fps" in record and float(record["grid_fps"]) != 1.0:
        raise ValueError("Grid playback must be 1 fps")
    numbered = [re.fullmatch(r"frame_(\d+)\.png", urllib.parse.urlparse(u).path.rsplit("/", 1)[-1]) for u in urls]
    if all(numbered) and [int(m.group(1)) for m in numbered] != list(range(n)):
        raise ValueError("Grid PNG filenames must follow slot order")
    video = str(record.get("video_url") or "")
    if not video:
        raise ValueError("Grid video_url is required")
    if not urllib.parse.urlparse(video).scheme and os.path.isfile(video):
        stat = os.stat(video)
        fps, count = _probe_local_grid(os.path.realpath(video), stat.st_mtime_ns, stat.st_size)
        if abs(fps - 1.0) > 1e-6 or count != n:
            raise ValueError(f"Expected a 1 fps/{n}-frame grid video, got {fps} fps/{count} frames")
    return n


def build_grid_from_manifest(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a manifest line into the runtime grid dict.

    Accepts either the new (``frame_urls``) layout or the legacy
    ``object_urls`` field name produced by ``preprocess_grids.py``.
    """

    frame_urls: Optional[List[str]] = record.get("frame_urls") or record.get(
        "object_urls"
    )
    if not frame_urls:
        raise ValueError(
            f"manifest record missing frame_urls/object_urls: keys={list(record)}"
        )
    n_frames = validate_grid_contract(record)
    duration = float(n_frames)  # Playback seconds; source times have separate fields.
    video_url = record.get("video_url")
    if not video_url:
        raise ValueError(
            f"manifest record missing video_url: keys={list(record)}"
        )
    return {
        "video_url": video_url,
        "frame_urls": list(frame_urls[:n_frames]),
        "n_frames": n_frames,
        "duration": duration,
        "grid_fps": 1.0,
        "url_offset": 0,
    }



def grid_frame_path(grid: Dict[str, Any], frame_index: int) -> Optional[str]:
    """Return a frame URL, or None for an out-of-range slot."""

    n = int(grid.get("n_frames", 0))
    urls = grid.get("frame_urls") or []
    if n <= 0 or not urls:
        return None
    i = parse_grid_index(frame_index)
    if not 0 <= i < n:
        return None
    if i >= len(urls):
        return None
    return urls[i]


def grid_slice(
    grid: Dict[str, Any],
    start_index: int,
    end_index: int,
) -> Optional[Dict[str, Any]]:
    """Return a valid half-open grid slice; reject every out-of-range request.

    url_offset records where the slice begins in its inherited playback video.
    """

    n = int(grid.get("n_frames", 0))
    urls = grid.get("frame_urls") or []
    if n <= 0 or not urls:
        return None
    s, e = parse_grid_index(start_index), parse_grid_index(end_index)
    if not 0 <= s < e <= n:
        return None
    sliced = list(urls[s:e])
    if not sliced:
        return None
    duration = float(grid.get("duration", 0.0))
    sub_duration = (e - s) * duration / n if n > 0 else 0.0
    # ``url_offset`` tracks the second-offset of frame 0 of THIS clip inside its
    # ``video_url``. A sliced clip inherits the parent ``video_url`` (its frames
    # are re-indexed 0..M-1), so cutting a nested sub-clip must seek at
    # parent_url_offset + s seconds, not at the clip-relative index. select_frame
    # composition resets this to 0 when it uploads a freshly re-indexed mp4.
    parent_offset = int(grid.get("url_offset", 0) or 0)
    return {
        "video_url": grid.get("video_url"),
        "frame_urls": sliced,
        "n_frames": len(sliced),
        "duration": sub_duration,
        "parent_range": [s, e],
        "url_offset": parent_offset + s,
    }


# ---------------------------------------------------------------------------
# Sub-clip composition
#
# The parent grid.mp4 was produced offline at 1fps, long_edge=512 (see
# preprocessing/preprocess_grids.py::compose_mp4_from_grid). That means
# frame_index N in the grid lands at second N of the mp4 — so cutting a
# sub-clip is a straight `ffmpeg -ss <si> -t <ei-si> -i parent.mp4` with
# accurate seek. We pay one parent-download per case (cached locally) and
# one ffmpeg re-encode per sub-clip.
# ---------------------------------------------------------------------------


def _safe_filename(url: str) -> str:
    """Return a short deterministic filename derived from ``url``."""

    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def _local_video_path(video_url: str) -> tuple[bool, Optional[str]]:
    """Resolve a filesystem-backed video reference.

    Returns ``(is_local_reference, usable_path)``.  Plain paths and ``file:``
    URLs are local references even when missing, so they fail immediately
    instead of being retried through ``urllib`` as malformed network URLs.
    """

    parsed = urllib.parse.urlparse(video_url)
    if parsed.scheme not in ("", "file"):
        return False, None
    if parsed.scheme == "file":
        path = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
        if parsed.netloc and parsed.netloc != "localhost":
            path = f"//{parsed.netloc}{path}"
    else:
        path = video_url
    path = os.path.abspath(path)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return True, path
    return True, None


def cache_parent_grid_mp4(
    video_url: str,
    cache_dir: str,
) -> Optional[str]:
    """Materialise ``video_url`` locally once and return its path.

    Existing local files are returned directly; remote URLs are downloaded
    once and cached. Returns ``None`` for missing or empty local references
    and failed remote downloads.
    """

    if not video_url:
        return None
    is_local, local_source = _local_video_path(video_url)
    if is_local:
        if local_source is None:
            logger.warning(
                "cache_parent_grid_mp4: local video is missing or empty: %s",
                video_url,
            )
        return local_source
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("cache_parent_grid_mp4: cannot create cache dir: %s", exc)
        return None
    local_path = os.path.join(cache_dir, f"{_safe_filename(video_url)}.mp4")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    # Retry the download a few times: under the 128-shard rollout herd a
    # presigned URL can hit a transient 5xx / connection reset, and a bare
    # single attempt would needlessly drop the whole sub-clip to the parent.
    attempts = max(1, int(os.environ.get("GRID_MP4_DOWNLOAD_RETRIES", "3")))
    for attempt in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(video_url, local_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "parent grid mp4 download failed (attempt %d/%d, %s): %s",
                attempt, attempts, video_url, exc,
            )
            try:
                os.remove(local_path)
            except OSError:
                pass
            if attempt >= attempts:
                return None
            time.sleep(min(2 ** (attempt - 1), 5))
            continue
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        if attempt >= attempts:
            return None
    return None


def cut_subclip(
    parent_mp4: str,
    start_index: int,
    end_index: int,
    dest_path: str,
    *,
    start_offset: int = 0,
    crf: int = 23,
    timeout: int = 120,
) -> bool:
    """Cut ``[start_index, end_index)`` seconds out of ``parent_mp4``.

    ``start_offset`` is the second-offset of frame 0 of the SOURCE handle inside
    ``parent_mp4`` (the grid's ``url_offset``). When a clip inherited its parent
    ``video_url`` (compose fell back), its frame indices are relative to the
    parent, so the real seek is ``start_offset + start_index``. For a fresh grid
    ``url_offset`` is 0 and this is a no-op.

    Re-encodes with libx264 ultrafast for frame-accurate cuts.

    Returns ``False`` (never raises) on any failure — missing ffmpeg,
    unwritable dest dir, ffmpeg error/timeout — so callers can fall back to the
    parent video instead of propagating an exception out of the tool call.
    """

    binary = shutil.which("ffmpeg")
    if binary is None:
        logger.error("ffmpeg not on PATH; cannot cut sub-clip.")
        return False
    duration = max(0.0, float(end_index) - float(start_index))
    if duration <= 0:
        return False
    seek = max(0.0, float(start_offset) + float(start_index))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    except OSError as exc:
        logger.warning("cut_subclip: cannot create dest dir: %s", exc)
        return False
    cmd = [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        parent_mp4,
        "-ss",
        f"{seek:.3f}",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        str(crf),
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-avoid_negative_ts",
        "make_zero",
        dest_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ) as exc:
        logger.warning("ffmpeg sub-clip cut failed: %s", exc)
        return False
    return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0

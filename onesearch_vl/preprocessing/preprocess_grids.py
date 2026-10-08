"""Original frame sampling and silent 1-FPS grid encoding helpers.

Use prepare_video.py for the portable publishing entrypoint.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from typing import Any, Dict, Optional

# Make sibling packages importable when run as `python tools/preprocess_grids.py`.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)



DEFAULT_LONG_EDGE = 1024
DEFAULT_MAX_FRAMES = 128
DEFAULT_TARGET_FPS = 2.0




def _probe_duration_fps(src_path: str) -> tuple[float, float, int]:
    """Return (duration_seconds, src_fps, src_nb_frames). Falls back to 0.

    Tries ffprobe first (richest output), then ffmpeg -i stderr parsing
    (some runtime images ship ffmpeg but
    no ffprobe), then decord as a last resort.
    """

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate,nb_frames",
        "-of", "json",
        src_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30)
        data = json.loads(out)
        s = (data.get("streams") or [{}])[0]
        duration = float(s.get("duration") or 0.0)
        rfr = s.get("r_frame_rate") or "0/1"
        num, _, den = rfr.partition("/")
        try:
            src_fps = float(num) / float(den) if den else float(num)
        except (ValueError, ZeroDivisionError):
            src_fps = 0.0
        try:
            nb_frames = int(s.get("nb_frames") or 0)
        except (TypeError, ValueError):
            nb_frames = 0
        if nb_frames <= 0 and duration > 0 and src_fps > 0:
            nb_frames = int(round(duration * src_fps))
        if duration > 0:
            return duration, src_fps, nb_frames
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Fallback 1: ffmpeg -i stderr parsing.
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", src_path],
            capture_output=True, text=True, timeout=30,
        )
        text = proc.stderr or ""
        duration = 0.0
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            duration = h * 3600 + mn * 60 + s
        src_fps = 0.0
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
        if m2:
            src_fps = float(m2.group(1))
        nb_frames = int(round(duration * src_fps)) if duration > 0 and src_fps > 0 else 0
        if duration > 0:
            return duration, src_fps, nb_frames
    except Exception:
        pass

    # Fallback 2: decord (slow on AV1/VP9 8K but always works).
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(src_path, ctx=cpu(0))
        nb_frames = len(vr)
        src_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 0.0
        duration = (nb_frames / src_fps) if src_fps > 0 else 0.0
        del vr
        if duration > 0:
            return duration, src_fps, nb_frames
    except Exception:
        pass

    return 0.0, 0.0, 0


def _prep_video(
    src_path: str,
    dst_path: str,
    *,
    long_edge: int = DEFAULT_LONG_EDGE,
    target_fps: float = DEFAULT_TARGET_FPS,
    crf: int = 23,
) -> bool:
    """Down-sample + downscale ``src_path`` into a small H264 mp4.

    Output is ``target_fps`` (default 2) and long-edge ``long_edge``
    (default 1024), ready for frame extraction with decord.

    The scale filter clamps the long edge without upscaling small inputs:
    ``min(long_edge, iw)`` / ``min(long_edge, ih)``. ``-2`` on the other
    axis tells libx264-friendly even dimensions while preserving aspect.
    """

    vf = (
        f"fps={target_fps},"
        f"scale='if(gt(iw,ih),min({long_edge},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({long_edge},ih))'"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", str(crf),
        "-an",
        "-pix_fmt", "yuv420p",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=7200)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg prep failed for %s: %s", src_path, exc)
        return False
    return os.path.exists(dst_path) and os.path.getsize(dst_path) > 0


def materialize_frame_grid(
    src_path: str,
    workdir: str,
    *,
    long_edge: int = DEFAULT_LONG_EDGE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> Optional[Dict[str, Any]]:
    """Extract evenly-spaced PNG frames into ``workdir``.

    Pipeline: ffmpeg-prep to (target_fps, long_edge) H264 → decord-open →
    np.linspace pick N <= 128 indices over the prepped clip's frame count
    → decode one-at-a-time → save PNG (no further resize needed; the prep
    already matches the target long edge).

    Returns ``{n_frames, duration, src_fps, src_total, src_indices,
    long_edge, workdir}`` on success, ``None`` on failure. Writes
    ``workdir/frame_NNN.png`` for N in [0, n_frames). ``duration``,
    ``src_fps``, ``src_total`` are reported as observed on the SOURCE
    video (via ffprobe) for downstream traceability.
    """

    import numpy as np  # local import keeps preprocess_grids importable
    from decord import VideoReader, cpu
    from PIL import Image

    os.makedirs(workdir, exist_ok=True)

    # Probe the SOURCE for trace fields (duration / fps / total). The
    # prepped clip we hand decord later has its own frame count which
    # drives sampling, but the manifest records the original numbers.
    src_duration, src_fps, src_total = _probe_duration_fps(src_path)
    if src_duration <= 0:
        logger.warning("source probe failed for %s", src_path)
        return None

    # Stage 1: prep clip.
    prepped = os.path.join(workdir, "_prepped.mp4")
    if not _prep_video(src_path, prepped, long_edge=long_edge, target_fps=target_fps):
        return None

    try:
        vr = VideoReader(prepped, ctx=cpu(0))
    except Exception as exc:
        logger.warning("decord open failed for prepped %s: %s", prepped, exc)
        try:
            os.remove(prepped)
        except OSError:
            pass
        return None

    prepped_total = len(vr)
    if prepped_total <= 0:
        logger.warning("prepped clip empty for %s", src_path)
        del vr
        try:
            os.remove(prepped)
        except OSError:
            pass
        return None

    target_frames = max(1, min(int(round(target_fps * src_duration)), max_frames))
    target_frames = min(target_frames, prepped_total)

    # Pick evenly-spaced indices over the prepped clip's frame count
    # (which is roughly target_fps * src_duration; might be a few frames
    # off because of trailing-fractional-second handling).
    raw_idx = np.linspace(
        0, prepped_total, num=target_frames, endpoint=False
    ) + (prepped_total / target_frames / 2.0)
    prepped_indices = [
        int(min(prepped_total - 1, max(0, round(x)))) for x in raw_idx
    ]
    # Trace field maps the prepped index back into the source frame space.
    if src_total > 0:
        src_indices = [
            int(min(src_total - 1, round(p * src_total / prepped_total)))
            for p in prepped_indices
        ]
    else:
        src_indices = list(prepped_indices)

    n_frames = 0
    try:
        for i, idx in enumerate(prepped_indices):
            frame_np = vr.get_batch([idx]).asnumpy()[0]
            img = Image.fromarray(frame_np)
            del frame_np
            out_path = os.path.join(workdir, f"frame_{i:03d}.png")
            img.save(out_path, format="PNG", optimize=True)
            img.close()
            n_frames += 1
    except Exception as exc:
        logger.warning("decord/PIL decode failed for %s: %s", src_path, exc)
        return None
    finally:
        # decord doesn't expose explicit close on all builds; drop the ref
        # so the underlying ffmpeg handle is freed before prepped cleanup.
        del vr
        try:
            os.remove(prepped)
        except OSError:
            pass

    if n_frames <= 0:
        logger.warning("no frames written for %s", src_path)
        return None

    return {
        "n_frames": n_frames,
        "duration": src_duration,
        "src_fps": src_fps,
        "src_total": src_total,
        "src_indices": src_indices[:n_frames],
        "long_edge": long_edge,
        "workdir": workdir,
    }


def compose_mp4_from_grid(
    grid: Dict[str, Any],
    dest_path: str,
    *,
    crf: int = 23,
    video_long_edge: Optional[int] = None,
) -> bool:
    """Build a 1-fps mp4 by stitching frame_NNN.png in order.

    The grid mp4 lives next to the PNGs on S3 so the agent can read either
    a single frame or the full clip without a second pass through ffmpeg.

    ``video_long_edge`` controls the composed video's resolution independently
    of the source PNGs. Defaults to keeping the PNG's native size.
    """

    workdir = grid.get("workdir")
    n = int(grid.get("n_frames") or 0)
    if not workdir or n <= 0:
        return False
    pattern = os.path.join(workdir, "frame_%03d.png")
    if not os.path.exists(os.path.join(workdir, "frame_000.png")):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-framerate", "1",
        "-i", pattern,
        "-frames:v", str(n),
    ]
    if video_long_edge and video_long_edge > 0:
        # scale long edge to video_long_edge, other side -2 to keep even
        cmd += [
            "-vf",
            f"scale='if(gt(iw,ih),min({video_long_edge},iw),-2)':"
            f"'if(gt(iw,ih),-2,min({video_long_edge},ih))'",
        ]
    cmd += [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dest_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("compose_mp4 failed (%s): %s", dest_path, exc)
        return False
    return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0


logging.basicConfig(
    format="[%(asctime)s][%(levelname)5s][%(processName)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("preprocess_grids")


# ---------------------------------------------------------------------------

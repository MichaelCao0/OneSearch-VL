"""Native audio/video handles and ordinary, synchronized FFmpeg crops.

Handle seconds start at the first AV stream on a common source timeline.
``inspect_av`` intervals are relative to the supplied handle, including clips;
``source_start_s``/``source_end_s`` always refer to that original timeline.
These handles are separate from the original inference path's frame grids.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import unquote, urlsplit


def _seconds(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number of seconds")
    return float(value)


def _local_path(value) -> Path:
    raw = os.fspath(value)
    parsed = urlsplit(raw)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        raw = unquote(parsed.path)
    elif parsed.scheme:
        raise ValueError("Native AV requires a local media file")
    path = Path(raw).resolve()
    if not path.is_file():
        raise ValueError("Native AV media file is missing")
    return path


def _probe(path: Path, timeout: float = 30) -> dict:
    binary = shutil.which("ffprobe")
    if binary is None:
        raise RuntimeError("ffprobe is required for native AV")
    try:
        output = subprocess.run([binary, "-v", "error", "-show_streams", "-show_format",
                                 "-of", "json", str(path)], check=True, capture_output=True,
                                timeout=timeout)
        return json.loads(output.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValueError("Unable to probe native AV media") from exc


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _duration_tag(value):
    try:
        hours, minutes, seconds = value.split(":")
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    except (ValueError, TypeError, AttributeError):
        return None


def probe_av(source_path) -> dict:
    """Read basic playback bounds without requiring a CFR sampling grid.

    MP4 stream durations are durations; Matroska's DURATION tags instead denote
    the stream endpoint even when its first timestamp is nonzero.
    """
    path = _local_path(source_path)
    data = _probe(path)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") in {"video", "audio"}
               and not s.get("disposition", {}).get("attached_pic")]
    videos = [s for s in streams if s["codec_type"] == "video"]
    if not videos:
        raise ValueError("Native AV requires a video stream")
    selected = videos[:1] + [s for s in streams if s["codec_type"] == "audio"][:1]
    format_data = data.get("format", {})
    format_start = _number(format_data.get("start_time")) or 0.0
    origin = min((_number(s.get("start_time")) if _number(s.get("start_time")) is not None
                  else format_start) for s in selected)
    endpoints = []
    audio_bounds = None
    for stream in selected:
        start = _number(stream.get("start_time"))
        duration = _number(stream.get("duration"))
        tag = _duration_tag(stream.get("tags", {}).get("DURATION"))
        endpoint = None
        if duration is not None and duration > 0:
            endpoint = (start if start is not None else origin) + duration
        elif tag is not None and math.isfinite(tag) and tag > origin:
            endpoint = tag
        if endpoint is not None:
            endpoints.append(endpoint)
        if stream["codec_type"] == "audio":
            audio_bounds = ((start if start is not None else origin) - origin,
                            endpoint - origin if endpoint is not None else None)
    if not endpoints:
        # Container fallback supports normal files whose streams omit durations.
        duration = _number(format_data.get("duration"))
        if duration is not None:
            endpoints.append(duration if "matroska" in format_data.get("format_name", "")
                             else format_start + duration)
    duration = max(endpoints, default=origin) - origin
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Native AV has no finite positive duration")
    return {"source_path": str(path), "duration": duration,
            "source_origin_s": origin, "has_audio": audio_bounds is not None,
            "audio_start_s": audio_bounds[0] if audio_bounds is not None else None,
            "audio_end_s": (audio_bounds[1] if audio_bounds[1] is not None else duration)
                           if audio_bounds is not None else None}


def make_native_av_handle(source_path) -> dict:
    """Register a local original media file without exposing its path in text."""
    info = probe_av(source_path)
    return {"kind": "native_av", "media_type": "native_av", "source_path": info["source_path"],
            "video_url": info["source_path"], "source_start_s": 0.0,
            "source_end_s": info["duration"], "duration": info["duration"],
            "source_origin_s": info["source_origin_s"], "has_audio": info["has_audio"]}


def crop_native_av(handle: dict, start_s, end_s, dest_path, *, timeout: float = 120) -> dict:
    """Crop a handle-relative half-open interval, always from the original file.

    Both streams subtract the same timestamp. Audio is padded with silence to
    that common zero, so decoders that return samples without stream timestamps
    still retain a delayed audio onset. VFR video remains VFR; no grid gate is
    imposed. Output is a local H.264/AAC MP4 (video-only when no audio exists).
    """
    if not isinstance(handle, dict) or handle.get("media_type", handle.get("kind")) != "native_av":
        raise ValueError("inspect_av requires a registered native AV handle")
    start, end = _seconds(start_s, "start_s"), _seconds(end_s, "end_s")
    duration = _seconds(handle.get("duration"), "handle duration")
    offset = _seconds(handle.get("source_start_s"), "source_start_s")
    source_end = _seconds(handle.get("source_end_s"), "source_end_s")
    if duration <= 0 or offset < 0 or not math.isclose(source_end - offset, duration, abs_tol=1e-6):
        raise ValueError("Invalid native AV handle time bounds")
    if not 0 <= start < end <= duration:
        raise ValueError("inspect_av requires 0 <= start_s < end_s <= handle duration")
    source = _local_path(handle.get("source_path", ""))
    info = probe_av(source)
    absolute_start, absolute_end = offset + start, offset + end
    if absolute_end > info["duration"] + 1e-6:
        raise ValueError("Requested interval exceeds original media duration")
    output = Path(dest_path).resolve()
    if output == source:
        raise ValueError("A native crop cannot overwrite its original source")
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg is required for native AV crops")
    output.parent.mkdir(parents=True, exist_ok=True)
    begin, finish = info["source_origin_s"] + absolute_start, info["source_origin_s"] + absolute_end
    include_audio = (info["has_audio"] and absolute_end > info["audio_start_s"]
                     and absolute_start < info["audio_end_s"])
    filters = [f"[0:v:0]trim=start={begin:.9f}:end={finish:.9f},"
               f"setpts=PTS-({begin:.9f})/TB,scale=trunc(iw/2)*2:trunc(ih/2)*2[v]"]
    if include_audio:
        filters.append(f"[0:a:0]atrim=start={begin:.9f}:end={finish:.9f},"
                       f"asetpts=PTS-({begin:.9f})/TB,aresample=async=1:first_pts=0[a]")
    with tempfile.NamedTemporaryFile(prefix="native-av-", suffix=".mp4", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    cmd = [binary, "-nostdin", "-y", "-v", "error", "-copyts", "-i", str(source),
           "-filter_complex", ";".join(filters), "-map", "[v]"]
    if include_audio:
        cmd += ["-map", "[a]", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-bf", "0",
            "-pix_fmt", "yuv420p", "-fps_mode", "vfr", "-t", f"{end-start:.9f}",
            "-map_metadata", "-1", "-movflags", "+faststart", str(temporary)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        result = _probe(temporary)
        if not any(s.get("codec_type") == "video" for s in result.get("streams", [])):
            raise ValueError("Requested interval contains no video frames")
        has_audio = any(s.get("codec_type") == "audio" for s in result["streams"])
        os.replace(temporary, output)
    except subprocess.SubprocessError as exc:
        raise ValueError("Unable to crop native AV media") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {"kind": "native_av", "media_type": "native_av", "source_path": str(source),
            "video_url": str(output), "source_start_s": absolute_start,
            "source_end_s": absolute_end, "source_origin_s": info["source_origin_s"],
            "duration": end-start, "has_audio": has_audio,
            "parent_start_s": start, "parent_end_s": end}

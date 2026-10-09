"""Original presentation clocks and synchronized, mapped AV clipping.

Clips use one shared timestamp subtraction, not per-stream STARTPTS. No frame
rate conversion, audio resampling, padding or source synchronization is applied.
Outputs use lossless FFV1/PCM float in Matroska; originals are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
from time import perf_counter

from .media_cost import instrument_media_args, record_media_operation

from .schemas import DERIVED_SCHEMA_VERSION, TOOL_REVISION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_media(args: list[str]) -> bytes:
    """Execute argument arrays and account successes and unknown failed work."""
    stdout, stderr = b"", b""
    status, returncode, failure_kind = "failed", None, None
    started = perf_counter()
    try:
        result = subprocess.run(instrument_media_args(args), check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=300)
        stdout, stderr = result.stdout, result.stderr
        status, returncode = "success", result.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None) or b""
        stderr = getattr(exc, "stderr", None) or b""
        returncode = getattr(exc, "returncode", None)
        failure_kind = ("timeout" if isinstance(exc, subprocess.TimeoutExpired) else
                        "nonzero_exit" if isinstance(exc, subprocess.CalledProcessError) else
                        "os_error")
        detail = stderr.decode("utf-8", errors="replace")[-2000:] if isinstance(stderr, bytes) else str(stderr)
        raise ValueError(f"Media tool failed: {args[0]}: {detail or type(exc).__name__}") from exc
    finally:
        record_media_operation(args, stdout=stdout, stderr=stderr,
                               wall_seconds=perf_counter()-started, status=status,
                               returncode=returncode, failure_kind=failure_kind)
    return stdout


def tool_version(tool: str) -> str:
    return run_media([tool, "-version"]).decode().splitlines()[0]


def probe_media(path: str | Path) -> dict[str, Any]:
    """Decode original frame timestamps; metadata alone cannot establish VFR."""
    raw = run_media([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_frames",
        "-show_entries",
        "format=start_time,duration,format_name:stream=index,codec_type,codec_name,time_base,start_pts,start_time,duration,sample_rate,channels,channel_layout,width,height,r_frame_rate,avg_frame_rate,disposition:frame=media_type,stream_index,pts,pts_time,best_effort_timestamp,best_effort_timestamp_time,duration,duration_time,pkt_duration,pkt_duration_time,nb_samples",
        "-of", "json", str(Path(path).resolve()),
    ])
    return json.loads(raw)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def clock_specs(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    """Record decode PTS/sample coordinates against the original container axis."""
    streams = [s for s in data.get("streams", []) if s.get("codec_type") in ("audio", "video")]
    if not streams:
        raise ValueError("Asset has no decodable audio/video streams")
    start = _number(data.get("format", {}).get("start_time"))
    basis = "container_start_time"
    if start is None:
        starts = [_number(s.get("start_time")) for s in streams]
        known = [s for s in starts if s is not None]
        if known:
            start, basis = min(known), "earliest_stream_start_time_fallback"
        else:
            # WAV and similar elementary formats have decode PTS but no start_time.
            original_times = []
            by_index = {s["index"]: s for s in streams}
            for frame in data.get("frames", []):
                pts = frame.get("pts", frame.get("best_effort_timestamp"))
                if pts is not None and frame["stream_index"] in by_index:
                    original_times.append(float(int(pts) * Fraction(by_index[frame["stream_index"]]["time_base"])))
            if not original_times:
                raise ValueError("Cannot establish presentation origin: no original timestamps")
            start, basis = min(original_times), "earliest_decoded_pts_fallback"
    origin = {"kind": basis, "container_start_time_s": start,
              "axis": "source_local_presentation", "unit": "seconds", "interval": "half_open",
              "pts_to_source": {"scale": 1.0, "offset_s": -start},
              "format_duration_s": _number(data.get("format", {}).get("duration"))}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in data.get("frames", []):
        grouped[int(frame["stream_index"])].append(frame)
    specs = []
    latest_end = start
    for stream in streams:
        if stream.get("disposition", {}).get("attached_pic"):
            raise ValueError("Attached-picture video is not a timed AV stream")
        spec = dict(stream)
        index = int(stream["index"])
        try:
            time_base = Fraction(stream["time_base"])
            if time_base <= 0:
                raise ValueError("nonpositive time_base")
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Stream {index} has no valid time_base") from exc
        pts_records = []
        sample_records = []
        sample_cursor = 0
        for frame in grouped[index]:
            pts = frame.get("pts", frame.get("best_effort_timestamp"))
            if pts is None:
                raise ValueError(f"Stream {index} has a frame with no original presentation PTS")
            absolute = float(int(pts) * time_base)
            duration = _number(frame.get("duration_time", frame.get("pkt_duration_time")))
            if stream["codec_type"] == "audio":
                count = int(frame.get("nb_samples", 0))
                rate = int(stream.get("sample_rate", 0))
                if count <= 0 or rate <= 0:
                    raise ValueError(f"Audio stream {index} has no sample mapping")
                duration = count / rate
                sample_records.append({"sample_start": sample_cursor, "sample_count": count,
                    "pts": int(pts), "source_start_s": absolute - start,
                    "sample_rate": rate, "seconds_per_sample": 1.0 / rate})
                sample_cursor += count
            if duration is None or duration <= 0:
                rate = Fraction(stream.get("avg_frame_rate", "0/1"))
                duration = float(1 / rate) if rate > 0 else 0.0
            pts_records.append({"pts": int(pts), "source_time_s": absolute - start,
                                "duration_s": duration})
            latest_end = max(latest_end, absolute + duration)
        if not pts_records:
            raise ValueError(f"Stream {index} has no decoded frame timestamps")
        source_times = [f["source_time_s"] for f in pts_records]
        deltas = [b - a for a, b in zip(source_times, source_times[1:])]
        if any(d <= 0 for d in deltas):
            raise ValueError(f"Stream {index} has nonmonotonic presentation timestamps")
        spec.update({"source_offset_s": source_times[0], "frame_pts": pts_records,
                     "sample_mapping": sample_records,
                     "vfr": stream["codec_type"] == "video" and bool(deltas) and max(deltas)-min(deltas) > float(time_base) * 1.5,
                     "timestamp_status": "decoded_original_pts"})
        specs.append(spec)
    duration_s = latest_end - start
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Asset presentation duration is not positive")
    return origin, specs, duration_s


def clip_media(source_record: dict, start_s: float, end_s: float, output_dir: str | Path) -> dict:
    """Clip source-local [start_s,end_s), retaining every timed source AV stream."""
    from .validate import validate_source, validate_derived

    errors = validate_source(source_record, verify_hash=True)
    if errors:
        raise ValueError("Invalid SourceRecord: " + "; ".join(errors))
    if (isinstance(start_s, bool) or isinstance(end_s, bool) or
            not isinstance(start_s, (int, float)) or not isinstance(end_s, (int, float)) or
            _number(start_s) is None or _number(end_s) is None or
            start_s < 0 or end_s <= start_s or end_s > source_record["duration_s"] + 1e-9):
        raise ValueError("Interval must be finite source-local 0 <= start_s < end_s <= duration_s")
    start_s, end_s = float(start_s), float(end_s)
    origin_s = source_record["timestamp_origin"]["container_start_time_s"]
    absolute_start, absolute_end = origin_s + start_s, origin_s + end_s
    filters, maps, stream_mapping = [], [], []
    for output_index, spec in enumerate(source_record["stream_specs"]):
        index, kind = spec["index"], spec["codec_type"]
        # A stream with no actual data in this interval cannot be honestly retained.
        overlaps = [f for f in spec["frame_pts"] if start_s <= f["source_time_s"] < end_s]
        if kind == "audio":
            overlaps = [f for f in spec["frame_pts"] if f["source_time_s"] < end_s and
                        f["source_time_s"] + f["duration_s"] > start_s]
        if not overlaps:
            raise ValueError(f"Stream {index} has no decodable media in selected interval")
        trim, setpts = ("trim", "setpts") if kind == "video" else ("atrim", "asetpts")
        label = f"out{output_index}"
        filters.append(f"[0:{index}]{trim}=start={absolute_start:.12f}:end={absolute_end:.12f},{setpts}=PTS-{absolute_start:.12f}/TB[{label}]")
        maps += ["-map", f"[{label}]"]
        stream_mapping.append({"source_stream_index": index, "output_stream_index": output_index,
            "codec_type": kind, "source_time_base": spec["time_base"],
            "source_offset_s": spec["source_offset_s"]})
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".clip-", suffix=".mkv", dir=directory)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        args = ["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i", source_record["asset_path"],
                "-filter_complex", ";".join(filters), *maps, "-c:v", "ffv1", "-c:a", "pcm_f32le",
                "-fps_mode", "passthrough", "-enc_time_base:v", "filter",
                "-avoid_negative_ts", "disabled", "-y", str(temp_path)]
        run_media(args)
        output_data = probe_media(temp_path)
        output_streams = output_data.get("streams", [])
        if [s["codec_type"] for s in output_streams] != [s["codec_type"] for s in source_record["stream_specs"]]:
            raise ValueError("Clip did not retain every original AV stream")
        for mapping, stream in zip(stream_mapping, output_streams):
            mapping["output_time_base"] = stream["time_base"]
            mapping["output_start_time_s"] = float(stream.get("start_time", 0))
        output_hash = sha256_file(temp_path)
        destination = directory / f"derived-{output_hash}.mkv"
        if destination.exists():
            if sha256_file(destination) != output_hash:
                raise ValueError("Content-addressed derived destination is corrupt")
            temp_path.unlink()
        else:
            # Linking is atomic and cannot overwrite an existing artifact.
            os.link(temp_path, destination)
            temp_path.unlink()
        record = {"schema_version": DERIVED_SCHEMA_VERSION, "derived_id": f"derived-{output_hash[:24]}",
            "parent_source_id": source_record["source_id"], "parent_sha256": source_record["asset_sha256"],
            "created_at": utc_now(), "created_by": {"tool": "clip_media", "revision": TOOL_REVISION},
            "transform": {"kind": "source_clock_clip", "video_codec": "ffv1", "audio_codec": "pcm_f32le",
                          "frame_rate_conversion": False, "audio_resampling": False},
            "tool_version": tool_version("ffmpeg"),
            "original_interval": {"start_s": start_s, "end_s": end_s},
            "mapping": {"axis": "source_local_presentation", "output_axis": "presentation_pts_seconds",
                        "interval": "half_open", "output_to_source": {"scale": 1.0, "offset_s": start_s},
                        "source_to_output": {"scale": 1.0, "offset_s": -start_s}, "streams": stream_mapping},
            "output_sha256": output_hash, "output_bytes": destination.stat().st_size,
            "output_path": str(destination)}
        errors = validate_derived(record, source_record)
        if errors:
            raise ValueError("Derived validation failed: " + "; ".join(errors))
        return record
    finally:
        if temp_path.exists():
            temp_path.unlink()

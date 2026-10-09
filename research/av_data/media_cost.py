"""Context-local, public-safe observations of actual media subprocess work.

FFprobe emitted frames and FFmpeg verbose *input* decoder summaries are observed
work, never inferred from clip outputs or duration. Missing/partial observations
remain unknown. File sizes are repeated asset footprints, not bytes read; AVIO
bytes are separate reported counters, not physical disk/network I/O. This does
not measure decoder-internal discarded frames, hash reads, CPU, memory or GPU
work, so full_decode_work_accounting_ready deliberately remains false.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Iterator

_ACTIVE: ContextVar[tuple] = ContextVar("av_media_work_ledgers", default=())
_INPUT_STREAM = re.compile(r"Input stream #(\d+):(\d+) \((video|audio)\):([^\r\n]*)")
_DECODED = re.compile(r"(\d+) frames decoded")
_SAMPLES = re.compile(r"\((\d+) samples\)")
_ERRORS = re.compile(r"(\d+) decode errors")
_IO_READ = re.compile(r"Statistics: (\d+) bytes read")
_IO_WRITE = re.compile(r"Statistics: (\d+) bytes written")


def _operation(args: list[str]) -> tuple[str, str]:
    tool = Path(args[0]).name if args else "other"
    tool = tool if tool in {"ffprobe", "ffmpeg"} else "other"
    if "-version" in args:
        return tool, "tool_version"
    if tool == "ffprobe":
        return tool, "ffprobe_frame_scan" if "-show_frames" in args else "ffprobe_metadata_probe"
    if tool == "ffmpeg" and "-i" in args:
        return tool, "ffmpeg_decode_transform"
    return tool, "unclassified_media_operation"


def instrument_media_args(args: list[str]) -> list[str]:
    """Only diagnostics change; original filters, codecs and clock args stay exact."""
    if not _ACTIVE.get() or _operation(args)[1] != "ffmpeg_decode_transform":
        return args
    result = list(args)
    found = False
    for index, arg in enumerate(result[:-1]):
        if arg in {"-v", "-loglevel"}:
            result[index + 1] = "verbose"
            found = True
    if not found:
        result[1:1] = ["-loglevel", "verbose"]
    return result


def _file_size(path: str) -> int | None:
    try:
        file = Path(path)
        return file.stat().st_size if file.is_file() else None
    except (OSError, ValueError):
        return None


def _asset_bytes(args: list[str], operation: str) -> tuple[int | None, int | None]:
    if operation == "tool_version":
        return 0, 0
    if operation.startswith("ffprobe_"):
        return _file_size(args[-1]), 0
    if operation == "ffmpeg_decode_transform":
        sizes = [_file_size(args[i + 1]) for i, arg in enumerate(args[:-1]) if arg == "-i"]
        return (sum(sizes) if sizes and all(s is not None for s in sizes) else None,
                _file_size(args[-1]))
    return None, None


def _probe_observation(stdout: bytes) -> tuple[dict, bool]:
    values = {"observed_input_video_frames": None, "observed_input_audio_frames": None,
              "observed_input_audio_samples": None}
    try:
        data = json.loads(stdout)
        frames = data["frames"]
        if not isinstance(frames, list):
            return values, False
        video = audio = samples = 0
        complete = True
        for frame in frames:
            if not isinstance(frame, dict):
                complete = False
            elif frame.get("media_type") == "video":
                video += 1
            elif frame.get("media_type") == "audio":
                audio += 1
                count = frame.get("nb_samples")
                if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    samples += count
                else:
                    complete = False
            else:
                complete = False
        values.update(observed_input_video_frames=video, observed_input_audio_frames=audio,
                      observed_input_audio_samples=samples)
        return values, complete
    except (KeyError, TypeError, ValueError):
        return values, False


def _ffmpeg_observation(stderr: bytes) -> tuple[dict, bool]:
    values = {"observed_input_video_frames": None, "observed_input_audio_frames": None,
              "observed_input_audio_samples": None}
    text = stderr.decode("utf-8", errors="replace")
    streams = _INPUT_STREAM.findall(text)
    # Count only input stream summaries; output encoder/progress counters are excluded.
    if not streams or len({(i, s) for i, s, _, _ in streams}) != len(streams):
        return values, False
    video = audio = samples = 0
    measured_video = measured_audio = measured_samples = False
    kinds = {kind for _, _, kind, _ in streams}
    complete = True
    for _, _, kind, details in streams:
        decoded = _DECODED.search(details)
        errors = _ERRORS.search(details)
        if errors and int(errors[1]) != 0:
            complete = False
        if decoded is None:
            complete = False
            continue
        if kind == "video":
            video += int(decoded[1])
            measured_video = True
        else:
            audio += int(decoded[1])
            measured_audio = True
            sample_match = _SAMPLES.search(details)
            if sample_match is None:
                complete = False
            else:
                samples += int(sample_match[1])
                measured_samples = True
    values.update(observed_input_video_frames=video if measured_video or "video" not in kinds else None,
                  observed_input_audio_frames=audio if measured_audio or "audio" not in kinds else None,
                  observed_input_audio_samples=samples if measured_samples or "audio" not in kinds else None)
    return values, complete


def record_media_operation(args: list[str], *, stdout: bytes, stderr: bytes,
                           wall_seconds: float, status: str, returncode: int | None,
                           failure_kind: str | None) -> None:
    """Build an allowlisted event; raw args, paths and diagnostic text are not stored."""
    ledgers = _ACTIVE.get()
    if not ledgers:
        return
    tool, operation = _operation(args)
    input_bytes, output_bytes = _asset_bytes(args, operation)
    values = {"observed_input_video_frames": None, "observed_input_audio_frames": None,
              "observed_input_audio_samples": None}
    complete, basis = False, "unavailable"
    if operation == "ffprobe_frame_scan":
        values, complete = _probe_observation(stdout)
        basis = "ffprobe_emitted_frame_records"
    elif operation == "ffmpeg_decode_transform":
        values, complete = _ffmpeg_observation(stderr)
        basis = "ffmpeg_verbose_input_stream_summary"
    elif operation == "tool_version":
        values = dict.fromkeys(values, 0)
        complete, basis = True, "no_media_input"
    complete = complete and status == "success"
    text = stderr.decode("utf-8", errors="replace")
    read_bytes = [int(value) for value in _IO_READ.findall(text)]
    written_bytes = [int(value) for value in _IO_WRITE.findall(text)]
    event = {"tool": tool, "operation": operation, "status": status,
             "returncode": returncode, "failure_kind": failure_kind,
             "wall_seconds": wall_seconds, "stdout_bytes": len(stdout),
             "stderr_bytes": len(stderr), "input_asset_bytes": input_bytes,
             "output_asset_bytes": output_bytes, "subprocess_cache_hit": False,
             "decode_measurement_basis": basis, **values,
             "decode_observation_complete": complete,
             "unknown_partial_work": not complete,
             "observed_io_read_bytes": sum(read_bytes) if read_bytes else None,
             "observed_io_write_bytes": sum(written_bytes) if written_bytes else None}
    for ledger in ledgers:
        ledger._append(event)


class MediaWorkLedger:
    """Append-only collector for one scope, including nested helper scopes."""
    def __init__(self) -> None:
        self._events: list[dict] = []
        self._lock = Lock()

    def _append(self, event: dict) -> None:
        with self._lock:
            self._events.append(deepcopy(event))

    def public_summary(self) -> dict:
        with self._lock:
            events = deepcopy(self._events)
        scans = [e for e in events if e["operation"] == "ffprobe_frame_scan"]
        decodes = [e for e in events if e["operation"] == "ffmpeg_decode_transform"]
        def total(key, group=events):
            return sum(e[key] for e in group if e[key] is not None)
        summary = {"schema_version": "media-work-ledger-0.1",
                   "scope": "instrumented_run_media_subprocesses",
                   "operation_count": len(events),
                   "successful_operations": sum(e["status"] == "success" for e in events),
                   "failed_operations": sum(e["status"] != "success" for e in events),
                   "wall_seconds": total("wall_seconds"),
                   "stdout_bytes": total("stdout_bytes"), "stderr_bytes": total("stderr_bytes"),
                   "input_asset_bytes": total("input_asset_bytes"),
                   "output_asset_bytes": total("output_asset_bytes"),
                   "asset_bytes_basis": "sum_of_per_operation_file_sizes_not_io",
                   "observed_io_read_bytes": total("observed_io_read_bytes"),
                   "observed_io_write_bytes": total("observed_io_write_bytes"),
                   "io_bytes_basis": "ffmpeg_AVIO_reported_counters_not_physical_io",
                   "unknown_io_read_operations": sum(e["observed_io_read_bytes"] is None for e in events if e["operation"] != "tool_version"),
                   "unknown_input_asset_bytes_operations": sum(e["input_asset_bytes"] is None for e in events),
                   "subprocess_cache_hits": total("subprocess_cache_hit"),
                   "ffprobe_scan_operations": len(scans), "ffmpeg_decode_operations": len(decodes),
                   "ffprobe_observation_complete": all(e["decode_observation_complete"] for e in scans),
                   "ffmpeg_input_decode_observation_complete": all(e["decode_observation_complete"] for e in decodes),
                   "unknown_partial_work_operations": total("unknown_partial_work"),
                   "full_decode_work_accounting_ready": False,
                   "observed_totals_are_lower_bounds": True,
                   "unmeasured_work": ["decoder_internal_discarded_work", "hash_copy_and_physical_io", "cpu_memory_gpu_resources"],
                   "events": events}
        for prefix, group in (("observed_ffprobe", scans), ("observed_ffmpeg_input", decodes)):
            for suffix in ("video_frames", "audio_frames", "audio_samples"):
                summary[f"{prefix}_{suffix}"] = total(f"observed_input_{suffix}", group)
        return summary


@contextmanager
def collect_media_work() -> Iterator[MediaWorkLedger]:
    """Capture this context's runner calls; concurrent contexts retain their own ledger."""
    ledger = MediaWorkLedger()
    token = _ACTIVE.set((*_ACTIVE.get(), ledger))
    try:
        yield ledger
    finally:
        _ACTIVE.reset(token)

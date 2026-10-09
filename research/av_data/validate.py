"""Fail-closed source and derived integrity/clock validation.

Validators return error strings (empty list means the tested contract passed).
Byte/timing validation establishes replay integrity, not semantic evidence truth.
"""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path

from .media_clock import clock_specs, probe_media, sha256_file
from .schemas import SOURCE_SCHEMA_VERSION, DERIVED_SCHEMA_VERSION

_HASH = re.compile(r"^[0-9a-f]{64}$")


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _asset_errors(path, digest, size, verify_hash=True):
    errors = []
    if not isinstance(path, str) or not path:
        return ["asset path is missing"]
    asset = Path(path)
    if not asset.is_file():
        return ["asset does not exist"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or asset.stat().st_size != size:
        errors.append("asset byte count mismatch")
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        errors.append("asset sha256 is invalid")
    elif verify_hash and sha256_file(asset) != digest:
        errors.append("asset sha256 mismatch")
    return errors


def validate_source(record, verify_hash=True) -> list[str]:
    if not isinstance(record, dict):
        return ["SourceRecord must be a JSON object"]
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError):
        return ["SourceRecord must contain finite JSON values"]
    required = {"schema_version", "source_id", "origin_url", "fetched_at", "media_type", "asset_bytes",
                "duration_s", "asset_sha256", "asset_path", "rights_basis", "redistribution_scope",
                "public_metadata", "stream_specs", "timestamp_origin", "access_status", "created_by", "probe_tool_version"}
    errors = [f"missing source field: {k}" for k in sorted(required-record.keys())]
    if errors:
        return errors
    if record["schema_version"] != SOURCE_SCHEMA_VERSION:
        errors.append("unsupported source schema_version")
    digest = record["asset_sha256"]
    if not isinstance(digest, str) or record["source_id"] != f"src_{digest[:24]}":
        errors.append("source_id does not match opaque content identity")
    if record["access_status"] != "available_local":
        errors.append("source asset is not available_local")
    for key in ("origin_url", "fetched_at", "rights_basis", "probe_tool_version"):
        if not isinstance(record[key], str) or not record[key].strip():
            errors.append(f"invalid source {key}")
    if not isinstance(record["redistribution_scope"], str) or record["redistribution_scope"] not in {"metadata_only", "annotation_only", "full_asset", "unknown"}:
        errors.append("unsupported redistribution_scope")
    if not isinstance(record["public_metadata"], dict):
        errors.append("public_metadata is not an object")
    if not _finite(record["duration_s"]) or record["duration_s"] <= 0:
        errors.append("invalid source presentation duration")
    origin = record["timestamp_origin"]
    if not isinstance(origin, dict) or origin.get("axis") != "source_local_presentation" or not _finite(origin.get("container_start_time_s")):
        return errors + ["invalid timestamp_origin"]
    streams = record["stream_specs"]
    if not isinstance(streams, list) or not streams:
        return errors + ["source has no stream_specs"]
    indices = []
    kinds = set()
    latest_end = 0.0
    for stream in streams:
        if not isinstance(stream, dict):
            errors.append("stream spec is not an object")
            continue
        index = stream.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            errors.append("stream index must be a nonnegative integer")
            continue
        indices.append(index)
        kind = stream.get("codec_type")
        if kind not in ("audio", "video"):
            errors.append("unsupported timed stream type")
            continue
        kinds.add(kind)
        try:
            base = Fraction(stream.get("time_base", "0/1"))
            if base <= 0:
                raise ValueError()
        except (ValueError, TypeError, ZeroDivisionError):
            errors.append(f"stream {index} invalid time_base")
            continue
        frames = stream.get("frame_pts")
        if not isinstance(frames, list) or not frames:
            errors.append(f"stream {index} missing original frame_pts")
            continue
        if not all(isinstance(frame, dict) for frame in frames):
            errors.append(f"stream {index} malformed original frame PTS")
            continue
        previous = None
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("pts"), int) or not _finite(frame.get("source_time_s")) or not _finite(frame.get("duration_s")) or frame["duration_s"] <= 0:
                errors.append(f"stream {index} malformed original frame PTS")
                break
            at = frame["source_time_s"]
            if abs(float(frame["pts"]*base)-origin["container_start_time_s"]-at) > 1e-8:
                errors.append(f"stream {index} original PTS/source mapping mismatch")
                break
            if previous is not None and at <= previous:
                errors.append(f"stream {index} nonmonotonic original PTS")
                break
            previous = at
            latest_end = max(latest_end, at+frame["duration_s"])
        if not _finite(stream.get("source_offset_s")) or stream["source_offset_s"] != frames[0].get("source_time_s"):
            errors.append(f"stream {index} offset does not match original first PTS")
        if kind == "audio":
            samples = stream.get("sample_mapping")
            if not isinstance(samples, list) or len(samples) != len(frames):
                errors.append(f"stream {index} missing audio sample mapping")
            else:
                cursor = 0
                for sample, frame in zip(samples, frames):
                    if (not isinstance(sample, dict) or sample.get("sample_start") != cursor or
                        not isinstance(sample.get("sample_count"), int) or sample["sample_count"] <= 0 or
                        not isinstance(sample.get("sample_rate"), int) or sample["sample_rate"] <= 0 or
                        sample.get("pts") != frame["pts"] or sample.get("source_start_s") != frame["source_time_s"] or
                        not _finite(sample.get("seconds_per_sample")) or
                        abs(sample["seconds_per_sample"]-1.0/sample["sample_rate"]) > 1e-12 or
                        abs(frame["duration_s"]-sample["sample_count"]/sample["sample_rate"]) > 1e-9):
                        errors.append(f"stream {index} invalid sample-to-source mapping")
                        break
                    cursor += sample["sample_count"]
    if len(set(indices)) != len(indices):
        errors.append("duplicate source stream indices")
    expected_type = "audio_video" if kinds == {"audio", "video"} else next(iter(kinds), None)
    if record["media_type"] != expected_type:
        errors.append("media_type does not match original AV streams")
    if _finite(record["duration_s"]) and abs(latest_end-record["duration_s"]) > 1e-7:
        errors.append("duration_s does not match original presentation end")
    errors += _asset_errors(record["asset_path"], digest, record["asset_bytes"], verify_hash)
    if verify_hash and not errors:
        try:
            actual_origin, actual_streams, actual_duration = clock_specs(probe_media(record["asset_path"]))
            if origin != actual_origin or streams != actual_streams or abs(record["duration_s"]-actual_duration) > 1e-9:
                errors.append("source clock manifest disagrees with original asset bytes")
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"original clock probe failed: {exc}")
    return errors


def validate_derived(record, parent) -> list[str]:
    if not isinstance(record, dict):
        return ["DerivedRecord must be a JSON object"]
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError):
        return ["DerivedRecord must contain finite JSON values"]
    errors = [f"invalid parent: {e}" for e in validate_source(parent)]
    required = {"schema_version", "derived_id", "parent_source_id", "parent_sha256", "created_at",
                "created_by", "transform", "tool_version", "original_interval", "mapping",
                "output_sha256", "output_bytes", "output_path"}
    errors += [f"missing derived field: {k}" for k in sorted(required-record.keys())]
    if errors:
        return errors
    if record["schema_version"] != DERIVED_SCHEMA_VERSION:
        errors.append("unsupported derived schema_version")
    if record["parent_sha256"] != parent["asset_sha256"] or record["parent_source_id"] != parent["source_id"]:
        errors.append("derived parent identity mismatch")
    digest = record["output_sha256"]
    if not isinstance(digest, str) or record["derived_id"] != f"derived-{digest[:24]}":
        errors.append("derived_id does not match output identity")
    interval = record["original_interval"]
    if not isinstance(interval, dict):
        return errors + ["invalid original_interval"]
    start, end = interval.get("start_s"), interval.get("end_s")
    if not _finite(start) or not _finite(end) or not 0 <= start < end <= parent["duration_s"]+1e-9:
        return errors + ["original_interval is outside parent source clock"]
    mapping = record["mapping"]
    if (not isinstance(mapping, dict) or mapping.get("axis") != "source_local_presentation" or
        mapping.get("output_axis") != "presentation_pts_seconds" or mapping.get("interval") != "half_open" or
        mapping.get("output_to_source") != {"scale": 1.0, "offset_s": start} or
        mapping.get("source_to_output") != {"scale": 1.0, "offset_s": -start}):
        return errors + ["invalid reversible source/output mapping"]
    stream_map = mapping.get("streams")
    specs = parent["stream_specs"]
    if not isinstance(stream_map, list) or len(stream_map) != len(specs):
        return errors + ["derived mapping did not retain every original AV stream"]
    for output_index, (entry, spec) in enumerate(zip(stream_map, specs)):
        if (not isinstance(entry, dict) or entry.get("source_stream_index") != spec["index"] or
            entry.get("output_stream_index") != output_index or entry.get("codec_type") != spec["codec_type"] or
            entry.get("source_time_base") != spec["time_base"] or entry.get("source_offset_s") != spec["source_offset_s"]):
            errors.append("derived stream identity/offset mismatch")
    errors += _asset_errors(record["output_path"], digest, record["output_bytes"])
    if errors:
        return errors
    try:
        actual = probe_media(record["output_path"])
        actual_streams = actual.get("streams", [])
        if [s["codec_type"] for s in actual_streams] != [s["codec_type"] for s in specs]:
            errors.append("output AV streams do not match parent")
        for entry, actual_stream in zip(stream_map, actual_streams):
            if entry.get("output_time_base") != actual_stream.get("time_base") or abs(entry.get("output_start_time_s", math.inf)-float(actual_stream.get("start_time", 0))) > 1e-7:
                errors.append("derived stream clock disagrees with actual bytes")
        for frame in actual.get("frames", []):
            time = float(frame.get("pts_time", frame.get("best_effort_timestamp_time", "nan")))
            if not math.isfinite(time) or time < -.002 or time+start >= end+.002:
                errors.append("output frame PTS outside mapped original_interval")
                break
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"output clock probe failed: {exc}")
    return errors

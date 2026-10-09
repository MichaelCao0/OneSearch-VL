"""Ingest immutable original media without re-timing or stripping audio."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .media_clock import clock_specs, probe_media, sha256_file, tool_version, utc_now
from .schemas import SOURCE_SCHEMA_VERSION, TOOL_REVISION


def ingest_media(local_path, origin_url, asset_root, public_metadata=None,
                 rights_basis="unknown", redistribution_scope="metadata_only") -> dict:
    """Copy exact bytes to a content-addressed store and return a JSON SourceRecord.

    No network acquisition or rights inference occurs. public_metadata must be
    actual supplied public metadata; private actor export is a separate contract.
    """
    source = Path(local_path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("Original local media is absent or empty")
    if not isinstance(origin_url, str) or not origin_url.strip():
        raise ValueError("origin_url must identify the supplied source")
    if not isinstance(rights_basis, str) or not rights_basis.strip():
        raise ValueError("rights_basis must be explicit, using 'unknown' when unverified")
    if redistribution_scope not in {"metadata_only", "annotation_only", "full_asset", "unknown"}:
        raise ValueError("Unsupported redistribution_scope")
    metadata = {} if public_metadata is None else public_metadata
    if not isinstance(metadata, dict):
        raise ValueError("public_metadata must be a JSON object")
    try:
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("public_metadata must contain finite JSON values") from exc
    digest = sha256_file(source)
    root = Path(asset_root).resolve()
    directory = root / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    # Do not retain URL/title/private labels in filesystem names.
    suffix = source.suffix.lower()
    if not suffix or not suffix[1:].isalnum() or len(suffix) > 12:
        suffix = ".media"
    destination = directory / f"{digest}{suffix}"
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError("Existing content-addressed original asset is corrupt")
    else:
        descriptor, name = tempfile.mkstemp(prefix=".ingest-", dir=directory)
        os.close(descriptor)
        tmp = Path(name)
        try:
            shutil.copyfile(source, tmp)
            if sha256_file(tmp) != digest:
                raise ValueError("Original asset changed while ingesting")
            try:
                os.link(tmp, destination)
            except FileExistsError:
                if sha256_file(destination) != digest:
                    raise ValueError("Concurrent content-addressed original asset is corrupt")
        finally:
            tmp.unlink(missing_ok=True)
    data = probe_media(destination)
    origin, specs, duration = clock_specs(data)
    kinds = {s["codec_type"] for s in specs}
    record = {"schema_version": SOURCE_SCHEMA_VERSION, "source_id": f"src_{digest[:24]}",
        "origin_url": origin_url, "fetched_at": utc_now(),
        "media_type": "audio_video" if kinds == {"audio", "video"} else next(iter(kinds)),
        "asset_bytes": destination.stat().st_size, "duration_s": duration,
        "asset_sha256": digest, "asset_path": str(destination),
        "rights_basis": rights_basis, "redistribution_scope": redistribution_scope,
        "public_metadata": metadata, "stream_specs": specs, "timestamp_origin": origin,
        "access_status": "available_local", "created_by": {"tool": "ingest_media", "revision": TOOL_REVISION},
        "probe_tool_version": tool_version("ffprobe")}
    from .validate import validate_source
    errors = validate_source(record)
    if errors:
        raise ValueError("Invalid ingested source: " + "; ".join(errors))
    return record

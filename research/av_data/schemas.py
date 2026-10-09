"""JSON contracts for immutable source assets and reversible derived views.

Source times are original presentation PTS minus the explicit container origin.
Stream start offsets and VFR frame PTS are data, never an alignment instruction.
"""
from typing import Any, TypedDict

SOURCE_SCHEMA_VERSION = "av-deepresearch.source.v1"
DERIVED_SCHEMA_VERSION = "av-deepresearch.derived.v1"
TOOL_REVISION = "phase1-v1"


class SourceRecord(TypedDict):
    schema_version: str
    source_id: str  # opaque src_<24 lowercase hex> from original asset SHA-256
    origin_url: str
    fetched_at: str
    media_type: str
    asset_bytes: int
    duration_s: float
    asset_sha256: str
    asset_path: str
    rights_basis: str
    redistribution_scope: str
    public_metadata: dict[str, Any]
    stream_specs: list[dict[str, Any]]
    timestamp_origin: dict[str, Any]
    access_status: str
    created_by: dict[str, str]
    probe_tool_version: str


class DerivedRecord(TypedDict):
    schema_version: str
    derived_id: str
    parent_source_id: str
    parent_sha256: str
    created_at: str
    transform: dict[str, Any]
    tool_version: str
    original_interval: dict[str, float]
    mapping: dict[str, Any]
    output_sha256: str
    output_bytes: int
    output_path: str
    created_by: dict[str, str]

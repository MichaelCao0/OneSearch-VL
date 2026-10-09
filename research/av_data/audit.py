"""Actor visibility export and private split auditing."""
from __future__ import annotations

from .split import _digest, _validate_components
from .tasks import SPLITS, _json_copy, _opaque, reject_private_fields, validate_task_public


_PUBLIC_SOURCE_FIELDS = frozenset({
    "source_id", "schema_version", "created_by", "revision", "origin_url", "fetched_at",
    "media_type", "asset_bytes", "duration_s", "asset_sha256", "rights_basis",
    "redistribution_scope", "public_metadata", "stream_specs", "timestamp_origin",
    "access_status", "probe_tool_version", "corpus_id", "split", "text", "text_kind",
    "text_sha256", "body_text",
})


def export_actor_task(task, visible_sources):
    """Validate and deep-copy only TaskPublic for the current episode.

    visible_sources is a source_id-keyed mapping or list of already-public source
    records carrying corpus_id and split, or a frozen corpus scope envelope
    {corpus_id, split, sources} returned by visible_corpus. Internal asset paths,
    event/recording IDs, gold labels, hidden relations and future observations
    cause a failure before export. Public metadata prose is left intact.
    """
    task = _json_copy(task)
    visible_sources = _json_copy(visible_sources)
    if not isinstance(task, dict):
        raise ValueError("Task must be an object")
    envelope = None
    if isinstance(visible_sources, dict) and "sources" in visible_sources:
        if set(visible_sources) != {"corpus_id", "split", "sources"}:
            raise ValueError("Unexpected frozen corpus scope fields")
        envelope = visible_sources
        if (envelope["corpus_id"] != task.get("corpus_id") or
                envelope["split"] != task.get("split")):
            raise ValueError("Frozen corpus scope belongs to another episode")
        visible_sources = envelope["sources"]
    if isinstance(visible_sources, list):
        registry = {}
        for source in visible_sources:
            if not isinstance(source, dict):
                raise ValueError("Visible source must be an object")
            source_id = _opaque(source.get("source_id"))
            if source_id in registry:
                raise ValueError("Duplicate visible source")
            registry[source_id] = source
    elif isinstance(visible_sources, dict):
        registry = visible_sources
    else:
        raise ValueError("visible_sources must be public source records")
    allowed = set(registry)
    for source_id, source in registry.items():
        _opaque(source_id)
        if not isinstance(source, dict) or source.get("source_id") != source_id:
            raise ValueError("Visible source key/source_id mismatch")
        if set(source) - _PUBLIC_SOURCE_FIELDS:
            raise ValueError("Unclassified/private field in visible source")
        source_corpus = source.get("corpus_id", envelope["corpus_id"] if envelope else None)
        source_split = source.get("split", envelope["split"] if envelope else None)
        if source_corpus != task.get("corpus_id") or source_split != task.get("split"):
            raise ValueError("Visible source belongs to another episode corpus/split")
        reject_private_fields(source, allowed)
    return validate_task_public(task, allowed, task.get("corpus_id"), task.get("split"))


def audit_split_manifest(components, manifest):
    """Reject altered frozen manifests, split clusters and exploration in test."""
    components = _validate_components(components)
    manifest = _json_copy(manifest)
    if not isinstance(manifest, dict) or manifest.get("frozen") is not True:
        raise ValueError("Split manifest must be frozen")
    source_splits = manifest.get("source_splits")
    rows = manifest.get("component_splits")
    if not isinstance(source_splits, dict) or not isinstance(rows, list):
        raise ValueError("Split manifest lacks assignments")
    expected_sources = {source for component in components for source in component["source_ids"]}
    if set(source_splits) != expected_sources:
        raise ValueError("Split source coverage differs from components")
    by_component = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"component_id", "source_ids", "split"}:
            raise ValueError("Invalid component assignment")
        identifier = row["component_id"]
        if identifier in by_component:
            raise ValueError("Duplicate component assignment")
        by_component[identifier] = row
    if set(by_component) != {component["component_id"] for component in components}:
        raise ValueError("Split component coverage differs")
    for component in components:
        names = {source_splits[source] for source in component["source_ids"]}
        if len(names) != 1 or not names <= SPLITS:
            raise ValueError("Same recording/event component crosses splits")
        name = next(iter(names))
        if component["exploration"] and name != "exploration":
            raise ValueError("Exploration cannot enter evaluation or training splits")
        row = by_component[component["component_id"]]
        if sorted(row["source_ids"]) != sorted(component["source_ids"]) or row["split"] != name:
            raise ValueError("Source and component assignment disagree")
    payload = dict(manifest)
    revision = payload.pop("revision", None)
    if not isinstance(revision, str) or _digest(payload) != revision:
        raise ValueError("Frozen split revision no longer matches content")
    return manifest

"""Private claim and lineage contracts; these records must never enter actor export."""
from __future__ import annotations

import re
from collections.abc import Mapping

from .tasks import _finite, _identity, _json_copy, _nonempty, _opaque

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RELATION_TYPES = frozenset({
    "url", "uploader", "recording", "event", "independent_observation",
    "near_duplicate", "lineage", "shared_knowledge",
})
RELATION_STATUSES = frozenset({"same", "different", "independent", "dependent", "unknown"})


def _registry(source_registry):
    if not isinstance(source_registry, Mapping):
        raise ValueError("source_registry must map source_id to records")
    for source_id, source in source_registry.items():
        _opaque(source_id)
        if not isinstance(source, dict) or source.get("source_id") != source_id:
            raise ValueError("Registry key/source_id mismatch")
    return source_registry


def _evidence(reference, registry):
    if not isinstance(reference, dict):
        raise ValueError("Evidence reference must be an object")
    if set(reference) not in (
            {"source_id", "asset_sha256", "interval"},
            {"source_id", "asset_sha256", "quote"}):
        raise ValueError("Evidence requires exactly source/hash and interval or quote")
    source_id = _opaque(reference["source_id"])
    if source_id not in registry:
        raise ValueError("Evidence source does not exist")
    source = registry[source_id]
    sha = reference["asset_sha256"]
    if not isinstance(sha, str) or not _SHA256.fullmatch(sha) or sha != source.get("asset_sha256"):
        raise ValueError("Evidence hash does not match registered source version")
    if "interval" in reference:
        interval = reference["interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Evidence interval must be [start_s,end_s)")
        start, end = (_finite(x, "interval") for x in interval)
        duration = _finite(source.get("duration_s"), "source duration_s")
        if not 0 <= start < end <= duration:
            raise ValueError("Evidence interval is empty or outside source clock")
    else:
        quote = reference["quote"]
        if not isinstance(quote, dict) or set(quote) != {"start_char", "end_char", "text"}:
            raise ValueError("Quote requires character bounds and exact text")
        start, end = quote["start_char"], quote["end_char"]
        body = source.get("body_text")
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or
                not isinstance(end, int) or not isinstance(body, str) or
                not 0 <= start < end <= len(body) or body[start:end] != quote["text"]):
            raise ValueError("Quote is not the exact registered source text")
    return reference


def validate_claim_gold(gold, task_id, source_registry):
    """Validate AND evidence packages inside an OR list, plus private bindings.

    Validation checks registered version/clock/quote identity, not semantic truth.
    Unknown/undetermined is confined to within_snapshot and needs cited context
    and an explicit missing-evidence explanation. Independent human adjudication
    is still required before these records can be accepted as gold.
    """
    gold = _json_copy(gold)
    fields = {
        "schema_version", "created_by", "revision", "task_id", "claim_id", "proposition",
        "expected_decision", "status", "qualifiers", "valid_evidence_sets",
        "required_bindings", "missing_evidence", "scope",
    }
    if not isinstance(gold, dict) or set(gold) != fields:
        raise ValueError("ClaimGold has missing or unexpected fields")
    _identity(gold)
    for key in ("task_id", "claim_id", "proposition", "expected_decision", "scope"):
        _nonempty(gold[key], key)
    if gold["task_id"] != task_id:
        raise ValueError("Gold belongs to another task")
    if gold["status"] not in {"supported", "contradicted", "undetermined"}:
        raise ValueError("Invalid claim status")
    if not isinstance(gold["qualifiers"], dict):
        raise ValueError("qualifiers must be an object")
    missing = gold["missing_evidence"]
    if not isinstance(missing, list) or any(not isinstance(x, str) or not x.strip() for x in missing):
        raise ValueError("missing_evidence must be a list of explanations")
    if gold["status"] == "undetermined" or gold["expected_decision"] == "unknown":
        if gold["scope"] != "within_snapshot" or not missing:
            raise ValueError("Unknown needs within_snapshot scope and missing evidence")
    registry = _registry(source_registry)
    packages = gold["valid_evidence_sets"]
    if not isinstance(packages, list) or not packages:
        raise ValueError("At least one OR evidence package is required")
    for package in packages:
        if not isinstance(package, list) or not package:
            raise ValueError("An AND evidence package cannot be empty")
        for reference in package:
            _evidence(reference, registry)
    bindings = gold["required_bindings"]
    if not isinstance(bindings, list):
        raise ValueError("required_bindings must be a list")
    for relation in bindings:
        validate_lineage(relation, registry)
    return gold


def validate_lineage(lineage, source_registry):
    """Validate one private source relation without assuming independence.

    recording/event describe different relations from URL or uploader identity.
    relation_status=unknown is explicit; any decided relationship needs basis
    references and an adjudication explanation. A fingerprint reference can be a
    basis for manual review but this validator does not turn it into semantic
    proof of an event, independence or causal attribution.
    """
    relation = _json_copy(lineage)
    required = {
        "schema_version", "created_by", "revision", "source_ids", "relation_type",
        "relation_status", "basis_refs", "adjudication",
    }
    optional = {"recording_id", "event_id", "relation_id"}
    if not isinstance(relation, dict) or not required <= set(relation) or set(relation) - required - optional:
        raise ValueError("LineageGold has missing or unexpected fields")
    _identity(relation)
    registry = _registry(source_registry)
    source_ids = relation["source_ids"]
    if (not isinstance(source_ids, list) or len(source_ids) < 2 or
            any(not isinstance(x, str) for x in source_ids) or len(set(source_ids)) != len(source_ids)):
        raise ValueError("Lineage requires distinct sources")
    for source_id in source_ids:
        _opaque(source_id)
        if source_id not in registry:
            raise ValueError("Lineage source does not exist")
    if relation["relation_type"] not in RELATION_TYPES or relation["relation_status"] not in RELATION_STATUSES:
        raise ValueError("Invalid or implicit lineage relationship")
    if (relation["relation_status"] in {"independent", "dependent"} and
            relation["relation_type"] != "independent_observation"):
        raise ValueError("Independence is a separate relation dimension")
    _nonempty(relation["adjudication"], "adjudication")
    refs = relation["basis_refs"]
    if not isinstance(refs, list):
        raise ValueError("basis_refs must be a list")
    if relation["relation_status"] != "unknown" and not refs:
        raise ValueError("A decided relation needs auditable basis references")
    for reference in refs:
        _evidence(reference, registry)
    for field in optional & set(relation):
        _nonempty(relation[field], field)
    return relation

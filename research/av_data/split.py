"""Private conservative event/recording components and reproducible split freezing."""
from __future__ import annotations

import hashlib
import json
import math
import random

from .gold import RELATION_STATUSES, RELATION_TYPES
from .tasks import SCHEMA_VERSION, SPLITS, _finite, _json_copy, _nonempty, _opaque

_CONNECTING = frozenset({"recording", "event", "near_duplicate", "lineage"})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_lineage_components(records):
    """Build private connected components before questions/trajectories exist.

    records are SourceRecord sidecars with optional recording_id/event_id,
    near_duplicate_ids, lineage_links=[{source_id, relation_type, relation_status}],
    record_role=event_evidence|shared_knowledge and stage/split=exploration.
    Same recording/event, duplicate candidates and unresolved explicit event
    links conservatively stay together. Generic knowledge pages, URL/uploader
    equality and independence claims do not connect event samples.
    """
    records = _json_copy(records)
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    by_id = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Lineage source record must be an object")
        source_id = _opaque(record.get("source_id"))
        if source_id in by_id:
            raise ValueError("Duplicate source_id")
        if record.get("record_role", "event_evidence") not in {"event_evidence", "shared_knowledge"}:
            raise ValueError("Unknown record_role")
        by_id[source_id] = record
    parent = {source_id: source_id for source_id in by_id}
    def find(source_id):
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id
    def join(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    def event_source(source_id):
        return by_id[source_id].get("record_role", "event_evidence") != "shared_knowledge"
    recording_groups, event_groups = {}, {}
    for source_id, record in by_id.items():
        if event_source(source_id):
            for field, groups in (("recording_id", recording_groups), ("event_id", event_groups)):
                identifier = record.get(field)
                if identifier is not None:
                    _nonempty(identifier, field)
                    if identifier in groups:
                        join(source_id, groups[identifier])
                    else:
                        groups[identifier] = source_id
        near = record.get("near_duplicate_ids", [])
        links = record.get("lineage_links", [])
        if not isinstance(near, list) or not isinstance(links, list):
            raise ValueError("Lineage links and duplicate candidates must be lists")
        for target in near:
            _opaque(target)
            if target not in by_id:
                raise ValueError("Dangling near-duplicate source")
            if event_source(source_id) and event_source(target):
                join(source_id, target)
        for relation in links:
            if not isinstance(relation, dict) or set(relation) != {"source_id", "relation_type", "relation_status"}:
                raise ValueError("Malformed lineage sidecar link")
            target = _opaque(relation["source_id"])
            if target not in by_id:
                raise ValueError("Dangling lineage link")
            kind, status = relation["relation_type"], relation["relation_status"]
            if kind not in RELATION_TYPES or status not in RELATION_STATUSES:
                raise ValueError("Unknown lineage relation")
            if (kind in _CONNECTING and status in {"same", "unknown", "dependent"} and
                    event_source(source_id) and event_source(target)):
                join(source_id, target)
    groups = {}
    for source_id in sorted(by_id):
        groups.setdefault(find(source_id), []).append(source_id)
    components = []
    for source_ids in groups.values():
        component = {
            "component_id": "cmp_" + _digest(source_ids)[:24],
            "source_ids": source_ids,
            "exploration": any(by_id[x].get("stage") == "exploration" or
                               by_id[x].get("split") == "exploration" for x in source_ids),
            "recording_ids": sorted({by_id[x]["recording_id"] for x in source_ids
                                     if event_source(x) and by_id[x].get("recording_id") is not None}),
            "event_ids": sorted({by_id[x]["event_id"] for x in source_ids
                                 if event_source(x) and by_id[x].get("event_id") is not None}),
        }
        components.append(component)
    return sorted(components, key=lambda component: component["component_id"])


def _validate_components(components):
    components = _json_copy(components)
    if not isinstance(components, list):
        raise ValueError("components must be a list")
    ids, sources = set(), set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("Component must be an object")
        identifier = _nonempty(component.get("component_id"), "component_id")
        members = component.get("source_ids")
        if identifier in ids or not isinstance(members, list) or not members:
            raise ValueError("Duplicate or empty component")
        ids.add(identifier)
        for member in members:
            _opaque(member)
            if member in sources:
                raise ValueError("Source appears in more than one component")
            sources.add(member)
        if not isinstance(component.get("exploration"), bool):
            raise ValueError("Component exploration flag must be explicit")
    return sorted(components, key=lambda component: component["component_id"])


def assign_event_splits(components, seed, ratios):
    """Return a frozen JSON manifest; allocate whole components by ratios.

    Exploration-tagged components are forced to exploration before ratio
    allocation. Largest remainder counts are based on components, never source
    or question counts. Input order cannot change the assignment.
    """
    components = _validate_components(components)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    ratios = _json_copy(ratios)
    if not isinstance(ratios, dict) or not ratios or set(ratios) - SPLITS:
        raise ValueError("Ratios must name known splits")
    for value in ratios.values():
        if _finite(value, "ratio") < 0:
            raise ValueError("Ratios cannot be negative")
    if not math.isclose(sum(ratios.values()), 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")
    names = sorted(ratios)
    eligible = [component for component in components if not component["exploration"]]
    random.Random(seed).shuffle(eligible)
    quotas = {name: len(eligible) * ratios[name] for name in names}
    counts = {name: math.floor(quotas[name]) for name in names}
    remainder = len(eligible) - sum(counts.values())
    order = sorted(names, key=lambda name: (-(quotas[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    assignments = {component["component_id"]: "exploration" for component in components if component["exploration"]}
    offset = 0
    for name in names:
        for component in eligible[offset:offset + counts[name]]:
            assignments[component["component_id"]] = name
        offset += counts[name]
    source_splits = {}
    rows = []
    for component in components:
        name = assignments[component["component_id"]]
        rows.append({"component_id": component["component_id"], "source_ids": sorted(component["source_ids"]), "split": name})
        for source_id in component["source_ids"]:
            source_splits[source_id] = name
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_by": "research.av_data.split",
        "seed": seed, "ratios": {name: ratios[name] for name in names},
        "frozen": True, "component_splits": rows,
        "source_splits": dict(sorted(source_splits.items())),
    }
    manifest["revision"] = _digest(manifest)
    return manifest

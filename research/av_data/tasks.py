"""JSON TaskPublic contract. Private annotations are never silently stripped."""
from __future__ import annotations

import json
import math
import re

SCHEMA_VERSION = "1.0"
SPLITS = frozenset({"exploration", "train", "dev", "test", "iid_test", "ood_test", "ood"})
_OPAQUE_SOURCE = re.compile(r"src_[0-9a-f]{16,64}\Z")
_SOURCE_TOKEN = re.compile(r"\bsrc_[A-Za-z0-9_-]+")
_PRIVATE_KEYS = frozenset({
    "answer", "answers", "expected_decision", "valid_evidence_sets", "evidence_sets",
    "required_bindings", "hidden_bindings", "bindings", "binding", "gold", "claim_gold",
    "lineage_gold", "private", "private_path", "asset_path", "asset_root", "source_path",
    "recording_id", "event_id", "lineage_links", "near_duplicate_ids", "independent_observation",
    "adjudication", "review_summary", "reviewer_summary", "annotation_summary",
    "future_observation", "future_observations", "missing_evidence", "effective_sources",
    "valid_sources", "gold_sources", "modality_requirement", "label", "labels",
})
_TASK_FIELDS = frozenset({
    "schema_version", "created_by", "revision", "task_id", "question", "target_questions",
    "initial_sources", "corpus_id", "budget_recipe", "split",
})


def _json_copy(value):
    """Return an independent JSON value, rejecting non-JSON keys/types and NaN."""
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("Expected JSON values, not " + type(item).__name__)
    check(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Record must be finite JSON") from exc


def _nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + " must be a nonempty string")
    return value


def _opaque(source_id):
    if not isinstance(source_id, str) or not _OPAQUE_SOURCE.fullmatch(source_id):
        raise ValueError("source_id must be opaque src_<16-64 lowercase hex>, not encode answers")
    return source_id


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(field + " must be a finite number")
    return value


def _identity(record):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported schema_version")
    for key in ("created_by", "revision"):
        _nonempty(record.get(key), key)


def reject_private_fields(value, allowed_source_ids=None):
    """Recursively inspect public JSON; actual prose and titles remain unchanged."""
    allowed = None if allowed_source_ids is None else set(allowed_source_ids)
    def walk(item, path):
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = key.casefold().replace("-", "_")
                compact = re.sub(r"[^a-z0-9]", "", normalized)
                if (normalized in _PRIVATE_KEYS or
                        any(token in compact for token in ("gold", "evidence", "binding", "private")) or
                        compact.startswith(("hidden", "future", "answer"))):
                    raise ValueError("Private field in actor payload: " + path + "." + key)
                walk(child, path + "." + key)
        elif isinstance(item, list):
            for number, child in enumerate(item):
                walk(child, path + "[" + str(number) + "]")
        elif isinstance(item, str) and allowed is not None:
            for token in _SOURCE_TOKEN.findall(item):
                _opaque(token)
                if token not in allowed:
                    raise ValueError("Foreign source ID in actor payload: " + token)
    walk(_json_copy(value), "$")


def validate_task_public(task, allowed_source_ids, corpus_id, current_split):
    """Validate a closed TaskPublic dict against one frozen episode's corpus/split.

    source IDs are random/content-neutral hex IDs. A caller must supply the full
    current corpus allowlist; no per-task gold source filtering is performed here.
    target_questions contains 3-6 {claim_id, question} objects; initial_sources
    contains 1-2 opaque source IDs. Returned data is a JSON deep copy.
    """
    task = _json_copy(task)
    if not isinstance(task, dict) or set(task) != _TASK_FIELDS:
        raise ValueError("TaskPublic has missing or unexpected fields")
    allowed = set(allowed_source_ids)
    for source_id in allowed:
        _opaque(source_id)
    reject_private_fields(task, allowed)
    _identity(task)
    for key in ("task_id", "question", "corpus_id"):
        _nonempty(task[key], key)
    if task["corpus_id"] != corpus_id:
        raise ValueError("Task belongs to another corpus")
    if current_split not in SPLITS or task["split"] != current_split:
        raise ValueError("Task belongs to another split")
    targets = task["target_questions"]
    if not isinstance(targets, list) or not 3 <= len(targets) <= 6:
        raise ValueError("Expected 3-6 public target questions")
    seen = set()
    for target in targets:
        if not isinstance(target, dict) or set(target) != {"claim_id", "question"}:
            raise ValueError("Target questions must contain only claim_id and question")
        _nonempty(target["claim_id"], "claim_id")
        _nonempty(target["question"], "target question")
        if target["claim_id"] in seen:
            raise ValueError("Duplicate claim_id")
        seen.add(target["claim_id"])
    initial = task["initial_sources"]
    if (not isinstance(initial, list) or not 1 <= len(initial) <= 2 or
            any(not isinstance(x, str) for x in initial) or len(set(initial)) != len(initial)):
        raise ValueError("Expected 1-2 distinct initial source IDs")
    for source_id in initial:
        _opaque(source_id)
        if source_id not in allowed:
            raise ValueError("Initial source is outside the current episode corpus")
    if not isinstance(task["budget_recipe"], dict) or not task["budget_recipe"]:
        raise ValueError("budget_recipe must be a nonempty public object")
    return task

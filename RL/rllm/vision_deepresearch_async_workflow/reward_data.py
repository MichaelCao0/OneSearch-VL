"""Portable task/annotation joins and access to actual trajectory steps."""
from __future__ import annotations
import os
import re
import json
import threading
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

MODALITIES = ("single_image", "multi_image", "video")


GT_COMMON = ("id", "claim_facts")
GROUND_ANNOTATION_FIELDS = ("frame_index", "object_description")


def ground_frames(gt):
    """Read target order from minimal annotations or explicit input targets."""
    if 'gt_frames' in gt:
        return gt['gt_frames']
    return [a['frame_index'] for a in gt.get('grounding_annotations', [])]


def validate_gt_fields(gt, modality=None):
    """Keep judge targets minimal; reject metadata and multi-image/video boxes."""
    missing = set(GT_COMMON) - set(gt)
    if missing:
        raise ValueError(f"Missing reward GT fields: {sorted(missing)}")
    modality = modality or gt.get('modality') or (
        'single_image' if 'gt_bboxes' in gt or 'grounding_contract' in gt else 'multi_image')
    if modality not in MODALITIES:
        raise ValueError('Unsupported GT modality')
    extra = {'gt_bboxes', 'grounding_contract'} if modality == 'single_image' else {'gt_frames', 'grounding_annotations'}
    # Raw annotation bundles may still carry alignment fields. The release
    # projector removes these; runtime scoring never needs a duplicate answer.
    unused = set(gt) - set(GT_COMMON) - extra - {'answer', 'modality'}
    if unused:
        raise ValueError(f"Unused reward GT fields: {sorted(unused)}; project the training bundle first")
    for annotation in gt.get('grounding_annotations', []):
        if (not isinstance(annotation, Mapping) or 'frame_index' not in annotation
                or not set(annotation) <= set(GROUND_ANNOTATION_FIELDS)):
            raise ValueError('Ground annotation requires only frame_index and object_description')


_QUESTION_PREFIX_RE = re.compile(
    r"^(?:image|video)_id:\s*\d+\s*\n\s*Question:\s*", re.IGNORECASE
)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


_DATA_URI_RE = re.compile(
    r"data:(?:image|video)/[^;,\s]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE
)


def normalize_question(question: Any) -> str:
    return _QUESTION_PREFIX_RE.sub("", str(question or "")).strip()


def _path_identity(path: Any) -> str:
    parts = str(path or "").rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(path or "")


def media_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    # Registration records the portable identity before resolving media locally.
    # Use that identity even when local filenames differ.
    if row.get("evgr_media_identity") is not None:
        return tuple(row["evgr_media_identity"])
    images = row.get("images") or ()
    if images:
        return tuple(_path_identity(path) for path in images)
    grid = row.get("video_grid") or {}
    video_name = os.path.basename(str(grid.get("video_url") or ""))
    frames = tuple(_path_identity(path) for path in (grid.get("frame_urls") or ()))
    return (video_name, *frames)


def task_identity(
    row: Mapping[str, Any], modality: str | None = None
) -> tuple[Any, ...]:
    return (
        modality or str(row.get("modality") or ""),
        normalize_question(row.get("question")),
        str(row.get("answer") or ""),
        media_identity(row),
    )


class GroundTruthIndex:
    """Validated one-to-one mapping from runtime tasks to EVGR rubric rows."""

    def __init__(self, data_root: str | os.PathLike[str]):
        self.data_root = Path(data_root)
        self._rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.counts: dict[str, int] = {}
        for modality in MODALITIES:
            raw_path = self.data_root / "data" / f"{modality}.jsonl"
            gt_path = self.data_root / "evgr_gt" / f"{modality}.jsonl"
            if not raw_path.is_file() or not gt_path.is_file():
                raise FileNotFoundError(
                    f"EVGR pair missing for {modality}: raw={raw_path} gt={gt_path}"
                )
            count = 0
            with raw_path.open(encoding="utf-8") as raw_file, gt_path.open(
                encoding="utf-8"
            ) as gt_file:
                for line_no, pair in enumerate(
                    zip_longest(raw_file, gt_file, fillvalue=None), 1
                ):
                    raw_line, gt_line = pair
                    if raw_line is None or gt_line is None:
                        raise ValueError(
                            f"EVGR raw/GT length mismatch for {modality} "
                            f"at line {line_no}"
                        )
                    raw = json.loads(raw_line)
                    gt = json.loads(gt_line)
                    if str(raw.get('id')) != str(gt.get('id')):
                        raise ValueError(f"EVGR raw/GT ID mismatch for {modality} line {line_no}")
                    if "answer" in gt and str(raw.get("answer") or "") != str(gt.get("answer") or ""):
                        raise ValueError(
                            f"EVGR raw/GT answer mismatch for {modality} line {line_no}"
                        )
                    if raw.get('modality') != modality or gt.get('modality', modality) != modality:
                        raise ValueError(
                            f"EVGR GT modality mismatch for {modality} line {line_no}"
                        )
                    key = task_identity(raw, modality)
                    if key in self._rows:
                        raise ValueError(
                            f"ambiguous EVGR task identity for {modality} line {line_no}"
                        )
                    self._rows[key] = gt
                    count += 1
            self.counts[modality] = count

    def __len__(self) -> int:
        return len(self._rows)

    def lookup(self, task: Mapping[str, Any]) -> dict[str, Any]:
        key = task_identity(task)
        try:
            return self._rows[key]
        except KeyError as exc:
            raise KeyError(
                "EVGR GT missing for "
                f"modality={task.get('modality')!r} id={task.get('id')!r} "
                f"question={normalize_question(task.get('question'))[:120]!r}"
            ) from exc


_INDEX_LOCK = threading.Lock()


_INDEX_BY_ROOT: dict[str, GroundTruthIndex] = {}


def get_gt_index(data_root: str | os.PathLike[str]) -> GroundTruthIndex:
    root = os.path.realpath(os.fspath(data_root))
    with _INDEX_LOCK:
        index = _INDEX_BY_ROOT.get(root)
        if index is None:
            index = GroundTruthIndex(root)
            _INDEX_BY_ROOT[root] = index
        return index


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def value(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(
        value, name, default
    )


def parse_tool_call(step: Any) -> tuple[str, dict[str, Any]] | None:
    """Parse a tool call from persisted dictionaries or live Step objects."""

    action = value(step, "action")
    action_payload = getattr(action, "action", action)
    if isinstance(action_payload, Mapping) and action_payload.get("type") == "tool_call":
        raw = action_payload.get("tool_call")
        if isinstance(raw, Mapping):
            call = raw
        else:
            try:
                call = json.loads(str(raw or ""))
            except json.JSONDecodeError:
                call = None
        if isinstance(call, Mapping) and call.get("name"):
            return str(call["name"]), dict(as_mapping(call.get("arguments")))

    response = str(value(step, "model_response", "") or "")
    for raw in reversed(_TOOL_CALL_RE.findall(response)):
        try:
            call = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(call, Mapping) and call.get("name"):
            return str(call["name"]), dict(as_mapping(call.get("arguments")))
    return None


def frame_indices(values: Any, count: int) -> list[int]:
    """Validate explicit source frame numbers, independent of media filenames."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError("EVGR frame indices must align with the ordered media list")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("EVGR frame indices must be nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("EVGR frame indices must be unique within a task")
    return list(values)


def episode_payload(
    episode: Any,
) -> tuple[Mapping[str, Any], Sequence[Any], int | None, Any, bool, bool]:
    task = as_mapping(value(episode, "task", {}))
    trajectories = value(episode, "trajectories", ()) or ()
    steps = value(trajectories[0], "steps", ()) if trajectories else ()
    metrics = as_mapping(value(episode, "metrics", {}))
    info = as_mapping(value(episode, "info", {}))
    fatal = bool(metrics.get("is_fatal", info.get("is_fatal", False)))
    raw_index = metrics.get("fatal_step_index", info.get("fatal_step_index"))
    try:
        fatal_index = int(raw_index) if fatal and raw_index is not None else None
    except (TypeError, ValueError):
        fatal_index = None
    episode_id = value(episode, "episode_id", value(episode, "id"))
    correct = bool(value(episode, "is_correct", False))
    return task, steps, fatal_index, episode_id, correct, fatal

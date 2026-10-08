"""Typed execution evidence shared by the independent EVGR judges.

Only executor observations create evidence. Actor thoughts and final answers
are deliberately never read. Local media locations live in ``_media`` only;
public records use stable IDs. Crop attachments are judge region previews,
not a claim that a stored executor output was recovered byte-for-byte.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from vision_deepresearch_async_workflow import reward_data as data
from vision_deepresearch_async_workflow.reward_client import RewardError

RETRIEVAL = {"image_search": "image_match", "text_search": "search_result", "visit_webpage": "page_content"}
ALIASES = {"search_image": "image_search", "search_text": "text_search", "web_search": "text_search", "visit": "visit_webpage"}
VISUAL_TYPES = {"source_image", "selected_frame", "crop"}
_IMAGE_HANDLE = re.compile(r"New image (?:ID|handle):\s*(img_\d+)", re.I)
_CLIP_HANDLE = re.compile(r"New clip handle:\s*(clip_\d+)", re.I)
_ERROR = re.compile(r"^(?:Tool execution error\s*:|Error(?:\s+executing\b|\s*:)|\w+ failed after\b|No (?:search )?results\b)", re.I)


def format_value(value):
    """Readable data only: no outer JSON, truncation, or model summarization."""
    if isinstance(value, Mapping):
        return "\n".join(f"{key}: {format_value(item)}" for key, item in value.items()) or "(none)"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    if value is None:
        return "(none)"
    return str(value)


def format_arguments(arguments):
    # This exact representation is also used to validate Ground use_quote.
    return "\n".join(f"{key}: {format_value(arguments[key])}" for key in sorted(arguments)) or "(none)"


def format_calls(calls):
    blocks = []
    for call in calls:
        lines = [f"{call['call_id']} | step {call['step']} | {call['tool']} | {call['status']}",
                 "  Arguments:\n" + format_arguments(call['arguments']),
                 "  Inputs: " + format_value(call['input_refs']) + "; outputs: " + format_value(call['output_refs']),
                 "  Previously available visuals: " + format_value(call['available_visual_refs'])]
        if 'grid_source_frames' in call:
            lines.append("  Active source frames: " + format_value(call['grid_source_frames']))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or "(No tool calls.)"


def _body(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value or "")
    text = re.sub(r"</?(?:tool_response|observation)>\s*", "", text).strip()
    text = re.sub(r"^Tool execution result:\s*", "", text).strip()
    # Normalize presentational emphasis BEFORE presenting/citing the evidence;
    # validators still require exact spans, not fuzzy post-hoc quote repair.
    return re.sub(r"\*\*([^\n]+?)\*\*", r"\1", text)


def _safe(value):
    """Omit binary blobs and local path strings from textual judge inputs."""
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, str):
        value = data._DATA_URI_RE.sub("<binary-media-omitted>", value)
        return re.sub(r"(?<!\w)/(?:mnt|home|tmp|nfs|apdcephfs)/[^\s<>\"']+",
                      lambda m: "asset_" + hashlib.sha256(m[0].encode()).hexdigest()[:16], value)
    return value


def _source(value):
    if isinstance(value, Mapping):
        return str(value.get("path") or value.get("url") or value.get("image_url") or "")
    return str(value or "")


def _frame_numbers(paths, explicit):
    if explicit is not None:
        return data.frame_indices(explicit, len(paths))
    result = []
    for i, path in enumerate(paths):
        match = re.search(r"frame_(\d+)", str(path))
        result.append(int(match[1]) if match else i)
    if len(set(result)) != len(result):
        raise RewardError("Ambiguous frame identities; provide explicit frame_indices")
    return result


def _open(source):
    if source.startswith("data:image/"):
        source = io.BytesIO(base64.b64decode(source.split(",", 1)[1], validate=True))
    elif "://" in source:
        raise RewardError("Ground evidence requires materialized local pixels")
    elif not Path(source).is_file():
        raise RewardError("Ground evidence pixels are missing locally")
    with Image.open(source) as image:
        return image.convert("RGB")


def _pixels(spec):
    image = _open(spec["source"])
    for op in spec.get("crops", []):
        w, h = image.size
        if "bbox_1000" in op:
            box = op["bbox_1000"]
            x, y = round(box[0] / 1000 * w), round(box[1] / 1000 * h)
            cw, ch = round((box[2] - box[0]) / 1000 * w), round((box[3] - box[1]) / 1000 * h)
        else:
            x, y, cw, ch = (op[k] for k in ("x", "y", "width", "height"))
        if cw <= 0 or ch <= 0:
            raise RewardError("Empty crop region in execution evidence")
        image = image.crop((x, y, x + cw, y + ch))
    return image


def _crop(args):
    box = args.get("bbox_1000", args.get("bbox"))
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise RewardError("Malformed crop coordinates")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise RewardError("Non-finite crop coordinates")
        box = [max(0, min(1000, int(v))) for v in box]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise RewardError("Inverted crop coordinates")
        return {"bbox_1000": box}
    op = {k: args.get(k, 0) for k in ("x", "y", "width", "height")}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in op.values()):
        raise RewardError("Malformed pixel crop coordinates")
    if op["width"] <= 0 or op["height"] <= 0:
        raise RewardError("Empty crop coordinates")
    return op


def _results(text):
    """Preserve actual result boundaries; never synthesize supporting text."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, list):
        return [json.dumps(v, ensure_ascii=False) for v in value]
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return [json.dumps(v, ensure_ascii=False) for v in value["results"]]
    passages = re.split(r"(?=\[Passage \d+\])", text)
    return [p.strip() for p in passages if p.strip()]


def _ground_descriptions(gt):
    """Read explicit, frame-aligned labels only; never infer from claim facts."""
    if 'grounding_annotations' not in gt:
        return {}
    annotations = gt['grounding_annotations']
    if not isinstance(annotations, list):
        raise RewardError('Malformed grounding annotations')
    labels = {}
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise RewardError('Malformed grounding annotation')
        frame = annotation.get('frame_index')
        if type(frame) is not int or frame in labels:
            raise RewardError('Invalid or duplicate grounding annotation frame')
        description = annotation.get('object_description')
        if description is not None and (not isinstance(description, str) or not description.strip()):
            raise RewardError('Invalid object description')
        # Ground reads only the reference appearance. No quality/status labels,
        # provenance, or multi-image/video boxes are part of its target input.
        labels[frame] = dict(object_description=_safe(description))
    if set(labels) != set(data.ground_frames(gt) or []):
        raise RewardError('Grounding annotations do not match target frames')
    return labels


def normalize_evidence(task, steps, gt, *, fatal_step_index=None):
    """Normalize the complete pre-fatal prefix, without reasoning or answers.

    Video indices are positions in the active grid, not source frame numbers.
    New handles exist only after successful executor receipts. A missing or
    failed receipt never creates a selected frame or crop. OCR and unknown
    tool outputs are retained as excluded records, not retrieval evidence.
    """
    modality = task.get("modality")
    if modality not in data.MODALITIES:
        raise ValueError("Unsupported evidence modality")
    selected = list(steps)
    if fatal_step_index is not None:
        if not 0 <= fatal_step_index <= len(selected):
            raise ValueError("Invalid fatal prefix index")
        selected = selected[:fatal_step_index]
    records, calls, media, targets, diagnostics = [], [], {}, [], []
    handles, grids, roots = {}, {}, {}
    descriptions = _ground_descriptions(gt) if modality != 'single_image' else {}

    def visual(eid, kind, spec, step, *, parents=(), handle=None, frame=None):
        record = dict(evidence_id=eid, evidence_type=kind, status="available",
                      step=step, parent_refs=list(parents), frame_index=frame)
        records.append(record)
        media[eid] = spec
        if handle:
            if handle in handles:
                raise RewardError("Reused visual handle in execution log")
            handles[handle] = eid
            record["handle"] = handle
        return eid

    if modality in ("single_image", "multi_image"):
        paths = [_source(v) for v in task.get("images") or []]
        if not paths or (modality == "single_image" and len(paths) != 1):
            raise RewardError("Invalid source image count")
        numbers = _frame_numbers(paths, task.get("image_frame_indices"))
        for i, (path, frame) in enumerate(zip(paths, numbers), 1):
            eid = visual(f"source_{i}", "source_image", {"source": path}, 0,
                         handle=f"img_{i}", frame=frame if modality == "multi_image" else None)
            roots[frame] = eid
        if modality == "single_image":
            boxes = gt.get("gt_bboxes") or []
            if not boxes and gt.get("grounding_contract") != "source_only":
                raise RewardError("Missing single-image grounding targets")
            for i, box in enumerate(boxes, 1):
                tid = f"target_{i}"
                media[tid] = {"source": paths[0], "crops": [_crop({"bbox_1000": box})]}
                targets.append(dict(target_id=tid, source_ref="source_1", bbox_1000=box))
            if not boxes:
                media["target_1"] = media["source_1"]
                targets.append(dict(target_id="target_1", source_ref="source_1"))
        else:
            for i, frame in enumerate(data.ground_frames(gt) or [], 1):
                if frame not in roots:
                    raise RewardError("Ground target absent from multi-image input")
                tid = f"target_{i}"
                media[tid] = media[roots[frame]]
                targets.append(dict(target_id=tid, source_ref=roots[frame], frame_index=frame,
                                    **descriptions.get(frame, {})))
    else:
        grid = task.get("video_grid") or {}
        paths = [_source(v) for v in grid.get("frame_urls") or []]
        numbers = data.frame_indices(grid.get("frame_indices", list(range(len(paths)))), len(paths))
        n = min(int(grid.get("n_frames") or len(paths)), len(paths))
        grids["video_1"] = list(zip(numbers, paths))[:n]
        by_frame = dict(grids["video_1"])
        for i, frame in enumerate(data.ground_frames(gt) or [], 1):
            if frame not in by_frame:
                raise RewardError("Ground target absent from video grid")
            tid = f"target_{i}"
            media[tid] = {"source": by_frame[frame]}
            targets.append(dict(target_id=tid, frame_index=frame, judge_reference_only=True,
                                **descriptions.get(frame, {})))
    if not targets:
        raise RewardError("No visual targets in rubric")

    for i, step in enumerate(selected, 1):
        # Accept frozen normalized trials as well as live/persisted Step objects.
        raw = data.value(step, "tool_call")
        action = data.value(step, "action")
        action = getattr(action, "action", action)
        if not raw and isinstance(action, Mapping) and action.get("type") and action["type"] != "tool_call":
            continue  # A quoted tool block inside a final answer was not executed.
        parsed = (str(raw["name"]), dict(raw.get("arguments") or {})) if isinstance(raw, Mapping) and raw.get("name") else data.parse_tool_call(step)
        if not parsed:
            continue
        name, args = parsed
        name = ALIASES.get(name, name)
        raw_observation = data.value(step, "observation", "")
        body = _safe(_body(raw_observation))
        state = "missing_observation" if not body else "error" if _ERROR.search(body) else "success"
        cid = f"call_{i}"
        source_handle = str(args.get("image") or args.get("url") or args.get("source") or "")
        source_id = handles.get(source_handle)
        call = dict(call_id=cid, step=i, tool=name, arguments=_safe(args), status=state,
                    input_refs=[source_id] if source_id else [], output_refs=[],
                    available_visual_refs=list(handles.values()))
        calls.append(call)
        if name in RETRIEVAL and state == "success":
            for j, result in enumerate(_results(body), 1):
                eid = f"ev_{i}_{j}"
                records.append(dict(evidence_id=eid, call_id=cid, step=i, evidence_type=RETRIEVAL[name],
                                    status="success", parent_refs=list(call["input_refs"]), text=result))
                call["output_refs"].append(eid)
            if not call["output_refs"]:
                call["status"] = "empty"
        elif name == "select_timespan" and state == "success" and body.lower().startswith("select_timespan succeeded"):
            parent = str(args.get("video") or args.get("clip") or "")
            match = _CLIP_HANDLE.search(body)
            grid = grids.get(parent, [])
            try:
                start = max(0, min(int(args.get("start_index", args.get("start_idx"))), len(grid)-1))
                end = min(int(args.get("end_index", args.get("end_idx"))), len(grid))
                if not match or not grid or end <= start or match[1] in grids:
                    raise ValueError()
                grids[match[1]] = grid[start:end]
                call["grid_source_frames"] = [v[0] for v in grids[match[1]]]
            except (ValueError, TypeError):
                raise RewardError("Invalid successful timespan receipt") from None
        elif name == "select_frame" and state == "success" and body.lower().startswith("select_frame succeeded"):
            parent = str(args.get("video") or args.get("clip") or "")
            grid = grids.get(parent, [])
            match = _IMAGE_HANDLE.search(body)
            try:
                index = max(0, min(int(args.get("frame_index", args.get("idx"))), len(grid)-1))
                if not grid or not match:
                    raise ValueError()
                frame, path = grid[index]
            except (TypeError, ValueError):
                raise RewardError("Invalid successful frame receipt") from None
            eid = visual(f"ev_{i}_visual", "selected_frame", {"source": path}, i, handle=match[1], frame=frame)
            call["output_refs"].append(eid)
        elif name == "crop" and state == "success" and body.lower().startswith("image cropped successfully"):
            match = _IMAGE_HANDLE.search(body)
            if not source_id or not match:
                diagnostics.append(dict(call_id=cid, issue="crop_source_not_verified"))
                continue
            spec = dict(media[source_id])
            spec["crops"] = [*spec.get("crops", []), _crop(args)]
            eid = visual(f"ev_{i}_visual", "crop", spec, i, parents=[source_id], handle=match[1])
            call["output_refs"].append(eid)
        elif body:
            kind = "excluded_ocr" if name in {"layout_parsing", "ocr"} else "execution_receipt"
            records.append(dict(evidence_id=f"ev_{i}_receipt", call_id=cid, step=i,
                                evidence_type=kind, status=state, parent_refs=call["input_refs"], text=body))
            if name == "super_resolution":
                diagnostics.append(dict(call_id=cid, issue="enhanced_pixels_not_materialized"))

    facts = gt.get("claim_facts") or []
    if not facts or any(not isinstance(v, str) or not v.strip() for v in facts):
        raise RewardError("Trace requires nonempty structured claim_facts")
    trace_targets = [dict(target_id=f"claim_{i}", claim=_safe(fact)) for i, fact in enumerate(dict.fromkeys(facts), 1)]
    return dict(modality=modality, question=_safe(data.normalize_question(task.get("question"))),
                records=records, calls=calls, trace_targets=trace_targets, ground_targets=targets,
                diagnostics=diagnostics, prefix_steps=len(selected), _media=media)


def attach_pixels(evidence, *, max_images=32, max_edge=768):
    """Return lossless judge previews, deduplicated by their rendered pixels."""
    parts, manifest, seen = [], [], {}
    for eid, spec in evidence["_media"].items():
        image = _pixels(spec)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content = buf.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        if digest not in seen:
            aid = f"image_{len(seen)+1}"
            seen[digest] = aid
            parts.extend([{"type": "text", "text": f"Attachment {aid}"},
                          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(content).decode(), "detail": "high"}}])
        manifest.append(dict(evidence_id=eid, attachment_id=seen[digest],
                             preview_size=list(image.size), region_preview=bool(spec.get("crops"))))
    if len(seen) > max_images:
        raise RewardError("Ground image budget exceeded; no silent frame dropping")
    return manifest, parts


def evidence_spans(record):
    """Addressable contiguous spans; offsets always point into original text."""
    text = record["text"]
    boundaries = list(re.finditer(r"(?<=[.!?])\s+|\n+", text))
    ranges = zip([0] + [m.end() for m in boundaries], [m.start() for m in boundaries] + [len(text)])
    spans = []
    for start, end in ranges:
        while start < end:
            while start < end and text[start].isspace():
                start += 1
            stop = end
            if end-start > 480:
                stop = text.rfind(" ", start+160, start+480)
                if stop <= start:
                    stop = start+480
            while stop > start and text[stop-1].isspace():
                stop -= 1
            if stop > start:
                spans.append(dict(span_id=f"{record['evidence_id']}.s{len(spans)+1}",
                                  start=start, end=stop, quote=text[start:stop]))
            start = max(stop, start+1)
    return spans


def organize_ground_evidence(view):
    visuals = {v["evidence_id"]: v for v in view["visual_evidence"]}
    creators = {eid: c for c in view["calls"] for eid in c["output_refs"] if eid in visuals}
    attachments = {a["evidence_id"]: a for a in view["attachments"]}

    def attachment(eid):
        if eid not in attachments:
            raise RewardError("Ground packet is missing attached pixels")
        return {k: attachments[eid][k] for k in
                ("attachment_id", "preview_size", "region_preview")}

    def lineage(eid, visited=()):
        if eid in visited or eid not in visuals:
            raise RewardError("Invalid visual lineage in ground packet")
        chain = []
        for parent in visuals[eid].get("parent_refs", []):
            chain.extend(lineage(parent, (*visited, eid)))
        return list(dict.fromkeys([*chain, eid]))

    actor_packets = []
    for eid, visual in visuals.items():
        chain = lineage(eid)
        creator = creators.get(eid)
        actor_packets.append(dict(
            **visual, role="actor_visual", **attachment(eid),
            source_frame_indices=list(dict.fromkeys(visuals[p]["frame_index"] for p in chain
                                                    if visuals[p].get("frame_index") is not None)),
            lineage=[dict(evidence_id=p, **attachment(p)) for p in chain],
            creation_call=({k: creator[k] for k in ("call_id", "tool", "arguments", "status")}
                           if creator else None),
            explicit_image_use_ids=[u["use_id"] for u in view["use_records"]
                                    if u["visual_evidence_id"] == eid and u["binding"] == "explicit_image_argument"],
            unverified_query_candidate_ids=[u["use_id"] for u in view["use_records"]
                                           if u["visual_evidence_id"] == eid and u["binding"] != "explicit_image_argument"],
        ))
    actor_by_id = {v["evidence_id"]: v for v in actor_packets}

    target_packets = []
    for target in view["targets"]:
        candidates = []
        for actor in actor_packets:
            parents = {p["evidence_id"] for p in actor["lineage"]}
            if target.get("source_ref") and target["source_ref"] not in parents:
                continue
            same_frame = target.get("frame_index") is not None and target["frame_index"] in actor["source_frame_indices"]
            candidates.append(dict(
                visual_evidence_id=actor["evidence_id"], attachment_id=actor["attachment_id"],
                reference_relation=("same_source_image" if target.get("source_ref") else
                                    "same_source_frame" if same_frame else "different_source_frame"),
                object_match="not_established_requires_pixel_comparison",
                explicit_image_use_ids=actor["explicit_image_use_ids"],
                unverified_query_candidate_ids=actor["unverified_query_candidate_ids"],
            ))
        target_packets.append(dict(
            target, role="judge_reference_only", **attachment(target["target_id"]),
            region_scope=("annotated_bbox" if target.get("bbox_1000") is not None else
                          "whole_reference_image_object_region_unspecified"),
            object_description=target.get("object_description"),
            actor_candidates=candidates,
        ))

    retrieval_packets = []
    for call in view["calls"]:
        if call["tool"] not in {"image_search", "text_search"}:
            continue
        explicit, unverified = [], []
        for use in view["use_records"]:
            if use["use_call_id"] != call["call_id"]:
                continue
            actor = actor_by_id[use["visual_evidence_id"]]
            record = dict(**use, attachment_id=actor["attachment_id"], lineage=actor["lineage"])
            if use["binding"] == "explicit_image_argument":
                record["relation_status"] = "verified_tool_input_not_verified_object_identity"
                explicit.append(record)
            else:
                record["relation_status"] = "availability_only_query_object_relation_unverified"
                unverified.append(record)
        retrieval_packets.append(dict(
            **{k: call[k] for k in ("call_id", "step", "tool", "arguments", "status")},
            explicit_image_uses=explicit, unverified_query_candidates=unverified,
            image_matches=[r for r in view["image_matches"] if r["call_id"] == call["call_id"]],
        ))
    return dict(question=view["question"], modality=view["modality"],
                target_packets=target_packets, actor_visual_packets=actor_packets,
                retrieval_packets=retrieval_packets, execution_calls=view["calls"])


def label_ground_images(view, packets, images):
    """Put all aliases and their distinct roles immediately beside each image.

    Reference/actor aliases can share identical pixels. Deduplication must not
    turn a reference image into an actor action, or hide an actual actor alias.
    """
    aids = list(dict.fromkeys(a["attachment_id"] for a in view["attachments"]))
    if len(images) != 2 * len(aids):
        raise RewardError("Ground attachment count does not match pixel manifest")
    by_aid = {}
    for aid, label, image in zip(aids, images[::2], images[1::2]):
        if label.get("text") != "Attachment " + aid or image.get("type") != "image_url":
            raise RewardError("Ground attachment order does not match pixel manifest")
        by_aid[aid] = image
    # Reference frames first; execution pixels then appear in actor order.
    ordered = list(dict.fromkeys(p["attachment_id"] for p in
                                [*packets["target_packets"], *packets["actor_visual_packets"]]))
    result = []
    for aid in ordered:
        roles = []
        for target in packets["target_packets"]:
            if target["attachment_id"] == aid:
                roles.append(f"Reference {target['target_id']} | judge_reference_only | "
                             f"frame: {format_value(target.get('frame_index'))} | region: {target['region_scope']}")
        for visual in packets["actor_visual_packets"]:
            if visual["attachment_id"] == aid:
                roles.append(f"Actor {visual['evidence_id']} | actor_visual | {visual['handle']} | "
                             f"{visual['evidence_type']} | source frames: {format_value(visual['source_frame_indices'])}")
        text = "\n".join(roles)
        result.extend([dict(type="text", text=text + "\nAttachment " + aid), by_aid[aid]])
    return result

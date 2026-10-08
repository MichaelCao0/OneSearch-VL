#!/usr/bin/env python3
"""Entrypoint for video agent inference.

Reads a QA CSV and the video/frame manifest produced by ``prepare_video.py``.
The manifest may reference local media or public URLs. Tool calls select frames
and slice clips at runtime; see README.md for the manifest schema.

Example (from the repository root):

    python onesearch_vl/run_infer_video.py --model 8b --gpus 0 \\
        --csv data/video.csv \\
        --manifest data/grids.jsonl \\
        --output-dir outputs/video --limit 100
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from rollout_admission import wait_for_tool_capacity, ToolCapacityUnavailable
import shutil
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from onesearch_infer import config, pipeline
from onesearch_infer.runners import InferenceConfig, build_runner


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OneSearch-VL Video Deep Research Agent. Tools: select_timespan, "
            "select_frame, crop, image_search, text_search."
        )
    )
    # Model config
    parser.add_argument(
        "--model",
        choices=sorted(config.MODEL_REGISTRY.keys()),
        default="30b-a3b",
        help="Model variant.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    # Data config
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="VideoDR CSV with columns (index, question, answer, ...).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "Path to grids_manifest.jsonl produced by "
            "preprocessing/preprocess_grids.py. One line per case_id; "
            "carries video_url + object_urls (frame PNG URLs)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where per-case trajectories and intermediate artifacts go.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Run only these specific row indices (overrides start/end).",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def _load_qa_csv(path: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read a QA CSV in either of the two supported layouts.

    Supported formats:

    1. **VideoDR-Benchmark ``test.csv``** (with header):
       ``id, category, video_url, question, answer, evidence, remarks`` — the
       on-disk layout shipped by HuggingFace ``strike20023/VideoDR``.

    2. **Headerless 3-column CSV**: ``index, question, answer`` — the
       compact format used by ``VideoDR-Agent`` and the smoke tests.

    Returns a DataFrame with normalised columns ``index``, ``question``,
    ``answer``, plus any extras from the source file (``category`` etc.).
    """

    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in df.columns}

    if "id" in cols and "question" in cols and "answer" in cols:
        rename = {cols["id"]: "index", cols["question"]: "question", cols["answer"]: "answer"}
        df = df.rename(columns=rename)
        df["index"] = df["index"].astype(int)
        if logger:
            logger.info("Detected VideoDR-Benchmark CSV layout (columns: %s)", list(df.columns))
        return df

    if "index" in cols and "question" in cols and "answer" in cols:
        rename = {cols["index"]: "index", cols["question"]: "question", cols["answer"]: "answer"}
        df = df.rename(columns=rename)
        df["index"] = df["index"].astype(int)
        if logger:
            logger.info("Detected explicit-header (index,question,answer) CSV layout")
        return df

    if logger:
        logger.info("Falling back to headerless 3-column CSV (index,question,answer)")
    return pd.read_csv(
        path,
        header=None,
        names=["index", "question", "answer"],
        encoding="utf-8-sig",
        usecols=[0, 1, 2],
    )


def _load_manifest(path: str, logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Load grids_manifest.jsonl indexed by ``case_id`` (only ok=True rows)."""

    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            cid = rec.get("case_id")
            if cid is None or not rec.get("ok"):
                continue
            out[str(cid)] = rec
    logger.info("Loaded %d ok manifest records from %s", len(out), path)
    return out


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s",
        level=getattr(logging, level),
    )


def _resume_search_roots(output_dir: str) -> List[str]:
    """Directories to scan for an existing trajectory besides ``output_dir``."""

    roots: List[str] = []
    seen = set()

    def _add(path: str) -> None:
        ab = os.path.abspath(path)
        if ab in seen or not os.path.isdir(ab):
            return
        seen.add(ab)
        roots.append(ab)

    output_base = (os.environ.get("OUTPUT_BASE") or "").strip()
    if output_base:
        _add(output_base)

    parent = os.path.dirname(os.path.abspath(output_dir))
    if os.path.basename(output_dir).startswith("shard_"):
        _add(parent)

    return roots


def _is_complete_trajectory(path: str) -> bool:
    """A trajectory counts as done only if it is fully-written valid JSON and
    was not aborted by an inference error.

    Guards against two failure modes:
    - preemption mid-write leaving a truncated (non-empty) file -> JSON fails
      to parse -> not complete, will be regenerated.
    - a case that died on an inference error (status="error") -> not complete,
      so it can be retried instead of being permanently skipped.

    Legacy trajectories predating the ``status`` field have no such key; we
    treat missing status as complete so already-produced outputs are not
    needlessly regenerated.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return False
    if isinstance(obj, dict) and obj.get("status") == "error":
        return False
    return True


def _find_existing_trajectory(
    case_index: int,
    output_dir: str,
    *,
    search_roots: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Return path to a *complete* ``{case_index}_trajectory.json`` if any."""

    name = f"{case_index}_trajectory.json"
    candidates: List[str] = [os.path.join(output_dir, name)]

    for root in search_roots or ():
        candidates.append(os.path.join(root, "merged", name))
        candidates.extend(sorted(glob.glob(os.path.join(root, "shard_*", name))))

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0 and _is_complete_trajectory(path):
            return path
    return None


def _materialize_resume_copy(existing: str, local_path: str) -> None:
    """Copy a trajectory found under another shard into this shard's dir."""

    if os.path.abspath(existing) == os.path.abspath(local_path):
        return
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0 and _is_complete_trajectory(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy2(existing, local_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_infer_video")

    if not os.path.exists(args.csv):
        logger.error("CSV not found: %s", args.csv)
        return 1
    if not os.path.exists(args.manifest):
        logger.error("Manifest not found: %s", args.manifest)
        return 1

    runner = build_runner(
        model_name=args.model,
        checkpoint=args.checkpoint,
        gpus=args.gpus,
        dtype=args.dtype,
    )
    logger.info("Model: %s (supports_video=%s)", runner.display_name, runner.spec.supports_video)
    try:
        runner.load()
    except Exception as exc:
        logger.error("Failed to initialise runner: %s", exc, exc_info=True)
        return 1

    df = _load_qa_csv(args.csv, logger)
    logger.info("Loaded %d rows from %s", len(df), args.csv)

    manifest = _load_manifest(args.manifest, logger)

    if args.indices:
        rows = df[df["index"].isin(args.indices)]
        if rows.empty:
            logger.error("None of the requested indices found in CSV.")
            return 1
        case_iter: List[tuple[int, pd.Series]] = list(rows.iterrows())
    else:
        start = max(0, int(args.start))
        if args.end is not None:
            end = min(int(args.end), len(df))
        elif args.limit is not None:
            end = min(start + int(args.limit), len(df))
        else:
            end = len(df)
        case_iter = [
            (df.index[i], df.iloc[i]) for i in range(start, end)
        ]

    os.makedirs(args.output_dir, exist_ok=True)
    inference_cfg = InferenceConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    success, failure = 0, 0
    skipped_done = 0
    resume = os.environ.get("RESUME_TRAJECTORIES", "1").lower() in ("1", "true", "yes")
    resume_roots = _resume_search_roots(args.output_dir) if resume else []
    if resume and resume_roots:
        logger.info(
            "Resume enabled; also searching under: %s",
            ", ".join(resume_roots),
        )
    for pos, (_, row) in enumerate(case_iter):
        case_index = int(row["index"])
        question = str(row["question"]) if pd.notna(row["question"]) else ""
        ground_truth = str(row["answer"]) if pd.notna(row["answer"]) else ""

        manifest_record = manifest.get(str(case_index))
        if manifest_record is None:
            logger.warning(
                "No manifest record for case_id=%d; skipping.", case_index
            )
            failure += 1
            continue

        # Resume: skip cases with a non-empty trajectory in this shard dir or
        # anywhere under OUTPUT_BASE (merged/ + sibling shard_*). Copies found
        # trajectories into the local shard dir so later restarts stay local.
        out_path = os.path.join(args.output_dir, f"{case_index}_trajectory.json")
        if resume:
            existing = _find_existing_trajectory(
                case_index, args.output_dir, search_roots=resume_roots
            )
            if existing:
                _materialize_resume_copy(existing, out_path)
                skipped_done += 1
                success += 1
                continue

        logger.info(
            "[%d/%d] index=%d  video_url=%s",
            pos + 1, len(case_iter), case_index,
            (manifest_record.get("video_url") or "")[:80],
        )

        category = (
            str(row["category"]) if "category" in row.index and pd.notna(row.get("category"))
            else "video_dr"
        )
        synthetic_row: dict = {
            "data_id": str(case_index),
            "category": category,
            "data_source": "VideoDR",
            "question": question,
            "ground_truth": ground_truth,
            "prompt": [{"role": "user", "content": question}],
        }
        if "evidence" in row.index and pd.notna(row.get("evidence")):
            synthetic_row["evidence"] = str(row["evidence"])

        try:
            wait_for_tool_capacity()
            traj = pipeline.process_single_case(
                row=synthetic_row,
                runner=runner,
                output_dir=args.output_dir,
                case_idx=case_index,
                dataset_type="test",
                inference_cfg=inference_cfg,
                manifest_record=manifest_record,
            )
            # A case that returned but aborted on an inference error is a
            # failure, not a success — otherwise error trajectories get
            # counted as done and skipped forever on resume.
            if isinstance(traj, dict) and traj.get("status") == "error":
                failure += 1
                logger.error("Case index=%d ended with status=error: %s",
                             case_index, traj.get("error"))
            else:
                success += 1
        except ToolCapacityUnavailable:
            raise
        except Exception as exc:
            failure += 1
            logger.error("Case index=%d failed: %s", case_index, exc)
            traceback.print_exc()

    logger.info(
        "Done. success=%d failure=%d skipped_resume=%d output=%s",
        success, failure, skipped_done, args.output_dir,
    )
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Entrypoint for **Multi-Image Deep Research** agent inference.

Reads a QA CSV and a multi-image manifest (see ``README.md``) and
runs the dedicated MI pipeline (``pipeline_mi.process_single_case_mi``).

Example (from the repository root):

    python onesearch_vl/run_infer_mi.py \\
        --model 8b \\
        --gpus 0 \\
        --csv data/multi_image.csv \\
        --manifest data/multi_image_manifest.jsonl \\
        --output-dir outputs/multi_image \\
        --limit 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from rollout_admission import wait_for_tool_capacity, ToolCapacityUnavailable
import sys
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd

from onesearch_infer import config
from onesearch_infer.pipeline_mi import process_single_case_mi
from onesearch_infer.runners import InferenceConfig, build_runner

# Reuse CSV + resume helpers from the video entrypoint.
from run_infer_video import (  # noqa: E402
    _configure_logging,
    _find_existing_trajectory,
    _load_qa_csv,
    _materialize_resume_copy,
    _resume_search_roots,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OneSearch-VL Multi-Image Deep Research Agent. "
            "Bootstrap: img_1..img_n (unordered). No select_timespan/select_frame."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(config.MODEL_REGISTRY.keys()),
        default="30b-a3b",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="mi_manifest.jsonl (modality=multi_image, image_urls only).",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
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


def _load_mi_manifest(path: str, logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
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
            if rec.get("modality") != "multi_image":
                logger.warning("Skipping non-MI manifest row case_id=%s", cid)
                continue
            out[str(cid)] = rec
    logger.info("Loaded %d ok MI manifest records from %s", len(out), path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_infer_mi")

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
    logger.info("Model: %s", runner.display_name)
    try:
        runner.load()
    except Exception as exc:
        logger.error("Failed to initialise runner: %s", exc, exc_info=True)
        return 1

    df = _load_qa_csv(args.csv, logger)
    logger.info("Loaded %d rows from %s", len(df), args.csv)

    manifest = _load_mi_manifest(args.manifest, logger)

    if args.indices:
        rows = df[df["index"].isin(args.indices)]
        if rows.empty:
            logger.error("None of the requested indices found in CSV.")
            return 1
        case_iter = list(rows.iterrows())
    else:
        start = max(0, int(args.start))
        if args.end is not None:
            end = min(int(args.end), len(df))
        elif args.limit is not None:
            end = min(start + int(args.limit), len(df))
        else:
            end = len(df)
        case_iter = [(df.index[i], df.iloc[i]) for i in range(start, end)]

    os.makedirs(args.output_dir, exist_ok=True)
    inference_cfg = InferenceConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    success, failure = 0, 0
    skipped_done = 0
    resume = os.environ.get("RESUME_TRAJECTORIES", "1").lower() in ("1", "true", "yes")
    resume_roots = _resume_search_roots(args.output_dir) if resume else []

    for pos, (_, row) in enumerate(case_iter):
        case_index = int(row["index"])
        question = str(row["question"]) if pd.notna(row["question"]) else ""
        ground_truth = str(row["answer"]) if pd.notna(row["answer"]) else ""

        manifest_record = manifest.get(str(case_index))
        if manifest_record is None:
            logger.warning("No MI manifest for case_id=%d; skipping.", case_index)
            failure += 1
            continue

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
            "[%d/%d] index=%d  n_images=%s  mi_id=%s",
            pos + 1,
            len(case_iter),
            case_index,
            manifest_record.get("n_images"),
            manifest_record.get("mi_id"),
        )

        structure = (
            str(row["structure"])
            if "structure" in row.index and pd.notna(row.get("structure"))
            else "mi_dr"
        )
        synthetic_row: dict = {
            "data_id": str(case_index),
            "category": structure,
            "data_source": "MultiImageDR",
            "question": question,
            "ground_truth": ground_truth,
            "prompt": [{"role": "user", "content": question}],
        }

        try:
            wait_for_tool_capacity()
            traj = process_single_case_mi(
                row=synthetic_row,
                runner=runner,
                output_dir=args.output_dir,
                case_idx=case_index,
                manifest_record=manifest_record,
                inference_cfg=inference_cfg,
            )
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
        success,
        failure,
        skipped_done,
        args.output_dir,
    )
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

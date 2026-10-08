#!/usr/bin/env python3
"""Entrypoint for single-image agent inference.

Reads parquet rows with image bytes, paths, or URLs and runs the single-image
pipeline. See README.md for the input schema.

Example (from the repository root):

    python onesearch_vl/run_infer_single.py --model 8b --gpus 0 \\
        --data-path data/single_image.parquet \\
        --output-dir outputs/single_image --limit 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from rollout_admission import wait_for_tool_capacity, ToolCapacityUnavailable
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from onesearch_infer import config
from onesearch_infer.pipeline_single import process_single_case_single
from onesearch_infer.runners import InferenceConfig, build_runner


def _eval_skip_indices() -> set[int]:
    """Return explicitly quarantined row indices for this eval task only."""

    raw = os.environ.get("EVAL_SKIP_CASE_INDICES", "").strip()
    if not raw:
        return set()
    try:
        return {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(f"invalid EVAL_SKIP_CASE_INDICES={raw!r}") from exc


def _write_quarantined_trajectory(
    *, row: Any, case_id: str, case_idx: int, output_dir: str
) -> None:
    """Record one known deterministic input failure without blocking its shard."""

    trajectory = {
        "case_id": case_id,
        "case_idx": case_idx,
        "category": row.get("category", "single_dr"),
        "data_source": row.get("data_source", "SingleImageDR"),
        "modality": "single_image",
        "n_images": 0,
        "prompt": row.get("prompt", []),
        "turns": [],
        "timestamp": datetime.now().isoformat(),
        "final_response_text": "",
        "status": "error",
        "error": "quarantined deterministic multimodal preprocessing stall",
    }
    out_path = os.path.join(output_dir, f"{case_id}_trajectory.json")
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(trajectory, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)
    logging.getLogger(__name__).warning(
        "Quarantined case %d as an error trajectory: %s", case_idx, out_path
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OneSearch-VL Single-Image Deep Research Agent. "
            "Bootstrap: img_1 (direct). No select_timespan/select_frame."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(config.MODEL_REGISTRY.keys()),
        default="8b",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--indices-file",
        type=Path,
        default=None,
        help="Optional newline-delimited original row indices to process.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s",
        level=getattr(logging, level),
    )


def _selected_indices(
    *, start: int, end: int, num_shards: int, shard_index: int
) -> range:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >=1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0,{num_shards}), got {shard_index}"
        )
    return range(start + shard_index, end, num_shards)


def _load_indices(path: Path, *, total_rows: int) -> list[int]:
    values: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        try:
            case_idx = int(value)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: not an integer: {value!r}") from exc
        if case_idx < 0 or case_idx >= total_rows:
            raise ValueError(
                f"{path}:{line_number}: index {case_idx} outside [0,{total_rows})"
            )
        values.append(case_idx)
    if not values:
        raise ValueError(f"indices file is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"indices file contains duplicate values: {path}")
    return values


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_infer_single")

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
        logger.error("Failed to initialize runner: %s", exc, exc_info=True)
        return 1

    if not os.path.exists(args.data_path):
        logger.error("Data file not found: %s", args.data_path)
        return 1

    df = pd.read_parquet(args.data_path)
    logger.info("Loaded %d rows from %s", len(df), args.data_path)

    if args.category:
        if args.indices_file is not None:
            logger.error("--category cannot be combined with --indices-file")
            return 2
        df = df[df["category"] == args.category]
        logger.info("Filtered to category=%s: %d rows", args.category, len(df))

    start = max(0, int(args.start))
    if args.end is not None:
        end = min(int(args.end), len(df))
    elif args.limit is not None:
        end = min(start + int(args.limit), len(df))
    else:
        end = len(df)

    try:
        if args.indices_file is not None:
            all_indices = _load_indices(args.indices_file, total_rows=len(df))
            if args.num_shards < 1:
                raise ValueError(f"num_shards must be >=1, got {args.num_shards}")
            if args.shard_index < 0 or args.shard_index >= args.num_shards:
                raise ValueError(
                    f"shard_index must be in [0,{args.num_shards}), got {args.shard_index}"
                )
            selected_indices = all_indices[args.shard_index :: args.num_shards]
        else:
            selected_indices = _selected_indices(
                start=start,
                end=end,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            )
    except ValueError as exc:
        logger.error("Invalid shard selection: %s", exc)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(
        "Processing rows [%d, %d) shard=%d/%d selected=%d -> %s",
        start,
        end,
        args.shard_index,
        args.num_shards,
        len(selected_indices),
        args.output_dir,
    )

    inference_cfg = InferenceConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    resume = os.environ.get("RESUME_TRAJECTORIES", "1").lower() in ("1", "true", "yes")
    skip_indices = _eval_skip_indices()
    success, failure, skipped_done, quarantined = 0, 0, 0, 0

    for idx in selected_indices:
        try:
            row = df.iloc[idx]
            case_id = (
                row.get("data_id", f"case_{idx}") if hasattr(row, "get") else f"case_{idx}"
            )
            out_path = os.path.join(args.output_dir, f"{case_id}_trajectory.json")
            if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                skipped_done += 1
                success += 1
                continue
            if idx in skip_indices:
                _write_quarantined_trajectory(
                    row=row,
                    case_id=str(case_id),
                    case_idx=idx,
                    output_dir=args.output_dir,
                )
                quarantined += 1
                success += 1
                continue
            wait_for_tool_capacity()
            process_single_case_single(
                row=row,
                runner=runner,
                output_dir=args.output_dir,
                case_idx=idx,
                dataset_type=args.dataset,
                inference_cfg=inference_cfg,
            )
            success += 1
        except ToolCapacityUnavailable:
            raise
        except Exception as exc:
            failure += 1
            logger.error("Case %d failed: %s", idx, exc)
            traceback.print_exc()

    logger.info(
        "Done. success=%d failure=%d skipped_resume=%d quarantined=%d output=%s",
        success,
        failure,
        skipped_done,
        quarantined,
        args.output_dir,
    )
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

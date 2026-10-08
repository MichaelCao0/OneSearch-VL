"""Single-image deep research inference pipeline.

Separate from video (``pipeline.process_single_case``) and multi-image
(``pipeline_mi.process_single_case_mi``):
- Bootstrap: ``img_1`` (optionally more) via parquet / URL / local dir.
- System prompt: the shared single-image prompt.
- Entry: ``run_infer_single.py`` (``run_infer.py`` delegates here).
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .pipeline import _bootstrap_images, run_agent_loop
from .runners import BaseRunner, InferenceConfig
from .prompts import get_system_prompt

logger = logging.getLogger(__name__)

FILENAME_PREFIX_TRAIN = "single_dr_train"
FILENAME_PREFIX_TEST = "single_dr_test"


def bootstrap_single_image(
    row: Any,
    case_id: str,
    case_idx: int,
    filename_prefix: str,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Register input image(s) as ``img_1..img_n`` without extra user text."""

    return _bootstrap_images(
        row, case_id, case_idx, filename_prefix, image_urls_dict
    )


def _extract_prompt_text(row: Any) -> str:
    prompt_list = row.get("prompt", []) if hasattr(row, "get") else []
    if isinstance(prompt_list, list) and prompt_list:
        first = prompt_list[0]
        return first.get("content", "") if isinstance(first, dict) else str(first)
    if hasattr(row, "get"):
        q = row.get("question", "")
        if isinstance(q, str) and q:
            return re.sub(
                r"^\s*image_id:\s*[^\s]+\s+Question:\s*",
                "",
                q,
                flags=re.IGNORECASE,
            ).strip()
    return str(prompt_list) if prompt_list else ""


def process_single_case_single(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
    dataset_type: str = "train",
    visual_lookup: Optional[Callable[..., object]] = None,
    inference_cfg: Optional[InferenceConfig] = None,
) -> Dict[str, Any]:
    """Drive one single-image QA through the ReAct agent loop."""

    filename_prefix = (
        FILENAME_PREFIX_TEST if dataset_type == "test" else FILENAME_PREFIX_TRAIN
    )

    if hasattr(row, "get"):
        case_id = row.get("data_id", f"case_{case_idx}")
        category = row.get("category", "single_dr")
        data_source = row.get("data_source", "SingleImageDR")
        prompt_list = row.get("prompt", [])
    else:
        case_id = f"case_{case_idx}"
        category = "single_dr"
        data_source = "SingleImageDR"
        prompt_list = []

    prompt_text = _extract_prompt_text(row)

    logger.info(
        "Processing single-image case %d (%s, category=%s, source=%s)",
        case_idx + 1,
        case_id,
        category,
        data_source,
    )

    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    image_paths_dict, initial_parts = bootstrap_single_image(
        row,
        case_id,
        case_idx,
        filename_prefix,
        image_urls_dict,
    )
    if not initial_parts:
        logger.warning(
            "Single-image case %s has no resolvable visual input; agent will rely on text only.",
            case_id,
        )

    initial_parts.append({"text": prompt_text})
    sys_prompt = get_system_prompt("single_image")

    gemini_contents: List[Dict[str, Any]] = [
        {"role": "user", "parts": initial_parts}
    ]

    n_imgs = sum(1 for k in image_paths_dict if k.startswith("img_"))
    trajectory: Dict[str, Any] = {
        "case_id": case_id,
        "case_idx": case_idx,
        "category": category,
        "data_source": data_source,
        "modality": "single_image",
        "n_images": n_imgs,
        "prompt": prompt_list,
        "turns": [],
        "timestamp": datetime.now().isoformat(),
    }

    return run_agent_loop(
        runner=runner,
        gemini_contents=gemini_contents,
        image_paths_dict=image_paths_dict,
        trajectory=trajectory,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        case_id=case_id,
        case_idx=case_idx,
        filename_prefix=filename_prefix,
        runner_supports_video=False,
        visual_lookup=visual_lookup,
        inference_cfg=inference_cfg,
        system_instruction=sys_prompt,
        modality="single_image",
    )

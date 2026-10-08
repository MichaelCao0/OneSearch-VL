"""Multi-image deep research inference pipeline.

Separate from ``pipeline.process_single_case`` (video / FVQA):
- Bootstrap: ``img_1..img_n`` URLs only (no ``video_1``).
- System prompt: the shared multi-image prompt (no select_timespan/frame).
- Entry: ``run_infer_mi.py``.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .pipeline import _add_image_url, run_agent_loop
from .prompts import get_system_prompt
from .runners import BaseRunner, InferenceConfig

logger = logging.getLogger(__name__)

FILENAME_PREFIX = "mi_dr"


def bootstrap_mi_from_manifest(
    manifest_record: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Register shuffled ``image_urls`` as ``img_1..img_n``."""

    image_urls = list(manifest_record.get("image_urls") or [])
    image_paths_dict: Dict[str, Any] = {}
    initial_parts: List[Dict[str, Any]] = []
    for url in image_urls:
        url_s = str(url or "")
        if url_s.startswith(("http://", "https://")):
            _add_image_url(image_paths_dict, initial_parts, url_s)
    return image_paths_dict, initial_parts


def process_single_case_mi(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    manifest_record: Dict[str, Any],
    visual_lookup: Optional[Callable[..., object]] = None,
    inference_cfg: Optional[InferenceConfig] = None,
) -> Dict[str, Any]:
    """Drive one multi-image QA through the ReAct agent loop."""

    if manifest_record.get("modality") != "multi_image":
        raise ValueError(
            f"Expected modality=multi_image, got {manifest_record.get('modality')!r}"
        )
    if not manifest_record.get("ok"):
        raise ValueError(f"Manifest record case_id={manifest_record.get('case_id')} ok=false")

    if hasattr(row, "get"):
        case_id = row.get("data_id", f"case_{case_idx}")
        category = row.get("category", "mi_dr")
        data_source = row.get("data_source", "MultiImageDR")
        prompt_list = row.get("prompt", [])
    else:
        case_id = f"case_{case_idx}"
        category = "mi_dr"
        data_source = "MultiImageDR"
        prompt_list = []

    if isinstance(prompt_list, list) and prompt_list:
        first = prompt_list[0]
        prompt_text = (
            first.get("content", "") if isinstance(first, dict) else str(first)
        )
    else:
        prompt_text = ""

    if not prompt_text and hasattr(row, "get"):
        q = row.get("question", "")
        if isinstance(q, str) and q:
            prompt_text = q.strip()

    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    image_paths_dict, initial_parts = bootstrap_mi_from_manifest(manifest_record)
    if not initial_parts:
        logger.warning(
            "Multi-image case %s has no resolvable image URLs; agent will rely on text only.",
            case_id,
        )

    initial_parts.append({"text": prompt_text})
    sys_prompt = get_system_prompt("multi_image")

    gemini_contents: List[Dict[str, Any]] = [
        {"role": "user", "parts": initial_parts}
    ]

    trajectory: Dict[str, Any] = {
        "case_id": case_id,
        "case_idx": case_idx,
        "category": category,
        "data_source": data_source,
        "modality": "multi_image",
        "n_images": manifest_record.get("n_images"),
        "mi_id": manifest_record.get("mi_id"),
        "source_qa_key": manifest_record.get("source_qa_key"),
        "youtube_id": manifest_record.get("youtube_id"),
        "mi_primitive": manifest_record.get("mi_primitive"),
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
        filename_prefix=FILENAME_PREFIX,
        runner_supports_video=False,
        visual_lookup=visual_lookup,
        inference_cfg=inference_cfg,
        system_instruction=sys_prompt,
        modality="multi_image",
    )

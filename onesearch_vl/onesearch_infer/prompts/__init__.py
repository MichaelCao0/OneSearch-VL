"""The three complete actor prompts shared by RL and inference."""
from functools import lru_cache
import json
from pathlib import Path

MODALITIES = ("single_image", "multi_image", "video")


@lru_cache(maxsize=None)
def get_system_prompt(modality: str = "video") -> str:
    """Return the complete system slot, including the frozen tool definitions."""
    if modality not in MODALITIES:
        raise ValueError(f"Unknown visual modality: {modality!r}")
    return Path(__file__).with_name(modality + ".txt").read_text(encoding="utf-8")


def get_tool_specs(modality: str = "video") -> list[dict]:
    """Read tool signatures from the same prompt shown to the actor."""
    prompt = get_system_prompt(modality)
    block = prompt.split("\n<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    return [json.loads(line) for line in block.splitlines() if line.strip()]


@lru_cache(maxsize=None)
def tool_names(modality: str = "video") -> frozenset[str]:
    return frozenset(spec["function"]["name"] for spec in get_tool_specs(modality))

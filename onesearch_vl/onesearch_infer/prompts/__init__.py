"""The actor prompts shared by RL and inference; retrieval is enabled by default."""
from functools import lru_cache
import json
import os
from pathlib import Path

MODALITIES = ("single_image", "multi_image", "video", "audio_video")


def retrieval_disabled() -> bool:
    """Gate retrieval tools only; model, OCR and judge services remain available."""
    return os.getenv("ONESEARCH_DISABLE_RETRIEVAL", "0").strip() == "1"


def is_retrieval_tool(name: str) -> bool:
    """Cover canonical search/visit names and the legacy Lens tag alias."""
    name = str(name).lower()
    return "search" in name or name.startswith("visit") or name == "lens_scan"


def _specs_from_prompt(prompt: str) -> list[dict]:
    block = prompt.split("\n<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    return [json.loads(line) for line in block.splitlines() if line.strip()]


@lru_cache(maxsize=None)
def _system_prompt(modality: str, disable_retrieval: bool) -> str:
    if modality not in MODALITIES:
        raise ValueError(f"Unknown visual modality: {modality!r}")
    original = Path(__file__).with_name(modality + ".txt").read_text(encoding="utf-8")
    if not disable_retrieval:
        return original
    specs = [spec for spec in _specs_from_prompt(original)
             if not is_retrieval_tool(spec["function"]["name"])]
    definitions = "\n".join(json.dumps(spec, ensure_ascii=False) for spec in specs)
    if modality == "audio_video":
        return (
            "You are an audio/video question-answering agent. External retrieval is disabled "
            "for this run. Inspect the supplied native media, including audio when present. "
            "Use inspect_av to replay a bounded interval. start_s/end_s are seconds relative "
            "to the input video_n/clip_n handle; clips start at 0. They are ordinary playback "
            "seconds, not frame-grid indices. Do not invent sources or external verification. "
            "If the media cannot establish the requested fact, state that evidence is insufficient.\n"
            "Before a tool call, reason briefly in <think>...</think>, then emit exactly one "
            '<tool_call>{"name":"...","arguments":{...}}</tool_call> and wait for its '
            "observation. Finish with <response>...</response>.\n\n<tools>\n"
            + definitions + "\n</tools>\n"
        )
    return (
        f"You are a visual inspection agent for {modality} inputs.\n"
        "External retrieval is disabled for this run. Use the provided media and "
        "the available visual tools only. Do not call search or webpage tools, "
        "invent sources, or claim new external verification. If the question "
        "requires facts absent from the supplied media, state that evidence is insufficient.\n"
        "Inspect visible content and any relevant regions or frames. Video/clip "
        "selection uses sampled frame-grid indices, not original-video seconds. "
        "Use the supplied img_n, video_n and clip_n handles.\n"
        "Before a tool call, reason briefly in <think>...</think>, then emit exactly "
        'one <tool_call>{"name":"...","arguments":{...}}</tool_call> and wait '
        "for its observation. Finish with a concise grounded answer in "
        "<response>...</response>.\n\n<tools>\n"
        + definitions + "\n</tools>\n"
    )


def get_system_prompt(modality: str = "video") -> str:
    """Read the current gate each call; cache entries are keyed by that mode."""
    return _system_prompt(modality, retrieval_disabled())


def get_tool_specs(modality: str = "video") -> list[dict]:
    """Read tool signatures from the exact prompt shown to the actor."""
    return _specs_from_prompt(get_system_prompt(modality))


@lru_cache(maxsize=None)
def _tool_names(modality: str, disable_retrieval: bool) -> frozenset[str]:
    return frozenset(spec["function"]["name"] for spec in
                     _specs_from_prompt(_system_prompt(modality, disable_retrieval)))


def tool_names(modality: str = "video") -> frozenset[str]:
    return _tool_names(modality, retrieval_disabled())


# Retain the existing cache-clear API used by callers and validation utilities.
get_system_prompt.cache_clear = _system_prompt.cache_clear
tool_names.cache_clear = _tool_names.cache_clear

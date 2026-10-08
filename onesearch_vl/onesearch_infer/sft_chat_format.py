"""Tool observations in the Qwen SFT chat layout."""
from typing import Any, Dict, List, Optional, Sequence


def observation_user_message(
    observation_text: str,
    media_parts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Keep produced media inside the tool response, before observation text."""
    media = list(media_parts or [])
    if media:
        parts: List[Dict[str, Any]] = [
            {"text": "<tool_response>\n"}, *media,
            {"text": f"{observation_text}\n</tool_response>"},
        ]
    else:
        parts = [{"text": f"<tool_response>\n{observation_text}\n</tool_response>"}]
    return {"role": "user", "parts": parts}

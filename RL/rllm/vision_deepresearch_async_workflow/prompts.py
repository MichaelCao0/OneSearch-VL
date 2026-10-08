"""Load the same complete actor prompts as inference."""
import sys
from pathlib import Path

_infer_root = str(Path(__file__).resolve().parents[3] / "onesearch_vl")
if _infer_root not in sys.path:
    sys.path.insert(0, _infer_root)
from onesearch_infer.prompts import get_system_prompt, get_tool_specs, tool_names  # noqa: E402

__all__ = ["get_system_prompt", "get_tool_specs", "tool_names"]

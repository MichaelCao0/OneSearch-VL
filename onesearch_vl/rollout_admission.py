"""Load the dependency-free shared admission controller for evaluation."""

import importlib.util
from pathlib import Path

_source = Path(__file__).resolve().parents[1] / "RL/rllm/rllm/tool_admission.py"
_spec = importlib.util.spec_from_file_location("_onesearch_tool_admission", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
wait_for_tool_capacity = _module.wait_for_tool_capacity
ToolCapacityUnavailable = _module.ToolCapacityUnavailable

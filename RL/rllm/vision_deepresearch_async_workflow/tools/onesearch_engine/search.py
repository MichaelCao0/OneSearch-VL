"""Use the same provider implementation as inference."""
import sys
from pathlib import Path

_infer_root = str(Path(__file__).resolve().parents[5] / "onesearch_vl")
if _infer_root not in sys.path:
    sys.path.insert(0, _infer_root)
from onesearch_infer import search as _shared  # noqa: E402
# Alias the module itself so runtime statistics and injected test clients observe
# calls inside the canonical backend as well as calls at the RL boundary.
sys.modules[__name__] = _shared

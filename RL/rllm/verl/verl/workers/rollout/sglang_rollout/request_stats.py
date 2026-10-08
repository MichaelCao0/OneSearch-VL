"""Small, dependency-free helpers for rollout server observability."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def summarize_scheduler_load(loads: Iterable[Any]) -> dict[str, int]:
    """Combine SGLang ``GetLoadReqOutput`` values across DP ranks.

    Keeping this conversion outside the GPU-dependent SGLang server module
    makes the accounting testable on CPU-only development hosts.
    """

    num_reqs = 0
    num_waiting_reqs = 0
    num_tokens = 0
    ranks = 0
    for load in loads:
        ranks += 1
        num_reqs += max(0, int(getattr(load, "num_reqs", 0) or 0))
        num_waiting_reqs += max(
            0, int(getattr(load, "num_waiting_reqs", 0) or 0)
        )
        num_tokens += max(0, int(getattr(load, "num_tokens", 0) or 0))

    return {
        "ranks": ranks,
        "num_reqs": num_reqs,
        "num_running_reqs": max(0, num_reqs - num_waiting_reqs),
        "num_waiting_reqs": min(num_reqs, num_waiting_reqs),
        "num_tokens": num_tokens,
    }

"""Ordering helpers for launching repeated rollout tasks."""

from collections import Counter
from typing import NamedTuple


class RolloutLaunch(NamedTuple):
    """One rollout slot, annotated without changing its input position."""

    input_index: int
    task: dict
    task_id: str
    rollout_index: int


def rollout_major_launch_plan(
    tasks: list[dict], task_ids: list[str]
) -> list[RolloutLaunch]:
    """Return repeated tasks in rollout-major launch order.

    ``DataProto.repeat(..., interleave=True)`` groups repeats by source, for
    example ``A0, A1, B0, B1``.  Launching that order lets same-source cache
    waiters occupy scarce preprocessing slots.  The rollout-major order
    ``A0, B0, A1, B1`` exposes distinct sources first while retaining each
    slot's original input index for deterministic result reconstruction.
    """
    if len(tasks) != len(task_ids):
        raise ValueError(
            "tasks and task_ids must have equal lengths: "
            f"{len(tasks)} != {len(task_ids)}"
        )

    next_rollout_index: Counter[str] = Counter()
    plan = []
    for input_index, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
        rollout_index = next_rollout_index[task_id]
        next_rollout_index[task_id] += 1
        plan.append(
            RolloutLaunch(
                input_index=input_index,
                task=task,
                task_id=task_id,
                rollout_index=rollout_index,
            )
        )

    plan.sort(key=lambda item: (item.rollout_index, item.input_index))
    return plan

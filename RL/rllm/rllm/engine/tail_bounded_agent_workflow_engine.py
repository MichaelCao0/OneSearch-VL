"""Group-safe rollout tail truncation for RL training.

The default engine waits for every trajectory.  For RLOO, dropping arbitrary
individual trajectories changes group sizes, so this opt-in engine stops only
after a configured fraction of *whole* rollout groups has completed and keeps
exactly those complete groups.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid
from collections import Counter, defaultdict, deque

from tqdm import tqdm

from rllm.engine.agent_workflow_engine import (
    AgentWorkflowEngine,
    _episode_is_strictly_valid,
)
from rllm.engine.rollout_launch_order import rollout_major_launch_plan


logger = logging.getLogger(__name__)


def complete_group_fraction_from_env() -> float:
    raw = os.environ.get("RLLM_COMPLETE_GROUP_FRACTION", "1.0")
    try:
        fraction = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "RLLM_COMPLETE_GROUP_FRACTION must be a float in (0, 1], "
            f"got {raw!r}"
        ) from exc
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "RLLM_COMPLETE_GROUP_FRACTION must be in (0, 1], "
            f"got {fraction}"
        )
    return fraction


class TailBoundedAgentWorkflowEngine(AgentWorkflowEngine):
    """Stop a training rollout after enough complete RLOO groups finish."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complete_group_fraction = complete_group_fraction_from_env()
        self.tail_completion_priority = os.environ.get(
            "RLLM_TAIL_COMPLETION_PRIORITY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        print(f"tail_completion_priority: {self.tail_completion_priority}")
        self._last_group_tail_stats: dict[str, int | float] = {}

    async def _run_completion_priority(
        self,
        launch_plan,
        expected_by_group,
        target_groups,
        stagger_s,
        task_count,
        kwargs,
    ):
        """Keep broad group coverage, then refill slots toward completion.

        The legacy path creates every coroutine up front.  The workflow-worker
        queue therefore admits the next rollout-major slot, even when another
        group is one response away from becoming trainable.  Here the first
        ``n_parallel_tasks`` slots are still admitted round-robin across prompt
        groups, preserving replica diversity.  Later slots are created only as
        capacity is released and prefer groups that have already made progress.
        """

        pending_by_group = defaultdict(deque)
        group_order = []
        group_rank = {}
        for launch in launch_plan:
            group_id = str(launch.task_id)
            if group_id not in group_rank:
                group_rank[group_id] = len(group_order)
                group_order.append(group_id)
            pending_by_group[group_id].append(launch)

        capacity = max(1, min(int(self.n_parallel_tasks), len(launch_plan)))
        inflight = {}
        inflight_by_group: Counter[str] = Counter()
        launched_by_group: Counter[str] = Counter()
        completed_by_group: Counter[str] = Counter()
        complete_groups: set[str] = set()
        slot_episodes = {}
        launch_count = 0
        consumed = 0

        def submit_next(group_id):
            nonlocal launch_count
            launch = pending_by_group[group_id].popleft()
            delay = (
                launch_count * stagger_s / task_count
                if stagger_s > 0 and task_count > 1
                else 0.0
            )

            async def run():
                if delay > 0:
                    await asyncio.sleep(delay)
                return await self.process_task_with_retry(
                    launch.task,
                    str(launch.task_id),
                    launch.rollout_index,
                    **kwargs,
                )

            future = asyncio.create_task(run())
            inflight[future] = group_id
            inflight_by_group[group_id] += 1
            launched_by_group[group_id] += 1
            launch_count += 1

        # Fill initial slots round-robin across prompt groups.
        while len(inflight) < capacity:
            made_progress = False
            for group_id in group_order:
                if len(inflight) >= capacity:
                    break
                if pending_by_group[group_id]:
                    submit_next(group_id)
                    made_progress = True
            if not made_progress:
                break

        try:
            with tqdm(total=len(launch_plan), desc="Generating trajectories") as pbar:
                stop = False
                while inflight and not stop:
                    done, _ = await asyncio.wait(
                        tuple(inflight), return_when=asyncio.FIRST_COMPLETED
                    )
                    progressed_groups = []
                    for future in done:
                        group_id = inflight.pop(future)
                        inflight_by_group[group_id] -= 1
                        task_id, rollout_idx, episode = await future
                        task_id = str(task_id)
                        slot_episodes[(task_id, rollout_idx)] = episode
                        completed_by_group[task_id] += 1
                        progressed_groups.append(task_id)
                        consumed += 1
                        pbar.update(1)
                        if (
                            completed_by_group[task_id]
                            == expected_by_group[task_id]
                        ):
                            complete_groups.add(task_id)
                        if len(complete_groups) >= target_groups:
                            stop = True
                            break

                    if stop:
                        break

                    # Refill available slots from groups that just made progress.
                    for group_id in progressed_groups:
                        if len(inflight) >= capacity:
                            break
                        if pending_by_group[group_id]:
                            submit_next(group_id)

                    # A fully launched group can leave spare capacity.  Admit
                    # an unseen group first (needed when groups > capacity),
                    # then favor the closest-to-complete remaining group.
                    while len(inflight) < capacity:
                        unseen = [
                            group_id
                            for group_id in group_order
                            if pending_by_group[group_id]
                            and launched_by_group[group_id] == 0
                        ]
                        if unseen:
                            submit_next(unseen[0])
                            continue
                        candidates = [
                            group_id
                            for group_id in group_order
                            if pending_by_group[group_id]
                        ]
                        if not candidates:
                            break
                        group_id = max(
                            candidates,
                            key=lambda item: (
                                completed_by_group[item],
                                -inflight_by_group[item],
                                -group_rank[item],
                            ),
                        )
                        submit_next(group_id)
        except BaseException:
            for future in inflight:
                if not future.done():
                    future.cancel()
            await asyncio.gather(*inflight, return_exceptions=True)
            raise

        unconsumed_done = sum(future.done() for future in inflight)
        cancelled = 0
        for future in inflight:
            if not future.done():
                future.cancel()
                cancelled += 1
        await asyncio.gather(*inflight, return_exceptions=True)
        not_submitted = sum(len(pending) for pending in pending_by_group.values())
        return (
            slot_episodes,
            complete_groups,
            consumed,
            cancelled,
            {
                "early_stop_priority_enabled": 1,
                "early_stop_launched_future_count": launch_count,
                "early_stop_not_submitted_episode_count": not_submitted,
                "early_stop_unconsumed_done_count": unconsumed_done,
            },
        )

    async def execute_tasks(
        self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs
    ):
        fraction = getattr(self, "complete_group_fraction", 1.0)
        if fraction >= 1.0 or self.current_mode == "val":
            self._last_group_tail_stats = {}
            return await super().execute_tasks(tasks, task_ids, **kwargs)

        if self.workflow_queue is None:
            await self.initialize_pool()
        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]
        if len(tasks) != len(task_ids):
            raise ValueError(
                "tasks and task_ids must have equal lengths: "
                f"{len(tasks)} != {len(task_ids)}"
            )

        expected_by_group = Counter(str(task_id) for task_id in task_ids)
        total_groups = len(expected_by_group)
        target_groups = max(1, math.ceil(total_groups * fraction))
        completed_by_group: Counter[str] = Counter()
        complete_groups: set[str] = set()
        slot_episodes = {}
        submitted_slots = [None] * len(tasks)

        stagger_s = float(os.getenv("ROLLOUT_STAGGER_S", "0.0"))
        task_count = len(tasks)

        async def staggered(coro, delay: float):
            if delay > 0:
                await asyncio.sleep(delay)
            return await coro

        launch_plan = rollout_major_launch_plan(tasks, task_ids)
        for launch in launch_plan:
            submitted_slots[launch.input_index] = (
                str(launch.task_id),
                launch.rollout_index,
            )

        priority_stats = {}
        if getattr(self, "tail_completion_priority", False):
            (
                slot_episodes,
                complete_groups,
                consumed,
                cancelled,
                priority_stats,
            ) = await self._run_completion_priority(
                launch_plan,
                expected_by_group,
                target_groups,
                stagger_s,
                task_count,
                kwargs,
            )
        else:
            futures = []
            for launch_number, launch in enumerate(launch_plan):
                delay = (
                    launch_number * stagger_s / task_count
                    if stagger_s > 0 and task_count > 1
                    else 0.0
                )
                coro = self.process_task_with_retry(
                    launch.task,
                    str(launch.task_id),
                    launch.rollout_index,
                    **kwargs,
                )
                futures.append(
                    asyncio.create_task(
                        staggered(coro, delay) if delay > 0 else coro
                    )
                )

            consumed = 0
            cancelled = 0
            try:
                with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
                    for future in asyncio.as_completed(futures):
                        task_id, rollout_idx, episode = await future
                        task_id = str(task_id)
                        slot_episodes[(task_id, rollout_idx)] = episode
                        completed_by_group[task_id] += 1
                        consumed += 1
                        pbar.update(1)
                        if (
                            completed_by_group[task_id]
                            == expected_by_group[task_id]
                        ):
                            complete_groups.add(task_id)
                        if len(complete_groups) >= target_groups:
                            break
            except BaseException:
                for future in futures:
                    if not future.done():
                        future.cancel()
                await asyncio.gather(*futures, return_exceptions=True)
                raise

            for future in futures:
                if not future.done():
                    future.cancel()
                    cancelled += 1
            await asyncio.gather(*futures, return_exceptions=True)

        # Keep exact, full groups only.  A fast partial group that happened to
        # finish before the threshold is deliberately discarded so RLOO never
        # sees a variable response count.
        accepted_groups = set(complete_groups)
        results = [
            slot_episodes.get(slot) if slot[0] in accepted_groups else None
            for slot in submitted_slots
        ]
        accepted_episodes = sum(
            expected_by_group[group_id] for group_id in accepted_groups
        )
        self._last_group_tail_stats = {
            "early_stop_enabled": 1,
            "early_stop_target_complete_groups": target_groups,
            "early_stop_complete_group_count": len(accepted_groups),
            "early_stop_dropped_group_count": total_groups - len(accepted_groups),
            "early_stop_accepted_episode_count": accepted_episodes,
            "early_stop_dropped_episode_count": len(tasks) - accepted_episodes,
            "early_stop_consumed_future_count": consumed,
            "early_stop_cancelled_future_count": cancelled,
            "early_stop_complete_group_fraction": (
                len(accepted_groups) / total_groups if total_groups else 1.0
            ),
            **priority_stats,
        }
        logger.warning(
            "[group-tail-stop] accepted_groups=%d/%d accepted_episodes=%d/%d "
            "consumed=%d cancelled=%d",
            len(accepted_groups),
            total_groups,
            accepted_episodes,
            len(tasks),
            consumed,
            cancelled,
        )
        if priority_stats:
            logger.warning(
                "[group-tail-priority] launched=%d not_submitted=%d "
                "unconsumed_done=%d",
                priority_stats["early_stop_launched_future_count"],
                priority_stats["early_stop_not_submitted_episode_count"],
                priority_stats["early_stop_unconsumed_done_count"],
            )

        await self._apply_rewards(results)
        self._last_episodes = results
        if self.episode_logger is not None:
            try:
                self.episode_logger.log_episodes_batch(
                    [episode for episode in results if episode is not None],
                    self.current_step,
                    self.current_mode,
                    self.current_epoch,
                )
            except Exception:
                logger.exception("Failed to log group-tail-bounded episodes")
        return results

    def _validate_strict_tail_slots(self, episodes, task_ids) -> None:
        """Validate accepted slots while allowing deliberately dropped groups.

        ``require_all_valid`` normally requires every submitted slot to contain
        a valid episode.  Tail stopping intentionally represents every slot in
        a dropped group as ``None`` so ``repeat_counts`` can remove the matching
        rows from the original Verl batch.  Accept that one explicit case while
        retaining strict UID, validity, and whole-group invariants for all rows
        that reach training.
        """
        if len(episodes) != len(task_ids):
            raise RuntimeError(
                "strict tail transform slot count mismatch: "
                f"episodes={len(episodes)} task_ids={len(task_ids)}"
            )

        expected_by_group = Counter(str(task_id) for task_id in task_ids)
        present_by_group: Counter[str] = Counter()
        next_rollout_idx: Counter[str] = Counter()
        accepted_episode_count = 0

        for slot, (episode, task_id) in enumerate(zip(episodes, task_ids)):
            group_id = str(task_id)
            rollout_idx = next_rollout_idx[group_id]
            next_rollout_idx[group_id] += 1
            if episode is None:
                continue

            expected_episode_id = f"{group_id}:{rollout_idx}"
            actual_episode_id = str(getattr(episode, "id", ""))
            if actual_episode_id != expected_episode_id:
                raise RuntimeError(
                    "strict tail transform episode/task/slot UID mismatch at "
                    f"slot {slot}: expected={expected_episode_id!r} "
                    f"actual={actual_episode_id!r}"
                )
            if not _episode_is_strictly_valid(episode):
                raise RuntimeError(
                    "strict tail transform received invalid accepted episode: "
                    f"{actual_episode_id}"
                )
            present_by_group[group_id] += 1
            accepted_episode_count += 1

        partial_groups = {
            group_id: (present_by_group[group_id], expected_count)
            for group_id, expected_count in expected_by_group.items()
            if present_by_group[group_id] not in (0, expected_count)
        }
        if partial_groups:
            raise RuntimeError(
                "strict tail transform refuses partial RLOO groups: "
                f"{partial_groups}"
            )

        accepted_group_count = sum(
            present_by_group[group_id] == expected_count
            for group_id, expected_count in expected_by_group.items()
        )
        stats = self._last_group_tail_stats
        expected_episodes = stats.get("early_stop_accepted_episode_count")
        expected_groups = stats.get("early_stop_complete_group_count")
        if expected_episodes is not None and accepted_episode_count != expected_episodes:
            raise RuntimeError(
                "strict tail transform accepted episode count mismatch: "
                f"validated={accepted_episode_count} engine={expected_episodes}"
            )
        if expected_groups is not None and accepted_group_count != expected_groups:
            raise RuntimeError(
                "strict tail transform accepted group count mismatch: "
                f"validated={accepted_group_count} engine={expected_groups}"
            )

    def transform_results_for_verl(self, episodes, task_ids):
        stats = getattr(self, "_last_group_tail_stats", {})
        strict_tail = bool(
            stats
            and getattr(self, "require_all_valid", False)
            and self.current_mode != "val"
        )
        if strict_tail:
            self._validate_strict_tail_slots(episodes, task_ids)

        # The base transform must retain one repeat_count per submitted slot,
        # including zero for each deliberate tail drop.  Temporarily bypass its
        # all-slots-present assertion only after the stronger tail-aware checks
        # above have passed.
        original_require_all_valid = getattr(self, "require_all_valid", False)
        if strict_tail:
            self.require_all_valid = False
        try:
            result = super().transform_results_for_verl(episodes, task_ids)
        finally:
            if strict_tail:
                self.require_all_valid = original_require_all_valid

        stats = getattr(self, "_last_group_tail_stats", {})
        if stats:
            raw_metrics = result.meta_info.setdefault("raw_rollout_metrics", {})
            raw_metrics.update(stats)
            # The base transform treats every ``None`` as an ERROR.  These
            # slots were cancelled intentionally and are already reported as
            # early_stop_dropped_episode_count, so do not misclassify them as
            # model/workflow failures.
            error_key = "termination_error_count"
            if error_key in raw_metrics:
                raw_metrics[error_key] = max(
                    0,
                    int(raw_metrics[error_key])
                    - int(stats.get("early_stop_dropped_episode_count", 0)),
                )
        return result

import asyncio
from rllm.tool_admission import ToolAdmission
import logging
import os
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from rllm.agents.agent import Episode
from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.engine.rollout_launch_order import rollout_major_launch_plan
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)

def _config_get(config, path: tuple[str, ...], default=None):
    """Read nested OmegaConf or plain mock configuration defensively."""
    current = config
    for key in path:
        if current is None:
            return default
        try:
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                getter = getattr(current, "get", None)
                if getter is not None:
                    value = getter(key, default)
                    current = default if value is None else value
                else:
                    current = getattr(current, key, default)
        except (AttributeError, KeyError, TypeError):
            return default
    return current


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _flag_is_set(value) -> bool:
    """Interpret bool-like fatal/masked values without treating ``"0"`` as true."""
    return _as_bool(value, default=False)


def _episode_trainable_steps(episode: Episode | None) -> int:
    if episode is None:
        return 0
    return sum(len(getattr(trajectory, "steps", ()) or ()) for trajectory in episode.trajectories)


def _episode_fatal_step_index(episode: Episode | None) -> int | None:
    """Return a validated fatal boundary, preferring workflow ``info``.

    ``fatal_step_index`` is a step boundary: steps before it remain trainable,
    while the boundary and everything after it are masked.  A value of zero
    therefore has no salvageable prefix and must never enter an actor update.
    """
    if episode is None:
        return None
    info = getattr(episode, "info", None) or {}
    metrics = getattr(episode, "metrics", None) or {}
    raw_index = info.get("fatal_step_index", metrics.get("fatal_step_index"))
    try:
        fatal_step_index = int(raw_index)
    except (TypeError, ValueError):
        return None
    return fatal_step_index if fatal_step_index >= 0 else None


def _episode_is_strictly_valid(episode: Episode | None) -> bool:
    """Return whether an episode is safe to occupy one training group slot.

    Normal episodes must end cleanly.  Fatal-aware episodes are also admitted
    when they carry a non-empty learnable prefix: their suffix is masked in
    ``transform_results_for_verl`` and their prefix advantage is one-sided in
    the trainer.  Rejecting every fatal episode here would silently disable the
    fatal-aware algorithm whenever strict group completion is enabled.
    """
    if episode is None:
        return False
    termination = getattr(episode, "termination_reason", None)
    termination = getattr(termination, "value", termination)
    trainable_steps = _episode_trainable_steps(episode)
    if trainable_steps < 1:
        return False
    info = getattr(episode, "info", None) or {}
    metrics = getattr(episode, "metrics", None) or {}
    if _flag_is_set(info.get("masked", metrics.get("masked", 0))):
        return False
    if _flag_is_set(metrics.get("masked", info.get("masked", 0))):
        return False
    is_fatal = _flag_is_set(
        info.get("is_fatal", metrics.get("is_fatal", False))
    ) or _flag_is_set(metrics.get("is_fatal", info.get("is_fatal", False)))
    if is_fatal:
        fatal_step_index = _episode_fatal_step_index(episode)
        return (
            fatal_step_index is not None
            and 0 < fatal_step_index <= trainable_steps
        )
    return termination == TerminationReason.ENV_DONE.value


def _episode_attempt_reason(episode: Episode | None) -> dict:
    info = (getattr(episode, "info", None) or {}) if episode is not None else {}
    metrics = (getattr(episode, "metrics", None) or {}) if episode is not None else {}
    termination = getattr(episode, "termination_reason", None)
    termination = getattr(termination, "value", termination)
    return {
        "termination_reason": str(termination or TerminationReason.UNKNOWN.value),
        "is_fatal": _flag_is_set(info.get("is_fatal", metrics.get("is_fatal", False)))
        or _flag_is_set(metrics.get("is_fatal", info.get("is_fatal", False))),
        "masked": _flag_is_set(info.get("masked", metrics.get("masked", 0)))
        or _flag_is_set(metrics.get("masked", info.get("masked", 0))),
        "fatal_step_index": _episode_fatal_step_index(episode),
        "fatal_reason": info.get("fatal_reason", metrics.get("fatal_reason", "")) or "",
        "mask_reason": info.get("mask_reason", metrics.get("mask_reason", "")) or "",
        "trainable_steps": _episode_trainable_steps(episode),
    }


def _apply_fatal_step_mask(
    response_mask: torch.Tensor, fatal_step_index: int
) -> torch.Tensor:
    """Zero out tokens for assistant steps >= *fatal_step_index* in a
    cumulative ``response_mask``.

    In cumulative mode the mask alternates between blocks of **1** (assistant
    tokens) and **0** (user / observation tokens).  Each contiguous 1-block
    corresponds to one agent step.  This helper walks through the mask,
    counts 1-blocks, and zeros everything from the *fatal_step_index*-th
    block onward.
    """
    mask = response_mask.clone()
    step = 0
    prev_val = 0
    for t in range(len(mask)):
        val = mask[t].item()
        if val == 1 and prev_val == 0:
            if step >= fatal_step_index:
                mask[t:] = 0
                return mask
        if val == 0 and prev_val == 1:
            step += 1
        prev_val = val
    return mask


class AgentWorkflowEngine:
    def __init__(
        self,
        workflow_cls: type[Workflow],
        workflow_args: dict,
        rollout_engine: RolloutEngine,
        config=None,
        n_parallel_tasks: int = 128,
        n_parallel_tools: int | None = None,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger=None,
        **kwargs,
    ):
        """Initialize the AgentWorkflowEngine.

        Args:
            workflow_cls: The workflow class to instantiate for each task.
            workflow_args: Arguments to pass to workflow instances.
            rollout_engine: Engine for model inference and rollout.
            config: Optional configuration object for training.
            n_parallel_tasks: Number of parallel workflow instances to maintain.
            n_parallel_tools: Shared blocking-tool worker count. Defaults to
                ``min(max(32, n_parallel_tasks), RLLM_MAX_TOOL_WORKERS)``;
                provider-specific bulkheads impose the actual backend limits.
            retry_limit: Maximum number of retry attempts for failed tasks.
            raise_on_error: Whether to raise exceptions on permanent failures.
            episode_logger: Optional logger for saving episode data to files.
            **kwargs: Additional keyword arguments.
        """
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args or {}

        self.rollout_engine = rollout_engine
        self.config = config  # if training
        self.require_all_valid = _as_bool(
            _config_get(
                config,
                ("rllm", "workflow", "require_all_valid"),
                False,
            )
        )
        raw_max_attempts = _config_get(
            config,
            ("rllm", "workflow", "all_valid_max_attempts"),
            8,
        )
        try:
            self.all_valid_max_attempts = int(raw_max_attempts)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"all_valid_max_attempts must be an integer in [1, 8], "
                f"got {raw_max_attempts!r}"
            ) from exc
        if not 1 <= self.all_valid_max_attempts <= 8:
            raise ValueError(
                f"all_valid_max_attempts must be in [1, 8], "
                f"got {self.all_valid_max_attempts}"
            )

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        if n_parallel_tools is None:
            try:
                worker_cap = max(
                    1,
                    int(os.environ.get("RLLM_MAX_TOOL_WORKERS", "512")),
                )
            except (TypeError, ValueError):
                worker_cap = 512
            self.n_parallel_tools = min(
                max(32, self.n_parallel_tasks),
                worker_cap,
            )
        else:
            self.n_parallel_tools = max(1, int(n_parallel_tools))
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tools)
        self.workflow_queue = None
        self.tool_admission = ToolAdmission(self.n_parallel_tasks)

        # Episode logging support
        self.episode_logger = episode_logger
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"  # "train" or "val"

        # Stores the raw Episode list from the most recent execute_tasks call
        self._last_episodes: list[Episode] = []
        self._reward_state = None

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        """Set current training step for episode logging.

        Args:
            step: Current training step number
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    async def _apply_rewards(self, episodes) -> None:
        """Apply the current reward before logging and conversion to Verl."""
        if not _as_bool(os.environ.get("FORMAL_EVGR_ENABLED", "0")):
            return
        if self.current_mode == "val" and not _as_bool(os.environ.get("FORMAL_EVGR_SCORE_VALIDATION", "0")):
            return
        from vision_deepresearch_async_workflow import reward
        from vision_deepresearch_async_workflow.reward_client import JudgeConfig
        from vision_deepresearch_async_workflow.reward_data import get_gt_index
        from vision_deepresearch_async_workflow.reward_config import RewardWeights
        from vision_deepresearch_async_workflow.reward_preflight import validate_environment

        validate_environment()
        weights = RewardWeights.from_env()
        max_concurrency = int(os.environ.get("FORMAL_EVGR_MAX_CONCURRENCY", "32"))
        if self._reward_state is None:
            data_root = os.environ.get("FORMAL_EVGR_DATA_ROOT", "").strip()
            if not data_root:
                raise ValueError("FORMAL_EVGR_DATA_ROOT is required when rewards are enabled")
            self._reward_state = (get_gt_index(data_root), reward.RewardJudge(JudgeConfig.from_env()))

        index, judge = self._reward_state
        scores = await reward.score_batch(episodes, index, judge, max_concurrency=max_concurrency)
        applied = attempts = 0
        for episode, score in zip(episodes, scores, strict=True):
            if episode is None or score is None:
                continue
            reward.apply_reward(episode, score, mode="both", process_weight=weights.evgr)
            applied += 1
            attempts += score["judge_attempts"]
        logger.warning("[reward] scored=%d/%d api_calls=%d model=%s",
                       applied, len(episodes), attempts, judge.config.model)

    async def initialize_pool(self):
        """Initialize the workflow pool with parallel workflow instances.

        Creates and populates the workflow queue with workflow instances
        for parallel task processing. This method is idempotent and will
        not recreate the pool if it already exists.
        """
        if self.workflow_queue is not None:
            return
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for i in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(
                rollout_engine=self.rollout_engine,
                executor=self.executor,
                **self.workflow_args,
            )
            assert (
                workflow.is_multithread_safe()
            ), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)

    async def process_task_with_retry(
        self, task: dict, task_id: str, rollout_idx: int, **kwargs
    ) -> tuple[str, int, Episode]:
        """Run one stable task/slot UID, optionally backfilling invalid episodes."""
        workflow = await self.workflow_queue.get()
        uid = f"{task_id}:{rollout_idx}"
        hard_mode = (
            self.require_all_valid
            and self.current_mode != "val"
            and not _as_bool(kwargs.get("validate"), default=False)
            and not _as_bool(getattr(self.rollout_engine, "validate", False))
        )
        max_attempts = self.all_valid_max_attempts if hard_mode else self.retry_limit
        attempt_lineage = []
        episode = None
        admitted = False
        try:
            await self.tool_admission.acquire()
            admitted = True
            for attempt_count in range(1, max_attempts + 1):
                episode = await workflow.run_with_termination_handling(
                    task=task, uid=uid, **kwargs
                )
                attempt_reason = _episode_attempt_reason(episode)
                attempt_reason["attempt"] = attempt_count
                attempt_lineage.append(attempt_reason)

                rewards_str = ", ".join(
                    f"{traj.name}: {traj.reward:.1f}"
                    for traj in (episode.trajectories if episode is not None else [])
                )
                colorful_print(
                    f"[{uid}] Rollout completed. Rewards: {rewards_str}, "
                    f"Termination: {getattr(episode, 'termination_reason', None)}",
                    fg="green"
                    if getattr(episode, "is_correct", False)
                    else "yellow",
                )

                if hard_mode:
                    if _episode_is_strictly_valid(episode):
                        episode.info = dict(getattr(episode, "info", None) or {})
                        episode.info.update(
                            {
                                "attempt_count": attempt_count,
                                "backfill_attempt_count": attempt_count - 1,
                                "attempt_lineage": attempt_lineage,
                            }
                        )
                        return task_id, rollout_idx, episode
                    if attempt_count < max_attempts:
                        logger.warning(
                            "[%s] strict-valid retry %d/%d: termination=%s fatal=%s "
                            "fatal_reason=%s masked=%s trainable_steps=%d",
                            uid,
                            attempt_count,
                            max_attempts,
                            attempt_reason["termination_reason"],
                            attempt_reason["is_fatal"],
                            attempt_reason["fatal_reason"],
                            attempt_reason["masked"],
                            attempt_reason["trainable_steps"],
                        )
                        continue
                    summary = "; ".join(
                        "attempt {attempt}: termination={termination_reason}, "
                        "fatal={is_fatal}, fatal_reason={fatal_reason}, "
                        "masked={masked}, trainable_steps={trainable_steps}".format(**reason)
                        for reason in attempt_lineage
                    )
                    raise RuntimeError(
                        f"[{uid}] strict-valid rollout exhausted after "
                        f"{max_attempts} attempts; reasons: {summary}"
                    )

                # Preserve the existing default behavior: only ERROR episodes retry.
                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, episode

                _err = (episode.info or {}).get("error")
                error_tb = (
                    _err.get("traceback")
                    if isinstance(_err, dict)
                    else (_err or None)
                )
                if error_tb:
                    print(error_tb)
                if attempt_count < self.retry_limit:
                    print(
                        f"[{uid}] Rollout failed on attempt "
                        f"{attempt_count}/{self.retry_limit}, retrying..."
                    )
                    continue

            if not self.raise_on_error:
                print(
                    f"[{uid}] Rollout failed permanently after "
                    f"{self.retry_limit} attempts."
                )
            else:
                raise Exception(
                    f"[{uid}] Rollout failed permanently after "
                    f"{self.retry_limit} attempts."
                )
            return task_id, rollout_idx, episode
        finally:
            if admitted:
                self.tool_admission.release()
            await self.workflow_queue.put(workflow)

    async def execute_tasks(
        self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs
    ) -> list[Episode]:
        """Run asynchronous workflow execution with retry logic for multiple tasks.

        Args:
            tasks: List of task dictionaries to process.
            task_ids: Optional list of task identifiers. If None, UUIDs are generated.
            **kwargs: Additional arguments passed to individual task processing.

        Returns:
            list[Episode]: List of completed episodes from all tasks.
        """
        if self.workflow_queue is None:
            await self.initialize_pool()

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_states = defaultdict(
            lambda: {
                "idx": None,
                "task": None,
                "episodes": [],
                "completed": 0,
                "total_rollouts": 0,
                "is_complete": False,
            }
        )

        _stagger_s = float(os.getenv("ROLLOUT_STAGGER_S", "0.0"))
        _n_tasks = len(tasks)

        async def _staggered(coro, delay: float):
            if delay > 0:
                await asyncio.sleep(delay)
            return await coro

        submitted_slots = [None] * len(tasks)
        slot_episodes = {}

        futures = []
        idx_counter = 0
        launch_plan = rollout_major_launch_plan(tasks, task_ids)
        for i, launch in enumerate(launch_plan):
            task = launch.task
            task_id = launch.task_id
            rollout_idx = launch.rollout_index
            state = task_states[task_id]
            if state["idx"] is None:  # First time seeing this task_id
                state["idx"] = idx_counter
                state["task"] = task
                idx_counter += 1
            submitted_slots[launch.input_index] = (task_id, rollout_idx)
            delay = (
                i * _stagger_s / _n_tasks
                if _stagger_s > 0 and _n_tasks > 1
                else 0.0
            )
            coro = self.process_task_with_retry(task, task_id, rollout_idx, **kwargs)
            futures.append(
                asyncio.create_task(_staggered(coro, delay) if delay > 0 else coro)
            )
            state["total_rollouts"] = max(
                state["total_rollouts"], rollout_idx + 1
            )

        try:
            with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
                for future in asyncio.as_completed(futures):
                    task_id, rollout_idx, episode = await future
                    slot_episodes[(task_id, rollout_idx)] = episode

                    state = task_states[task_id]
                    state["episodes"].append(episode)
                    state["completed"] += 1
                    pbar.update(1)
        except BaseException:
            for future in futures:
                if not future.done():
                    future.cancel()
            await asyncio.gather(*futures, return_exceptions=True)
            raise

        # Reconstruct input order so each task/group/slot keeps its stable UID.
        results = [slot_episodes[slot] for slot in submitted_slots]

        await self._apply_rewards(results)

        # Keep a reference so the trainer can inspect raw episodes
        self._last_episodes = results

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                logger.info(
                    f"Logging {len(results)} episodes to step={self.current_step}, mode={self.current_mode}, epoch={self.current_epoch}"
                )
                self.episode_logger.log_episodes_batch(
                    results, self.current_step, self.current_mode, self.current_epoch
                )
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")
                import traceback

                traceback.print_exc()

        return results

    async def execute_tasks_verl(self, batch: "DataProto", **kwargs) -> "DataProto":
        """Execute tasks from a Verl DataProto batch and return results.

        Args:
            batch: Verl DataProto containing tasks and metadata.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            DataProto: Transformed results compatible with Verl training.
        """
        await self.rollout_engine.wake_up()
        previous_validate = getattr(self.rollout_engine, "validate", False)
        previous_mode = self.current_mode
        sleep_called = False
        is_validation = batch.meta_info.get("validate", False)
        try:
            if is_validation:
                self.rollout_engine.validate = True
                self.current_mode = "val"
            else:
                self.current_mode = "train"
            tasks = batch.non_tensor_batch["extra_info"].tolist()
            task_ids = batch.non_tensor_batch["task_ids"].tolist()
            results = await self.execute_tasks(
                tasks, task_ids, **kwargs
            )  # list of Episodes

            # Keep validation mode active through transformation so hard-mode
            # backfill is never applied to evaluation data.
            sleep_called = True
            await self.rollout_engine.sleep()
            return self.transform_results_for_verl(results, task_ids)
        finally:
            self.rollout_engine.validate = previous_validate
            try:
                if not sleep_called:
                    sleep_called = True
                    await self.rollout_engine.sleep()
            finally:
                self.current_mode = previous_mode

    def transform_results_for_verl(
        self, episodes: list[Episode], task_ids: np.ndarray
    ) -> "DataProto":
        """Transform episode results into Verl-compatible DataProto format.

        Args:
            episodes: List of completed episodes from workflow execution.
            task_ids: Array of task identifiers corresponding to episodes.

        Returns:
            DataProto: Formatted data ready for Verl training pipeline.
        """
        # Local import to keep verl optional
        from verl import DataProto
        from verl.utils.torch_functional import pad_sequence_to_length

        hard_mode = (
            getattr(self, "require_all_valid", False)
            and self.current_mode != "val"
            and not _as_bool(getattr(self.rollout_engine, "validate", False))
        )
        if hard_mode:
            if len(episodes) != len(task_ids):
                raise RuntimeError(
                    "strict-valid training transform slot count mismatch: "
                    f"episodes={len(episodes)} task_ids={len(task_ids)}"
                )
            expected_slot_ids = []
            next_rollout_idx = Counter()
            for task_id in task_ids:
                task_key = str(task_id)
                rollout_idx = next_rollout_idx[task_key]
                expected_slot_ids.append(f"{task_key}:{rollout_idx}")
                next_rollout_idx[task_key] += 1
            actual_episode_ids = [
                str(getattr(episode, "id", "")) for episode in episodes
            ]
            duplicate_ids = [
                episode_id
                for episode_id, count in Counter(actual_episode_ids).items()
                if count > 1
            ]
            if duplicate_ids:
                raise RuntimeError(
                    "strict-valid training transform received duplicate episode.id: "
                    + ", ".join(duplicate_ids)
                )
            if actual_episode_ids != expected_slot_ids:
                raise RuntimeError(
                    "strict-valid training transform episode/task/slot UID mismatch: "
                    f"expected={expected_slot_ids[:8]} actual={actual_episode_ids[:8]}"
                )
            invalid = [
                str(getattr(episode, "id", f"episode_{index}"))
                for index, episode in enumerate(episodes)
                if not _episode_is_strictly_valid(episode)
            ]
            if invalid:
                raise RuntimeError(
                    "strict-valid training transform received invalid episode(s): "
                    + ", ".join(invalid)
                )

        prompts = []
        responses = []
        traj_rewards = []
        step_rewards = []
        episode_ids = []
        trajectory_ids = []
        step_ids = []
        step_nums = []
        repeat_counts = []
        is_last_step = []
        is_correct = []
        traj_mask = []
        termination_reasons = []
        metrics = []
        multi_modal_inputs_list = []
        is_fatal_flags = []
        fatal_step_indices = []
        episode_task_ids = []
        raw_termination_counts = Counter()
        expected_rollouts_by_task = Counter(str(task_id) for task_id in task_ids)
        if hard_mode:
            configured_group_size = _config_get(
                getattr(self, "config", None),
                ("actor_rollout_ref", "rollout", "n"),
                None,
            )
            if configured_group_size is not None:
                try:
                    configured_group_size = int(configured_group_size)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "strict-valid training transform requires an integer "
                        f"actor_rollout_ref.rollout.n, got {configured_group_size!r}"
                    ) from exc
                if configured_group_size < 1:
                    raise RuntimeError(
                        "strict-valid training transform requires "
                        "actor_rollout_ref.rollout.n >= 1"
                    )
                mismatched_groups = {
                    task_id: count
                    for task_id, count in expected_rollouts_by_task.items()
                    if count != configured_group_size
                }
                if mismatched_groups:
                    raise RuntimeError(
                        "strict-valid training transform requires exactly "
                        f"{configured_group_size} slots per group; got "
                        f"{mismatched_groups}"
                    )
        attempt_episode_count = 0
        backfill_attempt_count = 0
        max_attempts_used = 0
        attempt_termination_counts = Counter()
        attempt_fatal_count = 0
        attempt_masked_count = 0
        attempt_zero_step_count = 0
        episode_id_counts = Counter(
            str(getattr(episode, "id", ""))
            for episode in episodes
            if episode is not None and getattr(episode, "id", None) is not None
        )
        duplicate_episode_id_count = sum(
            max(count - 1, 0) for count in episode_id_counts.values()
        )

        for i, episode in enumerate(episodes):
            if episode is not None:
                ep_info = getattr(episode, "info", None) or {}
                lineage = ep_info.get("attempt_lineage") or []
                for attempt in lineage:
                    if not isinstance(attempt, dict):
                        continue
                    reason = str(
                        attempt.get(
                            "termination_reason",
                            TerminationReason.UNKNOWN.value,
                        )
                    )
                    attempt_termination_counts[reason] += 1
                    if _flag_is_set(attempt.get("is_fatal", False)):
                        attempt_fatal_count += 1
                    if _flag_is_set(attempt.get("masked", 0)):
                        attempt_masked_count += 1
                    try:
                        if int(attempt.get("trainable_steps", 0)) <= 0:
                            attempt_zero_step_count += 1
                    except (TypeError, ValueError):
                        attempt_zero_step_count += 1
                try:
                    used_attempts = max(
                        1, int(ep_info.get("attempt_count", len(lineage) or 1))
                    )
                except (TypeError, ValueError):
                    used_attempts = len(lineage) or 1
                attempt_episode_count += used_attempts
                backfill_attempt_count += max(used_attempts - 1, 0)
                max_attempts_used = max(max_attempts_used, used_attempts)
            total_steps = 0
            raw_reason = (
                episode.termination_reason
                if episode is not None and episode.termination_reason is not None
                else TerminationReason.ERROR
            )
            raw_termination_counts[
                getattr(raw_reason, "value", str(raw_reason))
            ] += 1

            if episode is None:
                print(f"Episode {i} is None (failed task), dropping it from the batch")
                repeat_counts.append(0)
                continue

            if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
                # termination hits before an agent finishes it's first step
                # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
                # we delete the episode from the batch by setting repeat_counts to 0
                print(
                    f"Episode {episode.id} has no valid trajectories, dropping it from the batch"
                )
                repeat_counts.append(0)
                continue

            ep_info = episode.info or {}
            ep_is_fatal = ep_info.get("is_fatal", False)
            ep_fatal_step_idx = ep_info.get("fatal_step_index")

            for trajectory in episode.trajectories:
                name = trajectory.name
                trajectory_id = f"{task_ids[i]}_{name}"  # unique trajectory identifier e.g., 1234567890_solver

                if len(trajectory.steps) == 0:
                    logger.info(f"Trajectory {trajectory_id} has no steps, skipping")
                    continue

                if not self.config.rllm.stepwise_advantage.enable:
                    if len(trajectory.steps) > 1:
                        if not trajectory.is_cumulative():
                            logger.warning(
                                f"Warning: Multi-step trajectory {trajectory_id} is not cumulative, but stepwise mode is not enabled. There could be a token mismatch during trajectory generation."
                            )

                        chat_completions = trajectory.steps[-1].chat_completions
                        prompt, response, mask = (
                            self.rollout_engine.chat_parser.tokenize_and_mask_cumulative(
                                chat_completions
                            )
                        )
                        if ep_is_fatal and ep_fatal_step_idx is not None:
                            mask = _apply_fatal_step_mask(mask, ep_fatal_step_idx)
                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append({})  # empty dict

                    elif isinstance(trajectory.steps[0].model_output, ModelOutput):
                        step = trajectory.steps[0]

                        prompt_ids = torch.tensor(
                            step.model_output.prompt_ids, dtype=torch.long
                        )
                        prompts.append(prompt_ids)

                        response_ids = torch.tensor(
                            step.model_output.completion_ids, dtype=torch.long
                        )
                        responses.append(response_ids)

                        mask = torch.ones_like(response_ids, dtype=torch.long)
                        if ep_is_fatal and ep_fatal_step_idx == 0:
                            mask = torch.zeros_like(response_ids, dtype=torch.long)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append(
                            step.model_output.multi_modal_inputs or {}
                        )

                    else:
                        chat_completions = trajectory.steps[0].chat_completions
                        prompt, response, mask = (
                            self.rollout_engine.chat_parser.tokenize_and_mask(
                                chat_completions
                            )
                        )
                        if ep_is_fatal and ep_fatal_step_idx == 0:
                            mask = torch.zeros_like(mask)
                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append({})  # empty dict

                    step_rewards.append(trajectory.reward)
                    step_ids.append(trajectory_id)
                    n_steps = 1

                else:
                    for step_idx, step in enumerate(trajectory.steps):
                        _mask_this_step = (
                            ep_is_fatal
                            and ep_fatal_step_idx is not None
                            and step_idx >= ep_fatal_step_idx
                        )

                        if isinstance(step.model_output, ModelOutput):
                            prompt_ids = torch.tensor(
                                step.model_output.prompt_ids, dtype=torch.long
                            )
                            prompts.append(prompt_ids)

                            response_ids = torch.tensor(
                                step.model_output.completion_ids, dtype=torch.long
                            )
                            responses.append(response_ids)

                            mask = torch.ones_like(response_ids, dtype=torch.long)
                            if _mask_this_step:
                                mask = torch.zeros_like(response_ids, dtype=torch.long)
                            traj_mask.append(mask)
                            multi_modal_inputs_list.append(
                                step.model_output.multi_modal_inputs or {}
                            )

                        else:
                            chat_completions = step.chat_completions
                            prompt, response, mask = (
                                self.rollout_engine.chat_parser.tokenize_and_mask(
                                    chat_completions
                                )
                            )
                            if _mask_this_step:
                                mask = torch.zeros_like(mask)
                            prompts.append(prompt)
                            responses.append(response)
                            traj_mask.append(mask)
                            multi_modal_inputs_list.append({})  # empty dict

                        step_rewards.append(step.reward)
                        step_ids.append(
                            f"{trajectory_id}_step{step_idx}"
                        )  # unique step identifier e.g., 1234567890_solver_step0

                    n_steps = len(trajectory.steps)

                trajectory_ids.extend([trajectory_id] * n_steps)
                step_nums.extend([n_steps] * n_steps)
                traj_rewards.extend([trajectory.reward] * n_steps)
                is_last_step.extend([False] * n_steps)
                is_last_step[-1] = True
                is_fatal_flags.extend([ep_is_fatal] * n_steps)
                fatal_step_indices.extend(
                    [ep_fatal_step_idx if ep_fatal_step_idx is not None else -1]
                    * n_steps
                )
                total_steps += n_steps

            episode_ids.extend([episode.id] * total_steps)
            episode_task_ids.extend([str(task_ids[i])] * total_steps)
            is_correct.extend([episode.is_correct] * total_steps)
            termination_reasons.extend(
                [
                    (
                        episode.termination_reason
                        if episode.termination_reason is not None
                        else TerminationReason.UNKNOWN
                    )
                ]
                * total_steps
            )
            metrics.extend([episode.metrics] * total_steps)
            repeat_counts.append(total_steps)
        raw_rollout_metrics = {
            "raw_episode_count": len(episodes),
            "valid_episode_count": 0,
            "complete_group_count": 0,
            "attempt_episode_count": attempt_episode_count,
            "backfill_attempt_count": backfill_attempt_count,
            "max_attempts_used": max_attempts_used,
            "expected_group_count": len(expected_rollouts_by_task),
            "incomplete_group_count": 0,
            "valid_group_size_min": 0,
            "valid_group_size_max": 0,
            "duplicate_episode_id_count": duplicate_episode_id_count,
            "attempt_fatal_count": attempt_fatal_count,
            "attempt_masked_count": attempt_masked_count,
            "attempt_zero_step_count": attempt_zero_step_count,
            "non_masked_response_tokens": 0,
            "timeout_rate": (
                raw_termination_counts[TerminationReason.TIMEOUT.value]
                / max(len(episodes), 1)
            ),
        }
        raw_rollout_metrics.update(
            {
                f"termination_{reason.value}_count": raw_termination_counts[
                    reason.value
                ]
                for reason in TerminationReason
            }
        )
        raw_rollout_metrics.update(
            {
                f"attempt_termination_{reason.value}_count": (
                    attempt_termination_counts[reason.value]
                )
                for reason in TerminationReason
            }
        )

        # Guard: compact_filtering (mask_timeout + mask_error + mask_unknown) can
        # eliminate every trajectory in the batch.  pad_sequence on an empty list
        # raises RuntimeError("received an empty list of sequences"), crashing the
        # entire training job.  Instead, signal the trainer to skip this step via
        # an empty DataProto so the trainer's _empty_batch path handles it cleanly.
        if not prompts:
            logger.warning(
                "[transform_results_for_verl] all trajectories were masked (empty batch); "
                "returning empty DataProto so trainer skips this step"
            )
            from verl import DataProto
            empty = DataProto.from_single_dict({})
            empty.meta_info = {
                "repeat_counts": repeat_counts,
                "all_masked": True,
                "raw_rollout_metrics": raw_rollout_metrics,
            }
            return empty

        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in prompts],
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        ).flip(dims=[1])
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = pad_sequence_to_length(
            prompts_batch,
            max_prompt_length,
            self.rollout_engine.tokenizer.pad_token_id,
            left_pad=True,
        )
        prompts_batch = prompts_batch[:, -max_prompt_length:]  # truncate if necessary

        response_batch = torch.nn.utils.rnn.pad_sequence(
            responses,
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        )
        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(
            response_batch,
            max_response_length,
            self.rollout_engine.tokenizer.pad_token_id,
            left_pad=False,
        )
        response_batch = response_batch[
            :, :max_response_length
        ]  # truncate if necessary

        input_ids = torch.concat([prompts_batch, response_batch], dim=1)

        prompt_lengths = torch.as_tensor([len(t) for t in prompts]).clamp_(
            min=0, max=max_prompt_length
        )
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in responses]).clamp_(
            min=0, max=max_response_length
        )
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        if (
            hasattr(self.rollout_engine, "processor")
            and self.rollout_engine.processor is not None
        ):
            position_ids = self._handle_multimodal_position_ids(
                processor=self.rollout_engine.processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs_list,
            )
        else:
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        traj_mask = torch.nn.utils.rnn.pad_sequence(
            traj_mask, batch_first=True, padding_value=0
        )
        traj_mask = pad_sequence_to_length(
            traj_mask, max_response_length, 0, left_pad=False
        )
        traj_mask = traj_mask[:, :max_response_length]  # truncate if necessary

        # Place all rewards to last response token of the last_step response
        traj_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        step_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, (traj_reward, step_reward) in enumerate(
            zip(traj_rewards, step_rewards, strict=False)
        ):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= traj_rewards_batch.shape[1]:
                traj_rewards_batch[i, resp_len - 1] = traj_reward
                step_rewards_batch[i, resp_len - 1] = step_reward

        # compact filtering
        cf = self.config.rllm.compact_filtering
        is_valid = [True] * len(episode_ids)
        if cf.enable:
            for i in range(len(episode_ids)):
                termination_reason = termination_reasons[i]
                if (
                    (
                        cf.mask_max_prompt_length_exceeded
                        and termination_reason
                        == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
                    )
                    or (
                        cf.mask_max_response_length_exceeded
                        and termination_reason
                        == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
                    )
                    or (
                        cf.mask_env_done
                        and termination_reason == TerminationReason.ENV_DONE
                    )
                    or (
                        cf.mask_max_turns_exceeded
                        and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED
                    )
                    or (
                        cf.mask_timeout
                        and termination_reason == TerminationReason.TIMEOUT
                    )
                    or (
                        cf.mask_unknown
                        and termination_reason == TerminationReason.UNKNOWN
                    )
                    or (cf.mask_error and termination_reason == TerminationReason.ERROR)
                ):
                    is_valid[i] = (
                        False  # set flag to filter out the episode later (after advantages are computed)
                    )

        # Override: keep fatal trajectories that have a recoverable prefix.
        # Their response_mask already zeros post-fatal tokens; the trainer
        # will apply advantage clamping.
        for i in range(len(is_valid)):
            if (
                not is_valid[i]
                and i < len(is_fatal_flags)
                and is_fatal_flags[i]
                and fatal_step_indices[i] > 0
            ):
                is_valid[i] = True
        valid_episode_to_task = {}
        for row_index, row_is_valid in enumerate(is_valid):
            if row_is_valid:
                valid_episode_to_task[str(episode_ids[row_index])] = (
                    episode_task_ids[row_index]
                )
        valid_rollouts_by_task = Counter(valid_episode_to_task.values())
        valid_group_sizes = [
            valid_rollouts_by_task[task_id]
            for task_id in expected_rollouts_by_task
        ]
        incomplete_group_count = sum(
            valid_rollouts_by_task[task_id] < expected_rollouts
            for task_id, expected_rollouts in expected_rollouts_by_task.items()
        )
        valid_row_mask = torch.as_tensor(
            is_valid, dtype=torch.bool, device=traj_mask.device
        )
        raw_rollout_metrics.update(
            {
                "valid_episode_count": len(valid_episode_to_task),
                "complete_group_count": sum(
                    valid_rollouts_by_task[task_id] >= expected_rollouts
                    for task_id, expected_rollouts in expected_rollouts_by_task.items()
                ),
                "incomplete_group_count": incomplete_group_count,
                "valid_group_size_min": min(valid_group_sizes)
                if valid_group_sizes
                else 0,
                "valid_group_size_max": max(valid_group_sizes)
                if valid_group_sizes
                else 0,
                "non_masked_response_tokens": int(
                    traj_mask[valid_row_mask].sum().item()
                ),
            }
        )

        non_tensors = {
            "episode_ids": np.array(episode_ids),  # unique identifier for each rollout
            "trajectory_ids": np.array(
                trajectory_ids
            ),  # unique identifier for each trajectory (shares prefix with task_id) and shared across rollouts
            "step_ids": np.array(
                step_ids
            ),  # unique identifier for each step (shares prefix with task_id) and shared across rollouts
            "batch_ids": np.array(
                [str(uuid.uuid4())] * len(episode_ids)
            ),  # unique identifier for each batch
            "step_nums": np.array(step_nums),
            "is_correct": np.array(is_correct),
            "termination_reasons": np.array([x.value for x in termination_reasons]),
            "metrics": np.array(metrics),
            "is_valid": np.array(is_valid),
            "is_last_step": np.array(is_last_step),
            "is_pad_step": np.array([False] * len(episode_ids)),
            "is_fatal": np.array(is_fatal_flags) if is_fatal_flags else np.array([False] * len(episode_ids)),
        }

        if any(mm_inputs is not None for mm_inputs in multi_modal_inputs_list):
            non_tensors["multi_modal_inputs"] = np.array(
                multi_modal_inputs_list, dtype=object
            )

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_batch,
                "responses": response_batch,
                "response_mask": traj_mask,
                "traj_rewards": traj_rewards_batch,
                "step_rewards": step_rewards_batch,
            },
            non_tensors=non_tensors,
            meta_info={
                "repeat_counts": repeat_counts,
                "raw_rollout_metrics": raw_rollout_metrics,
            },
        )

    def _handle_multimodal_position_ids(
        self,
        processor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: list[dict],
    ) -> torch.Tensor:
        """Handle multimodal position ids calculation. Borrowed from verl.utils.dataset.rl_dataset.py"""
        batch_size = input_ids.shape[0]
        position_ids_list = []

        if (
            processor is not None
            and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
        ):
            # qwen-vl mrope
            if "Qwen3VLProcessor" in processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            for i in range(batch_size):
                model_inputs = (
                    multi_modal_inputs[i] if i < len(multi_modal_inputs) else {}
                )
                vision_position_ids = get_rope_index(
                    processor,
                    input_ids=input_ids[i],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[i],
                )  # (3, seq_length)
                valid_mask = attention_mask[i].bool()
                text_position_ids = torch.ones((1, len(input_ids[i])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids_list.append(
                    torch.cat((text_position_ids, vision_position_ids), dim=0)
                )  # (4, seq_length)

        else:
            # Fallback: should not reach here if called correctly
            raise ValueError(
                f"Unsupported processor type: {processor.__class__.__name__ if processor else None}"
            )

        # Stack all position_ids to form batch: (batch_size, 4, seq_length)
        position_ids = torch.stack(position_ids_list, dim=0)
        return position_ids

    def shutdown(self):
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

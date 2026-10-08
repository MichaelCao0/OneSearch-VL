from io import BytesIO
import asyncio
import contextvars
import json
import os
import re
import traceback
import urllib.request
import time
from typing import Any, List, Optional

from PIL import Image
from vision_deepresearch_async_workflow.reward_config import RewardWeights
from vision_deepresearch_async_workflow.query_reward import (
    _QUERY_SYSTEM_PROMPT_TEMPLATE, _build_trajectory_summary, build_query_judge_prompt, parse_query,
)

from vision_deepresearch_async_workflow.deepresearch_agent import DeepResearchAgent
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationReason, Workflow

import base64


def as_pil_image(image: Any) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, str) and image.startswith("data:image/"):
        try:
            _, encoded = image.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            try:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        data_str = None
        if "data" in image and isinstance(image["data"], str):
            data_str = image["data"]
        elif "path" in image and isinstance(image["path"], str):
            data_str = image["path"]
        elif "url" in image and isinstance(image["url"], str):
            data_str = image["url"]
        if data_str:
            if data_str.startswith("data:image/"):
                try:
                    _, encoded = data_str.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception:  # noqa: BLE001
                    return None
            try:
                return Image.open(data_str).convert("RGB")
            except Exception:  # noqa: BLE001
                return None

    if isinstance(image, str):
        try:
            return Image.open(image).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    return None


def _extract_action_from_response(response: str) -> Action:
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<response>" in response and "</response>" in response:
        answer = response.split("<response>")[1].split("</response>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})


def _is_valid_format(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    pattern = (
        r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<response>.*?</response>|<answer>.*?</answer>)\s*$"
    )
    return re.match(pattern, content, re.DOTALL) is not None


def _format_reward_for_step(step: Step) -> float:
    if step.info.get("step_error"):
        return 0.0
    if _has_tool_error_observation(step.observation):
        return 0.0
    content = step.model_response if isinstance(step.model_response, str) else ""
    return 1.0 if _is_valid_format(content) else 0.0


def _has_tool_error_observation(observation: Any) -> bool:
    if not isinstance(observation, str):
        return False
    error_markers = (
        "[Json Parse Error]",
        "Tool execution error:",
        "Error executing",
        "Error: Image reference",
        "Error: OpenCV not available",
    )
    return any(marker in observation for marker in error_markers)


def _is_step_error(step: Step) -> bool:
    if step.info.get("step_error"):
        return True
    return _has_tool_error_observation(step.observation)


# ---------------------------------------------------------------------------
# Query-utility reward: LLM judge evaluates search trajectory quality
# ---------------------------------------------------------------------------

_QUERY_JUDGE_MARKER = os.getenv("JUDGE_MODEL_MARKER", os.getenv("JUDGE_MODEL", "gpt-4o-mini"))


def _query_reward_enabled() -> bool:
    return os.getenv("QUERY_REWARD_ENABLED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )

async def _judge_query_utility(
    question: str,
    ground_truth: str,
    prediction: str,
    steps: list[Step],
) -> float:
    """Call LLM judge to evaluate search query utility. Returns 0.0-1.0."""

    trajectory_summary = _build_trajectory_summary(steps)
    if trajectory_summary == "(No tool calls in trajectory)":
        return 0.0

    prompt = build_query_judge_prompt(
        question=question,
        ground_truth=ground_truth or "(not provided)",
        prediction=prediction or "(no prediction)",
        trajectory_summary=trajectory_summary,
    )

    try:
        from vision_deepresearch_async_workflow.utils.api_gateway_client import (
            api_gateway_chat,
            is_api_gateway_configured,
        )

        if not is_api_gateway_configured():
            print("[QueryUtilityReward] API gateway not configured, defaulting to 0.0")
            return 0.0

        messages = [
            {"role": "system", "content": _QUERY_SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": prompt},
        ]

        data = await asyncio.to_thread(
            api_gateway_chat,
            messages=messages,
            model_marker=_QUERY_JUDGE_MARKER,
            max_tokens=500,
            timeout=60,
        )

        choices = data.get("choices") or [{}]
        content = ""
        first_choice = choices[0] or {}
        if isinstance(first_choice, dict):
            msg = first_choice.get("message")
            if isinstance(msg, dict):
                content = msg.get("content", "")

        score = parse_query(content)
        if score is not None:
            print(f"[QueryUtilityReward] score={score:.2f}")
            return score

        print(f"[QueryUtilityReward] Failed to parse score from: {content[:200]}")
        return 0.0

    except Exception as exc:  # noqa: BLE001
        print(f"[QueryUtilityReward] Judge call failed: {exc}")
        return 0.0


def _get_next_observation(messages: list[dict], current_index: int) -> str:
    if current_index + 1 < len(messages):
        next_msg = messages[current_index + 1]
        if next_msg["role"] == "user" and "<tool_response>" in next_msg["content"]:
            return next_msg["content"]
    return ""


def _map_termination_reason(termination: str) -> TerminationReason:
    mapping = {
        "answer": TerminationReason.ENV_DONE,
        "timeout": TerminationReason.UNKNOWN,
        "max_rounds_reached": TerminationReason.UNKNOWN,
        "token_limit_no_answer": TerminationReason.UNKNOWN,
        "answer_token_limit": TerminationReason.UNKNOWN,
        "exceed available llm calls": TerminationReason.UNKNOWN,
        "prompt_budget_reached": TerminationReason.UNKNOWN,
        "max_rounds_reached_no_answer": TerminationReason.UNKNOWN,
        "repetition_detected": TerminationReason.UNKNOWN,  # Will be masked
        "tool_errors_too_many": TerminationReason.UNKNOWN,  # Will be masked
        "consecutive_step_errors": TerminationReason.UNKNOWN,  # Will be masked
        "error": TerminationReason.UNKNOWN,  # Will be masked
    }
    return mapping.get(termination, TerminationReason.UNKNOWN)


def _evaluate_answer(prediction: str, ground_truth: str) -> bool:
    if not prediction or not ground_truth:
        return False
    return prediction.strip().lower() == ground_truth.strip().lower()


def _should_mask_episode(result: dict, episode: Episode) -> tuple[bool, str]:
    """Determine if the entire episode should be masked based on answer/step error conditions."""
    steps = episode.trajectories[0].steps if episode.trajectories else []
    termination = result.get("termination", "")

    # Mask directly if no final answer is produced.
    if termination != "answer":
        return True, termination or "no_final_answer"

    # Check if there are too many step errors.
    total_steps = len(steps)

    if total_steps > 0:
        step_error_steps = sum(1 for step in steps if _is_step_error(step))
        if step_error_steps / total_steps > 0.5:
            return True, "tool_errors_too_many"

    return False, ""


# ---------------------------------------------------------------------------
# Step-level fatal-after detection (replaces trajectory-level hard mask)
# ---------------------------------------------------------------------------

_CONSECUTIVE_ERROR_THRESHOLD = 3


def _find_fatal_step_index(
    steps: list[Step], termination: str
) -> tuple[int | None, str]:
    """Detect the first fatal step using a consecutive-error counter.

    Returns ``(fatal_step_index, reason)`` where *fatal_step_index* is the
    0-based index of the **first step to mask** (this step and all subsequent
    steps will be masked).  Returns ``(None, "")`` for normal trajectories
    that terminated with an answer and contain no error cascades.

    A single error followed by recovery resets the counter — only cascading
    failures (>= threshold consecutive errors) trigger masking.
    """
    if not steps:
        return 0, "no_steps"

    consecutive_errors = 0

    for idx, step in enumerate(steps):
        if _is_step_error(step):
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        if consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD:
            fatal_start = idx - _CONSECUTIVE_ERROR_THRESHOLD + 1
            return fatal_start, "consecutive_errors"

    if termination == "answer":
        return None, ""

    # Abnormal termination without error cascade — all existing steps are
    # usable but the trajectory is incomplete.  Mark fatal *after* the last
    # step so every step is kept, yet the trajectory is still flagged for
    # advantage clamping.
    return len(steps), termination or "no_final_answer"


def _extract_video_grid(task: dict):
    """Build a runtime frame-grid handle from a dataset sample.

    Accepts a ready grid dict under ``video_grid`` / ``grid`` / ``video`` (with
    ``frame_urls`` or ``object_urls`` + ``video_url``) and normalises it via
    ``tools.video_io.build_grid_from_manifest``.

    Returns ``(grid, error)``:
      * ``(None, None)``  — the sample carries no video field (image-only task);
                            the rollout proceeds unchanged.
      * ``(grid, None)``  — a valid grid was built.
      * ``(None, msg)``   — a video field WAS present but malformed. The caller
                            must NOT silently treat this as an image-only task:
                            a broken video sample should be flagged so its
                            trajectory can be masked instead of training on a
                            video task with no usable video handle.
    """
    record = None
    for key in ("video_grid", "grid", "video"):
        cand = task.get(key)
        if isinstance(cand, dict):
            record = cand
            break
    if record is None:
        return None, None

    try:
        from vision_deepresearch_async_workflow.tools import video_io

        return video_io.build_grid_from_manifest(record), None
    except Exception as exc:  # noqa: BLE001
        msg = f"malformed video grid ({type(exc).__name__}: {exc})"
        print(f"[workflow] _extract_video_grid failed: {msg}")
        return None, msg


# Best-effort timeout for fetching bootstrap frame images from a grid.
_BOOTSTRAP_FETCH_TIMEOUT = 30


def _resize_frame_to_max_pixels(pil: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale a frame so its pixel count <= max_pixels, keeping aspect ratio.

    Dimensions are rounded to multiples of 28 for the Qwen3-VL processor.
    Upscaling is skipped.
    """
    w, h = pil.size
    if w * h <= max_pixels:
        return pil
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(28, int((w * scale) // 28) * 28)
    new_h = max(28, int((h * scale) // 28) * 28)
    return pil.resize((new_w, new_h), Image.BICUBIC)


def _bootstrap_frames_from_grid(
    grid: dict, max_frames: int = 128, max_pixels: int = 100352
) -> List[Image.Image]:
    """Fetch up to ``max_frames`` evenly-spaced frames from a grid as PIL images.

    The frames provide a visual overview for subsequent frame selection.
    Unavailable frames are skipped; an empty result provides no overview.
    """
    urls = (grid or {}).get("frame_urls") or []
    n = len(urls)
    if n == 0:
        return []
    k = min(max_frames, n)
    # Evenly spaced indices across the whole grid (always includes first/last).
    if k == 1:
        idxs = [0]
    else:
        idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    seen = set()
    pil_images: List[Image.Image] = []
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        url = urls[i]
        if not isinstance(url, str):
            continue
        try:
            if url.startswith(("http://", "https://")):
                with urllib.request.urlopen(url, timeout=_BOOTSTRAP_FETCH_TIMEOUT) as resp:
                    data = resp.read()
                pil = Image.open(BytesIO(data)).convert("RGB")
            else:
                pil = Image.open(url).convert("RGB")
            pil_images.append(_resize_frame_to_max_pixels(pil, max_pixels))
        except Exception as exc:  # noqa: BLE001
            print(f"[workflow] bootstrap frame {i} fetch failed: {type(exc).__name__}: {exc}")
            continue
    return pil_images


def _to_pil_image(img: Any) -> Optional[Image.Image]:
    """Best-effort conversion to PIL.Image for downstream multi-modal pipeline."""
    if isinstance(img, Image.Image):
        return img
    pil = as_pil_image(img)
    if pil is not None:
        if isinstance(img, str):
            # Preserve immutable source identity after eager loading/conversion;
            # ChatTemplateParser uses this to deduplicate fetch_image work
            # across the N responses sampled from the same dataset row.
            pil.info["_rllm_immutable_source"] = img
        return pil
    if isinstance(img, dict) and "bytes" in img:
        try:
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(img, str):
        try:
            pil = Image.open(img).convert("RGB")
            pil.info["_rllm_immutable_source"] = img
            return pil
        except Exception:  # noqa: BLE001
            return None
    return None


def _attach_timing_metrics(
    episode: Episode,
    *,
    workflow_started: float,
    workflow_prepare_s: float,
    agent_wall_s: float,
    reward_function_s: float,
    query_judge_s: float,
) -> None:
    """Expose per-trajectory stage timings through the trainer's batch metrics."""
    timing = episode.info.setdefault("timing", {})
    workflow_total_s = time.perf_counter() - workflow_started
    llm_total_s = float(timing.get("llm_total_s", 0.0))
    tool_total_s = float(timing.get("tool_total_s", 0.0))
    unattributed_s = max(
        0.0,
        workflow_total_s
        - llm_total_s
        - tool_total_s
        - reward_function_s
        - query_judge_s,
    )
    timing.update(
        {
            "workflow_total_s": workflow_total_s,
            "workflow_prepare_s": workflow_prepare_s,
            "agent_wall_s": agent_wall_s,
            "reward_function_s": reward_function_s,
            "query_judge_s": query_judge_s,
            "unattributed_s": unattributed_s,
        }
    )
    for key, value in timing.items():
        if isinstance(value, (int, float)):
            episode.metrics[f"timing/{key}"] = float(value)
    for key, value in episode.info.get("tool_statistics", {}).items():
        episode.metrics[f"tools/{key}"] = float(value)


_ENGINE_GAUGE_SUFFIXES = (
    "_mm_preprocess_max_active",
    "_mm_preprocess_max_waiters",
    "_mm_source_cache_entries",
    "_mm_source_cache_bytes",
)


def _merge_engine_timing(
    totals: dict[str, float], key: str, value: float
) -> None:
    """Merge one per-turn engine metric using its actual metric semantics.

    Durations and counters are additive across turns. Peak concurrency and
    cache size are gauges; summing them made a six-turn trajectory report a
    bogus peak greater than the configured semaphore capacity.
    """
    numeric_value = float(value)
    if key.endswith(_ENGINE_GAUGE_SUFFIXES):
        totals[key] = max(totals.get(key, numeric_value), numeric_value)
    else:
        totals[key] = totals.get(key, 0.0) + numeric_value


class DeepResearchWorkflow(Workflow):
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        tools: dict | None = None,
        system_prompt: str | None = None,
        reward_function: RewardFunction | None = None,
        **kwargs,
    ):
        self.reward_weights = RewardWeights.from_env()
        self.reward_weights.validate_enabled()
        super().__init__(rollout_engine, executor, **kwargs)

        self.tools = tools or {}
        if system_prompt is None:
            from vision_deepresearch_async_workflow.tool_contract import validate_sft_tools
            validate_sft_tools(self.tools)
        for tool in self.tools.values():
            if hasattr(tool, "set_executor"):
                tool.set_executor(self.executor)
        self.system_prompt = system_prompt
        self.reward_function = reward_function

        self.agent = DeepResearchAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
            executor=self.executor,
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)
        workflow_started = time.perf_counter()
        workflow_prepare_s = 0.0
        agent_started: float | None = None
        agent_wall_s = 0.0
        reward_function_s = 0.0
        query_judge_s = 0.0

        question = task.get("question", task.get("query", "No question provided"))
        answer = task.get("answer", "")

        print(f"🚀 Starting DeepResearch workflow for task {uid}")
        print(f"   Question: {question}")

        try:
            raw_images = None
            if "images" in task:
                raw_images = task.get("images")

            # ---- Video Deep Research: build the video_1 grid handle ----
            # A sample can carry a frame-grid either as a ready dict under
            # ``video_grid`` / ``grid`` or as a raw manifest record under
            # ``video`` (with frame_urls/object_urls + video_url). We normalise
            # to the runtime grid dict so select_timespan / select_frame work.
            video_grid, grid_error = _extract_video_grid(task)
            # Preserve one stable rollout application id across every LLM turn
            # in this trajectory.  Without this, VerlEngine generates a fresh
            # UUID per call, defeating AsyncLLMServerManager sticky routing and
            # SGLang prefix/KV-cache reuse for multi-turn conversations.
            agent_kwargs = dict(kwargs)
            agent_kwargs.setdefault("application_id", uid)
            if grid_error is not None:
                # A video field was present but unusable. The sample's episode
                # is flagged for masking after it is built (see below) so RL
                # does not train on a "video task" the model can never see.
                print(f"⚠️  task {uid}: {grid_error}; proceeding without video handle")

            # ---- Infer modality for prompt selection ----
            # Priority: explicit task["modality"] field > structural inference.
            # Prepared RL data tags every row with an explicit
            # ``modality`` (video / single_image / multi_image); this branch is
            # only a fallback for older data lacking the field.
            # Structural rules (mirror onesearch_vl pipeline routing):
            #   valid video_grid, or a video field that failed to parse → "video"
            #   ≥2 images, no video                 → "multi_image"
            #   single image, no video              → "single_image"
            # NOTE: test the VALUE, not key presence. On the Mixed dataset every
            # row carries both ``images`` and ``video_grid`` columns (the unused
            # one is None), so ``"video_grid" in task`` is always True and would
            # misroute image rows to the video pipeline.
            task_modality = (task.get("modality") or "").strip().lower()
            if task_modality in ("video", "multi_image", "single_image"):
                modality = task_modality
            elif video_grid is not None or grid_error is not None:
                modality = "video"
            else:
                raw_image_count = len(raw_images) if isinstance(raw_images, list) else (1 if raw_images else 0)
                modality = "multi_image" if raw_image_count >= 2 else "single_image"

            pil_images: List[Image.Image] = []
            if raw_images is not None:
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                for img in raw_images:
                    pil = _to_pil_image(img)
                    if pil is not None:
                        pil_images.append(pil)
                if len(pil_images) != len(raw_images):
                    raise ValueError(
                        "failed to load every initial image: "
                        f"expected={len(raw_images)} loaded={len(pil_images)}"
                    )

            # Video-only sample: pass video_url directly to the model, matching
            # SFT/infer behaviour (VIDEO_FPS=2, VIDEO_MAX_PIXELS=100352 by default,
            # overridable via env vars). This avoids the bootstrap-frames approach
            # (fetching static PNGs) which diverges from the training distribution.
            video_inputs: list = []
            if not pil_images and video_grid is not None:
                video_url = (video_grid or {}).get("video_url")
                if video_url:
                    video_fps = float(os.environ.get("VIDEO_FPS", "2.0"))
                    video_max_pixels = int(os.environ.get("VIDEO_MAX_PIXELS", "100352"))
                    video_max_frames = int(os.environ.get("VIDEO_MAX_FRAMES", "128"))
                    video_inputs = [{
                        "video": video_url,
                        "fps": video_fps,
                        "min_pixels": 256,
                        "max_pixels": video_max_pixels,
                        "max_frames": video_max_frames,
                        "n_frames": int(video_grid.get("n_frames") or 0),
                    }]

            workflow_prepare_s = time.perf_counter() - workflow_started
            agent_started = time.perf_counter()
            if pil_images:
                # ``image_path`` is the legacy single-image handle.  Passing
                # the first path for a multi-image task prevents the agent from
                # registering the remaining inputs as img_2..img_n.  Leave it
                # unset for multi-image tasks so the complete PIL list becomes
                # the initial tool-handle set.
                single_image_path = (
                    raw_images[0]
                    if modality == "single_image" and len(raw_images) == 1
                    else None
                )
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    images=pil_images,
                    image_path=single_image_path,
                    video_grid=video_grid,
                    modality=modality,
                    **agent_kwargs,
                )
            elif video_inputs:
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    videos=video_inputs,
                    video_grid=video_grid,
                    modality=modality,
                    **agent_kwargs,
                )
            else:
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    video_grid=video_grid,
                    modality=modality,
                    **agent_kwargs,
                )
            agent_wall_s = time.perf_counter() - agent_started

            episode = self._convert_result_to_episode(result, task, uid)

            # A malformed video grid means the model answered a video question
            # with no usable video handle — mask the trajectory so RL does not
            # train on it (mirrors the hard-mask path used for degenerate
            # episodes below).
            if grid_error is not None:
                episode.info["video_grid_error"] = grid_error
                episode.info["is_fatal"] = True
                episode.info["fatal_step_index"] = 0
                episode.info["mask_reason"] = "malformed_video_grid"
                episode.info["masked"] = 1.0

            prediction = result.get("prediction", "")
            if self.reward_function is not None and prediction:
                # The workflow may infer modality for legacy rows that do not
                # carry an explicit field. Pass the resolved value to reward so
                # judge routing is identical for explicit and inferred tasks.
                reward_task_info = dict(task)
                reward_task_info["modality"] = modality
                from vision_deepresearch_async_workflow.tool_statistics import CallStatistics, observe_judge
                judge_stats = CallStatistics(uid, None)
                judge_scope = observe_judge(judge_stats)
                judge_scope.__enter__()
                judge_failed = False
                reward_started = time.perf_counter()
                try:
                    if asyncio.iscoroutinefunction(self.reward_function):
                        reward_out = await self.reward_function(
                            reward_task_info, prediction
                        )
                    else:
                        loop = asyncio.get_running_loop()
                        reward_out = await loop.run_in_executor(
                            self.executor,
                            contextvars.copy_context().run, self.reward_function,
                            reward_task_info, prediction,
                        )
                except Exception as err:  # noqa: BLE001
                    judge_failed = True
                    print(f"Reward function failed: {err}")
                else:
                    if reward_out.is_correct is not None:
                        episode.is_correct = bool(reward_out.is_correct)
                    if isinstance(reward_out.metadata, dict):
                        reward_metadata = episode.info.setdefault("reward_metadata", {})
                        for key, value in reward_out.metadata.items():
                            if key not in reward_metadata:
                                reward_metadata[key] = value
                    if getattr(reward_out, "reward", None) is not None:
                        episode.info["reward_function_reward"] = float(
                            reward_out.reward
                        )
                finally:
                    reward_function_s = time.perf_counter() - reward_started
                    judge_scope.__exit__(None, None, None)
                    episode.info.setdefault("judge_statistics", []).append(judge_stats.finish("answer_judge", "exception" if judge_failed else "completed", judge_failed, {}))

            # --- Step-level fatal-after detection ---
            # Replaces the old trajectory-level hard mask.  Instead of
            # discarding entire anomalous episodes, we detect the first
            # "fatal" point (consecutive-error cascade) and only mask
            # tokens *after* that point.  The prefix is kept for learning
            # with a one-sided advantage clamp (see trainer).
            first_traj = (
                episode.trajectories[0] if episode.trajectories else None
            )
            all_steps = first_traj.steps if first_traj else []

            fatal_step_index, fatal_reason = _find_fatal_step_index(
                all_steps, result.get("termination", "")
            )
            is_fatal = fatal_step_index is not None

            query_reward_enabled = _query_reward_enabled()

            num_turns = sum(
                len(t.steps) for t in episode.trajectories
            )

            if is_fatal and fatal_step_index == 0:
                # Nothing salvageable — degenerate to hard mask.
                episode.termination_reason = TerminationReason.UNKNOWN
                episode.info["fatal_step_index"] = 0
                episode.info["is_fatal"] = True
                episode.info["fatal_reason"] = fatal_reason
                for trajectory in episode.trajectories:
                    trajectory.reward = 0.0
                episode.metrics = {
                    "reward/accuracy": 0.0,
                    "reward/query_utility": 0.0,
                    "reward/query_utility_enabled": float(query_reward_enabled),
                    "reward/format": 0.0,
                    "reward/total": 0.0,
                    "masked": 1.0,
                    "is_fatal": 1.0,
                    "fatal_step_index": 0,
                    "num_turns": num_turns,
                }
                episode.info["mask_reason"] = fatal_reason
            else:
                # Compute full reward for group advantage statistics.
                # For fatal trajectories, format and optional query utility are
                # scored only on the learnable prefix (pre-fatal steps).
                r_accuracy = 1.0 if episode.is_correct else 0.0

                scored_steps = (
                    all_steps[:fatal_step_index]
                    if is_fatal and fatal_step_index < len(all_steps)
                    else all_steps
                )
                if scored_steps:
                    fmt_scores = [_format_reward_for_step(s) for s in scored_steps]
                    r_format = sum(fmt_scores) / len(fmt_scores)
                else:
                    r_format = 0.0

                if query_reward_enabled:
                    query_steps = (
                        all_steps[:fatal_step_index]
                        if is_fatal and fatal_step_index < len(all_steps)
                        else all_steps
                    )
                    from vision_deepresearch_async_workflow.tool_statistics import CallStatistics, observe_judge
                    query_stats = CallStatistics(uid, None)
                    query_scope = observe_judge(query_stats, "query_judge")
                    query_scope.__enter__()
                    query_failed = False
                    query_judge_started = time.perf_counter()
                    try:
                        r_query = await _judge_query_utility(
                            question=question,
                            ground_truth=answer,
                            prediction=prediction,
                            steps=query_steps,
                        )
                    except Exception as exc:  # noqa: BLE001
                        query_failed = True
                        print(f"[QueryUtilityReward] Exception: {exc}")
                        r_query = 0.0
                    finally:
                        query_judge_s = time.perf_counter() - query_judge_started
                        query_scope.__exit__(None, None, None)
                        episode.info.setdefault("judge_statistics", []).append(query_stats.finish("query_judge", "exception" if query_failed else "completed", query_failed, {}))
                else:
                    r_query = 0.0

                total_reward = self.reward_weights.base_reward(
                    r_format, r_accuracy, r_query, query_enabled=query_reward_enabled,
                )

                for trajectory in episode.trajectories:
                    trajectory.reward = total_reward

                if is_fatal:
                    episode.info["fatal_step_index"] = fatal_step_index
                    episode.info["is_fatal"] = True
                    episode.info["fatal_reason"] = fatal_reason

                episode.metrics = {
                    "reward/accuracy": r_accuracy,
                    "reward/query_utility": r_query,
                    "reward/query_utility_enabled": float(query_reward_enabled),
                    "reward/format": r_format,
                    "reward/total": total_reward,
                    "masked": 0.0,
                    "is_fatal": 1.0 if is_fatal else 0.0,
                    "fatal_step_index": (
                        fatal_step_index if fatal_step_index is not None else -1
                    ),
                    "num_turns": num_turns,
                }

            _attach_timing_metrics(
                episode,
                workflow_started=workflow_started,
                workflow_prepare_s=workflow_prepare_s,
                agent_wall_s=agent_wall_s,
                reward_function_s=reward_function_s,
                query_judge_s=query_judge_s,
            )
            print(f"✅ DeepResearch workflow completed for task {uid}")
            print(f"   Prediction: {result.get('prediction', 'No prediction')}")
            print(f"   True Answer: {answer}")
            print(f"   Metrics: {episode.metrics}")
            if episode.info.get("mask_reason"):
                print(f"   Mask Reason: {episode.info['mask_reason']}")
            return episode

        except Exception as exc:  # noqa: BLE001
            if agent_started is not None and agent_wall_s == 0.0:
                agent_wall_s = time.perf_counter() - agent_started
            error_traceback = traceback.format_exc()
            print(f"❌ DeepResearch workflow failed for task {uid}: {exc}")
            print(error_traceback, end="")
            episode = Episode()
            episode.id = uid
            episode.task = task
            episode.termination_reason = TerminationReason.ERROR
            episode.is_correct = False
            episode.trajectories = []
            episode.metrics = {
                "reward/accuracy": 0.0,
                "reward/query_utility": 0.0,
                "reward/query_utility_enabled": float(_query_reward_enabled()),
                "reward/format": 0.0,
                "reward/total": 0.0,
                "masked": 1.0,
                "is_fatal": 1.0,
                "fatal_step_index": 0,
                "num_turns": 0,
            }
            fatal_reason = f"workflow_exception:{type(exc).__name__}"
            episode.info = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "error_traceback": error_traceback,
                # Keep exception returns inside the same fatal-aware contract
                # as zero-prefix rollout failures.  There are no materialised
                # steps to salvage, so the fatal boundary is necessarily zero
                # and strict collection must backfill this attempt.
                "is_fatal": True,
                "fatal_step_index": 0,
                "fatal_reason": fatal_reason,
                "mask_reason": fatal_reason,
                "masked": 1.0,
            }
            _attach_timing_metrics(
                episode,
                workflow_started=workflow_started,
                workflow_prepare_s=workflow_prepare_s,
                agent_wall_s=agent_wall_s,
                reward_function_s=reward_function_s,
                query_judge_s=query_judge_s,
            )
            return episode

    def _convert_result_to_episode(self, result: dict, task: dict, uid: str) -> Episode:
        messages = result.get("messages", [])
        prediction = result.get("prediction", "")
        termination = result.get("termination", "unknown")
        rounds = result.get("rounds", 0)
        time_taken = result.get("time_taken", 0.0)

        trajectories: list[Trajectory] = []
        steps: list[Step] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant":
                context = [{k:v for k,v in message.items() if k != "tool_statistics"} for message in messages[: i + 1]]
                assistant_content = msg.get("content", "")
                action = _extract_action_from_response(assistant_content)
                observation = _get_next_observation(messages, i)
                step = Step(
                    chat_completions=context.copy(),
                    model_response=assistant_content,
                    action=action,
                    observation=observation,
                    reward=0.0,
                )
                step.model_output = None
                # Per-turn token length (prompt_length INCLUDES expanded video /
                # image tokens; completion_length is this turn's generated tokens).
                # Stamped onto the assistant message by the agent so each dumped
                # Step is token-auditable and comparable to inference token counts.
                if msg.get("prompt_length") is not None:
                    step.info["prompt_length"] = msg["prompt_length"]
                if msg.get("completion_length") is not None:
                    step.info["completion_length"] = msg["completion_length"]
                # Per-turn LLM generation wall-clock (this turn's rollout-engine
                # call), stamped by the agent onto the assistant message.
                if msg.get("llm_duration_s") is not None:
                    step.info["llm_duration_s"] = msg["llm_duration_s"]
                for key, value in msg.items():
                    if key.startswith("engine_") and isinstance(value, (int, float)):
                        step.info[key] = float(value)
                if i + 1 < len(messages):
                    _nxt = messages[i + 1]
                    if _nxt.get("role") == "user" and _nxt.get("tool_duration_s") is not None:
                        step.info["tool_name"] = _nxt.get("tool_name", "unknown")
                        step.info["tool_duration_s"] = _nxt["tool_duration_s"]
                        if isinstance(_nxt.get("tool_statistics"), dict):
                            step.info["tool_statistics"] = _nxt["tool_statistics"]
                        phase_timing = _nxt.get("tool_phase_timing")
                        if isinstance(phase_timing, dict):
                            for name, seconds in phase_timing.items():
                                if isinstance(seconds, (int, float)):
                                    step.info[f"tool_phase/{name}"] = float(seconds)
                if msg.get("step_error"):
                    step.info["step_error"] = True
                if _has_tool_error_observation(observation):
                    step.info["step_error"] = True
                steps.append(step)
            i += 1

        # Aggregate per-step llm/tool timings into trajectory.info["timing"] so
        # EpisodeLogger and analyze_rollout_bottleneck.py can read them directly
        # without re-parsing every step.info.  The per-step flat fields
        # (llm_duration_s / tool_duration_s / tool_name) stay in step.info as-is
        # for fine-grained analysis; this adds a trajectory-level summary.
        _traj_llm_times: list[float] = []
        _traj_tool_times: dict[str, list[float]] = {}
        _traj_engine_totals: dict[str, float] = {}
        _traj_tool_phase_totals: dict[str, float] = {}
        for _st in steps:
            _si = _st.info
            _ld = _si.get("llm_duration_s")
            if isinstance(_ld, (int, float)) and _ld > 0:
                _traj_llm_times.append(float(_ld))
            _td = _si.get("tool_duration_s")
            _tn = _si.get("tool_name")
            if isinstance(_td, (int, float)) and _tn:
                _traj_tool_times.setdefault(_tn, []).append(float(_td))
            for _key, _value in _si.items():
                if _key.startswith("engine_") and isinstance(_value, (int, float)):
                    _merge_engine_timing(
                        _traj_engine_totals, _key, float(_value)
                    )
            for _key, _value in _si.items():
                if _key.startswith("tool_phase/") and isinstance(
                    _value, (int, float)
                ):
                    _traj_tool_phase_totals[_key] = (
                        _traj_tool_phase_totals.get(_key, 0.0) + float(_value)
                    )
        _traj_timing = {
            "llm_total_s": sum(_traj_llm_times),
            "llm_calls": len(_traj_llm_times),
            "tool_total_s": sum(v for vs in _traj_tool_times.values() for v in vs),
            "tool_calls": sum(len(vs) for vs in _traj_tool_times.values()),
            "per_tool_total_s": {k: sum(v) for k, v in _traj_tool_times.items()},
            "per_tool_calls": {k: len(v) for k, v in _traj_tool_times.items()},
            **_traj_engine_totals,
            **_traj_tool_phase_totals,
        }

        from vision_deepresearch_async_workflow.tool_statistics import aggregate_steps
        tool_statistics = aggregate_steps(steps)
        trajectory = Trajectory(
            name="deepresearch_agent",
            task=task,
            steps=steps,
            reward=0.0,
            info={"timing": _traj_timing, "tool_statistics": tool_statistics},
        )
        trajectories.append(trajectory)

        answer_text = task.get("answer", "")
        is_correct = _evaluate_answer(prediction, answer_text) if answer_text else False

        # Episode-level timing: sum across all trajectories (single traj here).
        _ep_llm_total = _traj_timing["llm_total_s"]
        _ep_tool_total = _traj_timing["tool_total_s"]

        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = _map_termination_reason(termination)
        episode.is_correct = is_correct
        episode.trajectories = trajectories
        episode.metrics = {}
        episode.info = {
            "tool_statistics": tool_statistics,
            "rounds": rounds,
            "time_taken": time_taken,
            "prediction": prediction,
            "answer": answer_text,
            "token_usage": result.get("token_usage", {}),
            "timing": {
                "llm_total_s": _ep_llm_total,
                "tool_total_s": _ep_tool_total,
                **_traj_engine_totals,
                **_traj_tool_phase_totals,
                "llm_frac": (
                    _ep_llm_total / (_ep_llm_total + _ep_tool_total)
                    if (_ep_llm_total + _ep_tool_total) > 0 else None
                ),
            },
        }
        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        # MultiTurnReactAgent handles per-run state; nothing to reset here.
        return

    def is_multithread_safe(self) -> bool:
        return True


__all__ = ["DeepResearchWorkflow"]

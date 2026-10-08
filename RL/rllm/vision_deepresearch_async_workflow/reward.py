"""Current Trace/Ground scoring with per-component zero fallback."""
from __future__ import annotations
import asyncio
import concurrent.futures
import logging
import os
import time
from vision_deepresearch_async_workflow import trace_reward, ground_reward
from vision_deepresearch_async_workflow.reward_client import RewardError, RequestError, JudgeClient
from vision_deepresearch_async_workflow.reward_data import episode_payload, value as get_field
from vision_deepresearch_async_workflow.reward_config import RewardWeights
from vision_deepresearch_async_workflow.evidence import normalize_evidence

logger = logging.getLogger(__name__)


def failed_component(component, error, *, attempts=0, latency_s=0., validation_errors=(), validation_retries=0):
    """Keep scoring failures distinguishable from valid judge-assigned zeros."""
    logger.warning("%s reward failed; using zero (%s)", component, type(error).__name__)
    return {f"r_{component}": 0., f"{component}_items": [],
            f"{component}_judge_failed": True,
            f"{component}_judge_error": f"{type(error).__name__}: {error}",
            f"{component}_judge_attempts": attempts,
            f"{component}_judge_latency_s": latency_s,
            f"{component}_response_model": None,
            f"{component}_validation_retries": validation_retries,
            f"{component}_validation_errors": list(validation_errors)}


_JUDGE_RETRY_PROMPT_TEMPLATE = """\
Validation failed: {error}. Regenerate the complete JSON using the SAME supplied evidence. Copy citation IDs and spans exactly. Do not invent evidence. Keep unsupported targets unsupported; provide every target."""


def build_inputs(task, steps, gt, *, fatal_step_index=None):
    evidence = normalize_evidence(task, steps, gt, fatal_step_index=fatal_step_index)
    prepared = dict(evidence=evidence, **trace_reward.build_trace_inputs(evidence),
                    **ground_reward.build_ground_inputs(evidence))
    limit = int(os.getenv("EVGR_SPLIT_MAX_TEXT_CHARS", "200000"))
    image_label_chars = sum(len(part["text"]) for part in prepared["ground_images"] if part["type"] == "text")
    if max(len(prepared["trace_prompt"]), len(prepared["ground_prompt"]) + image_label_chars) > limit:
        raise RewardError("Split judge input exceeds text budget; no silent truncation")
    return prepared


class RewardJudge(JudgeClient):
    def parse_component(self, text, prepared, component):
        return {"trace": trace_reward.parse_trace, "ground": ground_reward.parse_ground}[component](text, prepared)

    def score_component(self, prepared, component):
        result = {}
        validation_retries = int(os.getenv("EVGR_SPLIT_VALIDATION_RETRIES", "1"))
        if not 0 <= validation_retries <= 2:
            raise ValueError("EVGR_SPLIT_VALIDATION_RETRIES must be 0, 1 or 2")
        prompt = prepared[f"{component}_prompt"]
        content = prompt if component == "trace" else [{"type": "text", "text": prompt}, *prepared["ground_images"]]
        payload = dict(model=self.config.model, messages=[dict(role="user", content=content)],
                       temperature=0., max_tokens=self.config.max_tokens, stream=False)
        started = time.monotonic()
        attempts, validation_errors = 0, []
        for retry in range(validation_retries + 1):
            text = ""
            try:
                response, count = self._post(payload)
                attempts += count
                choices = response.get("choices") or []
                text = str(((choices[0].get("message") or {}).get("content") if choices else "") or "")
                if not choices or choices[0].get("finish_reason") not in (None, "stop"):
                    raise RewardError(f"{component} judge has missing or truncated output")
                parsed = self.parse_component(text, prepared, component)
                break
            except RequestError as exc:
                return failed_component(component, exc, attempts=attempts + exc.attempts,
                                        latency_s=time.monotonic()-started,
                                        validation_errors=validation_errors, validation_retries=retry)
            except Exception as exc:
                validation_errors.append(str(exc))
                if retry == validation_retries:
                    return failed_component(component, exc, attempts=attempts,
                                            latency_s=time.monotonic()-started,
                                            validation_errors=validation_errors, validation_retries=retry)
                # Ask the model to correct its own contract violation using
                # exactly the same evidence. Never repair by deleting a
                # citation or inventing a span. Exhausted scoring falls back separately.
                payload["messages"] = [payload["messages"][0],
                    dict(role="assistant", content=text),
                    dict(role="user", content=_JUDGE_RETRY_PROMPT_TEMPLATE.format(error=exc))]
        result.update(parsed)
        result.update({f"{component}_judge_failed": False,
                       f"{component}_judge_attempts": attempts,
                       f"{component}_judge_latency_s": time.monotonic()-started,
                       f"{component}_response_model": response.get("model"),
                       f"{component}_validation_retries": len(validation_errors),
                       f"{component}_validation_errors": validation_errors})
        return result

    def combine_components(self, prepared, trace, ground):
        result = dict(trace, **ground)
        result.update(judge_model=self.config.model, judge_backend=self.config.backend,
                      judge_attempts=result["trace_judge_attempts"]+result["ground_judge_attempts"],
                      score_aggregation="holistic-continuous",
                      evidence_diagnostics=prepared.get("evidence", {}).get("diagnostics", {}))
        return result

    def score_prepared(self, prepared):
        return self.combine_components(prepared, self.score_component(prepared, "trace"),
                                       self.score_component(prepared, "ground"))

    def score(self, task, steps, gt, *, fatal_step_index=None):
        try:
            prepared = build_inputs(task, steps, gt, fatal_step_index=fatal_step_index)
        except Exception as exc:
            return self.failed_score(exc)
        return self.score_prepared(prepared)

    def failed_score(self, error):
        """Shared input/GT failures make both components unavailable."""
        return self.combine_components({}, failed_component("trace", error),
                                       failed_component("ground", error))


async def score_batch(episodes, index, judge, *, max_concurrency=32):
    if max_concurrency < 1:
        raise ValueError("Split judge concurrency must be positive")
    output = [None] * len(episodes)
    work = []
    for slot, ep in enumerate(episodes):
        if ep is None:
            continue
        def flagged(v):
            return str(v).lower() in {"true", "1", "yes", "on"}
        if any(flagged((get_field(ep, field, {}) or {}).get("masked")) for field in ("info", "metrics")):
            continue
        task, steps, fatal, _, _, _ = episode_payload(ep)
        if not steps or fatal == 0:
            continue
        work.append((slot, task, steps, fatal))
    if not work:
        return output
    loop = asyncio.get_running_loop()
    def score(row):
        _, task, steps, fatal = row
        try:
            return judge.score(task, steps, index.lookup(task), fatal_step_index=fatal)
        except Exception as exc:
            return judge.failed_score(exc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_concurrency, len(work))) as pool:
        values = await asyncio.gather(*[loop.run_in_executor(pool, score, row) for row in work])
    for row, value in zip(work, values, strict=True):
        output[row[0]] = value
    return output


def blend_reward(
    *,
    r_format: float,
    r_accuracy: float,
    r_query: float,
    r_evgr: float,
    process_weight: float | None = None,
) -> float:
    """Combine rewards using the same validated weights as launch preflight."""
    weights = RewardWeights.from_env(process_weight=process_weight)
    return weights.combine(r_format, r_accuracy, r_query, r_evgr)


def apply_reward(episode, score, *, mode, process_weight):
    if mode != "both":
        raise ValueError("The current reward requires mode=both")
    metrics = episode.metrics
    component = 0.5 * (float(score["r_trace"]) + float(score["r_ground"]))
    total = blend_reward(
        r_format=metrics["reward/format"], r_accuracy=metrics["reward/accuracy"],
        r_query=metrics["reward/query_utility"], r_evgr=component, process_weight=process_weight)
    failed = {name: bool(score.get(f"{name}_judge_failed", False)) for name in ("trace", "ground")}
    metrics.update({"reward/formal_evgr": component, "reward/formal_evgr_trace": score["r_trace"],
                    "reward/formal_evgr_ground": score["r_ground"], "reward/formal_evgr_valid": float(not any(failed.values())),
                    "reward/formal_evgr_applied": 1.0, "reward/total": total})
    for name in ("trace", "ground"):
        metrics[f"reward/formal_evgr_{name}_failed"] = float(failed[name])
        metrics[f"timing/formal_evgr_{name}_judge_s"] = score[f"{name}_judge_latency_s"]
        metrics[f"reward/formal_evgr_{name}_request_attempts"] = score[f"{name}_judge_attempts"]
    episode.info["formal_evgr"] = score
    for trajectory in episode.trajectories:
        trajectory.reward = total
    return total

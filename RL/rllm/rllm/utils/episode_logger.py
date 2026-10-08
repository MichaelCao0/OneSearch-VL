"""Episode JSON Logger for saving detailed episode information."""

import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from rllm.agents.agent import Episode


class EpisodeLogger:
    """Logger to save episodes to individual JSON files by step and episode ID."""

    def __init__(self, base_dir: str, subdirectory: str = "episodes"):
        """Initialize the episode logger.

        Args:
            base_dir: Base directory for episode logs. Can be configured via
                     config.trainer.episode_log_dir
                     (default: "logs/${trainer.project_name}/${trainer.experiment_name}")
            subdirectory: Subdirectory within base_dir for episodes (default: "episodes")
                         Final path will be: {base_dir}/{subdirectory}/
        """
        self.log_dir = Path(base_dir) / subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the directory path for a specific training or validation step.

        Args:
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0

        Returns:
            Path object for the step directory
        """
        step_dir = self.log_dir / f"{mode}_step_{step}_epoch_{epoch}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def get_episode_filename(self, episode: Episode, step: int) -> str:
        """Generate filename for an episode.

        Format: episode_id{encoded_episode_id}.json

        Args:
            episode: The episode to save
            step: Current training step (not used in filename, but kept for compatibility)

        Returns:
            Filename string
        """
        # Escape separators without merging distinct episode IDs.
        episode_id_safe = quote(str(episode.id), safe="")

        filename = f"episode_id{episode_id_safe}.json"
        return filename

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0):
        """Log a single episode to its own JSON file in a step-specific directory.

        Args:
            episode: The episode to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        episode_data = {"training_step": step, "epoch": epoch, "episode_id": episode.id, "task": episode.task, "is_correct": episode.is_correct, "termination_reason": episode.termination_reason.value if episode.termination_reason else None, "metrics": episode.metrics, "timing": episode.info.get("timing", {}), "trajectories": []}
        episode_data["judge_statistics"] = episode.info.get("judge_statistics", [])
        episode_data["fatal_diagnostics"] = {key: episode.info.get(key) for key in ("fatal_reason", "mask_reason", "fatal_step_index", "is_fatal")}
        episode_data["tool_statistics"] = episode.info.get("tool_statistics", {})

        for traj in episode.trajectories:
            traj_data = {
                "name": traj.name,
                "uid": traj.uid,
                "reward": traj.reward,
                "num_steps": len(traj.steps),
                "timing": traj.info.get("timing", {}),
                "tool_statistics": traj.info.get("tool_statistics", {}),
                "steps": [
                    {
                        "observation": step.observation,
                        "thought": step.thought,
                        "action": step.action,
                        "reward": step.reward,
                        "done": step.done,
                        "model_response": step.model_response,
                        "chat_completions": step.chat_completions,
                        "timing": step.info.get("timing", {}),  # Add step-level timing
                        "info": step.info,  # flat per-tool timing keys (tool_name/tool_duration_s/llm_duration_s)
                    }
                    for step in traj.steps
                ],
            }
            episode_data["trajectories"].append(traj_data)

        # Write to individual file in step-specific directory
        step_dir = self.get_step_dir(step, mode, epoch)
        filename = self.get_episode_filename(episode, step)
        filepath = step_dir / filename

        try:
            with open(filepath, "w") as f:
                json_str = json.dumps(episode_data, indent=2, default=str)
                f.write(json_str + "\n")
                f.flush()  # Ensure data is written to disk
        except Exception as e:
            print(f"Error writing episode to {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise

    def log_episodes(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0):
        """Log multiple episodes, each to its own file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        print(f"[EpisodeLogger] Logging {len(episodes)} episodes for step={step}, mode={mode}, epoch={epoch}")
        for i, episode in enumerate(episodes):
            try:
                self.log_episode(episode, step, mode, epoch)
                print(f"[EpisodeLogger] Successfully logged episode {i + 1}/{len(episodes)}: {episode.id}")
            except Exception as e:
                print(f"[EpisodeLogger] Failed to log episode {i + 1}/{len(episodes)}: {e}")
                raise

    def log_episodes_batch(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0, batch_summary: bool = True):
        """Log multiple episodes and optionally create a batch summary in step-specific directory.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
            batch_summary: Whether to create a summary file for the batch
        """
        # Log individual episodes
        self.log_episodes(episodes, step, mode, epoch)

        # Optionally create batch summary in step-specific directory
        if batch_summary and episodes:
            tool_counts = Counter()
            judge_counts = Counter()
            for ep in episodes:
                tool_counts.update(ep.info.get("tool_statistics", {}))
                for judge in ep.info.get("judge_statistics", []):
                    judge_counts["invocations"] += 1
                    judge_counts["invocations_without_http"] += int(not judge.get("requests"))
                    judge_counts["decision_observed"] += int(judge.get("judge_decision_valid") is not None)
                    judge_counts["decision_valid"] += int(judge.get("judge_decision_valid") is True)
                    judge_counts.update(judge.get("counts", {}))
            def rate(numerator, denominator):
                count = tool_counts.get(denominator, 0)
                return tool_counts.get(numerator, 0) / count if count else None
            tool_rates = {
                "degraded_return": rate("degraded_return_calls", "degradation_observed_success_calls"),
                "degradation_coverage": rate("degradation_observed_success_calls", "success"),
                "full_quality_success": rate("full_quality_success_calls", "full_quality_observed_calls"),
                "full_quality_coverage": rate("full_quality_observed_calls", "quality_eligible_calls"),
                "page_degraded": rate("pages_degraded", "pages_recorded"),
                "page_full_quality": rate("pages_full_quality", "pages_quality_known"),
                "page_quality_coverage": rate("pages_quality_known", "pages_recorded"),
                "summary_completion_coverage": rate("summary_calls", "summary_started"),
                "execution_success": rate("success", "calls"),
                "page_fetch_success": rate("body_fetch_success_pages", "body_fetch_observed_pages"),
                "summary_generation_success": rate("summary_generated", "summary_calls"),
                "summary_fallback": rate("summary_fallbacks", "summary_calls"),
            }
            for key in tool_counts:
                if key.startswith("tool/") and key.endswith("/calls"):
                    tool_rates[key.removesuffix("/calls") + "/execution_success"] = rate(key.removesuffix("/calls") + "/success", key)
                    tool_prefix = key.removesuffix("/calls") + "/"
                    for label, numerator, denominator in (
                        ("degraded_return", "degraded_return_calls", "degradation_observed_success_calls"),
                        ("degradation_coverage", "degradation_observed_success_calls", "success"),
                        ("page_fetch_success", "body_fetch_success_pages", "body_fetch_observed_pages"),
                        ("summary_generation_success", "summary_generated", "summary_calls"),
                        ("full_quality_success", "full_quality_success_calls", "full_quality_observed_calls"),
                        ("full_quality_coverage", "full_quality_observed_calls", "calls"),
                    ):
                        tool_rates[tool_prefix + label] = rate(tool_prefix + numerator, tool_prefix + denominator)
                if key.startswith("http_") and key.endswith("_attempts"):
                    prefix = key.removesuffix("_attempts")
                    tool_rates[prefix + "_retry_fraction"] = rate(prefix + "_retries", key)
                    tool_rates[prefix + "_success"] = rate(prefix + "_success", prefix + "_completed")
                    tool_rates[prefix + "_completion_coverage"] = rate(prefix + "_completed", key)
            summary_data = {
                "training_step": step,
                "epoch": epoch,
                "mode": mode,
                "num_episodes": len(episodes),
                "episode_files": [self.get_episode_filename(ep, step) for ep in episodes],
                "summary_stats": {
                    "judge_statistics": {"counts":dict(judge_counts), "decision_valid_rate": judge_counts["decision_valid"] / judge_counts["decision_observed"] if judge_counts["decision_observed"] else None},
                    "tool_statistics": {"counts": dict(tool_counts), "rates": tool_rates,
                        "episodes_observed": sum(bool(ep.info.get("tool_statistics")) for ep in episodes)},
                    "total_correct": sum(1 for ep in episodes if ep.is_correct),
                    "total_incorrect": sum(1 for ep in episodes if not ep.is_correct),
                    "accuracy": sum(1 for ep in episodes if ep.is_correct) / len(episodes) if episodes else 0,
                    "avg_trajectories_per_episode": sum(len(ep.trajectories) for ep in episodes) / len(episodes) if episodes else 0,
                },
            }

            step_dir = self.get_step_dir(step, mode, epoch)
            summary_file = step_dir / "batch_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

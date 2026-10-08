"""Answer-aware Query inputs, distinct from the Trace/Ground evidence ledger.

No pixel attachments or judge-side target annotations are accepted here.
Tool observations are truncated summaries, not validated ledger evidence spans.
"""
import json
import re


_QUERY_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant."""


_QUERY_UTILITY_PROMPT_TEMPLATE = """\
You are an impartial judge evaluating the quality and utility of an agent's search trajectory.

[Original Question]
{question}

[Ground Truth Answer]
{ground_truth}

[Agent's Final Answer]
{prediction}

[Search Trajectory]
{trajectory_summary}

[Evaluation Criteria]
Evaluate the overall utility of the agent's search queries and retrieved results:

1. **Image search utility**: Did image searches retrieve visual evidence that genuinely supports answering the question? Were the images relevant, or just noise (captchas, unrelated pages)?
2. **Text search utility**: Did text searches find relevant textual information that contributes to the answer? Were queries well-formed and targeted?
3. **Query progression**: Did the queries show logical progression — refining, narrowing, or covering different aspects of the question? Or did they repeat / drift aimlessly?
4. **Complementarity**: Did image and text searches complement each other, providing evidence that one modality alone couldn't supply?
5. **Evidence vs. noise ratio**: What fraction of retrieved results actually contained useful evidence versus irrelevant content?

Score the overall query utility from 0.0 to 1.0:
- 0.0: No useful information retrieved; all searches irrelevant or failed
- 0.3: Mostly noise with occasional marginally relevant results
- 0.5: Mixed — some useful evidence found but significant noise or inefficiency
- 0.7: Good search strategy; majority of results were relevant
- 1.0: Excellent — targeted, efficient queries that retrieved highly relevant evidence

Output format (strictly follow):
score: [a single float between 0.0 and 1.0]
reasoning: [brief explanation in 2-3 sentences]"""


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + " ... " + text[-(limit // 2) :]


def _build_trajectory_summary(steps: list, max_steps: int = 20) -> str:
    """Build a concise summary of search-related steps for the judge."""
    lines: list[str] = []
    search_count = 0

    for idx, step in enumerate(steps):
        action_dict = step.action.action if step.action else {}
        action_type = action_dict.get("type", "")

        if action_type == "tool_call":
            tool_call_raw = action_dict.get("tool_call", "")
            tool_name = "unknown"
            tool_args_str = ""
            try:
                if isinstance(tool_call_raw, str):
                    parsed = json.loads(tool_call_raw)
                else:
                    parsed = tool_call_raw
                tool_name = parsed.get("name", "unknown")
                tool_args_str = json.dumps(
                    parsed.get("arguments", {}), ensure_ascii=False
                )
            except (json.JSONDecodeError, TypeError, AttributeError):
                tool_name = "parse_error"
                tool_args_str = _truncate(str(tool_call_raw), 200)

            obs = step.observation or ""
            if isinstance(obs, str) and "<tool_response>" in obs:
                obs = obs.split("<tool_response>")[-1].split("</tool_response>")[0]

            is_error = step.info.get("step_error", False)
            status = " [ERROR]" if is_error else ""

            lines.append(
                f"Step {idx + 1}: {tool_name}({_truncate(tool_args_str, 200)}){status}\n"
                f"  Result: {_truncate(obs.strip(), 400)}"
            )
            search_count += 1
            if search_count >= max_steps:
                lines.append(f"... ({len(steps) - idx - 1} more steps omitted)")
                break

        elif action_type == "final_answer":
            lines.append(
                f"Step {idx + 1}: final_answer\n"
                f"  Answer: {_truncate(action_dict.get('answer', ''), 300)}"
            )

    if not lines:
        return "(No tool calls in trajectory)"
    return "\n\n".join(lines)



def build_query_judge_prompt(*, question, ground_truth, prediction, trajectory_summary):
    return _QUERY_UTILITY_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth or "(not provided)",
        prediction=prediction or "(no prediction)",
        trajectory_summary=trajectory_summary,
    )


def parse_query(text: str) -> float | None:
    """Extract the score float from judge response."""
    for line in text.split("\n"):
        line_stripped = line.strip().lower()
        if line_stripped.startswith("score:"):
            val_str = line_stripped[len("score:"):].strip()
            try:
                val = float(val_str)
                return max(0.0, min(1.0, val))
            except ValueError:
                continue
    m = re.search(r"score\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return None

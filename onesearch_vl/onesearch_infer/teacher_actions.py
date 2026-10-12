"""Serialize a teacher-selected JSON action into the existing OneSearch tags.

The provider text is retained separately. This conversion never chooses an
action, supplies missing arguments, or invents observations.
"""
from __future__ import annotations

import json
import math

PROTOCOL = "onesearch_action_json.v1"
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["inspect_av", "text_search", "visit_webpage", "final"]},
        "summary": {"type": "STRING", "description": "A brief visible next-step summary, not private reasoning."},
        "arguments": {"type": "OBJECT", "properties": {
            "video": {"type": "STRING", "nullable": True},
            "start_s": {"type": "NUMBER", "nullable": True}, "end_s": {"type": "NUMBER", "nullable": True},
            "q": {"type": "STRING", "nullable": True}, "top_k": {"type": "INTEGER", "nullable": True},
            "url": {"type": "STRING", "nullable": True}},
            "required": ["video", "start_s", "end_s", "q", "top_k", "url"]},
        "answer": {"type": "STRING", "description": "Final answer for final; empty string for a tool action."}},
    "required": ["action", "summary", "arguments", "answer"],
}
TRANSPORT_INSTRUCTION = """The API transport uses one JSON action object per request.
Return ONLY the JSON object specified by responseSchema, without XML or code fences.
The external loop renders your action into OneSearch tags in previous history.
Choose action inspect_av, text_search, visit_webpage, or final. summary is a brief
visible next-step summary, not private reasoning. For a tool action, answer is
empty and arguments contains that tool's documented parameters. All six argument
keys must be present; set every unused parameter to JSON null. start_s and end_s
are separate numeric fields, never embedded in the video handle. For final,
all six arguments are null and answer contains the final answer. Do not simulate any
observations or combine a tool request and a final answer. Never include protocol
tags in summary or answer. The same tools and evidence requirements still apply.
"""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate teacher action key")
        result[key] = value
    return result


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def render_teacher_action(provider_text):
    value = json.loads(provider_text, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {"action", "summary", "arguments", "answer"}:
        raise ValueError("teacher must select exactly one complete action")
    name, summary, args, answer = (value[k] for k in ("action", "summary", "arguments", "answer"))
    if (not isinstance(summary, str) or not isinstance(answer, str) or not isinstance(args, dict)
            or any(tag in text for text in (summary, answer) for tag in
                   ("<tool_call", "</tool_call", "<response", "</response", "<think", "</think", "<tool_response"))):
        raise ValueError("invalid teacher action text or arguments")
    if set(args) - {"video", "start_s", "end_s", "q", "top_k", "url"}:
        raise ValueError("unknown teacher argument")
    args = {key: val for key, val in args.items() if val is not None}
    prefix = f"<think>{summary}</think>\n" if summary.strip() else ""
    if name == "final":
        if args or not answer.strip():
            raise ValueError("final answer cannot include an unexecuted action")
        return prefix + f"<response>{answer}</response>"
    if answer:
        raise ValueError("tool action cannot contain an unexecuted final answer")
    if name == "text_search":
        valid = (set(args) <= {"q", "top_k"} and isinstance(args.get("q"), str) and args["q"].strip()
                 and ("top_k" not in args or type(args["top_k"]) is int and 1 <= args["top_k"] <= 20))
    elif name == "visit_webpage":
        valid = set(args) == {"url"} and isinstance(args["url"], str) and args["url"].strip()
    elif name == "inspect_av":
        valid = (set(args) == {"video", "start_s", "end_s"} and isinstance(args["video"], str)
                 and args["video"].strip() and _finite(args["start_s"]) and _finite(args["end_s"])
                 and 0 <= args["start_s"] < args["end_s"])
    else:
        valid = False
    if not valid:
        raise ValueError("invalid or unsupported teacher tool action")
    action = json.dumps({"name": name, "arguments": args}, ensure_ascii=False, separators=(",", ":"))
    return prefix + f"<tool_call>{action}</tool_call>"

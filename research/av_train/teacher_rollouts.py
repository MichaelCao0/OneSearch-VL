"""Sharded native teacher candidates through the original OneSearch agent loop.

Explicit plan/run/status CLI; no scheduler, GPU loading or automatic retries.
Completion records execution only. Semantic retention remains a separate step.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "onesearch_vl"))

from onesearch_infer.gemini_av_runner import GeminiAVRunner, GeminiAVSettings
from onesearch_infer.openai_av_runner import OpenAIAVRunner, OpenAIAVSettings
from onesearch_infer.pipeline import process_native_av_case
from onesearch_infer.runners import InferenceConfig, _empty_response
from onesearch_infer.search_backends import FrozenSearchBackend
from .native_searchqa import prepare_cases
from .provider_guard import ProviderGuard
from .native_sft import file_sha, validate_trajectory

SCHEMA = "onesearch-native-teacher-plan.v1"
SOURCE_FILES = ["research/av_train/provider_guard.py", "research/av_train/teacher_rollouts.py", "research/av_train/native_searchqa.py",
                "research/av_train/native_sft.py", "research/av_train/generation_receipts.py",
                "research/av_train/teacher_media.py",
                "onesearch_vl/onesearch_infer/gemini_av_runner.py",
                "onesearch_vl/onesearch_infer/teacher_actions.py",
                "onesearch_vl/onesearch_infer/openai_av_runner.py",
                "onesearch_vl/onesearch_infer/sft_chat_format.py",
                "onesearch_vl/onesearch_infer/pipeline.py", "onesearch_vl/onesearch_infer/messages.py",
                "onesearch_vl/onesearch_infer/tools.py", "onesearch_vl/onesearch_infer/av_io.py",
                "onesearch_vl/onesearch_infer/search_backends.py",
                "onesearch_vl/onesearch_infer/config.py", "onesearch_vl/onesearch_infer/runners.py",
                "research/av_search/corpus.py", "research/av_search/retrieval.py", "pixi.lock"]


def teacher_settings(value):
    schema = value.get("schema") if isinstance(value, dict) else None
    if schema == "onesearch-gemini-av-teacher.v1":
        return GeminiAVSettings.from_dict(value)
    if schema == "onesearch-openai-av-teacher.v1":
        return OpenAIAVSettings.from_dict(value)
    raise ValueError("Unsupported native AV teacher config schema")


def make_teacher(settings):
    return OpenAIAVRunner(settings) if isinstance(settings, OpenAIAVSettings) else GeminiAVRunner(settings)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def create_json_once(path, value):
    """Publish a complete create-only JSON even when different shards race."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    atomic_json(temp, value)
    try:
        os.link(temp, path)
    except FileExistsError:
        if read_json(path) != value:
            raise ValueError("Resolved teacher identity changed; use a new versioned plan")
    finally:
        temp.unlink()


def prepare_plan(batch_dir, out_dir, teacher_config, prompt_file, *, candidates_per_task=3,
                 shard_size=20, index_dir=None, max_turns=8, max_new_tokens=2048, temperature=.7):
    """Bind train-only anonymous tasks and split candidate IDs into fixed shards."""
    if (type(candidates_per_task) is not int or not 1 <= candidates_per_task <= 4
            or type(shard_size) is not int or not 1 <= shard_size <= 256
            or type(max_turns) is not int or not 1 <= max_turns <= 12
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8192
            or isinstance(temperature, bool) or not 0 <= temperature <= 2):
        raise ValueError("Invalid teacher candidate or generation budget")
    batch = Path(batch_dir).resolve()
    config = Path(teacher_config).resolve()
    prompt = Path(prompt_file).resolve()
    index = Path(index_dir or batch / "corpus").resolve()
    settings = teacher_settings(read_json(config))
    scope = read_json(batch / "scope.json")
    tasks = [json.loads(line) for line in (batch / "tasks-public.jsonl").read_text().splitlines() if line]
    if scope.get("split") != "train" or not tasks or any(t.get("split") != "train" for t in tasks):
        raise ValueError("Teacher training candidates require a train-only batch")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    cases = prepare_cases(batch)
    queue = []
    for case in sorted(cases, key=lambda c: c["case_id"]):
        task_id = case["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
            raise ValueError("Task ID cannot be used as a candidate path")
        for candidate in range(1, candidates_per_task + 1):
            queue.append({**case, "task_id": task_id, "case_id": f"{task_id}__c{candidate:02d}",
                          "candidate_index": candidate})
    # Validate the frozen retrieval runtime in the actual producer environment.
    backend = FrozenSearchBackend(index, scope)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "shards").mkdir()
    (out / "episodes").mkdir()
    (out / "locks").mkdir()
    (out / "frozen-source").mkdir()
    atomic_json(out / "teacher-config.json", asdict(settings))
    (out / "prompt.txt").write_bytes(prompt.read_bytes())
    files = {}
    for name in SOURCE_FILES:
        source = ROOT / name
        target = out / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        files[str(source)] = file_sha(source)
    for path in (batch / "scope.json", batch / "tasks-public.jsonl", batch / "clue-bindings.private.json",
                 index / "manifest.json", index / "documents.jsonl"):
        files[str(path)] = file_sha(path)
    shards = []
    for start in range(0, len(queue), shard_size):
        path = out / "shards" / f"shard-{len(shards):04d}.jsonl"
        rows = queue[start:start + shard_size]
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        shards.append({"shard_id": len(shards), "path": str(path.relative_to(out)),
                       "sha256": file_sha(path), "candidates": len(rows)})
    atomic_json(out / "episodes/inputs.private.json", {"batch_dir": str(batch), "cases": queue,
                "input_mode": "native_audio_video", "private_answers_in_actor": False})
    manifest = {"schema": SCHEMA, "created_at": now(), "batch_dir": str(batch), "index_dir": str(index),
                "scope": scope, "files_sha256": files, "shards": shards,
                "planned_candidates": len(queue), "distinct_tasks": len(cases),
                "candidates_per_task": candidates_per_task, "teacher_status_at_plan": settings.status(),
                "config_sha256": file_sha(out / "teacher-config.json"),
                "prompt_sha256": file_sha(out / "prompt.txt"),
                "inputs_sha256": file_sha(out / "episodes/inputs.private.json"),
                "max_turns": max_turns, "max_new_tokens": max_new_tokens, "temperature": temperature,
                "agent_loop": "onesearch_infer.pipeline.run_agent_loop", "search_mode": "frozen",
                "production_sft_target": 15000, "retained_sft_rows": 0,
                "source_family_review": "must be completed for formal source-disjoint training/evaluation",
                "gpu_jobs_submitted": 0, "model_loaded": False}
    atomic_json(out / "plan.json", manifest)
    (out / "plan.sha256").write_text(file_sha(out / "plan.json") + "\n")
    return manifest


def load_plan(plan_dir):
    root = Path(plan_dir).resolve()
    sha = (root / "plan.sha256").read_text().strip()
    if file_sha(root / "plan.json") != sha:
        raise ValueError("Teacher plan changed")
    plan = read_json(root / "plan.json")
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unsupported teacher plan")
    for name, expected in plan["files_sha256"].items():
        if file_sha(name) != expected:
            raise ValueError("Frozen source or batch changed: " + Path(name).name)
    for name, key in (("teacher-config.json", "config_sha256"), ("prompt.txt", "prompt_sha256"),
                      ("episodes/inputs.private.json", "inputs_sha256")):
        if file_sha(root / name) != plan[key]:
            raise ValueError("Teacher plan config/prompt/inputs changed")
    for row in read_json(root / "episodes/inputs.private.json")["cases"]:
        if file_sha(row["source_path"]) != row["source_sha256"]:
            raise ValueError("Bound teacher input media changed")
    return root, plan, sha


class BudgetedRunner:
    """Charge request slots before transport, retaining uncertain-call evidence."""
    supports_audio_video = True

    def __init__(self, runner, limit, guard=None):
        self.guard = guard
        self.runner = runner
        self.spec = runner.spec
        self.limit = limit
        self.calls = 0
        self.charge = None

    def infer(self, *args, **kwargs):
        if self.guard and self.guard.stopped:
            return _empty_response(self.runner.display_name, "Teacher provider queue stopped; request not sent")
        if self.calls >= self.limit:
            return _empty_response(self.runner.display_name, "Teacher invocation API request budget exhausted")
        self.calls += 1
        self.charge()
        response = self.runner.infer(*args, **kwargs)
        if self.guard:
            self.guard.observe(response)
        return response


def reconcile_candidate(folder, state, plan_sha):
    """Running/no-state folders are uncertain attempts; never automatically resend."""
    if state and state.get("plan_sha256") != plan_sha:
        raise ValueError("Candidate belongs to another plan")
    if state.get("status") == "completed":
        path = folder / state["trajectory_file"]
        if file_sha(path) != state["trajectory_sha256"]:
            raise ValueError("Completed candidate trajectory changed")
    elif not state or state.get("status") == "running":
        state = {**state, "status": "interrupted", "plan_sha256": plan_sha,
                 "api_calls": state.get("api_calls", 0), "updated_at": now(),
                 "reason": "Previous candidate did not publish a terminal state; preserved without automatic retry"}
        atomic_json(folder / "candidate-state.json", state)
    return state


def run_shard(plan_dir, shard_id, *, max_candidates, max_api_calls, reconcile_only=False):
    if type(max_candidates) is not int or not 1 <= max_candidates <= 256 or type(max_api_calls) is not int or not 1 <= max_api_calls <= 4096:
        raise ValueError("Explicit positive candidate and API request budgets required")
    root, plan, plan_sha = load_plan(plan_dir)
    shard = next((x for x in plan["shards"] if x["shard_id"] == shard_id), None)
    if shard is None or file_sha(root / shard["path"]) != shard["sha256"]:
        raise ValueError("Unknown or changed teacher shard")
    rows = [json.loads(line) for line in (root / shard["path"]).read_text().splitlines()]
    result = {"shard_id": shard_id, "new_api_calls": 0, "new_completed_candidates": 0,
              "new_failed_candidates": 0, "attempted_candidates": 0, "retained_sft_rows": 0}
    with (root / "locks" / f"shard-{shard_id:04d}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for row in rows:
            folder = root / "episodes" / row["case_id"]
            if folder.exists():
                state_path = folder / "candidate-state.json"
                reconcile_candidate(folder, read_json(state_path) if state_path.exists() else {}, plan_sha)
            else:
                pending.append(row)
        guard = ProviderGuard(root)
        result["provider_stopped"] = guard.stopped
        if reconcile_only or not pending or guard.stopped:
            return result
        settings = teacher_settings(read_json(root / "teacher-config.json"))
        teacher = make_teacher(settings)
        create_json_once(root / "teacher-resolved.private.json", teacher.identity)
        runner = BudgetedRunner(teacher, max_api_calls, guard)
        backend = FrozenSearchBackend(plan["index_dir"], plan["scope"])
        old_turns = os.environ.get("AV_AGENT_MAX_TURNS")
        old_retrieval = os.environ.get("ONESEARCH_DISABLE_RETRIEVAL")
        os.environ["AV_AGENT_MAX_TURNS"] = str(plan["max_turns"])
        os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = "0"
        try:
            for row in pending[:max_candidates]:
                if runner.calls >= max_api_calls or guard.stopped:
                    break
                folder = root / "episodes" / row["case_id"]
                folder.mkdir()
                state = {"status": "running", "plan_sha256": plan_sha, "case_id": row["case_id"],
                         "task_id": row["task_id"], "api_calls": 0, "started_at": now()}
                atomic_json(folder / "candidate-state.json", state)

                def charge():
                    state["api_calls"] += 1
                    state["updated_at"] = now()
                    atomic_json(folder / "candidate-state.json", state)

                runner.charge = charge
                result["attempted_candidates"] += 1
                try:
                    trace = process_native_av_case(question=row["question"], source_path=row["source_path"],
                              runner=runner, output_dir=str(folder), case_id=row["case_id"], search_backend=backend,
                              inference_cfg=InferenceConfig(max_tokens=plan["max_new_tokens"], temperature=plan["temperature"]),
                              system_instruction=(root / "prompt.txt").read_text())
                    validate_trajectory(trace)
                    path = folder / f'{row["case_id"]}_trajectory.json'
                    state.update(status="completed", trajectory_file=path.name, trajectory_sha256=file_sha(path),
                                 semantic_review="pending", success_verdict=False)
                    result["new_completed_candidates"] += 1
                except Exception as exc:
                    # Provider failures are already sanitized; no server body,
                    # auth header or exception text is persisted here.
                    state.update(status="failed", error_type=type(exc).__name__, success_verdict=False)
                    result["new_failed_candidates"] += 1
                state["updated_at"] = now()
                atomic_json(folder / "candidate-state.json", state)
            result["new_api_calls"] = runner.calls
            result["provider_stopped"] = guard.stopped
        finally:
            for key, value in (("AV_AGENT_MAX_TURNS", old_turns), ("ONESEARCH_DISABLE_RETRIEVAL", old_retrieval)):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return result


def summarize_plan(plan_dir):
    root = Path(plan_dir).resolve()
    plan = read_json(root / "plan.json")
    states = [read_json(path) for path in (root / "episodes").glob("*/candidate-state.json")]
    counts = {key: sum(s.get("status") == key for s in states) for key in ("completed", "failed", "interrupted", "running")}
    return {"planned_candidates": plan["planned_candidates"], "distinct_tasks": plan["distinct_tasks"],
            **{key + "_candidates": value for key, value in counts.items()},
            "pending_candidates": plan["planned_candidates"] - len(states),
            "awaiting_semantic_review": counts["completed"], "retained_sft_rows": 0,
            "api_request_slots_charged": sum(s.get("api_calls", 0) for s in states),
            "teacher_status": teacher_settings(read_json(root / "teacher-config.json")).status(),
            "gpu_jobs_submitted": 0, "benchmark_gold_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--batch-dir", required=True)
    plan.add_argument("--out-dir", required=True)
    plan.add_argument("--teacher-config", required=True)
    plan.add_argument("--prompt-file", required=True)
    plan.add_argument("--index-dir")
    plan.add_argument("--candidates-per-task", type=int, default=3)
    plan.add_argument("--shard-size", type=int, default=20)
    run = commands.add_parser("run")
    run.add_argument("--plan-dir", required=True)
    run.add_argument("--shard-id", type=int, required=True)
    run.add_argument("--max-candidates", type=int, required=True)
    run.add_argument("--max-api-calls", type=int, required=True)
    run.add_argument("--reconcile-only", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--plan-dir", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        prepare_plan(args.batch_dir, args.out_dir, args.teacher_config, args.prompt_file,
                     candidates_per_task=args.candidates_per_task, shard_size=args.shard_size, index_dir=args.index_dir)
        value = summarize_plan(args.out_dir)
    elif args.command == "run":
        value = run_shard(args.plan_dir, args.shard_id, max_candidates=args.max_candidates,
                          max_api_calls=args.max_api_calls, reconcile_only=args.reconcile_only)
    else:
        value = summarize_plan(args.plan_dir)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

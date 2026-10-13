"""Sharded producer behavior with real AV and local HTTP, never teacher data."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "onesearch_vl"))
sys.path.insert(0, str(ROOT / "validation/research"))
from media_fixtures import make_offset_vfr
from onesearch_infer.av_io import make_native_av_handle, crop_native_av
from teacher_http_fixtures import fixture_server, reply
from research.av_search.corpus import build_corpus
from research.av_train.native_sft import file_sha, validate_trajectory
from research.av_train.teacher_rollouts import prepare_plan, run_shard, summarize_plan


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        source = self.root / "private_answer_title.mkv"
        make_offset_vfr(source)
        media = self.root / "anonymous.mp4"
        crop_native_av(make_native_av_handle(source), 0, 4, media)
        self.media = media
        sha = file_sha(media)
        source_id = "src_" + sha[:24]
        self.doc = {"source_id": source_id, "origin_url": "https://example.org/bell",
                    "asset_sha256": sha, "asset_path": str(media), "media_type": "video/mp4",
                    "duration_s": 4.0,
                    "text": "UNIT TEST bell source: TEST answer fact.", "text_kind": "web_body",
                    "public_metadata": {"title": "UNIT TEST source"}}
        manifest = build_corpus([self.doc], self.batch / "corpus", "unit-teacher", "train")
        self.scope = {"corpus_id": "unit-teacher", "split": "train", "source_ids": [source_id]}
        (self.batch / "scope.json").write_text(json.dumps(self.scope))
        tasks = [{"task_id": f"unit-task-{i}", "split": "train", "question": f"UNIT TEST question {i}",
                  "initial_sources": ["anonymous-clue"]} for i in (1, 2)]
        (self.batch / "tasks-public.jsonl").write_text("".join(json.dumps(x) + "\n" for x in tasks))
        (self.batch / "clue-bindings.private.json").write_text(json.dumps([
            {"clue_id": "anonymous-clue", "source": {"asset_path": str(media), "asset_sha256": sha}}]))
        (self.batch / "answer-candidates.private.jsonl").write_text('"PRIVATE_GOLD_SENTINEL"\n')
        self.config = self.root / "teacher.json"
        self.config.write_text(json.dumps({"schema": "onesearch-gemini-av-teacher.v1"}))
        self.prompt = ROOT / "onesearch_vl/onesearch_infer/prompts/audio_video_search_v2.txt"
        self.env = patch.dict(os.environ, {"ONESEARCH_TEACHER_API_KEY": "UNIT_TEST_NOT_A_SECRET",
                    "ONESEARCH_TEACHER_MODEL": "test-only-gemini", "ONESEARCH_DISABLE_RETRIEVAL": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def plan(self, **kwargs):
        out = self.root / "plan"
        return out, prepare_plan(self.batch, out, self.config, self.prompt,
                                 candidates_per_task=kwargs.pop("candidates_per_task", 2),
                                 shard_size=kwargs.pop("shard_size", 2), **kwargs)

    def replies(self):
        return [(200, reply('<tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":1,"end_s":3}}</tool_call>')),
                (200, reply('<tool_call>{"name":"text_search","arguments":{"q":"bell"}}</tool_call>')),
                (200, reply('<tool_call>{"name":"visit_webpage","arguments":{"url":"https://example.org/bell"}}</tool_call>')),
                (200, reply('<response>TEST answer fact.</response>'))]

    def test_plan_has_unique_candidate_ids_and_no_gold_in_actor_queue(self):
        out, manifest = self.plan()
        self.assertEqual(manifest["planned_candidates"], 4)
        self.assertEqual(manifest["distinct_tasks"], 2)
        rows = [json.loads(line) for path in sorted((out / "shards").glob("*.jsonl"))
                for line in path.read_text().splitlines()]
        self.assertEqual(len({r["case_id"] for r in rows}), 4)
        self.assertEqual([r["task_id"] for r in rows], ["unit-task-1"] * 2 + ["unit-task-2"] * 2)
        self.assertNotIn("PRIVATE_GOLD_SENTINEL", json.dumps(rows))
        status = summarize_plan(out)
        self.assertEqual(status["pending_candidates"], 4)
        self.assertEqual(status["retained_sft_rows"], 0)

    def test_real_loop_executes_crop_search_visit_and_resume_sends_no_duplicate_requests(self):
        out, _ = self.plan(candidates_per_task=1, shard_size=2)
        with fixture_server(self.replies() * 2) as (url, received):
            os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
            result = run_shard(out, 0, max_candidates=1, max_api_calls=8)
            self.assertEqual(result["new_completed_candidates"], 1)
            self.assertEqual(result["new_api_calls"], 4)
            self.assertEqual(len(received), 4)
            first = out / "episodes/unit-task-1__c01/unit-task-1__c01_trajectory.json"
            first_hash = file_sha(first)
            second = run_shard(out, 0, max_candidates=1, max_api_calls=8)
        self.assertEqual(second["new_completed_candidates"], 1)
        self.assertEqual(len(received), 8)
        self.assertEqual(file_sha(first), first_hash)
        with fixture_server([]) as (url, no_requests):
            # Completed shards return before resolving a changed endpoint.
            os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
            third = run_shard(out, 0, max_candidates=1, max_api_calls=8)
        self.assertEqual(third["new_api_calls"], 0)
        self.assertEqual(no_requests, [])
        status = summarize_plan(out)
        self.assertEqual(status["completed_candidates"], 2)
        self.assertEqual(status["retained_sft_rows"], 0)
        self.assertEqual(status["awaiting_semantic_review"], 2)
        trace = json.loads(first.read_text())
        validate_trajectory(trace)
        self.assertEqual(len(trace["turns"]), 4)
        self.assertIn("TEST answer fact", json.dumps(received[2]["payload"]))
        self.assertNotIn("PRIVATE_GOLD_SENTINEL", json.dumps(received))
        self.assertNotIn("UNIT_TEST_NOT_A_SECRET", (out / "plan.json").read_text())

    def test_openai_gateway_queue_uses_full_av_and_original_tools(self):
        self.config.write_text(json.dumps({"schema":"onesearch-openai-av-teacher.v1"}))
        os.environ["ONESEARCH_TEACHER_MODEL"]="gemini-test-only"
        out,_=self.plan(candidates_per_task=1)
        replies=[]
        for code,body in self.replies():
            text=body["candidates"][0]["content"]["parts"][0]["text"]
            replies.append((code,{"model":"gemini-test-only","choices":[{"message":{"content":text},"finish_reason":"stop"}],
                                  "usage":{"completion_tokens":20}}))
        with fixture_server(replies) as (url,received):
            os.environ["ONESEARCH_TEACHER_BASE_URL"]=url
            result=run_shard(out,0,max_candidates=1,max_api_calls=4)
        self.assertEqual(result["new_completed_candidates"],1)
        self.assertEqual(result["new_api_calls"],4)
        self.assertTrue(received[0]["path"].endswith('/chat/completions'))
        self.assertIn('data:video/mp4;base64,',json.dumps(received[0]["payload"]))
        self.assertEqual(summarize_plan(out)["teacher_status"]["provider"],"openai_compatible_av")

    def test_api_budget_counts_uncertain_calls_and_preserves_failed_candidate(self):
        out, _ = self.plan(candidates_per_task=1)
        with fixture_server(self.replies()) as (url, received):
            os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
            result = run_shard(out, 0, max_candidates=2, max_api_calls=1)
        self.assertEqual(len(received), 1)
        self.assertEqual(result["new_api_calls"], 1)
        self.assertEqual(result["new_completed_candidates"], 0)
        self.assertEqual(summarize_plan(out)["failed_candidates"], 1)
        state = json.loads((out / "episodes/unit-task-1__c01/candidate-state.json").read_text())
        self.assertEqual(state["api_calls"], 1)
        with fixture_server([]) as (url, received):
            os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
            # Same endpoint was frozen: changing it is rejected before traffic.
            with self.assertRaises(ValueError):
                run_shard(out, 0, max_candidates=1, max_api_calls=4)
        self.assertEqual(received, [])

    def test_provider_http_failure_stops_other_shards_before_paid_calls(self):
        out, _ = self.plan(candidates_per_task=1, shard_size=1)
        with fixture_server([(503, {"error": "PRIVATE_PROVIDER_BODY"})]) as (url, received):
            os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
            first = run_shard(out, 0, max_candidates=1, max_api_calls=8)
            second = run_shard(out, 1, max_candidates=1, max_api_calls=8)
        self.assertEqual(first["new_failed_candidates"], 1)
        self.assertEqual(len(received), 1)
        self.assertEqual(second["new_api_calls"], 0)
        self.assertEqual(summarize_plan(out)["pending_candidates"], 1)
        stop = (out / "provider-stop.json").read_text()
        self.assertNotIn("PRIVATE_PROVIDER_BODY", stop)
        self.assertNotIn("UNIT_TEST_NOT_A_SECRET", stop)

    def test_provider_review_failure_leaves_other_cases_unattempted(self):
        from research.av_train.teacher_silver_review import review
        refs = [{"task_id": f"unit-task-{i}", "answer_fields": [], "support_quotes": []} for i in (1, 2)]
        (self.batch / "answer-candidates.private.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))
        out, _ = self.plan(candidates_per_task=1, shard_size=2)
        with fixture_server(self.replies() * 2 + [(503, {"error": "PRIVATE_PROVIDER_BODY"})]) as (url, received):
            os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
            run_shard(out, 0, max_candidates=2, max_api_calls=8)
            reviewed = review(out, self.root / "review", workers=1, max_candidates=2)
            again = review(out, self.root / "review", workers=1, max_candidates=2)
        self.assertEqual(len(received), 9)
        self.assertEqual(reviewed["failed_reviews"], 1)
        self.assertEqual(reviewed["retained_silver_candidates"], 0)
        self.assertEqual(again["attempted_reviews"], 1)
        self.assertTrue(reviewed["provider_stopped"])

    def test_running_candidate_after_crash_is_preserved_without_auto_retry(self):
        out, _ = self.plan(candidates_per_task=1)
        candidate = out / "episodes/unit-task-1__c01"
        candidate.mkdir()
        state = {"status": "running", "api_calls": 1, "plan_sha256": file_sha(out / "plan.json")}
        (candidate / "candidate-state.json").write_text(json.dumps(state))
        # No transport configuration needed to reconcile an interrupted attempt.
        with patch.dict(os.environ, {"ONESEARCH_TEACHER_API_KEY": ""}):
            result = run_shard(out, 0, max_candidates=1, max_api_calls=1, reconcile_only=True)
        self.assertEqual(result["new_api_calls"], 0)
        updated = json.loads((candidate / "candidate-state.json").read_text())
        self.assertEqual(updated["status"], "interrupted")
        self.assertEqual(updated["api_calls"], 1)

    def test_non_train_batch_and_changed_media_are_rejected_before_requests(self):
        self.scope["split"] = "dev"
        (self.batch / "scope.json").write_text(json.dumps(self.scope))
        with self.assertRaisesRegex(ValueError, "train"):
            self.plan()
        self.scope["split"] = "train"
        (self.batch / "scope.json").write_text(json.dumps(self.scope))
        out, _ = self.plan(candidates_per_task=1)
        self.media.write_bytes(b"CHANGED UNIT TEST MEDIA")
        with fixture_server([]) as (url, received):
            os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
            with self.assertRaisesRegex(ValueError, "media"):
                run_shard(out, 0, max_candidates=1, max_api_calls=4)
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()

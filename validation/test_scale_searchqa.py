"""Scaling regressions: count exported questions once, require matching CPU audit."""
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import test_native_omni_sft as export_fixtures
from research.av_train.native_sft import export_episodes, file_sha

class ScaleTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module("research.av_train.scale_searchqa")
        except ModuleNotFoundError:self.fail("Reusable scale batch/count API is missing")

    def test_pool_counts_unique_tasks_and_requires_hash_bound_audit(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            _,episodes,verdict,resolver=export_fixtures.NativeSftTests().fixture(root)
            data=root/"export"
            with patch("research.av_train.native_sft.audio_sidecar",side_effect=lambda path,*a:resolver(path)):
                export_episodes(episodes,data,verdicts=verdict)
            copy=root/"duplicate";shutil.copytree(data,copy)
            result=api.verified_pool([data,copy])
            self.assertEqual(result["distinct_retained_questions"],1)
            self.assertEqual(result["cpu_verified_questions"],0)
            audit=root/"silver-cpu-audit";audit.mkdir()
            receipt={"status":"passed","audit_only":False,"dataset_summary_sha256":file_sha(data/"summary.json"),
                     "original_sampler_native_video_coverage_verified":True,"rows":[{"dataset":"trajectory","row_index":0},{"dataset":"answer_only","row_index":0}]}
            (audit/"label-media-audit.private.json").write_text(json.dumps(receipt))
            self.assertEqual(api.verified_pool([data,copy])["cpu_verified_questions"],1)
            rows=json.loads((data/"trajectory.json").read_text());rows[0]["conversations"][-1]["value"]="TAMPERED"
            (data/"trajectory.json").write_text(json.dumps(rows))
            with self.assertRaises(ValueError):api.verified_pool([data])

    def test_slice_preserves_corpus_and_selects_only_requested_task(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);batch=root/"batch";batch.mkdir();corpus=batch/"corpus";corpus.mkdir()
            (corpus/"documents.jsonl").write_text('CORPUS_IMMUTABLE_SENTINEL')
            tasks=[{"task_id":"a","question":"第一题","initial_sources":["clue_a"],"split":"train"},
                   {"task_id":"b","question":"SECOND_TASK_SENTINEL","initial_sources":["clue_b"],"split":"train"}]
            (batch/"tasks-public.jsonl").write_text("".join(json.dumps(x)+"\n" for x in tasks))
            (batch/"scope.json").write_text(json.dumps({"split":"train","corpus_id":"same-corpus","source_ids":["s1"]}))
            (batch/"clue-bindings.private.json").write_text(json.dumps([{"clue_id":"clue_a"},{"clue_id":"clue_b"}]))
            (batch/"answer-candidates.private.jsonl").write_text("".join(json.dumps({"task_id":x["task_id"]})+"\n" for x in tasks))
            (batch/"clue-lineage.private.json").write_text(json.dumps([{"task_id":x["task_id"]} for x in tasks]))
            out=root/"slice";api.slice_batch(batch,out,["a"])
            self.assertNotIn("SECOND_TASK_SENTINEL",(out/"tasks-public.jsonl").read_text())
            self.assertEqual(json.loads((out/"clue-bindings.private.json").read_text()),[{"clue_id":"clue_a"}])
            self.assertEqual((corpus/"documents.jsonl").read_text(),'CORPUS_IMMUTABLE_SENTINEL')
            self.assertEqual(json.loads((out/"scope.json").read_text())["corpus_id"],"same-corpus")
            with self.assertRaises(ValueError):api.slice_batch(batch,root/"bad",["missing"])

if __name__=="__main__":unittest.main()

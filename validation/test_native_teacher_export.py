"""Provider-neutral export tests. All generated text here is explicit test data."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_native_omni_sft as original_tests
from research.av_train.native_sft import export_episodes, file_sha, validate_trajectory


class TeacherExportTests(unittest.TestCase):
    def test_structured_generation_receipt_validates_in_standalone_exporter_process(self):
        with tempfile.TemporaryDirectory() as root:
            trace, _, _, _ = self.teacher_fixture(root)
            provider = json.dumps({"action": "final", "summary": "UNIT TEST", "arguments": {}, "answer": "UNIT TEST final"})
            canonical = '<think>UNIT TEST</think>\n<response>UNIT TEST final</response>'
            snapshot = deepcopy(trace["turns"][-1]["chat_completions"])
            snapshot[-1]["content"] = canonical
            response = deepcopy(trace["turns"][-1]["response"])
            response["candidates"][0]["content"]["parts"] = [{"text": canonical}]
            raw = response["raw_response"]
            raw["structured_action"] = {"protocol": "onesearch_action_json.v1", "provider_text": provider}
            raw["generation_receipt"].update(output_transport="onesearch_action_json.v1",
                provider_output_sha256=hashlib.sha256(provider.encode()).hexdigest(),
                output_sha256=hashlib.sha256(canonical.encode()).hexdigest())
            fixture = Path(root)/'receipt.json';fixture.write_text(json.dumps([response,snapshot]))
            repo = Path(__file__).resolve().parents[1]
            env = dict(os.environ);env.pop('PYTHONPATH',None)
            code = 'import json,sys; from research.av_train.generation_receipts import generation_identity; print(generation_identity(*json.load(open(sys.argv[1])))["provider"])'
            result = subprocess.run([sys.executable,'-c',code,str(fixture)],cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'gemini_native')

    def teacher_fixture(self, root):
        helper = original_tests.NativeSftTests(methodName="runTest")
        trace, episodes, verdict, resolver = helper.fixture(root)
        media = Path(root) / "query.mp4"
        for turn in trace["turns"]:
            turn["response"]["raw_response"] = {"generation_receipt": {
                "schema": "onesearch-native-generation.v1", "provider": "gemini_native",
                "model_id": "test-only-gemini", "model_version": "test-only-gemini",
                "endpoint_sha256": "2" * 64, "input_mode": "native_audio_video", "status": "complete",
                "request_sha256": "1" * 64, "generated_tokens": 10,
                "output_sha256": hashlib.sha256(turn["response_text"].encode()).hexdigest(),
                "media": [{"kind": "video", "sha256": file_sha(media), "bytes": media.stat().st_size}]}}
        path = episodes / "unit-train-001/unit-train-001_trajectory.json"
        path.write_text(json.dumps(trace))
        verdict["unit-train-001"]["trajectory_sha256"] = file_sha(path)
        return trace, episodes, verdict, resolver

    def test_external_native_teacher_can_export_same_assistant_targets(self):
        with tempfile.TemporaryDirectory() as root:
            trace, episodes, verdict, resolver = self.teacher_fixture(root)
            self.assertEqual(validate_trajectory(trace)[-1]["content"], trace["turns"][-1]["response_text"])
            with patch("research.av_train.native_sft.audio_sidecar", side_effect=lambda path, *args: resolver(path)):
                result = export_episodes(episodes, Path(root) / "sft", verdicts=verdict)
            self.assertEqual(result["retained_count"], 1)
            lineage = json.loads((Path(root) / "sft/lineage.private.json").read_text())
            self.assertEqual(lineage[0]["teacher_identity"]["model_id"], "test-only-gemini")
            rows = json.loads((Path(root) / "sft/trajectory.json").read_text())
            self.assertEqual(rows[0]["conversations"][1]["value"], trace["turns"][0]["response_text"])

    def test_multiple_candidates_map_to_one_task_without_losing_lineage(self):
        with tempfile.TemporaryDirectory() as root:
            trace, episodes, _, resolver = self.teacher_fixture(root)
            inputs = json.loads((episodes / "inputs.private.json").read_text())
            binding = inputs["cases"][0]
            inputs["cases"] = []
            verdicts = {}
            original_path = episodes / "unit-train-001/unit-train-001_trajectory.json"
            original_path.unlink()
            for index in (1, 2):
                candidate = f"unit-train-001__c{index:02d}"
                folder = episodes / candidate
                folder.mkdir()
                value = deepcopy(trace)
                value["case_id"] = candidate
                path = folder / f"{candidate}_trajectory.json"
                path.write_text(json.dumps(value))
                inputs["cases"].append({**binding, "case_id": candidate, "task_id": "unit-train-001"})
                verdicts[candidate] = {"success": True, "split": "train", "reviewer_id": "unit-test-only",
                                      "trajectory_sha256": file_sha(path), "quality_tier": "silver"}
            (episodes / "inputs.private.json").write_text(json.dumps(inputs))
            with patch("research.av_train.native_sft.audio_sidecar", side_effect=lambda path, *args: resolver(path)):
                result = export_episodes(episodes, Path(root) / "sft", verdicts=verdicts)
            self.assertEqual(result["retained_count"], 2)
            self.assertEqual(result["distinct_task_count"], 1)
            lineage = json.loads((Path(root) / "sft/lineage.private.json").read_text())
            self.assertEqual([row["task_id"] for row in lineage], ["unit-train-001"] * 2)
            self.assertEqual([row["case_id"] for row in lineage], ["unit-train-001__c01", "unit-train-001__c02"])

    def test_text_only_or_missing_native_receipts_remain_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            trace, _, _, _ = self.teacher_fixture(root)
            for kind in ("provider", "media", "receipt"):
                value = deepcopy(trace)
                raw = value["turns"][0]["response"]["raw_response"]
                if kind == "provider": raw["generation_receipt"]["provider"] = "text_only"
                elif kind == "media": raw["generation_receipt"]["media"] = []
                else: raw.clear()
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    validate_trajectory(value)


if __name__ == "__main__":
    unittest.main()

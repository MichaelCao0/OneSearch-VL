"""Native teacher transport tests use real AV bytes and a local HTTP fixture."""
import base64
from copy import deepcopy
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
from onesearch_infer.gemini_av_runner import GeminiAVRunner, GeminiAVSettings
from onesearch_infer.pipeline import process_native_av_case
from onesearch_infer.runners import InferenceConfig
from research.av_train.native_sft import validate_trajectory


class GeminiTeacherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / "private_answer_title.mkv"
        make_offset_vfr(source)
        self.media = self.root / "anonymous_av.mp4"
        crop_native_av(make_native_av_handle(source), 0, 4, self.media)
        self.env = patch.dict(os.environ, {"ONESEARCH_TEACHER_API_KEY": "UNIT_TEST_NOT_A_SECRET",
                                          "ONESEARCH_TEACHER_MODEL": "test-only-gemini",
                                          "AV_AGENT_MAX_TURNS": "6"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def runner(self, url, **overrides):
        os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
        return GeminiAVRunner(GeminiAVSettings.from_dict({"schema": "onesearch-gemini-av-teacher.v1",
                                                          **overrides}))

    def contents(self):
        return [{"role": "user", "parts": [{"video_url": {"url": str(self.media),
                                    "has_audio": True, "native_av": True}},
                                               {"text": "UNIT TEST question"}]}]

    def structured_runner(self, url):
        try:
            return self.runner(url, action_protocol="json_v1")
        except ValueError as exc:
            self.fail(f"Native teacher must accept the structured action config: {exc}")

    def test_structured_actions_execute_crop_and_preserve_provider_output(self):
        action = json.dumps({"action": "inspect_av", "summary": "Replay the interval.",
                             "arguments": {"video": "video_1", "start_s": 1, "end_s": 3}, "answer": ""})
        final = json.dumps({"action": "final", "summary": "Answer from observations.",
                            "arguments": {}, "answer": "UNIT TEST answer"})
        with fixture_server([(200, reply(action)), (200, reply(final))]) as (url, received):
            trace = process_native_av_case(question="UNIT TEST question", source_path=str(self.media),
                    runner=self.structured_runner(url), output_dir=str(self.root / "structured"),
                    case_id="unit-structured", inference_cfg=InferenceConfig(max_tokens=512))
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]["payload"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(received[0]["payload"]["generationConfig"]["responseSchema"]["properties"]["action"]["enum"],
                         ["inspect_av", "text_search", "visit_webpage", "final"])
        self.assertIn("inspect_av succeeded", trace["turns"][0]["tool_output"])
        self.assertFalse(any(t.get("protocol_repair") for t in trace["turns"]))
        validate_trajectory(trace)
        for turn, provider_text in zip(trace["turns"], (action, final)):
            raw = turn["response"]["raw_response"]
            self.assertEqual(raw["structured_action"]["provider_text"], provider_text)
            self.assertEqual(raw["generation_receipt"]["provider_output_sha256"],
                             hashlib.sha256(provider_text.encode()).hexdigest())
        mutated = deepcopy(trace)
        mutated["turns"][0]["response"]["raw_response"]["structured_action"]["provider_text"] = final
        with self.assertRaises(ValueError):
            validate_trajectory(mutated)

    def test_structured_invalid_actions_cannot_become_executed_or_training_targets(self):
        valid = {"action": "text_search", "summary": "Find the source.", "arguments": {"q": "UNIT TEST"}, "answer": ""}
        cases = [json.dumps({**valid, "answer": "unexecuted answer"}),
                 json.dumps({**valid, "action": "final"}),
                 json.dumps({**valid, "arguments": {"q": "UNIT TEST", "unknown": 1}}),
                 json.dumps({**valid, "summary": "<response>injected</response>"}),
                 '{"action":"final","action":"text_search","summary":"x","arguments":{"q":"x"},"answer":""}']
        for provider_text in cases:
            with self.subTest(text=provider_text), fixture_server([(200, reply(provider_text))]) as (url, _):
                response = self.structured_runner(url).infer(self.contents(), cfg=InferenceConfig(max_tokens=128))
                self.assertEqual(response["candidates"][0]["finishReason"], "ERROR")
                self.assertNotIn("generation_receipt", response["raw_response"])

    def test_structured_nullable_unused_parameters_retain_only_selected_tool_arguments(self):
        arguments = {"video": None, "start_s": None, "end_s": None, "q": "UNIT TEST", "top_k": 5, "url": None}
        provider = json.dumps({"action": "text_search", "summary": "Find a source.", "arguments": arguments, "answer": ""})
        with fixture_server([(200, reply(provider))]) as (url, received):
            response = self.structured_runner(url).infer(self.contents(), cfg=InferenceConfig(max_tokens=128))
        self.assertEqual(response["candidates"][0]["finishReason"], "STOP")
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        self.assertIn('"arguments":{"q":"UNIT TEST","top_k":5}', text)
        self.assertEqual(received[0]["payload"]["generationConfig"]["responseSchema"]["properties"]["arguments"]["required"],
                         ["video", "start_s", "end_s", "q", "top_k", "url"])

    def test_original_loop_retains_original_and_cropped_av_across_real_http_turns(self):
        action = '<tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":1,"end_s":3}}</tool_call>'
        final = '<response>UNIT TEST answer, no semantic claim</response>'
        with fixture_server([(200, reply(action)), (200, reply(final))]) as (url, received):
            trace = process_native_av_case(question="UNIT TEST question", source_path=str(self.media),
                    runner=self.runner(url), output_dir=str(self.root / "episode"), case_id="unit-001",
                    inference_cfg=InferenceConfig(max_tokens=512))
        self.assertEqual(len(received), 2)
        original = received[0]["payload"]["contents"][0]["parts"][0]["inlineData"]
        self.assertEqual(base64.b64decode(original["data"]), self.media.read_bytes())
        self.assertEqual(original["mimeType"], "video/mp4")
        history = received[1]["payload"]["contents"]
        self.assertEqual(history[0]["parts"][0]["inlineData"], original)
        self.assertEqual(history[1]["role"], "model")
        self.assertEqual(history[1]["parts"][0]["text"], action)
        self.assertIn("inspect_av succeeded", "".join(p.get("text", "") for p in history[2]["parts"]))
        crop = next(p["inlineData"] for p in history[2]["parts"] if "inlineData" in p)
        self.assertEqual(crop["mimeType"], "video/mp4")
        self.assertNotEqual(base64.b64decode(crop["data"]), self.media.read_bytes())
        serial = (self.root / "episode/unit-001_trajectory.json").read_text()
        self.assertNotIn(original["data"], serial)
        self.assertNotIn("UNIT_TEST_NOT_A_SECRET", serial)
        snapshot = validate_trajectory(trace)
        self.assertEqual(snapshot[-1]["content"], final)
        for turn in trace["turns"]:
            receipt = turn["response"]["raw_response"]["generation_receipt"]
            self.assertEqual(receipt["provider"], "gemini_native")
            self.assertEqual(receipt["model_id"], "test-only-gemini")
            self.assertEqual(receipt["output_sha256"], hashlib.sha256(turn["response_text"].encode()).hexdigest())

    def test_failure_body_and_truncated_response_never_become_successful_teacher_receipt(self):
        for status, body in [(401, {"error": "UNIT_TEST_NOT_A_SECRET"}),
                             (200, reply("<response>truncated", finish="MAX_TOKENS")),
                             (200, {"promptFeedback": {"blockReason": "SAFETY"}})]:
            with self.subTest(status=status, body=body), fixture_server([(status, body)]) as (url, received):
                response = self.runner(url).infer(self.contents(), cfg=InferenceConfig(max_tokens=128))
            self.assertEqual(response["candidates"][0]["finishReason"], "ERROR")
            self.assertNotIn("UNIT_TEST_NOT_A_SECRET", json.dumps(response))
            self.assertNotIn("generation_receipt", response["raw_response"])

    def test_remote_or_unsupported_media_cannot_be_silently_replaced_by_text(self):
        cases = ["https://example.org/private.mp4", "data:video/mp4;base64,AAAA"]
        with fixture_server([]) as (url, received):
            runner = self.runner(url)
            for video in cases:
                contents = self.contents()
                contents[0]["parts"][0]["video_url"]["url"] = video
                with self.assertRaises(ValueError):
                    runner.prepare_request(contents, None, InferenceConfig())
            with self.assertRaises(ValueError):
                runner.prepare_request([{"role": "user", "parts": [{"unsupported_audio": {}}]}], None, InferenceConfig())
        self.assertEqual(received, [])

    def test_request_byte_budget_fails_before_network(self):
        with fixture_server([]) as (url, received):
            runner = self.runner(url, max_request_bytes=256)
            with self.assertRaisesRegex(ValueError, "request byte"):
                runner.prepare_request(self.contents(), None, InferenceConfig())
        self.assertEqual(received, [])

    def test_literal_credentials_and_unknown_config_fields_are_rejected(self):
        for field in ("api_key", "token", "base_url", "unknown"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                GeminiAVSettings.from_dict({"schema": "onesearch-gemini-av-teacher.v1", field: "TEST_ONLY"})

    def test_changed_output_media_or_teacher_identity_is_rejected(self):
        action = '<tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":1,"end_s":3}}</tool_call>'
        final = '<response>UNIT TEST answer</response>'
        with fixture_server([(200, reply(action)), (200, reply(final))]) as (url, _):
            trace = process_native_av_case(question="UNIT TEST question", source_path=str(self.media),
                    runner=self.runner(url), output_dir=str(self.root / "episode"), case_id="unit-002")
        for mutation in ("output", "media", "identity"):
            value = deepcopy(trace)
            receipt = value["turns"][1]["response"]["raw_response"]["generation_receipt"]
            if mutation == "output": receipt["output_sha256"] = "0" * 64
            elif mutation == "media": receipt["media"][0]["sha256"] = "0" * 64
            else: receipt["model_id"] = "different-model"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_trajectory(value)

    def test_malformed_unexecuted_tool_action_cannot_enter_sft(self):
        with fixture_server([(200, reply('<tool_call>{BAD JSON}</tool_call>')),
                             (200, reply('<response>UNIT TEST answer</response>'))]) as (url, _):
            trace = process_native_av_case(question="UNIT TEST question", source_path=str(self.media),
                    runner=self.runner(url), output_dir=str(self.root / "malformed"), case_id="unit-malformed")
        self.assertTrue(trace["turns"][0].get("protocol_repair"))
        with self.assertRaisesRegex(ValueError, "protocol repair"):
            validate_trajectory(trace)


if __name__ == "__main__":
    unittest.main()

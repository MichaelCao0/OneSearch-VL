"""Real HTTP AV serialization for Gemini through an OpenAI-compatible gateway."""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest

import test_gemini_av_teacher as gemini_tests
from teacher_http_fixtures import fixture_server
from onesearch_infer.pipeline import process_native_av_case
from onesearch_infer.runners import InferenceConfig
from research.av_train.native_sft import validate_trajectory


def reply(text, *, finish="stop"):
    return {"model": "test-only-gemini", "choices": [{"message": {"role": "assistant", "content": text},
               "finish_reason": finish}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


class OpenAITeacherTests(unittest.TestCase):
    def setUp(self):
        self.helper = gemini_tests.GeminiTeacherTests(methodName="runTest")
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.root, self.media = self.helper.root, self.helper.media
        os.environ["ONESEARCH_TEACHER_MODEL"] = "gemini-test-only"

    def runner(self, url, **overrides):
        from onesearch_infer.openai_av_runner import OpenAIAVRunner, OpenAIAVSettings
        os.environ["ONESEARCH_TEACHER_BASE_URL"] = url
        return OpenAIAVRunner(OpenAIAVSettings.from_dict({"schema": "onesearch-openai-av-teacher.v1", **overrides}))

    def test_native_video_bytes_and_real_crop_survive_original_multi_turn_loop(self):
        action='<tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":1,"end_s":3}}</tool_call>'
        final='<response>UNIT TEST answer</response>'
        with fixture_server([(200,reply(action)),(200,reply(final))]) as (url,received):
            trace=process_native_av_case(question="UNIT TEST question",source_path=str(self.media),
                  runner=self.runner(url),output_dir=str(self.root/"openai"),case_id="test-openai")
        self.assertTrue(received[0]["path"].endswith("/chat/completions"))
        messages=received[0]["payload"]["messages"]
        original=next(p["video_url"]["url"] for p in messages[1]["content"] if p["type"]=="video_url")
        self.assertTrue(original.startswith("data:video/mp4;base64,"))
        self.assertEqual(base64.b64decode(original.split(",",1)[1]),self.media.read_bytes())
        history=received[1]["payload"]["messages"]
        self.assertEqual(history[2]["role"],"assistant")
        self.assertEqual(history[2]["content"][0]["text"],action)
        crop=next(p["video_url"]["url"] for p in history[3]["content"] if p["type"]=="video_url")
        self.assertNotEqual(crop,original)
        snapshot=validate_trajectory(trace)
        self.assertEqual(snapshot[-1]["content"],final)
        receipt=trace["turns"][-1]["response"]["raw_response"]["generation_receipt"]
        self.assertEqual(receipt["provider"],"openai_compatible_av")
        self.assertEqual(receipt["media_transport"],"video_url_data_uri")
        self.assertEqual(receipt["output_sha256"],hashlib.sha256(final.encode()).hexdigest())
        serial=json.dumps(trace)
        self.assertNotIn(original,serial);self.assertNotIn("UNIT_TEST_NOT_A_SECRET",serial)

    def test_error_truncation_or_missing_usage_cannot_enter_sft(self):
        missing=reply('<response>TEST</response>');missing.pop('usage')
        for code,body in [(400,{"error":"UNIT_TEST_NOT_A_SECRET"}),
                          (200,reply('<response>TEST',finish="length")),(200,missing)]:
            with self.subTest(code=code),fixture_server([(code,body)]) as (url,_):
                result=self.runner(url).infer(self.helper.contents(),cfg=InferenceConfig(max_tokens=128))
            self.assertEqual(result["candidates"][0]["finishReason"],"ERROR")
            self.assertNotIn("UNIT_TEST_NOT_A_SECRET",json.dumps(result))
            self.assertNotIn("generation_receipt",result["raw_response"])

    def test_no_frames_text_fallback_or_non_gemini_model(self):
        with fixture_server([]) as (url,received):
            runner=self.runner(url,max_request_bytes=256)
            with self.assertRaisesRegex(ValueError,"request byte"):
                runner.prepare_request(self.helper.contents(),None,InferenceConfig())
            os.environ["ONESEARCH_TEACHER_MODEL"]="text-only-model"
            with self.assertRaises(ValueError):self.runner(url)
        self.assertEqual(received,[])


if __name__=='__main__':unittest.main()

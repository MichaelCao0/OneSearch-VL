"""Real concurrent calls retain results and collect only sanitized timing metadata."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

class TelemetryTests(unittest.TestCase):
    def api(self):
        from research.av_train import teacher_telemetry
        return teacher_telemetry

    def runner(self):
        class Runner:
            def prepare_request(self, *args):
                time.sleep(.005)
                return b"SECRET_PAYLOAD", [{"bytes": 17, "sha256": "PRIVATE_MEDIA_HASH"}]
            def infer(self, *args, **kwargs):
                self.prepare_request(*args)
                time.sleep(.01)
                return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "PRIVATE_ANSWER"}]}}],
                        "usageMetadata": {"promptTokenCount": 21, "candidatesTokenCount": 7}}
        return Runner

    def test_result_identity_timing_and_no_payload_recorded(self):
        api=self.api(); cls=self.runner(); api.instrument_runner_class(cls)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"ONESEARCH_TEACHER_PROFILE_DIR": folder, "ONESEARCH_TEACHER_PROFILE_STAGE": "author"}):
            result=cls().infer("PRIVATE_INPUT", "PRIVATE_SYSTEM", "PRIVATE_KEY")
            self.assertEqual(result["candidates"][0]["content"]["parts"][0]["text"], "PRIVATE_ANSWER")
            records=list(Path(folder).glob("*.json")); self.assertEqual(len(records), 1)
            text=records[0].read_text(); row=json.loads(text)
            for secret in ["PRIVATE_INPUT", "PRIVATE_SYSTEM", "PRIVATE_KEY", "SECRET_PAYLOAD", "PRIVATE_MEDIA_HASH", "PRIVATE_ANSWER"]:
                self.assertNotIn(secret, text)
            self.assertEqual(row["stage"], "author")
            self.assertEqual(row["request_bytes"], len(b"SECRET_PAYLOAD"))
            self.assertEqual(row["generated_tokens"], 7)
            self.assertGreater(row["prepare_s"], 0)
            self.assertGreater(row["api_wait_and_parse_s"], 0)
            self.assertAlmostEqual(row["elapsed_s"], row["prepare_s"]+row["api_wait_and_parse_s"], places=5)

    def test_exception_is_reraised_and_private_error_text_is_not_logged(self):
        api=self.api(); cls=self.runner()
        def fail(*args, **kwargs): raise ValueError("PRIVATE_EXCEPTION")
        cls.infer=fail; api.instrument_runner_class(cls)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"ONESEARCH_TEACHER_PROFILE_DIR": folder}):
            with self.assertRaisesRegex(ValueError, "PRIVATE_EXCEPTION"): cls().infer("PRIVATE_INPUT")
            text=next(Path(folder).glob("*.json")).read_text()
            self.assertNotIn("PRIVATE_EXCEPTION", text)
            self.assertEqual(json.loads(text)["error_type"], "ValueError")

    def test_eight_inflight_calls_have_distinct_receipts_and_per_call_timing(self):
        api=self.api(); cls=self.runner(); original=cls.infer; active=0; peak=0; lock=threading.Lock(); barrier=threading.Barrier(8)
        def concurrent(self, *args, **kwargs):
            nonlocal active, peak
            with lock: active+=1; peak=max(active, peak)
            try:
                barrier.wait(timeout=5)
                return original(self, *args, **kwargs)
            finally:
                with lock: active-=1
        cls.infer=concurrent; api.instrument_runner_class(cls); api.instrument_runner_class(cls)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"ONESEARCH_TEACHER_PROFILE_DIR": folder}):
            with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(lambda _:cls().infer("PRIVATE_INPUT"), range(8)))
            receipts=[json.loads(p.read_text()) for p in Path(folder).glob("*.json")]
            self.assertEqual(peak, 8); self.assertEqual(len(results), 8); self.assertEqual(len(receipts), 8)
            self.assertTrue(all(r["prepare_s"]>0 for r in receipts))

if __name__ == "__main__": unittest.main()

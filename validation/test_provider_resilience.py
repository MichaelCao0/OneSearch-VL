import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

def response(status):
    return {"candidates":[{"finishReason":"ERROR","content":{"parts":[{"text":f"Native teacher HTTP {status}; response body withheld"}]}}]}

class ResilienceTests(unittest.TestCase):
    def guard(self, path):
        from research.av_train.provider_resilience import BurstProviderGuard
        return BurstProviderGuard(path)
    def test_default_policy_stops_at_first_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"ONESEARCH_TEACHER_STOP_FILE":"", "ONESEARCH_TEACHER_TRANSIENT_LIMIT":"1"}):
            g=self.guard(folder);g.observe(response(503));self.assertTrue(g.stopped)
    def test_isolated_503_does_not_stop_but_burst_of_three_does(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"ONESEARCH_TEACHER_STOP_FILE":"", "ONESEARCH_TEACHER_TRANSIENT_LIMIT":"3"}):
            g=self.guard(folder);g.observe(response(503));self.assertFalse(g.stopped)
            g.observe(response(502));self.assertFalse(g.stopped)
            g.observe(response(503));self.assertTrue(g.stopped)
    def test_auth_and_rate_limit_failures_stop_immediately(self):
        for status in [401,403,429]:
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"ONESEARCH_TEACHER_STOP_FILE":"", "ONESEARCH_TEACHER_TRANSIENT_LIMIT":"3"}):
                g=self.guard(folder);g.observe(response(status));self.assertTrue(g.stopped)
    def test_single_bad_sample_does_not_stop_other_samples(self):
        for status in [400,413,422]:
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"ONESEARCH_TEACHER_STOP_FILE":"", "ONESEARCH_TEACHER_TRANSIENT_LIMIT":"3"}):
                g=self.guard(folder);g.observe(response(status));self.assertFalse(g.stopped)
    def test_isolated_transport_failure_does_not_replay_or_stop_whole_queue(self):
        failure={"candidates":[{"finishReason":"ERROR","content":{"parts":[{"text":"Native teacher transport failed (ReadTimeout); response unavailable"}]}}]}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"ONESEARCH_TEACHER_STOP_FILE":"", "ONESEARCH_TEACHER_TRANSIENT_LIMIT":"3"}):
            g=self.guard(folder);g.observe(failure);self.assertFalse(g.stopped)
            g.observe(failure);self.assertFalse(g.stopped)
            g.observe(failure);self.assertTrue(g.stopped)
    def test_shared_burst_record_between_sibling_queues(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"ONESEARCH_TEACHER_STOP_FILE":str(Path(folder)/"shared-stop.json"), "ONESEARCH_TEACHER_TRANSIENT_LIMIT":"3"}):
            g1=self.guard(Path(folder)/"one");g2=self.guard(Path(folder)/"two")
            g1.observe(response(503));g2.observe(response(503));self.assertFalse(g1.stopped)
            g1.observe(response(503));self.assertTrue(g1.stopped);self.assertTrue(g2.stopped)
    def test_expired_failure_is_removed_without_replay(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"ONESEARCH_TEACHER_STOP_FILE":"", "ONESEARCH_TEACHER_TRANSIENT_LIMIT":"3"}):
            with patch("research.av_train.provider_resilience.time.time",return_value=100.):self.guard(folder).observe(response(503))
            with patch("research.av_train.provider_resilience.time.time",return_value=170.):
                g=self.guard(folder);g.observe(response(503));g.observe(response(503));self.assertFalse(g.stopped)
if __name__=="__main__":unittest.main()

import importlib.util
from pathlib import Path
import tempfile
import unittest

class GenerationBoundaryTests(unittest.TestCase):
    def api(self):
        from research.av_train import searchqa
        return searchqa
    def test_action_boundary_stops_before_imagined_tool_responses(self):
        m=self.api();self.assertTrue(hasattr(m,'decision_is_complete'),'Missing single-decision generation stop')
        self.assertFalse(m.decision_is_complete('{"action":{"tool":"search_sources"'))
        self.assertTrue(m.decision_is_complete('{"action":{"tool":"search_sources","arguments":{"query":"engine {1919}","top_k":3}}}'))
        self.assertFalse(m.decision_is_complete('{"action":{"query":"escaped \\" brace }'))
    def test_fenced_final_stops_only_after_closing_fence(self):
        m=self.api();self.assertTrue(hasattr(m,'decision_is_complete'))
        self.assertFalse(m.decision_is_complete('```json\n{"final":{"status":"ok","answer":"x"}}'))
        self.assertTrue(m.decision_is_complete('```json\n{"final":{"status":"ok","answer":"x"}}\n```'))
    def test_multiple_episode_receipts_do_not_overwrite_each_other(self):
        m=self.api();self.assertTrue(hasattr(m,'write_episode_summary'),'Missing create-only per-task progress')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m.write_episode_summary(root,{'task_id':'qa-001','status':'controller_failed'})
            m.write_episode_summary(root,{'task_id':'qa-002','status':'ok'})
            self.assertEqual(len(list((root/'episode-summaries').glob('*.json'))),2)
            with self.assertRaises(FileExistsError):m.write_episode_summary(root,{'task_id':'qa-001','status':'ok'})

if __name__=='__main__':unittest.main()

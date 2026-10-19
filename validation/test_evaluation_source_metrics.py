"""Posthoc source metrics use executed tool receipts, never actor mentions."""
import unittest
from research.av_train import evaluation_analysis

class SourceMetricTests(unittest.TestCase):
    def turn(self, name, output):
        return {"response_text": '<tool_call>{"name":"'+name+'","arguments":{}}</tool_call>', "tool_output":output}
    def test_target_url_in_real_passage_counts_without_internal_id(self):
        trace={"turns":[self.turn("text_search", "<observation>\n[Passage 1]\nURL: https://example.org/File:Bell%27s.webm\nExcerpt: ringing\n</observation>")]}
        actual=evaluation_analysis.source_metrics(trace, {"src_bell":"https://example.org/File:Bell's.webm"})
        self.assertTrue(actual['target_retrieved']);self.assertFalse(actual['target_read'])
        self.assertEqual(actual['matched_reference_source_ids'], ['src_bell'])
    def test_url_in_excerpt_or_actor_and_url_prefix_do_not_count(self):
        trace={"turns":[self.turn("text_search", "URL: https://example.org/source-other\nExcerpt: https://example.org/source src_target")]}
        actual=evaluation_analysis.source_metrics(trace, {'src_target':'https://example.org/source'})
        self.assertFalse(actual['target_retrieved']);self.assertFalse(actual['target_read'])
    def test_visit_requires_actual_target_url_in_successful_tool_output(self):
        trace={"turns":[self.turn("visit_webpage", "URL: https://example.org/source\nBody: source text")]}
        actual=evaluation_analysis.source_metrics(trace, {'src_target':'https://example.org/source'})
        self.assertTrue(actual['target_read']);self.assertFalse(actual['target_retrieved'])
        trace['turns'][0]['error']='execution failed'
        self.assertFalse(evaluation_analysis.source_metrics(trace, {'src_target':'https://example.org/source'})['target_read'])

if __name__=='__main__':unittest.main()

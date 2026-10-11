import importlib.util
import unittest
from research.av_train import searchqa

class SeedExpansionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('research.av_train.seed_expansion'),'Missing bounded seed expansion')
        return __import__('research.av_train.seed_expansion',fromlist=['*'])
    def test_collector_rejects_source_overlap_between_seed_and_diagnostic(self):
        m=self.api()
        with self.assertRaisesRegex(ValueError,'source overlap'):
            m.check_source_partition([{'initial_sources':['src_a']}],[{'initial_sources':['src_a']}])
        m.check_source_partition([{'initial_sources':['src_a']}],[{'initial_sources':['src_b']}])
    def test_fact_match_is_subset_check_not_accuracy_claim(self):
        m=self.api()
        x=m.answer_diagnostic({'status':'ok','answer':'Eugene, Oregon'},['Eugene','Oregon'])
        self.assertTrue(x['requested_field_match']);self.assertFalse(x['full_answer_semantic_verified'])
        x=m.answer_diagnostic({'status':'controller_failed','answer':'Eugene, Oregon'},['Eugene','Oregon'])
        self.assertFalse(x['requested_field_match'])
    def test_fact_match_detects_missing_and_translated_names_explicitly(self):
        m=self.api()
        self.assertFalse(m.answer_diagnostic({'status':'ok','answer':'尤金，俄勒冈'},['Eugene','Oregon'])['requested_field_match'])
        self.assertFalse(m.answer_diagnostic({'status':'ok','answer':'Eugene'},['Eugene','Oregon'])['requested_field_match'])
    def test_canonical_student_history_does_not_convert_to_native_tags(self):
        self.assertTrue(hasattr(searchqa,'controller_history'),'Missing explicit deployment protocol')
        history=[{'role':'assistant','content':'{"action":{"tool":"search_sources","arguments":{"query":"rain","top_k":3}}}'},
                 {'role':'user','content':'<tool_response>\n{"status":"no_results"}\n</tool_response>'}]
        self.assertEqual(searchqa.controller_history(history,'canonical'),history)
        self.assertIn('<tool_call>',searchqa.controller_history(history,'native')[0]['content'])
    def test_unknown_protocol_rejected_before_model_execution(self):
        self.assertTrue(hasattr(searchqa,'controller_history'))
        with self.assertRaises(ValueError):searchqa.controller_history([],'unknown')
if __name__=='__main__':unittest.main()

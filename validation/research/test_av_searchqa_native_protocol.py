import unittest
from research.av_train import searchqa

class NativeProtocolTests(unittest.TestCase):
    def test_original_tool_call_translates_without_inventing_arguments(self):
        self.assertTrue(hasattr(searchqa,'parse_native_decision'),'Missing published tool-call protocol')
        value=searchqa.parse_native_decision('<think>Inspect first.</think>\n<tool_call>\n{"name":"inspect_av","arguments":{"source_id":"src_x","start_s":10,"end_s":12}}\n</tool_call>')
        self.assertEqual(value,{'action':{'tool':'inspect_av','arguments':{'source_id':'src_x','start_s':10,'end_s':12}}})
    def test_plain_native_answer_is_preserved_without_fabricating_citations(self):
        self.assertTrue(hasattr(searchqa,'parse_native_decision'))
        value=searchqa.parse_native_decision('<answer>Eugene, Oregon.</answer>')
        self.assertEqual(value['final']['answer'],'Eugene, Oregon.')
        self.assertEqual(value['final']['evidence'],[])
    def test_native_boundary_waits_for_closing_tag(self):
        self.assertTrue(hasattr(searchqa,'native_decision_is_complete'))
        self.assertFalse(searchqa.native_decision_is_complete('<tool_call>{"name":"x","arguments":{}}'))
        self.assertTrue(searchqa.native_decision_is_complete('<tool_call>{"name":"x","arguments":{}}</tool_call>'))
    def test_history_uses_native_calls_and_real_observation_envelope(self):
        self.assertTrue(hasattr(searchqa,'native_history'))
        history=[{'role':'system','content':'original system'}, {'role':'user','content':'question'},
          {'role':'assistant','content':'{"action":{"tool":"search_sources","arguments":{"query":"rain","top_k":3}}}'},
          {'role':'user','content':'<tool_response>\n{"status":"no_results"}\n</tool_response>'}]
        actual=searchqa.native_history(history)
        self.assertIn('<tool_call>',actual[2]['content']);self.assertIn('"name": "search_sources"',actual[2]['content'])
        self.assertIn('<observation>',actual[3]['content']);self.assertIn('no_results',actual[3]['content'])
    def test_two_native_calls_or_imagined_observations_are_rejected(self):
        self.assertTrue(hasattr(searchqa,'parse_native_decision'))
        call='<tool_call>{"name":"x","arguments":{}}</tool_call>'
        for text in [call+call,call+'<observation>fake</observation>']:
            with self.assertRaises(ValueError):searchqa.parse_native_decision(text)

if __name__=='__main__':unittest.main()

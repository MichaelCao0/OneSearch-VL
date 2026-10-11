import importlib.util,unittest
SID='src_'+'a'*24
class SearchQAMetricTests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_bench.searchqa_metrics'))
  return __import__('research.av_bench.searchqa_metrics',fromlist=['*'])
 def fixture(self):
  trace={'task':{'task_id':'qa-1','budget_recipe':{'max_tool_calls':6,'max_requested_av_seconds':12.}},'steps':[{'action':{'tool':'search_sources'},'observation':{'status':'ok','results':[{'source_id':SID}]}},{'action':{'tool':'read_text'},'observation':{'status':'ok','source_id':SID}}],'final':{'status':'ok','answer':'Otto Recke, 1903.','evidence':[],'cost':{'tool_calls':2,'requested_av_seconds':0.}}}
  ref={'task_id':'qa-1','status':'pending_human_review','benchmark_gold':False,'target_source_id':SID,'answer_fields':[{'field_id':'maker','accepted_values':['Otto Recke']},{'field_id':'year','accepted_values':['1903']}]}
  return trace,ref
 def test_pending_publisher_fields_yield_diagnostics_not_answer_accuracy(self):
  m=self.api();trace,ref=self.fixture();r=m.score_candidate_diagnostic(trace,ref)
  self.assertTrue(r['all_requested_fields_match']);self.assertIsNone(r['answer_accuracy']);self.assertFalse(r['semantic_verified']);self.assertTrue(r['target_retrieved']);self.assertTrue(r['target_read'])
 def test_missing_field_and_failed_model_final_do_not_score_as_success(self):
  m=self.api();trace,ref=self.fixture();trace['final']['answer']='Otto Recke'
  self.assertEqual(m.score_candidate_diagnostic(trace,ref)['requested_field_match_fraction'],.5)
  trace['final']['status']='controller_failed';self.assertFalse(m.score_candidate_diagnostic(trace,ref)['all_requested_fields_match'])
 def test_failed_media_preparation_and_successful_semantic_check_are_distinct(self):
  m=self.api();trace,ref=self.fixture()
  trace['steps']+=[{'action':{'tool':'inspect_av'},'observation':{'status':'media_ready_observer_unavailable'}},{'action':{'tool':'inspect_av'},'observation':{'status':'ok','model_executed':True}}]
  r=m.score_candidate_diagnostic(trace,ref);self.assertEqual(r['media_preparation_receipts'],1);self.assertEqual(r['semantic_observer_receipts'],1)
 def test_trace_reference_mismatch_rejected(self):
  m=self.api();trace,ref=self.fixture();ref['task_id']='other'
  with self.assertRaises(ValueError):m.score_candidate_diagnostic(trace,ref)

class FieldBoundaryTests(unittest.TestCase):
 def test_short_name_is_not_counted_inside_unrelated_word(self):
  from research.av_bench.searchqa_metrics import score_candidate_diagnostic
  trace,ref=SearchQAMetricTests().fixture();trace['final']['answer']='California, 19030'
  ref['answer_fields']=[{'field_id':'name','accepted_values':['Al']},{'field_id':'year','accepted_values':['1903']}]
  self.assertEqual(score_candidate_diagnostic(trace,ref)['requested_field_match_fraction'],0.)

class PublisherFamilyMetricTests(unittest.TestCase):
 def test_related_publisher_page_counts_as_target_discovery_and_read(self):
  from research.av_bench.searchqa_metrics import score_candidate_diagnostic
  trace,ref=SearchQAMetricTests().fixture();page='src_'+'b'*24
  ref['support_quotes']=[{'source_id':page,'field_id':'maker','quote':'Otto Recke'}]
  trace['steps'][0]['observation']['results']=[{'source_id':page}]
  trace['steps'][1]['observation']['source_id']=page
  result=score_candidate_diagnostic(trace,ref)
  self.assertTrue(result['target_retrieved']);self.assertTrue(result['target_read'])
 def test_unrelated_source_does_not_count_as_target_family(self):
  from research.av_bench.searchqa_metrics import score_candidate_diagnostic
  trace,ref=SearchQAMetricTests().fixture();page='src_'+'b'*24
  trace['steps'][0]['observation']['results']=[{'source_id':page}]
  trace['steps'][1]['observation']['source_id']=page
  result=score_candidate_diagnostic(trace,ref)
  self.assertFalse(result['target_retrieved']);self.assertFalse(result['target_read'])

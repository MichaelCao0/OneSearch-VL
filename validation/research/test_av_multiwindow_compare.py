import contextlib,io,json,tempfile,unittest
from pathlib import Path
from research.av_bench.multiwindow_compare import compare
class MultiwindowCompareTests(unittest.TestCase):
 def fixture(self,r):
  for name in ['base','answer_only','trajectory']:
   p=r/f'dev-{name}-v1';p.mkdir();rows=[]
   for i in range(16):rows.append({'task_id':f'q{i}','diagnostic':{'all_requested_fields_match':name=='trajectory','target_retrieved':True,'target_read':name!='base','tool_attempts_in_trace':5,'budget_compliant':True,'cost':{}}})
   (p/'summary.json').write_text(json.dumps({'mode':'evaluate','attempts':16,'results':rows}))
   (p/'input-manifest.private.json').write_text(json.dumps({'mode':'evaluate','tasks_sha256':'tasks','bindings_sha256':'bindings','observer_dir':'observer','sampling':{'do_sample':False},'model_tool_calls':6,'max_new_tokens':256,'initial_context_policy':'executed_av_observations_v1','checkpoint':'base','adapter':None if name=='base' else name}))
 def test_complete_paired_diagnostics_never_become_gold_accuracy(self):
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
   r=Path(d);self.fixture(r);out=compare(r,r/'comparison.json')
   self.assertEqual(len(out['paired_questions']),16);self.assertEqual(out['summary']['trajectory']['all_requested_field_matches'],16)
   self.assertIsNone(out['summary']['trajectory']['answer_accuracy']);self.assertFalse(out['formal_benchmark'])
 def test_missing_or_changed_task_ids_reject_paired_claim(self):
  for altered in ['missing','changed']:
   with tempfile.TemporaryDirectory() as d:
    r=Path(d);self.fixture(r);p=r/'dev-trajectory-v1/summary.json';s=json.loads(p.read_text())
    if altered=='missing':s['results'].pop()
    else:s['results'][0]['task_id']='other'
    p.write_text(json.dumps(s))
    with self.assertRaises(ValueError):compare(r,r/'comparison.json')
if __name__=='__main__':unittest.main()

class MatchedInputManifestTests(MultiwindowCompareTests):
 def fixture(self,r):
  super().fixture(r)
  for name in ['base','answer_only','trajectory']:
   (r/f'dev-{name}-v1/input-manifest.private.json').write_text(json.dumps({'mode':'evaluate','tasks_sha256':'tasks','bindings_sha256':'bindings','observer_dir':'observer','sampling':{'do_sample':False},'model_tool_calls':6,'max_new_tokens':256,'initial_context_policy':'executed_av_observations_v1','checkpoint':'base','adapter':None if name=='base' else name}))
 def test_changed_budget_or_observer_cannot_enter_matched_comparison(self):
  for key,value in [('model_tool_calls',5),('observer_dir','different'),('sampling',{'do_sample':True}),('initial_context_policy','old_prefix')]:
   with tempfile.TemporaryDirectory() as d:
    r=Path(d);self.fixture(r);p=r/'dev-trajectory-v1/input-manifest.private.json';m=json.loads(p.read_text());m[key]=value;p.write_text(json.dumps(m))
    with self.assertRaises(ValueError):compare(r,r/'comparison.json')

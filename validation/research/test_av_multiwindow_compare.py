import contextlib,io,json,tempfile,unittest
from pathlib import Path
from research.av_bench.multiwindow_compare import compare
class MultiwindowCompareTests(unittest.TestCase):
 def fixture(self,r):
  for name in ['base','answer_only','trajectory']:
   p=r/f'dev-{name}-v1';p.mkdir();rows=[]
   for i in range(16):rows.append({'task_id':f'q{i}','diagnostic':{'all_requested_fields_match':name=='trajectory','target_retrieved':True,'target_read':name!='base','tool_attempts_in_trace':5,'budget_compliant':True,'cost':{}}})
   (p/'summary.json').write_text(json.dumps({'mode':'evaluate','attempts':16,'results':rows}))
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

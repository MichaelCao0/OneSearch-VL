import copy,json,unittest,tempfile
from pathlib import Path
from research.av_train import teacher_silver_review as r
class SilverReviewTests(unittest.TestCase):
 def setUp(self):
  self.verdict={k:True for k in r.CHECKS};self.verdict['reason']='Actual AV and returned evidence support the answer.'
 def test_intact_fence_or_tags_preserves_verdict(self):
  for text in [json.dumps(self.verdict),'```json\n'+json.dumps(self.verdict)+'\n```','<response>'+json.dumps(self.verdict)+'</response>']:
   self.assertEqual(r.parse_verdict(text),self.verdict)
 def test_strings_and_missing_checks_cannot_be_retained(self):
  for change in ['string','missing','extra']:
   bad=copy.deepcopy(self.verdict)
   if change=='string':bad[r.CHECKS[0]]='true'
   elif change=='missing':bad.pop(r.CHECKS[0])
   else:bad['score']=1
   with self.assertRaises(ValueError):r.parse_verdict(json.dumps(bad))
 def test_producer_terminal_hash_must_match_before_paid_review(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'case_trajectory.json';path.write_text('{"case_id":"case"}')
   state={'status':'completed','plan_sha256':'a'*64,'trajectory_file':path.name,'trajectory_sha256':r.file_sha(path)}
   (path.parent/'candidate-state.json').write_text(json.dumps(state))
   self.assertEqual(r.load_completed_trajectory(path,'a'*64)[0]['case_id'],'case')
   path.write_text('{"case_id":"edited"}')
   with self.assertRaises(ValueError):r.load_completed_trajectory(path,'a'*64)
   state['trajectory_sha256']=r.file_sha(path);state['plan_sha256']='b'*64
   (path.parent/'candidate-state.json').write_text(json.dumps(state))
   with self.assertRaises(ValueError):r.load_completed_trajectory(path,'a'*64)
 def test_paid_attempt_reservation_is_create_only(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'case.json';first={'case_id':'case','api_calls':1,'status':'running'}
   self.assertTrue(r.reserve_attempt(p,first));self.assertFalse(r.reserve_attempt(p,{'case_id':'overwritten'}))
   self.assertEqual(json.loads(p.read_text()),first)
 def test_false_check_retains_actual_rejection(self):
  bad=copy.deepcopy(self.verdict);bad['media_matches_reference']=False
  self.assertFalse(all(r.parse_verdict(json.dumps(bad))[k] for k in r.CHECKS))
if __name__=='__main__':unittest.main()

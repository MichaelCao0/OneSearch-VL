import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research.av_train import recover_multiwindow_rows as m
from research.av_train.trajectory import digest
class MultiwindowRecoveryTests(unittest.TestCase):
 def run_recovery(self,root):
  with patch('sys.argv',['recover','--episodes-dir',str(root/'episodes'),'--batch-dir',str(root/'batch'),'--out-dir',str(root/'out')]),contextlib.redirect_stdout(io.StringIO()):m.main()
 def fixture(self,root,count=6):
  (root/'episodes').mkdir();(root/'batch').mkdir();refs=[]
  for i in range(count):
   tid=f'q{i}';refs.append({'task_id':tid,'target_source_id':f'src_{i%3}'})
   folder=root/'episodes'/f'attempt-{i:04d}';folder.mkdir();row={'conversations':[{'from':'human','value':tid},{'from':'gpt','value':'fixture'}]};key=digest(row)
   for name,value in [('row.private.json',row),('note.private.json',{'task_id':tid,'quality_label':'silver_anonymous_publisher_fields','authored_assistant_targets':0,'row_sha256':key}),('retention.json',{'retained':True,'row_sha256':key})]:(folder/name).write_text(json.dumps(value))
  (root/'batch/answer-candidates.private.jsonl').write_text('\n'.join(json.dumps(x) for x in refs))
 def test_interrupted_unfinished_episode_does_not_discard_completed_rows(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r);(r/'episodes/attempt-unfinished').mkdir();self.run_recovery(r)
   s=json.loads((r/'out/summary.json').read_text());self.assertEqual(s['completed_attempts'],6);self.assertEqual(s['retained_recordings'],3);self.assertTrue(s['bounded_training_ready'])
 def test_changed_completed_row_is_rejected_before_export(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r);(r/'episodes/attempt-0000/row.private.json').write_text('{}')
   with self.assertRaises(ValueError):self.run_recovery(r)
   self.assertFalse((r/'out').exists())
 def test_insufficient_rows_are_preserved_but_cannot_start_paired_training(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r,2)
   with self.assertRaises(SystemExit):self.run_recovery(r)
   s=json.loads((r/'out/summary.json').read_text());self.assertEqual(s['retained_rows'],2);self.assertFalse(s['bounded_training_ready'])
if __name__=='__main__':unittest.main()

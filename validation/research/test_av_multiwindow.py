import unittest
from research.av_train import multiwindow as m

class MultiWindowTests(unittest.TestCase):
 def test_three_windows_cover_head_middle_tail_of_long_input(self):
  self.assertEqual(m.window_schedule(12.),[{'start_s':0.,'end_s':2.},{'start_s':5.,'end_s':7.},{'start_s':10.,'end_s':12.}])
 def test_short_real_recording_has_bounded_nonduplicated_windows(self):
  w=m.window_schedule(5.445);self.assertEqual(len(w),3)
  self.assertTrue(all(0<=v['start_s']<v['end_s']<=5.445 and v['end_s']-v['start_s']<=2.00000001 for v in w))
  self.assertEqual(m.window_schedule(2.),[{'start_s':0.,'end_s':2.}])
 def test_invalid_duration_and_unbounded_count_are_rejected(self):
  for d in [0.,-1.,True,float('nan')]:
   with self.assertRaises(ValueError):m.window_schedule(d)
 def test_same_clip_across_questions_is_one_observer_source(self):
  a={'clue_id':'src_'+'a'*24,'source':{'source_id':'src_'+'c'*24,'asset_sha256':'c'*64,'asset_path':'/same','duration_s':12.}}
  b=dict(a,clue_id='src_'+'b'*24)
  result=m.group_clues([('train',[a,b])]);self.assertEqual(len(result),1);self.assertEqual(len(result[0]['clue_ids']),2)
 def test_identical_clip_bytes_cannot_enter_train_and_dev_observer_plan(self):
  a={'clue_id':'src_'+'a'*24,'source':{'source_id':'src_'+'c'*24,'asset_sha256':'c'*64,'asset_path':'/same','duration_s':12.}}
  with self.assertRaisesRegex(ValueError,'split'):m.group_clues([('train',[a]),('dev',[a])])
 def test_changed_source_binding_cannot_hide_behind_same_hash(self):
  a={'clue_id':'src_'+'a'*24,'source':{'source_id':'src_'+'c'*24,'asset_sha256':'c'*64,'asset_path':'/same','duration_s':12.}}
  b={'clue_id':'src_'+'b'*24,'source':dict(a['source'],duration_s=8.)}
  with self.assertRaises(ValueError):m.group_clues([('train',[a,b])])
if __name__=='__main__':unittest.main()

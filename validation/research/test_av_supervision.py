import unittest
from research.av_train import supervision as m
class SupervisionTests(unittest.TestCase):
 def fixture(self):
  return [-100,-100,10,11,-100,20,21],[{'assistant_token_start':2,'assistant_token_count':2},{'assistant_token_start':5,'assistant_token_count':2}]
 def test_trajectory_preserves_all_verified_assistant_labels(self):
  labels,pairs=self.fixture();self.assertEqual(m.apply_supervision(labels,pairs,'trajectory'),labels)
 def test_answer_only_masks_actions_and_preserves_final_and_input(self):
  labels,pairs=self.fixture();out=m.apply_supervision(labels,pairs,'answer_only')
  self.assertEqual(out,[-100,-100,-100,-100,-100,20,21]);self.assertEqual(labels[2],10)
 def test_invalid_ranges_cannot_mask_input_or_another_target(self):
  labels,pairs=self.fixture();pairs[0]['assistant_token_start']=0
  with self.assertRaises(ValueError):m.apply_supervision(labels,pairs,'answer_only')
 def test_unknown_mode_and_empty_final_rejected(self):
  labels,pairs=self.fixture()
  with self.assertRaises(ValueError):m.apply_supervision(labels,pairs,'other')
  with self.assertRaises(ValueError):m.apply_supervision(labels,[],'answer_only')
if __name__=='__main__':unittest.main()

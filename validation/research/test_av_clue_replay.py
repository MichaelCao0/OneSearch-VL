import importlib.util,unittest
class ClueReplayTests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_train.clue_replay'))
  return __import__('research.av_train.clue_replay',fromlist=['*'])
 def test_exact_source_intervals_rebased_without_changing_description(self):
  m=self.api();original={'events':[{'description':'Music playing','modality':'audio','source_interval':{'start_s':10.,'end_s':12.},'uncertainty':'uncertain'}],'uncertainty':'unreviewed','cannot_determine':[]}
  result=m.rebase_observation(original,10.,2.)
  self.assertEqual(result['events'][0]['source_interval'],{'start_s':0.,'end_s':2.})
  self.assertEqual(result['events'][0]['description'],'Music playing');self.assertEqual(original['events'][0]['source_interval']['start_s'],10.)
 def test_outside_crop_intervals_are_rejected_not_clamped(self):
  m=self.api();original={'events':[{'source_interval':{'start_s':9.,'end_s':12.}}]}
  with self.assertRaises(ValueError):m.rebase_observation(original,10.,2.)
 def test_exact_decoded_media_signature_required_for_replay(self):
  m=self.api()
  self.assertTrue(m.same_decoded_media({'video':'v','audio':'a'},{'video':'v','audio':'a'}))
  self.assertFalse(m.same_decoded_media({'video':'v','audio':'a'},{'video':'v','audio':'different'}))
  self.assertFalse(m.same_decoded_media({'video':'v'},{'video':'v'}))

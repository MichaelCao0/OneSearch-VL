import unittest
from research.av_train.grouped_content_audit import validate_group,usable_search
def row(i):
 return {'question_id':i,'content_category':'music_performance','observed_visual_cue':'musician','observed_audio_cue':'violin','media_matches_question':True,'answer_fields_supported_by_publisher':True,'search_requirement':'external_attribute','modality_role':'joint','question_leaks_identifying_clue':False,'specificity':'linked_instance','decision':'keep','reason':'concrete cue and external attribute'}
class GroupAuditTests(unittest.TestCase):
 def test_all_question_identities_preserved(self):
  x=validate_group({'verdicts':[row('b'),row('a')]},['a','b']);self.assertEqual(len(x),2);self.assertTrue(usable_search(x[0]['verdict']))
 def test_missing_question_cannot_silently_raise_quality(self):
  with self.assertRaises(ValueError):validate_group({'verdicts':[row('a')]},['a','b'])
 def test_duplicate_and_extra_fields_rejected(self):
  with self.assertRaises(ValueError):validate_group({'verdicts':[row('a'),row('a')]},['a','b'])
  x=row('a');x['arm']='publisher'
  with self.assertRaises(ValueError):validate_group({'verdicts':[x]},['a'])
 def test_perception_and_leaked_question_excluded_from_search_yield(self):
  x=row('a');x.pop('question_id');x['search_requirement']='perception_only';self.assertFalse(usable_search(x));x['search_requirement']='external_attribute';x['question_leaks_identifying_clue']=True;self.assertFalse(usable_search(x))

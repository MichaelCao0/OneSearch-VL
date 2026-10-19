import unittest,hashlib
from research.av_train.candidate_content_audit import validate_audit
from research.av_train.content_audit_report import main_search_disposition

class CandidateContentAudit(unittest.TestCase):
 def good(self):
  return {'content_category':'speech_lecture','observed_visual_cue':'A speaker at a lectern','observed_audio_cue':'Spoken English with applause','media_matches_question':True,'answer_fields_supported_by_publisher':True,'search_requirement':'external_attribute','modality_role':'joint','question_leaks_identifying_clue':False,'specificity':'linked_instance','decision':'keep','reason':'Venue needs external lookup'}
 def test_real_content_category_can_disagree_with_acquisition_query(self):
  verdict=validate_audit(self.good())
  self.assertEqual(verdict['content_category'],'speech_lecture')
 def test_unknown_category_cannot_be_silently_counted(self):
  value=self.good();value['content_category']='bird_0'
  with self.assertRaises(ValueError):validate_audit(value)
 def test_human_gold_or_new_fields_cannot_be_injected(self):
  value=self.good();value['human_gold']=True
  with self.assertRaises(ValueError):validate_audit(value)
 def test_empty_audio_description_is_not_audited(self):
  value=self.good();value['observed_audio_cue']=''
  with self.assertRaises(ValueError):validate_audit(value)
 def test_perception_only_can_be_control_not_search_evidence(self):
  value=self.good();value.update(search_requirement='perception_only',decision='control')
  self.assertEqual(validate_audit(value)['decision'],'control')
 def test_wrong_types_cannot_pass_boolean_flags(self):
  value=self.good();value['media_matches_question']='true'
  with self.assertRaises(ValueError):validate_audit(value)

class MainSearchDisposition(unittest.TestCase):
 def test_perception_only_not_kept_as_main_search_when_teacher_says_keep(self):
  self.assertEqual(main_search_disposition({'decision':'keep','search_requirement':'perception_only','media_matches_question':True,'answer_fields_supported_by_publisher':True,'specificity':'generic_class','question_leaks_identifying_clue':False}),'control')
 def test_unsupported_instance_not_kept(self):
  self.assertEqual(main_search_disposition({'decision':'keep','search_requirement':'external_attribute','media_matches_question':True,'answer_fields_supported_by_publisher':True,'specificity':'unsupported_link','question_leaks_identifying_clue':False}),'reject')
 def test_unsupported_answer_requires_revision(self):
  self.assertEqual(main_search_disposition({'decision':'keep','search_requirement':'external_attribute','media_matches_question':True,'answer_fields_supported_by_publisher':False,'specificity':'linked_instance','question_leaks_identifying_clue':False}),'revise')

if __name__=='__main__':unittest.main()

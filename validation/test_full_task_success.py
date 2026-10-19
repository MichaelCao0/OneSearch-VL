import unittest
from research.av_train.full_task_success import validate_rating
class FullTaskSuccessTests(unittest.TestCase):
 def rating(self):return {'ratings':[{'answer_id':'a','correct_field_ids':['f'],'contradicts_reference':False,'unsupported_assertion':False,'reason':'supported'}]}
 def test_valid_closed_rating(self):self.assertEqual(len(validate_rating(self.rating(),['a'],['f'])),1)
 def test_unknown_model_and_field_rejected(self):
  for key,value in [('answer_id','b'),('correct_field_ids',['x']),('correct_field_ids',['f','f'])]:
   r=self.rating();r['ratings'][0][key]=value
   with self.assertRaises(ValueError):validate_rating(r,['a'],['f'])
 def test_missing_and_duplicate_model_rejected(self):
  r=self.rating()
  with self.assertRaises(ValueError):validate_rating(r,['a','b'],['f'])
  r['ratings']*=2
  with self.assertRaises(ValueError):validate_rating(r,['a','b'],['f'])
 def test_nonboolean_verdict_rejected(self):
  r=self.rating();r['ratings'][0]['contradicts_reference']='false'
  with self.assertRaises(ValueError):validate_rating(r,['a'],['f'])

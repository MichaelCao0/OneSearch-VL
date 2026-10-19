import unittest
class TaskSuccessTests(unittest.TestCase):
 def test_correct_think_is_not_final_success(self):
  from research.av_train.task_success import outcome
  grade={'all_fields_correct':True,'contradicts_reference':False,'unsupported_assertion':False}
  self.assertFalse(outcome({'status':'ok','protocol_final':False},grade)['answer_success'])
 def test_answer_correctness_and_supported_success_are_separate(self):
  from research.av_train.task_success import outcome
  grade={'all_fields_correct':True,'contradicts_reference':False,'unsupported_assertion':True}
  row=outcome({'status':'ok','protocol_final':True},grade)
  self.assertTrue(row['answer_success']);self.assertFalse(row['supported_success'])
 def test_unknown_reviews_and_execution_errors_do_not_become_wrong_answers(self):
  from research.av_train.task_success import summarize_outcomes
  preds={'a':{'status':'ok','protocol_final':True},'b':{'status':'ok','protocol_final':True},'c':{'status':'error','protocol_final':False}}
  grades={'a':{'all_fields_correct':True,'contradicts_reference':False,'unsupported_assertion':False},'c':{'all_fields_correct':False,'contradicts_reference':False,'unsupported_assertion':False}}
  got=summarize_outcomes(preds,grades,['a','b','c'])
  self.assertEqual(got['assessed'],1);self.assertEqual(got['unknown'],2)
  self.assertEqual(got['answer_success_count'],1);self.assertEqual(got['supported_success_count'],1)
  self.assertEqual(got['answer_success_planned_bounds'],[1/3,1.])
if __name__=='__main__':unittest.main()

import copy,unittest
from research.av_train.pilot_freeze import select_questions

def record(i):
 return {'task_id':str(i),'recording':str(i),'family_id':str(i),'question':'question '+str(i),'input_media_sha256':'media'+str(i),'trajectory_row_sha256':'tr'+str(i),'answer_only_row_sha256':'an'+str(i)}

class PilotFreezeTests(unittest.TestCase):
 def test_nested_subset_and_reproducibility(self):
  records=[record(i) for i in range(12)]
  small,_=select_questions(records,{},[],4,7)
  large,_=select_questions(records,{},[r['task_id'] for r in small],9,7)
  self.assertEqual({r['task_id'] for r in small}&{r['task_id'] for r in large},{r['task_id'] for r in small})
  self.assertEqual(large,select_questions(list(reversed(records)),{},[r['task_id'] for r in small],9,7)[0])
 def test_known_reject_cannot_reenter_via_anchor(self):
  records=[record(i) for i in range(5)];q=copy.deepcopy(records[0]);q['quality_view']='exclude_candidate'
  selected,report=select_questions(records,{'0':q},['0','1'],3,7)
  self.assertNotIn('0',[r['task_id'] for r in selected]);self.assertEqual(report['excluded_previous_ids'],['0'])
 def test_tampered_quality_binding_is_error(self):
  records=[record(i) for i in range(3)];q=copy.deepcopy(records[0]);q.update(quality_view='main_search_candidate',input_media_sha256='wrong')
  with self.assertRaisesRegex(ValueError,'quality binding'):select_questions(records,{'0':q},[],2,7)
 def test_reworded_whitespace_does_not_inflate_pool(self):
  records=[record(i) for i in range(3)];dup=copy.deepcopy(records[0]);dup.update(task_id='dup',question=' question 0 ')
  with self.assertRaisesRegex(ValueError,'duplicate'):select_questions(records+[dup],{},[],3,7)
 def test_insufficient_pool_never_emits_short_freeze(self):
  with self.assertRaisesRegex(ValueError,'eligible'):select_questions([record(0)],{},[],2,7)
 def test_missing_pinned_anchor_cannot_silently_change_subset(self):
  with self.assertRaisesRegex(ValueError,'missing anchor'):select_questions([record(0),record(1)],{},['absent'],2,7)

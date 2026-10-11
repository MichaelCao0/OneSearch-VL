import copy,unittest
from test_av_anonymous_teacher import AnonymousTeacherTests
from research.av_train.trajectory import make_step,source_ids
from research.av_train import anonymous_teacher as a
from research.av_train import multiwindow_teacher as m
class MultiwindowTeacherTests(unittest.TestCase):
 def fixture(self):
  t=AnonymousTeacherTests().fixture();first=t['steps'][0]
  prefix=[]
  for start in [0.,5.,10.]:
   action=copy.deepcopy(first['action']);action['arguments'].update(start_s=start,end_s=start+2.)
   obs=copy.deepcopy(first['observation']);obs['original_interval']={'start_s':start,'end_s':start+2.}
   prefix.append(make_step(action,obs))
  t['steps']=prefix+t['steps'][1:];t['task']['budget_recipe']={'max_tool_calls':9,'max_requested_av_seconds':6.}
  t['teacher'].update(assistance=a.MULTI_ASSISTANCE,trajectory_origin=a.MULTI_ORIGIN)
  t['referenced_source_ids']=sorted(source_ids(t));return t
 def test_three_input_observations_are_context_and_never_assistant_targets(self):
  row,note=a.anonymous_silver_row(self.fixture(),{'src_aaaaaaaaaaaaaaaa':'train','src_bbbbbbbbbbbbbbbb':'train'},['Eugene','Oregon'],initial_observation_count=3)
  import json
  self.assertEqual(len(json.loads(row['conversations'][0]['value'])['initial_av_observations']),3)
  self.assertNotIn('inspect_av',''.join(x['value'] for x in row['conversations'] if x['from']=='gpt'))
  self.assertEqual(note['model_decided_assistant_targets'],3)
 def test_partial_or_duplicate_initial_observations_cannot_train(self):
  for duplicate in [True,False]:
   t=self.fixture()
   if duplicate:t['steps'][1]=copy.deepcopy(t['steps'][0])
   else:t['steps'][1]['observation']['model_executed']=False;t['steps'][1]=make_step(t['steps'][1]['action'],t['steps'][1]['observation'])
   with self.assertRaises(ValueError):a.anonymous_silver_row(t,{'src_aaaaaaaaaaaaaaaa':'train','src_bbbbbbbbbbbbbbbb':'train'},['Eugene'],initial_observation_count=3)
 def test_attempt_schedule_records_64_attempts_over_21_questions(self):
  tasks=[{'task_id':str(i)} for i in range(21)];s=m.attempt_schedule(tasks,64,100)
  self.assertEqual(len(s),64);self.assertEqual(len({r['task']['task_id'] for r in s}),21)
  self.assertEqual(len({r['seed'] for r in s}),64)
 def test_invalid_sampling_and_attempt_budgets_rejected(self):
  for n in [0,129,True]:
   with self.assertRaises(ValueError):m.attempt_schedule([{'task_id':'x'}],n,1)
  for v in [0.,float('nan'),True]:
   with self.assertRaises(ValueError):m.sampling_options(v,.9)
  self.assertEqual(m.sampling_options(.7,.9),{'do_sample':True,'temperature':.7,'top_p':.9})
if __name__=='__main__':unittest.main()

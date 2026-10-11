import json,unittest
from research.av_train.rollout import run_episode
from research.av_train.trajectory import source_ids
from research.av_train import anonymous_teacher as a
class Runtime:
 def __init__(self,fail=False):self.source_ids=['src_aaaaaaaaaaaaaaaa','src_bbbbbbbbbbbbbbbb'];self.cost={'tool_calls':0,'requested_av_seconds':0.};self.fail=fail
 def tool_definitions(self):return []
 def execute(self,action):
  self.cost['tool_calls']+=1;args=action['arguments'];sid=args.get('source_id')
  if action['tool']=='inspect_av':
   self.cost['requested_av_seconds']+=args['end_s']-args['start_s']
   return {'status':'observer_failed' if self.fail else 'ok','model_executed':not self.fail,'source_id':sid,'original_interval':{'start_s':args['start_s'],'end_s':args['end_s']},'observation':{'events':[]}}
  if action['tool']=='search_sources':return {'status':'ok','results':[{'source_id':'src_bbbbbbbbbbbbbbbb','excerpt':'Eugene, Oregon'}]}
  return {'status':'ok','source_id':sid,'quote':{'text':'Eugene, Oregon'}}
class ContextAlignmentTests(unittest.TestCase):
 def task(self):return {'schema_version':'av-researchbench.task.v1','task_id':'t','split':'train','corpus_id':'c','question':'Which city?','initial_sources':['src_aaaaaaaaaaaaaaaa'],'target_questions':[{'fact_id':'answer','question':'Which city?'}],'budget_recipe':{'max_tool_calls':5,'max_requested_av_seconds':6.}}
 def actions(self):return [{'tool':'inspect_av','arguments':{'source_id':'src_aaaaaaaaaaaaaaaa','start_s':s,'end_s':s+2.}} for s in [0.,5.,10.]]
 def teacher(self):return {'kind':'model','model':'test_fixture_not_real','revision':'fixture','assistance':a.MULTI_ASSISTANCE,'trajectory_origin':a.MULTI_ORIGIN}
 def test_generation_context_equals_export_and_actions_remain_in_private_trace(self):
  histories=[]
  def controller(history):
   histories.append(history)
   return [{'action':{'tool':'search_sources','arguments':{'query':'saxophone','top_k':3}}},{'action':{'tool':'read_text','arguments':{'source_id':'src_bbbbbbbbbbbbbbbb','start_char':0,'end_char':14}}},{'final':{'status':'ok','answer':'Eugene, Oregon','evidence':[]}}][len(histories)-1]
  t=run_episode(self.task(),Runtime(),controller,self.teacher(),system_prompt='fixture',recover_tool_errors=True,initial_actions=self.actions(),include_budget_prompt=False)
  row,note=a.anonymous_silver_row(t,dict.fromkeys(['src_aaaaaaaaaaaaaaaa','src_bbbbbbbbbbbbbbbb'],'train'),['Eugene','Oregon'],initial_observation_count=3)
  self.assertEqual(histories[0][1]['content'],row['conversations'][0]['value'])
  self.assertEqual(histories[0][0]['content'],row['system'])
  self.assertEqual(len(histories[0]),2);self.assertEqual(len(t['steps']),5)
  self.assertEqual(t['final']['cost']['tool_calls'],5)
  self.assertEqual(len(histories[-1]),6);self.assertEqual(note['authored_assistant_targets'],0)
 def test_failed_input_observation_never_reaches_controller(self):
  calls=[]
  t=run_episode(self.task(),Runtime(fail=True),lambda h:calls.append(h),self.teacher(),initial_actions=self.actions())
  self.assertEqual(calls,[]);self.assertEqual(t['final']['status'],'initial_observation_failed')
 def test_prefix_cannot_read_metadata_or_consume_more_than_total_budget(self):
  for actions in [[{'tool':'read_text','arguments':{'source_id':'src_bbbbbbbbbbbbbbbb','start_char':0,'end_char':10}}],self.actions()*2]:
   with self.assertRaises(ValueError):run_episode(self.task(),Runtime(),lambda h:None,self.teacher(),initial_actions=actions)
if __name__=='__main__':unittest.main()

import copy
import importlib.util
import unittest
from research.av_train.trajectory import make_step
from test_research_pipeline import task,A,B,C

def trace():
    t=task();t['split']='train'
    return {'schema_version':'av-research-trajectory.v1','task':t,
      'teacher':{'kind':'model','model':'TEST_MODEL_IDENTITY_ONLY','revision':'TEST_REVISION'},
      'system':'TEST FIXTURE ONLY','tools':[{'name':'search_sources'},{'name':'inspect_av'}],
      'steps':[make_step({'tool':'inspect_av','arguments':{'source_id':A,'start_s':0,'end_s':1}},
         {'status':'ok','source_id':A,'original_interval':{'start_s':0,'end_s':1},'model_executed':True,'semantic_status':'observed_unreviewed'}),
        make_step({'tool':'read_text','arguments':{'source_id':A,'start_char':0,'end_char':43}},
         {'status':'ok','source_id':A,'quote':{'text':'Street musicians perform in Eugene, Oregon','start_char':0,'end_char':43}})],
      'final':{'status':'ok','answer':'The recording is in Eugene, Oregon.','evidence':[]},
      'execution_kind':'live_tools','referenced_source_ids':[A]}

class SilverExportTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('research.av_train.silver'),'Missing explicit publisher-fact silver route')
        return __import__('research.av_train.silver',fromlist=['*'])
    def test_publisher_fact_silver_keeps_actual_actions_and_masks_observations_by_role(self):
        m=self.api();row,note=m.publisher_silver_row(trace(),{A:'train',B:'train',C:'train'},['Eugene','Oregon'])
        self.assertEqual(note['quality_label'],'silver_publisher_fact_match');self.assertFalse(note['semantic_verified'])
        self.assertEqual([x['from'] for x in row['conversations']],['human','gpt','observation','gpt','observation','gpt'])
        self.assertIn('Eugene',row['conversations'][-1]['value'])
        self.assertIn('ONE JSON object',row['system'])
    def test_matching_answer_without_actual_read_support_is_rejected(self):
        m=self.api();t=trace();t['steps']=t['steps'][:1]
        with self.assertRaisesRegex(ValueError,'observed publisher'):m.publisher_silver_row(t,{A:'train',B:'train',C:'train'},['Eugene','Oregon'])
    def test_wrong_answer_and_failed_episodes_do_not_train(self):
        m=self.api()
        for update in [{'answer':'New York'},{'status':'controller_failed'}]:
            t=trace();t['final'].update(update)
            with self.assertRaises(ValueError):m.publisher_silver_row(t,{A:'train',B:'train',C:'train'},['Eugene','Oregon'])
    def test_script_fixture_and_reserved_sources_are_rejected(self):
        m=self.api();t=trace();t['teacher']['kind']='fixture'
        with self.assertRaises(ValueError):m.publisher_silver_row(t,{A:'train',B:'train',C:'train'},['Eugene','Oregon'])
        with self.assertRaises(ValueError):m.publisher_silver_row(trace(),{A:'dev',B:'train',C:'train'},['Eugene','Oregon'])
if __name__=='__main__':unittest.main()

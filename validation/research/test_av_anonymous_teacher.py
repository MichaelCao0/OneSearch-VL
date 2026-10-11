import unittest
from research.av_train.trajectory import make_step,source_ids
from research.av_train import anonymous_teacher as m

class AnonymousTeacherTests(unittest.TestCase):
    def fixture(self):
        task={'task_id':'t','schema_version':'av-researchbench.task.v1','corpus_id':'c','split':'train',
          'question':'Which city?','initial_sources':['src_aaaaaaaaaaaaaaaa'],
          'target_questions':[{'fact_id':'answer','question':'Which city?'}],
          'budget_recipe':{'max_tool_calls':4,'max_requested_av_seconds':2.}}
        steps=[make_step({'tool':'inspect_av','arguments':{'source_id':'src_aaaaaaaaaaaaaaaa','start_s':0.,'end_s':2.}},
                    {'status':'ok','source_id':'src_aaaaaaaaaaaaaaaa','model_executed':True,'model_executed_this_episode':False,'observation':{'events':[]}}),
          make_step({'tool':'search_sources','arguments':{'query':'saxophone','top_k':3}},
                    {'status':'ok','results':[{'source_id':'src_bbbbbbbbbbbbbbbb','excerpt':'Eugene, Oregon'}]}),
          make_step({'tool':'read_text','arguments':{'source_id':'src_bbbbbbbbbbbbbbbb','start_char':0,'end_char':14}},
                    {'status':'ok','source_id':'src_bbbbbbbbbbbbbbbb','quote':{'text':'Eugene, Oregon'}})]
        teacher={'kind':'model','model':'test','revision':'r','assistance':m.ASSISTANCE,'trajectory_origin':m.ORIGIN}
        trace={'schema_version':'av-research-trajectory.v1','task':task,'teacher':teacher,'system':'test',
          'tools':[],'steps':steps,'final':{'status':'ok','answer':'Eugene, Oregon','cost':{}},'execution_kind':'live_tools','referenced_source_ids':[]}
        trace['referenced_source_ids']=sorted(source_ids(trace))
        return trace
    def test_initial_inspection_is_authored_but_search_is_model_decided(self):
        calls=[]
        def actual(history):calls.append(history);return {'action':{'tool':'search_sources','arguments':{'query':'music','top_k':3}}}
        actual.identity={'kind':'model'};actual.total_tokens=0
        proxy=m.InitialInspectionController(actual,'src_aaaaaaaaaaaaaaaa',2.)
        self.assertEqual(proxy([])['action']['tool'],'inspect_av');self.assertEqual(len(calls),0)
        self.assertEqual(proxy([{'role':'user','content':'observation'}])['action']['tool'],'search_sources')
        self.assertEqual(len(calls),1)
    def test_authored_action_never_enters_sft_assistant_targets(self):
        row,note=m.anonymous_silver_row(self.fixture(),{'src_aaaaaaaaaaaaaaaa':'train','src_bbbbbbbbbbbbbbbb':'train'},['Eugene','Oregon'])
        targets=[c['value'] for c in row['conversations'] if c['from']=='gpt']
        self.assertEqual(len(targets),3);self.assertNotIn('inspect_av',''.join(targets))
        self.assertIn('initial_av_observation',row['conversations'][0]['value'])
        self.assertFalse(note['fully_autonomous']);self.assertEqual(note['authored_assistant_targets'],0)
    def test_observed_search_excerpt_without_actual_read_cannot_train(self):
        t=self.fixture();t['steps']=t['steps'][:2];t['referenced_source_ids']=sorted(source_ids(t))
        with self.assertRaisesRegex(ValueError,'actual successful text read'):
            m.anonymous_silver_row(t,{'src_aaaaaaaaaaaaaaaa':'train','src_bbbbbbbbbbbbbbbb':'train'},['Eugene','Oregon'])
    def test_wrong_requested_field_and_foreign_split_are_rejected(self):
        with self.assertRaises(ValueError):m.anonymous_silver_row(self.fixture(),{'src_aaaaaaaaaaaaaaaa':'train','src_bbbbbbbbbbbbbbbb':'train'},['California'])
        with self.assertRaises(ValueError):m.anonymous_silver_row(self.fixture(),{'src_aaaaaaaaaaaaaaaa':'train','src_bbbbbbbbbbbbbbbb':'dev'},['Eugene'])
if __name__=='__main__':unittest.main()

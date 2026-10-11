import json
from pathlib import Path
import tempfile
import unittest
import test_av_searchqa
from test_av_searchqa import SID
from research.av_train.searchqa import SearchQARuntime,run_guided_curriculum

class GuidedCurriculumTests(unittest.TestCase):
    def test_authored_actions_and_actual_answer_are_distinguished(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=test_av_searchqa.SearchQATests().fixture(root)
            from research.av_train.searchqa import prepare_search_batch
            prepare_search_batch(pool,reserved,q,root/'batch')
            source=json.loads((pool/'sources.private.jsonl').read_text())
            class Observer:
                available_windows=[{'source_id':SID,'start_s':0,'end_s':1}]
                def __call__(self,s,a,b,o):return {'status':'ok','source_id':SID,'model_executed':False,'semantic_status':'contract_fixture'}
            class Controller:
                identity={'kind':'fixture','model':'TEST_ONLY','revision':'TEST'}
                def __call__(self,history):
                    self.history=history
                    return {'final':{'status':'fixture_completed','answer':'tractor'}}
            controller=Controller()
            task=json.loads((root/'batch/tasks-public.jsonl').read_text().strip())
            runtime=SearchQARuntime(root/'batch/corpus',json.loads((root/'batch/scope.json').read_text()),{SID:source},root/'tools',task['budget_recipe'],Observer())
            trace=run_guided_curriculum(task,runtime,controller)
            self.assertEqual([s['action']['tool'] for s in trace['steps']],['inspect_av','search_sources','read_text'])
            self.assertEqual(trace['teacher']['trajectory_origin'],'authored_tool_curriculum_with_actual_model_final')
            self.assertEqual(trace['teacher']['kind'],'fixture')
            self.assertIn('A vintage tractor',json.dumps(controller.history))
            self.assertEqual(trace['final']['answer'],'tractor')
if __name__=='__main__':unittest.main()

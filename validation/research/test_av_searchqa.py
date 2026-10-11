import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from research.av_search.corpus import build_corpus

SID='src_'+'a'*24

def api(test):
    test.assertIsNotNone(importlib.util.find_spec('research.av_train.searchqa'),'Missing direct search-QA route')
    return __import__('research.av_train.searchqa',fromlist=['*'])

class SearchQATests(unittest.TestCase):
    def fixture(self,root):
        pool=root/'pool';pool.mkdir();asset=root/'fixture.bin';asset.write_bytes(b'fixture-original')
        sha=hashlib.sha256(asset.read_bytes()).hexdigest()
        source={'source_id':SID,'asset_sha256':sha,'asset_path':str(asset),'origin_url':'https://example.org/source',
          'media_type':'audio_video','duration_s':3.,'public_metadata':{'title':'A vintage tractor'}}
        (pool/'sources.private.jsonl').write_text(json.dumps(source)+'\n')
        build_corpus([dict(source,text='A vintage tractor',text_kind='public_metadata')],pool/'authoring-corpus','qa-fixture','train')
        reserved=root/'reserved.jsonl';reserved.write_text('')
        questions=[{'task_id':'qa-001','question':'What kind of machine is shown?', 'initial_sources':[SID]}]
        return pool,reserved,questions

    def test_pending_gold_does_not_prevent_single_source_exploration(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root)
            summary=m.prepare_search_batch(pool,reserved,q,root/'batch')
            self.assertTrue(summary['rollout_allowed']);self.assertFalse(summary['benchmark_gold'])
            self.assertFalse(summary['formal_train_split_frozen'])
            task=json.loads((root/'batch/tasks-public.jsonl').read_text().strip())
            self.assertEqual(task['question'],'What kind of machine is shown?')
            self.assertEqual(len(task['target_questions']),1)
            self.assertFalse((root/'batch/references.private.jsonl').exists())
            from research.av_train.rollout import ToolRuntime
            sources={SID:json.loads((pool/'sources.private.jsonl').read_text())}
            runtime=ToolRuntime(root/'batch/corpus',json.loads((root/'batch/scope.json').read_text()),sources,root/'tools',task['budget_recipe'])
            receipt=runtime.execute({'tool':'search_sources','arguments':{'query':'tractor','top_k':3}})
            self.assertEqual(receipt['results'][0]['source_id'],SID)

    def test_reserved_dev_bytes_block_exploration_before_output(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root)
            reserved.write_text((pool/'sources.private.jsonl').read_text())
            with self.assertRaisesRegex(ValueError,'reserved'):m.prepare_search_batch(pool,reserved,q,root/'batch')
            self.assertFalse((root/'batch').exists())

    def test_private_answers_cannot_enter_actor_questions(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root);q[0]['gold_answer']='tractor'
            with self.assertRaises(ValueError):m.prepare_search_batch(pool,reserved,q,root/'batch')

    def test_changed_asset_bytes_are_not_admitted(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root);(root/'fixture.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):m.prepare_search_batch(pool,reserved,q,root/'batch')

    def test_json_decision_keeps_answer_and_rejects_truncated_or_extra_text(self):
        m=api(self)
        value=m.parse_decision('```json\n{"final":{"status":"ok","answer":"tractor","evidence":[]}}\n```')
        self.assertEqual(value['final']['answer'],'tractor')
        for s in ['{"action":', 'prefix {"final":{}}','{"final":{},"final":{}}']:
            with self.assertRaises(ValueError):m.parse_decision(s)

    def test_qa_profile_records_simple_answer_from_real_tool_search(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root);m.prepare_search_batch(pool,reserved,q,root/'batch')
            task=json.loads((root/'batch/tasks-public.jsonl').read_text().strip())
            from research.av_train.rollout import ToolRuntime,run_episode
            runtime=ToolRuntime(root/'batch/corpus',json.loads((root/'batch/scope.json').read_text()),
                {SID:json.loads((pool/'sources.private.jsonl').read_text())},root/'tools',task['budget_recipe'])
            decisions=iter([{'action':{'tool':'search_sources','arguments':{'query':'tractor','top_k':3}}},
                {'final':{'status':'ok','answer':'A tractor.','evidence':[]}}])
            trace=run_episode(task,runtime,lambda history:next(decisions),
                {'kind':'fixture','model':'script','revision':'v1'},system_prompt=m.QA_SYSTEM)
            self.assertEqual(trace['final']['answer'],'A tractor.')
            self.assertEqual(trace['steps'][0]['observation']['results'][0]['source_id'],SID)
            self.assertIn('answer',trace['system'])
            self.assertNotIn('private_answer',json.dumps(trace))

    def test_failed_inspection_is_returned_to_controller_for_recovery(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root);m.prepare_search_batch(pool,reserved,q,root/'batch')
            task=json.loads((root/'batch/tasks-public.jsonl').read_text().strip())
            from research.av_train.rollout import ToolRuntime,run_episode
            def unavailable(source,start,end,out):raise ValueError('No exact executed observer window')
            runtime=ToolRuntime(root/'batch/corpus',json.loads((root/'batch/scope.json').read_text()),
                {SID:json.loads((pool/'sources.private.jsonl').read_text())},root/'tools',task['budget_recipe'],unavailable)
            decisions=iter([{'action':{'tool':'inspect_av','arguments':{'source_id':SID,'start_s':0,'end_s':1}}},
                {'action':{'tool':'search_sources','arguments':{'query':'tractor','top_k':3}}},
                {'final':{'status':'ok','answer':'A tractor.','evidence':[]}}])
            histories=[]
            def controller(history):histories.append(history);return next(decisions)
            trace=run_episode(task,runtime,controller,{'kind':'fixture','model':'script','revision':'v1'},
                system_prompt=m.QA_SYSTEM,recover_tool_errors=True)
            self.assertEqual(trace['final']['answer'],'A tractor.')
            self.assertEqual(trace['steps'][0]['observation']['status'],'tool_error')
            self.assertEqual(trace['steps'][1]['observation']['results'][0]['source_id'],SID)
            self.assertIn('tool_error',histories[1][-1]['content'])

    def test_runtime_rebuild_preserves_source_text_and_corpus_identity(self):
        m=api(self);self.assertTrue(hasattr(m,'prepare_runtime_corpus'),'Missing runtime-specific index rebuild')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root);m.prepare_search_batch(pool,reserved,q,root/'batch')
            before=json.loads((root/'batch/corpus/manifest.json').read_text())
            manifest=m.prepare_runtime_corpus(root/'batch',root/'runtime-corpus')
            self.assertEqual(manifest['documents_sha256'],before['documents_sha256'])
            self.assertEqual(manifest['corpus_hash'],before['corpus_hash'])
            self.assertEqual(manifest['source_ids'],[SID])

    def test_read_error_exposes_valid_bounds_without_returning_unread_content(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root);m.prepare_search_batch(pool,reserved,q,root/'batch')
            runtime=m.SearchQARuntime(root/'batch/corpus',json.loads((root/'batch/scope.json').read_text()),
                {SID:json.loads((pool/'sources.private.jsonl').read_text())},root/'tools',{'max_tool_calls':6,'max_requested_av_seconds':8})
            result=runtime.execute({'tool':'read_text','arguments':{'source_id':SID,'start_char':0,'end_char':1000}})
            self.assertEqual(result['status'],'tool_error')
            self.assertEqual(result['valid_text_range'],{'start_char':0,'end_char':17})
            self.assertNotIn('quote',result)
            valid=runtime.execute({'tool':'read_text','arguments':{'source_id':SID,'start_char':0,'end_char':17}})
            self.assertEqual(valid['quote']['text'],'A vintage tractor')

    def test_runtime_rebuild_can_change_sqlite_recipe_without_changing_documents(self):
        m=api(self)
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch('research.av_search.corpus.sqlite3.sqlite_version','0.0.0'):
                pool,reserved,q=self.fixture(root);m.prepare_search_batch(pool,reserved,q,root/'batch')
            old=json.loads((root/'batch/corpus/manifest.json').read_text())
            new=m.prepare_runtime_corpus(root/'batch',root/'runtime-corpus')
            self.assertEqual(old['documents_sha256'],new['documents_sha256'])
            self.assertNotEqual(old['corpus_hash'],new['corpus_hash'])

    def test_last_controller_turn_requests_answer_after_bounded_tools(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool,reserved,q=self.fixture(root);m.prepare_search_batch(pool,reserved,q,root/'batch')
            task=json.loads((root/'batch/tasks-public.jsonl').read_text().strip());task['budget_recipe']['max_tool_calls']=1
            runtime=m.SearchQARuntime(root/'batch/corpus',json.loads((root/'batch/scope.json').read_text()),
                {SID:json.loads((pool/'sources.private.jsonl').read_text())},root/'tools',task['budget_recipe'])
            calls=[]
            def controller(history):
                calls.append(history)
                if len(calls)==1:return {'action':{'tool':'search_sources','arguments':{'query':'tractor','top_k':3}}}
                self.assertIn('No tool turns remain',history[-1]['content'])
                return {'final':{'status':'ok','answer':'A tractor.','evidence':[]}}
            from research.av_train.rollout import run_episode
            trace=run_episode(task,runtime,controller,{'kind':'fixture','model':'script','revision':'v1'},
                system_prompt=m.QA_SYSTEM,recover_tool_errors=True)
            self.assertEqual(trace['final']['answer'],'A tractor.')

if __name__=='__main__':unittest.main()

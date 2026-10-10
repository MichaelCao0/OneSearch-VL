import json
from pathlib import Path
import tempfile
import unittest
import test_semantic_observer as semantic_fixture
from research.av_train import rollout


def parser(test):
    test.assertTrue(hasattr(rollout,'make_parser'),'Missing explicit research observer CLI')
    return rollout.make_parser()

class ObserverCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        semantic_fixture.SemanticObserverTests.setUpClass();cls.source=semantic_fixture.SemanticObserverTests.source
    @classmethod
    def tearDownClass(cls):semantic_fixture.SemanticObserverTests.tearDownClass()
    def args(self,options=None):return parser(self).parse_args(['--batch-dir','batch','--out-dir','out']+(options or []))
    def test_unavailable_default_does_not_load_any_backend(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(rollout.build_observer(self.args(),Path(d)))
    def test_native_mode_requires_explicit_model_and_runtime_before_import(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.args(['--observer-mode','native'])
            with self.assertRaisesRegex(ValueError,'Explicit native observer'):
                rollout.build_observer(a,Path(d))
    def test_recorded_mode_requires_expected_identity_and_execution_records(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.args(['--observer-mode','recorded'])
            with self.assertRaisesRegex(ValueError,'Independent observer'):
                rollout.build_observer(a,Path(d))
    def test_unavailable_mode_cannot_ignore_explicit_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.args(['--observer-model-manifest','some-path'])
            with self.assertRaisesRegex(ValueError,'unavailable'):
                rollout.build_observer(a,Path(d))
    def test_recorded_cli_replays_independent_prepared_fixture(self):
        from research.av_train.native_observer import NativeObserverAdapter
        from research.av_agent.semantic_observer import SemanticObserver
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache'
            actual=NativeObserverAdapter(SemanticObserver(semantic_fixture.ContractBackend()))(self.source,.2,1.2,cache)
            identity=root/'identity.json';identity.write_text(json.dumps(actual['backend_identity']))
            a=self.args(['--observer-mode','recorded','--observer-execution-dir',str(cache),
                         '--observer-backend-identity',str(identity)])
            replay=rollout.build_observer(a,root)
            self.assertEqual(replay(self.source,.2,1.2,root/'view'),actual)
            self.assertEqual(actual['semantic_status'],'contract_fixture')
    def test_actor_receives_native_window_limit_before_action(self):
        from research.av_train.native_observer import NativeObserverAdapter
        from research.av_agent.semantic_observer import SemanticObserver
        from test_research_pipeline import task,A,B,C
        self.assertTrue(hasattr(rollout.ToolRuntime,'tool_definitions'),'Missing advertised observer constraints')
        runtime=object.__new__(rollout.ToolRuntime)
        runtime.observer=NativeObserverAdapter(SemanticObserver(semantic_fixture.ContractBackend()),max_clip_seconds=2)
        runtime.source_ids=[A,B,C];runtime.cost={'tool_calls':0,'requested_av_seconds':0}
        seen=[]
        def controller(history):
            seen.append(history)
            return {'final':{'status':'fixture_completed'}}
        trace=rollout.run_episode(task(),runtime,controller,{'kind':'fixture','model':'script','revision':'v1'})
        definition=next(t for t in trace['tools'] if t['name']=='inspect_av')
        self.assertEqual(definition['constraints']['max_window_seconds'],2)
        self.assertIn('max_window_seconds',seen[0][0]['content'])

if __name__=='__main__':unittest.main()

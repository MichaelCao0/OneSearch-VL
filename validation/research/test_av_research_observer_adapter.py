from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import test_semantic_observer as semantic_fixture
from research.av_agent.semantic_observer import SemanticObserver


def api(test):
    name='research.av_train.native_observer'
    test.assertIsNotNone(importlib.util.find_spec(name),'Missing native AV episode adapter')
    return __import__(name,fromlist=['*'])

class NativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        semantic_fixture.SemanticObserverTests.setUpClass()
        cls.source=semantic_fixture.SemanticObserverTests.source;cls.derived=semantic_fixture.SemanticObserverTests.derived
    @classmethod
    def tearDownClass(cls):semantic_fixture.SemanticObserverTests.tearDownClass()
    def record(self,kind='contract_test'):
        from research.av_agent.semantic_observer import make_observer_input
        backend=semantic_fixture.ContractBackend();backend.manifest['execution_kind']=kind
        return SemanticObserver(backend).observe(make_observer_input(self.source,self.derived))
    def test_contract_fixture_cannot_be_semantic_training_observation(self):
        m=api(self);receipt=m.observation_receipt(self.record(),{'output_bytes':3},{})
        self.assertEqual(receipt['semantic_status'],'contract_fixture')
        self.assertNotEqual(receipt['status'],'ok')
    def test_model_record_projection_keeps_original_time_and_model_identity(self):
        m=api(self);record=self.record('model_inference');receipt=m.observation_receipt(record,{'output_bytes':3},{})
        self.assertEqual(receipt['original_interval'],{'start_s':.2,'end_s':1.2})
        self.assertEqual(receipt['status'],'ok')
        self.assertEqual(receipt['semantic_status'],'observed_unreviewed')
        self.assertEqual(receipt['backend_identity']['model_revision'],'f'*40)
        self.assertFalse(receipt['semantic_verified'])
        # This is an explicitly declared-model contract fixture, not real generation.
    def test_public_observation_contains_no_paths_or_record_nonces(self):
        m=api(self);receipt=m.observation_receipt(self.record('model_inference'),{}, {})
        for key in ['clip_path','asset_path','observation_id','created_at','record_sha256','human_validation']:
            self.assertNotIn(key,json.dumps(receipt))
    def test_tampered_model_record_rejected(self):
        m=api(self);record=self.record('model_inference');record['raw_output']='invented'
        with self.assertRaisesRegex(ValueError,'record'):m.observation_receipt(record,{}, {})
    def test_original_structured_events_preserved(self):
        m=api(self);record=self.record('model_inference')
        receipt=m.observation_receipt(record,{}, {})
        self.assertEqual(receipt['observation'],record['structured_observation'])
    def test_actual_adapter_executes_existing_clock_preparation(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            native=m.NativeObserverAdapter(SemanticObserver(semantic_fixture.ContractBackend()),max_clip_seconds=2)
            receipt=native(self.source,.2,1.2,Path(d)/'observed')
            self.assertEqual(receipt['semantic_status'],'contract_fixture')
            self.assertTrue((Path(d)/'observed/observation.private.json').is_file())
            self.assertGreater(receipt['actual_cost']['prepared_video_frames'],0)
    def test_cached_replay_uses_independent_native_record_not_trace(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'observed'
            adapter=m.NativeObserverAdapter(SemanticObserver(semantic_fixture.ContractBackend()),max_clip_seconds=2)
            actual=adapter(self.source,.2,1.2,root)
            replay=m.RecordedObserverReplay([root],expected_backend=actual['backend_identity'])
            cached=replay(self.source,.2,1.2,Path(d)/'cache-view')
            self.assertEqual(cached,actual)
            with self.assertRaisesRegex(ValueError,'cache'):replay(self.source,.3,1.3,Path(d)/'wrong-view')
    def test_cache_rejects_changed_native_identity(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'observed';adapter=m.NativeObserverAdapter(SemanticObserver(semantic_fixture.ContractBackend()))
            receipt=adapter(self.source,.2,1.2,root)
            identity=deepcopy(receipt['backend_identity']);identity['model_revision']='e'*40
            with self.assertRaisesRegex(ValueError,'identity'):m.RecordedObserverReplay([root],expected_backend=identity)

if __name__=='__main__':unittest.main()

class NativeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):semantic_fixture.SemanticObserverTests.setUpClass();cls.source=semantic_fixture.SemanticObserverTests.source
    @classmethod
    def tearDownClass(cls):semantic_fixture.SemanticObserverTests.tearDownClass()
    def runtime(self,d):
        from research.av_search.corpus import build_corpus
        from research.av_train.rollout import ToolRuntime
        from research.av_train.native_observer import NativeObserverAdapter
        root=Path(d);s=self.source
        doc={k:s[k] for k in ('source_id','origin_url','asset_sha256','asset_path','media_type','duration_s','public_metadata')}
        doc.update(text='Synthetic contract fixture',text_kind='public_metadata')
        build_corpus([doc],root/'corpus','native-contract-dev','dev')
        scope={'corpus_id':'native-contract-dev','split':'dev','source_ids':[s['source_id']]}
        adapter=NativeObserverAdapter(SemanticObserver(semantic_fixture.ContractBackend()))
        return ToolRuntime(root/'corpus',scope,{s['source_id']:s},root/'tools',{'max_tool_calls':3,'max_requested_av_seconds':4},observer=adapter)
    def test_native_receipt_cost_is_charged_to_episode(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=self.runtime(d)
            receipt=runtime.execute({'tool':'inspect_av','arguments':{'source_id':self.source['source_id'],'start_s':.2,'end_s':1.2}})
            self.assertGreater(runtime.cost.get('prepared_video_frames',0),0)
            self.assertEqual(runtime.cost.get('observer_backend_attempts'),1)
            self.assertEqual(runtime.cost['requested_av_seconds'],1)
    def test_search_exposes_media_duration_for_actor_window_choice(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=self.runtime(d)
            receipt=runtime.execute({'tool':'search_sources','arguments':{'query':'fixture','top_k':1}})
            self.assertEqual(receipt['results'][0].get('media_type'),'audio_video')
            self.assertEqual(receipt['results'][0].get('duration_s'),self.source['duration_s'])

    def test_missing_native_tokens_remain_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=self.runtime(d)
            runtime.execute({'tool':'inspect_av','arguments':{'source_id':self.source['source_id'],'start_s':.2,'end_s':1.2}})
            self.assertNotIn('input_tokens',runtime.cost)
            self.assertNotIn('generated_tokens',runtime.cost)
            self.assertEqual(runtime.cost.get('input_tokens_unknown_observations'),1)
            self.assertEqual(runtime.cost.get('generated_tokens_unknown_observations'),1)

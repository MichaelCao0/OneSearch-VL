"""Fail-closed observer boundaries; fixtures never claim model inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from research.av_data.ingest import ingest_media
from research.av_data.media_clock import clip_media
try:
    from research.av_agent import semantic_observer as observer
except ImportError:
    observer = None

class ContractBackend:
    """No-model fixture at the external inference boundary."""
    def __init__(self, mode='native_joint_av', mismatch=None, failure=None):
        self.manifest={'backend_id':'contract-fixture','model_id':'unit-test-no-model',
          'model_revision':'f'*40,'config_sha256':'d'*64,'mode':mode,'transport':'local',
          'execution_kind':'contract_test','runtime_device':'cpu','backend_code_revision':'test-fixture-v1'}
        self.mismatch=mismatch;self.failure=failure
    def infer(self,bound_input):
        if self.failure:raise self.failure
        result={'backend_identity':deepcopy(self.manifest),'consumed_input_sha256':bound_input['input_sha256'],
          'raw_output':'{"events": [], "uncertainty": "Fixture only; no model ran."}\n',
          'structured_observation':{'events':[],'uncertainty':'Fixture only; no model ran.','cannot_determine':['sound source']},
          'provider_requests':0}
        if self.mismatch=='identity':result['backend_identity']['model_revision']='e'*40
        if self.mismatch=='input':result['consumed_input_sha256']='c'*64
        if self.mismatch=='nan':result['structured_observation']['events']=[{'source_interval':{'start_s':float('nan'),'end_s':1}}]
        return result

class SemanticObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);media=cls.root/'input.mkv'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=blue:s=32x32:r=5:d=2',
          '-f','lavfi','-i','sine=frequency=300:sample_rate=8000:duration=2','-c:v','ffv1','-c:a','pcm_s16le',str(media)],check=True)
        cls.source=ingest_media(media,'https://example.org/private-fixture',cls.root/'assets',{'title':'Synthetic contract fixture'})
        cls.derived=clip_media(cls.source,.2,1.2,cls.root/'clips')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):self.assertIsNotNone(observer,'semantic observer contract is missing')
    def bound_input(self):
        return observer.make_observer_input(self.source,self.derived,question='Report observed AV events and uncertainty.',probe_id='probe-02')
    def test_observation_ids_are_unique_across_runtime_instances(self):
        bound=self.bound_input()
        first=observer.SemanticObserver().observe(bound)
        second=observer.SemanticObserver().observe(bound)
        self.assertNotEqual(first['observation_id'],second['observation_id'])

    def test_invalid_unicode_output_fails_and_preserves_provider_denominator(self):
        class InvalidUnicodeBackend(ContractBackend):
            def __init__(self,field):super().__init__();self.field=field
            def infer(self,bound):
                result=super().infer(bound);result['provider_requests']=2
                bad=chr(0xd800)
                if self.field=='raw':result['raw_output']=bad
                else:result['structured_observation']['uncertainty']=bad
                return result
        for field in ('raw','structured'):
            with self.subTest(field=field):
                runtime=observer.SemanticObserver(InvalidUnicodeBackend(field))
                record=runtime.observe(self.bound_input())
                self.assertEqual(record['status'],'failed')
                self.assertEqual(runtime.counts['calls'],1)
                self.assertEqual(runtime.counts['failures'],1)
                self.assertEqual(runtime.counts['provider_requests_known'],2)
                self.assertEqual(record['actual_cost']['provider_requests'],2)
                self.assertEqual(observer.validate_observation(record),[])

    def test_unavailable_backend_retains_input_identity_and_no_semantic_claim(self):
        record=observer.SemanticObserver().observe(self.bound_input())
        self.assertEqual(record['status'],'unavailable');self.assertEqual(record['error']['code'],'backend_unavailable')
        self.assertFalse(record['model_executed']);self.assertFalse(record['raw_av_observer_ready'])
        self.assertEqual(record['human_validation']['status'],'pending');self.assertIsNone(record['raw_output'])
        self.assertEqual(record['input']['original_interval'],{'start_s':.2,'end_s':1.2})
        self.assertEqual(observer.validate_observation(record),[])
    def test_input_rejects_wrong_source_hash_and_changed_clip_bytes(self):
        wrong=deepcopy(self.derived);wrong['parent_sha256']='f'*64
        with self.assertRaises(ValueError):observer.make_observer_input(self.source,wrong)
        corrupted=self.root/'corrupt.mkv';corrupted.write_bytes(b'not actual clip')
        wrong=deepcopy(self.derived);wrong['output_path']=str(corrupted)
        with self.assertRaises(ValueError):observer.make_observer_input(self.source,wrong)
    def test_shared_source_clock_and_mapping_tamper(self):
        bound=self.bound_input();self.assertEqual(bound['mapping']['output_to_source'],{'scale':1.0,'offset_s':.2})
        bad=deepcopy(self.derived);bad['mapping']['output_to_source']['offset_s']=0
        with self.assertRaises(ValueError):observer.make_observer_input(self.source,bad)
        bad=deepcopy(bound);bad['question']='changed after binding'
        self.assertTrue(observer.validate_observer_input(bad))
    def test_minimal_input_excludes_gold_and_metadata(self):
        bound=self.bound_input();rendered=json.dumps(bound)
        for key in ('public_metadata','origin_url','rights_basis'):self.assertNotIn(key,rendered)
        bad=deepcopy(bound);bad['gold_answer']='injected';self.assertTrue(observer.validate_observer_input(bad))
    def test_contract_execution_retains_exact_raw_without_claiming_model_ran(self):
        record=observer.SemanticObserver(ContractBackend()).observe(self.bound_input())
        self.assertEqual(record['status'],'contract_test_executed');self.assertFalse(record['model_executed'])
        expected='{"events": [], "uncertainty": "Fixture only; no model ran."}\n'
        self.assertEqual(record['raw_output'],expected)
        self.assertEqual(record['raw_output_sha256'],hashlib.sha256(expected.encode()).hexdigest())
        self.assertEqual(record['human_validation']['status'],'pending');self.assertFalse(record['raw_av_observer_ready'])
        self.assertEqual(observer.validate_observation(record),[])
    def test_backend_identity_and_consumed_hash_mismatch_fail_in_denominator(self):
        for mismatch in ('identity','input'):
            with self.subTest(mismatch=mismatch):
                runtime=observer.SemanticObserver(ContractBackend(mismatch=mismatch));record=runtime.observe(self.bound_input())
                self.assertEqual(record['status'],'failed');self.assertEqual(record['error']['code'],'backend_contract_invalid')
                self.assertEqual(runtime.counts['calls'],1);self.assertEqual(runtime.counts['failures'],1)
                self.assertFalse(record['raw_av_observer_ready'])
    def test_exception_keeps_safe_type_not_secret_provider_text(self):
        runtime=observer.SemanticObserver(ContractBackend(failure=RuntimeError('SECRET_FROM_PROVIDER')))
        record=runtime.observe(self.bound_input());self.assertEqual(record['status'],'failed')
        self.assertEqual(record['error'],{'code':'backend_failed','type':'RuntimeError'})
        self.assertNotIn('SECRET_FROM_PROVIDER',json.dumps(record));self.assertEqual(runtime.counts['failures'],1)
    def test_budget_block_stays_in_denominator_without_backend_invocation(self):
        runtime=observer.SemanticObserver(ContractBackend(),max_calls=1);runtime.observe(self.bound_input())
        blocked=runtime.observe(self.bound_input());self.assertEqual(blocked['status'],'budget_exceeded')
        self.assertEqual(runtime.counts['calls'],2);self.assertEqual(runtime.counts['backend_attempts'],1)
        self.assertEqual(runtime.counts['failures'],1)
    def test_record_tamper_and_fake_human_pass_are_rejected(self):
        record=observer.SemanticObserver(ContractBackend()).observe(self.bound_input())
        for mutation in ({'raw_output':'tampered'},{'model_executed':True},{'human_validation':{'status':'passed'}},{'raw_av_observer_ready':True}):
            with self.subTest(mutation=mutation):
                bad=deepcopy(record);bad.update(mutation);self.assertTrue(observer.validate_observation(bad))
    def test_declared_model_execution_keeps_human_review_pending(self):
        backend=ContractBackend();backend.manifest['execution_kind']='model_inference'
        # A boundary fixture checks policy here; it is not actual model evidence.
        record=observer.SemanticObserver(backend).observe(self.bound_input())
        self.assertEqual(record['status'],'model_executed');self.assertTrue(record['model_executed'])
        self.assertEqual(record['human_validation']['status'],'pending');self.assertFalse(record['raw_av_observer_ready'])
        self.assertEqual(observer.validate_observation(record),[])
    def test_explicit_asr_frames_mode_and_nonfinite_output_rejection(self):
        record=observer.SemanticObserver(ContractBackend(mode='asr_plus_frames')).observe(self.bound_input())
        self.assertEqual(record['backend_identity']['mode'],'asr_plus_frames');self.assertFalse(record['raw_av_observer_ready'])
        malformed=observer.SemanticObserver(ContractBackend(mismatch='nan')).observe(self.bound_input())
        self.assertEqual(malformed['status'],'failed')
    def test_fixed_probe_pack_keeps_input_and_human_reference_pending(self):
        pack=observer.semantic_probe_templates();self.assertEqual(len(pack),8)
        self.assertEqual(len({item['probe_id'] for item in pack}),8)
        for item in pack:
            self.assertEqual(item['status'],'input_and_human_reference_pending');self.assertIsNone(item['input_binding'])
            self.assertEqual(item['human_reference']['status'],'pending');self.assertNotEqual(item['acceptance_criterion'],'')
            self.assertIn('natural',item['report_separately_by']);self.assertIn('synthetic_timing',item['report_separately_by'])
    def test_preflight_cli_is_non_network_and_preserves_old_run(self):
        destination=self.root/'preflight'
        command=[sys.executable,'-m','research.av_agent.semantic_observer','--preflight','--output-dir',str(destination)]
        result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
        status=json.loads((destination/'preflight.json').read_text());self.assertFalse(status['readiness']['raw_av_observer_ready'])
        self.assertEqual(status['semantic_probe_denominator'],8);self.assertEqual(status['actual_provider_requests'],0)
        self.assertEqual(status['model_executed_probe_count'],0)
        repeat=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(repeat.returncode,0)

    def test_visual_only_backend_cannot_emit_audible_event_as_observation(self):
        class VisualBackend(ContractBackend):
            def infer(self,bound):
                result=super().infer(bound)
                result['structured_observation']['events']=[{'modality':'audio','description':'A tone.',
                  'source_interval':{'start_s':.2,'end_s':.4},'uncertainty':'unknown sound source'}]
                return result
        record=observer.SemanticObserver(VisualBackend(mode='visual_only')).observe(self.bound_input())
        self.assertEqual(record['status'],'failed')
        self.assertEqual(record['error']['code'],'backend_contract_invalid')
    def test_malformed_nested_mapping_returns_errors_instead_of_crashing(self):
        bound=self.bound_input();bound['mapping']['streams']=[{'codec_type':None},{'codec_type':'audio'}]
        self.assertTrue(observer.validate_observer_input(bound))
    def test_service_failure_preserves_unknown_provider_work(self):
        backend=ContractBackend(failure=TimeoutError('private endpoint details'));backend.manifest['transport']='service'
        runtime=observer.SemanticObserver(backend);record=runtime.observe(self.bound_input())
        self.assertIsNone(record['actual_cost']['provider_requests'])
        self.assertEqual(runtime.counts['provider_requests_unknown_attempts'],1)
        self.assertEqual(runtime.counts['backend_attempts'],1)
    def test_unpinned_backend_revision_is_rejected_before_any_execution(self):
        backend=ContractBackend();backend.manifest['model_revision']='main'
        with self.assertRaises(ValueError):observer.SemanticObserver(backend)
    def test_backend_event_outside_requested_interval_fails_without_support(self):
        class OutsideBackend(ContractBackend):
            def infer(self,bound):
                result=super().infer(bound)
                result['structured_observation']['events']=[{'modality':'audio','description':'A tone.',
                  'source_interval':{'start_s':1.5,'end_s':1.8},'uncertainty':'unknown sound source'}]
                return result
        record=observer.SemanticObserver(OutsideBackend()).observe(self.bound_input())
        self.assertEqual(record['status'],'failed');self.assertIsNone(record['structured_observation'])
        self.assertIsNotNone(record['raw_output'])

if __name__=='__main__':unittest.main()

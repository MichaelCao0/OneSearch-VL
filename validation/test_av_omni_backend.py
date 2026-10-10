"""Mocked native-AV boundary tests; none count as model capability evidence."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from research.av_agent.semantic_observer import SemanticObserver, _digest
try:
    from research.av_agent.qwen_omni_backend import QwenOmniBackend, parse_clip_observation
except ImportError:
    QwenOmniBackend = None
    parse_clip_observation = None


class Batch(dict):
    def to(self, *args, **kwargs):
        return self


class Processor:
    def __init__(self, raw):
        self.raw = raw
        self.messages = None
        self.kwargs = None
        self.decoded = None
    def apply_chat_template(self, messages, **kwargs):
        self.messages = deepcopy(messages)
        return 'rendered-private-clip-prompt'
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return Batch(input_ids=torch.tensor([[10, 11, 12]]), attention_mask=torch.ones((1, 3)))
    def batch_decode(self, ids, **kwargs):
        self.decoded = ids.tolist()
        return [self.raw]


class Model:
    device = torch.device('cpu')
    dtype = torch.bfloat16
    def __init__(self):
        self.kwargs = None
        self.failure = False
    def eval(self):
        return self
    def generate(self, **kwargs):
        self.kwargs = kwargs
        if self.failure:
            raise RuntimeError('PRIVATE_PROVIDER_OR_PATH_TEXT')
        return torch.tensor([[10, 11, 12, 21, 22]])


class OmniBackendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(QwenOmniBackend, 'native AV backend is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.snapshot = self.directory / 'snapshot'
        self.snapshot.mkdir()
        self.config = b'{"model_type":"qwen2_5_omni"}\n'
        files = {'config.json': self.config, 'model-00001-of-00001.safetensors': b'NO_MODEL_TEST_FIXTURE'}
        records = []
        for name, data in files.items():
            (self.snapshot / name).write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            records.append({'path': name, 'size_bytes': len(data), 'expected_sha256': digest,
                            'actual_sha256': digest, 'verified': True})
        self.artifact = {'schema_version':'av-omni-model-manifest-v1','repository_id': 'Qwen/Qwen2.5-Omni-7B', 'revision': 'a'*40,
                         'snapshot_path': str(self.snapshot), 'status': 'verified', 'files': records,
                         'all_lfs_verified': True, 'total_size_bytes': sum(x['size_bytes'] for x in records),
                         'budget_bytes': 30_000_000_000}
        self.manifest_path = self.directory / 'model-manifest.json'
        self.manifest_path.write_text(json.dumps(self.artifact))
        self.clip = self.directory / 'opaque-clip.mkv'
        self.clip.write_bytes(b'contract media bytes, no actual video')
        media_hash = hashlib.sha256(self.clip.read_bytes()).hexdigest()
        self.bound = {'schema_version':'av-observer-input-0.1','source_id':'src_'+('b'*64)[:24],
          'asset_sha256':'b'*64,'derived_id':'derived-'+media_hash[:24],'output_sha256':media_hash,
          'output_bytes':self.clip.stat().st_size,'clip_path':str(self.clip),
          'original_interval':{'start_s':15.0,'end_s':17.0},
          'mapping':{'axis':'source_local_presentation','interval':'half_open',
                     'output_axis':'presentation_pts_seconds','output_to_source':{'scale':1.0,'offset_s':15.0},
                     'source_to_output':{'scale':1.0,'offset_s':-15.0},
                     'streams':[{'codec_type':'audio','output_start_time_s':0.0},{'codec_type':'video','output_start_time_s':0.0}]},
          'modalities':['audio','video'],'input_mode':'raw_av','question':'Describe directly heard and seen events.',
          'probe_id':None}
        self.bound['input_sha256'] = _digest(self.bound)
        self.raw = json.dumps({'events':[{'modality':'joint_av','description':'Fixture event.',
                    'clip_interval':{'start_s':.25,'end_s':1.5},'uncertainty':'Fixture only.'}],
                    'uncertainty':'Fixture only; no real model executed.','cannot_determine':['actual event']})
        self.processor = Processor(self.raw)
        self.model = Model()
        self.calls = []
        self.sample_fps = 2.0
        self.video_kwargs_override = None
        def process(messages, **kwargs):
            self.calls.append((deepcopy(messages), kwargs))
            video_kwargs = self.video_kwargs_override if self.video_kwargs_override is not None else {'fps':[self.sample_fps],'do_sample_frames':False}
            if kwargs.get('return_video_kwargs'):
                return ['RAW_AUDIO'], None, ['RAW_VIDEO'], video_kwargs
            return ['RAW_AUDIO'], None, ['RAW_VIDEO']
        transformer_module = SimpleNamespace(
            Qwen2_5OmniProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw:self.processor),
            Qwen2_5OmniForConditionalGeneration=SimpleNamespace(from_pretrained=self.load_model))
        self.module_patch = patch.dict(sys.modules, {'transformers':transformer_module,
                                      'qwen_omni_utils':SimpleNamespace(process_mm_info=process)})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.load_kwargs = None
    def load_model(self, *args, **kwargs):
        self.load_kwargs = kwargs
        return self.model
    def backend(self):
        return QwenOmniBackend(self.manifest_path, backend_code_revision='c'*40, device='cpu')
    def test_delayed_audio_fails_before_helper_and_next_attempt_is_recorded(self):
        bound=deepcopy(self.bound)
        bound['mapping']['streams'][0]['output_start_time_s']=1.012
        bound.pop('input_sha256')
        bound['input_sha256']=_digest(bound)
        backend=self.backend()
        observer=SemanticObserver(backend,max_calls=2)
        failed=observer.observe(bound)
        self.assertEqual(failed['status'],'failed')
        self.assertEqual(failed['error']['type'],'ClockAlignmentUnsupportedError')
        self.assertEqual(self.calls,[])
        self.assertFalse(backend.last_metrics['generation_completed'])
        self.assertEqual(backend.last_metrics['error_type'],'ClockAlignmentUnsupportedError')
        success=observer.observe(self.bound)
        self.assertEqual(success['status'],'model_executed')
        self.assertEqual(backend.last_metrics['attempt'],2)
        self.assertTrue(backend.last_metrics['generation_completed'])

    def test_shared_raw_av_kwargs_and_generation_token_slice(self):
        backend = self.backend()
        reply = backend.infer(self.bound)
        self.assertTrue(self.calls[0][1]['use_audio_in_video'])
        self.assertTrue(self.calls[0][1].get('return_video_kwargs'))
        self.assertEqual(self.processor.kwargs.get('fps'),2.0)
        self.assertIs(self.processor.kwargs.get('do_sample_frames'),False)
        self.assertTrue(self.processor.kwargs['use_audio_in_video'])
        self.assertEqual(self.processor.kwargs['audio'], ['RAW_AUDIO'])
        self.assertEqual(self.processor.kwargs['videos'], ['RAW_VIDEO'])
        self.assertTrue(self.model.kwargs['use_audio_in_video'])
        self.assertFalse(self.model.kwargs['return_audio'])
        self.assertEqual(self.model.kwargs['thinker_max_new_tokens'], 256)
        self.assertNotIn('max_new_tokens', self.model.kwargs)
        self.assertEqual(self.processor.decoded, [[21, 22]])
        self.assertEqual(reply['raw_output'], self.raw)
        self.assertEqual(reply['consumed_input_sha256'], self.bound['input_sha256'])
        self.assertEqual(reply['provider_requests'], 0)
        self.assertTrue(self.load_kwargs['local_files_only'])
        self.assertFalse(self.load_kwargs['enable_audio_output'])
    def test_actual_helper_fps_is_scalar_for_short_clip_and_four_fps(self):
        self.sample_fps = 4.0
        for duration, requested_fps in ((1.0,2.0),(2.0,4.0)):
            with self.subTest(duration=duration,requested_fps=requested_fps):
                bound = deepcopy(self.bound)
                bound['original_interval']['end_s'] = 15.0+duration
                bound['input_sha256'] = _digest({k:v for k,v in bound.items() if k!='input_sha256'})
                backend = QwenOmniBackend(self.manifest_path,backend_code_revision='c'*40,
                                          device='cpu',video_fps=requested_fps)
                backend.infer(bound)
                self.assertEqual(self.processor.kwargs.get('fps'),4.0)
                self.assertIs(self.processor.kwargs.get('do_sample_frames'),False)
                self.assertEqual(backend.last_metrics['actual_video_fps'],4.0)
    def test_invalid_actual_fps_or_resampling_request_blocks_before_processor(self):
        backend = self.backend()
        for invalid in ({'fps':[],'do_sample_frames':False},
                        {'fps':[2.0,4.0],'do_sample_frames':False},
                        {'fps':[float('nan')],'do_sample_frames':False},
                        {'fps':[True],'do_sample_frames':False},
                        {'fps':[0],'do_sample_frames':False},
                        {'fps':[2.0],'do_sample_frames':True}):
            self.video_kwargs_override = invalid
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):backend.infer(self.bound)
            self.assertIsNone(self.processor.kwargs)
        self.assertIsNone(self.model.kwargs)

    @unittest.skipUnless(importlib.metadata.version('transformers')=='4.57.6',
                         'requires pinned isolated Omni runtime; no weights are loaded')
    def test_backend_messages_pass_official_pinned_system_prompt_guard(self):
        backend = self.backend()
        messages = backend._messages(self.bound)
        self.module_patch.stop()
        from transformers import Qwen2_5OmniProcessor
        from transformers.processing_utils import ProcessorMixin
        # Execute the official Omni guard, which indexes the system content
        # before delegating to the shared tokenizer. No model weights needed.
        processor = object.__new__(Qwen2_5OmniProcessor)
        with patch.object(ProcessorMixin, 'apply_chat_template', return_value='guard-passed') as delegate:
            rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(rendered, 'guard-passed')
        delegate.assert_called_once()
        self.assertIn('co-occurrence alone', messages[1]['content'][1]['text'])

    @unittest.skipUnless(importlib.metadata.version('transformers')=='4.57.6',
                         'requires pinned isolated Omni runtime; no weights are loaded')
    def test_real_pinned_helper_and_processor_preserve_actual_fps_clock(self):
        # Genuine CPU media decoding and the official processor __call__ path,
        # with tokenizer/video geometry stubs only; no model weights or gold.
        self.module_patch.stop()
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniProcessor
        captured={}
        class VideoGeometry:
            temporal_patch_size=2
            def __call__(self,**kwargs):
                captured['video_kwargs']=kwargs
                return {'video_grid_thw':torch.tensor([[2,1,1]])}
        def merge(schema,**kwargs):
            return {'videos_kwargs':{'seconds_per_chunk':2.0,'position_id_per_seconds':25,
                        'use_audio_in_video':kwargs['use_audio_in_video'],'fps':kwargs['fps'],
                        'do_sample_frames':kwargs['do_sample_frames']},
                    'audio_kwargs':{},'images_kwargs':{},'text_kwargs':{}}
        def replace(text, *args, **kwargs):
            captured['seconds_per_grid']=list(kwargs['video_second_per_grid'])
            return text
        class Tokenizer:
            init_kwargs={}
            def __call__(self,text,**kwargs):return {'input_ids':[[1,2]]}
        target=SimpleNamespace(tokenizer=Tokenizer(),
                   video_processor=VideoGeometry(),_merge_kwargs=merge,
                   replace_multimodal_special_tokens=replace)
        for duration,requested_fps in ((1,2.0),(2,4.0)):
            clip=self.directory/f'real-fps-{duration}.mp4'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                            f'color=c=blue:s=64x64:r=8:d={duration}',
                            '-f','lavfi','-i',f'sine=frequency=300:sample_rate=16000:duration={duration}',
                            '-c:v','mpeg4','-q:v','5','-c:a','aac','-shortest',str(clip)],check=True)
            messages=[{'role':'user','content':[{'type':'video','video':str(clip),'fps':requested_fps}]}]
            audios,images,videos,kwargs=process_mm_info(messages,use_audio_in_video=True,return_video_kwargs=True)
            self.assertEqual(len(kwargs['fps']),1)
            self.assertEqual(kwargs['fps'][0],4.0)
            self.assertIs(kwargs['do_sample_frames'],False)
            self.assertEqual(videos[0].shape[0],duration*4)
            self.assertIsNotNone(audios)
            Qwen2_5OmniProcessor.__call__(target,text='CPU boundary fixture',videos=videos,
                    fps=kwargs['fps'][0],do_sample_frames=False,use_audio_in_video=True)
            self.assertEqual(captured['seconds_per_grid'],[.5])
            self.assertIs(captured['video_kwargs']['do_sample_frames'],False)
            with self.assertRaises(TypeError):
                Qwen2_5OmniProcessor.__call__(target,text='Wrong FPS type boundary',videos=videos,
                        fps=kwargs['fps'],do_sample_frames=False,use_audio_in_video=True)

    def test_prompt_uses_relative_clip_times_and_excludes_private_identity_metadata(self):
        backend = self.backend()
        backend.infer(self.bound)
        messages = self.processor.messages
        text = '\n'.join(str(item['content']) for item in messages if isinstance(item['content'], str))
        text += messages[-1]['content'][-1]['text']
        self.assertIn('clip_interval', text)
        self.assertIn('2.000000', text)
        for forbidden in (self.bound['source_id'], self.bound['asset_sha256'], '15.0', 'gold', 'origin_url'):
            self.assertNotIn(forbidden, text)
        video = messages[-1]['content'][0]
        self.assertEqual(video['video'], str(self.clip))
    def test_relative_event_maps_to_original_clock(self):
        reply = self.backend().infer(self.bound)
        self.assertEqual(reply['structured_observation']['events'][0]['source_interval'],
                         {'start_s':15.25,'end_s':16.5})
        self.assertNotIn('clip_interval', reply['structured_observation']['events'][0])
    def test_non_json_retains_raw_with_explicit_parse_issue_and_abstention(self):
        self.processor.raw = 'Natural-language output without JSON.'
        backend = self.backend()
        record = SemanticObserver(backend, max_calls=1).observe(self.bound)
        self.assertEqual(record['raw_output'], self.processor.raw)
        self.assertEqual(record['status'], 'model_executed')
        self.assertEqual(record['structured_observation']['events'], [])
        self.assertIn('observer_parse_issue:invalid_json', record['structured_observation']['cannot_determine'])
        self.assertEqual(backend.last_metrics['parse_issue'], 'invalid_json')
        self.assertFalse(record['raw_av_observer_ready'])
    def test_outside_clip_and_non_finite_or_duplicate_times_abstain(self):
        for raw, issue in (
            (self.raw.replace('1.5', '2.5'), 'event_outside_clip_interval'),
            (self.raw.replace('0.25', 'NaN'), 'invalid_json'),
            ('{"events":[],"events":[],"uncertainty":"x","cannot_determine":[]}', 'invalid_json')):
            with self.subTest(issue=issue):
                value, actual = parse_clip_observation(raw, self.bound)
                self.assertEqual(value['events'], [])
                self.assertEqual(actual, issue)
    def test_huge_json_integer_abstains_without_losing_raw_generation(self):
        self.processor.raw = self.raw.replace('0.25', str(10**400))
        backend = self.backend()
        record = SemanticObserver(backend).observe(self.bound)
        self.assertEqual(record['status'],'model_executed')
        self.assertEqual(record['raw_output'],self.processor.raw)
        self.assertEqual(record['structured_observation']['events'],[])
        self.assertEqual(backend.last_metrics['parse_issue'],'invalid_clip_interval')

    def test_identity_is_defensive_and_matches_verified_config(self):
        backend = self.backend()
        expected = hashlib.sha256(self.config).hexdigest()
        self.assertEqual(backend.manifest['config_sha256'], expected)
        self.assertEqual(backend.manifest['model_revision'], 'a'*40)
        self.assertEqual(backend.manifest['mode'], 'native_joint_av')
        returned = backend.manifest
        returned['model_id'] = 'MUTATED'
        self.assertEqual(backend.manifest['model_id'], 'Qwen/Qwen2.5-Omni-7B')
        self.assertEqual(backend.model_manifest_sha256, hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
    def test_changed_weight_bytes_or_unverified_manifest_rejected_before_load(self):
        weight = self.snapshot / 'model-00001-of-00001.safetensors'
        weight.write_bytes(b'CHANGED_BYTES')
        with self.assertRaises(ValueError):self.backend()
        self.assertIsNone(self.load_kwargs)
    def test_unpinned_remote_path_or_unverified_identity_rejected(self):
        for field, value in (('revision','main'),('snapshot_path','Qwen/Qwen2.5-Omni-7B'),
                             ('all_lfs_verified',False),('repository_id','other/model'),
                             ('status','downloaded'),('schema_version','unverified-schema')):
            with self.subTest(field=field):
                artifact = deepcopy(self.artifact)
                artifact[field] = value
                self.manifest_path.write_text(json.dumps(artifact))
                with self.assertRaises(ValueError):self.backend()
        self.assertIsNone(self.load_kwargs)
    def test_model_failure_cost_and_safe_observer_error(self):
        backend = self.backend()
        self.model.failure = True
        runtime = SemanticObserver(backend, max_calls=1)
        record = runtime.observe(self.bound)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['actual_cost']['provider_requests'], 0)
        self.assertEqual(runtime.counts['backend_attempts'],1)
        self.assertEqual(runtime.counts['failures'],1)
        self.assertEqual(backend.last_metrics['error_type'], 'RuntimeError')
        self.assertGreaterEqual(backend.last_metrics['timings']['total_infer_seconds'],0)
        self.assertNotIn('PRIVATE_PROVIDER_OR_PATH_TEXT', json.dumps(record))
    def test_long_clip_missing_audio_and_tampered_input_block_before_generate(self):
        backend = self.backend()
        for mutation in ('duration','audio','digest'):
            bad = deepcopy(self.bound)
            if mutation == 'duration':bad['original_interval']['end_s']=17.1
            if mutation == 'audio':
                bad['modalities']=['video'];bad['mapping']['streams']=[{'codec_type':'video'}]
            if mutation == 'digest':bad['question']='changed'
            else:bad['input_sha256']=_digest({k:v for k,v in bad.items() if k!='input_sha256'})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):backend.infer(bad)
        self.assertIsNone(self.model.kwargs)
    def test_cli_help_imports_no_cuda_or_omni_runtime(self):
        result = subprocess.run([sys.executable,'research/scripts/run_av_observer_smoke.py','--help'],
                                capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--bindings',result.stdout)
        self.assertIn('--model-manifest',result.stdout)
    def test_cli_persists_private_raw_costs_and_public_count_only(self):
        from research.scripts.run_av_observer_smoke import main
        path = self.directory/'bindings.json'
        path.write_text(json.dumps({'schema_version':'av-observer-smoke-bindings-0.1',
                                    'inputs':[self.bound]}))
        output = self.directory/'run'
        with patch('sys.stdout'), patch('research.scripts.run_av_observer_smoke.collect_runtime_receipt', return_value={'schema_version':'av-native-live-runtime-0.1','status':'verified','errors':[],'packages':{},'fixture':'no real runtime attestation'}):
            code = main(['--bindings',str(path),'--model-manifest',str(self.manifest_path),
                         '--output-dir',str(output),'--device','cpu'])
        self.assertEqual(code,0)
        record = json.loads((output/'observations.private/0001.json').read_text())
        cost = json.loads((output/'costs.private.jsonl').read_text())
        summary = json.loads((output/'summary.public.json').read_text())
        self.assertEqual(record['raw_output'],self.raw)
        self.assertEqual(summary['planned_clip_count'],1)
        self.assertEqual(summary['attempted_clip_count'],1)
        self.assertEqual(summary['model_executed_clip_count'],1)
        self.assertEqual(summary['human_validated_clip_count'],0)
        self.assertFalse(summary['raw_av_observer_ready'])
        self.assertEqual(cost['provider_requests'],0)
        self.assertEqual(cost['generated_tokens'],2)
        self.assertGreaterEqual(cost['timings']['generation_seconds'],0)
        rendered = json.dumps(summary)
        for private in (str(self.clip),self.bound['source_id'],self.bound['question'],'Fixture event.'):
            self.assertNotIn(private,rendered)
        self.assertEqual(output.stat().st_mode & 0o777,0o700)
        with patch('sys.stderr'):
            repeat = main(['--bindings',str(path),'--model-manifest',str(self.manifest_path),
                           '--output-dir',str(output),'--device','cpu'])
        self.assertEqual(repeat,2)

    def test_cli_empty_generation_count_never_becomes_negative(self):
        from research.scripts.run_av_observer_smoke import main
        self.processor.raw = ''
        path = self.directory/'bindings.json'
        path.write_text(json.dumps({'schema_version':'av-observer-smoke-bindings-0.1','inputs':[self.bound]}))
        output = self.directory/'empty-run'
        with patch('sys.stdout'), patch('research.scripts.run_av_observer_smoke.collect_runtime_receipt', return_value={'schema_version':'av-native-live-runtime-0.1','status':'verified','errors':[],'packages':{},'fixture':'no real runtime attestation'}):
            code = main(['--bindings',str(path),'--model-manifest',str(self.manifest_path),
                         '--output-dir',str(output),'--device','cpu'])
        summary = json.loads((output/'summary.public.json').read_text())
        self.assertEqual(code,1)
        self.assertEqual(summary['model_executed_clip_count'],0)
        self.assertEqual(summary['failed_clip_count'],1)
        self.assertEqual(summary['valid_structured_output_count'],0)

    def test_runtime_mismatch_preserves_planned_denominator_before_model_load(self):
        from research.scripts.run_av_observer_smoke import main
        path=self.directory/'bindings.json'
        path.write_text(json.dumps({'schema_version':'av-observer-smoke-bindings-0.1','inputs':[self.bound]*2}))
        output=self.directory/'runtime-mismatch'
        receipt={'schema_version':'av-native-live-runtime-0.1','status':'mismatch',
                 'errors':['package_version_mismatch:torch'],'packages':{'torch':{'distribution_version':'2.8.0+cpu'}}}
        with patch('sys.stdout'),patch('research.scripts.run_av_observer_smoke.collect_runtime_receipt',return_value=receipt):
            code=main(['--bindings',str(path),'--model-manifest',str(self.manifest_path),
                       '--output-dir',str(output),'--device','cpu'])
        self.assertEqual(code,1)
        summary=json.loads((output/'summary.public.json').read_text())
        self.assertEqual(summary['planned_clip_count'],2)
        self.assertEqual(summary['attempted_clip_count'],0)
        self.assertEqual(summary['blocked_clip_count'],2)
        self.assertEqual(summary['live_runtime_status'],'mismatch')
        identity=json.loads((output/'run-manifest.private.json').read_text())
        receipt_path=output/'live-runtime.private.json'
        self.assertEqual(identity['live_runtime_receipt_sha256'],hashlib.sha256(receipt_path.read_bytes()).hexdigest())
        self.assertIsNone(self.load_kwargs)

    def test_live_runtime_versions_python_cuda_and_module_identity_are_checked(self):
        from research.scripts.run_av_observer_smoke import collect_runtime_receipt
        from research.scripts import run_av_observer_smoke as runner
        modules={'torch':'torch','torchvision':'torchvision','transformers':'transformers',
                 'qwen-omni-utils':'qwen_omni_utils','accelerate':'accelerate','huggingface-hub':'huggingface_hub'}
        versions={'torch':'2.8.0+cu126','torchvision':'0.23.0+cu126','transformers':'4.57.6',
                  'qwen-omni-utils':'0.0.9','accelerate':'1.11.0','huggingface-hub':'0.36.0'}
        lock=self.directory/'runtime.lock';lock.write_text('fixture runtime lock')
        expected={'schema_version':'av-observer-runtime-manifest-v1','python_series':'3.11',
                  'platform':'linux-64','packages':versions,'cuda_wheel_runtime':'12.6',
                  'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest()}
        path=self.directory/'runtime.json';path.write_text(json.dumps(expected))
        module_file=self.directory/'imported_fixture.py';module_file.write_text('# mock module, no model')
        def module(name):
            package=next((key for key,value in modules.items() if value==name),None)
            return SimpleNamespace(__file__=str(module_file),__version__=versions.get(package),
                                   version=SimpleNamespace(cuda='12.6'))
        with patch.object(runner.importlib,'import_module',side_effect=module), \
             patch.object(runner.importlib.metadata,'version',side_effect=lambda name:versions[name]), \
             patch.object(runner.importlib.metadata,'distribution',return_value=SimpleNamespace(locate_file=lambda name:module_file)):
            receipt=collect_runtime_receipt(path,lock)
            self.assertEqual(receipt['status'],'verified',receipt)
            self.assertEqual(receipt['interpreter']['executable'],sys.executable)
            self.assertEqual(receipt['packages']['torch']['distribution_version'],'2.8.0+cu126')
            self.assertEqual(receipt['packages']['torch']['module_sha256'],hashlib.sha256(module_file.read_bytes()).hexdigest())
            with patch.object(runner.importlib.metadata,'version',return_value='BAD_VERSION'):
                bad=collect_runtime_receipt(path,lock)
            self.assertEqual(bad['status'],'mismatch')
            self.assertIn('package_version_mismatch:torch',bad['errors'])
            with patch.object(runner.importlib.metadata,'distribution',return_value=SimpleNamespace(locate_file=lambda name:self.directory/'wrong-module.py')):
                bad=collect_runtime_receipt(path,lock)
            self.assertIn('imported_package_path_mismatch:torch',bad['errors'])
            expected['cuda_wheel_runtime']='11.8';path.write_text(json.dumps(expected))
            bad=collect_runtime_receipt(path,lock)
            self.assertIn('torch_cuda_build_mismatch',bad['errors'])
            expected['cuda_wheel_runtime']='12.6';expected['python_series']='3.10';path.write_text(json.dumps(expected))
            bad=collect_runtime_receipt(path,lock)
            self.assertIn('python_series_mismatch',bad['errors'])
            expected['python_series']='3.11';path.write_text(json.dumps(expected));lock.write_text('changed lock')
            bad=collect_runtime_receipt(path,lock)
            self.assertIn('runtime_lock_mismatch',bad['errors'])

    def test_malformed_live_runtime_manifest_rejects_before_importing_packages(self):
        from research.scripts import run_av_observer_smoke as runner
        path=self.directory/'runtime.json'
        lock=self.directory/'runtime.lock';lock.write_text('fixture lock')
        for value in ({'schema_version':'wrong','packages':{}},
                      {'schema_version':'av-observer-runtime-manifest-v1','packages':[]},
                      {'schema_version':'av-observer-runtime-manifest-v1','packages':{'torch':'2.8.0+cu126'}}):
            path.write_text(json.dumps(value))
            with patch.object(runner.importlib,'import_module',side_effect=AssertionError('must not import')):
                receipt=runner.collect_runtime_receipt(path,lock)
            self.assertEqual(receipt['status'],'mismatch')
            self.assertEqual(receipt['errors'],['invalid_runtime_manifest'])

    def test_cli_rejects_binding_budget_before_model_load(self):
        from research.scripts.run_av_observer_smoke import load_bindings
        path = self.directory/'bindings.json'
        path.write_text(json.dumps({'schema_version':'av-observer-smoke-bindings-0.1',
                                    'inputs':[self.bound]*7}))
        with self.assertRaises(ValueError):load_bindings(path,max_clips=6,max_clip_seconds=2)


if __name__ == '__main__':unittest.main()

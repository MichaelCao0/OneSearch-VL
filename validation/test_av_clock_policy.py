"""Explicit clock policy contracts; no weights or semantic validation."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import validation.test_av_omni_backend as fixtures
from research.av_agent.qwen_omni_backend import QwenOmniBackend

class ClockPolicyContracts(unittest.TestCase):
    setUp = fixtures.OmniBackendTests.setUp
    load_model = fixtures.OmniBackendTests.load_model
    def test_explicit_audio_policy_records_identity_and_array_receipt(self):
        bound=deepcopy(self.bound)
        bound['mapping']['streams'][0]['output_start_time_s']=1.012
        from research.av_agent.semantic_observer import _digest
        bound['input_sha256']=_digest({k:v for k,v in bound.items() if k!='input_sha256'})
        receipt={'schema_version':'av-audio-pts-adapter-0.1','clip_interval':{'start_s':0.0,'end_s':2.0},'input_sha256':bound['input_sha256'],
                 'observed_coverage_intervals':[{'start_s':1.012,'end_s':2.0}],
                 'synthetic_filler_is_evidence':False}
        with patch('research.av_agent.qwen_omni_backend.prepare_pts_audio',return_value=('PTS_AUDIO',receipt)):
            backend=QwenOmniBackend(self.manifest_path,backend_code_revision='c'*40,device='cpu',clock_policy='audio_pts_v1')
            backend.infer(bound)
        self.assertEqual(self.processor.kwargs['audio'],['PTS_AUDIO'])
        self.assertEqual(backend.last_metrics['clock_policy'],'audio_pts_v1')
        self.assertEqual(backend.last_metrics['clock_receipt'],receipt)
        self.assertIn('audio-pts-v1',backend.manifest['backend_id'])
        self.assertEqual(backend.last_metrics['parse_issue'],'event_outside_audio_coverage')
        self.assertEqual(backend.last_metrics['clock_receipt_sha256'],_digest(receipt))

    def test_postdecode_parse_exception_retains_raw_and_token_sidecar(self):
        import hashlib
        backend=QwenOmniBackend(self.manifest_path,backend_code_revision='c'*40,device='cpu')
        with patch('research.av_agent.qwen_omni_backend.parse_clip_observation',side_effect=RuntimeError('private')):
            with self.assertRaises(RuntimeError):backend.infer(self.bound)
        metrics=backend.last_metrics
        self.assertTrue(metrics['generation_completed'])
        self.assertEqual(metrics['generated_token_ids'],[21,22])
        self.assertEqual(metrics['decoded_raw_output'],self.raw)
        self.assertEqual(metrics['decoded_raw_output_sha256'],hashlib.sha256(self.raw.encode()).hexdigest())

    def test_unknown_output_policy_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            QwenOmniBackend(self.manifest_path,backend_code_revision='c'*40,
                            device='cpu',output_policy='repair_raw')
        self.assertIsNone(self.load_kwargs)

    def test_explicit_video_grid_replaces_endpoint_helper_sampling(self):
        receipt={'schema_version':'av-audio-pts-adapter-0.1','clip_interval':{'start_s':0.0,'end_s':2.0},'observed_coverage_intervals':[{'start_s':0.0,'end_s':2.0}]}
        video_receipt={'schema_version':'av-video-pts-grid-0.1','actual_selected_pts_s':[0,.5,1,1.5],
                       'target_grid_pts_s':[0,.5,1,1.5]}
        with patch('research.av_agent.qwen_omni_backend.prepare_pts_audio',return_value=('PTS_AUDIO',receipt)), \
             patch('research.av_agent.qwen_omni_backend.prepare_pts_video',return_value=('GRID_VIDEO',2.0,video_receipt)):
            backend=QwenOmniBackend(self.manifest_path,backend_code_revision='c'*40,
                                    device='cpu',clock_policy='av_pts_grid_v1')
            backend.infer(self.bound)
        self.assertEqual(self.calls, [], 'explicit grid path must not use endpoint helper video')
        self.assertEqual(self.processor.kwargs['audio'],['PTS_AUDIO'])
        self.assertEqual(self.processor.kwargs['videos'],['GRID_VIDEO'])
        self.assertEqual(self.processor.kwargs['fps'],2.0)
        self.assertEqual(backend.last_metrics['video_clock_receipt'],video_receipt)
        self.assertIn('av-pts-grid-v1',backend.manifest['backend_id'])
        self.assertIsNone(backend.last_metrics['parse_issue'])

    def test_unknown_clock_policy_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            QwenOmniBackend(self.manifest_path,backend_code_revision='c'*40,device='cpu',clock_policy='silently_rebase')
        self.assertIsNone(self.load_kwargs)

class AudioCoverageBoundaryTests(unittest.TestCase):
    def test_short_event_entirely_in_filler_is_rejected(self):
        from research.av_agent.qwen_omni_backend import parse_clip_observation
        import json
        bound={'original_interval':{'start_s':.5,'end_s':2.5},'mapping':{'output_to_source':{'scale':1.0,'offset_s':.5}},'modalities':['audio','video']}
        receipt={'schema_version':'av-audio-pts-adapter-0.1','clip_interval':{'start_s':0.0,'end_s':2.0},'observed_coverage_intervals':[{'start_s':1.012,'end_s':1.9998125}]}
        raw=json.dumps({'events':[{'modality':'audio','description':'candidate event','clip_interval':{'start_s':1.011,'end_s':1.0115},'uncertainty':'fixture'}],'uncertainty':'fixture','cannot_determine':[]})
        parsed,issue=parse_clip_observation(raw,bound,audio_coverage_receipt=receipt)
        self.assertEqual(issue,'event_outside_audio_coverage')
        self.assertEqual(parsed['events'],[])
    def test_exact_coverage_endpoint_survives_source_offset_mapping(self):
        from research.av_agent.qwen_omni_backend import parse_clip_observation
        import json
        bound={'original_interval':{'start_s':.5,'end_s':2.5},'mapping':{'output_to_source':{'scale':1.0,'offset_s':.5}},'modalities':['audio','video']}
        receipt={'schema_version':'av-audio-pts-adapter-0.1','clip_interval':{'start_s':0.0,'end_s':2.0},'observed_coverage_intervals':[{'start_s':1.012,'end_s':1.9998125}]}
        raw=json.dumps({'events':[{'modality':'joint_av','description':'candidate event','clip_interval':{'start_s':1.012,'end_s':1.9998125},'uncertainty':'fixture'}],'uncertainty':'fixture','cannot_determine':[]})
        parsed,issue=parse_clip_observation(raw,bound,audio_coverage_receipt=receipt)
        self.assertIsNone(issue)
        self.assertEqual(parsed['events'][0]['source_interval'],{'start_s':1.512,'end_s':2.4998125})

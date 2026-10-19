"""Actual pinned ProcessorMixin routing and audio token/feature contracts."""
from copy import deepcopy
import importlib.metadata
import unittest
import torch
from validation import test_native_omni_runner as fixtures
RecordingProcessor = fixtures.RecordingProcessor
video = fixtures.video

class ProcessorRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.metadata.version('transformers') != '4.57.6':
            raise unittest.SkipTest('requires the pinned Omni inference processor runtime')

    def prepare(self, audio=True, fps=2.0):
        from transformers.processing_utils import ProcessorMixin
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
        class MergingProcessor(RecordingProcessor):
            def __call__(self, **kwargs):
                self.merged = ProcessorMixin._merge_kwargs(
                    self, Qwen2_5OmniProcessorKwargs,
                    tokenizer_init_kwargs={}, **deepcopy(kwargs))
                return super().__call__(**kwargs)
        runner, _, _ = fixtures.NativeOmniTests().runner()
        processor = MergingProcessor();runner._processor = processor
        runner.prepare_inputs([{'parts':[video('/fake-media', audio=audio, fps=fps)]}])
        return processor.merged

    def test_audio_flag_reaches_actual_processor_merge(self):
        self.assertIs(self.prepare()['videos_kwargs']['use_audio_in_video'], True)

    def test_actual_sampling_fps_reaches_processor_merge(self):
        self.assertEqual(self.prepare(fps=1.7)['videos_kwargs'].get('fps'), 1.7)

    def test_decoded_frames_are_not_resampled(self):
        self.assertIs(self.prepare()['videos_kwargs'].get('do_sample_frames'), False)

    def test_muted_input_remains_visual_only(self):
        self.assertIs(self.prepare(audio=False)['videos_kwargs']['use_audio_in_video'], False)

class AudioTokenBindingTests(unittest.TestCase):
    def processor(self):
        return RecordingProcessor()

    def test_decoded_features_without_audio_placeholders_raise(self):
        from onesearch_vl.onesearch_infer.omni_runner import audio_token_binding
        inputs={'input_ids':torch.tensor([[2,2]]),
                'input_features':torch.zeros(1,128,100),
                'feature_attention_mask':torch.ones(1,100,dtype=torch.long)}
        with self.assertRaisesRegex(ValueError, 'Audio token/feature'):
            audio_token_binding(self.processor(), inputs)

    def test_matching_real_feature_lengths_are_counted(self):
        from onesearch_vl.onesearch_infer.omni_runner import audio_token_binding
        inputs={'input_ids':torch.tensor([[1]*25+[2]*4]),
                'input_features':torch.zeros(1,128,100),
                'feature_attention_mask':torch.ones(1,100,dtype=torch.long)}
        result=audio_token_binding(self.processor(), inputs)
        self.assertEqual(result, {'audio_token_count':25,'expected_audio_token_count':25,
                                  'video_token_count':4,'audio_token_binding_verified':True})

    def test_audio_placeholders_without_features_raise(self):
        from onesearch_vl.onesearch_infer.omni_runner import audio_token_binding
        with self.assertRaisesRegex(ValueError, 'Audio token/feature'):
            audio_token_binding(self.processor(), {'input_ids':torch.tensor([[1,2]])})

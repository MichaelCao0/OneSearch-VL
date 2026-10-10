"""Actual-byte bounded video grid tests; these guard time-axis selection."""
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import torch

from research.av_agent.audio_pts_adapter import decode_audio_on_clip_axis
from research.av_agent.video_pts_sampler import VideoPTSSamplerError, prepare_video_on_clip_grid
from validation.test_av_audio_pts_adapter import bind, digest, rebind


def make_grid_fixture(path, *, source_fps=25, video_pts=None, video_origin=0.0):
    """FFV1 visual plateau and PCM audio plateau both begin at 1.000s."""
    if video_pts is None:
        video_pts = [round(i * 1000 / source_fps + video_origin * 1000)
                     for i in range(source_fps * 2)]
    with av.open(str(path), 'w') as container:
        video = container.add_stream('ffv1', rate=source_fps)
        video.width = 56; video.height = 56; video.pix_fmt = 'yuv420p'
        video.time_base = Fraction(1, 1000)
        video.codec_context.time_base = Fraction(1, 1000)
        audio = container.add_stream('pcm_f32le', rate=16000)
        audio.layout = 'mono'; audio.time_base = Fraction(1, 16000)
        for pts in video_pts:
            value = 255 if 1000 <= pts < 1040 else 0
            pixels = np.full((56, 56, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts = pts; frame.time_base = Fraction(1, 1000)
            for packet in video.encode(frame): container.mux(packet)
        samples = np.zeros((1, 32000), dtype=np.float32)
        samples[0, 16000:16160] = .75
        frame = av.AudioFrame.from_ndarray(samples, format='flt', layout='mono')
        frame.sample_rate = 16000; frame.pts = 0; frame.time_base = Fraction(1, 16000)
        for packet in audio.encode(frame): container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None): container.mux(packet)
    result = bind(path, audio_origin=0.0, video_origin=video_pts[0] / 1000)
    result['mapping']['streams'][0]['source_time_base'] = str(Fraction(1, source_fps))
    return rebind(result)


class VideoPTSSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def fixture(self, name='clip.mkv', **kwargs):
        return make_grid_fixture(Path(self.tmp.name) / name, **kwargs)

    def test_simultaneous_pulses_land_at_model_grid_one_second(self):
        # Endpoint linspace would miss the actual visual pulse at source 1.000s.
        bound = self.fixture()
        before = hashlib.sha256(Path(bound['clip_path']).read_bytes()).hexdigest()
        video, fps, receipt = prepare_video_on_clip_grid(bound)
        audio, _ = decode_audio_on_clip_axis(bound)
        self.assertEqual(fps, 2.0)
        self.assertEqual(tuple(video.shape[:2]), (4, 3))
        self.assertEqual(video.dtype, torch.float32)
        self.assertEqual(receipt['target_timestamps_s'], [0.0, .5, 1.0, 1.5])
        self.assertEqual(receipt['selected_indices'], [0, 12, 25, 37])
        self.assertEqual(receipt['actual_timestamps_s'], [0.0, .48, 1.0, 1.48])
        self.assertGreater(float(video[2].mean()), 250)
        self.assertLess(float(video[[0, 1, 3]].abs().max()), 2)
        self.assertGreater(float(audio[16000:16160].mean()), .74)
        self.assertEqual(before, hashlib.sha256(Path(bound['clip_path']).read_bytes()).hexdigest())
        self.assertEqual(receipt['input_media_sha256_before'], before)
        self.assertEqual(receipt['input_media_sha256_after'], before)
        self.assertEqual(receipt['input_sha256'], bound['input_sha256'])
        self.assertEqual(receipt['output_tensor_sha256'], hashlib.sha256(video.numpy().astype('<f4', copy=False).tobytes()).hexdigest())
        self.assertEqual(receipt['output_tensor_shape'], list(video.shape))
        self.assertEqual(receipt['receipt_sha256'], digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'}))

    def test_true_25fps_half_frame_ties_choose_earlier_and_disclose_quantization(self):
        _, _, receipt = prepare_video_on_clip_grid(self.fixture())
        self.assertEqual(receipt['quantization_policy'], 'nearest_actual_decoded_pts; ties_choose_earlier; unique_frames')
        self.assertEqual(receipt['per_frame_error_s'], [0.0, -.02, 0.0, -.02])
        self.assertEqual(receipt['max_abs_quantization_error_s'], .02)
        self.assertAlmostEqual(receipt['quantization_bound_s'], .021)
        self.assertEqual(receipt['model_temporal_patch_size'], 2)
        self.assertEqual(receipt['model_temporal_patch_timestamps_s'], [0.0, 1.0])
        self.assertEqual(receipt['model_temporal_patch_frame_indices'], [[0, 1], [2, 3]])
        self.assertEqual([x['pts'] for x in receipt['decoded_video_frames']], list(range(0, 2000, 40)))
        self.assertTrue(all(x['time_base'] == '1/1000' for x in receipt['decoded_video_frames']))
        self.assertEqual(len(receipt['sampler_module_sha256']), 64)
        self.assertEqual(receipt['resize']['package_version'], '0.0.9')
        self.assertEqual(len(receipt['resize']['source_module_sha256']), 64)

    def test_one_ms_quantized_origin_stays_on_common_clip_axis(self):
        _, _, receipt = prepare_video_on_clip_grid(self.fixture(video_origin=.001))
        self.assertEqual(receipt['actual_timestamps_s'], [.001, .481, 1.001, 1.481])
        self.assertEqual(receipt['target_timestamps_s'], [0.0, .5, 1.0, 1.5])
        self.assertAlmostEqual(receipt['per_frame_error_s'][0], .001)
        self.assertEqual(receipt['video_clock']['decoded_first_pts_s'], .001)

    def test_dropped_or_vfr_actual_pts_fail_despite_plausible_average_rate(self):
        altered = list(range(0, 2000, 40)); altered[20] += 10
        dropped = list(range(0, 2000, 40)); del dropped[20]
        for name, pts in [('vfr', altered), ('dropped', dropped)]:
            with self.subTest(name=name):
                bound = self.fixture(name + '.mkv', video_pts=pts)
                with av.open(bound['clip_path']) as container:
                    self.assertEqual(container.streams.video[0].average_rate, Fraction(25, 1))
                with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(bound)

    def test_odd_noninteger_and_unbounded_sample_budgets_fail_closed(self):
        bound = self.fixture()
        for fps in [.5, 1.5, 1.2, 4.5, 0, -1, True, float('nan'), float('inf'), 10**1000]:
            with self.subTest(fps_type=type(fps).__name__, fps=repr(fps)[:30]):
                with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(bound, sample_fps=fps)
        for fps, count in [(1.0, 2), (3.0, 6), (4.0, 8)]:
            with self.subTest(fps=fps):
                tensor, scalar, receipt = prepare_video_on_clip_grid(bound, sample_fps=fps)
                self.assertEqual(len(tensor), count)
                self.assertEqual(scalar, fps)
                self.assertEqual(len(set(receipt['selected_indices'])), count)
        other = deepcopy(bound); other['original_interval']['end_s'] = 1.999; other = rebind(other)
        with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(other)
        other = deepcopy(bound); other['original_interval']['end_s'] = 2.01; other = rebind(other)
        with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(other)

    def test_duplicate_nearest_frames_are_rejected_without_tail_padding(self):
        with self.assertRaises(VideoPTSSamplerError):
            prepare_video_on_clip_grid(self.fixture(source_fps=1))

    def test_invalid_binding_hash_clock_and_pixel_budgets_fail_closed(self):
        bound = self.fixture()
        changes = []
        for key, value in [('input_sha256', '0' * 64), ('output_sha256', '0' * 64)]:
            other = deepcopy(bound); other[key] = value; changes.append(other)
        other = deepcopy(bound); other['mapping']['streams'][0]['output_start_time_s'] = .04
        changes.append(rebind(other))
        for other in changes:
            with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(other)
        for value in [0, -1, True, 200704.5, float('nan'), float('inf'), 10**1000]:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(bound, max_pixels=value)
        with Path(bound['clip_path']).open('ab') as out: out.write(b'changed')
        with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(bound)

    def test_media_mutation_during_actual_processing_is_rejected(self):
        import research.av_agent.video_pts_sampler as sampler
        bound = self.fixture(); real_clock = sampler._video_clock
        def clock_then_mutate(*args, **kwargs):
            result = real_clock(*args, **kwargs)
            with Path(bound['clip_path']).open('ab') as out: out.write(b'concurrent-change')
            return result
        with patch.object(sampler, '_video_clock', side_effect=clock_then_mutate):
            with self.assertRaises(VideoPTSSamplerError): prepare_video_on_clip_grid(bound)


if __name__ == '__main__': unittest.main()

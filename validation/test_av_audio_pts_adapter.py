"""Actual-byte clock adapter probes; fixtures contain controlled PTS and pulses."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np

from research.av_agent.audio_pts_adapter import (
    AudioPTSAdapterError, coverage_contains_interval, decode_audio_on_clip_axis,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def bind(path, *, audio_origin=0.75, video_origin=0.0, duration=2.0):
    blob = Path(path).read_bytes(); h = hashlib.sha256(blob).hexdigest()
    b = {'schema_version':'av-observer-input-0.1', 'source_id':'src_'+h[:24], 'asset_sha256':h,
         'derived_id':'derived-'+h[:24], 'output_sha256':h, 'output_bytes':len(blob),
         'clip_path':str(Path(path).resolve()), 'original_interval':{'start_s':0.0, 'end_s':duration},
         'mapping':{'axis':'source_local_presentation', 'output_axis':'presentation_pts_seconds',
                    'interval':'half_open', 'output_to_source':{'scale':1.0,'offset_s':0.0},
                    'source_to_output':{'scale':1.0,'offset_s':-0.0},
                    'streams':[{'source_stream_index':0,'output_stream_index':0,'codec_type':'video',
                                'source_time_base':'1/25','source_offset_s':0.0,
                                'output_time_base':'1/1000','output_start_time_s':video_origin},
                               {'source_stream_index':1,'output_stream_index':1,'codec_type':'audio',
                                'source_time_base':'1/16000','source_offset_s':audio_origin,
                                'output_time_base':'1/1000','output_start_time_s':audio_origin}]},
         'modalities':['audio','video'], 'input_mode':'raw_av', 'question':'Describe direct observations.',
         'probe_id':'synthetic-clock-adapter'}
    b['input_sha256'] = digest(b)
    return b


def rebind(b):
    b = deepcopy(b); b.pop('input_sha256', None); b['input_sha256'] = digest(b); return b


def make_fixture(path, *, rate=16000, audio_spans=((0.75,0.5),(1.5,0.25)),
                 video_pts=None, silent=False, extra_audio=False):
    """PCM floats plus FFV1: waveform locations can be checked independently."""
    with av.open(str(path), 'w') as c:
        v = c.add_stream('ffv1', rate=25); v.width=16; v.height=16; v.pix_fmt='yuv420p'; v.time_base=Fraction(1,1000); v.codec_context.time_base=Fraction(1,1000)
        a = c.add_stream('pcm_f32le', rate=rate); a.layout='mono'; a.time_base=Fraction(1,rate)
        if extra_audio:
            other = c.add_stream('pcm_f32le', rate=rate); other.layout='mono'; other.time_base=Fraction(1,rate)
        for pts in (video_pts if video_pts is not None else [i*40 for i in range(50)]):
            f = av.VideoFrame.from_ndarray(np.zeros((16,16,3), dtype=np.uint8), format='rgb24')
            f.pts=pts; f.time_base=Fraction(1,1000)
            for p in v.encode(f): c.mux(p)
        for start, duration in audio_spans:
            n=round(duration*rate); samples=np.zeros((1,n), dtype=np.float32)
            if not silent:
                # A 10ms plateau starts exactly 50ms after each span origin.
                samples[0,round(.05*rate):round(.06*rate)] = .75
            f = av.AudioFrame.from_ndarray(samples, format='flt', layout='mono')
            f.sample_rate=rate; f.pts=round(start*rate); f.time_base=Fraction(1,rate)
            for p in a.encode(f): c.mux(p)
            if extra_audio:
                for p in other.encode(f): c.mux(p)
        for s in ([v,a,other] if extra_audio else [v,a]):
            for p in s.encode(None): c.mux(p)
    return bind(path, audio_origin=audio_spans[0][0], video_origin=(video_pts[0]/1000 if video_pts else 0.0))


class AudioPTSAdapterTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
    def fixture(self, name='clip.mkv', **kwargs): return make_fixture(Path(self.tmp.name)/name, **kwargs)

    def test_delayed_pulse_stays_at_actual_clip_pts_and_missing_audio_is_masked(self):
        b=self.fixture(); a,r=decode_audio_on_clip_axis(b)
        self.assertEqual((a.dtype,a.shape),(np.dtype('float32'),(32000,)))
        self.assertTrue(np.all(a[:12000] == 0))
        self.assertGreater(float(a[12840:12920].mean()), .70)
        self.assertGreater(float(a[24840:24920].mean()), .70)
        self.assertEqual(r['origin']['decoded_audio_first_pts_s'], .75)
        self.assertEqual(r['origin']['resampled_audio_first_pts_s'], .75)
        self.assertEqual([(x['start_sample'],x['end_sample']) for x in r['observed_coverage_intervals']],[(12000,20000),(24000,28000)])
        self.assertEqual([(x['start_sample'],x['end_sample']) for x in r['missing_regions']],[(0,12000),(20000,24000),(28000,32000)])
        self.assertFalse(coverage_contains_interval(r,0,.5))
        self.assertTrue(coverage_contains_interval(r,.8,1.0))
        self.assertFalse(coverage_contains_interval(r,1.2,1.6))
        self.assertEqual(r['zero_filler_meaning'],'no_decoded_audio_media; not recorded silence')
        self.assertEqual(r['output_array_sha256'],hashlib.sha256(a.astype('<f4',copy=False).tobytes()).hexdigest())
        mask=np.zeros(32000,dtype=np.uint8);mask[12000:20000]=1;mask[24000:28000]=1
        self.assertEqual(r['coverage_mask_sha256'],hashlib.sha256(mask.tobytes()).hexdigest())
        self.assertEqual(r['input_sha256'],b['input_sha256'])
        self.assertEqual(r['input_media_sha256'],b['output_sha256'])
        self.assertEqual(len(r['adapter_module_sha256']),64)
        self.assertEqual(hashlib.sha256(Path(b['clip_path']).read_bytes()).hexdigest(),b['output_sha256'])

    def test_recorded_silence_remains_observed_media(self):
        a,r=decode_audio_on_clip_axis(self.fixture(silent=True))
        self.assertTrue(np.all(a==0));self.assertTrue(coverage_contains_interval(r,.8,1.0))
        self.assertFalse(coverage_contains_interval(r,.1,.2))

    def test_real_44100_resampling_preserves_delayed_pulse(self):
        b=self.fixture(rate=44100,audio_spans=((.75,.5),));a,r=decode_audio_on_clip_axis(b)
        self.assertGreater(float(a[12840:12920].mean()),.68)
        self.assertLess(float(abs(a[:11980]).max()),1e-6)
        self.assertEqual(r['origin']['resampled_audio_first_pts_s'],.75)
        self.assertEqual(r['resampling']['output_sample_rate'],16000)
        self.assertEqual(r['resampling']['input_sample_rates'],[44100])
        self.assertTrue(coverage_contains_interval(r,.81,1.2))

    def test_untrusted_stored_origin_cannot_override_actual_pts(self):
        b=self.fixture();b['mapping']['streams'][1]['output_start_time_s']=0.0;b=rebind(b)
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b)

    def test_material_overlap_is_rejected_instead_of_overwriting_waveform(self):
        b=self.fixture(audio_spans=((.75,.5),(1.0,.5)))
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b)

    def test_unknown_or_nonmonotonic_actual_pts_cannot_be_called_coverage(self):
        # Matroska preserves duplicate frame PTS even with monotonic packet DTS.
        b=self.fixture(audio_spans=((.75,.25),(.75,.25)))
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b)

    def test_material_out_of_clip_audio_is_rejected(self):
        b=self.fixture(audio_spans=((1.75,.5),))
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b)

    def test_vfr_and_nonzero_video_origins_fail_closed_from_actual_bytes(self):
        pts=[i*40 for i in range(50)];pts[20]+=10
        for n, stamps in [('vfr',pts),('origin',[i*40+40 for i in range(49)])]:
            with self.subTest(n=n):
                with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(self.fixture(n+'.mkv',video_pts=stamps))

    def test_one_ms_video_rounding_supported_with_observed_cfr_receipt(self):
        a,r=decode_audio_on_clip_axis(self.fixture(video_pts=[i*40+1 for i in range(50)]))
        self.assertEqual(r['video_clock']['decoded_first_pts_s'],.001)
        self.assertEqual(r['video_clock']['support_policy'],'decoded_cfr_grid_with_at_most_one_container_tick_origin')

    def test_duplicate_actual_or_declared_streams_fail_closed(self):
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(self.fixture(extra_audio=True))
        b=self.fixture('single.mkv');b['mapping']['streams'].append(deepcopy(b['mapping']['streams'][1]));b=rebind(b)
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b)
        b=self.fixture('index.mkv');b['mapping']['streams'][1]['output_stream_index']=0;b=rebind(b)
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b)

    def test_input_integrity_unknown_clock_and_nonfinite_budgets_fail_closed(self):
        b=self.fixture()
        cases=[]
        for value in [None,True,float('nan'),float('inf'),10**1000]:
            x=deepcopy(b);x['mapping']['streams'][0]['output_start_time_s']=value;cases.append(x)
        x=deepcopy(b);x['output_sha256']='0'*64;cases.append(x)
        for x in cases:
            with self.subTest(value_type=type(x['mapping']['streams'][0]['output_start_time_s']).__name__):
                with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(x)
        for rate in [0,-1,True,16001,float('nan'),float('inf'),10**1000]:
            with self.subTest(rate_type=type(rate).__name__):
                with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b,target_sample_rate=rate)
        b=self.fixture('duration.mkv');b['original_interval']['end_s']=2.01;b=rebind(b)
        with self.assertRaises(AudioPTSAdapterError):decode_audio_on_clip_axis(b)

    def test_coverage_gate_requires_finite_nonempty_supported_interval(self):
        _,r=decode_audio_on_clip_axis(self.fixture())
        for start,end in [(float('nan'),1),(0,float('inf')),(False,1),(.9,.9),(-.1,.2),(1.5,2.1),(.8,1.6)]:
            self.assertFalse(coverage_contains_interval(r,start,end))
        self.assertTrue(coverage_contains_interval(r,.75,1.25))
        self.assertFalse(coverage_contains_interval({},.8,1))


if __name__ == '__main__':unittest.main()

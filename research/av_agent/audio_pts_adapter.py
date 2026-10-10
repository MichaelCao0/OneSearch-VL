"""Bounded decoded-PTS audio preparation on the shared clip presentation axis.

This adapter proves array placement and observed-media coverage, not semantic
AV timing. Video support is limited to a decoded CFR grid whose first PTS is
zero (or one <=1ms container tick); general VFR/nonzero origins fail closed.
Audio resampler output PTS anchor every continuous segment. Adjacent frame PTS
may be reconciled only within the measured container tick plus sample rounding,
and each reconciliation is disclosed. A larger positive source-frame gap starts
a fresh resampler segment with missing-media filler. Material overlap is rejected.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from research.av_agent.semantic_observer import validate_observer_input

SCHEMA_VERSION = 'av-audio-pts-adapter-0.1'
MAX_CLIP_SECONDS = 2.0
MAX_OUTPUT_SAMPLES = 32000
MAX_VIDEO_FRAMES = 512
MAX_AUDIO_FRAMES = 4096


class AudioPTSAdapterError(ValueError):
    """Actual clocks or a bounded placement cannot be demonstrated."""


def _finite(value):
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        return False


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _time_base(value):
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        raise AudioPTSAdapterError('unknown or invalid time base') from None
    if result <= 0 or result > Fraction(1,1000):
        raise AudioPTSAdapterError('time base coarser than supported 1ms clock')
    return result


def _frame_time(frame):
    if isinstance(frame.pts, bool) or not isinstance(frame.pts, int):
        raise AudioPTSAdapterError('unknown or invalid decoded PTS')
    tb = _time_base(frame.time_base)
    result = frame.pts * tb
    if not _finite(float(result)):
        raise AudioPTSAdapterError('nonfinite decoded PTS')
    return result, tb


def _nearest_sample(t, rate):
    # Rational half-up rounding avoids binary-float and ties-to-even surprises.
    value = t * rate
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _intervals(mask, value, sample_rate, duration):
    rows = []; start = None
    for i in range(len(mask) + 1):
        selected = i < len(mask) and bool(mask[i]) == value
        if selected and start is None:
            start = i
        elif not selected and start is not None:
            rows.append({'start_sample':start, 'end_sample':i,
                         'start_s':start / sample_rate, 'end_s':min(i / sample_rate, duration)})
            start = None
    return rows


def _mapped_streams(bound, actual):
    declared = bound['mapping']['streams']
    if (len(declared) != 2 or sorted(s.get('codec_type') for s in declared) != ['audio','video'] or
            len(actual.audio) != 1 or len(actual.video) != 1 or len(actual) != 2):
        raise AudioPTSAdapterError('exactly one uniquely mapped audio and video stream required')
    mapped = {}
    for kind in ('audio','video'):
        d = next(s for s in declared if s['codec_type'] == kind)
        stream = getattr(actual, kind)[0]
        index = d.get('output_stream_index')
        if isinstance(index,bool) or not isinstance(index,int) or index != stream.index:
            raise AudioPTSAdapterError('declared output stream does not match actual stream index')
        tb = _time_base(stream.time_base)
        if _time_base(d.get('output_time_base')) != tb or not _finite(d.get('output_start_time_s')):
            raise AudioPTSAdapterError('known declared clock must match actual output time base')
        mapped[kind] = (stream, d, tb)
    return mapped


def _video_clock(path, declared, duration):
    import av
    with av.open(str(path)) as container:
        stream, metadata, tb = _mapped_streams(declared, container.streams)['video']
        fps = stream.average_rate
        if fps is None or fps <= 0 or not _finite(float(fps)) or float(fps) > 240:
            raise AudioPTSAdapterError('known bounded CFR video rate required')
        times = []
        for frame in container.decode(stream):
            t, frame_tb = _frame_time(frame)
            if frame_tb != tb or (times and t <= times[-1]):
                raise AudioPTSAdapterError('nonmonotonic or changing video PTS clock')
            if not 0 <= t < duration or len(times) >= MAX_VIDEO_FRAMES:
                raise AudioPTSAdapterError('video frame outside bounded clip axis')
            times.append(t)
        if len(times) < 2:
            raise AudioPTSAdapterError('at least two actual video frames required for CFR proof')
        origin = times[0]
        if not 0 <= origin <= tb:
            raise AudioPTSAdapterError('nonzero video origin exceeds one container tick; unsupported')
        if abs(float(origin) - metadata['output_start_time_s']) > float(tb) + 1e-12:
            raise AudioPTSAdapterError('declared video origin disagrees with actual first PTS')
        deviation = max(abs(t - (origin + Fraction(i,1)/fps)) for i,t in enumerate(times))
        if deviation > tb:
            raise AudioPTSAdapterError('actual video PTS do not prove a CFR grid; VFR unsupported')
        return {'stream_index':stream.index, 'decoded_frame_count':len(times),
                'decoded_first_pts_s':float(origin), 'decoded_last_pts_s':float(times[-1]),
                'time_base':str(tb), 'verified_cfr_fps':float(fps),
                'max_cfr_grid_deviation_s':float(deviation),
                'origin_rounding_allowance_s':float(tb),
                'support_policy':'decoded_cfr_grid_with_at_most_one_container_tick_origin',
                'limits':'general VFR and material nonzero video origin are unsupported; no semantic precision claim'}


def decode_audio_on_clip_axis(bound_input, target_sample_rate=16000):
    """Return a float32 mono clip-axis array and an auditable coverage receipt.

    The input must be a full immutable observer binding. Media and binding hashes
    are verified before decode and media is rehashed afterwards. Audio gap filler
    is an input transport accommodation and cannot support silence/event claims.
    Coverage gates must be applied to all accepted audio or joint AV intervals.
    """
    import av
    import numpy as np

    if (isinstance(target_sample_rate,bool) or not isinstance(target_sample_rate,int) or
            not 1000 <= target_sample_rate <= 16000):
        raise AudioPTSAdapterError('integer target sample rate must be in [1000,16000]')
    try:
        errors = validate_observer_input(bound_input, verify_files=True)
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, OSError):
        raise AudioPTSAdapterError('invalid immutable observer binding') from None
    if errors or bound_input['modalities'] != ['audio','video']:
        raise AudioPTSAdapterError('invalid immutable joint AV observer binding')
    start = Fraction(str(bound_input['original_interval']['start_s']))
    end = Fraction(str(bound_input['original_interval']['end_s']))
    duration = end - start
    if not 0 < duration <= Fraction(str(MAX_CLIP_SECONDS)):
        raise AudioPTSAdapterError('clip duration exceeds bounded 2s adapter')
    count = math.ceil(duration * target_sample_rate)
    if count > MAX_OUTPUT_SAMPLES:
        raise AudioPTSAdapterError('output exceeds 32000-sample budget')
    path = Path(bound_input['clip_path'])
    video = _video_clock(path, bound_input, duration)
    segments = []; frames = []; previous = None; input_records = []; gaps = []
    with av.open(str(path)) as container:
        stream, metadata, stream_tb = _mapped_streams(bound_input, container.streams)['audio']
        actual_audio_index = stream.index
        input_rate = None
        for frame in container.decode(stream):
            t,tb = _frame_time(frame)
            rate = frame.sample_rate
            if (isinstance(rate,bool) or not isinstance(rate,int) or not 1000 <= rate <= 192000 or
                    isinstance(frame.samples,bool) or not isinstance(frame.samples,int) or frame.samples < 1):
                raise AudioPTSAdapterError('invalid decoded audio sample clock')
            if tb != stream_tb or (input_rate is not None and rate != input_rate):
                raise AudioPTSAdapterError('changing audio rate or time base unsupported')
            input_rate = rate
            tolerance = tb + Fraction(1,rate)
            finish = t + Fraction(frame.samples, rate)
            if t < -tolerance or finish > duration + tolerance:
                raise AudioPTSAdapterError('material audio interval outside common clip axis')
            if len(input_records) >= MAX_AUDIO_FRAMES:
                raise AudioPTSAdapterError('decoded audio frame budget exceeded')
            gap = None
            if previous is not None:
                if t <= previous['time']:
                    raise AudioPTSAdapterError('unknown or nonmonotonic audio PTS')
                gap = t - previous['end']
                if gap < -tolerance:
                    raise AudioPTSAdapterError('material decoded audio overlap cannot be uniquely placed')
                if gap > tolerance:
                    segments.append(frames); frames=[]
                    gaps.append({'previous_end_s':float(previous['end']), 'next_start_s':float(t),
                                 'gap_s':float(gap), 'explanation':'actual decoded PTS gap; no audio media; resampler reset'})
            input_records.append({'pts':frame.pts, 'time_base':str(tb), 'start_s':float(t),
                                  'end_s':float(finish), 'samples':frame.samples, 'sample_rate':rate,
                                  'gap_from_previous_s':None if gap is None else float(gap)})
            frames.append(frame); previous={'time':t,'end':finish}
        if not frames:
            raise AudioPTSAdapterError('actual decoded audio frames required')
        segments.append(frames)
        if abs(input_records[0]['start_s'] - metadata['output_start_time_s']) > float(stream_tb) + 1 / input_rate:
            raise AudioPTSAdapterError('declared audio origin disagrees with actual first decoded PTS')
    audio = np.zeros(count, dtype=np.float32); coverage = np.zeros(count, dtype=np.uint8)
    output_records = []; rounding = []; clipped = []; output_origin = None
    tolerance_samples = math.ceil((stream_tb + Fraction(1,input_rate) + Fraction(1,target_sample_rate)) * target_sample_rate)
    for segment_index, source_frames in enumerate(segments):
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sample_rate)
        outputs = []
        try:
            for frame in source_frames:
                outputs.extend(resampler.resample(frame))
            outputs.extend(resampler.resample(None))
        except (ValueError, av.FFmpegError):
            raise AudioPTSAdapterError('native resampler cannot preserve decoded PTS') from None
        cursor = None; previous_output_t = None
        for frame in outputs:
            t,tb = _frame_time(frame)
            if frame.sample_rate != target_sample_rate or frame.layout.name != 'mono' or frame.format.name != 'fltp':
                raise AudioPTSAdapterError('native resampler output clock/layout mismatch')
            if previous_output_t is not None and t <= previous_output_t:
                raise AudioPTSAdapterError('nonmonotonic actual resampler output PTS')
            previous_output_t = t
            native_start = _nearest_sample(t, target_sample_rate)
            placement = native_start
            if cursor is not None:
                delta = native_start - cursor
                if abs(delta) > tolerance_samples:
                    raise AudioPTSAdapterError('material resampled gap/overlap unexplained by source segment')
                if delta:
                    placement = cursor
                    rounding.append({'segment_index':segment_index, 'output_frame_index':len(output_records),
                                     'native_start_sample':native_start, 'placed_start_sample':placement,
                                     'correction_samples':-delta,
                                     'explanation':'continuous source segment; container tick/sample rounding only'})
            if output_origin is None: output_origin=float(t)
            samples = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
            if len(samples) != frame.samples or not np.isfinite(samples).all():
                raise AudioPTSAdapterError('invalid finite native resampled samples')
            finish = placement + len(samples)
            if placement < -tolerance_samples or finish > count + tolerance_samples:
                raise AudioPTSAdapterError('material resampled audio outside bounded clip axis')
            left = max(0,placement); right=min(count,finish)
            if right <= left:
                raise AudioPTSAdapterError('resampled frame has no support inside clip')
            if coverage[left:right].any():
                raise AudioPTSAdapterError('resampled segments overlap observed media')
            audio[left:right] = samples[left-placement:right-placement]
            coverage[left:right] = 1
            if left != placement or right != finish:
                clipped.append({'output_frame_index':len(output_records), 'left_samples':left-placement,
                                'right_samples':finish-right, 'explanation':'at most container tick/sample rounding at clip boundary'})
            output_records.append({'segment_index':segment_index,'pts':frame.pts,'time_base':str(tb),
                                   'native_start_s':float(t),'samples':len(samples),
                                   'placed_start_sample':left,'placed_end_sample':right})
            cursor=finish
    if not output_records:
        raise AudioPTSAdapterError('native resampler produced no timestamped media')
    media_hash = _sha256_file(path)
    if media_hash != bound_input['output_sha256']:
        raise AudioPTSAdapterError('media bytes changed during clock adaptation')
    receipt = {'schema_version':SCHEMA_VERSION, 'input_sha256':bound_input['input_sha256'],
               'input_media_sha256':media_hash, 'adapter_module_sha256':_sha256_file(__file__),
               'output_array_sha256':hashlib.sha256(audio.astype('<f4',copy=False).tobytes()).hexdigest(),
               'coverage_mask_sha256':hashlib.sha256(coverage.tobytes()).hexdigest(),
               'axis':'clip_local_presentation_pts_seconds', 'clip_interval':{'start_s':0.0,'end_s':float(duration)},
               'output_sample_rate':target_sample_rate, 'output_samples':count,
               'output_dtype':'float32', 'output_channels':1,
               'origin':{'declared_audio_first_pts_s':metadata['output_start_time_s'],
                         'decoded_audio_first_pts_s':input_records[0]['start_s'],
                         'resampled_audio_first_pts_s':output_origin,
                         'placement_uses':'actual native resampler output PTS; no metadata-based rebasing'},
               'video_clock':video, 'audio_stream_index':actual_audio_index,
               'decoded_audio_frames':input_records, 'resampled_audio_frames':output_records,
               'resampling':{'implementation':'PyAV AudioResampler', 'pyav_version':av.__version__,
                             'format':'fltp','layout':'mono','input_sample_rates':[input_rate],
                             'output_sample_rate':target_sample_rate, 'segments':len(segments),
                             'reset_at_material_decoded_gap':True, 'flush_each_segment':True},
               'rounding':{'sample_index_rule':'rational nearest, half up',
                           'max_continuity_correction_samples':tolerance_samples,
                           'corrections':rounding, 'clips':clipped,
                           'precision_limit':'container tick plus native input/output sample rounding; no semantic precision guarantee'},
               'decoded_material_gaps':gaps,
               'observed_coverage_intervals':_intervals(coverage,True,target_sample_rate,float(duration)),
               'missing_regions':_intervals(coverage,False,target_sample_rate,float(duration)),
               'zero_filler_meaning':'no_decoded_audio_media; not recorded silence',
               'semantic_review_status':'pending'}
    receipt['receipt_sha256'] = _digest(receipt)
    return audio, receipt


def coverage_contains_interval(receipt, start_s, end_s):
    """Fail closed unless the complete finite half-open interval has audio media.

    Use only the immutable receipt returned by this adapter; the function does
    not authenticate an externally supplied/modified receipt's provenance.
    """
    if not _finite(start_s) or not _finite(end_s) or not 0 <= start_s < end_s:
        return False
    if not isinstance(receipt,dict) or receipt.get('schema_version') != SCHEMA_VERSION:
        return False
    clip = receipt.get('clip_interval')
    if not isinstance(clip,dict) or not _finite(clip.get('end_s')) or end_s > clip['end_s']:
        return False
    intervals = receipt.get('observed_coverage_intervals')
    if not isinstance(intervals,list): return False
    for interval in intervals:
        if (not isinstance(interval,dict) or not _finite(interval.get('start_s')) or
                not _finite(interval.get('end_s')) or not 0 <= interval['start_s'] < interval['end_s'] <= clip['end_s']):
            return False
    return any(interval['start_s'] <= start_s < end_s <= interval['end_s'] for interval in intervals)

"""Bounded CFR video selection on the common clip/model presentation grid.

Actual decoded PTS select the nearest unique source frames at j/sample_fps.
The receipt discloses source-frame quantization and two-frame temporal pooling;
it does not establish event-level semantic time precision or support VFR.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path

from research.av_agent.audio_pts_adapter import (
    AudioPTSAdapterError, MAX_VIDEO_FRAMES, _digest, _finite, _frame_time,
    _mapped_streams, _sha256_file, _video_clock,
)
from research.av_agent.semantic_observer import validate_observer_input

SCHEMA_VERSION = 'av-video-pts-grid-0.1'
MAX_CLIP_SECONDS = 2.0
MAX_SAMPLED_FRAMES = 8
MAX_PIXELS = 200704
TEMPORAL_PATCH_SIZE = 2


class VideoPTSSamplerError(ValueError):
    """The bounded immutable clip/model grid cannot be demonstrated."""


def prepare_video_on_clip_grid(bound_input, *, sample_fps=2.0, max_pixels=MAX_PIXELS):
    """Return official-helper float TCHW video, scalar fps, and a private receipt.

    Only an integer, even duration*sample_fps in [2,8] is accepted. The common
    clip axis is preserved: a <=1ms source origin is disclosed, never rebased.
    The official pinned helper receives PIL frames rather than a video path,
    so its endpoint-linspace reader and odd-tail padding cannot select frames.
    """
    import av
    import torch
    from qwen_omni_utils import fetch_video

    if not _finite(sample_fps) or sample_fps <= 0:
        raise VideoPTSSamplerError('known finite positive sample fps required')
    if (isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or
            not 100352 <= max_pixels <= MAX_PIXELS):
        raise VideoPTSSamplerError('integer max_pixels must be in [100352,200704]')
    try:
        bound = json.loads(json.dumps(bound_input, allow_nan=False))
        errors = validate_observer_input(bound, verify_files=True)
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, OSError):
        raise VideoPTSSamplerError('invalid immutable observer binding') from None
    if errors or bound['modalities'] != ['audio', 'video']:
        raise VideoPTSSamplerError('invalid immutable joint AV observer binding')
    duration = (Fraction(str(bound['original_interval']['end_s'])) -
                Fraction(str(bound['original_interval']['start_s'])))
    if not 0 < duration <= Fraction(str(MAX_CLIP_SECONDS)):
        raise VideoPTSSamplerError('clip duration exceeds bounded 2s sampler')
    scalar = Fraction(str(sample_fps))
    sample_count = duration * scalar
    if (sample_count.denominator != 1 or not 2 <= sample_count <= MAX_SAMPLED_FRAMES or
            int(sample_count) % TEMPORAL_PATCH_SIZE):
        raise VideoPTSSamplerError('duration*sample_fps must be an integer even frame count in [2,8]')
    count = int(sample_count)
    targets = [Fraction(i, 1) / scalar for i in range(count)]
    path = Path(bound['clip_path'])
    media_before = _sha256_file(path)
    if media_before != bound['output_sha256']:
        raise VideoPTSSamplerError('media bytes changed before grid preparation')

    try:
        clock = _video_clock(path, bound, duration)
        records = []; times = []; nearest = [None] * count
        with av.open(str(path)) as container:
            stream, metadata, stream_tb = _mapped_streams(bound, container.streams)['video']
            source_fps = Fraction(stream.average_rate)
            if float(source_fps) != clock['verified_cfr_fps']:
                raise VideoPTSSamplerError('source rate changed after CFR proof')
            for frame in container.decode(stream):
                t, tb = _frame_time(frame)
                if (tb != stream_tb or (times and t <= times[-1]) or
                        not 0 <= t < duration or len(times) >= MAX_VIDEO_FRAMES):
                    raise VideoPTSSamplerError('decoded video clock changed after bounded CFR proof')
                index = len(times)
                times.append(t)
                records.append({'index': index, 'pts': frame.pts, 'time_base': str(tb),
                                'timestamp_s': float(t), 'timestamp_rational_s': str(t)})
                # Strict improvement preserves the earlier frame on an exact tie.
                # Keep at most eight native frames; the full decoded video is not buffered.
                for j, target in enumerate(targets):
                    candidate = nearest[j]
                    distance = abs(t - target)
                    if candidate is None or distance < candidate['distance']:
                        nearest[j] = {'index': index, 'time': t, 'distance': distance, 'frame': frame}
        if (len(times) != clock['decoded_frame_count'] or not times or
                float(times[0]) != clock['decoded_first_pts_s'] or
                float(times[-1]) != clock['decoded_last_pts_s'] or
                max(abs(t - (times[0] + Fraction(i, 1) / source_fps))
                    for i, t in enumerate(times)) > stream_tb):
            raise VideoPTSSamplerError('decoded source grid changed after CFR proof')
        indices = [row['index'] for row in nearest]
        if len(set(indices)) != count:
            raise VideoPTSSamplerError('nearest grid frames must be unique; duplication unsupported')
        bound_error = Fraction(1, 2) / source_fps + stream_tb
        errors = [row['time'] - target for row, target in zip(nearest, targets)]
        if any(abs(error) > bound_error for error in errors):
            raise VideoPTSSamplerError('nearest actual PTS exceeds half-source-frame plus container tick')
        images = [row['frame'].to_image().convert('RGB') for row in nearest]
    except AudioPTSAdapterError as exc:
        raise VideoPTSSamplerError(str(exc)) from None
    except (av.FFmpegError, OSError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, VideoPTSSamplerError): raise
        raise VideoPTSSamplerError('actual bounded video PTS decode failed') from None

    if version('qwen-omni-utils') != '0.0.9':
        raise VideoPTSSamplerError('only pinned qwen-omni-utils 0.0.9 resizing is supported')
    resize_module = importlib.import_module(fetch_video.__module__)
    if (fetch_video.__module__ != 'qwen_omni_utils.v2_5.vision_process' or
            resize_module.FRAME_FACTOR != TEMPORAL_PATCH_SIZE):
        raise VideoPTSSamplerError('unsupported official video helper/temporal frame factor')
    try:
        tensor, helper_fps = fetch_video({'video': images, 'sample_fps': float(scalar),
                                          'max_pixels': max_pixels}, return_video_sample_fps=True)
    except (TypeError, ValueError, RuntimeError, AssertionError):
        raise VideoPTSSamplerError('official pinned frame-list resize failed') from None
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or
            tensor.ndim != 4 or tuple(tensor.shape[:2]) != (count, 3) or
            tensor.device.type != 'cpu' or not torch.isfinite(tensor).all() or
            tensor.shape[2] * tensor.shape[3] > max_pixels or
            helper_fps != float(scalar)):
        raise VideoPTSSamplerError('official resize changed frame count, scalar fps, or bounded TCHW contract')
    tensor = tensor.contiguous()
    media_after = _sha256_file(path)
    if media_after != media_before or validate_observer_input(bound, verify_files=True):
        raise VideoPTSSamplerError('media bytes changed during grid preparation')
    output_bytes = tensor.numpy().astype('<f4', copy=False).tobytes()
    actual = [row['time'] for row in nearest]
    receipt = {
        'schema_version': SCHEMA_VERSION, 'input_sha256': bound['input_sha256'],
        'input_binding': bound, 'input_media_sha256_before': media_before,
        'input_media_sha256_after': media_after, 'input_media_sha256': media_after,
        'sampler_module_sha256': _sha256_file(__file__),
        'cfr_proof_module_sha256': _sha256_file(inspect.getfile(_video_clock)),
        'axis': 'clip_local_presentation_pts_seconds',
        'clip_interval': {'start_s': 0.0, 'end_s': float(duration)},
        'sample_fps': float(scalar), 'sample_fps_rational': str(scalar),
        'sampled_frame_count': count, 'video_clock': clock,
        'decoded_video_frames': records, 'selected_indices': indices,
        'target_timestamps_s': [float(t) for t in targets],
        'target_timestamps_rational_s': [str(t) for t in targets],
        'actual_timestamps_s': [float(t) for t in actual],
        'actual_timestamps_rational_s': [str(t) for t in actual],
        'per_frame_error_s': [float(error) for error in errors],
        'per_frame_error_rational_s': [str(error) for error in errors],
        'max_abs_quantization_error_s': float(max(abs(error) for error in errors)),
        'quantization_bound_s': float(bound_error),
        'quantization_bound_rational_s': str(bound_error),
        'quantization_policy': 'nearest_actual_decoded_pts; ties_choose_earlier; unique_frames',
        'quantization_bound_policy': 'half_source_frame_plus_container_tick; exact_rational_comparison',
        'model_temporal_patch_size': TEMPORAL_PATCH_SIZE,
        'model_temporal_patch_timestamps_s': [float(targets[i]) for i in range(0, count, TEMPORAL_PATCH_SIZE)],
        'model_temporal_patch_frame_indices': [list(range(i, i + TEMPORAL_PATCH_SIZE))
                                               for i in range(0, count, TEMPORAL_PATCH_SIZE)],
        'model_temporal_patch_meaning': 'leading model-grid frame timestamp; processor pools two frames; not exact event time',
        'resize': {'implementation': 'official qwen_omni_utils.fetch_video PIL RGB frame-list',
                   'package_version': version('qwen-omni-utils'), 'source_module': fetch_video.__module__,
                   'source_module_sha256': _sha256_file(resize_module.__file__),
                   'source_function_sha256': hashlib.sha256(inspect.getsource(fetch_video).encode()).hexdigest(),
                   'max_pixels': max_pixels, 'pyav_version': av.__version__,
                   'duplicate_tail_frames': 0, 'input_frame_count': count, 'output_frame_count': count},
        'output_tensor_sha256': hashlib.sha256(output_bytes).hexdigest(),
        'output_tensor_shape': list(tensor.shape), 'output_tensor_dtype': 'float32',
        'output_tensor_byte_order': 'little_endian', 'output_tensor_bytes': len(output_bytes),
        'limits': 'bounded decoded CFR only; source-frame quantization and temporal pooling remain; no semantic timing precision claim',
        'semantic_review_status': 'pending',
    }
    receipt['receipt_sha256'] = _digest(receipt)
    return tensor, float(scalar), receipt

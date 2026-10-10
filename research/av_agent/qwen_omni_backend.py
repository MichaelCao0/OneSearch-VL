"""Local-only Qwen2.5-Omni native joint-AV observer.

Runtime imports are lazy: CPU contract tests and CLI help need no Omni install.
Actual inference requires an independently pinned/verified local HF snapshot.
Structured outputs are model candidates; this backend never approves semantic
truth, human review, source identity, or citation support.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time

from research.av_agent.semantic_observer import (
    _finite, _structured_errors, validate_backend_identity, validate_observer_input,
)
from research.av_data.media_clock import sha256_file
from research.av_agent.native_clock_gate import check_native_clip_clocks

MODEL_ID = 'Qwen/Qwen2.5-Omni-7B'
_HASH = re.compile(r'^[0-9a-f]{64}$')
_REVISION = re.compile(r'^[0-9a-f]{40,64}$')


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('non-finite JSON value')


def _json_loads(text):
    return json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)


def _abstention(issue):
    return {'events': [],
            'uncertainty': 'Structured parsing failed; raw model output is retained for review. No event support is accepted.',
            'cannot_determine': ['observer_parse_issue:' + issue]}, issue


def parse_clip_observation(raw_output, bound_input):
    """Strictly parse relative clip intervals, then map to source presentation.

    The model sees a zero-based clip, not source offsets. No clamping/repair is
    allowed: one invalid event abstains the whole candidate structure. The exact
    raw output remains outside this function and is never replaced by the parse.
    """
    text = raw_output.strip()
    # A single fenced JSON block is a presentation wrapper, not text recovery.
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[len('```json\n'):-len('\n```')]
    try:
        value = _json_loads(text)
    except (ValueError, TypeError, RecursionError):
        return _abstention('invalid_json')
    if not isinstance(value, dict) or set(value) != {'events', 'uncertainty', 'cannot_determine'}:
        return _abstention('invalid_fields')
    if not isinstance(value['events'], list) or len(value['events']) > 100:
        return _abstention('invalid_events')
    duration = bound_input['original_interval']['end_s'] - bound_input['original_interval']['start_s']
    mapping = bound_input['mapping']['output_to_source']
    mapped = value
    for event in mapped['events']:
        if not isinstance(event, dict) or set(event) != {'modality', 'description', 'clip_interval', 'uncertainty'}:
            return _abstention('invalid_event_fields')
        interval = event.pop('clip_interval')
        if not isinstance(interval, dict) or set(interval) != {'start_s', 'end_s'}:
            return _abstention('invalid_clip_interval')
        start, end = interval['start_s'], interval['end_s']
        if any(not _finite(x) for x in (start, end)):
            return _abstention('invalid_clip_interval')
        if not 0 <= start < end <= duration:
            return _abstention('event_outside_clip_interval')
        event['source_interval'] = {'start_s': start * mapping['scale'] + mapping['offset_s'],
                                    'end_s': end * mapping['scale'] + mapping['offset_s']}
    if _structured_errors(mapped, bound_input, 'native_joint_av'):
        return _abstention('invalid_observation_content')
    return mapped, None


def _verified_snapshot(model_manifest_path):
    """Freeze manifest bytes and rehash every listed artifact before model load.

    HF LFS expected SHA-256 values come from the downloader's pinned repository
    metadata. Their match is verified here; this is integrity, not authentication
    of a user-supplied manifest. The downloader records the upstream source.
    """
    manifest_path = Path(model_manifest_path)
    blob = manifest_path.read_bytes()
    manifest = _json_loads(blob.decode('utf-8'))
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'av-omni-model-manifest-v1' or
            manifest.get('status') != 'verified' or manifest.get('repository_id') != MODEL_ID):
        raise ValueError('unsupported model repository')
    if not isinstance(manifest.get('revision'), str) or not _REVISION.fullmatch(manifest['revision']):
        raise ValueError('model revision must be immutable')
    raw_path = manifest.get('snapshot_path')
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute() or not Path(raw_path).is_dir():
        raise ValueError('existing absolute local snapshot directory required')
    snapshot = Path(raw_path).resolve()
    if manifest.get('all_lfs_verified') is not True:
        raise ValueError('upstream weight hashes not verified')
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('artifact file manifest required')
    seen = set()
    total = 0
    has_weights = False
    config_sha256 = None
    for item in files:
        if not isinstance(item, dict):
            raise ValueError('invalid artifact record')
        name = item.get('path')
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or name in seen:
            raise ValueError('invalid or duplicate artifact path')
        seen.add(name)
        artifact = snapshot / name
        # HF snapshot files are symlinks to content-addressed blobs; these are
        # intentionally supported, but no path may escape via a relative '..'.
        expected = item.get('expected_sha256')
        actual = item.get('actual_sha256')
        size = item.get('size_bytes')
        if (not isinstance(expected, str) or not _HASH.fullmatch(expected) or actual != expected or
                item.get('verified') is not True or isinstance(size, bool) or not isinstance(size, int) or size < 1 or
                not artifact.is_file() or artifact.stat().st_size != size or sha256_file(artifact) != expected):
            raise ValueError('artifact bytes or hash do not match pinned manifest')
        total += size
        has_weights |= name.endswith('.safetensors')
        if name == 'config.json':
            config_sha256 = expected
    if not has_weights or config_sha256 is None:
        raise ValueError('verified model weights and config required')
    if manifest.get('total_size_bytes') != total:
        raise ValueError('artifact total byte mismatch')
    return snapshot, manifest, config_sha256, hashlib.sha256(blob).hexdigest()


class QwenOmniBackend:
    """Real local shared-clip inference with a defensive immutable identity.

    One attempt generates once (no retries, no provider requests). The processor,
    multimedia helper and model generation all receive use_audio_in_video=True;
    Talker output is disabled at load and return_audio=False at generation.
    Detailed timing/resource costs stay in a private sidecar (last_metrics),
    keeping the existing five-field SemanticObserver reply unchanged.
    """
    def __init__(self, model_manifest_path, *, backend_code_revision, device='cuda:0',
                 dtype='bfloat16', max_new_tokens=256, max_clip_seconds=2.0,
                 video_fps=2.0, max_pixels=256 * 28 * 28):
        if dtype not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('unsupported floating-point dtype')
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 1024:
            raise ValueError('max_new_tokens must be in [1,1024]')
        for value, upper in ((max_clip_seconds, 2.0), (video_fps, 4.0)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError('invalid bounded AV preparation budget')
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or not 28 * 28 <= max_pixels <= 256 * 28 * 28:
            raise ValueError('invalid bounded frame pixels')
        if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device):
            raise ValueError('explicit CPU or CUDA device required')
        started = time.monotonic()
        snapshot, artifact_manifest, config_sha256, manifest_sha256 = _verified_snapshot(model_manifest_path)
        self.model_manifest_sha256 = manifest_sha256
        self.model_artifact_identity = {'repository_id': artifact_manifest['repository_id'],
                                        'revision': artifact_manifest['revision'],
                                        'config_sha256': config_sha256,
                                        'model_manifest_sha256': manifest_sha256,
                                        'verified_artifact_bytes': artifact_manifest['total_size_bytes']}
        self._identity = {'backend_id':'qwen2.5-omni-native-local-v1','model_id':MODEL_ID,
            'model_revision':artifact_manifest['revision'],'config_sha256':config_sha256,
            'mode':'native_joint_av','transport':'local','execution_kind':'model_inference',
            'runtime_device':device,'backend_code_revision':backend_code_revision}
        if validate_backend_identity(self._identity):
            raise ValueError('invalid pinned backend identity')
        self.model_hash_verification_seconds = time.monotonic() - started
        # Import only after local identity/weights/budgets have been checked.
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        self._torch = torch
        self._process_mm_info = process_mm_info
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._max_clip_seconds = max_clip_seconds
        self._video_fps = video_fps
        self._max_pixels = max_pixels
        self._attempts = 0
        self._last_metrics = None
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('requested CUDA is unavailable')
        self._sync()
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        self._processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True,
                                                             trust_remote_code=False)
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(snapshot), torch_dtype=getattr(torch, dtype), device_map={'': device},
            local_files_only=True, trust_remote_code=False, enable_audio_output=False)
        self._model.eval()
        self._sync()
        self.model_load_seconds = time.monotonic() - started
        self.model_load_peak_cuda_memory = self._memory()

    @property
    def manifest(self):
        return deepcopy(self._identity)

    @property
    def last_metrics(self):
        return deepcopy(self._last_metrics)

    def _sync(self):
        if self._device.startswith('cuda'):
            self._torch.cuda.synchronize(self._device)

    def _memory(self):
        if not self._device.startswith('cuda'):
            return {'peak_allocated_bytes':None, 'peak_reserved_bytes':None,
                    'measurement':'not_applicable_cpu'}
        return {'peak_allocated_bytes':self._torch.cuda.max_memory_allocated(self._device),
                'peak_reserved_bytes':self._torch.cuda.max_memory_reserved(self._device),
                'measurement':'torch_cuda_allocator; excludes non-PyTorch allocations'}

    def _messages(self, bound):
        duration = bound['original_interval']['end_s'] - bound['original_interval']['start_s']
        instruction = (
            'Analyze only the audible and visible content of this supplied clip. Describe observations, '
            'uncertainty, and plausible alternatives; do not assert a unique cause or sound source from '
            'co-occurrence alone. Return exactly one JSON object without commentary. The keys are '
            'events (list), uncertainty (nonempty text), cannot_determine (list of text). Each event must '
            'have modality (audio, visual, or joint_av), description (text), clip_interval '
            '({"start_s":number,"end_s":number}), and uncertainty (nonempty text). '
            f'All clip_interval times are relative to the supplied clip, within [0,{duration:.6f}) seconds, '
            'with start_s < end_s. Use an empty events list when support is absent. Do not infer identity '
            'or unseen events. This question is a request, not evidence:\n' + bound['question'])
        return [{'role':'system','content':[{'type':'text','text':
                    'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of '
                    'perceiving auditory and visual inputs, as well as generating text and speech.'}]},
                {'role':'user','content':[{'type':'video','video':bound['clip_path'],
                                          'fps':self._video_fps,'max_pixels':self._max_pixels},
                                         {'type':'text','text':instruction}]}]

    def infer(self, bound_input):
        if validate_observer_input(bound_input, verify_files=True):
            raise ValueError('invalid or changed observer input')
        duration = bound_input['original_interval']['end_s'] - bound_input['original_interval']['start_s']
        if bound_input['modalities'] != ['audio','video'] or duration > self._max_clip_seconds:
            raise ValueError('native AV requires both streams and a bounded shared clip')
        self._attempts += 1
        started = time.monotonic()
        timings = {name:0.0 for name in ('media_preprocess_seconds','processor_seconds','generation_seconds',
                                        'decode_seconds','parse_seconds','total_infer_seconds')}
        metrics = {'schema_version':'av-native-observer-cost-0.1','attempt':self._attempts,
                   'input_sha256':bound_input['input_sha256'],'provider_requests':0,
                   'requested_clip_seconds':duration,'video_fps':self._video_fps,
                   'max_frame_pixels':self._max_pixels,'max_new_tokens':self._max_new_tokens,
                   'input_tokens':None,'generated_tokens':None,'generation_completed':False,
                   'parse_issue':None,'error_type':None,'timings':timings,'peak_cuda_memory':None,
                   'actual_video_fps':None,'do_sample_frames':False,'prepared_video_frames':None}
        stage = 'media_preprocess_seconds'
        stage_start = time.monotonic()
        try:
            self._sync()
            if self._device.startswith('cuda'):
                self._torch.cuda.reset_peak_memory_stats(self._device)
            check_native_clip_clocks(bound_input)
            messages = self._messages(bound_input)
            audio, images, videos, video_kwargs = self._process_mm_info(
                messages, use_audio_in_video=True, return_video_kwargs=True)
            # The helper reports actual sample FPS (minimum frame count and
            # rounding can change it from the requested FPS). Pinned Omni 4.57.6
            # computes temporal grids with a scalar FPS, so unpack one clip.
            if (not isinstance(videos, list) or len(videos) != 1 or
                    not isinstance(video_kwargs, dict) or set(video_kwargs) != {'fps','do_sample_frames'} or
                    video_kwargs['do_sample_frames'] is not False or
                    not isinstance(video_kwargs['fps'], list) or len(video_kwargs['fps']) != 1 or
                    not _finite(video_kwargs['fps'][0]) or video_kwargs['fps'][0] <= 0):
                raise ValueError('one sampled video with explicit valid FPS required')
            actual_fps = float(video_kwargs['fps'][0])
            metrics['actual_video_fps'] = actual_fps
            shape = getattr(videos[0], 'shape', None)
            if shape is not None and len(shape) == 4:
                metrics['prepared_video_frames'] = int(shape[0])
            timings[stage] = time.monotonic() - stage_start
            stage = 'processor_seconds'
            stage_start = time.monotonic()
            text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self._processor(text=text, audio=audio, images=images, videos=videos,
                                     return_tensors='pt', padding=True, use_audio_in_video=True,
                                     fps=actual_fps, do_sample_frames=False)
            inputs = inputs.to(self._model.device).to(self._model.dtype)
            metrics['input_tokens'] = int(inputs['input_ids'].shape[1])
            self._sync()
            timings[stage] = time.monotonic() - stage_start
            stage = 'generation_seconds'
            stage_start = time.monotonic()
            with self._torch.inference_mode():
                generated = self._model.generate(**inputs, use_audio_in_video=True, return_audio=False,
                                                thinker_max_new_tokens=self._max_new_tokens, do_sample=False)
            self._sync()
            timings[stage] = time.monotonic() - stage_start
            metrics['generation_completed'] = True
            stage = 'decode_seconds'
            stage_start = time.monotonic()
            ids = generated.sequences if hasattr(generated, 'sequences') else generated
            # Thinker generation returns the full decoder sequence, including
            # prompt tokens. Retaining only generated tokens avoids treating the
            # schema/example text in our prompt as model-observed events.
            output_ids = ids[:, metrics['input_tokens']:]
            metrics['generated_tokens'] = int(output_ids.shape[1])
            raw = self._processor.batch_decode(output_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
            timings[stage] = time.monotonic() - stage_start
            stage = 'parse_seconds'
            stage_start = time.monotonic()
            structured, metrics['parse_issue'] = parse_clip_observation(raw, bound_input)
            timings[stage] = time.monotonic() - stage_start
            return {'backend_identity':self.manifest,'consumed_input_sha256':bound_input['input_sha256'],
                    'raw_output':raw,'structured_observation':structured,'provider_requests':0}
        except Exception as exc:
            # Preserve elapsed failing-stage work without recording exception
            # text (which may contain private paths or raw media fragments).
            timings[stage] = time.monotonic() - stage_start
            metrics['error_type'] = type(exc).__name__
            raise
        finally:
            self._sync()
            timings['total_infer_seconds'] = time.monotonic() - started
            metrics['peak_cuda_memory'] = self._memory()
            self._last_metrics = metrics

#!/usr/bin/env python3
"""Run at most six fixed, prebound natural AV clips with local verified weights.

All observations, raw outputs, binding paths and detailed costs are private.
Only summary.public.json is a publication candidate, and it makes no semantic
capability claim. CLI help uses standard/CPU modules and never loads a model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

# Direct script execution and python -m both resolve the existing main checkout.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.av_agent.semantic_observer import SemanticObserver, validate_observer_input
from research.av_data.media_clock import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def _utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def load_bindings(path, *, max_clips=6, max_clip_seconds=2.0):
    """Accept the exact minimized observer inputs, not source/task/gold packages."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(payload, dict) or set(payload) != {'schema_version','inputs'} or
            payload['schema_version'] != 'av-observer-smoke-bindings-0.1'):
        raise ValueError('exact smoke input binding schema required')
    inputs = payload['inputs']
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= max_clips:
        raise ValueError('clip count exceeds fixed smoke budget')
    for bound in inputs:
        if validate_observer_input(bound, verify_files=True):
            raise ValueError('invalid or changed clip binding')
        duration = bound['original_interval']['end_s'] - bound['original_interval']['start_s']
        if bound['modalities'] != ['audio','video'] or duration > max_clip_seconds:
            raise ValueError('both AV streams and <=2 second clips required')
    return inputs


def collect_runtime_receipt(identity_manifest_path, lock_path):
    """Record imported code/version identity and fail closed before model load.

    A lock hash describes intended resolution. This receipt identifies the live
    interpreter and imported packages, checks pinned versions and CUDA build,
    and hashes imported module files. Package versions/hashes are provenance,
    not GPU execution or semantic capability evidence.
    """
    modules = {'torch':'torch','torchvision':'torchvision','transformers':'transformers',
               'qwen-omni-utils':'qwen_omni_utils','accelerate':'accelerate',
               'huggingface-hub':'huggingface_hub'}
    receipt = {'schema_version':'av-native-live-runtime-0.1','created_at':_utc(),
               'status':'mismatch','errors':[], 'packages':{},'model_sources':{},
               'expected_manifest_sha256':None,'actual_lock_sha256':None,
               'interpreter':{'executable':sys.executable,'version':sys.version,
                              'python_series':f'{sys.version_info.major}.{sys.version_info.minor}',
                              'prefix':sys.prefix,'platform':sys.platform,'machine':platform.machine()},
               'torch_cuda_build':None,'model_loaded':False,'gpu_inference_verified':False}
    errors=receipt['errors']
    try:
        manifest_bytes=Path(identity_manifest_path).read_bytes()
        receipt['expected_manifest_sha256']=hashlib.sha256(manifest_bytes).hexdigest()
        expected=json.loads(manifest_bytes)
        receipt['actual_lock_sha256']=sha256_file(lock_path)
    except Exception as exc:
        errors.append('runtime_identity_unavailable:'+type(exc).__name__)
        return receipt
    if (not isinstance(expected,dict) or expected.get('schema_version')!='av-observer-runtime-manifest-v1' or
            not isinstance(expected.get('packages'),dict) or set(expected['packages'])!=set(modules) or
            any(not isinstance(value,str) or not value for value in expected['packages'].values())):
        errors.append('invalid_runtime_manifest')
        return receipt
    if expected.get('python_series')!=receipt['interpreter']['python_series']:
        errors.append('python_series_mismatch')
    if expected.get('platform')!='linux-64' or sys.platform!='linux' or platform.machine() not in ('x86_64','AMD64'):
        errors.append('runtime_platform_mismatch')
    if expected.get('lock_sha256')!=receipt['actual_lock_sha256']:
        errors.append('runtime_lock_mismatch')
    for package,module_name in modules.items():
        row={'distribution_version':None,'imported_version':None,'module_path':None,'module_sha256':None,'distribution_module_path':None}
        receipt['packages'][package]=row
        try:
            row['distribution_version']=importlib.metadata.version(package)
            module=importlib.import_module(module_name)
            row['imported_version']=getattr(module,'__version__',None)
            module_file=Path(module.__file__)
            row['module_path']=str(module_file.resolve())
            installed_module=Path(importlib.metadata.distribution(package).locate_file(
                module_name.replace('.','/')+'/__init__.py')).resolve()
            row['distribution_module_path']=str(installed_module)
            if module_file.resolve()!=installed_module:
                errors.append('imported_package_path_mismatch:'+package)
            row['module_sha256']=sha256_file(module_file)
            if package=='torch':
                receipt['torch_cuda_build']=module.version.cuda
        except Exception as exc:
            errors.append('package_identity_unavailable:'+package+':'+type(exc).__name__)
            continue
        if row['distribution_version']!=expected['packages'][package]:
            errors.append('package_version_mismatch:'+package)
        if row['imported_version'] is not None and row['imported_version']!=row['distribution_version']:
            errors.append('imported_package_version_mismatch:'+package)
    if receipt['torch_cuda_build']!=expected.get('cuda_wheel_runtime'):
        errors.append('torch_cuda_build_mismatch')
    # Freeze the actual model/processor and AV decoder helper source identities,
    # not just their top-level packages. Skip these imports on version failure.
    if not errors:
        for name in ('transformers.models.qwen2_5_omni.modeling_qwen2_5_omni',
                     'transformers.models.qwen2_5_omni.processing_qwen2_5_omni',
                     'qwen_omni_utils.v2_5.vision_process','qwen_omni_utils.v2_5.audio_process'):
            try:
                module=importlib.import_module(name)
                module_file=Path(module.__file__)
                receipt['model_sources'][name]={'path':str(module_file.resolve()),'sha256':sha256_file(module_file)}
            except Exception as exc:
                errors.append('model_source_identity_unavailable:'+name+':'+type(exc).__name__)
    receipt['status']='verified' if not errors else 'mismatch'
    return receipt


def _git_head():
    return subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def run(args):
    if not 1 <= args.max_clips <= 6 or not math.isfinite(args.max_clip_seconds) or not 0 < args.max_clip_seconds <= 2:
        raise ValueError('smoke budget can only be reduced below 6 clips x 2 seconds')
    if not 1 <= args.max_new_tokens <= 1024:
        raise ValueError('invalid generation token budget')
    bindings = load_bindings(args.bindings, max_clips=args.max_clips, max_clip_seconds=args.max_clip_seconds)
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=False)
    # Raw/paths/costs are private even on a shared persistent mount.
    destination.chmod(0o700)
    observations_dir = destination / 'observations.private'
    observations_dir.mkdir(mode=0o700)
    started = time.monotonic()
    head = _git_head()
    backend_file = ROOT / 'research/av_agent/qwen_omni_backend.py'
    runtime_manifest = ROOT / 'runtime/av-observer/pixi.toml'
    runtime_lock = ROOT / 'runtime/av-observer/pixi.lock'
    runtime_identity_manifest = ROOT / 'runtime/av-observer/runtime-manifest.json'
    run_identity = {'schema_version':'av-native-observer-run-0.1','created_at':_utc(),
       'source_git_head':head,'backend_code_sha256':sha256_file(backend_file),
       'cli_code_sha256':sha256_file(__file__), 'bindings_sha256':sha256_file(args.bindings),
       'model_manifest_sha256':sha256_file(args.model_manifest) if args.model_manifest.is_file() else None,
       'runtime_identity_manifest_sha256':sha256_file(runtime_identity_manifest) if runtime_identity_manifest.is_file() else None,
       'runtime_manifest_sha256':sha256_file(runtime_manifest) if runtime_manifest.is_file() else None,
       'runtime_lock_sha256':sha256_file(runtime_lock) if runtime_lock.is_file() else None,
       'default_environment_lock_sha256':sha256_file(ROOT/'pixi.lock'),
       'selection_claim':'Caller supplies fixed natural-source bindings; this CLI does not synthesize or adjudicate probes.',
       'binding_count':len(bindings),'budgets':{'max_clips':args.max_clips,'max_clip_seconds':args.max_clip_seconds,
                                             'max_new_tokens':args.max_new_tokens,'video_fps':args.video_fps,
                                             'max_frame_pixels':args.max_pixels},
       'device':args.device,'dtype':args.dtype,'input_bindings':bindings,
       'human_semantic_validation_status':'pending','raw_av_observer_ready':False}
    _write(destination/'run-manifest.private.json', run_identity)
    from research.av_agent.qwen_omni_backend import QwenOmniBackend
    backend = None
    records = []
    costs = []
    load_error = None
    live_receipt=collect_runtime_receipt(runtime_identity_manifest,runtime_lock)
    receipt_path=destination/'live-runtime.private.json'
    _write(receipt_path,live_receipt)
    run_identity['live_runtime_receipt_sha256']=sha256_file(receipt_path)
    run_identity['live_runtime_status']=live_receipt['status']
    _write(destination/'run-manifest.private.json',run_identity)
    try:
        if live_receipt['status']!='verified':
            raise RuntimeError('live runtime does not match immutable expected identity')
        backend = QwenOmniBackend(args.model_manifest, backend_code_revision=head, device=args.device,
                                  dtype=args.dtype, max_new_tokens=args.max_new_tokens,
                                  max_clip_seconds=args.max_clip_seconds, video_fps=args.video_fps,
                                  max_pixels=args.max_pixels)
    except Exception as exc:
        # No input is attempted after a load/identity failure. Preserve the fixed
        # planned denominator and safe class, never the private exception text.
        load_error = type(exc).__name__
    if backend is not None:
        observer = SemanticObserver(backend, max_calls=args.max_clips)
        for index, bound in enumerate(bindings, 1):
            record = observer.observe(bound)
            record_path = observations_dir / f'{index:04d}.json'
            _write(record_path, record)
            cost = backend.last_metrics
            cost['observation_id'] = record['observation_id']
            cost['observation_status'] = record['status']
            records.append(record)
            costs.append(cost)
            # Persist after every attempt so interrupted runs keep completed work.
            with (destination/'costs.private.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(cost, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
        counts = observer.counts
        _write(destination/'model-identity.private.json', {'backend_identity':backend.manifest,
               'artifact_identity':backend.model_artifact_identity,
               'model_hash_verification_seconds':backend.model_hash_verification_seconds,
               'model_load_seconds':backend.model_load_seconds,
               'model_load_peak_cuda_memory':backend.model_load_peak_cuda_memory})
    else:
        counts = {'calls':0,'backend_attempts':0,'failures':0,'provider_requests_known':0,'provider_requests_unknown_attempts':0}
        _write(destination/'load-failure.private.json', {'error_type':load_error,'attempted_clip_count':0,
               'planned_clip_count':len(bindings),'wall_seconds':time.monotonic()-started})
    executed = sum(record['model_executed'] for record in records)
    parse_issues = sum(cost['parse_issue'] is not None for cost in costs)
    failed = sum(record['status']=='failed' for record in records)
    peak_allocated = [cost['peak_cuda_memory']['peak_allocated_bytes'] for cost in costs
                      if cost['peak_cuda_memory']['peak_allocated_bytes'] is not None]
    peak_reserved = [cost['peak_cuda_memory']['peak_reserved_bytes'] for cost in costs
                     if cost['peak_cuda_memory']['peak_reserved_bytes'] is not None]
    if backend is not None:
        for key, values in (('peak_allocated_bytes',peak_allocated),('peak_reserved_bytes',peak_reserved)):
            if backend.model_load_peak_cuda_memory[key] is not None:
                values.append(backend.model_load_peak_cuda_memory[key])
    summary = {'schema_version':'av-native-observer-public-summary-0.1','created_at':_utc(),
       'status':'runtime_mismatch' if live_receipt['status']!='verified' else 'load_failed' if load_error else 'completed_with_failures' if failed else 'completed',
       'planned_clip_count':len(bindings),'attempted_clip_count':counts['backend_attempts'],
       'model_executed_clip_count':executed,'failed_clip_count':failed,
       'blocked_clip_count':len(bindings)-len(records),'structured_parse_issue_count':parse_issues,
       'valid_structured_output_count':sum(r['model_executed'] and c['parse_issue'] is None for r,c in zip(records,costs)),
       'cannot_determine_clip_count':sum(bool((r['structured_observation'] or {}).get('cannot_determine')) for r in records),
       'model_id':'Qwen/Qwen2.5-Omni-7B',
       'model_revision':backend.manifest['model_revision'] if backend else None,
       'config_sha256':backend.manifest['config_sha256'] if backend else None,
       'model_manifest_sha256':run_identity['model_manifest_sha256'],
       'source_git_head':head,'backend_code_sha256':run_identity['backend_code_sha256'],
       'runtime_manifest_sha256':run_identity['runtime_manifest_sha256'],
       'runtime_lock_sha256':run_identity['runtime_lock_sha256'],
       'runtime_identity_manifest_sha256':run_identity['runtime_identity_manifest_sha256'],
       'live_runtime_receipt_sha256':run_identity['live_runtime_receipt_sha256'],
       'live_runtime_status':run_identity['live_runtime_status'],
       'mode':'native_joint_av','transport':'local','actual_provider_requests':counts['provider_requests_known'],
       'requested_av_seconds':sum(x['original_interval']['end_s']-x['original_interval']['start_s'] for x in bindings),
       'peak_cuda_allocated_bytes':max(peak_allocated) if peak_allocated else None,
       'peak_cuda_reserved_bytes':max(peak_reserved) if peak_reserved else None,
       'cuda_memory_scope':'torch CUDA allocator peak; non-PyTorch allocations excluded',
       'model_load_seconds':backend.model_load_seconds if backend else None,
       'model_hash_verification_seconds':backend.model_hash_verification_seconds if backend else None,
       'generation_seconds':sum(x['timings']['generation_seconds'] for x in costs),
       'observer_infer_seconds':sum(x['timings']['total_infer_seconds'] for x in costs),
       'total_wall_seconds':time.monotonic()-started,'budgets':run_identity['budgets'],
       'human_validated_clip_count':0,'human_semantic_validation_status':'pending','raw_av_observer_ready':False,
       'semantic_claim':'Actual model execution and JSON contract checks are reported separately. No independent semantic probe passed here.'}
    _write(destination/'summary.public.json', summary)
    run_identity.update(completed_at=_utc(),status=summary['status'],observer_counts=counts,
                        load_error_type=load_error,total_wall_seconds=summary['total_wall_seconds'])
    _write(destination/'run-manifest.private.json', run_identity)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 1 if load_error or failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings',type=Path,required=True,help='Private exact av-observer-smoke-bindings-0.1 JSON inputs')
    parser.add_argument('--model-manifest',type=Path,required=True,help='Immutable pinned local model artifact manifest')
    parser.add_argument('--output-dir',type=Path,required=True,help='New private persistent run directory; overwrite forbidden')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    parser.add_argument('--max-clips',type=int,default=6)
    parser.add_argument('--max-clip-seconds',type=float,default=2.0)
    parser.add_argument('--max-new-tokens',type=int,default=256)
    parser.add_argument('--video-fps',type=float,default=2.0)
    parser.add_argument('--max-pixels',type=int,default=256*28*28)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({'status':'preflight_failed','error_type':type(exc).__name__,
                          'raw_av_observer_ready':False},sort_keys=True),file=sys.stderr)
        return 2

if __name__ == '__main__':raise SystemExit(main())

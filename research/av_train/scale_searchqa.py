"""Versioned SearchQA batch production and distinct audited-question counting.

Reuse one frozen corpus per authoring batch. All paid work still goes through
teacher_rollouts and the original OneSearch agent loop; audits use original SFT.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import time
import sys

from .native_sft import SCHEMA, digest, export_episodes, file_sha
from .provider_guard import ProviderGuard
from .teacher_rollouts import ROOT, atomic_json, now, prepare_plan, summarize_plan
from .teacher_silver_review import review


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def slice_batch(batch_dir, out_dir, task_ids):
    batch, out = Path(batch_dir).resolve(), Path(out_dir).resolve()
    ids = set(task_ids)
    tasks = [t for t in rows(batch / 'tasks-public.jsonl') if t['task_id'] in ids]
    if not ids or len(ids) != len(task_ids) or {t['task_id'] for t in tasks} != ids:
        raise ValueError('Explicit distinct task subset required')
    clues = {c for t in tasks for c in t['initial_sources']}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'scope.json').write_bytes((batch / 'scope.json').read_bytes())
    (out / 'tasks-public.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks))
    atomic_json(out / 'clue-bindings.private.json', [c for c in read(batch / 'clue-bindings.private.json') if c['clue_id'] in clues])
    atomic_json(out / 'clue-lineage.private.json', [c for c in read(batch / 'clue-lineage.private.json') if c['task_id'] in ids])
    refs = [c for c in rows(batch / 'answer-candidates.private.jsonl') if c['task_id'] in ids]
    (out / 'answer-candidates.private.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in refs))
    atomic_json(out / 'parent-batch.private.json', {'batch_dir': str(batch), 'task_ids': list(task_ids),
        'parent_files_sha256': {name: file_sha(batch / name) for name in ('scope.json', 'tasks-public.jsonl',
            'clue-bindings.private.json', 'clue-lineage.private.json', 'answer-candidates.private.jsonl')}})
    return out


def verified_pool(export_dirs):
    """Deduplicate tasks and recording/question pairs; never count paired controls.

    Counts are derived from hash-bound actual exports. CPU-qualified counts also
    require a matching complete audit of both trajectory and answer-only rows.
    """
    retained, qualified, task_keys = set(), set(), {}
    exports, model_counts = [], {}
    for value in dict.fromkeys(str(Path(d).resolve()) for d in export_dirs):
        data = Path(value)
        spec = read(data / 'summary.json')
        if spec.get('schema_version') != SCHEMA or spec.get('audit_only') or spec.get('source_split') != 'train':
            raise ValueError('Only retained train exports can enter the production pool')
        trajectory, answers, lineage = read(data / 'trajectory.json'), read(data / 'answer_only.json'), read(data / 'lineage.private.json')
        if (not len(trajectory) == len(answers) == len(lineage) == spec['retained_count']
                or digest(trajectory) != spec['trajectory_rows_sha256'] or digest(answers) != spec['answer_rows_sha256']):
            raise ValueError('Export rows changed or disagree with retention count')
        batch = Path(spec['batch_dir'])
        tasks = {t['task_id']: t for t in rows(batch / 'tasks-public.jsonl')}
        parent = {r['task_id']: r for r in read(batch / 'clue-lineage.private.json')} if (batch / 'clue-lineage.private.json').exists() else {}
        audited = False
        expected = {(name, i) for name in ('trajectory', 'answer_only') for i in range(len(lineage))}
        for audit_path in data.parent.glob('*/label-media-audit.private.json'):
            audit = read(audit_path)
            if (audit.get('status') == 'passed' and audit.get('audit_only') is False
                    and audit.get('dataset_summary_sha256') == file_sha(data / 'summary.json')
                    and audit.get('original_sampler_native_video_coverage_verified') is True
                    and {(r['dataset'], r['row_index']) for r in audit.get('rows', [])} == expected):
                audited = True
                break
        for i, item in enumerate(lineage):
            if (item['row_index'] != i or item['split'] != 'train' or digest(trajectory[i]) != item['row_sha256']
                    or digest(answers[i]) != item['answer_row_sha256'] or not item.get('retention_verdict_sha256')
                    or file_sha(item['trajectory_path']) != item['trajectory_sha256']):
                raise ValueError('Export lineage does not match actual retained trajectory')
            tid = item['task_id']
            origin = parent.get(tid, {})
            recording = origin.get('acquisition_source_id') or origin.get('parent_source_id')
            if not recording:
                recording = file_sha(trajectory[i]['videos'][0])
            key = (recording, re.sub(r'\s+', '', tasks[tid]['question']).casefold())
            if tid in task_keys and task_keys[tid] != key:
                raise ValueError('Same task ID now points to a different question or source')
            task_keys[tid] = key
            if key not in retained:
                model = item['teacher_identity']['model_id']
                model_counts[model] = model_counts.get(model, 0) + 1
            retained.add(key)
            if audited:
                qualified.add(key)
        exports.append({'dataset_dir': value, 'retained_rows': len(lineage), 'cpu_audit_passed': audited,
                        'summary_sha256': file_sha(data / 'summary.json')})
    return {'distinct_retained_questions': len(retained), 'cpu_verified_questions': len(qualified),
            'awaiting_cpu_audit_questions': len(retained - qualified), 'export_directories': exports,
            'teacher_models': model_counts, 'answer_only_counted_as_new_questions': False,
            'benchmark_gold': 0, 'gpu_jobs_submitted': 0}


def produce_batch(batch_dir, out_dir, teacher_config, prompt_file, model_manifest, *,
                  prior_exports=(), target=2000, chunk_size=25, workers=3, excluded_task_ids=()):
    """Single pass, bounded chunks; stop on provider or CPU audit failure.

    Failed or uncertain attempts are never requeued. A stopped invocation requires
    explicit inspection and a new versioned run. No background retry policy.
    """
    if not 1 <= chunk_size <= 50 or not 1 <= workers <= 8 or not 1 <= target <= 20000:
        raise ValueError('Explicit target/chunk/worker budget required')
    batch, out = Path(batch_dir).resolve(), Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    blocked = set(excluded_task_ids)
    tasks = [t['task_id'] for t in rows(batch / 'tasks-public.jsonl') if t['task_id'] not in blocked]
    exports = [Path(d).resolve() for d in prior_exports]
    completed, cursor = [], 0
    guard = ProviderGuard(out)
    try:
        while cursor < len(tasks):
            current = verified_pool(exports)
            if current['cpu_verified_questions'] >= target:
                break
            if guard.stopped:
                raise RuntimeError('Teacher service stopped; no automatic replay')
            chosen = tasks[cursor:cursor + min(chunk_size, target - current['cpu_verified_questions'])]
            cursor += len(chosen)
            chunk = out / f'chunk-{len(completed) + 1:04d}'
            chunk.mkdir()
            sliced = slice_batch(batch, chunk / 'batch', chosen)
            plan_dir = chunk / 'candidate-plan'
            manifest = prepare_plan(sliced, plan_dir, teacher_config, prompt_file, candidates_per_task=1,
                shard_size=1, index_dir=batch / 'corpus', max_turns=8, max_new_tokens=4096, temperature=.2)
            atomic_json(out / 'state.json', {'status': 'running', 'stage': 'rollout', 'active_chunk': chunk.name,
                'selected_questions': len(chosen), 'cursor': cursor, 'planned_questions': len(tasks),
                'target': target, 'pool': current, 'at_utc': now()})
            timings = {}
            stage_started = time.perf_counter()
            def worker(shard):
                with (chunk / f"rollout-{shard['shard_id']:04d}.log").open('x') as log:
                    module = 'research.av_train.teacher_telemetry' if os.environ.get('ONESEARCH_TEACHER_PROFILE_DIR') else 'research.av_train.teacher_rollouts'
                    command = [sys.executable, '-m', module]
                    if module.endswith('teacher_telemetry'): command.append('rollout')
                    p = subprocess.run(command + ['run',
                        '--plan-dir', str(plan_dir), '--shard-id', str(shard['shard_id']),
                        '--max-candidates', '1', '--max-api-calls', '8'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                return {'shard_id': shard['shard_id'], 'returncode': p.returncode}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes = [f.result() for f in as_completed([pool.submit(worker, s) for s in manifest['shards']])]
            timings['rollout_s'] = time.perf_counter() - stage_started
            atomic_json(chunk / 'performance.json', timings)
            atomic_json(chunk / 'rollout-summary.json', {'summary': summarize_plan(plan_dir), 'workers': outcomes})
            if any(r['returncode'] for r in outcomes) or guard.stopped or ProviderGuard(plan_dir).stopped:
                raise RuntimeError('Teacher rollout stopped; preserve attempts and pending work')
            stage_started = time.perf_counter()
            previous_stage = os.environ.get('ONESEARCH_TEACHER_PROFILE_STAGE')
            os.environ['ONESEARCH_TEACHER_PROFILE_STAGE'] = 'review'
            try:
                verdict = review(plan_dir, chunk / 'silver-review', workers=workers, max_candidates=len(chosen))
            finally:
                if previous_stage is None: os.environ.pop('ONESEARCH_TEACHER_PROFILE_STAGE', None)
                else: os.environ['ONESEARCH_TEACHER_PROFILE_STAGE'] = previous_stage
                timings['review_s'] = time.perf_counter() - stage_started
                atomic_json(chunk / 'performance.json', timings)
            if guard.stopped or verdict.get('provider_stopped'):
                raise RuntimeError('Teacher review stopped; preserve all attempts')
            exported = {'retained_count': 0}
            if verdict['retained_silver_candidates']:
                data = chunk / 'silver-sft'
                exported = export_episodes(plan_dir / 'episodes', data,
                    verdicts=read(chunk / 'silver-review/verdicts.private.json'))
                if exported['retained_count']:
                    atomic_json(out / 'state.json', {'status': 'running', 'stage': 'original_trainer_cpu_audit',
                        'active_chunk': chunk.name, 'pool': verified_pool(exports + [data]), 'target': target, 'at_utc': now()})
                    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
                    stage_started = time.perf_counter()
                    with (chunk / 'cpu-audit.log').open('x') as log:
                        p = subprocess.run(['pixi', 'run', '--locked', '--manifest-path', str(ROOT / 'runtime/sft-gpu/pixi.toml'),
                            'python', '-m', 'research.av_train.native_sft', 'audit', '--dataset-dir', str(data),
                            '--model-manifest', str(model_manifest), '--out-dir', str(chunk / 'silver-cpu-audit')],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    timings['cpu_audit_s'] = time.perf_counter() - stage_started
                    atomic_json(chunk / 'performance.json', timings)
                    if p.returncode:
                        raise RuntimeError('Original trainer CPU audit failed; do not count these rows toward target')
                    exports.append(data)
            completed.append({'chunk': chunk.name, 'questions': len(chosen), 'review': verdict, 'export': exported})
            atomic_json(chunk / 'summary.json', completed[-1])
            atomic_json(out / 'chunks.json', completed)
            atomic_json(out / 'state.json', {'status': 'running', 'stage': 'chunk_completed', 'target': target,
                'pool': verified_pool(exports), 'cursor': cursor, 'planned_questions': len(tasks), 'at_utc': now()})
        result = {'status': 'completed', 'stage': 'single_pass_finished', 'target': target,
            'pool': verified_pool(exports), 'attempted_question_slots': cursor, 'remaining_unattempted': len(tasks) - cursor,
            'chunks': completed, 'at_utc': now(), 'automatic_retries': 0}
        atomic_json(out / 'state.json', result)
        return result
    except Exception as exc:
        atomic_json(out / 'stopped.json', {'status': 'stopped', 'error_type': type(exc).__name__,
            'completed_chunks': len(completed), 'attempted_question_slots': cursor, 'at_utc': now(), 'automatic_retries': 0})
        raise

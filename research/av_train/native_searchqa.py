"""Native Omni AV search episodes through the original OneSearch agent loop."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'onesearch_vl'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def prepare_cases(batch_dir, task_ids=None):
    """Bind anonymous media privately; no reference answers enter actor inputs."""
    batch = Path(batch_dir)
    tasks = read_rows(batch / 'tasks-public.jsonl')
    bindings = {x['clue_id']: x['source'] for x in json.loads((batch / 'clue-bindings.private.json').read_text())}
    if task_ids:
        if set(task_ids) - {t['task_id'] for t in tasks}:
            raise ValueError('Unknown task ID')
        tasks = [t for t in tasks if t['task_id'] in task_ids]
    cases = []
    for task in tasks:
        if len(task['initial_sources']) != 1:
            raise ValueError('Initial pilot requires one anonymous AV clue')
        source = bindings[task['initial_sources'][0]]
        if sha256(source['asset_path']) != source['asset_sha256']:
            raise ValueError('Anonymous input bytes changed')
        cases.append({'case_id': task['task_id'], 'question': task['question'],
                      'source_path': source['asset_path'], 'source_sha256': source['asset_sha256']})
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', required=True)
    p.add_argument('--index-dir')
    p.add_argument('--model-manifest', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--task-id', action='append')
    p.add_argument('--max-tasks', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-turns', type=int, default=6)
    p.add_argument('--temperature', type=float, default=0.)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--direct', action='store_true')
    p.add_argument('--prompt-file', help='Versioned experimental prompt; no task answers or private hints')
    a = p.parse_args()
    if not 1 <= a.max_tasks <= 256 or not 1 <= a.max_turns <= 12:
        raise ValueError('Invalid task/turn count')
    batch = Path(a.batch_dir)
    cases = prepare_cases(batch, a.task_id)[:a.max_tasks]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'inputs.private.json', {'cases': cases,
        'batch_dir': str(batch), 'search_mode': 'direct' if a.direct else 'frozen',
        'private_answers_in_actor': False, 'input_mode': 'native_audio_video'})
    if a.prepare_only:
        print(json.dumps({'prepared': len(cases), 'out_dir': str(out)}), flush=True)
        return
    from onesearch_infer.runners import build_runner, InferenceConfig
    from onesearch_infer.pipeline import process_native_av_case
    from onesearch_infer.search_backends import FrozenSearchBackend
    from onesearch_infer.prompts import get_system_prompt
    artifact = json.loads(Path(a.model_manifest).read_text())
    snapshot = Path(artifact['snapshot_path'])
    os.environ['ONESEARCH_OMNI_MODEL_MANIFEST'] = str(Path(a.model_manifest).resolve())
    os.environ['AV_AGENT_MAX_TURNS'] = str(a.max_turns)
    os.environ['ONESEARCH_DISABLE_RETRIEVAL'] = '0'
    scope = json.loads((batch / 'scope.json').read_text())
    backend = None if a.direct else FrozenSearchBackend(a.index_dir or batch / 'corpus', scope)
    runner = build_runner('omni-7b', checkpoint=str(snapshot))
    runner.load()
    if a.direct and a.prompt_file:
        raise ValueError('Direct baseline and search prompt override are mutually exclusive')
    custom_prompt = Path(a.prompt_file).read_text() if a.prompt_file else None
    results = []
    started = time.monotonic()
    write_json(out / 'model-identity.json', {'model': artifact['repository_id'], 'revision': artifact['revision'],
        'input_mode': 'native_audio_video', 'agent_loop': 'onesearch_infer.pipeline.run_agent_loop',
        'search_mode': 'direct' if a.direct else 'frozen', 'max_turns': a.max_turns,
        'max_new_tokens': a.max_new_tokens, 'temperature': a.temperature,
        'prompt_file': a.prompt_file, 'prompt_sha256': sha256(a.prompt_file) if a.prompt_file else None})
    for idx, case in enumerate(cases):
        folder = out / case['case_id']
        folder.mkdir()
        before = time.monotonic()
        try:
            system = custom_prompt or get_system_prompt('audio_video')
            if a.direct:
                system = ('Answer the question from the supplied audio and video. Do not call tools. '
                          'Return your answer in <response>...</response>.')
            trajectory = process_native_av_case(question=case['question'], source_path=case['source_path'],
                runner=runner, output_dir=str(folder), case_id=case['case_id'], case_idx=idx,
                search_backend=backend, inference_cfg=InferenceConfig(max_tokens=a.max_new_tokens,
                    temperature=a.temperature), system_instruction=system)
            tool_calls = []
            from onesearch_infer.tools import extract_tool_call
            for turn in trajectory['turns']:
                call = extract_tool_call(turn.get('response_text', ''))
                if call:
                    parsed = json.loads(call) if isinstance(call, str) else call
                    tool_calls.append(parsed.get('name'))
            result = {'case_id': case['case_id'], 'status': trajectory.get('status'),
                'final_answer': trajectory['final_response_text'], 'tool_calls': tool_calls,
                'turns': len(trajectory['turns']),
                'protocol_final_response': bool(re.search(r'<response>.+?</response>', trajectory['final_response_text'], re.S)),
                'wall_seconds': time.monotonic() - before}
        except Exception as exc:
            result = {'case_id': case['case_id'], 'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                      'wall_seconds': time.monotonic() - before}
            write_json(folder / 'failure.json', result)
        results.append(result)
        write_json(out / 'summary.json', {'input_mode': 'native_audio_video', 'planned': len(cases),
            'completed': len(results), 'results': results, 'wall_seconds': time.monotonic()-started,
            'formal_benchmark': False, 'optimizer_updates': 0})
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

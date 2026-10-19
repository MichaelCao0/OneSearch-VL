"""Bounded online Search/Fetch probe and native OneSearch AV entrypoint."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import types
from .deepseek_backend import DeepSeekSearchBackend, atomic_json, utcnow

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['probe','run'])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--key-file', default='/home/jovyan/.config/onesearch-av/deepseek-search.env')
    p.add_argument('--max-search-requests', type=int, default=8)
    p.add_argument('--query', action='append')
    p.add_argument('--fetch-url', action='append')
    p.add_argument('--cases-json', help='List of case_id/question/source_path/source_sha256 only')
    p.add_argument('--model-manifest')
    p.add_argument('--full-checkpoint')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--need-content', action=argparse.BooleanOptionalAction, default=True,
                   help='Request Search page bodies as well as summaries; Fetch remains available.')
    p.add_argument('--prompt-file', help='Use the exact frozen evaluation prompt when comparing backends.')
    p.add_argument('--modality', choices=['native_av','muted_video'], default='native_av')
    p.add_argument('--max-turns', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=512)
    a = p.parse_args()
    if not 1 <= a.max_turns <= 12 or not 1 <= a.max_new_tokens <= 4096:
        raise ValueError('Invalid inference budget')
    out = Path(a.out_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    backend = DeepSeekSearchBackend(out/'search', key_file=a.key_file, max_requests=a.max_search_requests, need_content=a.need_content)
    if a.action == 'probe':
        results = []
        for query in a.query or ['OneSearch-VL multimodal search agent']:
            before = time.monotonic(); body, receipt = backend.search(query, 5)
            results.append({'endpoint':'search','query':query,'result_count':len(body['results']),
                            'urls':[h['url'] for h in body['results']],
                            'wall_seconds':time.monotonic()-before,'receipt_id':receipt['request_id']})
        for url in a.fetch_url or []:
            before = time.monotonic(); body, receipt = backend.fetch(url)
            results.append({'endpoint':'fetch','url':url,'content_characters':len(body['content']),
                            'wall_seconds':time.monotonic()-before,'receipt_id':receipt['request_id']})
        atomic_json(out/'probe.json', {'at_utc':utcnow(),'results':results,'summary':backend.summary()})
        print(json.dumps({'results':results,'summary':backend.summary()},ensure_ascii=False))
        return
    if not a.cases_json or not a.model_manifest:
        raise ValueError('run requires cases-json and model-manifest')
    cases = json.loads(Path(a.cases_json).read_text())
    if not isinstance(cases,list) or not 1 <= len(cases) <= 200:
        raise ValueError('Expected 1-200 cases')
    allowed = {'case_id','question','source_path','source_sha256'}
    for case in cases:
        if set(case) != allowed:
            raise ValueError('Actor case schema must exclude answers and reference URLs')
        if hashlib.sha256(Path(case['source_path']).read_bytes()).hexdigest() != case['source_sha256']:
            raise ValueError('Input media changed')
    manifest = json.loads(Path(a.model_manifest).read_text())
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    atomic_json(out/'inputs.private.json',cases)
    atomic_json(out/'preparation.json', {'prepared':True,'cases':len(cases),
                'modality':a.modality,'model_revision':manifest['revision'],
                'model_loaded':False,'model_requests':0,'search_requests':0,
                'prepare_only':a.prepare_only,'need_content':a.need_content,'max_search_requests':a.max_search_requests,
                'max_turns':a.max_turns,'max_new_tokens':a.max_new_tokens,
                'source_file_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if a.prepare_only:
        print(json.dumps({'prepared':len(cases),'modality':a.modality,'model_requests':0,'search_requests':0}))
        return
    sys.path.insert(0,str(ROOT/'onesearch_vl'))
    from onesearch_infer.runners import build_runner, InferenceConfig
    from onesearch_infer.pipeline import process_native_av_case
    from onesearch_infer.prompts import get_system_prompt
    manifest = json.loads(Path(a.model_manifest).read_text())
    os.environ.update(ONESEARCH_OMNI_MODEL_MANIFEST=str(Path(a.model_manifest).resolve()),
                      ONESEARCH_DISABLE_RETRIEVAL='0', AV_AGENT_MAX_TURNS=str(a.max_turns))
    runner = build_runner('omni-7b',checkpoint=manifest['snapshot_path']); runner.load()
    checkpoint_identity = None
    if a.full_checkpoint:
        from research.av_train.full_checkpoint import attach_full_thinker
        checkpoint_identity = attach_full_thinker(runner,a.full_checkpoint)
    # Only the native AV tool set is advertised; there is no reverse image API.
    prompt = Path(a.prompt_file).read_text() if a.prompt_file else get_system_prompt('audio_video') + (
        '\nOnline text_search q must contain 1-128 Unicode characters. '
        'Search returns web summaries and visit_webpage returns Markdown.\n')
    mute = None
    if a.modality == 'muted_video':
        from research.av_train.modality_inputs import MutedMedia
        mute = MutedMedia(out/'muted-media')
        original_infer = runner.infer
        def infer(self, contents, system_instruction=None, cfg=None):
            result = original_infer(mute.contents(contents), system_instruction=system_instruction, cfg=cfg)
            assert self.last_metrics['audio_count'] == 0
            assert not any(self.last_metrics['video_has_audio'])
            return result
        runner.infer = types.MethodType(infer,runner)
    atomic_json(out/'identity.json', {'at_utc':utcnow(),'model_revision':manifest['revision'],
                'checkpoint':checkpoint_identity,'search_mode':'deepseek_online','need_content':a.need_content,'modality':a.modality,
                'agent_loop':'onesearch_infer.pipeline.run_agent_loop',
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'max_turns':a.max_turns,'max_new_tokens':a.max_new_tokens,
                'formal_benchmark':False,'reference_answers_in_actor':False})
    results = []
    for i, case in enumerate(cases):
        trace = process_native_av_case(question=case['question'],source_path=mute.video(case['source_path']) if mute else case['source_path'],runner=runner,
                    output_dir=str(out/case['case_id']),case_id=case['case_id'],case_idx=i,search_backend=backend,
                    inference_cfg=InferenceConfig(max_tokens=a.max_new_tokens,temperature=0.),system_instruction=prompt)
        results.append({'case_id':case['case_id'],'status':trace['status'],
                        'final_response':trace['final_response_text'],'turns':len(trace['turns'])})
        atomic_json(out/'summary.json',{'at_utc':utcnow(),'completed':len(results),'planned':len(cases),
                                      'results':results,'search':backend.summary(),'semantic_review':'pending'})


if __name__ == '__main__': main()

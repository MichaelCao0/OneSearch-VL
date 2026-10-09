"""Deterministic infrastructure replay; this is not a teacher/agent experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from research.av_data.audit import export_actor_task
from research.av_data.validate import validate_source, validate_derived
from research.av_search.corpus import visible_corpus
from .adapter import EpisodeTools, strict_json


def run_pilot(package_dir, out_dir):
    package=Path(package_dir).resolve();out=Path(out_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    records=[strict_json(x, max_chars=16*1024*1024) for x in (package/'sources.jsonl').read_text().splitlines() if x]
    registry={r['source_id']:r for r in records}
    scope=strict_json((package/'scope.json').read_text())
    tasks=[strict_json(x) for x in (package/'tasks-public.jsonl').read_text().splitlines() if x]
    visible=visible_corpus(package/'corpus',scope)
    started=time.monotonic();cases=[]
    for task in tasks:
        sid=task['initial_sources'][0];record=registry[sid]
        export_actor_task(task,visible)
        tools=EpisodeTools(package/'corpus',registry,scope,out/task['task_id'],max_calls=6,max_av_seconds=3)
        # Tool path exercise: intervals selected from actual original stream clocks.
        start=max(0.0,max(s['source_offset_s'] for s in record['stream_specs']))+0.1
        end=min(record['duration_s'],start+2.0)
        title=record['public_metadata']['title']
        query=' '.join(re.findall(r'[^\W_]+',title,flags=re.UNICODE)[:8])
        actions=[{'name':'search_sources','arguments':{'query':query,'top_k':20}},
                 {'name':'provenance','arguments':{'source_id':sid}},
                 {'name':'inspect_av','arguments':{'source_id':sid,'start_s':start,'end_s':end}}]
        receipts=[tools.dispatch(json.dumps(a,ensure_ascii=False)) for a in actions]
        found=[r['source_id'] for r in receipts[0].get('results',[])][:2]
        receipts.append(tools.dispatch(json.dumps({'name':'compare_sources','arguments':{'source_ids':found or [sid]}})))
        actor_payload=json.dumps(receipts,ensure_ascii=False)
        if 'asset_path' in actor_payload or 'output_path' in actor_payload or '/home/jovyan' in actor_payload or 'private_gold' in actor_payload:
            raise ValueError('Internal path/annotation leaked into actor receipts')
        trusted=[strict_json(x, max_chars=16*1024*1024) for x in (out/task['task_id']/'events.jsonl').read_text().splitlines()]
        derived=[x['private_derived_record'] for x in trusted if x['private_derived_record']]
        integrity_errors=validate_source(record)
        for d in derived:
            integrity_errors+=validate_derived(d,record)
        searches=receipts[0].get('results',[])
        cases.append({'task_id':task['task_id'],'source_id':sid,
            'query':query,'seed_returned_by_search':sid in [r['source_id'] for r in searches],
            'retrieved_source_count':len(searches),'requested_interval':{'start_s':start,'end_s':end},
            'clip_status':receipts[2]['status'],'output_sha256':receipts[2].get('output_sha256'),
            'integrity_errors':integrity_errors,'costs':tools.costs,
            'semantic_review':'not_performed','actor_type':'deterministic_script'})
        (out/task['task_id']/'costs.json').write_text(json.dumps(tools.costs,indent=2))
    summary={'schema_version':'av-infrastructure-pilot-0.1','created_at':datetime.now(timezone.utc).isoformat(),
        'actor_type':'deterministic_script','case_count':len(cases),
        'search_seed_recall_count':sum(c['seed_returned_by_search'] for c in cases),
        'media_clip_success_count':sum(c['clip_status']=='media_ready_observer_unavailable' for c in cases),
        'integrity_pass_count':sum(not c['integrity_errors'] for c in cases),
        'semantic_success_count':None,'human_gold_count':0,'teacher_episodes':0,
        'G0':'pending','G1':'not_started','method_experiments':0,
        'raw_av_observer_ready':False,'wall_seconds':time.monotonic()-started,
        'totals':{k:sum(c['costs'][k] for c in cases) for k in cases[0]['costs']},
        'search_interpretation':'query uses public title; infrastructure lookup only, not research task retrieval evaluation',
        'cases':cases}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-dir',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();summary=run_pilot(a.package_dir,a.out_dir)
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

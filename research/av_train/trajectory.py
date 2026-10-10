"""Content-checked tool execution records. Hash integrity alone does not prove execution.

Retention requires independent tool replay in export.py and a semantic verdict.
Only nondeterministic runtime timing is excluded from replay identity.
"""
from __future__ import annotations
import hashlib
import json
from research.av_bench.contracts import json_copy,validate_task,nonempty

TOOLS={'search_sources','inspect_av','read_text'}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def stable_observation(observation):
    value=json_copy(observation)
    def clean(item):
        if isinstance(item,dict):return {k:clean(v) for k,v in item.items() if k not in {'wall_seconds','wall_time_seconds','elapsed_seconds','created_at','fetched_at'}}
        if isinstance(item,list):return [clean(x) for x in item]
        return item
    return clean(value)


def source_ids(value):
    out=set()
    def visit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='source_id':out.add(v)
                elif k in {'initial_sources','ranked_sources','referenced_source_ids'} and isinstance(v,list):out.update(v)
                else:visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(value)
    return out


def make_step(action,observation):
    a=json_copy(action);o=json_copy(observation)
    if not isinstance(a,dict) or set(a)!={'tool','arguments'} or a['tool'] not in TOOLS or not isinstance(a['arguments'],dict):raise ValueError('Malformed action')
    if not isinstance(o,dict) or 'status' not in o:raise ValueError('Executed observation requires status')
    return {'action':a,'observation':o,'observation_sha256':digest(o),'replay_sha256':digest(stable_observation(o))}


def validate_trace(trace,source_splits):
    t=json_copy(trace)
    fields={'schema_version','task','teacher','system','tools','steps','final','execution_kind','referenced_source_ids'}
    if not isinstance(t,dict) or set(t)!=fields or t['schema_version']!='av-research-trajectory.v1':raise ValueError('Malformed trace')
    if t['task'].get('split')!='train':raise ValueError('Training split required')
    validate_task(t['task'],{'corpus_id':t['task']['corpus_id'],'split':'train','source_ids':[s for s,p in source_splits.items() if p=='train']})
    if t['execution_kind']!='live_tools' or t['teacher'].get('kind')!='model':raise ValueError('Executed model teacher required; fixtures cannot train')
    for k in ('model','revision'):nonempty(t['teacher'].get(k),'teacher '+k)
    if not isinstance(t['tools'],list) or not isinstance(t['steps'],list) or not t['steps']:raise ValueError('Multi-turn tool trajectory required')
    nonempty(t['system'],'system')
    touched=source_ids(t)
    if any(source_splits.get(s)!='train' for s in touched):raise ValueError('Foreign/dev/test source split in training trace')
    for step in t['steps']:
        if set(step)!={'action','observation','observation_sha256','replay_sha256'}:raise ValueError('Malformed step')
        rebuilt=make_step(step['action'],step['observation'])
        if rebuilt!=step:raise ValueError('Observation hash mismatch')
    if not isinstance(t['final'],dict) or t['final'].get('status')!='ok':raise ValueError('Successful final submission required')
    return t

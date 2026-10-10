"""Closed actor tasks and separate human-adjudicated references for v2 research."""
from __future__ import annotations
import json
import math
from research.av_data.tasks import _opaque, reject_private_fields

TASK_VERSION='av-researchbench.task.v1'
REFERENCE_VERSION='av-researchbench.reference.v1'
FAMILIES=('source_discovery','event_localization','cross_source_comparison','research_composition')
TASK_FIELDS={'schema_version','task_id','question','initial_sources','target_questions','corpus_id','split','budget_recipe'}

def json_copy(value):
    return json.loads(json.dumps(value,allow_nan=False,ensure_ascii=False))

def number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(name+' must be finite numeric')
    return value

def nonempty(value,name):
    if not isinstance(value,str) or not value.strip():raise ValueError(name+' must be nonempty string')
    return value

def validate_task(task,scope):
    t=json_copy(task)
    if not isinstance(t,dict) or set(t)!=TASK_FIELDS or t['schema_version']!=TASK_VERSION:
        raise ValueError('Invalid closed benchmark task')
    reject_private_fields(t)
    for key in ('task_id','question','corpus_id'):nonempty(t[key],key)
    if t['split'] not in {'train','dev','test','train_validation'} or t['split']!=scope['split'] or t['corpus_id']!=scope['corpus_id']:
        raise ValueError('Task scope/split mismatch')
    allowed=set(scope['source_ids'])
    for sid in allowed:_opaque(sid)
    initial=t['initial_sources']
    if not isinstance(initial,list) or len(initial)>2 or len(initial)!=len(set(initial)) or not set(initial)<=allowed:
        raise ValueError('Invalid initial sources')
    targets=t['target_questions']
    if not isinstance(targets,list) or not 1<=len(targets)<=6:raise ValueError('Require 1..6 target questions')
    ids=[]
    for target in targets:
        if not isinstance(target,dict) or set(target)!={'fact_id','question'}:raise ValueError('Invalid public target')
        ids.append(nonempty(target['fact_id'],'fact_id'));nonempty(target['question'],'question')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate fact_id')
    b=t['budget_recipe']
    if not isinstance(b,dict) or set(b)!={'max_tool_calls','max_requested_av_seconds'}:raise ValueError('Invalid budget')
    if isinstance(b['max_tool_calls'],bool) or not isinstance(b['max_tool_calls'],int) or b['max_tool_calls']<1:raise ValueError('Invalid tool budget')
    if number(b['max_requested_av_seconds'],'AV budget')<=0:raise ValueError('Invalid AV budget')
    return t

def validate_evidence(item,sources):
    if not isinstance(item,dict):raise ValueError('Evidence must be object')
    allowed={'fact_id','source_id','start_s','end_s','event_id','start_char','end_char','quote'}
    if set(item)-allowed or not {'fact_id','source_id'}<=set(item):raise ValueError('Malformed evidence')
    nonempty(item['fact_id'],'fact_id');sid=_opaque(item['source_id'])
    if sid not in sources:raise ValueError('Evidence outside corpus')
    source=sources[sid]
    if source['media_type']=='webpage':
        if not {'start_char','end_char','quote'}<=set(item) or set(item)&{'start_s','end_s','event_id'}:raise ValueError('Text evidence needs quote coordinates')
        a,b=item['start_char'],item['end_char'];body=source.get('body_text',source.get('text',''))
        if any(isinstance(x,bool) or not isinstance(x,int) for x in (a,b)) or not 0<=a<b<=len(body) or body[a:b]!=item['quote']:
            raise ValueError('Text evidence must match frozen quote')
    else:
        if not {'start_s','end_s'}<=set(item) or set(item)&{'start_char','end_char','quote'}:raise ValueError('Timed evidence needs interval')
        a=number(item['start_s'],'start_s');b=number(item['end_s'],'end_s')
        if not 0<=a<b<=number(source['duration_s'],'duration_s'):raise ValueError('Evidence interval outside source')
        if 'event_id' in item:nonempty(item['event_id'],'event_id')
    return item

def validate_reference(reference,task,sources,allow_pending=False):
    r=json_copy(reference)
    fields={'schema_version','task_id','status','reviewer_ids','facts','evidence_packages','family'}
    if not isinstance(r,dict) or set(r)!=fields or r['schema_version']!=REFERENCE_VERSION or r['task_id']!=task['task_id'] or r['family'] not in FAMILIES:
        raise ValueError('Invalid reference identity/schema')
    if allow_pending and r['status']=='pending' and r['facts']==[] and r['evidence_packages']==[] and r['reviewer_ids']==[]:return r
    if r['status']!='adjudicated':raise ValueError('Reference must be adjudicated before scoring')
    reviewers=r['reviewer_ids']
    if not isinstance(reviewers,list) or len(set(reviewers))<2 or any(not isinstance(x,str) or not x.strip() for x in reviewers):raise ValueError('Two independent reviewer identities required')
    expected={x['fact_id'] for x in task['target_questions']}
    facts=r['facts']
    if not isinstance(facts,list) or len(facts)!=len(expected) or {x.get('fact_id') for x in facts}!=expected:raise ValueError('Reference facts must match necessary public targets')
    for fact in facts:
        if set(fact)!={'fact_id','accepted_values'} or not isinstance(fact['accepted_values'],list) or not fact['accepted_values']:
            raise ValueError('Every fact needs accepted values')
        for value in fact['accepted_values']:
            if isinstance(value,str):nonempty(value,'accepted value')
            elif isinstance(value,(int,float)) and not isinstance(value,bool):number(value,'accepted value')
            else:raise ValueError('Accepted value must be scalar string/number')
    packages=r['evidence_packages']
    if not isinstance(packages,list) or not packages:raise ValueError('Require at least one complete evidence package')
    for package in packages:
        if not isinstance(package,list) or not package:raise ValueError('Empty evidence package')
        for item in package:validate_evidence(item,sources)
        if {x['fact_id'] for x in package}!=expected:raise ValueError('Every alternative package must support every necessary fact')
        if len({json.dumps(x,sort_keys=True) for x in package})!=len(package):raise ValueError('Duplicate package evidence')
    return r

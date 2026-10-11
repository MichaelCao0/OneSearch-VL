"""Explicit silver seed export for publisher-attribute QA, separate from gold export.

String matches only check requested publisher facts against actually read text.
They do not validate every assertion, AV semantics, source independence or gold.
"""
from __future__ import annotations
import json
import unicodedata
from .trajectory import validate_trace,digest
from .export import public_observation

def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',str(text)).casefold() if c.isalnum())

def publisher_silver_row(trace,source_splits,required_terms):
    t=validate_trace(trace,source_splits)
    if not isinstance(required_terms,list) or not required_terms or any(not isinstance(v,str) or not v.strip() for v in required_terms):
        raise ValueError('Explicit required publisher terms needed')
    answer=t['final'].get('answer','')
    if not isinstance(answer,str) or not all(normalized(term) in normalized(answer) for term in required_terms):
        raise ValueError('Final answer misses required publisher facts')
    read=[]
    av=False
    for step in t['steps']:
        o=step['observation']
        if step['action']['tool']=='read_text' and o.get('status')=='ok':read.append(o.get('quote',{}).get('text',''))
        if step['action']['tool']=='search_sources' and o.get('status')=='ok':read.extend(r.get('excerpt','') for r in o.get('results',[]))
        if step['action']['tool']=='inspect_av' and o.get('status')=='ok' and o.get('model_executed') is True:av=True
    if not av:raise ValueError('Executed AV observation required')
    if not any(all(normalized(term) in normalized(text) for term in required_terms) for text in read):
        raise ValueError('No observed publisher text supports requested facts')
    turns=[{'from':'human','value':json.dumps(t['task'],ensure_ascii=False)}]
    for step in t['steps']:
        turns.extend([{'from':'gpt','value':json.dumps({'action':step['action']},ensure_ascii=False)},
                      {'from':'observation','value':json.dumps(public_observation(step['observation']),ensure_ascii=False)}])
    turns.append({'from':'gpt','value':json.dumps({'final':{k:v for k,v in t['final'].items() if k!='cost'}},ensure_ascii=False)})
    from .searchqa import QA_SYSTEM
    row={'system':QA_SYSTEM+'\nTools: '+json.dumps(t['tools'],ensure_ascii=False),'tools':'','conversations':turns}
    note={'quality_label':'silver_publisher_fact_match','semantic_verified':False,'benchmark_gold':False,
      'task_id':t['task']['task_id'],'trace_sha256':digest(t),'teacher':t['teacher'],'required_terms':required_terms,
      'independent_replay_verified':False,'source_family_review':'pending','scope':'engineering_seed_only',
      'target_protocol':'canonical_json_action_and_final; teacher used native protocol, payloads preserved',
      'limitation':'Checks requested publisher terms only; full answer and AV semantics require sampled review.'}
    return row,note

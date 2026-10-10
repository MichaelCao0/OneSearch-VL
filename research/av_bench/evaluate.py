"""Package-aware exact-fact scoring. Free-form semantic judging is a separate review.

A package is an AND of supporting evidence entries; packages are OR alternatives.
Source ranking uses actor final order, deduplicated before k, never tool result order.
An event_id, if supplied by annotation, must be a public ordinal/instance value,
not an arbitrary private identifier the actor cannot infer.
"""
from __future__ import annotations
import json
import unicodedata
from .contracts import validate_task,validate_reference,validate_evidence,json_copy,number

METRICS=('complete_package_source_recall_at_k','best_valid_package_coverage_at_k',
         'package_completion_reciprocal_rank','moment_package_coverage_at_k',
         'necessary_fact_accuracy','all_facts_correct','ebs','budget_compliant')

def interval_iou(a,b):
    overlap=max(0,min(a['end_s'],b['end_s'])-max(a['start_s'],b['start_s']))
    union=max(a['end_s'],b['end_s'])-min(a['start_s'],b['start_s'])
    return overlap/union if union>0 else 0

def unique(values):
    out=[];seen=set()
    for value in values:
        key=json.dumps(value,sort_keys=True,allow_nan=False)
        if key not in seen:seen.add(key);out.append(value)
    return out

def matches(pred,ref,threshold,require_fact=True):
    if pred.get('source_id')!=ref['source_id'] or (require_fact and pred.get('fact_id')!=ref['fact_id']):return False
    if 'start_s' in ref:
        if not {'start_s','end_s'}<=set(pred):return False
        if 'event_id' in ref and pred.get('event_id')!=ref['event_id']:return False
        return interval_iou(pred,ref)>=threshold
    return pred.get('quote')==ref['quote'] and pred.get('start_char')==ref['start_char'] and pred.get('end_char')==ref['end_char']

def normalized(value):
    if isinstance(value,str):return ('s',' '.join(unicodedata.normalize('NFKC',value).casefold().split()))
    return ('n',value)

def score_prediction(task,reference,prediction,sources,k=5,iou_threshold=.5):
    validate_task(task,{'corpus_id':task['corpus_id'],'split':task['split'],'source_ids':list(sources)})
    r=validate_reference(reference,task,sources)
    if isinstance(k,bool) or not isinstance(k,int) or k<1 or not 0<iou_threshold<=1:raise ValueError('Invalid scoring thresholds')
    result={'task_id':task['task_id'],'status':'failed','k':k,'iou_threshold':iou_threshold,**{m:0. for m in METRICS},'cost':{}}
    try:
        p=json_copy(prediction)
        if not isinstance(p,dict) or p.get('status')!='ok':
            result['status']=p.get('status','failed') if isinstance(p,dict) else 'malformed'
            return result
        ranked=p['ranked_sources'];windows=p['ranked_windows'];evidence=p['evidence'];facts=p['facts'];cost=p['cost']
        if not all(isinstance(x,list) for x in (ranked,windows,evidence,facts)):raise ValueError('Predicted arrays required')
        if any(not isinstance(s,str) or s not in sources for s in ranked):raise ValueError('Foreign ranking source')
        for item in evidence:validate_evidence(item,sources)
        for window in windows:validate_evidence(dict(window,fact_id='ranking'),sources)
        if not isinstance(cost,dict) or not {'tool_calls','requested_av_seconds'}<=set(cost):raise ValueError('Actual logical costs required')
        for value in cost.values():
            if number(value,'cost')<0:raise ValueError('Negative cost')
        if isinstance(cost['tool_calls'],bool) or not isinstance(cost['tool_calls'],int):raise ValueError('tool_calls must be integer')
        expected={f['fact_id'] for f in r['facts']};by_fact={}
        for fact in facts:
            if not isinstance(fact,dict) or set(fact)!={'fact_id','value'} or fact['fact_id'] not in expected or fact['fact_id'] in by_fact:
                raise ValueError('Malformed/duplicate answer fact')
            if isinstance(fact['value'],bool) or not isinstance(fact['value'],(str,int,float)):raise ValueError('Scalar answer required')
            by_fact[fact['fact_id']]=fact['value']
        ranked=unique(ranked);windows=unique(windows);evidence=unique(evidence)
        top=set(ranked[:k]);packages=r['evidence_packages'];ranks={sid:i+1 for i,sid in enumerate(ranked)}
        coverage=[len({i['source_id'] for i in pkg}&top)/len({i['source_id'] for i in pkg}) for pkg in packages]
        result['best_valid_package_coverage_at_k']=max(coverage)
        result['complete_package_source_recall_at_k']=float(max(coverage)==1)
        result['package_completion_reciprocal_rank']=max((1/max(ranks[s] for s in ids) if ids<=set(ranks) else 0)
            for ids in ({i['source_id'] for i in pkg} for pkg in packages))
        timed=[[i for i in pkg if 'start_s' in i] for pkg in packages]
        moment=[sum(any(matches(w,i,iou_threshold,False) for w in windows[:k]) for i in pkg)/len(pkg) for pkg in timed if pkg]
        result['moment_package_coverage_at_k']=max(moment,default=0.)
        correct=sum(f['fact_id'] in by_fact and normalized(by_fact[f['fact_id']]) in {normalized(x) for x in f['accepted_values']} for f in r['facts'])
        result['necessary_fact_accuracy']=correct/len(r['facts']);result['all_facts_correct']=float(correct==len(r['facts']))
        supported=any(all(any(matches(e,i,iou_threshold) for e in evidence) for i in pkg) for pkg in packages)
        budget=task['budget_recipe'];compliant=cost['tool_calls']<=budget['max_tool_calls'] and cost['requested_av_seconds']<=budget['max_requested_av_seconds']
        result['budget_compliant']=float(compliant);result['cost']=cost
        result['ebs']=float(result['all_facts_correct'] and supported and compliant)
        result['status']='ok' if compliant else 'over_budget'
    except (ValueError,TypeError,KeyError,OverflowError):result['status']='malformed'
    return result

def aggregate_scores(rows):
    if not rows:raise ValueError('No attempted tasks')
    if len({r['task_id'] for r in rows})!=len(rows):
        # Separate seeds/runs must be aggregated separately, never cherry-pick.
        raise ValueError('Duplicate attempted task IDs')
    return {'attempt_count':len(rows),**{m:sum(r[m] for r in rows)/len(rows) for m in METRICS},
            'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}}

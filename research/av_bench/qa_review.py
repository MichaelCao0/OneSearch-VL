"""Import human authoring reviews against exact task and answer snapshots.

Question retention alone does not confirm candidate answers. The output is a
private human-reviewed reference set; it does not declare a formal test split.
"""
import argparse,json,re
from pathlib import Path
from research.av_bench.contracts import json_copy
from research.av_train.trajectory import digest
from research.av_data.media_clock import sha256_file
from research.av_data.build_research_batch import write_json,write_jsonl

SCHEMA='av-searchqa-authoring-review-draft.v2'
CHOICES={'decision':{'pending','keep','revise','reject'},'audio_role':{'pending','necessary','helpful','redundant','unusable'},'needs_search':{'pending','required','perception_only','ambiguous'},'answer_support':{'pending','supported','revise','unsupported'}}

def validate_reference_quotes(references,documents):
    count=0
    for reference in references:
        fields={f['field_id'] for f in reference['answer_fields']};supported=set()
        for q in reference['support_quotes']:
            document=documents.get(q['source_id']);start=q['start_char'];end=q['end_char']
            if not document or isinstance(start,bool) or isinstance(end,bool) or not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(document['text']) or document['text'][start:end]!=q['quote']:raise ValueError('Support quote differs from frozen source text')
            if q['field_id'] not in fields:raise ValueError('Support quote names a foreign answer field')
            supported.add(q['field_id']);count+=1
        if supported!=fields:raise ValueError('Each answer field requires a frozen support quote')
    return count


def import_draft(draft,tasks,references,*,task_hash,reference_hash):
    d=json_copy(draft)
    required={'schema_version','reviewer_id','created_at','benchmark_gold','task_batch_sha256','candidate_references_sha256','reviews'}
    if not isinstance(d,dict) or set(d)!=required or d['schema_version']!=SCHEMA or d['benchmark_gold'] is not False:raise ValueError('Explicit v2 human draft required')
    if not isinstance(d['reviewer_id'],str) or not d['reviewer_id'].strip() or not isinstance(d['created_at'],str) or not d['created_at'].strip():raise ValueError('Human reviewer and submission time required')
    for key,expected in [('task_batch_sha256',task_hash),('candidate_references_sha256',reference_hash)]:
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected) or d[key]!=expected:raise ValueError('Review snapshot hash mismatch')
    ts={t['task_id']:t for t in tasks};rs={r['task_id']:r for r in references}
    if len(ts)!=len(tasks) or len(rs)!=len(references) or set(ts)!=set(rs):raise ValueError('Unique aligned tasks and references required')
    if not isinstance(d['reviews'],list):raise ValueError('Review entries required')
    seen=set();accepted=[];decisions=[]
    for review in d['reviews']:
        if not isinstance(review,dict) or set(review)!={'task_id','decision','audio_role','needs_search','answer_support','note'}:raise ValueError('Closed question review schema required')
        tid=review['task_id']
        if tid not in ts or tid in seen:raise ValueError('Duplicate or foreign review task')
        seen.add(tid)
        if any(not isinstance(review[k],str) or review[k] not in values for k,values in CHOICES.items()) or not isinstance(review['note'],str):raise ValueError('Unknown review verdict')
        ref=rs[tid]
        if ref.get('status')!='pending_human_review' or ref.get('benchmark_gold') is not False or not ref.get('answer_fields') or not ref.get('support_quotes'):raise ValueError('Pending supported candidate reference required')
        if review['decision']!='keep':disposition=review['decision']
        elif review['answer_support']!='supported':disposition='pending_answer_review' if review['answer_support']=='pending' else 'answer_'+review['answer_support']
        elif review['needs_search'] in {'pending','ambiguous'} or review['audio_role']=='pending':disposition='pending_task_characterization'
        else:
            disposition='human_reviewed';new=json_copy(ref)
            new.update(status='human_reviewed',reviewer_ids=[d['reviewer_id']],reviewed_at=d['created_at'],audio_role=review['audio_role'],needs_search=review['needs_search'],review_note=review['note'],human_review_sha256=digest(d),formal_test_reference=False)
            accepted.append(new)
        decisions.append({'task_id':tid,'disposition':disposition,'review':review})
    decisions.extend({'task_id':tid,'disposition':'missing_review'} for tid in sorted(set(ts)-seen))
    return {'schema_version':'av-searchqa-human-reviewed-references.v1','reviewer_id':d['reviewer_id'],'human_review_sha256':digest(d),'tasks_sha256':task_hash,'candidate_references_sha256':reference_hash,'references':accepted,'decisions':decisions,'formal_test_ready':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',required=True);p.add_argument('--review-file',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args()
    batch=Path(a.batch_dir);tasks=list(map(json.loads,(batch/'tasks-public.jsonl').read_text().splitlines()));refs=list(map(json.loads,(batch/'answer-candidates.private.jsonl').read_text().splitlines()))
    from research.av_search.corpus import load_frozen
    root,_=load_frozen(batch/'corpus')
    documents={d['source_id']:d for d in map(json.loads,(root/'documents.jsonl').read_text().splitlines())};validate_reference_quotes(refs,documents)
    result=import_draft(json.loads(Path(a.review_file).read_text()),tasks,refs,task_hash=sha256_file(batch/'tasks-public.jsonl'),reference_hash=sha256_file(batch/'answer-candidates.private.jsonl'))
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);write_json(out/'review-import.private.json',result);write_jsonl(out/'references.private.jsonl',result['references'])
    write_json(out/'summary.json',{'human_reviewed_references':len(result['references']),'candidate_tasks':len(tasks),'reviewer_id':result['reviewer_id'],'formal_test_ready':False,'task_batch_sha256':result['tasks_sha256'],'candidate_references_sha256':result['candidate_references_sha256']})
    print(json.dumps({'human_reviewed_references':len(result['references']),'candidate_tasks':len(tasks),'formal_test_ready':False}),flush=True)
if __name__=='__main__':main()

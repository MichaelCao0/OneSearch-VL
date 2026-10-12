"""Construct direct QA candidates with anonymous AV query clips.

Publisher answer candidates stay private and pending human review. Clips and
origin families inherit their source split reservation; no teacher or optimizer runs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import unicodedata
from urllib.parse import unquote,urlsplit,urlunsplit
from research.av_bench.contracts import TASK_VERSION,json_copy,number,validate_task
from research.av_search.corpus import load_frozen
from .build_research_batch import write_json,write_jsonl
from .media_clock import clip_media,sha256_file
from .ingest import ingest_media

CASE_FIELDS={'task_id','question','target_source_id','start_s','end_s','scenario','fields'}

def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKC',value).casefold() if c.isalnum())

def origin_key(url):
    u=urlsplit(url);return urlunsplit((u.scheme.casefold(),u.netloc.casefold(),unicodedata.normalize('NFC',unquote(u.path)).replace('_',' '),u.query,''))

def validate_authoring_cases(cases,docs,sources):
    if not isinstance(cases,list) or not cases:raise ValueError('Require authoring cases')
    ids=set();references=[]
    for case in cases:
        if not isinstance(case,dict) or set(case)!=CASE_FIELDS:raise ValueError('Closed private authoring case required')
        tid=case['task_id'];sid=case['target_source_id']
        if not isinstance(tid,str) or not tid or Path(tid).name!=tid or tid in ids:raise ValueError('Unique safe task IDs required')
        ids.add(tid)
        if sid not in sources or sources[sid]['media_type']!='audio_video':raise ValueError('Target must be reserved AV source')
        start=number(case['start_s'],'start_s');end=number(case['end_s'],'end_s')
        if not 0<=start<end or end>sources[sid]['duration_s']+1e-6 or end-start>30.+1e-6:raise ValueError('Bounded valid clue crop required')
        if not isinstance(case['question'],str) or not case['question'].strip() or not isinstance(case['scenario'],str):raise ValueError('Question/scenario required')
        fields=case['fields']
        if not isinstance(fields,list) or not fields:raise ValueError('Answer fields required')
        related=[d for d in docs.values() if origin_key(d['origin_url'])==origin_key(sources[sid]['origin_url'])]
        quotes=[];field_ids=set()
        for field in fields:
            if not isinstance(field,dict) or set(field)!={'field_id','accepted_values'}:raise ValueError('Closed answer field required')
            fid=field['field_id'];values=field['accepted_values']
            if not isinstance(fid,str) or not fid or fid in field_ids:raise ValueError('Unique answer field IDs required')
            field_ids.add(fid)
            if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values):raise ValueError('Answer strings required')
            # Canonical candidate must occur verbatim in a related frozen doc.
            match=next(((d,d['text'].casefold().find(values[0].casefold())) for d in related if values[0].casefold() in d['text'].casefold()),None)
            if match is None:raise ValueError('Requested candidate has no related publisher quote: '+fid)
            doc,pos=match;start_char=max(0,pos-100);end_char=min(len(doc['text']),pos+len(values[0])+100)
            quotes.append({'field_id':fid,'source_id':doc['source_id'],'start_char':start_char,'end_char':end_char,
              'quote':doc['text'][start_char:end_char],'basis':'frozen_publisher_claim; not independent truth verification'})
        references.append({'schema_version':'av-searchqa-reference-candidate.v1','task_id':tid,'status':'pending_human_review','reviewer_ids':[],
          'benchmark_gold':False,'target_source_id':sid,'answer_fields':json_copy(fields),'support_quotes':quotes,
          'scenario':case['scenario'],'query_interval_on_parent':{'start_s':case['start_s'],'end_s':case['end_s']},
          'audio_required':'pending_content_review','external_information_required':'publisher_attribute_lookup'})
    return references

def actor_task(case,clue_id,corpus_id,split):
    task={'schema_version':TASK_VERSION,'task_id':case['task_id'],'question':case['question'],'initial_sources':[clue_id],
      'target_questions':[{'fact_id':'answer','question':case['question']}],'corpus_id':corpus_id,'split':split,
      'budget_recipe':{'max_tool_calls':6,'max_requested_av_seconds':12.}}
    validate_task(task,{'corpus_id':corpus_id,'split':split,'source_ids':[clue_id]});return task

def build_clue_batch(pool_dir,train_registry,cases,out_dir):
    pool=Path(pool_dir);out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    corpus=pool/'corpus' if (pool/'corpus').exists() else pool/'authoring-corpus'
    _,manifest=load_frozen(corpus)
    if manifest['split'] not in {'train','dev','test'}:raise ValueError('Train/dev/test source reservation required')
    split=manifest['split']
    sources={s['source_id']:s for s in map(json.loads,(pool/'sources.private.jsonl').read_text().splitlines())}
    docs={d['source_id']:d for d in map(json.loads,(corpus/'documents.jsonl').read_text().splitlines())}
    if set(docs)!=set(sources) or set(docs)!=set(manifest['source_ids']):raise ValueError('Corpus/registry identity mismatch')
    for source in sources.values():
        if sha256_file(source['asset_path'])!=source['asset_sha256']:raise ValueError('Reserved asset hash mismatch')
    train=list(map(json.loads,Path(train_registry).read_text().splitlines()))
    shared_bytes={s['asset_sha256'] for s in sources.values()}&{s['asset_sha256'] for s in train}
    shared_origins={origin_key(s['origin_url']) for s in sources.values()}&{origin_key(s['origin_url']) for s in train}
    if shared_bytes or shared_origins:raise ValueError('Train/dev asset or origin family overlap')
    references=validate_authoring_cases(cases,docs,sources)
    out.mkdir(parents=True,exist_ok=False);shutil.copytree(corpus,out/'corpus')
    tasks=[];bindings=[];lineage=[];clue_public=[];crop_cache={}
    for case in cases:
        parent=sources[case['target_source_id']]
        crop_key=(parent['asset_sha256'],case['start_s'],case['end_s'])
        if crop_key not in crop_cache:
            derived=clip_media(parent,case['start_s'],case['end_s'],out/'query-derived'/case['task_id'])
            clue_source=ingest_media(derived['output_path'],'https://query.invalid/anonymous',out/'query-assets',{},
                rights_basis=parent['rights_basis'],redistribution_scope=parent['redistribution_scope'])
            crop_cache[crop_key]=(derived,clue_source)
        else:
            derived,clue_source=crop_cache[crop_key]
        (out/'query-derived'/case['task_id']).mkdir(parents=True,exist_ok=True)
        write_json(out/'query-derived'/case['task_id']/'derivation.private.json',derived)
        if clue_source['media_type']!='audio_video':raise ValueError('Query must retain timed audio and video')
        clue_id='src_'+secrets.token_hex(12)
        while clue_id in sources or any(b['clue_id']==clue_id for b in bindings):clue_id='src_'+secrets.token_hex(12)
        tasks.append(actor_task(case,clue_id,manifest['corpus_id'],split));bindings.append({'clue_id':clue_id,'source':clue_source})
        family=hashlib.sha256(origin_key(parent['origin_url']).encode()).hexdigest()
        lineage.append({'task_id':case['task_id'],'clue_id':clue_id,'clip_internal_source_id':clue_source['source_id'],
          'parent_source_id':parent['source_id'],'parent_sha256':parent['asset_sha256'],'clip_sha256':clue_source['asset_sha256'],
          'family_id':'origin_'+family[:24],'split':split,'original_interval':derived['original_interval'],'mapping':derived['mapping']})
        clue_public.append({'source_id':clue_id,'media_type':'audio_video','duration_s':clue_source['duration_s'],'time_axis':'query_clip_local_presentation'})
    write_jsonl(out/'tasks-public.jsonl',tasks);write_jsonl(out/'sources.private.jsonl',list(sources.values()))
    write_json(out/'scope.json',{k:manifest[k] for k in ('corpus_id','split','source_ids')});write_json(out/'clue-bindings.private.json',bindings)
    write_json(out/'clues-public.json',clue_public);write_jsonl(out/'answer-candidates.private.jsonl',references);write_json(out/'clue-lineage.private.json',lineage)
    write_json(out/'source-family-audit.private.json',{'same_bytes_train_dev':len(shared_bytes),'same_origin_train_dev':len(shared_origins),
      'grouping':'original publisher origin, including its HTML and AV derivatives','near_duplicate_family_review':'pending',
      'family_count':len({l['family_id'] for l in lineage}),'train_registry_sha256':sha256_file(train_registry)})
    summary={'schema_version':'av-anonymous-clue-batch.v1','split':split,'tasks':len(tasks),'query_clips':len(bindings),'unique_query_assets':len(crop_cache),
      'parent_recordings':len({l['parent_source_id'] for l in lineage}),'indexed_av_sources':sum(s['media_type']=='audio_video' for s in sources.values()),
      'indexed_webpages':sum(s['media_type']=='webpage' for s in sources.values()),'benchmark_gold':0,'semantic_review_status':'pending',
      'profile':'direct_audio_video_search_qa','initial_target_id_shared':False,'initial_publisher_metadata_shared':False,
      'query_clips_indexed':False,'comparison_registry_same_bytes':0,'comparison_registry_same_origin':0,'teacher_executed':False,'optimizer_updates':0,
      'tasks_sha256':sha256_file(out/'tasks-public.jsonl'),'candidate_references_sha256':sha256_file(out/'answer-candidates.private.jsonl'),
      'scenarios':{s:sum(c['scenario']==s for c in cases) for s in sorted({c['scenario'] for c in cases})}}
    write_json(out/'summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pool-dir','train-registry','authoring','out-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(build_clue_batch(a.pool_dir,a.train_registry,json.loads(Path(a.authoring).read_text()),a.out_dir),ensure_ascii=False))
if __name__=='__main__':main()

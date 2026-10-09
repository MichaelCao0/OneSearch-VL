"""Build versioned, unreviewed exploration packages from actual Commons bytes.

This does not create semantic gold, ASR, event independence or observer answers.
The input acquisition manifest and all exclusions are preserved separately.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .ingest import ingest_media
from .validate import validate_source
from .tasks import validate_task_public
from .split import build_lineage_components, assign_event_splits
from .audit import audit_split_manifest, export_actor_task
from research.av_search.corpus import build_corpus, visible_corpus, canonical

TARGETS={
 'technical_mechanisms':[
  '原始音视频中哪些可听变化可以定位到具体区间，各区间伴随什么可见机械状态？请分别给出声音和画面依据。',
  '所选声音节律或变化与可见机械运动能对应到什么程度？请保留画外声、后期音轨和未知声源的解释。',
  '搜索来源池中的相关录制并比较：它们能否核验这段具体录制的过程？区分发布者描述、相似实例与独立观察，无法判定时说明缺少什么。'],
 'field_performance':[
  '选取可定位的声音起止或节律变化，核验当时画面中的表演者、动作与对象；给出原始时间区间。',
  '现有声画支持何种对应？同期演奏、配乐、画外声或混合音轨有哪些已确认和未确认部分？',
  '从全局来源池查找相关现场材料，比较能支持的过程与来源边界；判断是否同一事件或独立观察需要哪些额外证据？'],
 'environment_animals':[
  '哪些可听环境变化能定位到具体区间，画面在相应区间呈现什么？分别引用声音和画面。',
  '可听事件是否能够归因于画面中的对象？区分同时出现、空间对应、声源归因和无法确定的解释。',
  '检索相关来源并比较：原始录制与发布者说明分别支持哪些结论？哪些空间、声源或事件一致性仍缺少证据？']}


def _write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def prepare_pilot(acquisition_manifest, out_dir, asset_root, corpus_id='commons-exploration-pilot-v1'):
    input_path=Path(acquisition_manifest).resolve()
    acquisition=json.loads(input_path.read_text())
    seeds=[s for s in acquisition['sources'] if s['status']=='candidate_unreviewed_av']
    if not 1<=len(seeds)<=10:
        raise ValueError('Pilot accepts 1..10 unreviewed seed candidates')
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    records=[]; mappings=[]; failures=[]; docs=[]
    for seed in seeds:
        meta=seed['public_metadata']
        public={k:meta[k] for k in ('title','description','creator','license','license_url') if k in meta}
        if meta.get('categories'):
            public['categories']=meta['categories'].split('|') if isinstance(meta['categories'],str) else meta['categories']
        try:
            record=ingest_media(seed['asset_path'],seed['origin_url'],asset_root,public_metadata=public,
                rights_basis='publisher_declared:'+meta.get('license','unknown'),redistribution_scope='metadata_only')
            if record['asset_sha256']!=seed['sha256'] or validate_source(record):
                raise ValueError('Acquisition/source integrity mismatch')
            records.append(record)
            mappings.append({'acquisition_id':seed['source_id'],'source_id':record['source_id'],
                'candidate_domain':seed['candidate_domain'],'origin_url':seed['origin_url'],
                'description_page_sha256':seed.get('description_page_sha256'),
                'description_page_revision':seed.get('description_page_revision'),
                'rights_metadata':{k:meta.get(k,'') for k in ('license','license_url','creator','credit','attribution','usage_terms')},
                'semantic_review':'not_performed','event_relationships':'unknown'})
            text='\n'.join(f'{k}: {public[k]}' for k in ('title','description','creator','license','license_url') if k in public)
            docs.append({k:record[k] for k in ('source_id','origin_url','asset_sha256','asset_path','public_metadata','media_type','duration_s')})
            docs[-1].update(text=text,text_kind='public_metadata')
        except (ValueError,OSError) as exc:
            failures.append({'acquisition_id':seed['source_id'],'error':str(exc),'status':'ingest_failed'})
    (out/'sources.jsonl').write_text(''.join(canonical(x)+'\n' for x in records))
    _write_json(out/'acquisition-to-source.json',mappings)
    _write_json(out/'ingest-failures.json',failures)
    if not records:
        raise ValueError('No valid original media; failures preserved')
    manifest=build_corpus(docs,out/'corpus',corpus_id,'exploration')
    scope={'corpus_id':corpus_id,'split':'exploration','source_ids':manifest['source_ids']}
    _write_json(out/'scope.json',scope)
    # Components here are structural candidates; no event IDs are asserted.
    sidecars=[{'source_id':r['source_id'],'stage':'exploration','record_role':'event_evidence',
               'near_duplicate_ids':[],'lineage_links':[]} for r in records]
    components=build_lineage_components(sidecars)
    splits=assign_event_splits(components,seed=20261009,ratios={'exploration':1.0})
    audit_split_manifest(components,splits)
    _write_json(out/'lineage-candidates.json',sidecars)
    _write_json(out/'candidate-components.json',components)
    _write_json(out/'split-manifest.json',splits)
    tasks=[];drafts=[]
    visible=visible_corpus(out/'corpus',scope)
    for n,mapping in enumerate(mappings,1):
        sid=mapping['source_id']
        task={'schema_version':'1.0','created_by':'prepare_pilot-v1','revision':'draft-v1',
              'task_id':f'av-pilot-{n:03d}',
              'question':'研究这段录制及相关来源能支持的声画过程与来源边界。所有结论必须引用实际读取的原始片段或公开文本；不足以判定时明确说明。',
              'target_questions':[{'claim_id':f'c{i}','question':q} for i,q in enumerate(TARGETS[mapping['candidate_domain']],1)],
              'initial_sources':[sid],'corpus_id':corpus_id,'split':'exploration',
              'budget_recipe':{'max_tool_calls':20,'max_requested_av_seconds':60,'provider_requests':0}}
        task=validate_task_public(task,scope['source_ids'],corpus_id,'exploration')
        export_actor_task(task,visible)
        tasks.append(task)
        drafts.append({'task_id':task['task_id'],'status':'draft_unreviewed','candidate_domain':mapping['candidate_domain'],
                       'source_id':sid,'gold_status':'not_created','human_review':'not_performed','audio_necessity':'not_evaluated'})
    (out/'tasks-public.jsonl').write_text(''.join(canonical(x)+'\n' for x in tasks))
    _write_json(out/'draft-status.json',drafts)
    summary={'schema_version':'av-pilot-package-0.1','created_at':datetime.now(timezone.utc).isoformat(),
             'status':'draft_unreviewed','acquisition_manifest_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
             'source_count':len(records),'draft_task_count':len(tasks),'ingest_failure_count':len(failures),
             'source_bytes':sum(r['asset_bytes'] for r in records),'source_seconds':sum(r['duration_s'] for r in records),
             'domain_counts':dict(Counter(m['candidate_domain'] for m in mappings)),
             'corpus_hash':manifest['corpus_hash'],'scope':'global exploration pool for every task',
             'event_independence':'not_adjudicated','same_event_cross_recordings':'not_established',
             'paired_description_pages':'publisher_assertions_not_independent_sources','human_gold':'not_created',
             'G0':'pending','raw_av_observer_ready':False,'method_experiments':0}
    _write_json(out/'package-summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acquisition-manifest',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--asset-root',required=True);p.add_argument('--corpus-id',default='commons-exploration-pilot-v1')
    a=p.parse_args();print(json.dumps(prepare_pilot(a.acquisition_manifest,a.out_dir,a.asset_root,a.corpus_id),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

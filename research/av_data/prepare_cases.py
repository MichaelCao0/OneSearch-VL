"""Freeze three bounded natural AV case drafts and blank human review packets.

All tasks search one global exploration corpus. Candidate roles and quotes are
private preparation aids, not gold or independent recording adjudications.
No fetching, model observations, crops, semantic labels or human signatures are
created here. Excluded acquisition rows remain excluded and are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from research.av_search.corpus import build_corpus, canonical, visible_corpus, strict_json_loads
from .audit import export_actor_task
from .tasks import validate_task_public, _opaque, reject_private_fields
from .validate import validate_source
from .review_forms import generate
from .web_sources import validate_web_source, web_corpus_document, quote_web_source

DOMAINS = frozenset({'technical_mechanisms','field_performance','environment_animals'})
PLAN_FIELDS = frozenset({'task_id','candidate_domain','question','target_questions','initial_sources','source_roles'})
ROLES = frozenset({'primary_recording','comparison_recording','publisher_assertion','external_context'})
ROLE_FIELDS = frozenset({'source_id','role','related_source_id','quote'})
REVISION = 'three-cases-draft-v1'


def _json(path):
    return strict_json_loads(Path(path).read_text(encoding='utf-8'))


def _jsonl(path):
    return [strict_json_loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _write(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_case_plan(plans, allowed_source_ids):
    """Validate three cases, each 4-6 distinct assets, without accepting gold fields."""
    plans=strict_json_loads(canonical(plans));allowed=set(allowed_source_ids)
    for source_id in allowed:_opaque(source_id)
    if not isinstance(plans,list) or len(plans)!=3:raise ValueError('Exactly three case drafts are required')
    if {p.get('candidate_domain') for p in plans if isinstance(p,dict)}!=DOMAINS:raise ValueError('Expected mechanical, performance and environment case domains')
    task_ids=set();selected=set()
    for plan in plans:
        if not isinstance(plan,dict) or set(plan)!=PLAN_FIELDS:raise ValueError('Unexpected/private case plan fields')
        task_id=plan['task_id']
        if not isinstance(task_id,str) or not task_id.strip() or task_id in task_ids:raise ValueError('Distinct nonempty task IDs required')
        task_ids.add(task_id)
        # Actor-visible portions obey the same closed contract as final tasks.
        public={'schema_version':'1.0','created_by':'prepare_cases','revision':REVISION,
            'task_id':task_id,'question':plan['question'],'target_questions':plan['target_questions'],
            'initial_sources':plan['initial_sources'],'corpus_id':'case-validation','split':'exploration',
            'budget_recipe':{'max_tool_calls':20,'max_requested_av_seconds':60,'provider_requests':0}}
        validate_task_public(public,allowed,'case-validation','exploration')
        roles=plan['source_roles']
        if not isinstance(roles,list) or not 4<=len(roles)<=6:raise ValueError('Each case needs 4-6 distinct available source assets')
        ids=[]
        for role in roles:
            if not isinstance(role,dict) or set(role)-ROLE_FIELDS or not {'source_id','role'}<=set(role):raise ValueError('Unexpected/private source role fields')
            sid=_opaque(role['source_id'])
            if sid not in allowed or sid in ids:raise ValueError('Duplicate or foreign case source')
            ids.append(sid)
            if role['role'] not in ROLES:raise ValueError('Unknown draft source role')
            if 'quote' in role and (not isinstance(role['quote'],str) or not role['quote'].strip()):raise ValueError('Source assertion quote must be exact nonempty text')
            if role['role']=='publisher_assertion' and 'related_source_id' not in role:raise ValueError('Publisher page must identify its related recording')
            if 'related_source_id' in role:_opaque(role['related_source_id'])
        if sum(r['role']=='primary_recording' for r in roles)!=1:raise ValueError('Each case requires one primary recording')
        if not set(plan['initial_sources'])<=set(ids):raise ValueError('Initial sources must belong to candidate case assets')
        for role in roles:
            if 'related_source_id' in role and (role['related_source_id'] not in ids or role['related_source_id']==role['source_id']):raise ValueError('Related source must be another case asset')
        selected.update(ids)
    if len(selected)>18:raise ValueError('Three-case asset cap is 18')
    return plans


def prepare_cases(sources_path, case_plan_path, acquisition_manifest, out_dir,
                  corpus_id='natural-three-cases-20261010-v1'):
    """Create a new package after validating selected media, raw pages and quotes."""
    sources_path,case_plan_path,acquisition_manifest=map(Path,(sources_path,case_plan_path,acquisition_manifest))
    out=Path(out_dir).resolve()
    if out.exists():raise FileExistsError(out)
    records=_jsonl(sources_path)
    registry={r['source_id']:r for r in records}
    if len(records)!=len(registry) or not 1<=len(records)<=18:raise ValueError('Unique global source registry must stay within 18 assets')
    plans=validate_case_plan(_json(case_plan_path),registry)
    selected={r['source_id'] for plan in plans for r in plan['source_roles']}
    if selected!=set(registry):raise ValueError('Global registry must contain exactly the union of selected case assets')
    acquisition=_json(acquisition_manifest)
    if not isinstance(acquisition,dict) or not isinstance(acquisition.get('sources'),list):raise ValueError('Acquisition manifest missing source rows')
    accepted={r.get('sha256'):r for r in acquisition['sources'] if r.get('status')=='candidate_unreviewed_av'}
    exclusions=[r for r in acquisition['sources'] if r.get('status')!='candidate_unreviewed_av']
    docs=[];mappings=[];replay=[]
    for record in records:
        if record.get('media_type')=='webpage':
            errors=validate_web_source(record)
            if errors:raise ValueError('Invalid webpage: '+'; '.join(errors))
            docs.append(web_corpus_document(record))
            replay.append({'source_id':record['source_id'],'media_type':'webpage','asset_path':record['asset_path'],
                'asset_sha256':record['asset_sha256'],'body_text_sha256':record['body_text_sha256'],
                'fetched_at':record['fetched_at'],'snapshot_identity':record['snapshot_identity'],
                'replay':'research.av_data.web_sources.quote_web_source with exact character bounds'})
        else:
            errors=validate_source(record)
            if errors:raise ValueError('Invalid original media: '+'; '.join(errors))
            if record['media_type']!='audio_video' or record['asset_sha256'] not in accepted:raise ValueError('Only accepted unreviewed AV acquisitions may be selected; exclusions are not overridden')
            row=accepted[record['asset_sha256']]
            mappings.append({'source_id':record['source_id'],'acquisition_id':row.get('source_id'),
                'acquisition_status':row['status'],'asset_sha256':record['asset_sha256']})
            public=record['public_metadata'];reject_private_fields(public)
            text='\n'.join(f'{k}: {public[k]}' for k in ('title','description','creator','license','license_url') if k in public)
            doc={k:record[k] for k in ('source_id','origin_url','asset_sha256','asset_path','public_metadata','media_type','duration_s')}
            doc.update(text=text,text_kind='public_metadata');docs.append(doc)
            replay.append({'source_id':record['source_id'],'media_type':record['media_type'],'asset_path':record['asset_path'],
                'asset_sha256':record['asset_sha256'],'original_interval':{'start_s':0.0,'end_s':record['duration_s']},
                'timestamp_origin':record['timestamp_origin'],'replay':'original presentation clock; no crop is an additional recording'})
    drafts=[];quotes=[]
    for plan in plans:
        relations=[]
        for role in plan['source_roles']:
            record=registry[role['source_id']]
            timed=role['role'] in {'primary_recording','comparison_recording'}
            if timed != (record['media_type']=='audio_video'):raise ValueError('Recording roles require original AV; text roles require webpages')
            if 'related_source_id' in role and registry[role['related_source_id']]['media_type']!='audio_video':raise ValueError('Publisher/context relation must refer to a recording')
            if 'quote' in role:
                start=record.get('body_text','').find(role['quote'])
                if start<0:raise ValueError('Candidate publisher quote does not occur in captured body text')
                quote=quote_web_source(record,start,start+len(role['quote']))
                quote.update(task_id=plan['task_id'],status='publisher_assertion_unreviewed')
                quotes.append(quote)
            if role['role']=='publisher_assertion':
                relations.append({'left_source_id':role['source_id'],'right_source_id':role['related_source_id'],
                    'candidate_relation':'publisher_description_of_asset','status':'candidate_unreviewed',
                    'independent_recording':False,'semantic_match':'not_adjudicated'})
            elif role['role']=='comparison_recording':
                primary=next(r['source_id'] for r in plan['source_roles'] if r['role']=='primary_recording')
                relations.append({'left_source_id':primary,'right_source_id':role['source_id'],
                    'candidate_relation':'comparison_candidate','status':'candidate_unreviewed',
                    'same_event':'unknown','same_recording':'unknown','independence':'not_adjudicated'})
        drafts.append({'task_id':plan['task_id'],'candidate_domain':plan['candidate_domain'],
            'status':'draft_unreviewed','source_roles':plan['source_roles'],'candidate_relations':relations,
            'human_review':'not_performed','human_gold_count':0,'audio_necessity':'pending',
            'multi_source_paths':'candidate_unreviewed_not_semantically_verified',
            'scope':'within_snapshot','needs_search_target':plan['target_questions'][-1]['claim_id']})
    # All failures above happen before making a partially reviewable package.
    out.mkdir(parents=True,exist_ok=False)
    (out/'sources.jsonl').write_text(''.join(canonical(r)+'\n' for r in records),encoding='utf-8')
    manifest=build_corpus(docs,out/'corpus',corpus_id,'exploration')
    scope={'corpus_id':corpus_id,'split':'exploration','source_ids':manifest['source_ids']};_write(out/'scope.json',scope)
    visible=visible_corpus(out/'corpus',scope);tasks=[]
    for plan in plans:
        task={'schema_version':'1.0','created_by':'prepare_cases','revision':REVISION,
            'task_id':plan['task_id'],'question':plan['question'],'target_questions':plan['target_questions'],
            'initial_sources':plan['initial_sources'],'corpus_id':corpus_id,'split':'exploration',
            'budget_recipe':{'max_tool_calls':20,'max_requested_av_seconds':60,'provider_requests':0}}
        tasks.append(export_actor_task(task,visible))
    (out/'tasks-public.jsonl').write_text(''.join(canonical(t)+'\n' for t in tasks),encoding='utf-8')
    _write(out/'case-drafts-private.json',drafts);_write(out/'publisher-quotes.json',quotes)
    _write(out/'acquisition-exclusions.json',exclusions);_write(out/'acquisition-to-source.json',mappings)
    _write(out/'replay-assets.json',replay)
    forms=generate(out/'tasks-public.jsonl',out/'sources.jsonl',out/'review-packets')
    web=out/'web-body-text';web.mkdir()
    for record in records:
        if record['media_type']=='webpage':(web/(record['source_id']+'.txt')).write_text(record['body_text'],encoding='utf-8')
    _write(out/'inputs-manifest.json',{'sources':{'path':str(sources_path.resolve()),'sha256':_digest(sources_path)},
        'case_plan':{'path':str(case_plan_path.resolve()),'sha256':_digest(case_plan_path)},
        'acquisition':{'path':str(acquisition_manifest.resolve()),'sha256':_digest(acquisition_manifest)},
        'code':{'prepare_cases_sha256':_digest(__file__),
                'web_sources_sha256':_digest(Path(__file__).with_name('web_sources.py'))}})
    summary={'schema_version':'av-three-case-package.v1','created_at':datetime.now(timezone.utc).isoformat(),
        'status':'draft_unreviewed','case_count':3,'source_count':len(records),
        'case_source_counts':{p['task_id']:len(p['source_roles']) for p in plans},
        'media_source_count':sum(r['media_type']=='audio_video' for r in records),
        'web_source_count':sum(r['media_type']=='webpage' for r in records),
        'source_bytes':sum(r['asset_bytes'] for r in records),
        'media_source_seconds':sum(r['duration_s'] for r in records if r['duration_s'] is not None),
        'excluded_acquisition_count':len(exclusions),'corpus_hash':manifest['corpus_hash'],
        'manifest_sha256':manifest['manifest_sha256'],'global_scope_source_count':len(scope['source_ids']),
        'quote_count':len(quotes),'review_forms':forms,'human_gold_count':0,'human_review':'not_performed',
        'audio_necessity':'pending','same_event_cross_recordings':'not_established',
        'independence':'not_adjudicated','multi_source_paths':'candidate_unreviewed_not_semantically_verified',
        'G0':'pending','method_experiments':0}
    _write(out/'package-summary.json',summary)
    (out/'REVIEW-README.md').write_text('\n'.join([
        '# 三案独立审核包（私有）','',
        '状态：draft_unreviewed；human gold = 0；音频必要性 pending。',
        'A/B 分别填写 review-packets/A 与 B，第一次提交前不得互看；adjudication 仍为空白。',
        '所有任务使用 scope.json 的同一全局 corpus。case-drafts-private.json 只列准备候选和未知关系，不是 gold。',
        'sources.jsonl 与 replay-assets.json 保存原始 bytes/hash、时钟或网页抓取版本。web-body-text 的字符坐标与 corpus web_body 完全一致。',
        'publisher-quotes.json 的 exact quote 可使用 pixi run python -m research.av_data.web_sources --sources SOURCES --source-id ID --char-start START --char-end END 重放。',
        '媒体只回放原始 [start_s,end_s)；任何新裁切只是视图，不是另一份独立录制。',
        '发布者说明不是独立观察；相似媒体不能自动判同事件。逐项裁决 proposition、AND/OR packages、bindings、关系和缺口。',
        '没有审核人身份、签名或裁决；空白不代表 unknown 或完成。','']),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',required=True);parser.add_argument('--case-plan',required=True)
    parser.add_argument('--acquisition-manifest',required=True);parser.add_argument('--out-dir',required=True)
    parser.add_argument('--corpus-id',default='natural-three-cases-20261010-v1')
    args=parser.parse_args()
    print(json.dumps(prepare_cases(args.sources,args.case_plan,args.acquisition_manifest,args.out_dir,args.corpus_id),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

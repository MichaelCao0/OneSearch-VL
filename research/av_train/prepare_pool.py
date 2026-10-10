"""Screen new acquisition seeds against reserved assets; prepare authoring-only train pool.

A different page/hash does not prove an independent recording. This module never
freezes split lineage or exports actor tasks before source-family adjudication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from research.av_bench.contracts import TASK_VERSION,REFERENCE_VERSION,validate_task
from research.av_data.ingest import ingest_media
from research.av_data.web_sources import ingest_web_snapshot,web_corpus_document
from research.av_data.build_research_batch import write_json,write_jsonl
from research.av_search.corpus import build_corpus


def screen_candidates(manifests,reserved_hashes):
    seen=set(reserved_hashes);accepted=[];rejected=[]
    for manifest in manifests:
        for row in manifest['sources']:
            if row.get('status')!='candidate_unreviewed_av':
                rejected.append({'page_id':row.get('page_id'),'reason':'acquisition_not_accepted_av'});continue
            sha=row.get('sha256')
            if sha in seen:
                rejected.append({'page_id':row.get('page_id'),'reason':'reserved_or_duplicate_asset'});continue
            if not isinstance(sha,str) or len(sha)!=64 or not row.get('asset_path'):raise ValueError('Missing original acquisition asset')
            seen.add(sha);accepted.append(row)
    return accepted,rejected


def prepare_pool(manifest_paths,reserved_sources_path,out_dir):
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    reserved=[json.loads(l) for l in Path(reserved_sources_path).read_text().splitlines() if l]
    manifests=[json.loads(Path(p).read_text()) for p in manifest_paths]
    seeds,rejected=screen_candidates(manifests,{r['asset_sha256'] for r in reserved})
    if len(seeds)!=4:raise ValueError('Authoring batch expects four new natural AV seeds')
    out.mkdir(parents=True,exist_ok=False);sources=[];by_domain={};relations=[];docs=[]
    for seed in seeds:
        if hashlib.sha256(Path(seed['asset_path']).read_bytes()).hexdigest()!=seed['sha256']:raise ValueError('Acquired asset hash mismatch')
        # Corpus metadata has a closed schema. Keep the complete publisher API in acquisition manifest.
        metadata={k:v for k,v in seed['public_metadata'].items() if k in {'title','description','creator','license','license_url'}}
        source=ingest_media(seed['asset_path'],seed['origin_url'],out/'originals',metadata,
                           rights_basis='publisher_declared:'+metadata.get('license','unknown'),redistribution_scope='metadata_only')
        sources.append(source);by_domain.setdefault(seed['candidate_domain'],[]).append(source['source_id'])
        relations.append({'source_id':source['source_id'],'acquisition_page_id':seed['page_id'],
                          'family_review_status':'pending','recording_id':None,'event_id':None,
                          'candidate_split':'train','split_frozen':False})
        d={k:source[k] for k in ('source_id','origin_url','asset_sha256','asset_path','media_type','duration_s','public_metadata')}
        d.update(text='\n'.join(f'{k}: {v}' for k,v in metadata.items()),text_kind='public_metadata');docs.append(d)
        if seed.get('description_page_html_path'):
            page=ingest_web_snapshot(seed['description_page_html_path'],seed['origin_url'],out/'snapshots',metadata,
                                    fetched_at=seed['fetched_at'],rights_basis='publisher_description_snapshot',redistribution_scope='metadata_only')
            sources.append(page);docs.append(web_corpus_document(page))
            relations.append({'source_id':page['source_id'],'related_recording_id':source['source_id'],
                              'relation':'publisher_description_not_independent_recording','family_review_status':'pending',
                              'candidate_split':'train','split_frozen':False})
    mechanical=by_domain.get('technical_mechanisms',[]);performance=by_domain.get('field_performance',[]);environment=by_domain.get('environment_animals',[])
    if len(mechanical)!=1 or len(performance)!=1 or len(environment)!=2:raise ValueError('Expected technical/performance/environment source groups 1/1/2')
    specs=[
      ('event_localization','technical_mechanisms',mechanical,mechanical,
       '在起始机械录制中定位可见部件完成两个连续运动周期的片段，记录这一段的实际声音及其与运动的关系。',
       ['哪个部件形成可追踪周期？','连续两周期的起止时间是什么？','片段的可听声音与运动有什么对应？'],
       '发布者声称这是拖拉机；必须先回放确认具体部件与周期，不能用标题代替观察。'),
      ('source_discovery','live_performance',performance,[],
       '在语料库寻找同时可见吹奏与拨弦演奏的录制，定位两名演奏者共同演奏的片段，并说明实际听到的声部。',
       ['可见演奏者与动作分别是什么？','音轨中能辨认哪些声部？','共同演奏的证据区间在哪里？'],
       '候选池目前只有一份演奏录制且标题直给，需要加入同类干扰来源后再用于有效搜索训练。'),
      ('cross_source_comparison','environment',environment,[environment[0]],
       '搜索另一份降雨环境录制，比较两份录制的连续背景声、离散声学变化和画面中的运动。分别引用原片区间。',
       ['两份录制的连续背景声如何比较？','有哪些可确认的离散声音或变化？','每项观察对应哪些源时间区间？'],
       'Duck Pond 搜索结果是池塘降雨，不能当鸭子活动；不能根据 thunder 标题预设片段有雷声。'),
      ('research_composition','environment',environment,[environment[1]],
       '搜索另一份降雨录制，结合两份原片和说明页，比较实际雨声/其他声音与可见环境，并列出发布者提供的地点及各自依据。',
       ['两份原片的实际声音与可见环境各是什么？','发布者分别提供哪些地点信息？','声画区间和文字引用分别在哪里？'],
       '这是一组雨景比较候选；必须核实是否有足够可区分的声音与画面，地点仅按公开说明引用。')]
    corpus_id='avrb-train-authoring-pool-20261010-v0';registry={r['source_id']:r for r in sources};rows=[]
    for i,(family,domain,candidates,initial,question,targets,risk) in enumerate(specs,1):
        tid='avrb-train-candidate-'+str(i).zfill(3)
        task={'schema_version':TASK_VERSION,'task_id':tid,'question':question,'initial_sources':initial,
              'target_questions':[{'fact_id':'f'+str(j),'question':q} for j,q in enumerate(targets,1)],
              'corpus_id':corpus_id,'split':'train','budget_recipe':{'max_tool_calls':20,'max_requested_av_seconds':120}}
        validate_task(task,{'corpus_id':corpus_id,'split':'train','source_ids':list(registry)})
        rows.append({'task':task,'reference':{'schema_version':REFERENCE_VERSION,'task_id':tid,'status':'pending','reviewer_ids':[],
                     'facts':[],'evidence_packages':[],'family':family},'family':family,'domain':domain,
                     'candidate_source_ids':candidates,'suitability_risk':risk,'status':'candidate_unreviewed_family_split_not_frozen','semantic_gold':False})
    build_corpus(docs,out/'authoring-corpus',corpus_id,'train')
    write_jsonl(out/'sources.private.jsonl',sources);write_json(out/'case-authoring.private.json',rows)
    write_json(out/'source-family-review.private.json',relations);write_json(out/'acquisition-rejections.private.json',rejected)
    # Deliberately omit public tasks and frozen scope: these sources are not cleared for teacher use.
    summary={'schema_version':'av-train-authoring-pool.v0','case_count':len(rows),'av_source_count':4,
             'text_source_count':len(sources)-4,'source_count':len(sources),'source_seconds':sum(r['duration_s'] for r in sources if r['media_type']=='audio_video'),
             'scope_split':'train_candidate','adjudicated_reference_count':0,'retained_sft_rows':0,
             'train_split_frozen':False,'teacher_rollout_allowed':False,'same_byte_overlap_with_dev':0,
             'independent_recording_adjudication':'pending','status':'new_source_authoring_candidates_ready_for_review'}
    write_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',action='append',required=True)
    p.add_argument('--reserved-sources',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();print(json.dumps(prepare_pool(a.manifest,a.reserved_sources,a.out_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

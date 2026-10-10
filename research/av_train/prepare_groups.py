"""Build arbitrary authoring task/source groups from explicit, versioned specs.

Acquisition query labels are not semantic labels. All accepted AV bytes can enter
the authoring corpus, while explicit task specs choose review groups. This module
never freezes source families, approves facts, or exports runnable actor tasks.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from research.av_bench.contracts import TASK_VERSION,REFERENCE_VERSION,FAMILIES,json_copy,nonempty,validate_task
from research.av_data.ingest import ingest_media
from research.av_data.media_clock import sha256_file
from research.av_data.web_sources import ingest_web_snapshot,web_corpus_document
from research.av_data.build_research_batch import write_json,write_jsonl
from research.av_search.corpus import build_corpus
from .prepare_pool import screen_candidates

SPEC_FIELDS={'schema_version','corpus_id','budget_recipe','tasks'}
TASK_SPEC_FIELDS={'task_id','family','domain','source_group','candidate_page_ids','initial_page_ids',
                  'question','target_questions','suitability_risk'}


def validate_group_specs(specs,seeds):
    s=json_copy(specs)
    if not isinstance(s,dict) or set(s)!=SPEC_FIELDS or s['schema_version']!='av-train-group-specs.v1':
        raise ValueError('Invalid group spec schema')
    nonempty(s['corpus_id'],'corpus_id')
    if not isinstance(s['tasks'],list) or not s['tasks']:raise ValueError('Require authoring tasks')
    pages={r['page_id'] for r in seeds};task_ids=set()
    # Public task validation also checks budget and atomic target IDs without using gold.
    dummy_sid='src_'+'a'*24
    for row in s['tasks']:
        if not isinstance(row,dict) or set(row)!=TASK_SPEC_FIELDS:raise ValueError('Invalid task spec schema')
        for key in ('task_id','domain','source_group','question','suitability_risk'):nonempty(row[key],key)
        if row['task_id'] in task_ids:raise ValueError('Duplicate task ID')
        task_ids.add(row['task_id'])
        if row['family'] not in FAMILIES:raise ValueError('Unknown task family')
        candidates=row['candidate_page_ids'];initial=row['initial_page_ids']
        if not isinstance(candidates,list) or not candidates or any(isinstance(p,bool) or not isinstance(p,int) for p in candidates):
            raise ValueError('Invalid candidate pages')
        if len(candidates)!=len(set(candidates)):raise ValueError('Duplicate candidate page')
        if not set(candidates)<=pages:raise ValueError('Task references unavailable page after acquisition/split screening')
        if not isinstance(initial,list) or len(initial)>2 or any(isinstance(p,bool) or not isinstance(p,int) for p in initial) or len(initial)!=len(set(initial)) or not set(initial)<=set(candidates):
            raise ValueError('Invalid initial pages outside source group')
        if row['family'] in {'source_discovery','cross_source_comparison','research_composition'} and len(candidates)<2:
            raise ValueError('Search/comparison/composition group requires at least two AV candidates')
        task={'schema_version':TASK_VERSION,'task_id':row['task_id'],'question':row['question'],
              'initial_sources':[],'target_questions':row['target_questions'],'corpus_id':s['corpus_id'],
              'split':'train','budget_recipe':s['budget_recipe']}
        validate_task(task,{'corpus_id':s['corpus_id'],'split':'train','source_ids':[dummy_sid]})
    return s


def prepare_group_pool(manifest_paths,reserved_sources_path,task_specs_path,out_dir):
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    reserved=[json.loads(l) for l in Path(reserved_sources_path).read_text().splitlines() if l]
    manifests=[json.loads(Path(p).read_text()) for p in manifest_paths]
    seeds,rejected=screen_candidates(manifests,{r['asset_sha256'] for r in reserved})
    if not seeds:raise ValueError('No non-reserved accepted AV sources')
    if len({r['page_id'] for r in seeds})!=len(seeds):raise ValueError('Duplicate acquisition page with different bytes requires revision review')
    specs=validate_group_specs(json.loads(Path(task_specs_path).read_text()),seeds)
    for seed in seeds:
        if sha256_file(seed['asset_path'])!=seed['sha256']:raise ValueError('Acquired asset hash mismatch')
    out.mkdir(parents=True,exist_ok=False)
    sources=[];docs=[];relations=[];page_to_source={}
    for seed in seeds:
        metadata={k:v for k,v in seed['public_metadata'].items() if k in {'title','description','creator','license','license_url'}}
        source=ingest_media(seed['asset_path'],seed['origin_url'],out/'originals',metadata,
                           rights_basis='publisher_declared:'+metadata.get('license','unknown'),redistribution_scope='metadata_only')
        sources.append(source);page_to_source[seed['page_id']]=source['source_id']
        relations.append({'source_id':source['source_id'],'acquisition_page_id':seed['page_id'],
                          'family_review_status':'pending','recording_id':None,'event_id':None,
                          'candidate_split':'train','split_frozen':False})
        doc={k:source[k] for k in ('source_id','origin_url','asset_sha256','asset_path','media_type','duration_s','public_metadata')}
        doc.update(text='\n'.join(f'{k}: {v}' for k,v in metadata.items()),text_kind='public_metadata');docs.append(doc)
        if seed.get('description_page_html_path'):
            page=ingest_web_snapshot(seed['description_page_html_path'],seed['origin_url'],out/'snapshots',metadata,
                                    fetched_at=seed['fetched_at'],rights_basis='publisher_description_snapshot',redistribution_scope='metadata_only')
            sources.append(page);docs.append(web_corpus_document(page))
            relations.append({'source_id':page['source_id'],'related_recording_id':source['source_id'],
                              'relation':'publisher_description_not_independent_recording','family_review_status':'pending',
                              'candidate_split':'train','split_frozen':False})
    rows=[];groups={};registry={r['source_id']:r for r in sources}
    for spec in specs['tasks']:
        candidates=[page_to_source[p] for p in spec['candidate_page_ids']]
        task={'schema_version':TASK_VERSION,'task_id':spec['task_id'],'question':spec['question'],
              'initial_sources':[page_to_source[p] for p in spec['initial_page_ids']],
              'target_questions':spec['target_questions'],'corpus_id':specs['corpus_id'],'split':'train',
              'budget_recipe':specs['budget_recipe']}
        validate_task(task,{'corpus_id':specs['corpus_id'],'split':'train','source_ids':list(registry)})
        rows.append({'task':task,'reference':{'schema_version':REFERENCE_VERSION,'task_id':task['task_id'],
                    'status':'pending','reviewer_ids':[],'facts':[],'evidence_packages':[],'family':spec['family']},
                     'family':spec['family'],'domain':spec['domain'],'source_group':spec['source_group'],
                     'candidate_source_ids':candidates,'suitability_risk':spec['suitability_risk'],
                     'status':'candidate_unreviewed_family_split_not_frozen','semantic_gold':False})
        group=groups.setdefault(spec['source_group'],{'group_id':spec['source_group'],'candidate_source_ids':[],
                                                      'task_ids':[],'semantic_review_status':'pending'})
        group['candidate_source_ids']=sorted(set(group['candidate_source_ids'])|set(candidates))
        group['task_ids'].append(task['task_id'])
    build_corpus(docs,out/'authoring-corpus',specs['corpus_id'],'train')
    write_jsonl(out/'sources.private.jsonl',sources);write_json(out/'case-authoring.private.json',rows)
    write_json(out/'source-family-review.private.json',relations);write_json(out/'source-groups.private.json',list(groups.values()))
    write_json(out/'acquisition-rejections.private.json',rejected)
    write_json(out/'input-lineage.private.json',{'task_specs_sha256':sha256_file(task_specs_path),
        'reserved_sources_sha256':sha256_file(reserved_sources_path),
        'manifest_sha256':[sha256_file(p) for p in manifest_paths],
        'page_to_source':page_to_source,'authoring_specs':specs})
    summary={'schema_version':'av-train-authoring-pool.v1','corpus_id':specs['corpus_id'],'case_count':len(rows),
             'family_counts':dict(Counter(r['family'] for r in rows)),'source_group_count':len(groups),
             'av_source_count':len(seeds),'text_source_count':len(sources)-len(seeds),'source_count':len(sources),
             'source_seconds':sum(r['duration_s'] for r in sources if r['media_type']=='audio_video'),
             'scope_split':'train_candidate','adjudicated_reference_count':0,'retained_sft_rows':0,
             'train_split_frozen':False,'teacher_rollout_allowed':False,'same_byte_overlap_with_dev':0,
             'independent_recording_adjudication':'pending','status':'source_group_candidates_ready_for_review'}
    write_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',action='append',required=True)
    p.add_argument('--reserved-sources',required=True);p.add_argument('--task-specs',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();print(json.dumps(prepare_group_pool(a.manifest,a.reserved_sources,a.task_specs,a.out_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Select nested pilot sets using existing hash-bound native trainer receipts."""
from collections import Counter
from copy import deepcopy
import hashlib,json,re
from pathlib import Path
from .native_sft import file_sha,digest,SCHEMA,materialize_configs
from .scale_searchqa import verified_pool,rows
from .learning_pilot import verify_row_audit
from .source_isolation import canonical_url
from .teacher_rollouts import atomic_json,now

ALLOWED={'main_search_candidate','not_content_audited'}
def select_questions(records,quality,previous_ids,size,seed):
 if size<1:raise ValueError('positive size required')
 seen_ids=set();seen_pairs=set();eligible={};counts=Counter()
 for original in records:
  row=deepcopy(original);tid=row['task_id'];pair=(row['recording'],re.sub(r'\s+','',row['question']).casefold())
  if tid in seen_ids or pair in seen_pairs:raise ValueError('duplicate question or recording/question pair')
  seen_ids.add(tid);seen_pairs.add(pair);q=quality.get(tid)
  if q:
   for k in ['question','input_media_sha256','trajectory_row_sha256','answer_only_row_sha256']:
    if q[k]!=row[k]:raise ValueError('quality binding mismatch: '+k)
   tier=q['quality_view']
  else:tier='not_content_audited'
  counts[tier]+=1;row['quality_view']=tier
  if tier in ALLOWED:eligible[tid]=row
 missing=set(previous_ids)-seen_ids
 if missing:raise ValueError('missing anchor IDs')
 if len(eligible)<size:raise ValueError('insufficient eligible questions')
 anchors=[tid for tid in dict.fromkeys(previous_ids) if tid in eligible]
 if len(anchors)>size:raise ValueError('anchors exceed requested size')
 selected=[eligible[tid] for tid in anchors];recordings=Counter(r['recording'] for r in selected);families=Counter(r['family_id'] for r in selected)
 def stable(row):return hashlib.sha256((str(seed)+'|'+row['task_id']).encode()).hexdigest()
 pending=[r for tid,r in eligible.items() if tid not in anchors]
 while len(selected)<size:
  row=min(pending,key=lambda r:(recordings[r['recording']],families[r['family_id']],stable(r)))
  pending.remove(row);selected.append(row);recordings[row['recording']]+=1;families[row['family_id']]+=1
 selected.sort(key=stable)
 return selected,{'available_questions':len(records),'eligible_questions':len(eligible),'quality_counts':dict(counts),'selected_questions':len(selected),'preserved_previous_ids':anchors,'excluded_previous_ids':[tid for tid in dict.fromkeys(previous_ids) if tid not in eligible],'selection_seed':seed,'unreviewed_silver_explicit':True,'human_gold':0}

def collect(production,quality_dir,reference_run):
 root=Path('/home/jovyan/OneSearch-VL');production=Path(production);quality_dir=Path(quality_dir);reference_run=Path(reference_run)
 exports=json.loads((production/'prior-exports.private.json').read_text())
 if (production/'export-index.private.json').exists():exports+=json.loads((production/'export-index.private.json').read_text())['exports']
 pool=verified_pool(exports);quality={r['task_id']:r for r in rows(quality_dir/'membership.private.jsonl')}
 previous=json.loads((reference_run/'train-members.private.json').read_text());oldmeta={r['task_id']:r for r in previous}
 old_res=json.loads((root/'research/runs/av-next-phase-20261014-v3/reservation.private.json').read_text());reserve=json.loads((production/'reservation.private.json').read_text())
 blocked_urls={canonical_url(u) for u in old_res['blocked_urls']+reserve['urls']};blocked_hashes=set(old_res['blocked_hashes'])|set(reserve['hashes']);blocked_ids=set(old_res['blocked_source_ids'])|set(old_res['family_members'])
 ev=json.loads((reference_run/'eval-members.private.json').read_text());assert file_sha(reference_run/'eval-members.private.json')==old_res['eval_members_sha256']
 eval_families={r['family_id'] for r in ev};eval_media={h for r in ev for h in r['media_hashes']}
 cache={};records=[];content={};audits={};seen=set();native=None
 def sha(path):
  path=str(path)
  if path not in cache:cache[path]=file_sha(path)
  return cache[path]
 for entry in pool['export_directories']:
  if not entry['cpu_audit_passed']:continue
  d=Path(entry['dataset_dir']);spec=json.loads((d/'summary.json').read_text());tr=json.loads((d/'trajectory.json').read_text());an=json.loads((d/'answer_only.json').read_text());lineage=json.loads((d/'lineage.private.json').read_text())
  ap=next(p for p in d.parent.glob('*/label-media-audit.private.json') if json.loads(p.read_text()).get('status')=='passed' and json.loads(p.read_text()).get('dataset_summary_sha256')==sha(d/'summary.json'))
  audit=json.loads(ap.read_text());audits[str(d)]=(str(ap),audit)
  if native is None:native=audit['native_source_sha256']
  assert native==audit['native_source_sha256']
  for name,h in native.items():assert sha(root/'SFT/src/llamafactory'/name)==h
  batch=Path(spec['batch_dir']);tasks={r['task_id']:r for r in rows(batch/'tasks-public.jsonl')};refs={r['task_id']:r for r in rows(batch/'answer-candidates.private.jsonl')}
  parents={r['task_id']:r for r in json.loads((batch/'clue-lineage.private.json').read_text())} if (batch/'clue-lineage.private.json').exists() else {}
  clues={r['clue_id']:r['source'] for r in json.loads((batch/'clue-bindings.private.json').read_text())}
  teacherplan=json.loads((Path(spec['episodes_dir']).parent/'plan.json').read_text())
  for i,line in enumerate(lineage):
   tid=line['task_id'];task=tasks[tid];ref=refs[tid];parent=parents.get(tid,{});clue=clues[task['initial_sources'][0]]
   recording=parent.get('acquisition_source_id') or parent.get('parent_source_id') or sha(tr[i]['videos'][0]);pair=(recording,re.sub(r'\s+','',task['question']).casefold())
   if pair in seen:continue
   seen.add(pair);family=parent.get('candidate_family_id') or oldmeta.get(tid,{}).get('family_id') or 'recording:'+recording
   assert not {recording,parent.get('parent_source_id'),ref.get('target_source_id')}&blocked_ids
   hashes={sha(p) for p in tr[i].get('videos',[])}|{sha(clue['asset_path'])};assert not hashes&(blocked_hashes|eval_media) and family not in eval_families
   trace=json.loads(Path(line['trajectory_path']).read_text());text=canonical_url(json.dumps(trace,ensure_ascii=False));assert not any(url in text for url in blocked_urls),'reserved source exposed in actual trace'
   assert sha(clue['asset_path'])==clue['asset_sha256']
   for receipt in line.get('audio_receipts',[]):assert sha(receipt['audio_path'])==receipt['audio_sha256']
   t=verify_row_audit(tr[i],line,audit,'trajectory',i);a=verify_row_audit(an[i],line,audit,'answer_only',i)
   assert 0<t['supervised_tokens']<t['total_tokens']<=audit['cutoff_len']
   row={'task_id':tid,'question':task['question'],'recording':recording,'family_id':family,'input_media_sha256':sha(tr[i]['videos'][0]),'trajectory_row_sha256':line['row_sha256'],'answer_only_row_sha256':line['answer_row_sha256'],'export_dir':str(d),'row_index':i,'batch_dir':str(batch),'index_dir':teacherplan['index_dir'],'task':task,'reference':ref,'source_path':clue['asset_path'],'source_sha256':clue['asset_sha256'],'media_hashes':sorted(hashes),'split':'train','teacher':line['teacher_identity']['model_id'],'scenario':ref.get('scenario','unknown'),'field_count':len(ref['answer_fields']),'trajectory_tokens':t['total_tokens'],'supervised_tokens':t['supervised_tokens'],'answer_tokens':a['total_tokens'],'answer_supervised_tokens':a['supervised_tokens'],'model_manifest':json.loads((production/'plan.json').read_text())['model_manifest']}
   records.append(row);content[tid]=(tr[i],an[i],line)
 assert len(records)==pool['cpu_verified_questions']
 return {'records':records,'quality':quality,'previous':previous,'pool':pool,'content':content,'audits':audits,'eval':ev,'production':str(production),'reference_run':str(reference_run),'quality_dir':str(quality_dir)}

def freeze(bundle,selected,out):
 out=Path(out);out.mkdir(exist_ok=False);data=out/'train-paired';data.mkdir();traj=[];answers=[];lineage=[];receipt_rows=[];coverage={};parents={}
 for row in selected:
  tr,an,line=bundle['content'][row['task_id']];j=len(traj);new=deepcopy(line);new.update(row_index=j,parent_row_index=row['row_index'],parent_export_dir=row['export_dir'],family_id=row['family_id']);traj.append(tr);answers.append(an);lineage.append(new)
  ap,audit=bundle['audits'][row['export_dir']];parents[ap]=file_sha(ap)
  for variant,actual in [('trajectory',tr),('answer_only',an)]:
   receipt=deepcopy(verify_row_audit(actual,line,audit,variant,row['row_index']));receipt.update(row_index=j,parent_row_index=row['row_index'],parent_audit_path=ap,parent_audit_sha256=parents[ap]);receipt_rows.append(receipt)
  for receipt in audit['video_sampling_coverage_receipts']:
   if receipt['video_path'] in tr['videos']+an['videos']:
    assert file_sha(receipt['video_path'])==receipt['video_sha256'];coverage[receipt['video_path']]=receipt
 for name,value in [('trajectory.json',traj),('answer_only.json',answers),('lineage.private.json',lineage)]:atomic_json(data/name,value)
 (data/'dataset_info.json').write_bytes((Path(selected[0]['export_dir'])/'dataset_info.json').read_bytes())
 summary={'schema_version':SCHEMA,'audit_only':False,'source_split':'train','retained_count':len(selected),'distinct_task_count':len(selected),'use_audio_in_video':True,'media_mode':'joint_av','retained_pool_sha256':digest(lineage),'trajectory_rows_sha256':digest(traj),'answer_rows_sha256':digest(answers),'benchmark_gold_verified':False,'training_ready':False,'cpu_label_media_audit':'inherited exact unchanged row receipts','audit_aggregation':True}
 atomic_json(data/'summary.json',summary)
 combined=deepcopy(next(iter(bundle['audits'].values()))[1]);combined.update(dataset_summary_sha256=file_sha(data/'summary.json'),rows=receipt_rows,video_sampling_coverage_receipts=list(coverage.values()),audit_aggregation=True,audit_execution='exact unchanged rows; original CPU decoding receipts reused',parent_audits=[{'path':p,'sha256':h} for p,h in parents.items()])
 ad=out/'train-cpu-audit-composed';ad.mkdir();atomic_json(ad/'label-media-audit.private.json',combined)
 materialize_configs(data,ad,selected[0]['model_manifest'],out/'trainer-configs')
 # Full Thinker uses the original trainer. LoRA pilots remain separate artifacts.
 import yaml
 for variant in ['trajectory','answer_only']:
  path=out/'trainer-configs'/('train-'+variant+'.yaml');cfg=yaml.safe_load(path.read_text());cfg={k:v for k,v in cfg.items() if not k.startswith('lora_')}
  cfg.update(finetuning_type='full',freeze_vision_tower=True,freeze_multi_modal_projector=True,freeze_language_model=False,flash_attn='sdpa',learning_rate=1e-5,num_train_epochs=2.,gradient_accumulation_steps=4,save_steps=250,save_total_limit=2,save_only_model=False,seed=42,data_seed=42,output_dir=str(out/(variant+'-thinker')))
  path.write_text(yaml.safe_dump(cfg,sort_keys=False))
 atomic_json(out/'train-members.private.json',selected);atomic_json(out/'eval-members.private.json',bundle['eval'])
 profile={'questions':len(selected),'recordings':len({r['recording'] for r in selected}),'families':len({r['family_id'] for r in selected}),'quality_views':dict(Counter(r['quality_view'] for r in selected)),'teachers':dict(Counter(r['teacher'] for r in selected)),'scenario_hints_not_content_labels':dict(Counter(r['scenario'] for r in selected)),'context_tokens_per_epoch':sum(r['trajectory_tokens'] for r in selected),'supervised_tokens_per_epoch':sum(r['supervised_tokens'] for r in selected),'answer_context_tokens_per_epoch':sum(r['answer_tokens'] for r in selected),'answer_supervised_tokens_per_epoch':sum(r['answer_supervised_tokens'] for r in selected),'max_context_tokens':max(r['trajectory_tokens'] for r in selected)}
 atomic_json(out/'token-profile.json',profile)
 result={'status':'data_frozen','at_utc':now(),'train_questions':len(selected),'train_members_sha256':file_sha(out/'train-members.private.json'),'eval_members_sha256':file_sha(out/'eval-members.private.json'),'dataset_summary_sha256':file_sha(data/'summary.json'),'cpu_audit_sha256':file_sha(ad/'label-media-audit.private.json'),'quality_membership_sha256':file_sha(Path(bundle['quality_dir'])/'membership.private.jsonl'),'source_pool_questions':bundle['pool']['cpu_verified_questions'],'source_production':bundle['production'],'source_observation_reserved_overlap':0,'known_excluded_questions':0,'inherited_audits_not_new_decoding':True,'gpu_resource_smoke_required':True,'full_thinker_training_executed':False,'training_ready':False,'gpu_jobs':0,'human_gold':0,'profile':profile}
 atomic_json(out/'data-freeze.json',result);return result

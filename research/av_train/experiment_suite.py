"""Versioned inputs for learning, modality and construction experiments.

Only reads existing generation and native trainer receipts. This module never
calls a teacher, launches GPU work, or promotes candidate annotations to gold.
"""
from collections import Counter
from pathlib import Path
import hashlib,json,re
from .native_sft import file_sha,digest
from .teacher_rollouts import atomic_json,now

OPERATIONS=['single_anchor_lookup','multi_hop_retrieval','knowledge_conditioned_count','multi_anchor_join','multi_anchor_arithmetic','multi_anchor_comparison']
MODALITY_ROLES=['audio_primary','joint_av','visual_primary','speech_text_control']

def normalized_question(value):
 return re.sub(r'\s+','',value).casefold()

def check_benchmark_members(rows,train):
 seen=set();pairs=set()
 train_recordings={r['recording'] for r in train};train_families={r['family_id'] for r in train}
 train_media={h for r in train for h in r['media_hashes']}
 for row in rows:
  pair=(row['recording'],normalized_question(row['question']))
  if row['task_id'] in seen or pair in pairs:raise ValueError('duplicate benchmark question')
  if row['recording'] in train_recordings or row['family_id'] in train_families or train_media.intersection(row['media_hashes']):raise ValueError('benchmark train source overlap')
  seen.add(row['task_id']);pairs.add(pair)
 return {'questions':len(rows),'recordings':len({r['recording'] for r in rows}),'families':len({r['family_id'] for r in rows}),'train_recording_family_or_media_overlap':0,'human_gold':0}

def construction_sample(rows,size,seed):
 eligible=[r for r in rows if r['quality_view'] in {'main_search_candidate','not_content_audited'}]
 eligible.sort(key=lambda r:hashlib.sha256((str(seed)+'|'+r['task_id']).encode()).hexdigest())
 chosen=[];recordings=set();families=set()
 for row in eligible:
  if row['recording'] in recordings or row['family_id'] in families:continue
  chosen.append(row);recordings.add(row['recording']);families.add(row['family_id'])
  if len(chosen)==size:return chosen
 raise ValueError('insufficient independent construction sources')

def experiment_matrix():
 arms=[]
 for n in [500,2000]:
  for supervision in ['trajectory','answer_only']:
   arms.append({'id':f'full_{n}_{supervision}','questions':n,'supervision':supervision,'finetuning_type':'full','epochs':2,'effective_batch':4,'expected_updates':n*2//4,'learning_rate':1e-5,'freeze_av_towers_and_projectors':True,'start_from_same_pinned_base':True})
 return {'schema':'onesearch-av-three-axis-experiments.v1','created_at':now(),
  'model':{'id':'Qwen/Qwen2.5-Omni-7B','revision':'ae9e1690543ffd5c0221dc27f79834d0294cba00','entrypoint':'SFT/src/train.py','talker_excluded':True},
  'axis_a':{'question':'Does complete search-trajectory supervision and data scale improve task success?',
   'training_arms':arms,'baselines':['direct_base','dev_selected_prompted_search'],
   'prompt_selection':{'split':'dev_only','heldout_for_selection':False,'required_protocol_probe':'Separate tool-protocol execution from AV clue-to-query ability','minimum_validity':'Report actual tool execution and final completion before treating prompted arm as an effective search baseline'},
   'comparison_policy':{'same_questions_for_supervision_controls':True,'clean500_nested_in_clean2000':True,'primary_scaling':'same epochs and recipe; 2K has 4x planned updates','report_context_and_supervised_tokens':True,'claim_compute_matched':False,'historical_lora500_is_separate':True},
   'gpu_preflight':{'world_size_candidate':4,'min_memory_gib_per_gpu_candidate':80,'deepspeed':'zero2','gradient_accumulation_for_candidate_world_size':1,'required':'actual AV full Thinker forward/backward and 3 optimizer updates; decoder change, AV freeze and checkpoint readback'},
   'metrics':['answer_success','supported_success','target_source_returned','effective_tool_calls','final_completion','supervised_tokens','optimizer_updates','gpu_hours']},
  'axis_b':{'question':'When do sound and visual context improve retrieval and task success?',
   'inputs':['native_av','muted_video','audio_only','video_plus_asr'],
   'modality_role_labels':MODALITY_ROLES,'research_operation_labels':OPERATIONS,
   'input_contract':{'all_turns':'Intervention also applies to every inspect_av return','muted_video':'No model-visible audio; preserve encoded video packets','audio_only':'Original audio without video frames or image/text captions derived from them','video_plus_asr':'Muted video and transcript from same interval; ASR model and transcript fixed for all compared policies'},
   'annotation_selection':'Do not select or relabel benchmark by student wins/losses',
   'metrics':['paired_answer_success','paired_supported_success','target_source_returned','query_or_entity_change','tool_budget','family_bootstrap_interval'],
   'formal_gold_requirements':['human fact and source verification','input interval and identity link','source-family separation','modality role annotation'],
   'human_gold':0},
  'axis_c':{'question':'Which data construction and sampling choices improve data quality and learning?',
   'construction_pilot':{'paired_sources':32,'arms':['native_av_plus_publisher','publisher_text_only'],'same_source_material':True,'same_teacher_and_output_budget':True,'same_fact_support_rules':True,'max_fresh_calls_if_both_arms_needed':64,'reuse_policy':'Reuse successful AV-author receipt only with exact question/media/publisher binding; never replay failed or unknown calls','status':'input selection; model generation not launched'},
   'blind_quality_review':['actual input match','reasonable clue-to-entity search path','external information needed','answer support','distinct facts and operation coverage'],
   'learning_controls':['natural_sample_500_trajectory','modality_stratified_500_trajectory'],
   'sampling_budget':'Match question count and teacher identity where feasible; report and match supervised-token budget in the data-method comparison',
   'role_labels':'Human labels or explicitly named automatic silver labels; never infer audio contribution from domain or student outcome',
   'metrics':['accepted_questions_per_source','search_needed_rate','input_answer_support','audited_main_search_yield','downstream_task_success','supervised_tokens']},
  'status':{'new_model_api_calls':0,'new_gpu_jobs':0,'human_gold':0,'formal_training_not_launched':True}}

def freeze_benchmark(batches,train,out_dir):
 from .scale_searchqa import rows
 from .source_isolation import canonical_url
 out=Path(out_dir);out.mkdir(exist_ok=False,parents=True);members=[];sources=[]
 for value in batches:
  b=Path(value);tasks=rows(b/'tasks-public.jsonl');refs={r['task_id']:r for r in rows(b/'answer-candidates.private.jsonl')}
  lineage={r['task_id']:r for r in json.loads((b/'clue-lineage.private.json').read_text())};bindings={r['clue_id']:r['source'] for r in json.loads((b/'clue-bindings.private.json').read_text())}
  corpus=b/'corpus';docs={r['source_id']:r for r in rows(corpus/'documents.jsonl')}
  for t in tasks:
   tid=t['task_id'];line=lineage[tid];ref=refs[tid];binding=bindings[t['initial_sources'][0]];target=docs[ref['target_source_id']]
   assert ref['benchmark_gold'] is False and t['split']=='dev'
   assert file_sha(binding['asset_path'])==binding['asset_sha256']
   parent_path=line['native_crop']['source_path'];assert file_sha(parent_path)==target['asset_sha256']
   for quote in ref['support_quotes']:
    doc=docs[quote['source_id']];body=doc['text'];assert hashlib.sha256(body.encode()).hexdigest()==doc['text_sha256'];assert body[quote['start_char']:quote['end_char']]==quote['quote']
   row={'task_id':tid,'question':t['question'],'question_sha256':hashlib.sha256(t['question'].encode()).hexdigest(),'recording':line['acquisition_source_id'],'family_id':line['candidate_family_id'],'media_hashes':sorted({binding['asset_sha256'],target['asset_sha256']}),'input_path':binding['asset_path'],'input_sha256':binding['asset_sha256'],'parent_path':parent_path,'parent_sha256':target['asset_sha256'],'source_url':canonical_url(target['origin_url']),'interval':line['interval'],'batch_dir':str(b),'reference':ref,'task':t,'task_row_sha256':digest(t),'reference_sha256':digest(ref),'scenario_hint_not_modality':ref['scenario'],'human_gold':False}
   members.append(row)
  sources.append({'batch':str(b),'questions':len(tasks),'file_sha256':{n:file_sha(b/n) for n in ['tasks-public.jsonl','answer-candidates.private.jsonl','clue-lineage.private.json','clue-bindings.private.json']}})
 summary=check_benchmark_members(members,train)
 # Search actual observations as well as declared train metadata.
 texts=[];seen=set()
 for r in train:
  e=Path(r['export_dir']);lineage_path=e/'lineage.private.json'
  if str(lineage_path) in seen:continue
  seen.add(str(lineage_path))
  for line in json.loads(lineage_path.read_text()):texts.append(canonical_url(Path(line['trajectory_path']).read_text()))
 all_text='\n'.join(texts);exposed=[m['task_id'] for m in members if m['source_url'] and m['source_url'] in all_text]
 if exposed:raise ValueError('Benchmark target URL exposed in actual train trajectory')
 summary.update(at_utc=now(),status='candidate_membership_frozen_not_gold',actual_train_target_url_exposure=0,source_batches=sources,scenario_hints=dict(Counter(m['scenario_hint_not_modality'] for m in members)),modality_labels_pending=len(members),operation_labels_pending=len(members))
 (out/'members.private.jsonl').write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in members))
 template=[]
 for m in members:
  template.append({'task_id':m['task_id'],'question_sha256':m['question_sha256'],'input_sha256':m['input_sha256'],'reference_sha256':m['reference_sha256'],'reviewer_id':None,'fact_support':'pending','input_interval_identity':'pending','search_necessity':'pending','modality_role':'pending','research_operation':'pending','note':'','human_gold':False})
 (out/'human-annotation-template.private.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in template))
 summary['membership_sha256']=file_sha(out/'members.private.jsonl');atomic_json(out/'summary.json',summary)
 return summary,members

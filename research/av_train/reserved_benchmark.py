"""Author dev candidates only from families reserved before SFT production.

Reuses native AV authoring/materialization. No student-score selection, paid
replay, training export, or promotion to human gold. Existing reservations and
production source freezes are never mutated.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json,os,time,socket,fcntl,re
from research.av_train.source_isolation import canonical_url


def select_reserved_families(reserved,seeds,families,*,limit,attempted_source_ids=(),blocked_urls=(),blocked_hashes=()):
 if type(limit) is not int or not 1<=limit<=1000:raise ValueError('Explicit bounded benchmark recording cap required')
 groups={}
 for group in families:
  for sid in group['source_ids']:
   if sid in groups:raise ValueError('Ambiguous source family')
   groups[sid]=group['candidate_family_id']
 by_id={s['source_id']:s for s in seeds}
 if len(by_id)!=len(seeds):raise ValueError('Repeated source IDs')
 if any(s['source_id'] not in groups or s['source_id'] not in by_id for s in reserved):raise ValueError('Every reservation requires an inventoried source family')
 urls={canonical_url(u) for u in blocked_urls};hashes=set(blocked_hashes)
 blocked={groups[s] for s in attempted_source_ids if s in groups}
 for s in seeds:
  if canonical_url(s['origin_url']) in urls or s['sha256'] in hashes:
   if s['source_id'] not in groups:raise ValueError('Blocked source lacks family')
   blocked.add(groups[s['source_id']])
 by_category=defaultdict(list)
 for reservation in reserved:
  s=by_id[reservation['source_id']];family=groups[s['source_id']]
  if (family in blocked or s.get('status')!='candidate_unreviewed_av' or not s.get('description_page_html_path')):continue
  by_category[s.get('candidate_domain','unknown').split('_')[0]].append({**s,'candidate_family_id':family})
 selected=[];seen=set()
 for i in range(max((len(x) for x in by_category.values()),default=0)):
  for category,items in sorted(by_category.items()):
   if i>=len(items):continue
   s=items[i];f=s['candidate_family_id']
   if f in seen:continue
   selected.append(s);seen.add(f)
   if len(selected)>=limit:break
  if len(selected)>=limit:break
 # The original author selector also operates on family identity. Leave aliases
 # of selected families available, and exclude every other family from authoring.
 used=sorted(s['source_id'] for s in seeds if groups.get(s['source_id']) not in seen)
 return {'selected':selected,'selected_family_ids':sorted(seen),'used_source_ids':used,'blocked_family_ids':sorted(blocked),'human_gold':False,'selection_basis':'reserved family identity; round-robin acquisition query category; no student outcomes','family_independence':'automatic grouping, human adjudication pending'}


def read_training_views(export_dirs):
 """Read actual traces through verified, current export lineage hashes."""
 import hashlib
 paths=set();hashes=set();urls=set()
 for directory in export_dirs:
  data=Path(directory);lineage=data/'lineage.private.json'
  for row in json.loads(lineage.read_text()):
   trace=Path(row['trajectory_path']);raw=trace.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=row['trajectory_sha256']:raise ValueError('Training trace changed after retained export')
   paths.add(str(trace));urls.update(canonical_url(u.rstrip('.,;')) for u in re.findall(r'https?://[^\s<>"\\]+',raw.decode()))
   hashes.update(x['source_sha256'] for x in row.get('audio_receipts',[]))
 if not paths:raise ValueError('No actual training trace provenance; view isolation unproven')
 return {'trace_paths':sorted(paths),'observed_urls':sorted(urls),'media_hashes':sorted(hashes)}


def observed_training_blocks(production):
 from research.av_train.scale_searchqa import verified_pool
 p=Path(production);exports=json.loads((p/'prior-exports.private.json').read_text())
 exports += [str(x.parent.resolve()) for x in p.glob('**/silver-sft/summary.json')]
 pool=verified_pool(exports)
 views=read_training_views([x['dataset_dir'] for x in pool['export_directories'] if x['cpu_audit_passed']])
 return {'cpu_qualified_questions':pool['cpu_verified_questions'],**views}


def paid_source_ids(root):
 ids=set()
 for p in Path(root).glob('**/results/wc_*.json'):
  d=json.loads(p.read_text())
  if d.get('api_calls',0)>0:ids.add(d.get('source_id',p.stem))
 return ids



def ready_reserved_cohorts(production,*,completed=(),cohort_name=None):
 """A cohort is visible only after its source reservations are written."""
 return [c for c in sorted(Path(production).glob('cohort-*')) if c.name not in set(completed)
  and (cohort_name is None or c.name==cohort_name)
  and all((c/n).exists() for n in ['benchmark-reserved-sources.private.json','inventory/inventory.private.json','reserved-registry.private.jsonl','combined-manifest.private.json'])]

def run(plan_file):
 from research.av_train.question_authoring import prepare,run as author,materialize
 from research.av_train.teacher_rollouts import atomic_json,now
 from research.av_train.native_sft import file_sha
 from research.av_train.teacher_telemetry import install
 from research.av_train.provider_resilience import install as resilience,BurstProviderGuard
 from research.av_data.training_sample_review import build
 p=json.loads(Path(plan_file).read_text());out=Path(plan_file).resolve().parent;production=Path(p['production_run']);root=production.parents[2]
 lock=(out/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pid=os.getpid();deadline=time.time()+p['max_elapsed_hours']*3600
 atomic_json(out/'worker-start.json',{'pid':pid,'host':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'process_start_ticks':Path('/proc/self/stat').read_text().split()[21],'at_utc':now(),'deadline_epoch':deadline})
 def state(stage,**extra):
  atomic_json(out/'state.json',{'status':'running','stage':stage,'at_utc':now(),'human_gold':0,'gpu_jobs':0,'paid_replays':0,**extra})
 def check():
  if time.time()>deadline:raise TimeoutError('Benchmark producer budget exhausted')
  if BurstProviderGuard(out).stopped:raise RuntimeError('Provider stopped; preserve attempts')
 try:
  state('waiting_for_pre_reserved_cohort')
  while True:
   check();ready=ready_reserved_cohorts(production,cohort_name=p.get('cohort_name'))
   if ready:break
   worker=json.loads((production/'worker-start.json').read_text())
   alive=Path('/proc',str(worker['pid'])).exists()
   if not alive:raise RuntimeError('Training producer ended before any reserved cohort was ready')
   time.sleep(30)
  cohort=ready[0];state('audit_reserved_source_isolation',cohort=cohort.name)
  seeds=json.loads((cohort/'combined-manifest.private.json').read_text())['sources'];inventory=json.loads((cohort/'inventory/inventory.private.json').read_text())
  blocks=observed_training_blocks(production);atomic_json(out/'training-view-blocks.private.json',blocks)
  selection=select_reserved_families(json.loads((cohort/'benchmark-reserved-sources.private.json').read_text()),seeds,inventory['candidate_families'],limit=p['max_author_api_calls'],attempted_source_ids=paid_source_ids(root/'research/runs'),blocked_urls=blocks['observed_urls'],blocked_hashes=blocks['media_hashes'])
  atomic_json(out/'selection.private.json',selection)
  if not selection['selected']:raise ValueError('No untouched independent reserved benchmark family')
  selected_families=set(selection['selected_family_ids']);groups={s:g['candidate_family_id'] for g in inventory['candidate_families'] for s in g['source_ids']}
  reservation=[json.loads(l) for l in (cohort/'reserved-registry.private.jsonl').read_text().splitlines() if l]
  kept=[s for s in reservation if groups.get(s['source_id']) not in selected_families]
  registry=out/'other-reservations.private.jsonl';registry.write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in kept))
  used=out/'used-sources.private.json';atomic_json(used,[{'acquisition_source_id':s} for s in selection['used_source_ids']])
  os.environ.update(dict(l.split('=',1) for l in Path(p['teacher_env']).read_text().splitlines() if l))
  os.environ.update(ONESEARCH_2K_TEACHER_MODEL=p['teacher_model'],ONESEARCH_TEACHER_STOP_FILE=str(out/'provider-stop.json'),ONESEARCH_TEACHER_PROFILE_DIR=str(out/'api-timing'),ONESEARCH_TEACHER_TRANSIENT_LIMIT='3',ONESEARCH_TEACHER_PROFILE_STAGE='reserved_benchmark_author')
  install();resilience();a=out/'authoring'
  state('prepare_actual_native_av',selected_families=len(selected_families))
  prepare(cohort/'combined-manifest.private.json',cohort/'inventory/inventory.private.json',registry,used,a,p['teacher_config'],len(selected_families))
  inputs=json.loads((a/'inputs.private.json').read_text());assert {groups[s['source_id']] for s in inputs}<=selected_families
  assert not {s['source_id'] for s in inputs}&paid_source_ids(root/'research/runs')
  prompt=(Path(p['author_prompt_file']).read_text() if p.get('author_prompt_file') else (a/'prompt.txt').read_text()).replace('Create training questions','Propose benchmark development CANDIDATES')+'\nThese are pending human-review candidates. The question must require identifying a concrete source or retrieving an external attribute. Species/name recognition alone is not sufficient for the main search set. Prefer substantive attributes: location, origin, work/composer, mechanism/model or event identity. Avoid details impossible to link from this clip, uploader/account/license/date trivia and descriptive clues that reveal the answer. Sound can be primary, joint, helpful or a visual-first control, but describe only actual sound and never assume all sources need audio. Acquisition query category is not a verified content label. Omit unanswerable or ambiguous tasks. Human reviewers will decide search necessity, factual support and modality dependence.\n'
  (a/'prompt.txt').write_text(prompt);spec=json.loads((a/'plan.json').read_text());spec['prompt_sha256']=file_sha(a/'prompt.txt');atomic_json(a/'plan.json',spec)
  check();state('native_av_authoring',prepared_recordings=len(inputs),workers=p['workers'])
  result=author(a,workers=p['workers'],max_sources=p['max_author_api_calls']);atomic_json(out/'author-result.json',result);check()
  if not result['authored_question_candidates']:
   atomic_json(out/'state.json',{'status':'no_candidates_from_completed_attempts','author_summary':result,'human_gold':0});return
  state('materialize_benchmark_dev')
  summary=materialize(a,cohort/'combined-manifest.private.json',cohort/'inventory/inventory.private.json',registry,out/'batch',split='dev');check()
  state('render_playable_review',tasks=summary['tasks'])
  build([out/'batch'],out/'review',split='dev',limit=min(24,summary['recording_candidates']))
  atomic_json(out/'state.json',{'status':'candidates_ready_for_human_review','at_utc':now(),'batch_summary':summary,'author_summary':result,'human_gold':0,'gpu_jobs':0,'paid_replays':0})
 except BaseException as exc:
  atomic_json(out/'state.json',{'status':'stopped','error_type':type(exc).__name__,'at_utc':now(),'human_gold':0,'paid_replays':0});raise

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);args=parser.parse_args();run(args.plan)

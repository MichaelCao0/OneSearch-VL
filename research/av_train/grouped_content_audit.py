"""Blind grouped quality audit; one native AV request per source."""
from pathlib import Path
from collections import Counter
import json,os,socket,time,fcntl,sys
from dataclasses import replace
from research.av_train.candidate_content_audit import validate_audit,SYSTEM as SINGLE_SYSTEM
from research.av_train.teacher_rollouts import atomic_json,now
from research.av_train.native_sft import file_sha

SYSTEM=SINGLE_SYSTEM+"\nEvaluate every question independently. Question identifiers are opaque. Do not infer which construction method wrote a question. Return exactly {\"verdicts\":[...]} with one entry per supplied question_id. Each entry must contain question_id and exactly the audit fields listed above. Do not add group-level conclusions."

def validate_group(value,ids):
 if not isinstance(value,dict) or set(value)!={'verdicts'} or not isinstance(value['verdicts'],list):raise ValueError('closed grouped verdict required')
 actual=[];out=[]
 for row in value['verdicts']:
  if not isinstance(row,dict) or not isinstance(row.get('question_id'),str):raise ValueError('question identity required')
  tid=row['question_id'];actual.append(tid);verdict=validate_audit({k:v for k,v in row.items() if k!='question_id'});out.append({'question_id':tid,'verdict':verdict})
 if len(actual)!=len(set(actual)) or set(actual)!=set(ids):raise ValueError('group membership mismatch')
 return out

def usable_search(v):
 return v['decision']=='keep' and v['media_matches_question'] and v['answer_fields_supported_by_publisher'] and not v['question_leaks_identifying_clue'] and v['search_requirement'] in {'external_attribute','source_identification'} and v['specificity']=='linked_instance'

def run(plan_file):
 from research.av_train.provider_resilience import install,BurstProviderGuard
 from research.av_train.question_authoring import parse_json_output
 from research.av_train.generation_receipts import generation_identity
 from onesearch_vl.onesearch_infer.gemini_av_runner import GeminiAVRunner,GeminiAVSettings
 from onesearch_vl.onesearch_infer.runners import InferenceConfig
 from onesearch_vl.onesearch_infer.messages import to_oai_messages
 plan_file=Path(plan_file);r=plan_file.parent;p=json.loads(plan_file.read_text())
 lock=(r/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic_json(r/'worker-start.json',{'pid':os.getpid(),'host':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'process_start_ticks':Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],'at_utc':now()})
 atomic_json(r/'state.json',{'status':'waiting','stage':'wait_for_benchmark_audit','at_utc':now(),'new_api_calls':0})
 deadline=time.monotonic()+p['max_wait_seconds'];dependency=Path(p['after_audit'])
 while not (dependency/'summary.json').exists():
  if (dependency/'provider-stop.json').exists():raise RuntimeError('prior audit provider stop; do not restart provider traffic')
  if time.monotonic()>deadline:raise TimeoutError('prior audit wait')
  time.sleep(30)
 if (dependency/'provider-stop.json').exists():raise RuntimeError('prior audit provider stop')
 rows=json.loads((r/'inputs.private.json').read_text());mapping=json.loads((r/'blind-map.private.json').read_text());assert len(rows)==p['max_api_calls']==28
 os.environ.update(dict(l.split('=',1) for l in Path(p['teacher_env']).read_text().splitlines() if l))
 os.environ.update(ONESEARCH_2K_TEACHER_MODEL='gemini-3.8-flash',ONESEARCH_TEACHER_STOP_FILE=str(r/'provider-stop.json'),ONESEARCH_TEACHER_TRANSIENT_LIMIT='3')
 install();settings=replace(GeminiAVSettings.from_dict(json.loads(Path(p['teacher_config']).read_text())),action_protocol='tags')
 frozen={str(x):file_sha(x) for x in [Path(__file__),plan_file,r/'inputs.private.json',r/'blind-map.private.json']};atomic_json(r/'source-freeze.json',frozen)
 (r/'results').mkdir(exist_ok=True);deadline=time.monotonic()+p['max_elapsed_seconds']
 for row in rows:
  if time.monotonic()>deadline or BurstProviderGuard(r).stopped:break
  for path,h in frozen.items():assert file_sha(path)==h
  dest=r/'results'/(row['recording']+'.json')
  if dest.exists():continue
  assert file_sha(row['media_path'])==row['media_sha256']
  contents=[{'role':'user','parts':[{'video_url':{'url':row['media_path'],'native_av':True}},{'text':json.dumps({'publisher':row['publisher_text'],'questions':row['questions']},ensure_ascii=False)}]}]
  result={'recording':row['recording'],'status':'running','api_calls':1,'started_at':now(),'media_sha256':row['media_sha256'],'question_ids':[q['question_id'] for q in row['questions']],'human_gold':False};atomic_json(dest,result)
  try:
   response=GeminiAVRunner(settings).infer(contents,SYSTEM,InferenceConfig(max_tokens=8192,temperature=0.));BurstProviderGuard(r).observe(response);result['response']=response
   text=''.join(x.get('text','') for x in response['candidates'][0]['content']['parts'])
   result['generation_identity']=generation_identity(response,[{'role':'system','content':SYSTEM}]+to_oai_messages(contents)+[{'role':'assistant','content':text}])
   result['verdicts']=validate_group(parse_json_output(text),result['question_ids']);result['status']='completed'
  except Exception as exc:result.update(status='failed',error_type=type(exc).__name__)
  result['finished_at']=now();atomic_json(dest,result)
  atomic_json(r/'state.json',{'status':'running','stage':'blind_paired_av_quality_audit','at_utc':now(),'completed_attempts':len(list((r/'results').glob('*.json'))),'max_api_calls':28,'paid_replays':0,'human_gold':0})
 results=[json.loads(x.read_text()) for x in (r/'results').glob('*.json')];good=[x for x in results if x['status']=='completed'];by_arm={};per_source=[]
 for rec in good:
  pair={}
  for item in rec['verdicts']:
   origin=mapping[item['question_id']];arm=origin['arm'];v=item['verdict'];pair.setdefault(arm,[]).append(usable_search(v))
   by_arm.setdefault(arm,[]).append({'question_id':item['question_id'],'recording':rec['recording'],'usable_search':usable_search(v),'verdict':v})
  assert set(pair)=={'native_av_generation','publisher_text_generation'}
  per_source.append({'recording':rec['recording'],**{a:sum(v)/len(v) for a,v in pair.items()}})
 summary={'status':'bounded_work_finished','at_utc':now(),'attempted_sources':len(results),'completed_sources':len(good),'failed_or_unknown_sources':len(results)-len(good),'questions_by_arm':{a:len(v) for a,v in by_arm.items()},'usable_search_by_arm':{a:sum(x['usable_search'] for x in v) for a,v in by_arm.items()},'source_mean_usable_fraction':{a:sum(x[a] for x in per_source)/len(per_source) for a in by_arm} if per_source else {},'provider_stopped':BurstProviderGuard(r).stopped,'human_gold':0,'paid_replays':0,'gpu_jobs':0,'new_training_rows':0,'scope':'Diagnostic conditional on historical AV-successful sources and publisher generation success; generation prompt/schema differed; not a causal AV-input or learning-gain estimate'}
 atomic_json(r/'question-verdicts.private.json',by_arm);atomic_json(r/'source-paired-results.private.json',per_source);atomic_json(r/'summary.json',summary);atomic_json(r/'state.json',summary)

if __name__=='__main__':run(sys.argv[1])

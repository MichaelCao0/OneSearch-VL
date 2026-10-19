"""Same question/prompt/budget evaluation: base, old172, clean500; AV/muted."""
from pathlib import Path
import sys,os,json,argparse,time,re,types
ROOT=Path('/home/jovyan/OneSearch-VL');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'onesearch_vl'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from research.av_train.native_sft import file_sha
from research.av_train.teacher_rollouts import atomic_json
from research.av_train.full_checkpoint import attach_full_thinker
from onesearch_infer.runners import build_runner,InferenceConfig
from onesearch_infer.pipeline import process_native_av_case
from onesearch_infer.search_backends import FrozenSearchBackend
from onesearch_infer.tools import extract_tool_call
from research.av_train.modality_inputs import MutedMedia
from research.av_search.corpus import visible_corpus

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--variant',choices=['base','trained'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--modality',choices=['native_av','muted_video']);a=p.parse_args()
 r=Path(a.run_dir).resolve();plan=json.loads((r/'plan.json').read_text());members=json.loads((r/'eval-members.private.json').read_text());freeze=json.loads((r/'data-freeze.json').read_text())
 assert file_sha(r/'eval-members.private.json')==freeze['eval_member_sha256'];assert file_sha(r/'plan.json')==freeze['plan_sha256']
 if a.smoke:members=[members[0],members[-1]]
 root=r/('evaluation-smoke' if a.smoke else 'gpu-v1/eval');root.mkdir(parents=True,exist_ok=True)
 outputs={}
 for mode in ([a.modality] if a.modality else plan['modalities']):
  out=root/(a.variant+'__'+mode);out.mkdir(exist_ok=True);outputs[mode]=out
 os.environ.update(ONESEARCH_OMNI_MODEL_MANIFEST=plan['model_manifest'],AV_AGENT_MAX_TURNS=str(plan['max_turns']),ONESEARCH_DISABLE_RETRIEVAL='0')
 artifact=json.loads(Path(plan['model_manifest']).read_text());runner=build_runner('omni-7b',checkpoint=artifact['snapshot_path']);runner.load()
 identity={'variant':a.variant,'model_revision':artifact['revision'],'optimizer_updates':0,'generation_vocab_policy':plan['generation_vocab_policy'],'max_turns':plan['max_turns'],'max_new_tokens':plan['max_new_tokens'],'prompt_sha256':file_sha(plan['prompt_file'])}
 if a.variant!='base':
  identity.update(attach_full_thinker(runner,plan['full_checkpoint']));identity['optimizer_updates']=plan['optimizer_updates_per_sft'];runner._artifact_identity.update(sft_variant=plan['arm'])

 for mode,out in outputs.items():atomic_json(out/'model-identity.json',dict(identity,input_mode=mode))
 original_infer=runner.infer;mute=MutedMedia(r/'muted-eval-media');mode_state={'mode':'native_av'}
 def infer(self,contents,system_instruction=None,cfg=None):
  if mode_state['mode']=='muted_video':contents=mute.contents(contents)
  result=original_infer(contents,system_instruction=system_instruction,cfg=cfg)
  if mode_state['mode']=='muted_video':
   assert self.last_metrics['audio_count']==0 and not any(self.last_metrics['video_has_audio'])
  return result
 runner.infer=types.MethodType(infer,runner)
 backends={};results={m:(json.loads((out/'summary.json').read_text())['results'] if (out/'summary.json').exists() else []) for m,out in outputs.items()};started=time.monotonic();prompt=Path(plan['prompt_file']).read_text()
 for mode,rows in results.items():
  assert all(x['status']!='error' for x in rows);assert len({x['task_id'] for x in rows})==len(rows)
 for case in members:
  assert file_sha(case['source_path'])==case['source_sha256']
  key=(case['batch_dir'],case['index_dir'])
  if key not in backends:backends[key]=FrozenSearchBackend(case['index_dir'],json.loads((Path(case['batch_dir'])/'scope.json').read_text()))
  for mode,out in outputs.items():
   if any(x['task_id']==case['task_id'] for x in results[mode]):continue
   mode_state['mode']=mode;folder=out/case['task_id'];folder.mkdir();before=time.monotonic()
   try:
    source=case['source_path'] if mode=='native_av' else mute.video(case['source_path'])
    trace=process_native_av_case(question=case['task']['question'],source_path=source,runner=runner,output_dir=str(folder),case_id=case['task_id'],case_idx=len(results[mode]),search_backend=backends[key],inference_cfg=InferenceConfig(max_tokens=plan['max_new_tokens'],temperature=0.),system_instruction=prompt)
    answer=trace['final_response_text'];calls=[];signatures=[];target=False;read=False
    documents=visible_corpus(case['index_dir'],backends[key].scope)['sources'];ids={case['reference']['target_source_id']}|{x['source_id'] for x in case['reference'].get('support_quotes',[])};urls={documents[sid]['origin_url'] for sid in ids};assert urls
    # Public URL metric; internal source IDs are never shown to the actor.
    for turn in trace['turns']:
     call=extract_tool_call(turn.get('response_text',''))
     if call:
      action=json.loads(call) if isinstance(call,str) else call;name=action.get('name');calls.append(name);signatures.append(json.dumps(action,ensure_ascii=False,sort_keys=True));obs=str(turn.get('tool_output',''));hit=any(u in obs for u in urls);target|=name in ['text_search','image_search'] and hit;read|=name=='visit_webpage' and hit
    metrics=[t.get('response',{}).get('raw_response',{}).get('metrics',{}) for t in trace['turns']]
    assert sum(x.get('generated_tokens',0) for x in metrics)>0,'No actual generation'
    if mode=='native_av':assert any(x.get('audio_count',0)>0 for x in metrics),'Native initial audio missing'
    result={'task_id':case['task_id'],'split':case['split'],'family_id':case['family_id'],'variant':a.variant,'modality':mode,'status':trace['status'],'final_answer':answer,
     'protocol_final':bool(re.search(r'<response>\s*\S.*?</response>',answer,re.S)),'tool_calls':calls,'turns':len(trace['turns']),'target_retrieved':target,'target_read':read,'reference_url_count':len(urls),
     'repeated_identical_calls':len(signatures)-len(set(signatures)),'wall_seconds':time.monotonic()-before,'trace_path':str(folder/(case['task_id']+'_trajectory.json')),'semantic_accuracy':None,
     'last_input_audio_count':runner.last_metrics['audio_count'],'last_input_video_count':runner.last_metrics['video_count']}
   except Exception as exc:
    import traceback
    (folder/'traceback.private.txt').write_text(traceback.format_exc());result={'task_id':case['task_id'],'split':case['split'],'family_id':case['family_id'],'variant':a.variant,'modality':mode,'status':'error','error_type':type(exc).__name__,'final_answer':'','tool_calls':[],'wall_seconds':time.monotonic()-before}
   results[mode].append(result);atomic_json(out/'summary.json',{'variant':a.variant,'modality':mode,'planned':len(members),'completed':len(results[mode]),'results':results[mode],'wall_seconds':time.monotonic()-started,'formal_benchmark':False,'semantic_accuracy':'pending independent review'})
   print(json.dumps({k:v for k,v in result.items() if k not in ['final_answer']},ensure_ascii=False),flush=True)
 for mode,x in results.items():
  if len(x)!=len(members) or any(z['status']=='error' for z in x):raise RuntimeError('Evaluation infrastructure failure; do not count as task failure')
 atomic_json(root/(a.variant+'__'+(a.modality or 'paired')+'-execution-receipt.json'),{'status':'passed','questions_per_modality':len(members),'modalities':list(outputs),'wall_seconds':time.monotonic()-started,'max_allocated_gib':__import__('torch').cuda.max_memory_allocated()/2**30})
if __name__=='__main__':main()

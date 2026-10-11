"""Bounded real teacher attempts on fixed, executed anonymous multiwindow inputs.

Authored input inspection is context, never an assistant training target.
Seeds and sampling parameters are recorded; attempts are distinct from retained rows.
"""
import argparse,json,math
from pathlib import Path
from research.av_bench.contracts import json_copy
from research.av_bench.searchqa_metrics import score_candidate_diagnostic
from research.av_data.build_research_batch import write_json
from research.av_data.media_clock import sha256_file
from research.av_train.trajectory import digest
from research.av_train.searchqa import LocalController,LocalOmniController,SmokeObserverReplay
from research.av_train.clue_searchqa import ClueSearchQARuntime
from research.av_train.anonymous_teacher import TEACHER_SYSTEM,MULTI_ASSISTANCE,MULTI_ORIGIN,anonymous_silver_row
from research.av_train.rollout import run_episode

def sampling_options(temperature,top_p):
    for v in (temperature,top_p):
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Finite sampling parameters required')
    if not 0<temperature<=2 or not 0<top_p<=1:raise ValueError('Sampling parameters outside bounded range')
    return {'do_sample':True,'temperature':temperature,'top_p':top_p}

def attempt_schedule(tasks,count,seed):
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=128 or not tasks:raise ValueError('Bounded nonempty attempt schedule required')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0 or seed+count>=2**32:raise ValueError('Bounded integer seed required')
    return [{'attempt':i+1,'task':json_copy(tasks[i%len(tasks)]),'seed':seed+i} for i in range(count)]

class InitialWindows:
    def __init__(self,controller,clue_id,windows):
        self.controller=controller;self.actions=[{'action':{'tool':'inspect_av','arguments':{'source_id':clue_id,'start_s':w['start_s'],'end_s':w['end_s']}}} for w in windows]
    @property
    def total_tokens(self):return self.controller.total_tokens
    def __call__(self,history):return self.actions.pop(0) if self.actions else self.controller(history)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('batch-dir','index-dir','observer-dir','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--controller-kind',choices=['onesearch','omni'],default='onesearch')
    p.add_argument('--mode',choices=['collect','evaluate'],default='collect');p.add_argument('--adapter')
    p.add_argument('--checkpoint',default='repro/assets/model-8b');p.add_argument('--model-manifest')
    p.add_argument('--attempts',type=int,default=64);p.add_argument('--seed',type=int,default=1021200)
    p.add_argument('--temperature',type=float,default=.7);p.add_argument('--top-p',type=float,default=.9)
    p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--model-tool-calls',type=int,default=6)
    a=p.parse_args();options=sampling_options(a.temperature,a.top_p) if a.mode=='collect' else {'do_sample':False}
    if a.adapter and (a.mode!='evaluate' or a.controller_kind!='onesearch'):raise ValueError('Adapters are only accepted for explicit OneSearch evaluation')
    if not 1<=a.model_tool_calls<=6 or not 32<=a.max_new_tokens<=384:raise ValueError('Bounded generation/tool budgets required')
    if a.controller_kind=='omni' and not a.model_manifest:raise ValueError('Pinned Omni manifest required')
    batch=Path(a.batch_dir);scope=json.loads((batch/'scope.json').read_text())
    tasks=list(map(json.loads,(batch/'tasks-public.jsonl').read_text().splitlines()))
    expected_split='train' if a.mode=='collect' else 'dev'
    if scope['split']!=expected_split or any(t['split']!=expected_split for t in tasks):raise ValueError('Task split does not match collection/evaluation mode')
    if a.mode=='evaluate' and a.attempts!=len(tasks):raise ValueError('Evaluate each dev question exactly once')
    schedule=attempt_schedule(tasks,a.attempts,a.seed)
    bindings=json.loads((batch/'clue-bindings.private.json').read_text());by_clue={b['clue_id']:b for b in bindings}
    sources={s['source_id']:s for s in map(json.loads,(batch/'sources.private.jsonl').read_text().splitlines())}
    refs={r['task_id']:r for r in map(json.loads,(batch/'answer-candidates.private.jsonl').read_text().splitlines())}
    observer=SmokeObserverReplay(a.observer_dir);windows={}
    for b in bindings:
        sid=b['source']['source_id'];w=sorted([w for w in observer.available_windows if w['source_id']==sid],key=lambda w:w['start_s'])
        if len(w)!=3:raise ValueError('Each training clue requires exactly three real executed windows')
        windows[b['clue_id']]=w
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'input-manifest.private.json',{'tasks_sha256':sha256_file(batch/'tasks-public.jsonl'),'bindings_sha256':sha256_file(batch/'clue-bindings.private.json'),'observer_dir':a.observer_dir,'sampling':options,'attempts':[{'attempt':x['attempt'],'task_id':x['task']['task_id'],'seed':x['seed']} for x in schedule],'private_references_given_to_controller':False,'mode':a.mode,'adapter':a.adapter})
    controller=(LocalController(a.checkpoint,out/'generations',a.max_new_tokens,protocol='canonical') if a.controller_kind=='onesearch' else LocalOmniController(a.model_manifest,out/'generations',a.max_new_tokens,protocol='canonical'))
    if a.adapter:
        from peft import PeftModel
        controller.model=PeftModel.from_pretrained(controller.model,a.adapter,is_trainable=False).eval()
        controller.identity.update(adapter_sha256=sha256_file(Path(a.adapter)/'adapter_model.safetensors'),role='trained_controller')
    controller.sampling_options=options;controller.identity.update(assistance=MULTI_ASSISTANCE,trajectory_origin=MULTI_ORIGIN,input_profile='anonymous_three_window_clue',observer_mode='fixed_executed_replay',sampling=options)
    write_json(out/'teacher-identity.private.json',controller.identity)
    source_splits={s:'train' for s in scope['source_ids']};source_splits.update({b['clue_id']:'train' for b in bindings})
    rows=[];notes=[];results=[];seen=set()
    system=TEACHER_SYSTEM.replace('The first input-clip observation is prepared for you.','Three input-clip window observations are prepared for you.')
    for item in schedule:
        task=item['task'];tid=task['task_id'];folder=out/f"attempt-{item['attempt']:04d}";folder.mkdir()
        clue=task['initial_sources'][0];w=windows[clue]
        task['budget_recipe']={'max_tool_calls':len(w)+a.model_tool_calls,'max_requested_av_seconds':sum(v['end_s']-v['start_s'] for v in w)}
        controller.episode_seed=item['seed'];controller.torch.manual_seed(item['seed']);controller.torch.cuda.manual_seed_all(item['seed'])
        runtime=ClueSearchQARuntime(a.index_dir,scope,sources,[by_clue[clue]],folder/'tools',task['budget_recipe'],observer)
        before=controller.calls;tokens_before=controller.total_tokens
        trace=run_episode(task,runtime,InitialWindows(controller,clue,w),controller.identity,system_prompt=system,recover_tool_errors=True)
        trace['final']['cost']['controller_tokens']=controller.total_tokens-tokens_before
        write_json(folder/'trace.private.json',trace);metric=score_candidate_diagnostic(trace,refs[tid]);write_json(folder/'diagnostic.json',metric)
        result={'attempt':item['attempt'],'task_id':tid,'seed':item['seed'],'model_decisions':controller.calls-before,'status':trace['final'].get('status'),'retained':False}
        if a.mode=='evaluate':
            result['diagnostic']=metric;write_json(folder/'retention.json',result);results.append(result);print(json.dumps(result),flush=True);continue
        try:
            row,note=anonymous_silver_row(trace,source_splits,[f['accepted_values'][0] for f in refs[tid]['answer_fields']],initial_observation_count=3)
            key=digest(row)
            if key in seen:result['rejection']='duplicate_retained_row'
            else:
                seen.add(key);note['row_sha256']=key;rows.append(row);notes.append(note);write_json(folder/'row.private.json',row);write_json(folder/'note.private.json',note);result.update(retained=True,row_sha256=key)
        except ValueError as exc:result['rejection']=str(exc)
        write_json(folder/'retention.json',result);results.append(result);print(json.dumps(result),flush=True)
    write_json(out/'rows.private.json',rows);write_json(out/'notes.private.json',notes)
    write_json(out/'summary.json',{'schema_version':'av-anonymous-multiwindow-teacher.v1','mode':a.mode,'attempts':len(results),'unique_questions':len({r['task_id'] for r in results}),'silver_rows':len(rows),'results':results,'new_optimizer_updates':0,'benchmark_gold':False,'formal_accuracy':False,'semantic_verified':False,'private_references_given_to_controller':False})
if __name__=='__main__':main()

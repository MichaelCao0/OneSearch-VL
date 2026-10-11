"""Matched-budget autonomous tool-use pilot, not formal heldout benchmark."""
import argparse
import json
from pathlib import Path
from research.av_bench.contracts import json_copy
from research.av_data.build_research_batch import write_json
from research.av_data.media_clock import sha256_file
from .searchqa import LocalController,SearchQARuntime,SmokeObserverReplay,QA_SYSTEM
from .rollout import run_episode
from .seed_expansion import answer_diagnostic

def bounded_tasks(tasks,task_ids,max_calls):
    if max_calls not in (2,3,4) or not task_ids or len(set(task_ids))!=len(task_ids):raise ValueError('Bounded unique task selection required')
    selected=[json_copy(t) for t in tasks if t['task_id'] in task_ids]
    if len(selected)!=len(task_ids):raise ValueError('Missing selected task')
    for t in selected:t['budget_recipe']={'max_tool_calls':max_calls,'max_requested_av_seconds':4.}
    return selected

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','batch-dir','index-dir','observer-smoke-dir','checkpoint','adapter','out-dir'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(a.spec).read_text());batch=Path(a.batch_dir)
    tasks=bounded_tasks(list(map(json.loads,(batch/'tasks-public.jsonl').read_text().splitlines())),['avqa-diag-music','avqa-diag-rain'],3)
    sources={r['source_id']:r for r in map(json.loads,(batch/'sources.private.jsonl').read_text().splitlines())};scope=json.loads((batch/'scope.json').read_text())
    observer=SmokeObserverReplay(a.observer_smoke_dir);controller=LocalController(a.checkpoint,out/'generations',128,protocol='canonical')
    results=[]
    for variant in ('base','expanded_adapter'):
        if variant!='base':
            from peft import PeftModel
            controller.model=PeftModel.from_pretrained(controller.model,a.adapter,is_trainable=False).eval()
        identity=json_copy(controller.identity);identity.update({'protocol':'canonical','variant':variant,'trajectory_origin':'autonomous_model_actions_and_final'})
        if variant!='base':identity.update({'adapter_path':a.adapter,'adapter_sha256':sha256_file(Path(a.adapter)/'adapter_model.safetensors')})
        for task in tasks:
            runtime=SearchQARuntime(a.index_dir,scope,sources,out/variant/task['task_id']/'tools',task['budget_recipe'],observer)
            before=controller.total_tokens;before_calls=controller.calls
            trace=run_episode(task,runtime,controller,identity,system_prompt=QA_SYSTEM,recover_tool_errors=True)
            trace['final']['cost']['episode_controller_tokens']=controller.total_tokens-before
            write_json(out/variant/task['task_id']/'trace.private.json',trace)
            result={'variant':variant,'task_id':task['task_id'],'final':trace['final'],
              'metric':answer_diagnostic(trace['final'],spec['terms'][task['task_id']]),'model_decisions':controller.calls-before_calls,
              'tools':[s['action']['tool'] for s in trace['steps']],
              'tool_statuses':[s['observation'].get('status') for s in trace['steps']],
              'inspect_av_executed':any(s['action']['tool']=='inspect_av' and s['observation'].get('status')=='ok' for s in trace['steps']),
              'search_executed':any(s['action']['tool']=='search_sources' and s['observation'].get('status')=='ok' for s in trace['steps'])}
            results.append(result);write_json(out/variant/task['task_id']/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    write_json(out/'summary.json',{'schema_version':'av-autonomous-matched-pilot.v1','results':results,'questions_per_variant':2,
      'max_tool_calls_per_episode':3,'max_requested_av_seconds':4.,'max_new_tokens_per_decision':128,
      'base_field_matches':sum(r['metric']['requested_field_match'] for r in results if r['variant']=='base'),
      'adapter_field_matches':sum(r['metric']['requested_field_match'] for r in results if r['variant']=='expanded_adapter'),
      'semantic_verified':False,'formal_heldout_benchmark':False,'scope':'train-internal exploration; initial AV source identifiers available; publisher facts seen in training retrieval context',
      'adapter':a.adapter})
if __name__=='__main__':main()

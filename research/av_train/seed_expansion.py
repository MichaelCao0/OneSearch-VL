"""Bounded assisted silver collection and source-disjoint answer diagnostic.

Tool actions are authored; actual final decisions come from the frozen base.
Requested-field match measures exact publisher fields, not QA accuracy.
"""
import argparse
import gc
import json
from pathlib import Path
from .silver import normalized,publisher_silver_row
from .searchqa import LocalController,SearchQARuntime,SmokeObserverReplay,run_guided_curriculum,QA_SYSTEM
from .trajectory import source_ids
from .export import public_observation
from research.av_data.build_research_batch import write_json

def check_source_partition(seed_tasks,diagnostic_tasks):
    seed={sid for t in seed_tasks for sid in t['initial_sources']}
    diagnostic={sid for t in diagnostic_tasks for sid in t['initial_sources']}
    if seed&diagnostic:raise ValueError('seed/diagnostic source overlap')
    return {'seed_sources':sorted(seed),'diagnostic_sources':sorted(diagnostic)}

def answer_diagnostic(final,required_terms):
    if not required_terms or any(not isinstance(v,str) or not v.strip() for v in required_terms):raise ValueError('Required field terms needed')
    answer=final.get('answer','')
    match=final.get('status')=='ok' and isinstance(answer,str) and all(normalized(t) in normalized(answer) for t in required_terms)
    return {'requested_field_match':bool(match),'full_answer_semantic_verified':False,'metric':'exact_requested_publisher_fields; not QA accuracy'}

def final_history(trace):
    history=[{'role':'system','content':QA_SYSTEM+'\nTools: '+json.dumps(trace['tools'],ensure_ascii=False)},
      {'role':'user','content':json.dumps(trace['task'],ensure_ascii=False)}]
    for step in trace['steps']:
        history.extend([{'role':'assistant','content':json.dumps({'action':step['action']},ensure_ascii=False)},
          {'role':'user','content':'<tool_response>\n'+json.dumps(public_observation(step['observation']),ensure_ascii=False)+'\n</tool_response>'}])
    history.append({'role':'user','content':'Final turn. Answer the original question in ONE concise English sentence, preserving publisher proper names exactly. Return only a complete JSON object: {"final":{"status":"ok","answer":"...","evidence":[]}}. No tools, reasoning or repeated sentences. If the requested field is missing, use status insufficient_evidence.'})
    return history

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','batch-dir','index-dir','observer-smoke-dir','checkpoint','adapter','out-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(a.spec).read_text());tasks={t['task_id']:t for t in map(json.loads,(Path(a.batch_dir)/'tasks-public.jsonl').read_text().splitlines())}
    seed=[tasks[i] for i in spec['seed_task_ids']];diagnostics=[tasks[i] for i in spec['diagnostic_task_ids']]
    partition=check_source_partition(seed,diagnostics);write_json(out/'source-partition.json',partition)
    source_splits={sid:'train' for sid in json.loads((Path(a.batch_dir)/'scope.json').read_text())['source_ids']}
    sources={r['source_id']:r for r in map(json.loads,(Path(a.batch_dir)/'sources.private.jsonl').read_text().splitlines())}
    scope=json.loads((Path(a.batch_dir)/'scope.json').read_text());observer=SmokeObserverReplay(a.observer_smoke_dir)
    controller=LocalController(a.checkpoint,out/'generations-base',128,protocol='canonical')
    rows=[];notes=[];seed_results=[];pairs=[]
    # Diagnostic first: do not risk using the remaining allocation before the comparison.
    for task in diagnostics:
        runtime=SearchQARuntime(a.index_dir,scope,sources,out/task['task_id']/'tools',task['budget_recipe'],observer)
        trace=run_guided_curriculum(task,runtime,controller,compact_canonical=True)
        write_json(out/task['task_id']/'trace-base.private.json',trace)
        pair={'task_id':task['task_id'],'base':trace['final'],'base_metric':answer_diagnostic(trace['final'],spec['terms'][task['task_id']]),'history':final_history(trace)}
        pairs.append(pair);write_json(out/task['task_id']/'base-diagnostic.json',{k:v for k,v in pair.items() if k!='history'})
        print(json.dumps({'stage':'base_diagnostic','task_id':task['task_id'],**pair['base_metric']}),flush=True)
    from peft import PeftModel
    controller.model=PeftModel.from_pretrained(controller.model,a.adapter,is_trainable=False).eval()
    for pair in pairs:
        try:final=controller(pair['history']).get('final',{'status':'no_final_answer'})
        except Exception as exc:final={'status':'controller_failed','error_type':type(exc).__name__}
        pair['adapter']=final;pair['adapter_metric']=answer_diagnostic(final,spec['terms'][pair['task_id']])
        write_json(out/pair['task_id']/'paired-diagnostic.json',{k:v for k,v in pair.items() if k!='history'})
        print(json.dumps({'stage':'adapter_diagnostic','task_id':pair['task_id'],**pair['adapter_metric']}),flush=True)
    controller.model=controller.model.unload().eval()
    for task in seed:
        runtime=SearchQARuntime(a.index_dir,scope,sources,out/task['task_id']/'tools',task['budget_recipe'],observer)
        trace=run_guided_curriculum(task,runtime,controller,compact_canonical=True)
        write_json(out/task['task_id']/'trace-base.private.json',trace)
        result={'task_id':task['task_id'],'final':trace['final'],'retained':False}
        try:
            if set(source_ids(trace))&set(partition['diagnostic_sources']):raise ValueError('Diagnostic source entered seed trace')
            row,note=publisher_silver_row(trace,source_splits,spec['terms'][task['task_id']]);rows.append(row);notes.append(note);result['retained']=True
        except ValueError as exc:result['rejection']=str(exc)
        seed_results.append(result);write_json(out/task['task_id']/'retention.json',result)
        print(json.dumps({'stage':'seed','task_id':task['task_id'],'retained':result['retained']}),flush=True)
    write_json(out/'rows.private.json',rows);write_json(out/'notes.private.json',notes)
    summary={'schema_version':'av-assisted-seed-and-answer-diagnostic.v1','seed_attempts':len(seed_results),'silver_rows':len(rows),
      'seed_results':seed_results,'paired_diagnostics':[{k:v for k,v in x.items() if k!='history'} for x in pairs],
      'base_field_matches':sum(x['base_metric']['requested_field_match'] for x in pairs),
      'adapter_field_matches':sum(x['adapter_metric']['requested_field_match'] for x in pairs),'diagnostic_count':len(pairs),
      'full_answer_semantic_verified':False,'formal_benchmark':False,'autonomous_search_evaluated':False,
      'diagnostic_scope':'train-internal initial-source-disjoint; identical real tool history; authored title-assisted tools',
      'adapter':a.adapter,'new_optimizer_updates':0,'max_new_tokens':128,'controller':controller.identity}
    write_json(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('seed_results','paired_diagnostics')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

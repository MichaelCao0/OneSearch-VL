"""Model-decided anonymous-clue search trajectories with one authored input inspect.

Input AV observations reuse exact decoded clips. Search, reads and final answers
come from actual model generations. Authored inspect is not an SFT target.
"""
import argparse
import json
from pathlib import Path
from research.av_bench.contracts import json_copy
from research.av_bench.searchqa_metrics import field_present,score_candidate_diagnostic
from research.av_data.build_research_batch import write_json
from .clue_replay import MappedClueReplay
from .clue_searchqa import ClueSearchQARuntime,CLUE_SYSTEM
from .export import public_observation
from .rollout import run_episode
from .searchqa import LocalOmniController,QA_SYSTEM
from .trajectory import validate_trace,digest

ASSISTANCE='one authored query-clip inspect; model decides all subsequent search/read/final turns; no title or reference fields supplied'
ORIGIN='authored_input_inspect_with_model_decided_search_qa'
MULTI_ASSISTANCE='three authored query-clip window inspections; model decides search/read/final; no publisher clues supplied'
MULTI_ORIGIN='authored_multiwindow_input_with_model_decided_search_qa'
TEACHER_SYSTEM=QA_SYSTEM+'\n'+CLUE_SYSTEM+'''\nThe first input-clip observation is prepared for you. Search using SHORT English or source-language object/sound keywords from that observation. Query terms may match descriptions in French as well as English. Read a discovered relevant source using the returned quote_start and quote_end; do not inspect the input again. Once the requested attributes are available, answer in ONE concise English sentence, preserving publisher proper names exactly. Return ONE complete JSON decision, no reasoning or imagined observations.'''

class InitialInspectionController:
    def __init__(self,controller,clue_id,duration):
        self.controller=controller;self.clue_id=clue_id;self.duration=duration;self.first=True
    @property
    def total_tokens(self):return self.controller.total_tokens
    def __call__(self,history):
        if self.first:
            self.first=False
            return {'action':{'tool':'inspect_av','arguments':{'source_id':self.clue_id,'start_s':0.,'end_s':self.duration}}}
        return self.controller(history)

def anonymous_silver_row(trace,source_splits,required_terms,*,initial_observation_count=1):
    t=validate_trace(trace,source_splits)
    if isinstance(initial_observation_count,bool) or initial_observation_count not in (1,3):raise ValueError('One or three initial observations required')
    expected=(ASSISTANCE,ORIGIN) if initial_observation_count==1 else (MULTI_ASSISTANCE,MULTI_ORIGIN)
    if (t['teacher'].get('assistance'),t['teacher'].get('trajectory_origin'))!=expected:raise ValueError('Explicit assistance identity required')
    if not required_terms or any(not isinstance(x,str) or not x.strip() for x in required_terms):raise ValueError('Required publisher fields needed')
    answer=t['final'].get('answer','')
    if not isinstance(answer,str) or not all(field_present(x,answer) for x in required_terms):raise ValueError('Requested fields missing from answer')
    initial_steps=t['steps'][:initial_observation_count]
    if len(initial_steps)!=initial_observation_count:raise ValueError('Missing initial observations')
    windows=set()
    for first in initial_steps:
        obs=first['observation'];args=first['action']['arguments']
        if first['action']['tool']!='inspect_av' or args['source_id'] not in t['task']['initial_sources'] or obs.get('status')!='ok' or obs.get('model_executed') is not True:raise ValueError('Executed input observation required')
        key=(args['source_id'],args['start_s'],args['end_s'])
        if key in windows:raise ValueError('Duplicate initial observation')
        windows.add(key)
    successful_reads=[s['observation'].get('quote',{}).get('text','') for s in t['steps'][initial_observation_count:] if s['action']['tool']=='read_text' and s['observation'].get('status')=='ok']
    if not any(all(field_present(x,text) for x in required_terms) for text in successful_reads):raise ValueError('Requested fields require an actual successful text read')
    if not any(s['action']['tool']=='search_sources' and s['observation'].get('status')=='ok' and s['observation'].get('results') for s in t['steps'][initial_observation_count:]):raise ValueError('Actual source discovery required')
    from .rollout import initial_av_context
    initial=initial_av_context(t['task'],[s['observation'] for s in initial_steps])
    turns=[{'from':'human','value':json.dumps(initial,ensure_ascii=False)}]
    for step in t['steps'][initial_observation_count:]:
        turns.extend([{'from':'gpt','value':json.dumps({'action':step['action']},ensure_ascii=False)},
          {'from':'observation','value':json.dumps(public_observation(step['observation']),ensure_ascii=False)}])
    turns.append({'from':'gpt','value':json.dumps({'final':{k:v for k,v in t['final'].items() if k!='cost'}},ensure_ascii=False)})
    row={'system':t['system']+'\nTools: '+json.dumps(t['tools'],ensure_ascii=False),'tools':'','conversations':turns}
    note={'task_id':t['task']['task_id'],'quality_label':'silver_anonymous_publisher_fields','semantic_verified':False,'benchmark_gold':False,
      'trace_sha256':digest(t),'teacher':t['teacher'],'required_terms':required_terms,'fully_autonomous':False,
      'authored_assistant_targets':0,'initial_observation_count':initial_observation_count,'model_decided_assistant_targets':len(t['steps'])-initial_observation_count+1,'fresh_observer_executed':all(s['observation'].get('model_executed_this_episode') is True for s in initial_steps),
      'scope':'training_exploration; exact field match plus actual text read; no formal QA accuracy'}
    return row,note

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch-dir','index-dir','admission-file','model-manifest','out-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-new-tokens',type=int,default=192);p.add_argument('--max-tool-calls',type=int,default=4)
    a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    batch=Path(a.batch_dir);scope=json.loads((batch/'scope.json').read_text())
    sources={s['source_id']:s for s in map(json.loads,(batch/'sources.private.jsonl').read_text().splitlines())}
    bindings=json.loads((batch/'clue-bindings.private.json').read_text())
    admitted=json.loads(Path(a.admission_file).read_text())['admitted'];by_task={x['task_id']:x for x in admitted}
    tasks=[t for t in map(json.loads,(batch/'tasks-public.jsonl').read_text().splitlines()) if t['task_id'] in by_task]
    if len(tasks)!=2 or any(t['split']!='train' for t in tasks):raise ValueError('Exactly two admitted train tasks required for bounded smoke')
    observer=MappedClueReplay(a.admission_file)
    controller=LocalOmniController(a.model_manifest,out/'generations',a.max_new_tokens,protocol='canonical')
    controller.identity.update({'protocol':'canonical','input_profile':'anonymous_query_clip','observer_mode':'exact_decoded_replay',
      'assistance':ASSISTANCE,'trajectory_origin':ORIGIN})
    source_splits={sid:'train' for sid in scope['source_ids']};source_splits.update({b['clue_id']:'train' for b in bindings})
    references={r['task_id']:r for r in map(json.loads,(batch/'answer-candidates.private.jsonl').read_text().splitlines())}
    rows=[];notes=[];summaries=[]
    for original in tasks:
        task=json_copy(original);task['budget_recipe']={'max_tool_calls':a.max_tool_calls,'max_requested_av_seconds':2.}
        tid=task['task_id'];selected=[b for b in bindings if b['clue_id'] in task['initial_sources']]
        runtime=ClueSearchQARuntime(a.index_dir,scope,sources,selected,out/tid/'tools',task['budget_recipe'],observer)
        proxy=InitialInspectionController(controller,task['initial_sources'][0],by_task[tid]['query_duration_s'])
        before=controller.calls
        trace=run_episode(task,runtime,proxy,controller.identity,system_prompt=TEACHER_SYSTEM,recover_tool_errors=True)
        write_json(out/tid/'trace.private.json',trace)
        metric=score_candidate_diagnostic(trace,references[tid]);write_json(out/tid/'diagnostic.json',metric)
        result={'task_id':tid,'final':trace['final'],'model_decisions':controller.calls-before,'tools':[s['action']['tool'] for s in trace['steps']],
          'retained':False,'diagnostic':metric}
        try:
            terms=[f['accepted_values'][0] for f in references[tid]['answer_fields']]
            row,note=anonymous_silver_row(trace,source_splits,terms);rows.append(row);notes.append(note);result['retained']=True
        except ValueError as exc:result['rejection']=str(exc)
        summaries.append(result);write_json(out/tid/'retention.json',result)
        # Per-episode durable export survives a later deadline.
        write_json(out/tid/'row.private.json',rows[-1] if result['retained'] else None)
        print(json.dumps({'task_id':tid,'status':trace['final'].get('status'),'model_decisions':controller.calls-before,'tools':result['tools'],'retained':result['retained']}),flush=True)
    write_json(out/'rows.private.json',rows);write_json(out/'notes.private.json',notes)
    write_json(out/'summary.json',{'schema_version':'av-anonymous-teacher-smoke.v1','episodes':summaries,'teacher':controller.identity,
      'attempted_tasks':len(summaries),'silver_rows':len(rows),'new_optimizer_updates':0,'formal_benchmark':False,
      'fully_autonomous':False,'private_references_given_to_controller':False,'semantic_verified':False})
if __name__=='__main__':main()

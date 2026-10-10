"""Bounded controller episodes over real tools; teacher never receives private references.

No model is selected implicitly. OpenAI-compatible controller uses explicit env
configuration. Semantic AV observer is injected, or its unavailability is returned.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import urllib.request
from research.av_bench.contracts import json_copy,validate_task,number
from research.av_agent.tools import inspect_av,read_text
from research.av_search.retrieval import search_sources
from .trajectory import make_step,source_ids
from .export import public_observation

SYSTEM='''You research a frozen corpus of audio-video and text sources. Use only actual tool observations available in history. Return one JSON object per turn: {"action":{"tool":"search_sources|inspect_av|read_text","arguments":{...}}} or {"final":{"status":"ok","facts":[{"fact_id":"f1","value":"..."}],"ranked_sources":["src_..."],"ranked_windows":[{"source_id":"src_...","start_s":0,"end_s":1}],"evidence":[{"fact_id":"f1","source_id":"src_...","start_s":0,"end_s":1}]}}. For text evidence include source_id, fact_id, start_char, end_char, exact quote. Never invent tool outputs. Event identities, when requested, are observable ordinals rather than private annotation IDs. Missing evidence should be reported, not guessed.'''
TOOL_DEFINITIONS=[
 {'name':'search_sources','arguments':{'query':'string','top_k':'integer 1..20'}},
 {'name':'inspect_av','arguments':{'source_id':'string','start_s':'number on original timeline','end_s':'number'}},
 {'name':'read_text','arguments':{'source_id':'string','start_char':'integer','end_char':'integer'}}]

class ToolRuntime:
    def __init__(self,index_dir,scope,sources,output_dir,budget,observer=None):
        self.index_dir=Path(index_dir);self.scope=json_copy(scope);self.sources=sources
        self.source_ids=list(scope['source_ids']);self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=False)
        self.budget=budget;self.observer=observer
        self.cost={'tool_calls':0,'requested_av_seconds':0.}
    def tool_definitions(self):
        definitions=json_copy(TOOL_DEFINITIONS)
        limit=getattr(self.observer,'max_clip_seconds',None)
        if limit is not None:
            next(t for t in definitions if t['name']=='inspect_av')['constraints']={'max_window_seconds':limit}
        return definitions
    def execute(self,action):
        tool=action['tool'];a=action['arguments'];expected={
          'search_sources':{'query','top_k'},'read_text':{'source_id','start_char','end_char'},
          'inspect_av':{'source_id','start_s','end_s'}}
        if tool not in expected or set(a)!=expected[tool]:raise ValueError('Invalid tool arguments')
        if self.cost['tool_calls']>=self.budget['max_tool_calls']:raise ValueError('Tool budget exceeded')
        if tool!='search_sources' and a['source_id'] not in self.source_ids:raise ValueError('Source outside split scope')
        if tool=='inspect_av':
            start=number(a['start_s'],'start_s');end=number(a['end_s'],'end_s')
            source=self.sources[a['source_id']]
            if source['media_type']!='audio_video' or not 0<=start<end<=source['duration_s']:raise ValueError('AV interval outside source')
            if self.cost['requested_av_seconds']+end-start>self.budget['max_requested_av_seconds']:raise ValueError('AV budget exceeded')
            self.cost['requested_av_seconds']+=end-start
        self.cost['tool_calls']+=1
        if tool=='search_sources':
            receipt=search_sources(self.index_dir,a['query'],self.scope,a['top_k'])
            for result in receipt['results']:
                source=self.sources[result['source_id']]
                result['media_type']=source['media_type']
                if source['media_type']=='audio_video':result['duration_s']=source['duration_s']
            return receipt
        if tool=='read_text':return read_text(self.index_dir,a['source_id'],a['start_char'],a['end_char'],self.scope)
        output=self.output_dir/('inspect-'+str(self.cost['tool_calls']))
        if self.observer is not None:
            receipt=self.observer(self.sources[a['source_id']],a['start_s'],a['end_s'],output)
        else:
            receipt,_=inspect_av(self.sources[a['source_id']],a['start_s'],a['end_s'],output)
        # Sum measured resource values, including failed/fixture observer attempts.
        # Missing metrics stay absent; counters distinguish partial sums from zero.
        actual=receipt.get('actual_cost',{})
        aliases={'returned_video_frames':'prepared_video_frames',
                 'decoded_audio_samples':'prepared_audio_samples','output_bytes':'prepared_output_bytes'}
        for key,value in actual.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0:
                key=aliases.get(key,key)
                self.cost[key]=self.cost.get(key,0)+value
        if self.observer is not None:
            for key in ('input_tokens','generated_tokens'):
                value=actual.get(key)
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0:
                    counter=key+'_unknown_observations'
                    self.cost[counter]=self.cost.get(counter,0)+1
        return receipt

class OpenAIController:
    def __init__(self,model,base_url_env='AV_TEACHER_BASE_URL',api_key_env='AV_TEACHER_API_KEY'):
        base=os.environ.get(base_url_env);key=os.environ.get(api_key_env)
        if not base or not key:raise ValueError('teacher_unavailable: required provider environment variables are absent')
        self.url=base.rstrip('/')+'/chat/completions';self.key=key;self.model=model;self.total_tokens=0
    def __call__(self,history):
        request=urllib.request.Request(self.url,data=json.dumps({'model':self.model,'messages':history,
            'temperature':0,'max_tokens':2048,'response_format':{'type':'json_object'}},ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=60) as response:body=json.loads(response.read())
        except Exception as exc:raise RuntimeError('teacher_request_failed: '+type(exc).__name__) from None
        self.total_tokens+=body.get('usage',{}).get('total_tokens',0)
        return json.loads(body['choices'][0]['message']['content'])

def run_episode(task,runtime,controller,teacher):
    validate_task(task,{'corpus_id':task['corpus_id'],'split':task['split'],'source_ids':runtime.source_ids})
    tools=runtime.tool_definitions() if hasattr(runtime,'tool_definitions') else json_copy(TOOL_DEFINITIONS)
    history=[{'role':'system','content':SYSTEM+'\nTools: '+json.dumps(tools,ensure_ascii=False,allow_nan=False)},
             {'role':'user','content':json.dumps(task,ensure_ascii=False)}]
    steps=[];final={'status':'tool_budget_exhausted'}
    for _ in range(task['budget_recipe']['max_tool_calls']+1):
        try:decision=json_copy(controller(json_copy(history)))
        except Exception as exc:
            final={'status':'controller_failed','error_type':type(exc).__name__};break
        if isinstance(decision,dict) and set(decision)=={'final'} and isinstance(decision['final'],dict):
            final=decision['final'];break
        if not isinstance(decision,dict) or set(decision)!={'action'}:
            final={'status':'invalid_controller_output'};break
        action=decision['action']
        try:
            receipt=runtime.execute(action);step=make_step(action,receipt)
        except Exception as exc:
            final={'status':'tool_execution_failed','error_type':type(exc).__name__};break
        steps.append(step)
        # The controller sees only public observations; raw path/error context stays private.
        history.extend([{'role':'assistant','content':json.dumps({'action':action},ensure_ascii=False,allow_nan=False)},
                        {'role':'user','content':'<tool_response>\n'+json.dumps(public_observation(receipt),ensure_ascii=False,allow_nan=False)+'\n</tool_response>'}])
    final['cost']=json_copy(runtime.cost)
    if hasattr(controller,'total_tokens'):final['cost']['controller_tokens']=controller.total_tokens
    trace={'schema_version':'av-research-trajectory.v1','task':json_copy(task),'teacher':json_copy(teacher),
           'system':SYSTEM,'tools':tools,'steps':steps,'final':final,'execution_kind':'live_tools',
           'referenced_source_ids':[]}
    trace['referenced_source_ids']=sorted(source_ids(trace))
    return trace


def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--task-id');p.add_argument('--teacher-model');p.add_argument('--teacher-revision')
    p.add_argument('--fixture-smoke',action='store_true')
    p.add_argument('--observer-mode',choices=['unavailable','native','recorded'],default='unavailable')
    p.add_argument('--observer-model-manifest');p.add_argument('--observer-code-revision')
    p.add_argument('--observer-runtime-manifest');p.add_argument('--observer-runtime-lock')
    p.add_argument('--observer-device',default='cuda:0')
    p.add_argument('--observer-execution-dir',action='append',default=[])
    p.add_argument('--observer-backend-identity')
    return p


def build_observer(args,out_dir):
    native_fields=(args.observer_model_manifest,args.observer_code_revision,
                   args.observer_runtime_manifest,args.observer_runtime_lock)
    recorded_fields=(args.observer_execution_dir,args.observer_backend_identity)
    if args.observer_mode=='unavailable':
        if any(native_fields) or any(recorded_fields):raise ValueError('Observer unavailable mode cannot ignore explicit configuration')
        return None
    if args.observer_mode=='recorded':
        if any(native_fields) or not args.observer_execution_dir or not args.observer_backend_identity:
            raise ValueError('Independent observer records and expected identity required; native fields are not accepted')
        from .native_observer import RecordedObserverReplay
        expected=json.loads(Path(args.observer_backend_identity).read_text())
        return RecordedObserverReplay(args.observer_execution_dir,expected_backend=expected)
    if not all(native_fields) or any(recorded_fields):
        raise ValueError('Explicit native observer model, code revision, runtime manifest and lock required')
    from research.scripts.run_av_observer_smoke import collect_runtime_receipt
    from research.av_data.build_research_batch import write_json
    receipt=collect_runtime_receipt(args.observer_runtime_manifest,args.observer_runtime_lock)
    write_json(Path(out_dir)/'observer-runtime.private.json',receipt)
    if receipt['status']!='verified':raise ValueError('Native observer runtime mismatch before model load')
    from research.av_agent.qwen_omni_backend import QwenOmniBackend
    from research.av_agent.semantic_observer import SemanticObserver
    from .native_observer import NativeObserverAdapter
    backend=QwenOmniBackend(args.observer_model_manifest,backend_code_revision=args.observer_code_revision,
        device=args.observer_device,max_clip_seconds=2.,clock_policy='av_pts_grid_v1',output_policy='json_object_v1')
    return NativeObserverAdapter(SemanticObserver(backend),max_clip_seconds=2.,backend=backend)


def main():
    a=make_parser().parse_args();batch=Path(a.batch_dir);out=Path(a.out_dir)
    if out.exists():raise FileExistsError(out)
    tasks=[json.loads(l) for l in (batch/'tasks-public.jsonl').read_text().splitlines() if l]
    task=next(t for t in tasks if a.task_id is None or t['task_id']==a.task_id)
    sources={r['source_id']:r for r in (json.loads(l) for l in (batch/'sources.private.jsonl').read_text().splitlines() if l)}
    scope=json.loads((batch/'scope.json').read_text());out.mkdir(parents=True,exist_ok=False)
    if a.fixture_smoke:
        counter=[0]
        def controller(history):
            counter[0]+=1
            if counter[0]==1:return {'action':{'tool':'search_sources','arguments':{'query':'steam engine','top_k':3}}}
            if counter[0]==2:
                result=json.loads(history[-1]['content'].removeprefix('<tool_response>\n').removesuffix('\n</tool_response>'))['results'][0]
                return {'action':{'tool':'read_text','arguments':{'source_id':result['source_id'],'start_char':result['quote_start'],'end_char':result['quote_end']}}}
            return {'final':{'status':'fixture_completed','facts':[],'ranked_sources':[],'ranked_windows':[],'evidence':[]}}
        teacher={'kind':'fixture','model':'metadata-connectivity-script','revision':'v0'}
    else:
        if not a.teacher_model or not a.teacher_revision:raise ValueError('Explicit teacher model/revision required')
        controller=OpenAIController(a.teacher_model);teacher={'kind':'model','model':a.teacher_model,'revision':a.teacher_revision}
    observer=build_observer(a,out)
    runtime=ToolRuntime(batch/'corpus',scope,sources,out/'tools',task['budget_recipe'],observer=observer)
    trace=run_episode(task,runtime,controller,teacher)
    from research.av_data.build_research_batch import write_json
    write_json(out/'trace.private.json',trace)
    write_json(out/'summary.json',{'status':trace['final']['status'],'executed_tool_calls':len(trace['steps']),
        'teacher_kind':teacher['kind'],'retained_sft_rows':0,'semantic_observer_configured':observer is not None,'observer_mode':a.observer_mode,
        'training_ready':False,'scope':scope['split']})
    print(json.dumps(json.loads((out/'summary.json').read_text()),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

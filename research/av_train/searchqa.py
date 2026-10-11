"""Direct AV search QA pilot: real model decisions, frozen search, fixed AV replay.

Exploration can run before benchmark gold. Candidate traces do not become
semantically verified SFT rows merely because their tool execution succeeds.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import shutil
import time
from research.av_bench.contracts import TASK_VERSION,validate_task,json_copy
from research.av_data.build_research_batch import write_json,write_jsonl
from research.av_data.media_clock import sha256_file
from research.av_search.corpus import load_frozen,strict_json_loads
from .rollout import ToolRuntime,run_episode

QA_SYSTEM='''You are an audio-video search assistant. Answer the user's question using the available tools when needed. The input may contain initial audio-video source clues. Inspect them to obtain observable clues, search the corpus using relevant keywords, and read relevant text. Search results may be in a different language: rewrite unsuccessful queries using short source-language keywords. A single source can suffice; do not manufacture comparisons or a research report. Treat AV model observations as fallible. The frozen pilot provides exact available AV windows in the tool definition; inspect_av must use one of these exact windows, and unavailable windows cannot be inferred. Return ONE JSON object per turn, with no extra prose: {"action":{"tool":"search_sources","arguments":{"query":"keywords","top_k":3}}}, {"action":{"tool":"inspect_av","arguments":{"source_id":"src_...","start_s":0,"end_s":2}}}, {"action":{"tool":"read_text","arguments":{"source_id":"src_...","start_char":0,"end_char":100}}}, or {"final":{"status":"ok","answer":"Direct answer to the question","evidence":[{"source_id":"src_...","start_s":0,"end_s":2}]}}. Text citations use source_id,start_char,end_char,quote. Cite only content actually inspected or read. If evidence is insufficient use final.status="insufficient_evidence" and explain the uncertainty. Do not invent source IDs, observations, identities or details.'''

def prepare_search_batch(pool_dir,reserved_sources,questions,out_dir):
    pool=Path(pool_dir);out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    _,manifest=load_frozen(pool/'authoring-corpus')
    if manifest['split']!='train':raise ValueError('Exploration requires train candidate sources')
    sources=[json.loads(l) for l in (pool/'sources.private.jsonl').read_text().splitlines() if l]
    reserved=[json.loads(l) for l in Path(reserved_sources).read_text().splitlines() if l]
    reserved_hashes={s['asset_sha256'] for s in reserved}
    if reserved_hashes&{s['asset_sha256'] for s in sources}:raise ValueError('reserved dev/test bytes overlap')
    if {s['source_id'] for s in sources}!=set(manifest['source_ids']):raise ValueError('Source registry/corpus mismatch')
    for source in sources:
        if sha256_file(source['asset_path'])!=source['asset_sha256']:raise ValueError('Original asset hash mismatch')
    scope={k:manifest[k] for k in ('corpus_id','split','source_ids')}
    tasks=[];ids=set()
    for q in questions:
        if not isinstance(q,dict) or set(q)!={'task_id','question','initial_sources'}:raise ValueError('Closed search-QA question schema required')
        if q['task_id'] in ids:raise ValueError('Duplicate task ID')
        ids.add(q['task_id'])
        task={'schema_version':TASK_VERSION,**json_copy(q),'corpus_id':scope['corpus_id'],'split':'train',
          'target_questions':[{'fact_id':'answer','question':q['question']}],
          'budget_recipe':{'max_tool_calls':6,'max_requested_av_seconds':8.}}
        validate_task(task,scope);tasks.append(task)
    if not tasks:raise ValueError('Require search-QA tasks')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(pool/'authoring-corpus',out/'corpus')
    write_jsonl(out/'sources.private.jsonl',sources);write_jsonl(out/'tasks-public.jsonl',tasks);write_json(out/'scope.json',scope)
    summary={'schema_version':'av-searchqa-exploration.v1','task_count':len(tasks),'source_count':len(sources),
      'rollout_allowed':True,'benchmark_gold':False,'formal_train_split_frozen':False,
      'source_family_review':'pending','same_byte_overlap_with_reserved':0,'retained_sft_rows':0,
      'profile':'direct_search_qa','pool_source_registry_sha256':sha256_file(pool/'sources.private.jsonl'),
      'reserved_registry_sha256':sha256_file(reserved_sources)}
    write_json(out/'summary.json',summary);return summary


def prepare_runtime_corpus(batch_dir,out_dir):
    from research.av_search.corpus import build_corpus,manifest_digest,DOC_FIELDS
    batch=Path(batch_dir);old=batch/'corpus'
    manifest=strict_json_loads((old/'manifest.json').read_text())
    if manifest_digest(manifest)!=manifest['manifest_sha256'] or sha256_file(old/'documents.jsonl')!=manifest['documents_sha256']:
        raise ValueError('Original public corpus identity changed')
    sources={r['source_id']:r for r in map(json.loads,(batch/'sources.private.jsonl').read_text().splitlines())}
    docs=[]
    for line in (old/'documents.jsonl').read_text().splitlines():
        public=strict_json_loads(line);doc={k:v for k,v in public.items() if k in DOC_FIELDS}
        source=sources[doc['source_id']]
        if source['asset_sha256']!=doc['asset_sha256']:raise ValueError('Runtime registry/hash mismatch')
        doc['asset_path']=source['asset_path'];docs.append(doc)
    build_corpus(docs,out_dir,manifest['corpus_id'],manifest['split'])
    _,new=load_frozen(out_dir)
    if any(new[k]!=manifest[k] for k in ('documents_sha256','source_ids')):
        raise ValueError('Runtime rebuild changed public corpus content')
    return new


def parse_decision(text):
    text=text.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
    value=strict_json_loads(text)
    if not isinstance(value,dict) or set(value) not in ({'action'},{'final'}):raise ValueError('One action or final JSON required')
    if not isinstance(next(iter(value.values())),dict):raise ValueError('Decision payload must be object')
    return value


def decision_is_complete(text):
    try:
        parse_decision(text)
    except (ValueError,TypeError):
        return False
    return True

def write_episode_summary(out_dir,summary):
    root=Path(out_dir)/'episode-summaries';root.mkdir(exist_ok=True)
    task_id=summary['task_id']
    if not isinstance(task_id,str) or not task_id or Path(task_id).name!=task_id:raise ValueError('Safe task ID required')
    write_json(root/(task_id+'.json'),summary)



NATIVE_SYSTEM="""You are an Audio-Video Search Agent, extending OneSearch-VL. Answer the user's question using observed audio-video clues and retrieved publisher facts. Source count is determined by the question. This pilot uses a frozen corpus, not the open web.
Call one function per turn using the published protocol:
<tool_call>{"name":"search_sources","arguments":{"query":"rain","top_k":3}}</tool_call>
Actual tool results are supplied in <observation>...</observation>. NEVER continue into an imagined observation or future action. Finish with <answer>your direct answer</answer>, optionally citing actual source IDs or URLs. You may use a short <think>...</think> before the call or answer.
The tool definitions list exact available AV windows. Use those exact start_s/end_s, not a default 0..2 interval. Failed tools return an error for recovery. For initial media, first inspect an available window to obtain clues, then search. At most 6 tool calls are available; once the requested attribute is in a search excerpt or text quote, finish with <answer> instead of repeatedly rereading it. Search uses short English or source-language terms: Chinese metadata queries often fail. Retry empty searches with translated keywords. read_text supports frozen metadata as well as webpage text; use returned quote_start/quote_end from search results to choose valid bounds. Treat AV descriptions as fallible; missing identities/locations must be looked up. A final answer can report insufficient evidence. Do not fabricate identities, sources or citations."""

def parse_native_decision(text):
    text=text.strip()
    if text.startswith('<think>'):
        end=text.find('</think>')
        if end<0:raise ValueError('Incomplete native thought envelope')
        text=text[end+8:].strip()
    if text.startswith('<tool_call>') and text.endswith('</tool_call>') and text.count('<tool_call>')==1:
        value=strict_json_loads(text[11:-12].strip())
        if not isinstance(value,dict) or set(value)!={'name','arguments'} or not isinstance(value['name'],str) or not isinstance(value['arguments'],dict):
            raise ValueError('Invalid native tool call')
        return {'action':{'tool':value['name'],'arguments':value['arguments']}}
    if text.startswith('<answer>') and text.endswith('</answer>') and text.count('<answer>')==1:
        answer=text[8:-9].strip()
        if not answer:raise ValueError('Empty native answer')
        return {'final':{'status':'ok','answer':answer,'evidence':[]}}
    return parse_decision(text)

def native_decision_is_complete(text):
    try:parse_native_decision(text)
    except (ValueError,TypeError):return False
    return True

def native_history(history):
    output=json_copy(history)
    for item in output:
        if item['role']=='assistant':
            decision=parse_decision(item['content'])
            if 'action' in decision:
                a=decision['action'];item['content']='<tool_call>'+json.dumps({'name':a['tool'],'arguments':a['arguments']},ensure_ascii=False)+'</tool_call>'
        elif item['role']=='user' and item['content'].startswith('<tool_response>'):
            item['content']=item['content'].replace('<tool_response>','<observation>',1).replace('</tool_response>','</observation>',1)
    return output


def controller_history(history,protocol):
    if protocol=='canonical':return json_copy(history)
    if protocol=='native':return native_history(history)
    raise ValueError('Unknown controller protocol')


class LocalController:
    """Published visual controller acting on text from a fixed AV observer."""
    def __init__(self,checkpoint,out_dir,max_new_tokens=384,protocol="native"):
        controller_history([],protocol);self.protocol=protocol
        import torch
        from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration,StoppingCriteria,StoppingCriteriaList
        checkpoint=Path(checkpoint);receipt=json.loads((checkpoint/'download-receipt.json').read_text())
        if receipt.get('repo')!='OneSearch-VL/OneSearch-VL-8B' or receipt.get('verified') is not True:
            raise ValueError('Verified published controller required')
        if sha256_file(checkpoint/'model.safetensors')!=receipt['expected_lfs_sha256']:raise ValueError('Controller weight hash mismatch')
        self.identity={'kind':'model','model':receipt['repo'],'revision':receipt['revision'],
          'role':'bootstrap_controller','input_mode':'text_with_fixed_av_observations'}
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        self.model.config.text_config.use_cache=True
        self.out=Path(out_dir);self.out.mkdir(parents=True,exist_ok=False)
        self.max_new_tokens=max_new_tokens;self.total_tokens=0;self.calls=0;self.torch=torch
        self.stopping_base=StoppingCriteria;self.stopping_list=StoppingCriteriaList
    def __call__(self,history):
        text=self.tokenizer.apply_chat_template(controller_history(history,self.protocol),tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(text,return_tensors='pt').to('cuda:0')
        if inputs['input_ids'].shape[1]>12000:raise ValueError('Controller context budget exceeded')
        start=time.monotonic()
        protocol=self.protocol;tokenizer=self.tokenizer;prefix_len=inputs['input_ids'].shape[1]
        class OneDecisionStop(self.stopping_base):
            def __call__(self,input_ids,scores,**kwargs):
                return (decision_is_complete if protocol=="canonical" else native_decision_is_complete)(tokenizer.decode(input_ids[0,prefix_len:],skip_special_tokens=True))
        with self.torch.inference_mode():
            tokens=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,**getattr(self,'sampling_options',{'do_sample':False}),use_cache=True,
                stopping_criteria=self.stopping_list([OneDecisionStop()]))
        raw=self.tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        self.calls+=1;self.total_tokens+=tokens.shape[1]
        write_json(self.out/f'{self.calls:04d}.private.json',{'raw_output':raw,'input_tokens':inputs['input_ids'].shape[1],
            'generated_tokens':tokens.shape[1]-inputs['input_ids'].shape[1],'wall_seconds':time.monotonic()-start,'sampling_options':getattr(self,'sampling_options',{'do_sample':False}),'episode_seed':getattr(self,'episode_seed',None)})
        return (parse_decision if self.protocol=="canonical" else parse_native_decision)(raw)


class LocalOmniController(LocalController):
    def __init__(self,manifest,out_dir,max_new_tokens=384,protocol="native"):
        controller_history([],protocol);self.protocol=protocol
        import torch
        from transformers import AutoTokenizer,Qwen2_5OmniForConditionalGeneration,StoppingCriteria,StoppingCriteriaList
        from research.av_agent.qwen_omni_backend import _verified_snapshot
        snapshot,artifact,_,_= _verified_snapshot(manifest)
        self.identity={'kind':'model','model':artifact['repository_id'],'revision':artifact['revision'],
          'role':'alternate_bootstrap_teacher','input_mode':'text_with_fixed_av_observations'}
        self.tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        self.parent_model=Qwen2_5OmniForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},enable_audio_output=False).eval()
        self.model=self.parent_model.thinker
        self.model.config.use_cache=True
        self.out=Path(out_dir);self.out.mkdir(parents=True,exist_ok=False)
        self.max_new_tokens=max_new_tokens;self.total_tokens=0;self.calls=0;self.torch=torch
        self.stopping_base=StoppingCriteria;self.stopping_list=StoppingCriteriaList


class SmokeObserverReplay:
    """Read independently executed Omni smoke records without regenerating them."""
    def __init__(self,smoke_dir):
        from research.av_agent.semantic_observer import validate_observation,validate_observer_input
        self.root=Path(smoke_dir)
        self.identity=json.loads((self.root/'model-identity.private.json').read_text())['backend_identity']
        self.records={};self.available_windows=[]
        self.hashes={}
        for p in sorted((self.root/'observations.private').glob('*.json')):
            record=json.loads(p.read_text())
            if validate_observation(record) or record['backend_identity']!=self.identity:raise ValueError('Invalid observer record or identity')
            if record['status']!='model_executed':continue
            if validate_observer_input(record['input'],verify_files=True):raise ValueError('Observer input hash mismatch')
            bound=record['input'];w=bound['original_interval'];key=(bound['source_id'],w['start_s'],w['end_s'])
            if key in self.records:raise ValueError('Ambiguous fixed observation')
            self.records[key]=p;self.hashes[p]=sha256_file(p)
            self.available_windows.append({'source_id':key[0],'start_s':key[1],'end_s':key[2]})
        if not self.records:raise ValueError('No executed AV observations')
    def __call__(self,source,start_s,end_s,out_dir):
        from research.av_agent.semantic_observer import validate_observation,validate_observer_input
        key=(source['source_id'],start_s,end_s)
        if key not in self.records:raise ValueError('No exact executed observer window')
        p=self.records[key]
        if sha256_file(p)!=self.hashes[p]:raise ValueError('Changed observer record')
        record=json.loads(p.read_text());bound=record['input']
        if validate_observation(record) or validate_observer_input(bound,verify_files=True):raise ValueError('Changed observer binding')
        if sha256_file(source['asset_path'])!=bound['asset_sha256'] or source['asset_sha256']!=bound['asset_sha256']:
            raise ValueError('Changed source asset')
        return {'schema_version':'av-searchqa-fixed-observation.v1','status':'ok','source_id':source['source_id'],
          'original_interval':json_copy(bound['original_interval']),'modalities':bound['modalities'],
          'model_executed':True,'model_executed_this_episode':False,'semantic_status':'observed_unreviewed',
          'semantic_verified':False,'observation':json_copy(record['structured_observation']),
          'backend_identity':json_copy(self.identity),'record_sha256':self.hashes[p],
          'actual_cost':{'observer_cache_reads':1,'observer_provider_requests':0},
          'original_observer_cost':json_copy(record['actual_cost'])}

class SearchQARuntime(ToolRuntime):
    def execute(self,action):
        try:return super().execute(action)
        except ValueError:
            a=action.get('arguments',{});sid=a.get('source_id')
            if sid not in self.source_ids or self.cost['tool_calls']>=self.budget['max_tool_calls']:raise
            if action.get('tool')=='read_text':
                from research.av_search.retrieval import get_source
                doc=get_source(self.index_dir,sid,self.scope)
                return {'status':'tool_error','error_type':'ValueError','source_id':sid,
                    'valid_text_range':{'start_char':0,'end_char':len(doc['text'])},
                    'message':'Requested quote was not read. Retry within valid_text_range; maximum 4000 characters per read.'}
            if action.get('tool')=='inspect_av' and self.observer is not None and hasattr(self.observer,'available_windows'):
                return {'status':'tool_error','error_type':'ValueError','source_id':sid,
                    'available_windows':[w for w in self.observer.available_windows if w['source_id']==sid],
                    'message':'Requested interval was not observed. Choose one exact available window.'}
            raise
    def tool_definitions(self):
        definitions=super().tool_definitions()
        if self.observer is not None and hasattr(self.observer,'available_windows'):
            next(t for t in definitions if t['name']=='inspect_av')['constraints']={
              'mode':'fixed_executed_observer_replay','available_windows':json_copy(self.observer.available_windows)}
        return definitions

def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--observer-smoke-dir');p.add_argument('--observer-mode',choices=['replay','media','native'],default='replay');p.add_argument('--out-dir',required=True)
    p.add_argument('--max-tasks',type=int,default=3);p.add_argument('--task-id',action='append')
    p.add_argument('--max-new-tokens',type=int,default=384);p.add_argument('--index-dir')
    p.add_argument('--controller-kind',choices=['onesearch','omni'],default='onesearch');p.add_argument('--controller-model-manifest')
    p.add_argument('--observer-model-manifest');p.add_argument('--observer-code-revision')
    p.add_argument('--observer-runtime-manifest');p.add_argument('--observer-runtime-lock')
    p.add_argument('--observer-device',default='cuda:0')
    p.add_argument('--clue-bindings',help='Private anonymous query-clip bindings; publisher lookup requires search')
    p.add_argument('--protocol',choices=['native','canonical'],default='native')
    p.add_argument('--guided-curriculum',action='store_true',help='Authored initial-source tool recipe plus actual model answer; not autonomous teacher actions')
    return p


def run_guided_curriculum(task,runtime,controller,*,compact_canonical=False):
    from .trajectory import make_step,source_ids
    from .export import public_observation
    if len(task['initial_sources'])!=1:raise ValueError('Guided seed requires one initial source')
    sid=task['initial_sources'][0];source=runtime.sources[sid]
    windows=[w for w in runtime.observer.available_windows if w['source_id']==sid]
    if not windows:raise ValueError('Guided seed requires an executed AV window')
    from research.av_search.retrieval import get_source
    # This authoring recipe uses public publisher title as an assisted lookup clue.
    # It is excluded from autonomous search performance results.
    initial_doc=get_source(runtime.index_dir,sid,runtime.scope)
    query=source['public_metadata']['title'].removeprefix('File:')
    actions=[{'tool':'inspect_av','arguments':windows[0]},
             {'tool':'search_sources','arguments':{'query':query,'top_k':3}},
             {'tool':'read_text','arguments':{'source_id':sid,'start_char':0,'end_char':min(4000,len(initial_doc['text']))}}]
    system=QA_SYSTEM if compact_canonical else NATIVE_SYSTEM
    steps=[];history=[{'role':'system','content':system+'\nTools: '+json.dumps(runtime.tool_definitions(),ensure_ascii=False)},
                     {'role':'user','content':json.dumps(task,ensure_ascii=False)}]
    for action in actions:
        receipt=runtime.execute(action)
        if receipt['status']!='ok':raise ValueError('Guided real tool execution failed')
        steps.append(make_step(action,receipt))
        history.extend([{'role':'assistant','content':json.dumps({'action':action},ensure_ascii=False)},
            {'role':'user','content':'<tool_response>\n'+json.dumps(public_observation(receipt),ensure_ascii=False)+'\n</tool_response>'}])
    history.append({'role':'user','content':'The initial-source inspection, title-assisted search and publisher text read have now actually executed. Answer the original question directly using the observed publisher description. Preserve proper names, dates and model numbers exactly as written in that text. This is the final turn: return only <answer>your answer</answer>, no further tool calls. If uncertain, say so.'})
    if compact_canonical:
        history[-1]['content']='Final turn. Answer the original question in ONE concise English sentence, preserving publisher proper names exactly. Return only a complete JSON object: {"final":{"status":"ok","answer":"...","evidence":[]}}. No tools, reasoning or repeated sentences. If the requested field is missing, use status insufficient_evidence.'
    try:decision=controller(history);final=decision.get('final',{'status':'no_final_answer'})
    except Exception as exc:final={'status':'controller_failed','error_type':type(exc).__name__}
    identity=json_copy(controller.identity);identity['trajectory_origin']='authored_tool_curriculum_with_actual_model_final'
    identity['assistance']='public_publisher_title_lookup; not autonomous source discovery'
    trace={'schema_version':'av-research-trajectory.v1','task':json_copy(task),'teacher':identity,
      'system':system,'tools':runtime.tool_definitions(),'steps':steps,'final':final,
      'execution_kind':'live_tools','referenced_source_ids':[]}
    trace['referenced_source_ids']=sorted(source_ids(trace));final['cost']=json_copy(runtime.cost)
    return trace


def main():
    a=make_parser().parse_args()
    if a.clue_bindings and a.guided_curriculum:raise ValueError('Anonymous query clues cannot use title-assisted guided curriculum')
    if a.guided_curriculum and a.observer_mode!='replay':raise ValueError('Guided curriculum requires fixed executed observer records')
    if a.controller_kind=='omni' and not a.controller_model_manifest:raise ValueError('Pinned Omni teacher manifest required')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    batch=Path(a.batch_dir);index=Path(a.index_dir) if a.index_dir else batch/'corpus';load_frozen(index);sources={r['source_id']:r for r in map(json.loads,(batch/'sources.private.jsonl').read_text().splitlines())}
    tasks=list(map(json.loads,(batch/'tasks-public.jsonl').read_text().splitlines()))
    if a.task_id:tasks=[t for t in tasks if t['task_id'] in a.task_id]
    tasks=tasks[:a.max_tasks]
    if a.observer_mode=='replay' and not a.observer_smoke_dir:raise ValueError('Replay observer requires an executed smoke directory')
    if a.observer_mode=='media' and a.observer_smoke_dir:raise ValueError('Media-only mode cannot ignore supplied replay directory')
    if a.observer_mode=='media' and a.guided_curriculum:raise ValueError('Guided curriculum requires executed observer records')
    native_fields=(a.observer_model_manifest,a.observer_code_revision,a.observer_runtime_manifest,a.observer_runtime_lock)
    if a.observer_mode!='native' and any(native_fields):raise ValueError('Native configuration cannot be ignored by another observer mode')
    if a.observer_mode=='native':
        if a.observer_smoke_dir or not all(native_fields):raise ValueError('Dynamic observer requires complete pinned native configuration')
        from types import SimpleNamespace
        from .rollout import build_observer
        observer=build_observer(SimpleNamespace(observer_mode='native',observer_model_manifest=a.observer_model_manifest,observer_code_revision=a.observer_code_revision,observer_runtime_manifest=a.observer_runtime_manifest,observer_runtime_lock=a.observer_runtime_lock,observer_device=a.observer_device,observer_execution_dir=[],observer_backend_identity=None),out)
    else:
        observer=SmokeObserverReplay(a.observer_smoke_dir) if a.observer_mode=='replay' else None
    controller=(LocalOmniController(a.controller_model_manifest,out/'controller-generations',a.max_new_tokens,protocol=a.protocol) if a.controller_kind=='omni' else LocalController(a.checkpoint,out/'controller-generations',a.max_new_tokens,protocol=a.protocol))
    summaries=[]
    bindings=json.loads(Path(a.clue_bindings).read_text()) if a.clue_bindings else None
    controller.identity.update({'protocol':a.protocol,'observer_mode':a.observer_mode,
      'input_mode':{'replay':'text_with_fixed_av_observations','media':'text_with_media_preparation_receipts_no_semantic_observer','native':'text_with_dynamic_av_observations'}[a.observer_mode],
      'input_profile':'anonymous_query_clip' if bindings is not None else 'source_visible_pilot'})
    system_prompt=QA_SYSTEM if a.protocol=='canonical' else NATIVE_SYSTEM
    if bindings is not None:
        from .clue_searchqa import CLUE_SYSTEM
        system_prompt+='\n'+CLUE_SYSTEM
    for task in tasks:
        before_calls=controller.calls;before_tokens=controller.total_tokens
        scope=json.loads((batch/'scope.json').read_text())
        if bindings is not None:
            from .clue_searchqa import ClueSearchQARuntime
            selected=[b for b in bindings if b['clue_id'] in task['initial_sources']]
            runtime=ClueSearchQARuntime(index,scope,sources,selected,out/task['task_id']/'tools',task['budget_recipe'],observer)
        else:
            runtime=SearchQARuntime(index,scope,sources,out/task['task_id']/'tools',task['budget_recipe'],observer)
        trace=(run_guided_curriculum(task,runtime,controller,compact_canonical=a.protocol=='canonical') if a.guided_curriculum else run_episode(task,runtime,controller,controller.identity,system_prompt=system_prompt,recover_tool_errors=True))
        trace['final']['cost']['episode_controller_tokens']=controller.total_tokens-before_tokens
        write_json(out/task['task_id']/'trace.private.json',trace)
        summary={'task_id':task['task_id'],'status':trace['final'].get('status'),'answer':trace['final'].get('answer'),
          'model_decisions':controller.calls-before_calls,'tools':[s['action']['tool'] for s in trace['steps']],
          'observer_cache_reads':runtime.cost.get('observer_cache_reads',0),'semantic_verified':False,'retained_sft_rows':0}
        summaries.append(summary);write_episode_summary(out,summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    write_json(out/'summary.json',{'schema_version':'av-searchqa-bootstrap-run.v1','controller':controller.identity,
      'observer_mode':{'replay':'fixed_executed_replay','media':'media_ready_observer_unavailable','native':'dynamic_native_av'}[a.observer_mode],'attempted_tasks':len(summaries),'model_decisions':controller.calls,
      'answer_submission_count':sum(s['status']=='ok' and bool(s['answer']) for s in summaries),
      'episodes':summaries,'semantic_verified_count':0,'retained_sft_rows':0,'optimizer_updates':0})

if __name__=='__main__':main()

"""Prepare and execute fixed multiwindow AV observations without publisher clues.

Long input duration is independent of each native observer call. Repeated
questions on identical clipped bytes reuse the same executed input observations.
"""
import argparse,json,math
from pathlib import Path
from research.av_bench.contracts import json_copy
from research.av_data.build_research_batch import write_json
from research.av_data.media_clock import sha256_file

def window_schedule(duration_s):
    if isinstance(duration_s,bool) or not isinstance(duration_s,(int,float)) or not math.isfinite(duration_s) or duration_s<=0:raise ValueError('Valid positive clip duration required')
    if duration_s<=2.:return [{'start_s':0.,'end_s':float(duration_s)}]
    span=duration_s-2.
    return [{'start_s':x,'end_s':min(duration_s,x+2.)} for x in [0.,span/2.,span]]

def group_clues(split_bindings):
    groups={}
    for split,bindings in split_bindings:
        for b in bindings:
            source=b['source'];key=source['asset_sha256']
            if key in groups:
                old=groups[key]
                if old['split']!=split:raise ValueError('Identical query clip crosses split')
                if old['source']!=source:raise ValueError('Ambiguous source binding for identical bytes')
                old['clue_ids'].append(b['clue_id'])
            else:groups[key]={'source':json_copy(source),'split':split,'clue_ids':[b['clue_id']]}
    return list(groups.values())

def make_plan(batch_dirs,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=False);splits=[];batch_hashes={}
    for dirname in batch_dirs:
        batch=Path(dirname);scope=json.loads((batch/'scope.json').read_text());bindings=json.loads((batch/'clue-bindings.private.json').read_text())
        splits.append((scope['split'],bindings));batch_hashes[str(batch)]={'tasks_sha256':sha256_file(batch/'tasks-public.jsonl'),'bindings_sha256':sha256_file(batch/'clue-bindings.private.json')}
    groups=group_clues(splits);items=[]
    for group in groups:
        source=group['source']
        if source.get('public_metadata')!={} or source.get('media_type')!='audio_video' or sha256_file(source['asset_path'])!=source['asset_sha256']:raise ValueError('Verified anonymous AV source required')
        for w in window_schedule(source['duration_s']):items.append({'source':source,'split':group['split'],'clue_ids':group['clue_ids'],'window':w})
    plan={'schema_version':'av-multiwindow-plan.v1','batches':batch_hashes,'unique_query_assets':len(groups),'window_count':len(items),'windows':items,
      'observer_executed':False,'title_or_answer_supplied':False,'semantic_verified':False}
    write_json(out/'plan.private.json',plan);return plan

def prepare(plan_file,out_dir):
    from research.av_agent.observer import prepare_av_input
    from research.av_agent.semantic_observer import make_observer_input
    plan=json.loads(Path(plan_file).read_text());out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    bound_inputs=[]
    for i,item in enumerate(plan['windows'],1):
        folder=out/f'{i:04d}';source=item['source'];w=item['window']
        derived,cost=prepare_av_input(source,w['start_s'],w['end_s'],folder/'media')
        bound=make_observer_input(source,derived,probe_id=f'multiwindow-{i:04d}')
        write_json(folder/'input.private.json',bound);write_json(folder/'derived.private.json',derived);write_json(folder/'preparation-cost.private.json',cost)
        bound_inputs.append(bound)
        print(json.dumps({'prepared':i,'total':len(plan['windows']),'split':item['split'],'window':w}),flush=True)
    payload={'schema_version':'av-multiwindow-prepared-inputs.v1','inputs':bound_inputs,'plan_sha256':sha256_file(plan_file)}
    write_json(out/'bindings.private.json',payload);return payload

def execute(bindings_file,model_manifest,out_dir):
    from research.av_agent.semantic_observer import SemanticObserver,validate_observer_input
    from research.av_agent.qwen_omni_backend import QwenOmniBackend
    from research.scripts.run_av_observer_smoke import collect_runtime_receipt
    import subprocess
    payload=json.loads(Path(bindings_file).read_text());inputs=payload['inputs']
    if payload.get('schema_version')!='av-multiwindow-prepared-inputs.v1' or not 1<=len(inputs)<=90:raise ValueError('Prepared bounded multiwindow inputs required')
    for bound in inputs:
        if validate_observer_input(bound,verify_files=True) or bound['modalities']!=['audio','video'] or bound['original_interval']['end_s']-bound['original_interval']['start_s']>2.00000001:raise ValueError('Changed or invalid prepared input')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=False);(out/'observations.private').mkdir()
    live=collect_runtime_receipt('runtime/av-observer/runtime-manifest.json','runtime/av-observer/pixi.lock');write_json(out/'live-runtime.private.json',live)
    if live['status']!='verified':raise ValueError('Observer runtime mismatch')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    backend=QwenOmniBackend(model_manifest,backend_code_revision=head,device='cuda:0',max_new_tokens=256,
      max_clip_seconds=2.,video_fps=2.,max_pixels=50176,clock_policy='av_pts_grid_v1',output_policy='json_object_v1')
    write_json(out/'model-identity.private.json',{'backend_identity':backend.manifest,'artifact_identity':backend.model_artifact_identity,
      'model_hash_verification_seconds':backend.model_hash_verification_seconds,'model_load_seconds':backend.model_load_seconds})
    write_json(out/'execution-inputs.private.json',{'bindings_sha256':sha256_file(bindings_file),'git_head':head,'planned_inputs':len(inputs)})
    observer=SemanticObserver(backend,max_calls=len(inputs));results=[]
    for i,bound in enumerate(inputs,1):
        record=observer.observe(bound);write_json(out/'observations.private'/f'{i:04d}.json',record)
        write_json(out/f'backend-metrics-{i:04d}.private.json',backend.last_metrics)
        result={'input_index':i,'status':record['status'],'source_id':bound['source_id'],'interval':bound['original_interval']};results.append(result)
        print(json.dumps(result),flush=True)
    summary={'schema_version':'av-multiwindow-execution.v1','planned_inputs':len(inputs),'executed_inputs':sum(x['status']=='model_executed' for x in results),
      'failures':sum(x['status']!='model_executed' for x in results),'semantic_verified':False,'results':results,'controller_executed':False,'optimizer_updates':0}
    write_json(out/'summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('plan');a.add_argument('--batch-dir',action='append',required=True);a.add_argument('--out-dir',required=True)
    a=sub.add_parser('prepare');a.add_argument('--plan',required=True);a.add_argument('--out-dir',required=True)
    a=sub.add_parser('execute');a.add_argument('--bindings',required=True);a.add_argument('--model-manifest',required=True);a.add_argument('--out-dir',required=True)
    a=p.parse_args()
    result=make_plan(a.batch_dir,a.out_dir) if a.command=='plan' else prepare(a.plan,a.out_dir) if a.command=='prepare' else execute(a.bindings,a.model_manifest,a.out_dir)
    print(json.dumps({k:v for k,v in result.items() if k not in {'windows','inputs','results'}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

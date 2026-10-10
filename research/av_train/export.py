"""ShareGPT controller export from independently replayed, semantically retained traces.

The modular route trains the controller on fixed observer TEXT, not native audio.
Observations are context turns. Check real LLaMA-Factory IGNORE_INDEX labels before
training; role correctness is not an executed token-label audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .trajectory import validate_trace,digest,stable_observation,source_ids

PRIVATE_OUTPUT_KEYS={'asset_path','output_path','private_path','asset_root'}


def public_observation(value):
    if isinstance(value,dict):return {k:public_observation(v) for k,v in value.items() if k not in PRIVATE_OUTPUT_KEYS}
    if isinstance(value,list):return [public_observation(x) for x in value]
    return value


def export_trajectory(trace,source_splits,replay,verdict):
    t=validate_trace(trace,source_splits)
    if verdict.get('semantic_verified') is not True or not verdict.get('reviewer_id') or verdict.get('evaluation',{}).get('ebs')!=1:
        raise ValueError('Successful semantic verification required')
    av_steps=[step for step in t['steps'] if step['action']['tool']=='inspect_av'
              and step['observation'].get('status')=='ok'
              and step['observation'].get('semantic_status') in {'observed','observed_unreviewed'}]
    if not av_steps:raise ValueError('At least one executed semantic AV observation is required')
    for citation in t['final'].get('evidence',[]):
        if 'start_s' in citation:
            observed=False
            for step in av_steps:
                obs=step['observation'];window=obs.get('original_interval',{})
                if obs.get('source_id')==citation.get('source_id') and window.get('start_s',float('inf'))<=citation['start_s']<citation['end_s']<=window.get('end_s',-1):observed=True
            if not observed:raise ValueError('Citation outside observed AV windows')
        elif 'start_char' in citation:
            observed=any(step['action']['tool']=='read_text' and step['observation'].get('source_id')==citation.get('source_id')
                and step['observation'].get('quote',{}).get('start_char',float('inf'))<=citation['start_char']<citation['end_char']<=step['observation'].get('quote',{}).get('end_char',-1)
                for step in t['steps'])
            if not observed:raise ValueError('Citation outside observed text')
    messages=[{'from':'human','value':json.dumps(t['task'],ensure_ascii=False,allow_nan=False)}]
    for step in t['steps']:
        replayed=replay(step['action'])
        if digest(stable_observation(replayed))!=step['replay_sha256']:raise ValueError('Independent tool replay mismatch')
        if any(source_splits.get(s)!='train' for s in source_ids(replayed)):raise ValueError('Replay source split mismatch')
        messages.extend([{'from':'gpt','value':json.dumps({'action':step['action']},ensure_ascii=False,allow_nan=False)},
                         {'from':'observation','value':json.dumps(public_observation(step['observation']),ensure_ascii=False,allow_nan=False)}])
    messages.append({'from':'gpt','value':json.dumps({'final':{k:v for k,v in t['final'].items() if k!='cost'}},ensure_ascii=False,allow_nan=False)})
    return {'system':t['system']+'\nTools: '+json.dumps(t['tools'],ensure_ascii=False,allow_nan=False),'tools':'','conversations':messages}


def dataset_registration(file_name='av_research/retained.json'):
    return {'file_name':file_name,'formatting':'sharegpt','columns':{'messages':'conversations','system':'system','tools':'tools'},
            'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt','observation_tag':'observation'}}


def export_batch(traces,source_splits,replay_factory,verdicts,out_dir):
    """Create-only retained dataset, rejection audit and trace/verdict hashes.

    replay_factory(trace) returns an independently configured tool executor, not
    a function reading observations out of that trace. Caller is trusted pipeline
    code. Content hashes cannot authenticate a dishonest caller/reviewer.
    """
    from research.av_data.build_research_batch import write_json
    from datetime import datetime,timezone
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    retained=[];rejections=[];lineage=[]
    for trace in traces:
        tid=trace.get('task',{}).get('task_id','missing-task-id')
        verdict=verdicts.get(tid,{})
        try:
            # Validate before constructing a potentially expensive replay backend.
            validate_trace(trace,source_splits)
            row=export_trajectory(trace,source_splits,replay_factory(trace),verdict)
        except (ValueError,KeyError,TypeError) as exc:
            rejections.append({'task_id':tid,'reason':str(exc),'trace_sha256':digest(trace)})
            continue
        retained.append(row);lineage.append({'row_index':len(retained)-1,'task_id':tid,
            'trace_sha256':digest(trace),'verdict_sha256':digest(verdict),'teacher':trace['teacher'],
            'source_ids':sorted(source_ids(trace)),'step_count':len(trace['steps']),
            'assistant_target_turns':len(trace['steps'])+1})
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'retained.json',retained);write_json(out/'rejections.private.json',rejections)
    write_json(out/'lineage.private.json',lineage)
    write_json(out/'dataset_info.json',{'av_research_sft':dataset_registration('retained.json')})
    (out/'data-settings.yaml').write_text('dataset: av_research_sft\ntrain_on_prompt: false\nmask_history: false\ncutoff_len: 32768\npacking: false\n',encoding='utf-8')
    summary={'schema_version':'av-research-sft-export.v1','created_at':datetime.now(timezone.utc).isoformat(),
             'attempt_count':len(traces),'retained_count':len(retained),'rejected_count':len(rejections),
             'retention_rate':len(retained)/len(traces) if traces else 0.,'training_ready':False,
             'token_label_audit':'required_not_run','sequence_length_audit':'required_not_run',
             'data_sha256':digest(retained),'source_split_sha256':digest(source_splits),
             'execution_replay_required':True,'semantic_review_required':True}
    write_json(out/'summary.json',summary)
    return summary

"""Paired development diagnostics; pending candidates are never reported as gold accuracy."""
import argparse,json
from pathlib import Path
from research.av_data.build_research_batch import write_json

def compare(run_dir,out_file):
    root=Path(run_dir);variants={};input_identity=None
    for name in ('base','answer_only','trajectory'):
        manifest=json.loads((root/f'dev-{name}-v1/input-manifest.private.json').read_text())
        shared_keys=('mode','tasks_sha256','bindings_sha256','observer_dir','sampling','model_tool_calls','max_new_tokens','initial_context_policy','checkpoint')
        current={k:manifest[k] for k in shared_keys}
        if input_identity is None:input_identity=current
        elif current!=input_identity:raise ValueError('Paired variants have different inputs, budget, sampling or base checkpoint')
        summary=json.loads((root/f'dev-{name}-v1/summary.json').read_text())
        if summary.get('mode')!='evaluate' or summary.get('attempts')!=16:raise ValueError('Complete 16-question development diagnostic required')
        rows=summary['results'];by_id={r['task_id']:r for r in rows}
        if len(by_id)!=16 or len(rows)!=16:raise ValueError('Repeated or missing dev question')
        variants[name]=by_id
    if any(set(x)!=set(variants['base']) for x in variants.values()):raise ValueError('Paired task identities differ')
    rows=[]
    for tid in sorted(variants['base']):
        values={name:items[tid]['diagnostic'] for name,items in variants.items()}
        rows.append({'task_id':tid,'variants':values})
    summary={}
    for name,items in variants.items():
        d=[r['diagnostic'] for r in items.values()]
        summary[name]={'questions':len(d),'all_requested_field_matches':sum(x['all_requested_fields_match'] for x in d),'target_retrieved':sum(x['target_retrieved'] for x in d),'target_read':sum(x['target_read'] for x in d),'tool_attempts':sum(x['tool_attempts_in_trace'] for x in d),'all_budget_compliant':all(x['budget_compliant'] for x in d),'answer_accuracy':None}
    result={'schema_version':'av-paired-development-diagnostics.v1','summary':summary,'matched_input_identity':input_identity,'paired_questions':rows,'formal_benchmark':False,'semantic_verified':False,'scope':'publisher field matches and retrieval/read diagnostics; human answer verdicts pending'}
    write_json(out_file,result);print(json.dumps(summary),flush=True);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--out-file',required=True);a=p.parse_args();compare(a.run_dir,a.out_file)
if __name__=='__main__':main()

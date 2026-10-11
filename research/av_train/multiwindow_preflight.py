"""Freeze already prepared train/dev windows and rebuild runtime search indices."""
import argparse,json,subprocess
from pathlib import Path
from research.av_data.build_research_batch import write_json
from research.av_data.media_clock import sha256_file
from research.av_agent.semantic_observer import validate_observer_input
from research.av_train.multiwindow import group_clues
from research.av_train.searchqa import prepare_runtime_corpus

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
    split_batches={'train':r/'train21-v2','dev':Path('research/runs/av-clue-searchqa-20261012-v1/development-context-final-v3')}
    inputs=[];identities={};groups=[]
    for split,batch in split_batches.items():
        planfile=r/f'{split}-plan/plan.private.json';prepared=r/f'{split}-prepared/bindings.private.json'
        plan=json.loads(planfile.read_text());bound=json.loads(prepared.read_text())
        if bound['plan_sha256']!=sha256_file(planfile) or len(bound['inputs'])!=len(plan['windows']):raise ValueError('Incomplete or changed prepared inputs')
        for item,value in zip(plan['windows'],bound['inputs']):
            if item['split']!=split or value['source_id']!=item['source']['source_id'] or value['asset_sha256']!=item['source']['asset_sha256'] or value['original_interval']!=item['window'] or validate_observer_input(value,verify_files=True):raise ValueError('Prepared input does not match frozen plan')
        frozen=plan['batches'][str(batch)]
        if frozen['tasks_sha256']!=sha256_file(batch/'tasks-public.jsonl') or frozen['bindings_sha256']!=sha256_file(batch/'clue-bindings.private.json'):raise ValueError('Batch changed during preparation')
        inputs.extend(bound['inputs']);groups.append((split,json.loads((batch/'clue-bindings.private.json').read_text())))
        identities[split]={'prepared_sha256':sha256_file(prepared),'plan_sha256':sha256_file(planfile),'windows':len(bound['inputs']),**frozen}
    group_clues(groups)
    if len(inputs)!=75 or identities['train']['windows']!=27 or identities['dev']['windows']!=48:raise ValueError('Expected exactly 27 train and 48 dev windows')
    out=r/'gpu-inputs-v1';out.mkdir(exist_ok=False)
    write_json(out/'bindings.private.json',{'schema_version':'av-multiwindow-prepared-inputs.v1','inputs':inputs,'split_receipts':identities})
    for split,batch in split_batches.items():prepare_runtime_corpus(batch,out/f'corpus-{split}-sft')
    write_json(out/'preflight.private.json',{'status':'prepared','git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs':identities,'window_count':75,'observer_executed':False,'locks':{str(p):sha256_file(p) for p in [Path('runtime/av-observer/pixi.lock'),Path('runtime/sft-gpu/pixi.lock')]}})
    print(json.dumps({'prepared_windows':75,'train_windows':27,'dev_windows':48,'observer_executed':False}),flush=True)
if __name__=='__main__':main()

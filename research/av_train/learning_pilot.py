"""Helpers for source-grouped, paired native-Omni learning experiments."""
import hashlib
from pathlib import Path
from .native_sft import digest,file_sha


def group_holdout(records, holdout_questions=40, seed=42):
    """Group transitive recording-family and shared AV-byte connections."""
    if not records or not 1 <= holdout_questions < len(records):
        raise ValueError('Require a nonempty train and held-out pool')
    ids=[r['task_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Repeated task IDs are not distinct examples')
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    owners={}
    for i,row in enumerate(records):
        keys=['family:'+row['family_id']]+['media:'+m for m in row['media_hashes']]
        for key in keys:
            if key in owners:parent[find(i)]=find(owners[key])
            owners[key]=i
    groups={}
    for i,row in enumerate(records):groups.setdefault(find(i),[]).append(row)
    ordered=sorted(groups.values(),key=lambda rows:hashlib.sha256((str(seed)+'|'+','.join(sorted(r['task_id'] for r in rows))).encode()).hexdigest())
    if len(ordered)<2:raise ValueError('All records share a recording group; no independent holdout')
    held=[];train=[]
    for index,group in enumerate(ordered):
        if len(held)<holdout_questions and index<len(ordered)-1:held.extend(group)
        else:train.extend(group)
    return sorted(train,key=lambda r:r['task_id']),sorted(held,key=lambda r:r['task_id'])


def verify_row_audit(row, lineage, audit, variant, index):
    """Bind unchanged exported rows to their original per-row CPU receipt."""
    key={'trajectory':'row_sha256','answer_only':'answer_row_sha256'}.get(variant)
    if key is None or digest(row)!=lineage[key]:raise ValueError('Export row differs from its audited lineage')
    matches=[r for r in audit['rows'] if r['dataset']==variant and r['row_index']==index]
    if len(matches)!=1:raise ValueError('Exactly one original row audit is required')
    return matches[0]


def attach_thinker_adapter(runner, adapter_dir):
    """Attach the original trainer's Thinker LoRA, verify every saved tensor."""
    import torch
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    folder=Path(adapter_dir).resolve()
    thinker=runner._model.thinker
    if isinstance(thinker,PeftModel):raise ValueError('Adapters must start from the original base Thinker')
    expected=load_file(folder/'adapter_model.safetensors')
    if not expected or any(not torch.isfinite(v).all() for v in expected.values()):raise ValueError('Finite saved adapter required')
    loaded=PeftModel.from_pretrained(thinker,folder,is_trainable=False,autocast_adapter_dtype=True)
    actual=get_peft_model_state_dict(loaded)
    if set(actual)!=set(expected) or any(not torch.equal(v,actual[k].detach().cpu()) for k,v in expected.items()):
        raise ValueError('Loaded adapter differs from original trainer output')
    loaded.eval();runner._model.thinker=loaded
    return {'adapter_path':str(folder),'adapter_sha256':file_sha(folder/'adapter_model.safetensors'),
            'adapter_reload_exact':True,'adapter_tensor_count':len(expected),'base_thinker_only':True}

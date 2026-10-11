"""Bounded LoRA engineering smoke on silver rows using native trainer token labels.

Custom PyTorch optimizer loop, not the full official OneSearch-VL SFT recipe.
Four updates on a tiny assisted seed do not demonstrate performance improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

def optimizer_step(model,batch,optimizer):
    import torch
    count=int((batch['labels']!=-100).sum().item())
    if count==0:raise ValueError('No supervised targets')
    optimizer.zero_grad(set_to_none=True)
    loss=model(**batch).loss
    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
    loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,1.)
    if not torch.isfinite(norm) or norm.item()<=0:raise ValueError('No finite nonzero gradient')
    optimizer.step()
    return {'loss':float(loss.detach()),'gradient_norm':float(norm),'supervised_tokens':count}

def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--checkpoint',default='repro/assets/model-8b');p.add_argument('--steps',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.steps<=8:raise ValueError('Bounded optimizer step count required')
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model,PeftModel
    from research.av_train.label_audit import audit_rows,native_components
    from research.av_data.build_research_batch import write_json
    from research.av_data.media_clock import sha256_file
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads(Path(a.rows).read_text())
    report=audit_rows(rows,a.checkpoint,cutoff_len=8192)
    write_json(out/'native-label-audit.private.json',report)
    converter,_,processor,_=native_components(a.checkpoint,8192)
    batches=[]
    for row in rows:
        aligned=converter(row)
        ids,labels=processor._encode_data_example(aligned['_prompt'],aligned['_response'],aligned['_system'],aligned['_tools'],[],[],[])
        batches.append({'input_ids':torch.tensor([ids],device='cuda:0'),
           'labels':torch.tensor([labels],device='cuda:0'),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long,device='cuda:0')})
    receipt=json.loads((Path(a.checkpoint)/'download-receipt.json').read_text())
    if sha256_file(Path(a.checkpoint)/'model.safetensors')!=receipt['expected_lfs_sha256']:raise ValueError('Base weight hash mismatch')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.checkpoint,local_files_only=True,torch_dtype=torch.bfloat16,
          device_map={'':'cuda:0'},attn_implementation='sdpa')
    model.config.text_config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
          target_modules=r'.*language_model.*\.(q_proj|v_proj)',task_type='CAUSAL_LM'))
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all('language_model' in n and 'lora_' in n and 'visual' not in n for n in names)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    initial=parameter_digest(model);updates=[];start=time.monotonic()
    for step in range(a.steps):
        result=optimizer_step(model,batches[step%len(batches)],optimizer);scheduler.step();result['step']=step+1
        updates.append(result);write_json(out/f'update-{step+1:03d}.json',result);print(json.dumps(result),flush=True)
    final=parameter_digest(model);assert initial!=final,'Trainable parameters did not change'
    checkpoint=out/'checkpoint';model.save_pretrained(checkpoint,safe_serialization=True)
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),
      'cuda_rng':torch.cuda.get_rng_state_all(),'global_step':a.steps},checkpoint/'training_state.pt')
    base=model.unload();reloaded=PeftModel.from_pretrained(base,checkpoint,is_trainable=True)
    reload_digest=parameter_digest(reloaded);assert reload_digest==final,'Adapter reload mismatch'
    summary={'schema_version':'onesearch-av-lora-engineering-smoke.v1','status':'passed','dataset_quality':'silver_assisted_curriculum',
      'row_count':len(rows),'optimizer_updates':len(updates),'updates':updates,'training_wall_seconds':time.monotonic()-start,
      'initial_trainable_sha256':initial,'final_trainable_sha256':final,'reload_trainable_sha256':reload_digest,
      'adapter_reload_verified':True,'optimizer_scheduler_rng_saved':True,'base_model_revision':receipt['revision'],
      'rows_sha256':sha256_file(a.rows),'lora_target_parameter_names':names,'performance_comparison_run':False,
      'recipe':'custom bounded PyTorch loop using actual LLaMA-Factory converter/template/labels; not official full recipe'}
    write_json(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in {'lora_target_parameter_names','updates'}},indent=2))
if __name__=='__main__':main()

"""Resource validation for full Thinker using the unchanged original trainer."""
from pathlib import Path
import json,os,re,sys,math

def allocation_mask(uuids,nvml,driver,expected=4):
 from research.scripts.av_gpu_driver_probe import bus_key
 if len(uuids)!=expected or len(set(uuids))!=expected or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+',u) for u in uuids):raise ValueError('distinct allocated GPU UUIDs required')
 ordinals=[]
 for u in uuids:
  if u not in nvml:raise ValueError('allocated GPU absent from NVML')
  matches=[d['ordinal'] for d in driver if bus_key(d['pci_bus_id'])==bus_key(nvml[u])]
  if len(matches)!=1:raise ValueError('allocated PCI mapping not unique')
  ordinals.append(matches[0])
 if len(set(ordinals))!=expected:raise ValueError('allocated PCI identities collide')
 return ','.join(map(str,ordinals))

def select_indices(rows,n=12):
 if n<3 or len(rows)<n:raise ValueError('enough distinct rows required')
 chosen=[]
 for key in [lambda r:r['trajectory_tokens'],lambda r:r['supervised_tokens'],lambda r:len(r['media_hashes'])]:
  i=max(range(len(rows)),key=lambda i:(key(rows[i]),-i))
  if i not in chosen:chosen.append(i)
 for i in sorted(range(len(rows)),key=lambda i:(-rows[i]['trajectory_tokens'],i)):
  if i not in chosen:chosen.append(i)
  if len(chosen)==n:break
 return chosen

def policy(model,expected_trainable=7615616512,expected_frozen=1316197376):
 rows=list(model.named_parameters())
 trainable=sum(p.numel() for n,p in rows if p.requires_grad);frozen=sum(p.numel() for n,p in rows if not p.requires_grad)
 assert (trainable,frozen)==(expected_trainable,expected_frozen),'full Thinker parameter counts changed'
 assert not any('lora_' in n or 'talker' in n for n,p in rows)
 assert all(not p.requires_grad for n,p in rows if 'audio_tower' in n or 'visual' in n)
 return {'trainable':trainable,'frozen':frozen,'trainable_tensors':sum(p.requires_grad for n,p in rows),'frozen_tensors':sum(not p.requires_grad for n,p in rows)}

def probe(run):
 import subprocess
 from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
 from research.av_train.teacher_rollouts import atomic_json,now
 assigned=os.environ.get('NVIDIA_VISIBLE_DEVICES','').split(',')
 assert len(assigned)==4
 nvml={}
 for u in assigned:
  assert re.fullmatch(r'GPU-[0-9a-fA-F-]+',u)
  bus,uuid=subprocess.check_output(['nvidia-smi','--id='+u,'--query-gpu=pci.bus_id,uuid','--format=csv,noheader'],text=True,timeout=30).strip().split(',')
  assert uuid.strip()==u;nvml[u]=bus.strip()
 os.environ.pop('CUDA_VISIBLE_DEVICES',None);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
 mask=allocation_mask(assigned,nvml,driver_inventory())
 (run/'cluster/cuda-mask.txt').write_text(mask+'\n')
 atomic_json(run/'cluster/driver-probe.json',{'at_utc':now(),'status':'mapped','allocated_uuids':assigned,'allocated_pci':[nvml[u] for u in assigned],'mask':mask,'unassigned_compute':False})

def runtime_preflight(run):
 import torch,hashlib
 from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
 from research.av_train.teacher_rollouts import atomic_json,now
 receipt=json.loads((run/'cluster/driver-probe.json').read_text());visible=driver_inventory()
 assert len(visible)==torch.cuda.device_count()==4
 assert [bus_key(d['pci_bus_id']) for d in visible]==[bus_key(b) for b in receipt['allocated_pci']]
 devices=[]
 for i in range(4):
  gib=torch.cuda.get_device_properties(i).total_memory/2**30
  assert gib>=120,'full parameter FP32/optimizer policy requires >=120 GiB for this smoke'
  a=torch.ones((32,32),device='cuda:'+str(i));b=a@a;torch.cuda.synchronize(i);assert b[0,0].item()==32
  devices.append({'gpu':torch.cuda.get_device_name(i),'total_gib':gib,'assigned_pci':receipt['allocated_pci'][i],'actual_matmul':True})
 freeze=json.loads((run/'cluster/source-freeze.json').read_text())
 for path,h in freeze['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'frozen source changed'
 manifest=json.loads(Path(freeze['model_manifest']).read_text())
 assert manifest['status']=='verified' and manifest['all_lfs_verified']
 for f in manifest['files']:
  h=hashlib.sha256()
  with Path(f['local_path']).open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==f['expected_sha256']
 atomic_json(run/'cluster/gpu-runtime.json',{'at_utc':now(),'devices':devices,'world_size':4,'model_rehashed':True,'original_entrypoint':'SFT/src/train.py'})

def train(run):
 import torch,runpy,time
 from transformers import TrainerCallback
 from research.av_train.teacher_rollouts import atomic_json,now
 import llamafactory.train.sft.workflow as workflow
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.cuda.reset_peak_memory_stats(local);start=time.monotonic()
 def sample(model):
  return {n:p.detach().reshape(-1)[:8].float().cpu().tolist() for n,p in model.named_parameters()}
 class Tracker(TrainerCallback):
  def on_train_begin(self,args,state,control,model=None,**kw):
   self.model=model;self.policy=policy(model);self.before=sample(model)
   atomic_json(run/f'policy-rank-{rank}.json',{'status':'started','rank':rank,'policy':self.policy,'at_utc':now()})
  def on_step_end(self,args,state,control,**kw):
   atomic_json(run/f'progress-rank-{rank}.json',{'global_step':state.global_step,'at_utc':now(),'max_allocated_gib':torch.cuda.max_memory_allocated(local)/2**30,'max_reserved_gib':torch.cuda.max_memory_reserved(local)/2**30})
  def on_train_end(self,args,state,control,model=None,**kw):
   final=sample(model);params=dict(model.named_parameters());changed=[n for n in final if self.before[n]!=final[n] and params[n].requires_grad]
   assert changed,'no sampled trainable parameter changed'
   assert all(self.before[n]==final[n] for n,p in params.items() if not p.requires_grad),'frozen AV sample changed'
   assert state.global_step==3 and policy(model)==self.policy
   for values in final.values():assert all(math.isfinite(v) for v in values)
   atomic_json(run/f'policy-rank-{rank}.json',{'status':'completed','at_utc':now(),'rank':rank,'policy':self.policy,'optimizer_updates':state.global_step,'changed_trainable_sample_tensors':len(changed),'frozen_parameter_samples_unchanged':True,'changed_tensor_samples':{n:final[n] for n in changed[:4]},'sample_values_per_tensor':8,'wall_seconds':time.monotonic()-start,'max_allocated_gib':torch.cuda.max_memory_allocated(local)/2**30,'max_reserved_gib':torch.cuda.max_memory_reserved(local)/2**30})
 original=workflow.CustomSeq2SeqTrainer
 class Instrumented(original):
  def __init__(self,*args,**kwargs):
   kwargs['callbacks']=list(kwargs.get('callbacks') or [])+[Tracker()];super().__init__(*args,**kwargs)
 workflow.CustomSeq2SeqTrainer=Instrumented
 sys.argv=[str(Path('/home/jovyan/OneSearch-VL/SFT/src/train.py')),str(run/'train.yaml')]
 runpy.run_path(sys.argv[0],run_name='__main__')

def validate_scheduler(engines,external,step=3):
 if external is not None:
  assert external['last_epoch']==step,'external HF scheduler step mismatch'
  return 'external_hf_scheduler'
 assert engines and all(isinstance(x.get('lr_scheduler'),dict) and x['lr_scheduler'].get('last_epoch')==step for x in engines),'DeepSpeed scheduler state absent'
 return 'deepspeed_scheduler'

def validate(run):
 import torch
 from safetensors import safe_open
 from research.av_train.teacher_rollouts import atomic_json,now
 output=run/'checkpoint';checkpoint=output/'checkpoint-3'
 state=json.loads((checkpoint/'trainer_state.json').read_text());assert state['global_step']==3
 logs=[x for x in state['log_history'] if 'loss' in x];assert len(logs)==3 and all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in logs)
 receipts=[json.loads((run/f'policy-rank-{i}.json').read_text()) for i in range(4)]
 assert all(x['status']=='completed' and x['optimizer_updates']==3 for x in receipts)
 index=json.loads((output/'model.safetensors.index.json').read_text())['weight_map'] if (output/'model.safetensors.index.json').exists() else None
 for name,values in receipts[0]['changed_tensor_samples'].items():
  f=output/index[name] if index else output/'model.safetensors'
  with safe_open(f,framework='pt',device='cpu') as tensors:actual=tensors.get_tensor(name).reshape(-1)[:8].float().tolist()
  assert actual==values,'saved model readback differs from trained values'
 optimizers=list(checkpoint.glob('**/*optim_states.pt'));model_states=list(checkpoint.glob('**/*model_states.pt'));assert len(optimizers)==4 and model_states
 optimizer_steps=[]
 def steps(v):
  result=[]
  if isinstance(v,dict):
   for k,x in v.items():
    if k=='step' and (isinstance(x,(int,float)) or isinstance(x,torch.Tensor) and x.numel()==1):result.append(int(x))
    elif isinstance(x,(dict,list,tuple)):result.extend(steps(x))
  elif isinstance(v,(list,tuple)):
   for x in v:result.extend(steps(x))
  return result
 for p in optimizers:
  v=torch.load(p,map_location='cpu',mmap=True,weights_only=False);actual=steps(v);assert actual and set(actual)=={3};optimizer_steps.append({'file':p.name,'steps':len(actual),'all_step_three':True});del v
 engines=[torch.load(p,map_location='cpu',mmap=True,weights_only=False) for p in model_states]
 assert all(x['global_steps']==3 for x in engines)
 external=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=False) if (checkpoint/'scheduler.pt').exists() else None
 validate_scheduler(engines,external)
 rngs=list(checkpoint.glob('rng_state*.pth'));assert len(rngs)==4
 for p in rngs:
  v=torch.load(p,map_location='cpu',weights_only=False);assert all(k in v for k in ['python','numpy','cpu','cuda'])
 result={'status':'passed','at_utc':now(),'optimizer_updates':3,'world_size':4,'real_full_thinker':True,'original_trainer':'SFT/src/train.py','frozen_av_policy_checked':True,'model_sample_readback_verified':True,'optimizer_scheduler_rng_metadata_verified':True,'optimizer_partitions':optimizer_steps,'per_rank':receipts,'losses':logs,'performance_gain_evaluated':False}
 atomic_json(run/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in {'per_rank','losses'}},ensure_ascii=False))

def main():
 mode,folder=sys.argv[1:];run=Path(folder).resolve()
 {'probe':probe,'preflight':runtime_preflight,'train':train,'validate':validate}[mode](run)
if __name__=='__main__':main()

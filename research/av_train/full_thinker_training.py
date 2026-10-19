"""Original-trainer full Thinker runs with explicit steps and checkpoint readback."""
from pathlib import Path
import json,os,sys,math
from .full_thinker_smoke import probe,runtime_preflight,policy,validate_scheduler

def validate_recipe(plan,cfg):
 steps=plan['optimizer_updates'];questions=plan['train_questions'];world=plan['world_size']
 if world!=4 or steps<1 or questions<1:raise ValueError('Explicit four-GPU recipe required')
 if cfg.get('finetuning_type')!='full' or any(k.startswith('lora_') for k in cfg):raise ValueError('Full Thinker required')
 if not cfg.get('freeze_vision_tower') or not cfg.get('freeze_multi_modal_projector') or cfg.get('freeze_language_model'):raise ValueError('AV frozen and language trainable required')
 if cfg.get('max_steps')!=steps:raise ValueError('Step count differs from run plan')
 batch=cfg['per_device_train_batch_size']*cfg['gradient_accumulation_steps']*world
 if batch!=4 or not cfg.get('use_audio_in_video') or cfg.get('cutoff_len')!=32768:raise ValueError('Native AV effective batch four required')
 if plan['purpose']=='formal_training':
  if questions not in {500,2000} or cfg.get('num_train_epochs')!=2. or cfg.get('learning_rate')!=1e-5:raise ValueError('Frozen scaling recipe required')
  if steps!=math.ceil(questions/batch)*2:raise ValueError('Formal recipe must complete two epochs')
 elif plan['purpose']!='resource_validation':raise ValueError('Known run purpose required')
 if cfg.get('resume_from_checkpoint'):raise ValueError('Each arm starts at the same pinned base')
 return steps

def train(run):
 import torch,runpy,time,yaml
 from transformers import TrainerCallback
 from .teacher_rollouts import atomic_json,now
 import llamafactory.train.sft.workflow as workflow
 plan=json.loads((run/'plan.json').read_text());expected=validate_recipe(plan,yaml.safe_load((run/'train.yaml').read_text()))
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.cuda.reset_peak_memory_stats(local);start=time.monotonic()
 def sample(model):return {n:p.detach().reshape(-1)[:8].float().cpu().tolist() for n,p in model.named_parameters()}
 class Tracker(TrainerCallback):
  def on_train_begin(self,args,state,control,model=None,**kw):
   self.model=model;self.policy=policy(model);self.before=sample(model)
   atomic_json(run/f'policy-rank-{rank}.json',{'status':'started','rank':rank,'policy':self.policy,'at_utc':now(),'expected_optimizer_updates':expected})
  def on_step_end(self,args,state,control,**kw):
   atomic_json(run/f'progress-rank-{rank}.json',{'global_step':state.global_step,'expected_optimizer_updates':expected,'at_utc':now(),'wall_seconds':time.monotonic()-start,'max_allocated_gib':torch.cuda.max_memory_allocated(local)/2**30,'max_reserved_gib':torch.cuda.max_memory_reserved(local)/2**30})
  def on_train_end(self,args,state,control,model=None,**kw):
   final=sample(model);params=dict(model.named_parameters());changed=[n for n in final if self.before[n]!=final[n] and params[n].requires_grad]
   assert changed and state.global_step==expected and policy(model)==self.policy
   assert all(self.before[n]==final[n] for n,p in params.items() if not p.requires_grad)
   assert all(math.isfinite(v) for values in final.values() for v in values)
   atomic_json(run/f'policy-rank-{rank}.json',{'status':'completed','at_utc':now(),'rank':rank,'policy':self.policy,'optimizer_updates':state.global_step,'changed_trainable_sample_tensors':len(changed),'frozen_parameter_samples_unchanged':True,'changed_tensor_samples':{n:final[n] for n in changed[:4]},'sample_values_per_tensor':8,'wall_seconds':time.monotonic()-start,'max_allocated_gib':torch.cuda.max_memory_allocated(local)/2**30,'max_reserved_gib':torch.cuda.max_memory_reserved(local)/2**30})
 original=workflow.CustomSeq2SeqTrainer
 class Instrumented(original):
  def __init__(self,*args,**kwargs):
   kwargs['callbacks']=list(kwargs.get('callbacks') or [])+[Tracker()];super().__init__(*args,**kwargs)
 workflow.CustomSeq2SeqTrainer=Instrumented
 sys.argv=[str(Path('/home/jovyan/OneSearch-VL/SFT/src/train.py')),str(run/'train.yaml')];runpy.run_path(sys.argv[0],run_name='__main__')

def optimizer_step_values(value):
 result=[]
 if isinstance(value,dict):
  for key,x in value.items():
   if key=='step' and (isinstance(x,(int,float)) or hasattr(x,'numel') and x.numel()==1):result.append(int(x))
   elif isinstance(x,(dict,list,tuple)):result.extend(optimizer_step_values(x))
 elif isinstance(value,(list,tuple)):
  for x in value:result.extend(optimizer_step_values(x))
 return result

def validate(run):
 import torch,yaml
 from safetensors import safe_open
 from .teacher_rollouts import atomic_json,now
 from .full_checkpoint import checkpoint_layout
 plan=json.loads((run/'plan.json').read_text());expected=validate_recipe(plan,yaml.safe_load((run/'train.yaml').read_text()));out=run/'checkpoint';checkpoint=out/f'checkpoint-{expected}'
 state=json.loads((checkpoint/'trainer_state.json').read_text());assert state['global_step']==expected
 losses=[x for x in state['log_history'] if 'loss' in x];assert losses and all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in losses)
 receipts=[json.loads((run/f'policy-rank-{i}.json').read_text()) for i in range(4)];assert all(x['status']=='completed' and x['optimizer_updates']==expected for x in receipts)
 _,config,mapping,names=checkpoint_layout(out)
 for name,values in receipts[0]['changed_tensor_samples'].items():
  with safe_open(out/mapping[name],framework='pt',device='cpu') as tensors:actual=tensors.get_tensor(name).reshape(-1)[:8].float().tolist()
  assert actual==values,'Saved model differs from trained model'
 optimizers=list(checkpoint.glob('**/*optim_states.pt'));model_states=list(checkpoint.glob('**/*model_states.pt'));assert len(optimizers)==4 and model_states
 partitions=[]
 for f in optimizers:
  data=torch.load(f,map_location='cpu',mmap=True,weights_only=False);steps=optimizer_step_values(data);assert steps and set(steps)=={expected};partitions.append({'file':f.name,'step_counters':len(steps),'expected_step':expected});del data
 engines=[torch.load(f,map_location='cpu',mmap=True,weights_only=False) for f in model_states];assert all(x['global_steps']==expected for x in engines)
 external=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=False) if (checkpoint/'scheduler.pt').exists() else None;scheduler=validate_scheduler(engines,external,expected)
 rngs=list(checkpoint.glob('rng_state*.pth'));assert len(rngs)==4
 for f in rngs:
  x=torch.load(f,map_location='cpu',weights_only=False);assert all(k in x for k in ['python','numpy','cpu','cuda'])
 result={'status':'passed','at_utc':now(),'purpose':plan['purpose'],'train_questions':plan['train_questions'],'variant':plan['variant'],'optimizer_updates':expected,'world_size':4,'real_full_thinker':True,'original_trainer':'SFT/src/train.py','frozen_av_policy_checked':True,'model_sample_readback_verified':True,'optimizer_scheduler_rng_metadata_verified':True,'scheduler_storage':scheduler,'optimizer_partitions':partitions,'per_rank':receipts,'losses':losses,'performance_gain_evaluated':False}
 atomic_json(run/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in {'per_rank','losses'}},ensure_ascii=False))

def main():
 mode,folder=sys.argv[1:];run=Path(folder).resolve();{'probe':probe,'preflight':runtime_preflight,'train':train,'validate':validate}[mode](run)
if __name__=='__main__':main()

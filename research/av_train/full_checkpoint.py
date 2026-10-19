"""Load full SFT weights into the same native Omni Thinker for evaluation."""
from pathlib import Path
import json,re,hashlib

BASE='Qwen/Qwen2.5-Omni-7B'
REVISION='ae9e1690543ffd5c0221dc27f79834d0294cba00'

def checkpoint_layout(folder):
 root=Path(folder).resolve();config=json.loads((root/'config.json').read_text())
 if config.get('model_type')!='qwen2_5_omni_thinker' or config.get('architectures')!=['Qwen2_5OmniThinkerForConditionalGeneration']:raise ValueError('full native Omni Thinker checkpoint required')
 index=json.loads((root/'model.safetensors.index.json').read_text());mapping=index['weight_map']
 if not mapping or any('lora_' in k or k.startswith('talker.') or k.startswith('thinker.') for k in mapping):raise ValueError('full Thinker state keys required')
 names=sorted(set(mapping.values()))
 for n in names:
  if not re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors',n) or not (root/n).is_file() or (root/n).resolve().parent!=root:raise ValueError('local bounded checkpoint shard required')
 return root,config,mapping,names

def attach_full_thinker(runner,folder):
 from transformers.modeling_utils import load_sharded_checkpoint
 root,config,mapping,names=checkpoint_layout(folder)
 identity=runner._artifact_identity
 if identity.get('repository_id')!=BASE or identity.get('revision')!=REVISION:raise ValueError('same pinned native Omni base required')
 if runner._model is None or not hasattr(runner._model,'thinker'):raise ValueError('load original native Omni runner first')
 thinker=runner._model.thinker
 if set(thinker.state_dict())!=set(mapping):raise ValueError('full checkpoint and native Thinker state keys differ')
 hashes={}
 for name in ['config.json','model.safetensors.index.json']+names:
  h=hashlib.sha256()
  with (root/name).open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  hashes[name]=h.hexdigest()
 result=load_sharded_checkpoint(thinker,str(root),strict=True,prefer_safe=True)
 assert not result.missing_keys and not result.unexpected_keys
 thinker.eval()
 for name in ['_generation_suppressed_ids','_generation_vocab_size']:
  if hasattr(runner,name):delattr(runner,name)
 receipt={'checkpoint_kind':'full_thinker','checkpoint':str(root),'base_revision':REVISION,'checkpoint_file_sha256':hashes,'strict_state_keys':True,'state_tensors':len(mapping),'inference_route':'original OmniRunner, original native AV and agent loop','inference_weight_dtype':str(next(thinker.parameters()).dtype)}
 runner._artifact_identity.update(full_thinker_checkpoint=receipt)
 return receipt

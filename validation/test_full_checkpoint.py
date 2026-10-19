from pathlib import Path
from types import SimpleNamespace
import tempfile,json,unittest
import torch
from safetensors.torch import save_file
from research.av_train.full_checkpoint import checkpoint_layout,attach_full_thinker,BASE,REVISION
class FullCheckpointTests(unittest.TestCase):
 def package(self,root,model):
  config={'model_type':'qwen2_5_omni_thinker','architectures':['Qwen2_5OmniThinkerForConditionalGeneration']}
  (root/'config.json').write_text(json.dumps(config));name='model-00001-of-00001.safetensors';state=model.state_dict();save_file(state,str(root/name))
  (root/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':sum(x.numel()*x.element_size() for x in state.values())},'weight_map':{k:name for k in state}}))
 def test_strict_full_state_load_changes_existing_thinker(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);trained=torch.nn.Linear(3,2);current=torch.nn.Linear(3,2);self.package(root,trained)
   runner=SimpleNamespace(_model=SimpleNamespace(thinker=current),_artifact_identity={'repository_id':BASE,'revision':REVISION},_generation_suppressed_ids=[1],_generation_vocab_size=2)
   result=attach_full_thinker(runner,root);self.assertEqual(result['checkpoint_kind'],'full_thinker');self.assertTrue(torch.equal(current.weight,trained.weight));self.assertFalse(hasattr(runner,'_generation_suppressed_ids'))
 def test_partial_language_state_rejected_before_loading(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.package(root,torch.nn.Linear(3,2));runner=SimpleNamespace(_model=SimpleNamespace(thinker=torch.nn.Sequential(torch.nn.Linear(3,2))),_artifact_identity={'repository_id':BASE,'revision':REVISION})
   with self.assertRaises(ValueError):attach_full_thinker(runner,root)
 def test_wrong_architecture_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.package(root,torch.nn.Linear(3,2));(root/'config.json').write_text(json.dumps({'model_type':'qwen2_5_omni'}))
   with self.assertRaises(ValueError):checkpoint_layout(root)
 def test_shard_path_escape_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.package(root,torch.nn.Linear(3,2));p=root/'model.safetensors.index.json';x=json.loads(p.read_text());x['weight_map']['weight']='../model-00001-of-00001.safetensors';p.write_text(json.dumps(x))
   with self.assertRaises(ValueError):checkpoint_layout(root)

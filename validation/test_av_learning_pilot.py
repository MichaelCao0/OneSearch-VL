import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from research.av_train.learning_pilot import group_holdout,verify_row_audit,attach_thinker_adapter
from research.av_train.native_sft import digest

class PilotIsolation(unittest.TestCase):
    def test_family_and_shared_media_are_kept_together(self):
        records=[{'task_id':'a','family_id':'f1','media_hashes':['m1']},
                 {'task_id':'b','family_id':'f1','media_hashes':['m2']},
                 {'task_id':'c','family_id':'f2','media_hashes':['m2']},
                 {'task_id':'d','family_id':'f3','media_hashes':['m3']},
                 {'task_id':'e','family_id':'f4','media_hashes':['m4']}]
        train,held=group_holdout(records,1,42)
        self.assertEqual(len(train)+len(held),5)
        held_ids={r['task_id'] for r in held}
        self.assertTrue(not held_ids.intersection({'a','b','c'}) or {'a','b','c'}<=held_ids)
        self.assertFalse({m for r in train for m in r['media_hashes']} & {m for r in held for m in r['media_hashes']})
        self.assertEqual((train,held),group_holdout(records,1,42))
    def test_duplicate_task_cannot_become_another_training_row(self):
        with self.assertRaises(ValueError):group_holdout([{'task_id':'a','family_id':'f1','media_hashes':[]},{'task_id':'a','family_id':'f2','media_hashes':[]}],1)
    def test_tampered_export_does_not_inherit_old_audit(self):
        row={'conversations':[{'from':'gpt','value':'actual'}]}
        lineage={'row_sha256':digest(row)};audit={'rows':[{'dataset':'trajectory','row_index':0,'total_tokens':5}]}
        self.assertEqual(verify_row_audit(row,lineage,audit,'trajectory',0)['total_tokens'],5)
        changed=copy.deepcopy(row);changed['conversations'][0]['value']='rewritten'
        with self.assertRaises(ValueError):verify_row_audit(changed,lineage,audit,'trajectory',0)
    def test_missing_row_audit_is_rejected(self):
        row={'conversations':[]}
        with self.assertRaises(ValueError):verify_row_audit(row,{'row_sha256':digest(row)},{'rows':[]},'trajectory',0)

class RealAdapter(unittest.TestCase):
    def test_adapter_updates_survive_attachment_to_thinker(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from peft import LoraConfig,get_peft_model
        config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
        torch.manual_seed(19);base=LlamaForCausalLM(config)
        initial=copy.deepcopy(base.state_dict())
        trained=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        ids=torch.tensor([[1,2,3,4]]);optimizer=torch.optim.AdamW(trained.parameters(),lr=.01)
        trained(input_ids=ids,labels=ids).loss.backward();optimizer.step();trained.eval()
        expected=trained(input_ids=ids).logits.detach()
        with tempfile.TemporaryDirectory() as folder:
            trained.save_pretrained(folder)
            original=LlamaForCausalLM(config);original.load_state_dict(initial);original.eval()
            runner=SimpleNamespace(_model=SimpleNamespace(thinker=original))
            receipt=attach_thinker_adapter(runner,folder)
            self.assertTrue(receipt['adapter_reload_exact'])
            actual=runner._model.thinker(input_ids=ids).logits.detach()
            self.assertTrue(torch.allclose(expected,actual,atol=1e-6,rtol=1e-5))

if __name__=='__main__':unittest.main()

import importlib.util
import unittest
import torch
class OptimizerSmokeTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('research.av_train.optimizer_smoke'),'Missing real optimizer smoke')
        return __import__('research.av_train.optimizer_smoke',fromlist=['*'])
    def test_actual_gradient_updates_weights_and_masks_context(self):
        m=self.api()
        class TinyModel(torch.nn.Module):
            def __init__(self):super().__init__();self.emb=torch.nn.Embedding(8,4);self.head=torch.nn.Linear(4,8)
            def forward(self,input_ids,labels,**kwargs):
                logits=self.head(self.emb(input_ids));loss=torch.nn.functional.cross_entropy(logits.reshape(-1,8),labels.reshape(-1),ignore_index=-100)
                return type('Output',(),{'loss':loss})()
        torch.manual_seed(3);model=TinyModel();before=model.head.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        result=m.optimizer_step(model,{'input_ids':torch.tensor([[1,2,3]]),'labels':torch.tensor([[-100,4,5]])},optimizer)
        self.assertGreater(result['gradient_norm'],0);self.assertEqual(result['supervised_tokens'],2)
        self.assertFalse(torch.equal(before,model.head.weight))
    def test_all_masked_targets_do_not_start_an_optimizer_update(self):
        m=self.api();model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        with self.assertRaisesRegex(ValueError,'supervised'):m.optimizer_step(model,{'labels':torch.tensor([[-100,-100]])},optimizer)
if __name__=='__main__':unittest.main()

import copy,unittest
from research.av_train.full_thinker_training import validate_recipe,optimizer_step_values
class FullThinkerTrainingTests(unittest.TestCase):
 def recipe(self,n=500):
  return {'optimizer_updates':n//2,'train_questions':n,'world_size':4,'purpose':'formal_training'},{'finetuning_type':'full','freeze_vision_tower':True,'freeze_multi_modal_projector':True,'freeze_language_model':False,'max_steps':n//2,'per_device_train_batch_size':1,'gradient_accumulation_steps':1,'use_audio_in_video':True,'cutoff_len':32768,'num_train_epochs':2.,'learning_rate':1e-5}
 def test_500_and_2000_complete_two_epochs(self):
  for n,expected in [(500,250),(2000,1000)]:
   p,c=self.recipe(n);self.assertEqual(validate_recipe(p,c),expected)
 def test_three_steps_cannot_be_formal(self):
  p,c=self.recipe();p['optimizer_updates']=c['max_steps']=3
  with self.assertRaises(ValueError):validate_recipe(p,c)
 def test_resource_three_steps_explicit(self):
  p,c=self.recipe();p.update(purpose='resource_validation',train_questions=12,optimizer_updates=3);c['max_steps']=3;self.assertEqual(validate_recipe(p,c),3)
 def test_accumulation_four_on_four_devices_rejected(self):
  p,c=self.recipe();c['gradient_accumulation_steps']=4
  with self.assertRaises(ValueError):validate_recipe(p,c)
 def test_changed_av_policy_and_lora_rejected(self):
  for name,value in [('freeze_vision_tower',False),('freeze_multi_modal_projector',False),('freeze_language_model',True),('lora_rank',8)]:
   p,c=self.recipe();c[name]=value
   with self.assertRaises(ValueError):validate_recipe(p,c)
 def test_resume_and_step_mismatch_rejected(self):
  for name,value in [('resume_from_checkpoint','other'),('max_steps',3)]:
   p,c=self.recipe();c[name]=value
   with self.assertRaises(ValueError):validate_recipe(p,c)
 def test_optimizer_metadata_nested_steps(self):
  self.assertEqual(optimizer_step_values({'optimizer_state_dict':{'state':{3:{'step':250,'exp_avg':[2.,4.]},4:{'step':250}}}}),[250,250])

import unittest
from research.av_train.full_thinker_smoke import allocation_mask,select_indices,policy,validate_scheduler
class FullThinkerSmokeTests(unittest.TestCase):
 def test_maps_only_four_allocated_devices_from_eight_visible(self):
  driver=[{'ordinal':i,'pci_bus_id':f'0000:{16+i:02x}:00.0'} for i in range(8)]
  ids=['GPU-'+str(i) for i in [1,3,5,7]];nvml={u:driver[i]['pci_bus_id'] for u,i in zip(ids,[1,3,5,7])}
  self.assertEqual(allocation_mask(ids,nvml,driver),'1,3,5,7')
 def test_duplicate_allocation_rejected(self):
  with self.assertRaises(ValueError):allocation_mask(['GPU-1']*4,{},[])
 def test_unknown_allocated_device_rejected(self):
  with self.assertRaises(ValueError):allocation_mask(['GPU-'+str(i) for i in range(4)],{},[])
 def test_extrema_selected_without_duplicate_rows(self):
  rows=[{'trajectory_tokens':i,'supervised_tokens':100-i,'media_hashes':[1]*(i%5)} for i in range(20)]
  ids=select_indices(rows);self.assertEqual(len(ids),len(set(ids)));self.assertEqual(len(ids),12);self.assertIn(19,ids);self.assertIn(0,ids);self.assertIn(4,ids)
 def test_policy_rejects_trainable_audio_even_with_matching_counts(self):
  class P:
   requires_grad=True
   def numel(self):return 1
  class M:
   def named_parameters(self):return [('audio_tower.p',P())]
  with self.assertRaises(AssertionError):policy(M(),1,0)

class SchedulerStorageTests(unittest.TestCase):
 def test_external_hf_scheduler_is_valid_with_none_in_deepspeed(self):
  self.assertEqual(validate_scheduler([{'lr_scheduler':None}],{'last_epoch':3}),'external_hf_scheduler')
 def test_external_scheduler_wrong_step_rejected(self):
  with self.assertRaises(AssertionError):validate_scheduler([{'lr_scheduler':None}],{'last_epoch':2})
 def test_absent_scheduler_never_counted_resumable(self):
  with self.assertRaises(AssertionError):validate_scheduler([{'lr_scheduler':None}],None)

import importlib.util
import unittest
class PairedSearchTests(unittest.TestCase):
 def test_budgeted_task_copy_does_not_mutate_training_spec(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_train.paired_searchqa'))
  from research.av_train.paired_searchqa import bounded_tasks
  original=[{'task_id':'a','question':'q','budget_recipe':{'max_tool_calls':6,'max_requested_av_seconds':8.}}]
  result=bounded_tasks(original,['a'],3)
  self.assertEqual(original[0]['budget_recipe']['max_tool_calls'],6)
  self.assertEqual(result[0]['budget_recipe'],{'max_tool_calls':3,'max_requested_av_seconds':4.})
 def test_missing_task_or_excessive_budget_rejected(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_train.paired_searchqa'))
  from research.av_train.paired_searchqa import bounded_tasks
  with self.assertRaises(ValueError):bounded_tasks([],['missing'],3)
  with self.assertRaises(ValueError):bounded_tasks([],[],100)

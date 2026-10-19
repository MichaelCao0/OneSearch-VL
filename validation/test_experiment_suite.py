import unittest
from research.av_train.experiment_suite import check_benchmark_members,construction_sample,experiment_matrix

def row(i):return {'task_id':str(i),'question':'question '+str(i),'recording':'r'+str(i),'family_id':'f'+str(i),'media_hashes':['m'+str(i)],'quality_view':'not_content_audited'}

class ExperimentSuiteTests(unittest.TestCase):
 def test_benchmark_deduplicates_normalized_question(self):
  a=row(0);b=dict(a,task_id='another',question=' question 0 ')
  with self.assertRaisesRegex(ValueError,'duplicate'):check_benchmark_members([a,b],[])
 def test_source_overlap_cannot_pass_due_to_new_task_id(self):
  a=row(0);b=dict(row(1),family_id=a['family_id'])
  with self.assertRaisesRegex(ValueError,'overlap'):check_benchmark_members([b],[a])
 def test_method_sample_is_eligible_source_distinct_and_order_independent(self):
  rows=[row(i) for i in range(8)];rows[0]['quality_view']='exclude_candidate';rows[2]['recording']=rows[1]['recording'];rows[4]['family_id']=rows[3]['family_id']
  chosen=construction_sample(rows,4,7)
  self.assertEqual(chosen,construction_sample(list(reversed(rows)),4,7))
  self.assertNotIn('0',[r['task_id'] for r in chosen]);self.assertEqual(len({r['recording'] for r in chosen}),4);self.assertEqual(len({r['family_id'] for r in chosen}),4)
 def test_arms_match_supervision_counts_and_world_size_budget(self):
  m=experiment_matrix();arms=m['axis_a']['training_arms'];self.assertEqual(len(arms),4)
  for n in [500,2000]:self.assertEqual({a['supervision'] for a in arms if a['questions']==n},{'trajectory','answer_only'})
  self.assertFalse(m['axis_a']['comparison_policy']['claim_compute_matched']);self.assertEqual(m['axis_b']['human_gold'],0);self.assertEqual(m['status']['new_gpu_jobs'],0)

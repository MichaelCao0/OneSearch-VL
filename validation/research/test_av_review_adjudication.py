import copy,hashlib,json,unittest
from test_av_annotation_import import review
from test_research_pipeline import task,sources
from research.av_bench import annotations

def digest(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def resolution(a,b):
 return {'schema_version':'av-researchbench.adjudication.v1','task_id':'t1','status':'submitted',
  'adjudicator_id':'third-human','submitted_at':'2026-10-10T14:00:00Z','review_a_sha256':digest(a),'review_b_sha256':digest(b),
  'facts':copy.deepcopy(a['facts']),'evidence_packages':copy.deepcopy(a['evidence_packages']),
  'modality_path':'joint_av','source_family_checked':True,'notes':'Replayed original fixture; synthetic test only.'}

class AdjudicationTests(unittest.TestCase):
 def setUp(self):
  self.a=review('first-human');self.b=review('second-human');self.b['facts'][0]['accepted_values']=['other']
 def call(self,r=None):return annotations.adjudicate_reviews(task(),sources(),self.a,self.b,r or resolution(self.a,self.b))
 def test_explicit_third_review_resolves_difference_and_preserves_original_hashes(self):
  self.assertTrue(hasattr(annotations,'adjudicate_reviews'),'Missing explicit adjudication path')
  r=self.call();self.assertEqual(r['status'],'adjudicated');self.assertEqual(r['reference']['facts'],self.a['facts'])
  self.assertEqual(r['review_a_sha256'],digest(self.a));self.assertEqual(r['review_b_sha256'],digest(self.b))
  self.assertEqual(r['reference']['reviewer_ids'],['first-human','second-human','third-human'])
 def test_modified_original_review_invalidates_resolution(self):
  self.assertTrue(hasattr(annotations,'adjudicate_reviews'),'Missing explicit adjudication path')
  r=resolution(self.a,self.b);self.b['notes']+=' modified'
  with self.assertRaisesRegex(ValueError,'hash'):self.call(r)
 def test_original_reviewer_cannot_be_third_adjudicator(self):
  self.assertTrue(hasattr(annotations,'adjudicate_reviews'),'Missing explicit adjudication path')
  r=resolution(self.a,self.b);r['adjudicator_id']=self.a['reviewer_id']
  with self.assertRaisesRegex(ValueError,'(?i)independent'):self.call(r)
 def test_pending_resolution_never_creates_reference(self):
  self.assertTrue(hasattr(annotations,'adjudicate_reviews'),'Missing explicit adjudication path')
  r=resolution(self.a,self.b);r['status']='pending'
  with self.assertRaises(ValueError):self.call(r)
 def test_bad_span_rejected_even_with_human_identity(self):
  self.assertTrue(hasattr(annotations,'adjudicate_reviews'),'Missing explicit adjudication path')
  r=resolution(self.a,self.b);r['evidence_packages'][0][0]['end_s']=100
  with self.assertRaises(ValueError):self.call(r)
 def test_different_task_family_requires_reauthoring(self):
  self.assertTrue(hasattr(annotations,'adjudicate_reviews'),'Missing explicit adjudication path')
  self.b['family']='event_localization'
  with self.assertRaisesRegex(ValueError,'family'):self.call()

import copy,unittest
from research.av_bench import qa_review as m
class QAReviewTests(unittest.TestCase):
 def task(self):return {'task_id':'q1','split':'dev','question':'Where?'}
 def reference(self):return {'task_id':'q1','status':'pending_human_review','benchmark_gold':False,'answer_fields':[{'field_id':'city','accepted_values':['Eugene']}],'support_quotes':[{'field_id':'city','source_id':'s1','start_char':0,'end_char':6,'quote':'Eugene'}],'scenario':'live_music','target_source_id':'s1','query_interval_on_parent':{'start_s':8.,'end_s':20.},'reviewer_ids':[]}
 def draft(self,decision='keep',support='supported'):
  return {'schema_version':'av-searchqa-authoring-review-draft.v2','reviewer_id':'human-test','created_at':'2026-10-12T12:00:00Z','benchmark_gold':False,'task_batch_sha256':'a'*64,'candidate_references_sha256':'b'*64,'reviews':[{'task_id':'q1','decision':decision,'audio_role':'helpful','needs_search':'required','answer_support':support,'note':''}]}
 def call(self,doc=None):return m.import_draft(doc or self.draft(),[self.task()],[self.reference()],task_hash='a'*64,reference_hash='b'*64)
 def test_keep_without_answer_verdict_does_not_create_reviewed_reference(self):
  out=self.call(self.draft(support='pending'));self.assertEqual(out['references'],[]);self.assertEqual(out['decisions'][0]['disposition'],'pending_answer_review')
 def test_reviewed_development_reference_preserves_values_quotes_and_human_identity(self):
  out=self.call();r=out['references'][0]
  self.assertEqual(r['status'],'human_reviewed');self.assertEqual(r['reviewer_ids'],['human-test']);self.assertEqual(r['answer_fields'],self.reference()['answer_fields']);self.assertEqual(r['support_quotes'],self.reference()['support_quotes'])
  self.assertEqual(r['audio_role'],'helpful');self.assertEqual(r['needs_search'],'required');self.assertFalse(r['formal_test_reference'])
 def test_wrong_task_or_answer_snapshot_and_duplicate_ids_are_rejected(self):
  for key,value in [('task_batch_sha256','c'*64),('candidate_references_sha256','d'*64)]:
   doc=self.draft();doc[key]=value
   with self.assertRaises(ValueError):self.call(doc)
  doc=self.draft();doc['reviews'].append(copy.deepcopy(doc['reviews'][0]))
  with self.assertRaises(ValueError):self.call(doc)
  doc=self.draft();doc['reviews'][0]['task_id']='foreign'
  with self.assertRaises(ValueError):self.call(doc)
 def test_revised_rejected_and_ambiguous_questions_do_not_silently_enter_reference_set(self):
  for decision in ['revise','reject']:
   self.assertEqual(self.call(self.draft(decision=decision))['references'],[])
  doc=self.draft();doc['reviews'][0]['needs_search']='ambiguous';self.assertEqual(self.call(doc)['references'],[])
 def test_pending_audio_or_unsupported_answers_remain_unreviewed(self):
  for support in ['revise','unsupported']:
   self.assertEqual(self.call(self.draft(support=support))['references'],[])
  doc=self.draft();doc['reviews'][0]['audio_role']='pending';self.assertEqual(self.call(doc)['references'],[])
 def test_unknown_fields_or_missing_human_identity_cannot_be_inferred(self):
  doc=self.draft();doc['reviews'][0]['answer_support']='yes'
  with self.assertRaises(ValueError):self.call(doc)
  doc=self.draft();doc['reviewer_id']=''
  with self.assertRaises(ValueError):self.call(doc)
if __name__=='__main__':unittest.main()

class QuoteIdentityTests(QAReviewTests):
 def test_reference_quotes_require_exact_frozen_publisher_text(self):
  docs={'s1':{'text':'Eugene'}};self.assertEqual(m.validate_reference_quotes([self.reference()],docs),1)
  with self.assertRaises(ValueError):m.validate_reference_quotes([self.reference()],{'s1':{'text':'Other!'}})
  with self.assertRaises(ValueError):m.validate_reference_quotes([self.reference()],{})
 def test_each_requested_answer_field_requires_support_and_valid_char_bounds(self):
  r=self.reference();r['answer_fields'].append({'field_id':'country','accepted_values':['USA']})
  with self.assertRaises(ValueError):m.validate_reference_quotes([r],{'s1':{'text':'Eugene'}})
  r=self.reference();r['support_quotes'][0]['start_char']=True
  with self.assertRaises(ValueError):m.validate_reference_quotes([r],{'s1':{'text':'Eugene'}})

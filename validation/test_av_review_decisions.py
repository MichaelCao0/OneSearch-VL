import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from research.av_data.review_decisions import import_reviews


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class ReviewImportTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.package=self.root/'package';self.package.mkdir();(self.package/'media').mkdir()
  (self.package/'media/input.mp4').write_bytes(b'bounded-native-media-fixture')
  task={'task_id':'q1','question':'片段中的作品由谁创作？'};ref={'task_id':'q1','answer_fields':[{'field_id':'composer','accepted_values':['Composer']}],'support_quotes':[{'field_id':'composer','quote':'Composer wrote the work.'}]}
  for name,value in [('tasks-public.jsonl',task),('answer-candidates.private.jsonl',ref)]: (self.package/name).write_text(json.dumps(value)+'\n')
  self.payload={'summary':{'split':'dev','tasks_sha256':digest(self.package/'tasks-public.jsonl'),'candidate_references_sha256':digest(self.package/'answer-candidates.private.jsonl')},'cases':[{'task':task,'reference_candidate':ref,'media':'media/input.mp4'}]}
  receipt={'task_id':'q1','source_id':'recording1','input_sha256':digest(self.package/'media/input.mp4'),'parent_sha256':'a'*64,'query_interval_on_parent':{'start_s':12.0,'end_s':42.0}}
  (self.package/'payload.json').write_text(json.dumps(self.payload));(self.package/'manifest.json').write_text(json.dumps({'cases':1,'receipts':[receipt],'task_batch_sha256':self.payload['summary']['tasks_sha256'],'candidate_references_sha256':self.payload['summary']['candidate_references_sha256']}))
 def draft(self,reviewer='r1',**changes):
  row={'task_id':'q1','decision':'keep','audio_role':'helpful','needs_search':'required','answer_support':'supported','note':'核对原声与来源'};row.update(changes)
  data={'schema_version':'av-searchqa-authoring-review-draft.v2','reviewer_id':reviewer,'created_at':'2026-10-15T10:00:00Z','benchmark_gold':False,'task_batch_sha256':self.payload['summary']['tasks_sha256'],'candidate_references_sha256':self.payload['summary']['candidate_references_sha256'],'reviews':[row]}
  path=self.root/(reviewer+'.json');path.write_text(json.dumps(data));return path
 def test_single_reviewer_is_not_consensus_or_gold(self):
  result=import_reviews(self.package,[self.draft()],self.root/'out')
  self.assertEqual(result['statuses'],{'awaiting_second_review':1});self.assertEqual(result['human_gold'],0)
 def test_two_agreeing_reviewers_preserve_media_and_interval_binding(self):
  result=import_reviews(self.package,[self.draft('r1'),self.draft('r2')],self.root/'out')
  self.assertEqual(result['statuses'],{'consensus_keep_candidate':1});self.assertEqual(result['human_gold'],0)
  row=json.loads((self.root/'out/review-records.private.jsonl').read_text())
  self.assertEqual(row['media_binding']['query_interval_on_parent'],{'start_s':12.0,'end_s':42.0})
  self.assertEqual(set(row['reviewer_ids']),{'r1','r2'})
 def test_audio_disagreement_requires_adjudication(self):
  result=import_reviews(self.package,[self.draft('r1'),self.draft('r2',audio_role='redundant')],self.root/'out')
  self.assertEqual(result['statuses'],{'needs_adjudication':1})
 def test_stale_reference_hash_rejected_before_output_created(self):
  p=self.draft();x=json.loads(p.read_text());x['candidate_references_sha256']='b'*64;p.write_text(json.dumps(x))
  with self.assertRaises(ValueError):import_reviews(self.package,[p],self.root/'out')
  self.assertFalse((self.root/'out').exists())
 def test_same_reviewer_cannot_count_twice(self):
  p=self.draft();q=self.root/'copy.json';q.write_bytes(p.read_bytes())
  with self.assertRaises(ValueError):import_reviews(self.package,[p,q],self.root/'out')
 def test_unknown_task_is_rejected(self):
  with self.assertRaises(ValueError):import_reviews(self.package,[self.draft(task_id='other')],self.root/'out')
 def test_mutated_media_cannot_reuse_annotation(self):
  p=self.draft();(self.package/'media/input.mp4').write_bytes(b'changed')
  with self.assertRaises(ValueError):import_reviews(self.package,[p],self.root/'out')
 def test_perception_only_is_control(self):
  result=import_reviews(self.package,[self.draft('r1',needs_search='perception_only'),self.draft('r2',needs_search='perception_only')],self.root/'out')
  self.assertEqual(result['statuses'],{'perception_control_candidate':1})
 def test_empty_import_reports_pending_without_invented_reviews(self):
  result=import_reviews(self.package,[],self.root/'out')
  self.assertEqual(result['statuses'],{'awaiting_review':1});self.assertEqual(result['reviewers'],0)
 def test_duplicate_task_in_draft_rejected(self):
  p=self.draft();x=json.loads(p.read_text());x['reviews'].append(copy.deepcopy(x['reviews'][0]));p.write_text(json.dumps(x))
  with self.assertRaises(ValueError):import_reviews(self.package,[p],self.root/'out')
 def test_browser_gold_flag_does_not_authorize_gold(self):
  p=self.draft();x=json.loads(p.read_text());x['benchmark_gold']=True;p.write_text(json.dumps(x))
  with self.assertRaises(ValueError):import_reviews(self.package,[p],self.root/'out')

if __name__=='__main__':unittest.main()

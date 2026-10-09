import copy
import json
from pathlib import Path
import tempfile
import unittest
from research.av_data import review_forms as r

SRC='src_'+'1'*24
TASK={'schema_version':'1.0','created_by':'test','revision':'fixture-v1','task_id':'../../outside',
'question':'原始片段与来源描述能支持哪些判断？','target_questions':[{'claim_id':str(i),'question':'待核验问题'} for i in range(3)],
'initial_sources':[SRC],'corpus_id':'c1','budget_recipe':{'cap':1},'split':'exploration'}
SOURCE={'source_id':SRC,'asset_sha256':'a'*64,'asset_path':'/remote/raw/a.webm','media_type':'video','duration_s':3.0,
'origin_url':'https://example.org/source','timestamp_origin':{'presentation_start_s':0},'public_metadata':{'title':'真实标题'},
'stream_specs':[{'codec_type':'video'},{'codec_type':'audio'}]}

class ReviewFormsTests(unittest.TestCase):
 def test_all_labels_and_human_identity_are_blank(self):
  form=r.blank_review(TASK,{SRC:SOURCE},'A')
  self.assertIsNone(form['reviewer_identity']);self.assertEqual(form['annotation_status'],'blank_unreviewed')
  for claim in form['target_annotations']:
   self.assertIsNone(claim['expected_decision']);self.assertIsNone(claim['status']);self.assertEqual(claim['valid_evidence_sets'],[])
   self.assertTrue(all(t['evidence_sufficient'] is None for t in claim['modality_trials']))
 def test_a_b_forms_are_independent_objects(self):
  a=r.blank_review(TASK,{SRC:SOURCE},'A');b=r.blank_review(TASK,{SRC:SOURCE},'B')
  a['target_annotations'][0]['status']='supported'
  self.assertIsNone(b['target_annotations'][0]['status']);self.assertEqual(TASK['target_questions'][0]['question'],'待核验问题')
 def test_foreign_initial_source_fails(self):
  task=copy.deepcopy(TASK);task['initial_sources']=['src_'+'2'*24]
  with self.assertRaises(ValueError):r.blank_review(task,{SRC:SOURCE},'A')
 def test_public_answer_field_fails(self):
  task=copy.deepcopy(TASK);task['answer']='hidden'
  with self.assertRaises(ValueError):r.blank_review(task,{SRC:SOURCE},'A')
 def test_filename_is_opaque_not_task_path(self):
  name=r.task_stem('../../outside')
  self.assertNotIn('/',name);self.assertNotIn('..',name)
 def test_generate_refuses_overwrite_and_preserves_hashes(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);tasks=root/'tasks.jsonl';sources=root/'sources.jsonl';out=root/'forms'
   tasks.write_text(json.dumps(TASK)+'\n');sources.write_text(json.dumps(SOURCE)+'\n')
   receipt=r.generate(tasks,sources,out)
   self.assertEqual(receipt['tasks'],1);self.assertEqual(len(list((out/'A').glob('*.json'))),1)
   with self.assertRaises(FileExistsError):r.generate(tasks,sources,out)

if __name__=='__main__':unittest.main()

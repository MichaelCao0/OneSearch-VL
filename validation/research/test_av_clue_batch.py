import importlib.util,unittest
SID='src_'+'a'*24;PAGE='src_'+'d'*24
class ClueBatchTests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_data.clue_batch'))
  return __import__('research.av_data.clue_batch',fromlist=['*'])
 def fixture(self):
  sources={SID:{'source_id':SID,'media_type':'audio_video','duration_s':8.,'origin_url':'https://example.org/source'}}
  docs={PAGE:{'source_id':PAGE,'origin_url':'https://example.org/source','text':'The producer is Otto Recke. The machine was made in 1903.'}}
  cases=[{'task_id':'qa-dev-001','question':'Who produced this machine and when?','target_source_id':SID,'start_s':1.,'end_s':3.,'scenario':'machinery','fields':[{'field_id':'maker','accepted_values':['Otto Recke']},{'field_id':'year','accepted_values':['1903']}]}]
  return cases,docs,sources
 def test_candidate_answer_fields_have_exact_actual_publisher_quotes(self):
  m=self.api();cases,docs,sources=self.fixture();refs=m.validate_authoring_cases(cases,docs,sources)
  self.assertEqual(refs[0]['status'],'pending_human_review');self.assertFalse(refs[0]['benchmark_gold'])
  self.assertEqual(len(refs[0]['support_quotes']),2)
  for q in refs[0]['support_quotes']:self.assertEqual(docs[q['source_id']]['text'][q['start_char']:q['end_char']],q['quote'])
 def test_unsupported_answer_is_rejected_before_materialization(self):
  m=self.api();cases,docs,sources=self.fixture();cases[0]['fields'][0]['accepted_values']=['Imaginary Manufacturer']
  with self.assertRaisesRegex(ValueError,'publisher'):m.validate_authoring_cases(cases,docs,sources)
 def test_text_from_unrelated_page_cannot_support_clue_target(self):
  m=self.api();cases,docs,sources=self.fixture();docs[PAGE]['origin_url']='https://example.org/different'
  with self.assertRaises(ValueError):m.validate_authoring_cases(cases,docs,sources)
 def test_duplicate_task_or_bad_crop_rejected(self):
  m=self.api();cases,docs,sources=self.fixture()
  with self.assertRaises(ValueError):m.validate_authoring_cases(cases+cases,docs,sources)
  cases[0]['end_s']=20
  with self.assertRaises(ValueError):m.validate_authoring_cases(cases,docs,sources)
 def test_private_answer_keys_never_enter_public_task(self):
  m=self.api();cases,_,_=self.fixture();t=m.actor_task(cases[0],'src_'+'b'*24,'corpus-dev','dev')
  self.assertEqual(t['initial_sources'],['src_'+'b'*24])
  text=__import__('json').dumps(t);self.assertNotIn('Otto Recke',text);self.assertNotIn(SID,text);self.assertNotIn('accepted_values',text)
if __name__=='__main__':unittest.main()

class ClueTrainScopeTests(unittest.TestCase):
 def test_train_pool_uses_same_anonymous_query_schema(self):
  import tempfile,json,hashlib
  from pathlib import Path
  from research.av_search.corpus import build_corpus
  from research.av_data.clue_batch import build_clue_batch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pool=root/'pool';pool.mkdir();asset=root/'asset';asset.write_bytes(b'fixture-source')
   source={'source_id':SID,'asset_path':str(asset),'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'origin_url':'https://example.org/target','media_type':'audio_video','duration_s':3.,'public_metadata':{'title':'tractor'}}
   (pool/'sources.private.jsonl').write_text(json.dumps(source)+'\n')
   build_corpus([dict(source,text='tractor',text_kind='public_metadata')],pool/'authoring-corpus','train-pool','train')
   comparison=root/'reserved.jsonl';comparison.write_text(json.dumps(source)+'\n')
   with self.assertRaisesRegex(ValueError,'overlap'):build_clue_batch(pool,comparison,[],root/'out')
   self.assertFalse((root/'out').exists())

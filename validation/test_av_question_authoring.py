import copy,json,unittest,tempfile
from pathlib import Path
from research.av_train import question_authoring as q
class AuthoringTests(unittest.TestCase):
 def setUp(self):
  self.item={'question':'结合画面和旋律，这段演奏的作品与作曲家是谁？','visual_cue':'演奏者弹奏钢琴','audio_cue':'钢琴旋律','media_matches_description':True,'external_information_required':True,'fields':[{'field_id':'composer','accepted_values':['Debussy']},{'field_id':'work','accepted_values':['Clair De Lune']}]}
  self.text='Recorded piano performance: Clair De Lune by Claude Debussy.'
 def test_real_publisher_quotes_required(self):
  value=q.validate_questions({'questions':[self.item]},self.text)
  self.assertEqual(len(value),1)
  bad=copy.deepcopy(self.item);bad['fields'][0]['accepted_values']=['Bach']
  with self.assertRaises(ValueError):q.validate_questions({'questions':[bad]},self.text)
 def test_reworded_same_answers_do_not_inflate_count(self):
  b=copy.deepcopy(self.item);b['question']='请识别这首曲子的作者和名称。';b['fields'].reverse()
  with self.assertRaises(ValueError):q.validate_questions({'questions':[self.item,b]},self.text)
 def test_private_answer_leak_in_question_rejected(self):
  bad=copy.deepcopy(self.item);bad['question']='这段 Debussy 的作品是什么？'
  with self.assertRaises(ValueError):q.validate_questions({'questions':[bad]},self.text)
 def test_two_character_answer_leak_and_swapped_alias_duplicate_rejected(self):
  bad=copy.deepcopy(self.item);bad['question']='肖邦演奏的曲目是什么？';bad['fields']=[{'field_id':'composer','accepted_values':['肖邦']}]
  with self.assertRaises(ValueError):q.validate_questions({'questions':[bad]},'作曲家：肖邦')
  a=copy.deepcopy(self.item);a['fields']=[{'field_id':'composer','accepted_values':['Debussy','Claude Debussy']}]
  b=copy.deepcopy(a);b['question']='这位作品作者是谁？';b['fields'][0]['accepted_values'].reverse()
  with self.assertRaises(ValueError):q.validate_questions({'questions':[a,b]},self.text)
 def test_raw_authored_result_binding_rejects_changed_questions_or_media(self):
  with tempfile.TemporaryDirectory() as folder:
   media=Path(folder)/'fixture.mp4';media.write_bytes(b'integrity-test-media')
   text=json.dumps({'questions':[self.item]});sha=q.file_sha(media)
   identity={'provider':'gemini_native','model_id':'fixture','model_version':'fixture','endpoint_sha256':'a'*64}
   receipt={**identity,'schema':'onesearch-native-generation.v1','status':'complete','input_mode':'native_audio_video',
    'generated_tokens':20,'request_sha256':'b'*64,'output_sha256':q.hashlib.sha256(text.encode()).hexdigest(),
    'media':[{'kind':'video','sha256':sha,'bytes':media.stat().st_size}]}
   result={'source_id':'wc_a','status':'completed','questions':[self.item],'generation_identity':identity,
    'response':{'raw_response':{'generation_receipt':receipt},'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}]}}
   row={'source_id':'wc_a','teacher_media_path':str(media),'teacher_media_sha256':sha,'publisher_text':self.text}
   q.verify_authored_result(result,row)
   result['questions'][0]['question']='篡改了问题'
   with self.assertRaises(ValueError):q.verify_authored_result(result,row)
   result['questions']=json.loads(text)['questions'];media.write_bytes(b'changed-media')
   with self.assertRaises(ValueError):q.verify_authored_result(result,row)
 def test_crop_validation_tolerates_float_roundoff_but_not_longer_clip(self):
  from research.av_data.clue_batch import validate_authoring_cases
  source={'source_id':'a','media_type':'audio_video','duration_s':85.738685,'origin_url':'https://example.com/a'}
  doc={'source_id':'b','origin_url':source['origin_url'],'text':'Composer Debussy'}
  case={'task_id':'test','target_source_id':'a','question':'作曲家是谁？','scenario':'music','start_s':17.147737000000003,'end_s':47.147737000000006,'fields':[{'field_id':'composer','accepted_values':['Debussy']}]}
  self.assertEqual(len(validate_authoring_cases([case],{'b':doc},{'a':source})),1)
  case['end_s']+=.001
  with self.assertRaises(ValueError):validate_authoring_cases([case],{'b':doc},{'a':source})
 def test_single_chinese_answer_is_rejected_in_question(self):
  bad=copy.deepcopy(self.item);bad['question']='这只虎属于什么动物？';bad['fields']=[{'field_id':'animal','accepted_values':['虎']}]
  with self.assertRaises(ValueError):q.validate_questions({'questions':[bad]},'虎')
 def test_silent_or_mismatched_authoring_is_not_accepted(self):
  for key,value in [('audio_cue',''),('media_matches_description',False),('external_information_required',False)]:
   bad=copy.deepcopy(self.item);bad[key]=value
   with self.assertRaises(ValueError):q.validate_questions({'questions':[bad]},self.text)
 def test_parser_rejects_extra_text_multiple_objects_and_duplicate_keys(self):
  self.assertEqual(q.parse_json_output('```json\n{"questions": []}\n```'),{'questions':[]})
  self.assertEqual(q.parse_json_output('<response>{"questions": []}</response>'),{'questions':[]})
  for text in ['prefix {"questions": []}','{"questions": [],"questions": []}','{"questions": []}\n{"questions": []}']:
   with self.assertRaises(ValueError):q.parse_json_output(text)
 def test_failed_acquisition_rows_without_hash_are_skipped(self):
  selected,audit=q.select_sources([{'source_id':'failed','status':'failed_download_or_probe'}, {'status':'rejected_declared_size'}],[],[],[],10)
  self.assertEqual(selected,[])
 def test_reserved_asset_family_propagates_without_exact_new_source_overlap(self):
  seeds=[{'source_id':'new','sha256':'newhash','origin_url':'https://example.com/new','status':'candidate_unreviewed_av','description_page_html_path':__file__}]
  fam=[{'candidate_family_id':'f1','source_ids':['heldout','new']}]
  reserved=[{'source_id':'heldout','asset_sha256':'devhash','origin_url':'https://example.com/dev'}]
  selected,audit=q.select_sources(seeds,fam,reserved,[],100)
  self.assertEqual(selected,[]);self.assertEqual(audit['excluded_reserved_family_sources'],['new'])
 def test_family_quarantine_propagates_to_related_recording(self):
  seeds=[{'source_id':s,'sha256':s,'origin_url':'https://example.com/'+s,'status':'candidate_unreviewed_av','description_page_html_path':__file__} for s in ['a','b','c']]
  fam=[{'candidate_family_id':'f1','source_ids':['a','b']},{'candidate_family_id':'f2','source_ids':['c']}]
  selected,audit=q.select_sources(seeds,fam,[{'asset_sha256':'a','origin_url':'https://example.com/a'}],[],100)
  self.assertEqual([s['source_id'] for s in selected],['c']);self.assertEqual(set(audit['excluded_reserved_family_sources']),{'a','b'})
if __name__=='__main__':unittest.main()

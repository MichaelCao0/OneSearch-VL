import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from research.av_search.corpus import build_corpus
SID='src_'+'a'*24;CLUE='src_'+'b'*24;CLIP='src_'+'c'*24
class ClueSearchQATests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_train.clue_searchqa'),'Missing independent media clue runtime')
  return __import__('research.av_train.clue_searchqa',fromlist=['*'])
 def runtime(self,root,observer=None):
  m=self.api();asset=root/'asset.bin';asset.write_bytes(b'target');clip=root/'clip.bin';clip.write_bytes(b'independent clip')
  target={'source_id':SID,'asset_path':str(asset),'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'origin_url':'https://example.org/target','media_type':'audio_video','duration_s':9.,'public_metadata':{'title':'A vintage tractor'}}
  build_corpus([dict(target,text='A vintage tractor from France',text_kind='public_metadata')],root/'corpus','clue-fixture','dev')
  source={'source_id':CLIP,'asset_path':str(clip),'asset_sha256':hashlib.sha256(clip.read_bytes()).hexdigest(),'origin_url':'https://example.invalid/query','media_type':'audio_video','duration_s':2.,'public_metadata':{}}
  scope={'source_ids':[SID],'corpus_id':'clue-fixture','split':'dev'}
  binding={'clue_id':CLUE,'source':source}
  return m.ClueSearchQARuntime(root/'corpus',scope,{SID:target},[binding],root/'tools',{'max_tool_calls':6,'max_requested_av_seconds':4.},observer),source
 def test_input_clue_has_no_publisher_text_lookup(self):
  with tempfile.TemporaryDirectory() as d:
   r,_=self.runtime(Path(d));o=r.execute({'tool':'read_text','arguments':{'source_id':CLUE,'start_char':0,'end_char':12}})
   self.assertEqual(o['status'],'tool_error');self.assertNotIn('quote',o);self.assertEqual(r.cost['tool_calls'],1)
 def test_target_reads_require_actual_search_discovery(self):
  with tempfile.TemporaryDirectory() as d:
   r,_=self.runtime(Path(d));o=r.execute({'tool':'read_text','arguments':{'source_id':SID,'start_char':0,'end_char':5}})
   self.assertEqual(o['status'],'tool_error');self.assertNotIn('valid_text_range',o)
   result=r.execute({'tool':'search_sources','arguments':{'query':'tractor','top_k':3}});self.assertEqual(result['results'][0]['source_id'],SID)
   self.assertEqual(r.execute({'tool':'read_text','arguments':{'source_id':SID,'start_char':0,'end_char':5}})['quote']['text'],'A vin')
 def test_initial_tool_definitions_do_not_advertise_target_id(self):
  with tempfile.TemporaryDirectory() as d:
   r,_=self.runtime(Path(d));text=json.dumps(r.tool_definitions())
   self.assertIn(CLUE,text);self.assertNotIn(SID,text);self.assertNotIn(CLIP,text);self.assertNotIn('vintage',text)
 def test_inspection_uses_actual_clipped_asset_and_sanitizes_identity(self):
  calls=[]
  def fixture(source,start,end,out):
   calls.append((source,start,end));return {'status':'fixture_observer','source_id':source['source_id'],'asset_sha256':source['asset_sha256'],'origin_url':'https://example.org/target','mapping':{'parent_source_id':SID},'observation':{'description':'rotating wheel','source_id':source['source_id']},'semantic_verified':False,'actual_cost':{'fixture_calls':1}}
  with tempfile.TemporaryDirectory() as d:
   r,source=self.runtime(Path(d),fixture);o=r.execute({'tool':'inspect_av','arguments':{'source_id':CLUE,'start_s':0,'end_s':2}})
   self.assertEqual(calls[0][0],source);self.assertEqual(o['source_id'],CLUE);self.assertEqual(o['observation']['source_id'],CLUE)
   self.assertEqual(o['observation']['description'],'rotating wheel');self.assertEqual(o['status'],'fixture_observer')
   for value in [SID,CLIP,source['asset_sha256'],'https://example.org/target','asset_path']:self.assertNotIn(value,json.dumps(o))
 def test_unseen_target_inspection_does_not_prepare_media(self):
  calls=[]
  def fixture(*args):calls.append(args);return {}
  with tempfile.TemporaryDirectory() as d:
   r,_=self.runtime(Path(d),fixture);o=r.execute({'tool':'inspect_av','arguments':{'source_id':SID,'start_s':0,'end_s':1}})
   self.assertEqual(o['status'],'tool_error');self.assertEqual(calls,[]);self.assertEqual(r.cost['requested_av_seconds'],0.)
 def test_clue_alias_collision_is_rejected(self):
  m=self.api()
  with tempfile.TemporaryDirectory() as d:
   r,source=self.runtime(Path(d))
   with self.assertRaises(ValueError):m.ClueSearchQARuntime(r.index_dir,r.scope,r.sources,[{'clue_id':SID,'source':source}],Path(d)/'other',r.budget)
 def test_binding_metadata_is_not_silently_admitted(self):
  m=self.api()
  with tempfile.TemporaryDirectory() as d:
   r,source=self.runtime(Path(d));source['public_metadata']={'title':'Answer-revealing title'}
   with self.assertRaises(ValueError):m.ClueSearchQARuntime(r.index_dir,r.scope,{SID:r.sources[SID]},[{'clue_id':CLUE,'source':source}],Path(d)/'other',r.budget)
if __name__=='__main__':unittest.main()

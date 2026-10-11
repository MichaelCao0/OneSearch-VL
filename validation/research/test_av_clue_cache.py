import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research.av_search.corpus import build_corpus
from research.av_data.clue_batch import build_clue_batch

class SharedClueCacheTests(unittest.TestCase):
 def test_shared_recording_crop_is_prepared_once_without_alias_or_answer_leakage(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pool=root/'pool';pool.mkdir();asset=root/'source';asset.write_bytes(b'original')
   h=hashlib.sha256(asset.read_bytes()).hexdigest();sid='src_'+h[:24]
   source={'source_id':sid,'asset_path':str(asset),'asset_sha256':h,'origin_url':'https://example.org/train',
     'media_type':'audio_video','rights_basis':'synthetic_fixture','redistribution_scope':'full_asset','duration_s':20.,'public_metadata':{'title':'Tractor','description':'Rochet-Schneider 1919'}}
   (pool/'sources.private.jsonl').write_text(json.dumps(source)+'\n')
   build_corpus([dict({k:v for k,v in source.items() if k not in {'rights_basis','redistribution_scope'}},text='Tractor Rochet-Schneider 1919',text_kind='public_metadata')],pool/'authoring-corpus','train-pool','train')
   other=root/'dev.jsonl';other.write_text(json.dumps(dict(source,asset_sha256='0'*64,origin_url='https://example.org/dev'))+'\n')
   clip=root/'clip';clip.write_bytes(b'clipped');ch=hashlib.sha256(clip.read_bytes()).hexdigest()
   query=dict(source,source_id='src_'+ch[:24],asset_sha256=ch,asset_path=str(clip),public_metadata={},duration_s=12.)
   derived={'output_path':str(clip),'original_interval':{'start_s':0.,'end_s':12.},'mapping':{'fixture':True}}
   cases=[{'task_id':'q-'+str(i),'question':q,'target_source_id':sid,'start_s':0.,'end_s':12.,'scenario':'machinery',
     'fields':[{'field_id':'field','accepted_values':[answer]}]} for i,(q,answer) in enumerate([('Which brand?','Rochet-Schneider'),('Which year?','1919')])]
   with patch('research.av_data.clue_batch.clip_media',return_value=derived) as prepare,patch('research.av_data.clue_batch.ingest_media',return_value=query) as ingest:
    result=build_clue_batch(pool,other,cases,root/'out')
   self.assertEqual(prepare.call_count,1);self.assertEqual(ingest.call_count,1)
   self.assertEqual(result['query_clips'],2);self.assertEqual(result['unique_query_assets'],1)
   bindings=json.loads((root/'out/clue-bindings.private.json').read_text());self.assertNotEqual(bindings[0]['clue_id'],bindings[1]['clue_id'])
   self.assertEqual(bindings[0]['source'],bindings[1]['source'])
   public=(root/'out/tasks-public.jsonl').read_text();self.assertNotIn('Rochet-Schneider',public);self.assertNotIn(sid,public)
if __name__=='__main__':unittest.main()

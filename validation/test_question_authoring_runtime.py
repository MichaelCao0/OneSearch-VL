"""Local native HTTP fixtures prove AV transfer, immutable attempts and export."""
import json,os,subprocess,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from teacher_http_fixtures import fixture_server,reply
from research.av_train import question_authoring as q
class AuthoringRuntimeTests(unittest.TestCase):
 def exercise(self,success):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);media=root/'original.mp4'
   subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=blue:s=96x64:r=4:d=1','-f','lavfi','-i','sine=frequency=300:duration=1','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(media)],check=True,capture_output=True)
   page=root/'page.html';page.write_text('<p>Recorded piano: Clair De Lune by Claude Debussy.</p>')
   seed={'source_id':'wc_fixture','page_id':1,'status':'candidate_unreviewed_av','sha256':q.file_sha(media),'asset_path':str(media),
    'origin_url':'https://example.com/fixture','public_metadata':{'title':'Recorded piano','description':'Clair De Lune by Claude Debussy.','license':'CC0'},
    'description_page_html_path':str(page),'description_page_revision':1,'fetched_at':'2026-10-12T00:00:00+00:00','candidate_domain':'music_0','duration_s':1.}
   paths={n:root/(n+'.json') for n in ['manifest','inventory','drafts','config']}
   paths['manifest'].write_text(json.dumps({'sources':[seed,{'status':'rejected_declared_size'}]}));paths['inventory'].write_text(json.dumps({'candidate_families':[{'candidate_family_id':'f1','source_ids':['wc_fixture']}]}))
   paths['drafts'].write_text('[]');reserved=root/'reserved.jsonl';reserved.write_text('')
   settings=q.GeminiAVSettings(model_env='AV_TEST_MODEL',api_key_env='AV_TEST_KEY',base_url_env='AV_TEST_URL')
   paths['config'].write_text(json.dumps(asdict(settings)))
   item={'question':'这段演奏的作曲家是谁？','visual_cue':'fixture visible clue','audio_cue':'fixture audible clue','media_matches_description':True,'external_information_required':True,'fields':[{'field_id':'composer','accepted_values':['Claude Debussy']} ]}
   responses=[(200,reply(json.dumps({'questions':[item]})))] if success else [(503,{'error':'fixture failure'})]
   with fixture_server(responses) as (url,received),patch.dict(os.environ,{'AV_TEST_MODEL':'test-only-gemini','AV_TEST_KEY':'fixture-no-real-key','AV_TEST_URL':url}):
    plan=root/'plan';q.prepare(paths['manifest'],paths['inventory'],reserved,paths['drafts'],plan,paths['config'],10)
    result=q.run(plan,workers=1,max_sources=10);again=q.run(plan,workers=1,max_sources=10)
    self.assertEqual(len(received),1);self.assertEqual(result,again);self.assertEqual(result['api_slots_charged'],1)
    self.assertIn('inlineData',received[0]['payload']['contents'][0]['parts'][0])
    if success:
     batch=root/'batch';report=q.materialize(plan,paths['manifest'],paths['inventory'],reserved,batch)
     self.assertEqual(report['tasks'],1)
     actor=(batch/'tasks-public.jsonl').read_text();self.assertNotIn('Debussy',actor);self.assertNotIn('example.com',actor)
     record_path=plan/'results/wc_fixture.json';record=json.loads(record_path.read_text());record['status']='failed';record.pop('questions');record_path.write_text(json.dumps(record))
     recovered=q.materialize(plan,paths['manifest'],paths['inventory'],reserved,root/'recovered')
     self.assertEqual(recovered['tasks'],1);self.assertEqual(json.loads(record_path.read_text())['status'],'failed')
    else:self.assertEqual(result['failed_author_calls'],1)
 def test_native_media_receipt_can_materialize_private_reference_without_actor_leak(self):self.exercise(True)
 def test_failed_http_attempt_is_not_replayed_on_resume(self):self.exercise(False)
if __name__=='__main__':unittest.main()

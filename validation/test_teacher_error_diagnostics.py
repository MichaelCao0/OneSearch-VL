import json,unittest
class TeacherErrorTests(unittest.TestCase):
 def test_missing_thought_signature_classified_without_body_or_key(self):
  from onesearch_vl.onesearch_infer.teacher_errors import summarize_http_error
  secret='UNIT_TEST_SECRET_DO_NOT_LOG'
  body=json.dumps({'error':{'code':400,'status':'INVALID_ARGUMENT','message':'Function call is missing a thought_signature. '+secret}}).encode()
  value=summarize_http_error(400,body)
  self.assertEqual(value['category'],'missing_thought_signature');self.assertEqual(value['provider_status'],'INVALID_ARGUMENT')
  self.assertNotIn(secret,json.dumps(value));self.assertNotIn('message',value)
 def test_media_parse_and_unknown_failure_do_not_store_body(self):
  from onesearch_vl.onesearch_infer.teacher_errors import summarize_http_error
  value=summarize_http_error(400,json.dumps({'error':{'message':'Unable to process input video'}}).encode())
  self.assertEqual(value['category'],'media_decode')
  value=summarize_http_error(503,b'<html>private upstream trace</html>')
  self.assertEqual(value['category'],'service_unavailable');self.assertNotIn('private',json.dumps(value))

class DurableDiagnosticsTests(unittest.TestCase):
 def test_failure_diagnostics_survive_aborted_rollout_in_timing_receipt(self):
  import os,tempfile
  from pathlib import Path
  from unittest.mock import patch
  from research.av_train.teacher_telemetry import instrument_runner_class
  class Runner:
   def prepare_request(self,*args):return b'',[]
   def infer(self,*args,**kwargs):
    return {'candidates':[{'finishReason':'ERROR','content':{'parts':[{'text':'Native teacher HTTP 400; response body withheld'}]}}],
      'raw_response':{'provider_error':{'http_status':400,'category':'missing_thought_signature','body_sha256':'a'*64,'body_bytes':42,'provider_status':'INVALID_ARGUMENT'}}}
  instrument_runner_class(Runner)
  with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'ONESEARCH_TEACHER_PROFILE_DIR':folder}):
   Runner().infer()
   row=json.loads(next(Path(folder).glob('*.json')).read_text())
   self.assertEqual(row.get('provider_error',{}).get('category'),'missing_thought_signature')

if __name__=='__main__':unittest.main()

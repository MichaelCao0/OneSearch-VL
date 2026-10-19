"""Allowlisted HTTP diagnostics; never retain messages, credentials or media."""
import hashlib,json
STATUS={'INVALID_ARGUMENT','FAILED_PRECONDITION','NOT_FOUND','PERMISSION_DENIED','UNAUTHENTICATED','RESOURCE_EXHAUSTED','UNAVAILABLE','INTERNAL','DEADLINE_EXCEEDED'}
def summarize_http_error(http_status,body):
 if not isinstance(body,bytes):raise ValueError('Diagnostic response bytes required')
 result={'http_status':int(http_status),'body_bytes':len(body),'body_sha256':hashlib.sha256(body).hexdigest(),'category':'unknown'}
 message=''
 if len(body)<=65536:
  try:
   data=json.loads(body);error=data.get('error',{}) if isinstance(data,dict) else {}
   if isinstance(error,dict):
    status=error.get('status')
    if status in STATUS:result['provider_status']=status
    message=error.get('message','') if isinstance(error.get('message',''),str) else ''
   elif isinstance(error,str):message=error
  except (ValueError,TypeError):pass
 text=message.casefold()
 patterns=[('missing_thought_signature',['thought_signature','thought signature']),('media_decode',['unable to process input video','failed to decode','invalid video','video format','failed to process video','unable to decode']),
 ('unsupported_schema',['responseschema','response_schema','unknown field','unsupported schema']),('credit_or_quota',['insufficient','quota','credit','balance']),('model_routing',['model not found','no available channel','no available provider','model is not supported']),('context_limit',['context length','token limit','too many tokens'])]
 for category,terms in patterns:
  if any(term in text for term in terms):result['category']=category;break
 if result['category']=='unknown':
  result['category']={401:'authentication',403:'permission',429:'rate_limit',413:'request_too_large',503:'service_unavailable',502:'upstream_failure',504:'upstream_timeout'}.get(http_status,'unknown')
 return result

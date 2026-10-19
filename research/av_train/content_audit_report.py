"""Display completed automatic audit suggestions without altering human drafts."""
from pathlib import Path
import json,shutil,os,hashlib


def main_search_disposition(verdict):
 if not verdict['media_matches_question'] or verdict['specificity']=='unsupported_link':return 'reject'
 if not verdict['answer_fields_supported_by_publisher'] or verdict['question_leaks_identifying_clue']:return 'revise'
 if verdict['search_requirement']=='perception_only':return 'control'
 if verdict['search_requirement']=='ambiguous_unanswerable':return 'revise'
 return verdict['decision']


def annotate_package(source,audit_run,destination):
 from research.av_data.training_sample_review import render_review
 from research.av_train.candidate_content_audit import validate_audit
 src=Path(source);audit=Path(audit_run);dst=Path(destination)
 payload=json.loads((src/'payload.json').read_text());inputs={x['task_id']:x for x in json.loads((audit/'inputs.private.json').read_text())}
 summary={'automatic_opinions':0,'unknown':0,'human_gold':0,'human_review_modified':False}
 for case in payload['cases']:
  tid=case['task']['task_id'];path=audit/'results'/(tid+'.json');row=inputs[tid]
  raw=(src/case['media']).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==row['media_sha256']
  assert case['task']['question']==row['question']
  record=json.loads(path.read_text()) if path.exists() else {}
  if record.get('status')=='completed':
   verdict=validate_audit(record['verdict'])
   case['automatic_content_audit']={'verdict':verdict,'main_search_disposition':main_search_disposition(verdict),'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_id':record['generation_identity']['model_id'],'human_gold':False,'human_reviewed':False}
   summary['automatic_opinions']+=1
  else:summary['unknown']+=1
 shutil.copytree(src,dst,copy_function=os.link)
 for name in ['index.html','payload.json','manifest.json']:(dst/name).unlink()
 payload['summary']['automatic_content_audit']=summary
 (dst/'payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 html=render_review(payload)
 marker='note.oninput=()=>{record.note=note.value;save()};}'
 assert marker in html
 extra='''note.oninput=()=>{record.note=note.value;save()};
if(c.automatic_content_audit){const a=c.automatic_content_audit,v=a.verdict,detail=h('details',null,box);h('summary','查看自动 AV 质量抽查意见（模型判断，待人审）',detail);h('p',`模型 ${a.model_id}；真实内容建议 ${v.content_category}；主搜索集建议 ${a.main_search_disposition}；搜索类型 ${v.search_requirement}；模态角色 ${v.modality_role}`,detail);h('p','画面：'+v.observed_visual_cue,detail);h('p','声音：'+v.observed_audio_cue,detail);h('p','依据：'+v.reason,detail);h('p','自动意见不修改你的选择，不证明音频增益；perception_only 作为感知控制，不自动进入主搜索集。',detail);}}
'''
 html=html.replace(marker,extra)
 (dst/'index.html').write_text(html)
 m=json.loads((src/'manifest.json').read_text());m.update(parent_review_package=str(src.resolve()),automatic_audit_run=str(audit.resolve()),automatic_audit_summary=summary,human_draft_schema_unchanged=True)
 (dst/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 return m

"""Import version-bound browser review drafts into auditable candidate decisions.

This consumes real human exports. Agreement is a construction candidate;
formal benchmark gold also needs interval/source-link annotation and split checks.
No existing package, browser draft or training data is modified.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json

OPTIONS={
 'decision':{'pending','keep','revise','reject'},
 'audio_role':{'pending','necessary','helpful','redundant','unusable'},
 'needs_search':{'pending','required','perception_only','ambiguous'},
 'answer_support':{'pending','supported','revise','unsupported'},
}

def file_sha(path):
 with Path(path).open('rb') as stream:
  return hashlib.file_digest(stream,'sha256').hexdigest()

def load_package(package):
 root=Path(package).resolve()
 payload=json.loads((root/'payload.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
 task_sha=file_sha(root/'tasks-public.jsonl');ref_sha=file_sha(root/'answer-candidates.private.jsonl')
 if task_sha!=payload['summary']['tasks_sha256'] or task_sha!=manifest['task_batch_sha256']:
  raise ValueError('Task batch does not match package identity')
 if ref_sha!=payload['summary']['candidate_references_sha256'] or ref_sha!=manifest['candidate_references_sha256']:
  raise ValueError('Reference batch does not match package identity')
 tasks=[json.loads(l) for l in (root/'tasks-public.jsonl').read_text().splitlines() if l]
 refs=[json.loads(l) for l in (root/'answer-candidates.private.jsonl').read_text().splitlines() if l]
 def keyed(rows,key):
  out={x[key]:x for x in rows}
  if len(out)!=len(rows):raise ValueError('Duplicate task identity')
  return out
 by_task=keyed(tasks,'task_id');by_ref=keyed(refs,'task_id');cases={}
 for c in payload['cases']:
  tid=c['task']['task_id']
  if tid in cases:raise ValueError('Duplicate package task')
  cases[tid]=c
 receipts=keyed(manifest['receipts'],'task_id')
 if set(by_task)!=set(by_ref) or set(by_task)!=set(cases) or set(by_task)!=set(receipts):
  raise ValueError('Package membership mismatch')
 for tid,c in cases.items():
  if c['task']!=by_task[tid] or c['reference_candidate']!=by_ref[tid]:
   raise ValueError('Rendered question/reference changed')
  path=(root/c['media']).resolve()
  if not path.is_relative_to(root) or file_sha(path)!=receipts[tid]['input_sha256']:
   raise ValueError('Media no longer matches reviewed input')
 return root,payload,manifest,cases,receipts,task_sha,ref_sha

def disposition(rows):
 active=[x for x in rows if x['decision']!='pending']
 if not active:return 'awaiting_review'
 complete=[x for x in active if all(x[k]!='pending' for k in OPTIONS)]
 if len(complete)<2:return 'awaiting_second_review' if complete else 'incomplete_review'
 signatures={tuple(x[k] for k in OPTIONS) for x in complete}
 if len(signatures)>1:return 'needs_adjudication'
 row=complete[0]
 if row['decision']=='reject':return 'excluded_candidate'
 if row['decision']=='revise' or row['answer_support']!='supported' or row['needs_search']=='ambiguous' or row['audio_role']=='unusable':
  return 'needs_revision'
 if row['needs_search']=='perception_only':return 'perception_control_candidate'
 return 'consensus_keep_candidate'

def import_reviews(package,draft_files,out_dir):
 root,payload,manifest,cases,receipts,task_sha,ref_sha=load_package(package)
 imported={tid:[] for tid in cases};reviewer_ids=set();sources=[]
 for value in draft_files:
  path=Path(value);data=json.loads(path.read_text())
  if data.get('schema_version')!='av-searchqa-authoring-review-draft.v2' or data.get('benchmark_gold') is not False:
   raise ValueError('Expected a browser review draft, never a gold assertion')
  reviewer=data.get('reviewer_id')
  if not isinstance(reviewer,str) or not reviewer.strip() or reviewer!=reviewer.strip() or len(reviewer)>200 or reviewer in reviewer_ids:
   raise ValueError('Unique nonempty reviewer identity required')
  created=data.get('created_at')
  try:
   stamp=datetime.fromisoformat(created.replace('Z','+00:00'))
  except (ValueError,TypeError,AttributeError):raise ValueError('Invalid reviewer timestamp') from None
  if stamp.tzinfo is None:raise ValueError('Reviewer timestamp must include timezone')
  if data.get('task_batch_sha256')!=task_sha or data.get('candidate_references_sha256')!=ref_sha:
   raise ValueError('Review belongs to another question/reference version')
  rows=data.get('reviews')
  if not isinstance(rows,list):raise ValueError('Review rows must be a list')
  seen=set()
  for row in rows:
   if not isinstance(row,dict) or set(row)!={'task_id','note',*OPTIONS}:
    raise ValueError('Closed browser review record required')
   tid=row['task_id']
   if tid not in cases or tid in seen:raise ValueError('Unknown or duplicate reviewed task')
   seen.add(tid)
   if any(row[k] not in allowed for k,allowed in OPTIONS.items()):raise ValueError('Unknown review choice')
   if not isinstance(row['note'],str) or len(row['note'])>20000:raise ValueError('Bounded reviewer note required')
   imported[tid].append(dict(row,reviewer_id=reviewer,created_at=created,draft_sha256=file_sha(path)))
  reviewer_ids.add(reviewer);sources.append({'file':str(path.resolve()),'sha256':file_sha(path),'reviewer_id':reviewer})
 records=[]
 for tid,case in cases.items():
  rows=imported[tid];receipt=receipts[tid]
  records.append({'schema':'onesearch-av-human-review-candidate.v1','task_id':tid,
   'status':disposition(rows),'reviewer_ids':[x['reviewer_id'] for x in rows],
   'reviews':rows,'media_binding':{k:receipt[k] for k in ['source_id','input_sha256','parent_sha256','query_interval_on_parent']},
   'question_sha256':hashlib.sha256(case['task']['question'].encode()).hexdigest(),
   'reference_sha256':hashlib.sha256(json.dumps(case['reference_candidate'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
   'human_gold':False,'remaining_gold_requirements':['explicit interval/source-identity link verification','recording-family train/dev/test exclusion','adjudication if reviewers disagree']})
 summary={'schema':'onesearch-av-review-import.v1','at_utc':datetime.now(timezone.utc).isoformat(),
  'split':payload['summary']['split'],'package':str(root),'task_batch_sha256':task_sha,
  'candidate_references_sha256':ref_sha,'source_manifest_sha256':file_sha(root/'manifest.json'),
  'questions':len(records),'reviewers':len(reviewer_ids),'draft_files':sources,
  'statuses':dict(Counter(x['status'] for x in records)),'human_gold':0,
  'formal_freeze_performed':False,'source_packages_modified':False}
 dest=Path(out_dir).resolve();dest.mkdir(parents=True,exist_ok=False)
 (dest/'review-records.private.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
 (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 return summary

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--package',required=True);parser.add_argument('--review',action='append',default=[])
 parser.add_argument('--out',required=True);args=parser.parse_args()
 print(json.dumps(import_reviews(args.package,args.review,args.out),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

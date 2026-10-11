"""Blank reviewer packets and deterministic import of independently submitted annotations.

Suitability UI notes are not semantic gold. Only explicit completed reviewer input
can create references. Differing answers/packages/intervals require adjudication;
this importer does not decide semantics, infer annotator independence, or authenticate
identities. Reviewer identity and actual replay remain the data team's responsibility.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import hashlib
from pathlib import Path
from .contracts import REFERENCE_VERSION,FAMILIES,json_copy,nonempty,validate_reference
from research.av_data.build_research_batch import write_json,write_jsonl

REVIEW_FIELDS={'schema_version','reviewer_id','submitted_at','status','task_id','family','facts',
               'evidence_packages','source_family_checked','modality_path','notes'}


def blank_review(task,family):
    if family not in FAMILIES:raise ValueError('Unknown task family')
    return {'schema_version':'av-researchbench.review.v1','reviewer_id':None,'submitted_at':None,
            'status':'pending','task_id':task['task_id'],'family':family,'facts':[],'evidence_packages':[],
            'source_family_checked':False,'modality_path':None,'notes':''}


def validate_review(review,task):
    r=json_copy(review)
    if not isinstance(r,dict) or set(r)!=REVIEW_FIELDS or r['schema_version']!='av-researchbench.review.v1':raise ValueError('Invalid submitted review schema')
    if r['status']!='submitted' or r['task_id']!=task['task_id'] or r['family'] not in FAMILIES:raise ValueError('Review pending or wrong task')
    nonempty(r['reviewer_id'],'reviewer_id');nonempty(r['submitted_at'],'submitted_at')
    try:when=datetime.fromisoformat(r['submitted_at'].replace('Z','+00:00'))
    except ValueError:raise ValueError('Submitted review requires ISO timestamp') from None
    if when.tzinfo is None:raise ValueError('Submitted review timestamp requires timezone')
    if r['source_family_checked'] is not True:raise ValueError('Original recording/source family review required')
    if r['modality_path'] not in {'joint_av','audio','video','av_text','text_shortcut'}:raise ValueError('Information path review required')
    nonempty(r['notes'],'review notes')
    return r


def annotation_identity(review):
    # Ordering of facts/packages/evidence and accepted synonyms must not create fake disagreement.
    def key(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)
    facts=[{'fact_id':f['fact_id'],'accepted_values':sorted(f['accepted_values'],key=key)} for f in review['facts']]
    packages=[sorted(package,key=key) for package in review['evidence_packages']]
    return key({'family':review['family'],'facts':sorted(facts,key=key),'evidence_packages':sorted(packages,key=key),
                'modality_path':review['modality_path']})


def merge_reviews(task,sources,review_a,review_b):
    a=validate_review(review_a,task);b=validate_review(review_b,task)
    if a['reviewer_id']==b['reviewer_id']:raise ValueError('Two independent reviewers required')
    def reference(review):
        return {'schema_version':REFERENCE_VERSION,'task_id':task['task_id'],'status':'adjudicated',
                'reviewer_ids':[a['reviewer_id'],b['reviewer_id']],'facts':review['facts'],
                'evidence_packages':review['evidence_packages'],'family':review['family']}
    # Check source coordinates and necessary fact coverage on BOTH submitted packages.
    validate_reference(reference(a),task,sources);validate_reference(reference(b),task,sources)
    if annotation_identity(a)!=annotation_identity(b):
        return {'status':'needs_adjudication','task_id':task['task_id'],'reference':None,
                'reviewer_ids':[a['reviewer_id'],b['reviewer_id']],'reason':'Answers, evidence packages, intervals or information path disagree.'}
    return {'status':'agreed','task_id':task['task_id'],'reference':reference(a),
            'reviewer_ids':[a['reviewer_id'],b['reviewer_id']],'modality_path':a['modality_path']}



ADJUDICATION_FIELDS={'schema_version','task_id','status','adjudicator_id','submitted_at',
                     'review_a_sha256','review_b_sha256','facts','evidence_packages',
                     'modality_path','source_family_checked','notes'}


def review_digest(review):
    return hashlib.sha256(json.dumps(review,sort_keys=True,separators=(',',':'),
                                    ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def adjudicate_reviews(task,sources,review_a,review_b,resolution):
    """Import an explicit third review; never average boundaries or infer decisions.

    Source replay, reviewer independence and identities require human evidence.
    This function validates coordinates, exact input identities and provenance.
    """
    a=validate_review(review_a,task);b=validate_review(review_b,task)
    if a['family']!=b['family']:raise ValueError('Review family disagreement requires task reauthoring')
    base=merge_reviews(task,sources,a,b)
    if base['status']!='needs_adjudication':raise ValueError('Adjudication requires an unresolved review difference')
    r=json_copy(resolution)
    if (not isinstance(r,dict) or set(r)!=ADJUDICATION_FIELDS or
            r['schema_version']!='av-researchbench.adjudication.v1' or
            r['task_id']!=task['task_id'] or r['status']!='submitted'):
        raise ValueError('Explicit submitted adjudication required')
    if r['review_a_sha256']!=review_digest(a) or r['review_b_sha256']!=review_digest(b):
        raise ValueError('Adjudication original review hash mismatch')
    adjudicator=nonempty(r['adjudicator_id'],'adjudicator_id')
    if adjudicator in (a['reviewer_id'],b['reviewer_id']):raise ValueError('Independent third adjudicator required')
    # Reuse completion, timezone, modality and source-review validation.
    submitted={'schema_version':'av-researchbench.review.v1','reviewer_id':adjudicator,
               'submitted_at':r['submitted_at'],'status':'submitted','task_id':task['task_id'],
               'family':a['family'],'facts':r['facts'],'evidence_packages':r['evidence_packages'],
               'source_family_checked':r['source_family_checked'],'modality_path':r['modality_path'],
               'notes':r['notes']}
    validate_review(submitted,task)
    ref={'schema_version':REFERENCE_VERSION,'task_id':task['task_id'],'status':'adjudicated',
         'reviewer_ids':[a['reviewer_id'],b['reviewer_id'],adjudicator],'family':a['family'],
         'facts':r['facts'],'evidence_packages':r['evidence_packages']}
    validate_reference(ref,task,sources)
    return {'status':'adjudicated','task_id':task['task_id'],'reference':ref,
            'reviewer_ids':ref['reviewer_ids'],'modality_path':r['modality_path'],
            'review_a_sha256':review_digest(a),'review_b_sha256':review_digest(b),
            'adjudication_sha256':review_digest(r),'resolution_notes':r['notes']}


def prepare_review_packets(batch_dir,out_dir):
    root=Path(batch_dir);out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    rows=json.loads((root/'case-authoring.private.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    for reviewer in ('A','B'):
        directory=out/reviewer;directory.mkdir()
        for row in rows:write_json(directory/(row['task']['task_id']+'.json'),blank_review(row['task'],row['family']))
    summary={'case_count':len(rows),'reviewer_packets':2,'submitted_review_count':0,'adjudicated_reference_count':0}
    write_json(out/'summary.json',summary)
    (out/'README.md').write_text('''# Semantic annotation packets

A and B replay source media independently before sharing annotations. Enter actual
reviewer_id and submitted_at; status becomes submitted only after facts, source
intervals/quotes and evidence packages have been checked. Each alternative package
must support every necessary fact; evidence within a package is AND, alternatives
are OR. Reference fact_ids must match target_questions. Use observable event ordinals
when the task asks for an instance, not hidden random event names. Do not copy model
observations into references without checking original media. Title assertions and
actual AV facts are distinct. Suitability UI notes do not satisfy this review.

The current candidates include grouped questions. Before formal annotation, split
combined questions into atomic necessary facts where needed, preserving a new task
revision. Free-text answer semantics require a calibrated semantic scorer; the v0
exact-value scorer only measures canonical labels/values.

When independently submitted annotations agree, import can create a versioned
reference. Disagreement stays needs_adjudication; no silent averaging of times.
''',encoding='utf-8')
    return summary


def import_batch_reviews(batch_dir,reviews_dir,out_dir,adjudications_dir=None):
    root=Path(batch_dir);reviews=Path(reviews_dir);out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    rows=json.loads((root/'case-authoring.private.json').read_text())
    sources={r['source_id']:r for r in (json.loads(l) for l in (root/'sources.private.jsonl').read_text().splitlines() if l)}
    refs=[];decisions=[]
    for row in rows:
        tid=row['task']['task_id'];paths=[reviews/person/(tid+'.json') for person in ('A','B')]
        if any(not p.is_file() for p in paths):
            decisions.append({'task_id':tid,'status':'missing_review'});continue
        a,b=(json.loads(p.read_text()) for p in paths)
        if any(r.get('status')!='submitted' for r in (a,b)):
            decisions.append({'task_id':tid,'status':'pending_review'});continue
        if a['family']!=row['family'] or b['family']!=row['family']:
            raise ValueError('Review family differs from authored task')
        decision=merge_reviews(row['task'],sources,a,b)
        if decision['status']=='needs_adjudication' and adjudications_dir is not None:
            resolution_path=Path(adjudications_dir)/(tid+'.json')
            if resolution_path.is_file():
                decision=adjudicate_reviews(row['task'],sources,a,b,json.loads(resolution_path.read_text()))
        decisions.append(decision)
        if decision['reference'] is not None:refs.append(decision['reference'])
    out.mkdir(parents=True,exist_ok=False)
    write_jsonl(out/'references.private.jsonl',refs);write_json(out/'decisions.private.json',decisions)
    summary={'case_count':len(rows),'adjudicated_reference_count':len(refs),
             'pending_or_disagreement_count':len(rows)-len(refs),'teacher_rollout_ready':False,
             'note':'Source split freeze and task admission are separate from semantic reference import.'}
    write_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','import'])
    p.add_argument('--batch-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--reviews-dir');p.add_argument('--adjudications-dir')
    a=p.parse_args()
    if a.action=='import' and not a.reviews_dir:p.error('--reviews-dir required for import')
    result=prepare_review_packets(a.batch_dir,a.out_dir) if a.action=='prepare' else import_batch_reviews(a.batch_dir,a.reviews_dir,a.out_dir,a.adjudications_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

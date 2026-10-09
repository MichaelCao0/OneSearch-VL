"""Generate separate Chinese A/B and adjudication forms, with all labels blank.

These private work forms are not ClaimGold and must never enter actor exports.
Only trusted technical source identities and public task wording are prefilled.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .tasks import validate_task_public

CONDITIONS = ('metadata-only','ASR-only','ASR+frames','raw AV')
SOURCE_FIELDS = ('source_id','asset_sha256','asset_path','media_type','duration_s',
                 'origin_url','timestamp_origin','public_metadata')


def task_stem(task_id):
    return 'task_'+hashlib.sha256(task_id.encode('utf-8')).hexdigest()[:20]


def _catalog(registry):
    catalog=[]
    for source_id, record in sorted(registry.items()):
        if record.get('source_id') != source_id: raise ValueError('registry ID mismatch')
        if not re.fullmatch(r'[0-9a-f]{64}',str(record.get('asset_sha256',''))):
            raise ValueError('source hash missing or malformed')
        entry={key:copy.deepcopy(record.get(key)) for key in SOURCE_FIELDS}
        entry['technical_stream_types']=sorted({s.get('codec_type','unknown') for s in record.get('stream_specs',[])})
        entry['interpretation']='技术身份与公开metadata；不预填自然性、声源、事件关系或独立性结论。'
        catalog.append(entry)
    return catalog


def blank_review(task,registry,slot):
    if slot not in {'A','B'}: raise ValueError('reviewer slot must be A or B')
    task=validate_task_public(task,registry,task.get('corpus_id'),task.get('split'))
    return {'schema_version':'av-independent-review.blank.v1','visibility':'annotation_private_only',
        'annotation_status':'blank_unreviewed','reviewer_slot':slot,'reviewer_identity':None,
        'started_at':None,'submitted_at':None,'elapsed_minutes':None,'prior_exposure':None,
        'inputs_used_manifest_hashes':[],'task_public_snapshot':task,'source_catalog':_catalog(registry),
        'source_semantic_reviews':[],
        'target_annotations':[{'claim_id':target['claim_id'],'target_question':target['question'],
            'question_disposition':None,'proposition':None,'qualifiers':{},'expected_decision':None,'status':None,
            'scope':'within_snapshot','direct_observations':[],'source_assertions':[],'external_facts':[],
            'valid_evidence_sets':[],'required_bindings':[],'missing_evidence':[],
            'modality_trials':[{'condition':condition,'availability':None,'inputs_manifest_hash':None,
                'decision':None,'evidence_sufficient':None,'basis_refs':[],'limitations':None}
                for condition in CONDITIONS],
            'modality_requirement':[],'modality_justification':None}
            for target in task['target_questions']],
        'source_relations':[],'seed_disposition':None,'submission_confirmation':None,
        'instructions_path':'docs/research/annotation/Independent-Review-Instructions-zh.md',
        'warnings':['本表无gold；空白不代表unknown。','原始录制与描述页不是两个独立观察。',
                    '相邻/相似与RMS不证明声源、身份或物理因果。',
                    'A/B首次提交前不得互看；ASR等输入不可用时填unavailable，不伪造评估。']}


def blank_adjudication(task):
    return {'schema_version':'av-independent-adjudication.blank.v1','visibility':'annotation_private_only',
        'annotation_status':'blank_unreviewed','task_id':task['task_id'],'task_revision':task['revision'],
        'corpus_id':task['corpus_id'],'adjudicator_identity':None,'started_at':None,'submitted_at':None,
        'reviewer_A_form_hash':None,'reviewer_B_form_hash':None,'review_independence_confirmed':None,
        'disagreements':[],'replayed_evidence':[],'claim_decisions':[],
        'gold_export_approved':None,'task_disposition':None,'remaining_pending':[],
        'event_recording_cluster_decision':None,'submission_confirmation':None}


def render_review(form):
    task=form['task_public_snapshot'];slot=form['reviewer_slot']
    lines=[f'# 审核者 {slot}：{task["task_id"]}', '',
        '状态：**blank_unreviewed**。私有空白表，不是gold。另一审核者提交前不可互看。', '',
        f'任务 revision：{task["revision"]}；corpus：{task["corpus_id"]}；split：{task["split"]}。', '',
        f'研究问题：{task["question"]}', '',
        '真实审核者 ID：____；开始/提交时间：____；耗时：____；prior_exposure：____。', '',
        '输入 manifest/hash：____。说明见 Independent-Review-Instructions-zh.md。', '',
        '初始 source IDs：'+', '.join(task['initial_sources']), '',
        '## 原始来源与技术身份', '',
        '下面 metadata 是 publisher assertion，不是语义、声源、独立性或因果真值。', '']
    for source in form['source_catalog']:
        lines.extend([f'- source_id：`{source["source_id"]}`；type：{source["media_type"]}；duration：{source["duration_s"]} s。',
                      f'  asset_sha256：`{source["asset_sha256"]}`',
                      f'  原始文件：`{source["asset_path"]}`；origin URL：{source["origin_url"]}',
                      '  自然性/编辑/配乐/旁白/同期声/时间问题：____。'])
    for claim in form['target_annotations']:
        lines.extend(['',f'## 目标项 {claim["claim_id"]}', '',claim['target_question'], '',
            'proposition / 对象 / 量词 / 时间与 qualifiers：____。',
            '题干是否预设无依据事实；保留/改写/拒绝：____。', '',
            '| 条件 | input manifest/hash | available/unavailable | 本条件判断/限制 | evidence sufficient? | refs/缺口 |',
            '|---|---|---|---|---|---|'])
        for condition in CONDITIONS: lines.append(f'| {condition} | ____ | ____ | ____ | ____ | ____ |')
        lines.extend(['', 'speech-sufficient / non-speech-acoustic / joint-AV-binding / control 与论证：____。',
            'Direct observations + source/hash/[start,end)：____。',
            'Source assertions / external facts + exact quote/char bounds：____。',
            'expected_decision：____；status（supported/contradicted/undetermined）：____。',
            'scope：within_snapshot；missing evidence/何种新证据可区分：____。',
            'Evidence P1（包内 AND）：____；P2（包间 OR）：____。',
            '必要 bindings 与 basis_refs；co-occurrence 到声源/因果的缺口：____。'])
    lines.extend(['','## 来源关系与提交','',
        'recording/event/independent_observation 是不同关系。原录制和描述页不能算两次观察。',
        'source pairs / relation_type / same-different-independent-dependent-unknown / basis_refs：____。',
        '建议接受/改写重标/exploration only/拒绝/pending 与理由：____。',
        '真实审核者提交确认：____。空白/pending/unavailable 不算完成。',''])
    return '\n'.join(lines)


def _read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path,data):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def generate(tasks_path,sources_path,out_dir):
    tasks_path,sources_path,out_dir=map(Path,(tasks_path,sources_path,out_dir))
    tasks=_read_jsonl(tasks_path);sources=_read_jsonl(sources_path)
    if not 1 <= len(tasks) <= 10: raise ValueError('pilot forms require 1-10 tasks')
    registry={source['source_id']:source for source in sources}
    if len(registry)!=len(sources):raise ValueError('duplicate source IDs')
    if len({task['task_id'] for task in tasks})!=len(tasks):raise ValueError('duplicate task IDs')
    forms=[(blank_review(task,registry,'A'),blank_review(task,registry,'B'),blank_adjudication(task)) for task in tasks]
    out_dir.mkdir(parents=True,exist_ok=False)
    for slot in ('A','B','adjudication'): (out_dir/slot).mkdir()
    for a,b,c in forms:
        stem=task_stem(c['task_id'])
        for slot,form in [('A',a),('B',b)]:
            _write_json(out_dir/slot/(stem+'.json'),form)
            with (out_dir/slot/(stem+'.md')).open('x',encoding='utf-8') as f:f.write(render_review(form))
        _write_json(out_dir/'adjudication'/(stem+'.json'),c)
        template=Path('docs/research/annotation/Adjudication-Blank-zh.md').read_text(encoding='utf-8')
        with (out_dir/'adjudication'/(stem+'.md')).open('x',encoding='utf-8') as f:
            f.write(f'<!-- task_id={c["task_id"]}; revision={c["task_revision"]}; 未填写裁决 -->\n'+template)
    receipt={'schema_version':'av-review-forms-generation.v1','generated_at':datetime.now(timezone.utc).isoformat(),
        'tasks':len(tasks),'registry_sources':len(sources),'status':'generated_blank_forms_not_human_review',
        'tasks_path':str(tasks_path.resolve()),'tasks_sha256':hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        'sources_path':str(sources_path.resolve()),'sources_sha256':hashlib.sha256(sources_path.read_bytes()).hexdigest(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'out_dir':str(out_dir.resolve()),'human_signatures':0,'gold_records':0,
        'files':[{'path':str(path.relative_to(out_dir)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(out_dir.rglob('*')) if path.is_file()]}
    _write_json(out_dir/'generation-receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',required=True);parser.add_argument('--sources',required=True)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    print(json.dumps(generate(args.tasks,args.sources,args.out_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

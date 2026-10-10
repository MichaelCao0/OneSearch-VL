"""Create-only natural development candidate batch, shared corpus and blank references.

Questions are concrete authoring candidates, not assertions that the media contains
all requested events. Existing explored assets are development-only. Training uses
new independently acquired source families; no test corpus is created here.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from research.av_bench.contracts import TASK_VERSION,REFERENCE_VERSION,validate_task,validate_reference
from research.av_search.corpus import build_corpus,canonical
from research.av_search.retrieval import search_sources
from research.av_agent.tools import read_text

SELECTORS={
 'engine_old':'steam engine 1903','engine_short':'file:steam engine.','engine_run':'steam engine operating',
 'press':'printing presses','cello':'tchêlistas','violin':'stroh violin','gong':'gong and drum',
 'drum':'cardio drumming','water':'shila waterfall','bird':'little corella feeding'}
# Public questions; private source keys guide author review, never actor retrieval.
SPECS=[
 ('source_discovery','technical_mechanisms',['engine_old','engine_short','engine_run','press'],['engine_old'],
  '参照起始录制中的机械运转，在语料库寻找另一份具有相似周期运动、但可听节律或可见部件不同的机械录制。返回所选来源及比较区间。',
  ['另一录制的主要可见运动是什么？','两份录制的可听节律有什么相同和不同？','哪些来源区间支持这次匹配？'],
  '必须能观察到可区分的机械运动与声音；标题匹配可能构成捷径。'),
 ('source_discovery','live_performance',['violin','cello','gong','drum'],['violin'],
  '参照起始演奏录制，寻找另一份可以观察到弓与弦相互作用的演奏录制。定位可见演奏动作与可听音符共同出现的片段，说明它与起始录制的区别。',
  ['所选来源中可见的演奏动作是什么？','该区间可听声音是持续还是离散的？','匹配来源和区间在哪里？'],
  '应核实弓弦动作可见、音轨确实可用，并检查 metadata-only 是否已足够。'),
 ('source_discovery','live_performance',['gong','drum','cello','violin'],['gong'],
  '参照起始录制中的打击演奏，搜索另一份由可见重复敲击动作形成节奏的录制。比较敲击对象和声音变化，并标出所用区间。',
  ['另一个来源里敲击对象是什么？','敲击动作与可听节奏有什么对应？','证据来源和区间在哪里？'],
  'Cardio Drumming 的录制音轨可能包含配乐；若无法听到实际敲击，需改写任务。'),
 ('source_discovery','animals_environment',['bird','water'],[],
  '在语料库寻找一份可见动物在户外活动的录制，定位活动片段，区分该片段可听见的动物发声与其他背景声。不要仅凭画面认定动物正在发声。',
  ['动物可见的主要活动是什么？','能确认的声音类别分别有哪些？','哪一来源和区间支持这些观察？'],
  '动物来源目前仅一份，干扰候选太少；正式版需增加同类自然录制。'),
 ('event_localization','technical_mechanisms',['engine_run'],['engine_run'],
  '在起始机械录制中，定位一段可以完整追踪至少两个连续运动周期的最短连续区间；同时记录该区间里可听的周期性声音。',
  ['完整周期依据哪个可见部件判断？','所定位区间的起止时间是什么？','可听周期与可见周期有什么对应？'],
  '不能预设存在启动/停机；审核者需核对周期计数和最短区间定义。'),
 ('event_localization','live_performance',['violin'],['violin'],
  '在起始演奏录制中，定位第一个能清楚看到弓运动方向改变的片段。描述转向前后的可听音符是否也发生变化，并给出源时间。',
  ['首次清楚的弓转向何时发生？','转向前后声音有何可听变化？','哪一段支持声画对应判断？'],
  '需要确认弓转向真正可见；声音变化不应被写入预设答案。'),
 ('event_localization','live_performance',['gong'],['gong'],
  '在起始打击乐录制中，定位最后一次清楚可见的敲击及其后的声音延续片段。给出敲击时刻与可听延续的结束边界。',
  ['最后一次清楚可见的敲击是什么动作？','敲击与声音延续的区间分别在哪里？','声音延续期间画面中还有哪些动作？'],
  '必须核实声音确为可听延续，素材末尾截断时标记右删失边界。'),
 ('event_localization','animals_environment',['bird'],['bird'],
  '在起始动物录制中，定位第一个清楚可见的喙接触地面并再次抬起的完整动作。记录这期间实际可听声音及其与动作的关系。',
  ['首次完整动作的起止时间是什么？','该动作包含哪些可见阶段？','动作期间可听到什么，是否能对应到该动物？'],
  '如果动作在片头已开始或喙不可见，则需改写实例定义；不要求一定存在动物声。'),
 ('cross_source_comparison','technical_mechanisms',['engine_old','engine_run'],['engine_old'],
  '寻找另一份机械运转录制，与起始录制比较主要运动部件和可听节律。分别给出每份录制中至少两个周期的证据片段。',
  ['两份录制的主要运动部件各是什么？','两段可听节律有什么差别？','两段比较证据的区间在哪里？'],
  '不能把不同剪裁当独立录制；两份录制的机械部件和音轨需实际核实。'),
 ('cross_source_comparison','live_performance',['violin','cello'],['violin'],
  '寻找另一份弓弦演奏录制，比较两份材料中的持琴/运弓动作、演奏人数和声音持续性。每项比较都注明支持它的来源区间。',
  ['两份录制的演奏人数和持琴方式各是什么？','运弓动作与声音持续性如何比较？','对应证据区间在哪里？'],
  '演奏人数不能只靠说明页；素材可能不是同一首曲目，问题不要求同曲比较。'),
 ('cross_source_comparison','live_performance',['gong','drum'],['gong'],
  '寻找另一份有可见重复敲击的录制，比较两份材料的敲击对象、动作序列和实际可听节奏。区分原声、配乐和无法判断的情况。',
  ['两份录制的敲击对象与动作序列各是什么？','两段实际音轨的节奏有何差别？','声画对应结论由哪些区间支持？'],
  '如果其中一段只有配乐，可以测观察能力，但不能作为两种敲击音色的比较。'),
 ('cross_source_comparison','animals_environment',['water','bird'],['water'],
  '寻找一份可见动物活动的户外录制，与起始自然环境录制比较连续背景声、离散声音和可见运动。返回两份来源及时间区间。',
  ['两份录制里主要可见运动各是什么？','连续背景声与离散声音如何比较？','哪两个来源区间支持比较？'],
  '目前环境/动物来源不足且语义差异过大；作为流程样例，不代表正式难度。'),
 ('research_composition','technical_mechanisms',['engine_old','engine_run','press'],['engine_old'],
  '搜索并比较两份机械运转录制，结合各自冻结说明页整理一张表：可见运动、实际可听声音、发布者声称的设备或用途。给出声画区间与说明页引用。',
  ['两份录制的运动与声音分别是什么？','发布者对设备或用途分别作了什么说明？','哪些 AV 区间和文字引用支持这张表？'],
  '两份原始 AV 必须分别观察；公开说明不能替代 AV 事实。'),
 ('research_composition','live_performance',['gong','drum'],['gong'],
  '搜索另一份打击或节奏演奏录制，利用两份原片和说明页完成对照：敲击对象、声音与动作对应、公开作者与许可。分别给出 AV 和文字依据。',
  ['两份录制可观察的敲击对象和声画对应是什么？','公开作者与许可分别是什么？','AV 与文字证据分别在哪里？'],
  '作者与许可容易搜索，主要分数应来自两份 AV 的实际内容组合。'),
 ('research_composition','animals_environment',['water','bird'],['water'],
  '搜索一份动物户外活动录制，与起始环境录制共同制作自然观察记录：可见活动、实际音轨、发布者是否提供地点。将原片观察和文字说明分别引用。',
  ['两份录制的可见活动与实际声音各是什么？','发布者是否提供地点，具体说了什么？','哪些原片区间和文字支持这些记录？'],
  '说明页未提供地点时允许“未提供”；不能从动物物种或风景自行补地点。'),
 ('research_composition','live_performance',['violin','cello'],['violin'],
  '搜索另一份弓弦演奏录制，组合两份原片与公开说明，比较乐器可见结构、演奏动作、实际声音及发布来源。输出带声画片段和文字引用的简短研究表。',
  ['两份录制的乐器结构与演奏动作各是什么？','实际音轨有什么可比较特征？','哪些片段和公开说明支持比较与来源归属？'],
  '结构观察应来自画面；不能把 Stroh 名称或说明自动提升为已观察结构。'),
]


def build_tasks(registry,corpus_id):
    selected={}
    for key,needle in SELECTORS.items():
        found=[r['source_id'] for r in registry.values() if r['media_type']=='audio_video' and needle in r['public_metadata'].get('title','').casefold()]
        # Fixture titles need not retain File: prefix/suffix.
        if key=='engine_short' and not found:
            found=[r['source_id'] for r in registry.values() if r['media_type']=='audio_video' and r['public_metadata'].get('title','').casefold()=='steam engine']
        if len(found)!=1:raise ValueError('Missing source or ambiguous source group: '+key)
        selected[key]=found[0]
    scope={'source_ids':list(registry),'split':'dev','corpus_id':corpus_id};rows=[]
    for index,(family,domain,keys,initial,question,targets,risk) in enumerate(SPECS,1):
        tid='avrb-dev-'+str(index).zfill(3)
        task={'schema_version':TASK_VERSION,'task_id':tid,'question':question,
              'initial_sources':[selected[k] for k in initial],
              'target_questions':[{'fact_id':'f'+str(j),'question':q} for j,q in enumerate(targets,1)],
              'corpus_id':corpus_id,'split':'dev','budget_recipe':{'max_tool_calls':20,'max_requested_av_seconds':120}}
        task=validate_task(task,scope)
        reference={'schema_version':REFERENCE_VERSION,'task_id':tid,'status':'pending','reviewer_ids':[],
                   'facts':[],'evidence_packages':[],'family':family}
        validate_reference(reference,task,registry,allow_pending=True)
        rows.append({'task':task,'reference':reference,'family':family,'domain':domain,
                     'candidate_source_ids':[selected[k] for k in keys],'suitability_risk':risk,
                     'status':'candidate_unreviewed','semantic_gold':False})
    return rows


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def write_jsonl(path,rows):
    with Path(path).open('x',encoding='utf-8') as f:
        for row in rows:f.write(canonical(row)+'\n')


def build_batch(registry_paths,out_dir,corpus_id='avrb-dev-candidates-20261010-v0'):
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    registry={};inputs=[]
    for path in map(Path,registry_paths):
        inputs.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line);sid=row['source_id']
            if sid in registry and any(registry[sid][k]!=row[k] for k in ('asset_sha256','asset_path','media_type')):raise ValueError('Conflicting original registry')
            registry[sid]=row
    rows=build_tasks(registry,corpus_id);docs=[]
    for row in registry.values():
        path=Path(row['asset_path']);actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=row['asset_sha256']:raise ValueError('Original asset hash mismatch')
        public=row['public_metadata']
        if row['media_type']=='webpage':text=row['body_text'];kind='web_body'
        else:text='\n'.join(f'{k}: {public[k]}' for k in ('title','description','creator','license','license_url') if k in public);kind='public_metadata'
        doc={k:row[k] for k in ('source_id','origin_url','asset_sha256','asset_path','media_type','duration_s','public_metadata')}
        doc.update(text=text,text_kind=kind);docs.append(doc)
    out.mkdir(parents=True,exist_ok=False)
    corpus=build_corpus(docs,out/'corpus',corpus_id,'dev')
    scope={'source_ids':corpus['source_ids'],'split':'dev','corpus_id':corpus_id}
    write_json(out/'scope.json',scope);write_jsonl(out/'sources.private.jsonl',list(registry.values()))
    write_jsonl(out/'tasks-public.jsonl',[r['task'] for r in rows]);write_jsonl(out/'references.private.jsonl',[r['reference'] for r in rows])
    write_json(out/'case-authoring.private.json',rows)
    # Conservatively reserve all sources already explored here; no random split of reused media.
    write_json(out/'split-manifest.private.json',{'source_splits':{s:'dev' for s in registry},
               'status':'development_reservation_only','train_source_count':0,'test_source_count':0,
               'reason':'Previously explored assets are reserved for dev; family-reviewed independent acquisition is required for train and test.'})
    receipts=[]
    for query in ('steam engine','violin cello','gong drum','waterfall bird'):
        search=search_sources(out/'corpus',query,scope,top_k=5);receipts.append({'tool':'search_sources','result':search})
        if search['results']:
            hit=search['results'][0];read=read_text(out/'corpus',hit['source_id'],hit['quote_start'],hit['quote_end'],scope)
            receipts.append({'tool':'read_text','result':read})
    write_json(out/'runtime-smoke.private.json',{'kind':'metadata_tool_connectivity_not_benchmark_attempt','receipts':receipts})
    write_json(out/'training-readiness.json',{'status':'waiting_for_independent_train_sources_and_teacher',
               'real_teacher_trajectory_count':0,'retained_sft_rows':0,'train_source_count':0,
               'token_label_audit':'not_run','teacher_configured':False,'training_started':False})
    summary={'schema_version':'av-research-batch.v0','created_at':datetime.now(timezone.utc).isoformat(),
             'case_count':len(rows),'family_counts':dict(Counter(r['family'] for r in rows)),
             'source_count':len(registry),'av_source_count':sum(r['media_type']=='audio_video' for r in registry.values()),
             'text_source_count':sum(r['media_type']=='webpage' for r in registry.values()),
             'source_seconds':sum(r['duration_s'] for r in registry.values() if r['media_type']=='audio_video'),
             'adjudicated_reference_count':0,'retained_sft_rows':0,'executed_metadata_tool_calls':len(receipts),
             'semantic_model_calls':0,'gpu_job_seconds':0,'corpus_hash':corpus['corpus_hash'],'input_manifests':inputs,
             'status':'candidate_batch_ready_for_user_review'}
    write_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',action='append',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();print(json.dumps(build_batch(a.sources,a.out_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

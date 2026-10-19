"""Render real online tasks and consolidate paired SFT for the original CPU audit."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from collections import Counter
from research.av_search.deepseek_backend import atomic_json,utcnow
from research.av_train.native_sft import digest,file_sha,SCHEMA,dataset_registration

def retained(out):
    return [r for p in sorted((out/'sources').glob('*/retained.private.json')) for r in json.loads(p.read_text())]

def render(out):
    out=Path(out).resolve();rows=retained(out);dest=out/'review';dest.mkdir(exist_ok=True);(dest/'media').mkdir(exist_ok=True)
    cards=[];index=[]
    for i,row in enumerate(rows,1):
        task=row['task'];source=Path(row['media']['source_path']);name=task['task_id']+'.mp4';target=dest/'media'/name
        if not target.exists():shutil.copyfile(source,target)
        if file_sha(target)!=row['media']['source_sha256']:raise ValueError('Review media bytes differ from actual teacher input')
        graph=json.loads(Path(row['graph_path']).read_text());pages=json.loads(Path(row['pages_path']).read_text())
        trace=json.loads(Path(row['trajectory_path']).read_text())
        data={'task':task,'metrics':row['metrics'],'graph':graph,'pages':pages,
              'teacher_final':trace['final_response_text'],'teacher_trace':[{'response':t['response_text'],'observation':t.get('tool_output')} for t in trace['turns']],
              'verification':row['verification'],'question_only':row['question_only'],'process_review':row['process_review'],
              'media_sha256':row['media']['source_sha256'],'duration_s':row['media']['duration_s'],'recording':row['recording']}
        atomic_json(dest/(task['task_id']+'.json'),data)
        e=html.escape;tid=task['task_id'];eid='v'+str(i)
        fact_rows=''.join('<tr><td>'+e(f['id'])+'</td><td>'+e(f['head']+' → '+f['relation']+' → '+f['tail'])+'</td><td><a target="_blank" rel="noreferrer" href="'+e(pages[f['source_id']]['url'],quote=True)+'">'+e(f['source_id'])+'</a><br>'+e(f['quote'])+'</td></tr>' for f in graph['facts'])
        cards.append('<article id="'+tid+'"><h2>'+str(i)+'. '+e(task['question'])+'</h2><p>操作：'+e(task['operation'])+'　声音作用（teacher 标注）：'+e(task.get('audio_role','unknown'))+'　输入：'+str(round(row['media']['duration_s'],1))+' 秒完整相关录制</p>'
          +'<video id="'+eid+'" preload="metadata" controls src="media/'+name+'"></video><div class="buttons"><button onclick="setMode(\''+eid+'\',\'av\')">原声画面</button><button onclick="setMode(\''+eid+'\',\'muted\')">静音画面</button><button onclick="setMode(\''+eid+'\',\'audio\')">只听声音</button></div>'
          +'<p><strong>参考答案：</strong>'+e(' / '.join(task['accepted_answers']))+'</p><p><strong>依赖：</strong>'+e(json.dumps(task['fact_paths'],ensure_ascii=False))+' → '+e(json.dumps(task['final_operation'],ensure_ascii=False))+'</p>'
          +'<p>'+e(task.get('modality_rationale',''))+'</p><details><summary>事实、URL 与实际抓取引文</summary><table><tr><th>ID</th><th>关系</th><th>网页证据</th></tr>'+fact_rows+'</table></details>'
          +'<details><summary>实际 teacher 解题与自动审核</summary><pre>'+e(json.dumps({k:data[k] for k in ['teacher_final','teacher_trace','verification','question_only','process_review']},ensure_ascii=False,indent=2))+'</pre></details>'
          +'<p>人工记录：<select data-task="'+tid+'" onchange="saveReview(this)"><option value="">未审核</option><option>保留；声音有帮助</option><option>保留；主要依赖画面</option><option>修改</option><option>删除</option></select> <a href="'+tid+'.json">完整结构 JSON</a></p></article>')
        index.append({'number':i,'task_id':tid,'operation':task['operation'],'duration_s':row['media']['duration_s'],'media_sha256':row['media']['source_sha256']})
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OneSearch-AV 在线组合构题审核</title><style>body{max-width:1100px;margin:36px auto;padding:0 22px;font:16px/1.7 system-ui;background:#f3f5f7;color:#172335}article{background:white;border:1px solid #dce1e7;border-radius:12px;padding:24px;margin:24px 0}video{width:100%;max-height:560px;background:#111;border-radius:8px}button,select{font:inherit;padding:7px 12px;margin:7px 6px 7px 0}table{width:100%;border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.buttons{display:flex;flex-wrap:wrap}.audio-mode{filter:brightness(0)}</style><h1>真实在线搜索 · 首批组合题审核</h1><p>构题和 teacher 解题均使用真实 DeepSeek Search/Fetch。以下均为自动审核保留的 silver 候选，尚未计为人工 gold。视频与实际 teacher 输入逐字节一致，支持完整播放、静音及黑屏只听声音。</p><p>当前保留 NUM 道。请重点检查：指代能否从输入定位、事实链是否必要、答案能否由网页事实推出、teacher 是否真的取得了证据。</p><button onclick="downloadReview()">导出人工审核记录</button>CARDS<script>
function setMode(id,mode){const v=document.getElementById(id);v.muted=mode==='muted';v.classList.toggle('audio-mode',mode==='audio');}
function saveReview(el){const key='onesearch-av-online-review-v1';const data=JSON.parse(localStorage.getItem(key)||'{}');data[el.dataset.task]={decision:el.value,at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(data));}
function downloadReview(){const b=new Blob([localStorage.getItem('onesearch-av-online-review-v1')||'{}'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='human-review.json';a.click();URL.revokeObjectURL(a.href);}
for(const el of document.querySelectorAll('select[data-task]')){const r=JSON.parse(localStorage.getItem('onesearch-av-online-review-v1')||'{}')[el.dataset.task];if(r)el.value=r.decision;}
</script></html>""".replace('NUM',str(len(rows))).replace('CARDS',''.join(cards))
    (dest/'index.html').write_text(page)
    atomic_json(dest/'index.json',{'at_utc':utcnow(),'retained':len(rows),'cases':index,'human_verified':0})
    return {'review_dir':str(dest),'questions':len(rows),'operations':dict(Counter(r['task']['operation'] for r in rows))}

def merge(out):
    out=Path(out).resolve();parts=json.loads((out/'exports.private.json').read_text());dest=out/'train-paired';dest.mkdir(exist_ok=False)
    full=[];answers=[];lineage=[];seen=set()
    for p in parts:
        src=Path(p['dataset_dir']);s=json.loads((src/'summary.json').read_text())
        t=json.loads((src/'trajectory.json').read_text());a=json.loads((src/'answer_only.json').read_text());l=json.loads((src/'lineage.private.json').read_text())
        if digest(t)!=s['trajectory_rows_sha256'] or digest(a)!=s['answer_rows_sha256'] or not len(t)==len(a)==len(l)==s['retained_count']:
            raise ValueError('Part export rows changed')
        for x,y,z in zip(t,a,l):
            if z['task_id'] in seen:raise ValueError('Duplicate task across exports')
            if digest(x)!=z['row_sha256'] or digest(y)!=z['answer_row_sha256'] or file_sha(z['trajectory_path'])!=z['trajectory_sha256']:
                raise ValueError('Export lineage mismatch')
            seen.add(z['task_id']);full.append(x);answers.append(y);lineage.append({**z,'row_index':len(full)-1})
    if not full:raise ValueError('No retained paired training rows')
    for name,data in [('trajectory.json',full),('answer_only.json',answers),('lineage.private.json',lineage)]:atomic_json(dest/name,data)
    atomic_json(dest/'dataset_info.json',{'native_omni_trajectory':dataset_registration('trajectory.json'),'native_omni_answer_only':dataset_registration('answer_only.json')})
    summary={'schema_version':SCHEMA,'audit_only':False,'source_split':'train','batch_dir':str(out/'batch'),'episodes_dir':str(out/'episodes'),
        'retained_count':len(full),'distinct_task_count':len(seen),'use_audio_in_video':True,'media_mode':'joint_av',
        'retained_pool_sha256':digest([r['case_id'] for r in lineage]),'trajectory_rows_sha256':digest(full),'answer_rows_sha256':digest(answers),
        'all_assistant_actions_supervised':True,'answers_are_actual_generated_responses':True,'train_on_prompt':False,'mask_history':False,
        'training_ready':False,'cpu_label_media_audit':'required','model_loaded':False,'optimizer_updates':0,'benchmark_gold_verified':False}
    atomic_json(dest/'summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['render','merge']);p.add_argument('--run-dir',required=True);a=p.parse_args()
    print(json.dumps((render if a.action=='render' else merge)(a.run_dir),ensure_ascii=False))
if __name__=='__main__':main()

"""Human review artifact: original-PTS contact sheets, media links and question decisions.

Publisher metadata is attribution, not semantic gold. The page exports reviewer
notes only; acceptance never silently creates benchmark answers or evidence packages.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote
from research.av_bench.contracts import FAMILIES
from .build_research_batch import write_json

LABELS={'source_discovery':'来源发现','event_localization':'事件实例定位',
        'cross_source_comparison':'跨源比较','research_composition':'研究组合'}

def esc(value):return html.escape(str(value),quote=True)


def source_card(source,initial=False):
    sid=source['source_id'];title=source['title']
    card=f'<article class="source"><div class="source-head"><b>{esc(title)}</b><span>{"起始素材" if initial else "待审候选"}</span></div>'
    if source.get('contact_sheet'):
        card+=f'<img loading="lazy" src="{esc(source["contact_sheet"])}" alt="{esc(title)}：三张原始时间轴抽帧">'
    if source.get('playback_url'):
        card+=f'<video controls preload="none" src="{esc(source["playback_url"])}"></video>'
    if source.get('body_text'):
        card+='<details><summary>冻结说明页正文（前 1500 字符）</summary><p>'+esc(source['body_text'][:1500])+'</p></details>'
    card+=f'<div class="provenance">{source.get("duration_s",0):.2f}s · <code>{esc(sid)}</code><br>{esc(source.get("rights_basis","unknown"))}</div>'
    card+=f'<a href="{esc(source["origin_url"])}" target="_blank" rel="noopener">打开原片与授权说明 ↗</a></article>'
    return card


def case_html(row,source_views):
    t=row['task'];tid=t['task_id'];family=row['family']
    targets=''.join('<li>'+esc(q['question'])+'</li>' for q in t['target_questions'])
    source_cards=[source_card(source,source['source_id'] in t['initial_sources']) for source in source_views]
    return f'''<section class="case" data-family="{esc(family)}" id="{esc(tid)}">
      <div class="case-label">{esc(tid)} · {esc(LABELS[family])} · {esc(row['domain'])}</div>
      <h2>{esc(t['question'])}</h2><ol>{targets}</ol>
      <div class="risk"><b>审核重点</b> {esc(row['suitability_risk'])}</div>
      <div class="grid">{''.join(source_cards)}</div>
      <details><summary>证据与评测要求</summary><p>最终提交有序来源、时间窗口、必要事实和逐事实引用。采用完整有效证据包、事件/区间匹配、事实准确率和 EBS；本题 reference 为空，暂不计性能分数。</p><p>所有模型搜索同一共享 corpus；下面展示的候选仅供作者审查，不提供给 actor。AV 派生描述索引尚未建成。</p></details>
      <div class="review"><label>题型决定 <select data-field="decision"><option value="pending">待审</option><option value="keep">保留方向</option><option value="rewrite">修改问题</option><option value="reject">淘汰</option></select></label>
      <label>声画需求 <select data-field="modality"><option value="pending">待核实</option><option value="joint_av">需要联合 AV</option><option value="audio">音频为主</option><option value="video">视觉为主</option><option value="text_shortcut">文字已足够</option></select></label>
      <textarea data-field="notes" placeholder="回放后的意见：合理问题、可核对事实、source_id 与时间区间、应增加的相似候选。"></textarea></div>
      </section>'''


def source_time(pts_seconds,origin):
    from research.av_bench.contracts import number
    mapping=origin.get('pts_to_source')
    if not isinstance(mapping,dict) or set(mapping)!={'scale','offset_s'}:
        raise ValueError('Missing exact original clock mapping')
    return number(pts_seconds,'PTS')*number(mapping['scale'],'clock scale')+number(mapping['offset_s'],'clock offset')


def frame_at_or_after(asset_path,target_source_time,origin):
    import io
    import re
    import subprocess
    from PIL import Image
    mapping=origin.get('pts_to_source',{})
    if mapping.get('scale')!=1:raise ValueError('Unsupported preview clock scale')
    original_pts=(target_source_time-mapping['offset_s'])/mapping['scale']
    command=['ffmpeg','-nostdin','-hide_banner','-copyts','-i',asset_path,'-map','0:v:0',
             '-vf',f'select=gte(t\\,{original_pts:.9f}),showinfo','-frames:v','1','-f','image2pipe','-c:v','png','pipe:1']
    process=subprocess.run(command,capture_output=True,timeout=60)
    timestamps=re.findall(r'pts_time:([0-9eE+.-]+)',process.stderr.decode(errors='replace'))
    if process.returncode or not process.stdout or not timestamps:raise ValueError('Original frame extraction failed')
    image=Image.open(io.BytesIO(process.stdout)).convert('RGB')
    return image,source_time(float(timestamps[0]),origin)


def contact_sheet(source,path):
    from PIL import Image,ImageDraw
    targets=[source['duration_s']*fraction for fraction in (.1,.5,.9)]
    chosen=[frame_at_or_after(source['asset_path'],t,source['timestamp_origin']) for t in targets]
    result=Image.new('RGB',(960,220),'#101923');draw=ImageDraw.Draw(result);times=[]
    for i,(im,seconds) in enumerate(chosen):
        im.thumbnail((312,184));result.paste(im,(i*320+(312-im.width)//2,(184-im.height)//2))
        draw.text((i*320+10,194),f'source time {seconds:.3f}s',fill='#eef6ff');times.append(seconds)
    result.save(path,quality=87)
    return times


def render_review(batch_dir,out_dir):
    root=Path(batch_dir);out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    rows=json.loads((root/'case-authoring.private.json').read_text());summary=json.loads((root/'summary.json').read_text())
    sources={r['source_id']:r for r in (json.loads(l) for l in (root/'sources.private.jsonl').read_text().splitlines() if l)}
    out.mkdir(parents=True,exist_ok=False);(out/'thumbs').mkdir();views={};frame_receipts=[]
    for sid,source in sources.items():
        if source['media_type']!='audio_video':continue
        path=out/'thumbs'/(sid+'.jpg');times=contact_sheet(source,path)
        title=source['public_metadata']['title'];name=title.removeprefix('File:')
        views[sid]={'source_id':sid,'title':title,'duration_s':source['duration_s'],'origin_url':source['origin_url'],
                    'rights_basis':source['rights_basis'],'contact_sheet':'thumbs/'+path.name,
                    'playback_url':'https://commons.wikimedia.org/wiki/Special:Redirect/file/'+quote(name,safe='')}
        frame_receipts.append({'source_id':sid,'asset_sha256':source['asset_sha256'],'actual_source_times_s':times,
                               'contact_sheet_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'timestamp_origin':source['timestamp_origin']})
    for sid,source in sources.items():
        if source['media_type']=='webpage':
            views[sid]={'source_id':sid,'title':source['public_metadata'].get('title','冻结说明页'),
                        'duration_s':0.,'origin_url':source['origin_url'],'rights_basis':source['rights_basis'],
                        'body_text':source['body_text']}
    def case_views(row):
        candidates=[views[s] for s in row['candidate_source_ids'] if s in views]
        origins={s['origin_url'] for s in candidates}
        return candidates+[view for sid,view in views.items() if sources[sid]['media_type']=='webpage' and view['origin_url'] in origins]
    sections=''.join(case_html(row,case_views(row)) for row in rows)
    buttons=f'<button class="active" data-family="all">全部 {len(rows)}</button>'+''.join(f'<button data-family="{f}">{LABELS[f]} {sum(r["family"]==f for r in rows)}</button>' for f in FAMILIES)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AV-ResearchBench · 16 cases 审核</title>
<style>
:root{color-scheme:light;font:15px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif;color:#193247;background:#eff4f7}body{margin:0}header{background:#122c40;color:#fff;padding:36px max(24px,calc((100vw - 1140px)/2))}header small{letter-spacing:.15em;color:#86d4dc}h1{font-size:34px;line-height:1.25;margin:10px 0}header p{max-width:850px;color:#c4dae8}main{max-width:1140px;margin:auto;padding:24px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.stat{padding:18px;background:white;border-radius:12px}.stat b{display:block;font-size:30px;color:#087d89}.bar{position:sticky;top:0;background:#eff4f7f0;backdrop-filter:blur(10px);padding:18px 0;display:flex;flex-wrap:wrap;gap:8px;z-index:5}button,select{font:inherit;padding:8px 12px;border:1px solid #b5c9d4;border-radius:7px;background:white;color:#193247;cursor:pointer}button.active,.export{background:#087d89;color:#fff;border-color:#087d89}.case{background:white;padding:26px;margin:0 0 24px;border:1px solid #dce7ed;border-radius:14px;scroll-margin-top:90px}.case-label{color:#087d89;font-weight:650;font-size:13px}h2{font-size:21px;line-height:1.55;margin:12px 0}.risk{background:#fff6e4;border-left:4px solid #d69e37;padding:10px 14px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.source{border:1px solid #dfebf0;border-radius:9px;overflow:hidden;padding:12px}.source-head{min-height:56px;display:flex;gap:8px;font-size:13px;line-height:1.4}.source-head b{word-break:break-word}.source-head span{white-space:nowrap;color:#087d89}.source img{width:100%;border-radius:5px;background:#101923}.source video{display:block;width:100%;max-height:270px;background:#101923;margin-top:8px}.provenance{font-size:12px;color:#617785;margin:8px 0;word-break:break-all}a{color:#087d89}.review{border-top:1px solid #dfebf0;margin-top:18px;padding-top:18px;display:flex;flex-wrap:wrap;gap:18px}.review label{display:flex;align-items:center;gap:8px}.review textarea{box-sizing:border-box;width:100%;min-height:100px;font:inherit;padding:12px;border:1px solid #c5d7df;border-radius:8px}details{margin-top:18px;color:#536b7b}summary{cursor:pointer}.hidden{display:none}footer{color:#536b7b;font-size:13px;padding:18px 0}@media(max-width:760px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:27px}.case{padding:18px}}
</style><header><small>AV RESEARCH · DEVELOPMENT REVIEW · V0</small><h1>16 个多源音视频研究候选</h1><p>先审查任务是否值得做、证据是否自然，再冻结 reference。缩略图来自实际原片；播放链接保留音轨。问题基于公开素材拟定，尚未核实其全部声画要求。</p></header><main>
<div class="stats"><div class="stat"><b>16</b>候选任务 · 四类各四题</div><div class="stat"><b>10</b>自然 AV 来源 · 约 299 秒</div><div class="stat"><b>0</b>已裁决 reference</div><div class="stat"><b>0</b>已保留 SFT 轨迹</div></div>
<p>本轮仅供 development。10 份素材和相关说明页已预留给 dev；train/test 需采集独立来源家族。视频直连 Commons；浏览器不支持 .ogv 时，可点击原片页面播放。导出的审核意见只保存在你的设备。</p>
<nav class="bar">BUTTONS<button class="export" id="export">导出审核意见 JSON</button></nav>SECTIONS
<footer>Metric: complete-package source recall · moment/instance coverage · necessary fact accuracy · EBS · inference cost. 这是一份作者候选审核包，未运行正式 benchmark/SFT。</footer></main>
<script>
const storeKey='avrb-review-20261010-v0';let previous={};try{previous=JSON.parse(localStorage.getItem(storeKey)||'{}')}catch(e){}
const cards=[...document.querySelectorAll('.case')];function snapshot(){return Object.fromEntries(cards.map(c=>[c.id,Object.fromEntries([...c.querySelectorAll('[data-field]')].map(x=>[x.dataset.field,x.value]))]))}
for(const card of cards)for(const el of card.querySelectorAll('[data-field]')){if(previous[card.id]?.[el.dataset.field])el.value=previous[card.id][el.dataset.field];el.addEventListener('input',()=>{try{localStorage.setItem(storeKey,JSON.stringify(snapshot()))}catch(e){}})}
for(const button of document.querySelectorAll('button[data-family]'))button.onclick=()=>{document.querySelectorAll('button[data-family]').forEach(b=>b.classList.toggle('active',b===button));cards.forEach(c=>c.classList.toggle('hidden',button.dataset.family!=='all'&&c.dataset.family!==button.dataset.family))};
document.querySelector('#export').onclick=()=>{const payload={schema_version:'avrb-user-review-notes.v1',created_at:new Date().toISOString(),status:'user_suitability_review_not_adjudicated_gold',decisions:snapshot()};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='AV-ResearchBench-review-notes.json';a.click();URL.revokeObjectURL(url)};
</script></html>'''.replace('BUTTONS',buttons).replace('SECTIONS',sections)
    is_train=summary.get('scope_split')=='train_candidate'
    count=len(rows);av_count=sum(r['media_type']=='audio_video' for r in sources.values())
    page=page.replace('16 cases 审核',f'{count} cases 审核').replace('16 个多源音视频研究候选',f'{count} 个'+('训练来源研究候选' if is_train else '多源音视频研究候选'))
    page=page.replace('<b>16</b>候选任务 · 四类各四题',f'<b>{count}</b>候选任务 · 四类任务')
    page=page.replace('<b>10</b>自然 AV 来源 · 约 299 秒',f'<b>{av_count}</b>自然 AV 来源 · 约 {summary["source_seconds"]:.0f} 秒')
    if is_train:
        page=page.replace('DEVELOPMENT REVIEW','TRAIN AUTHORING REVIEW')
        page=page.replace('本轮仅供 development。10 份素材和相关说明页已预留给 dev；train/test 需采集独立来源家族。',
                          '这批来自新采集来源，与 dev 原始文件无同字节重复；录制家族和语义标注尚待核实，train split 未冻结，尚不允许 teacher rollout。')
        page=page.replace('avrb-review-20261010-v0','avrb-train-review-20261010-v0')
        page=page.replace('所有模型搜索同一共享 corpus；下面展示的候选仅供作者审查，不提供给 actor。AV 派生描述索引尚未建成。',
                          '这是训练候选作者页；source-family 审查和必要事实完成后才输出 actor tasks。Teacher 不读取参考答案。')
    # Show the shared AV corpus, including natural candidates not assigned to a task.
    # These authoring cards are never sent to the actor as a shortlist.
    av_views=[view for sid,view in views.items() if sources[sid]['media_type']=='audio_video']
    gallery='<details class="corpus"><summary>查看共享语料：'+str(len(av_views))+' 份音视频与 '+str(sum(r['media_type']=='webpage' for r in sources.values()))+' 份说明页</summary><p>所有任务使用同一检索语料。这里展示全部原片，包括尚未归入任务组的潜在干扰素材；用途和独立录制关系仍待审核。</p><div class="grid">'+''.join(source_card(view) for view in av_views)+'</div></details>'
    page=page.replace('<nav class="bar">',gallery+'<nav class="bar">')
    review_id=hashlib.sha256(json.dumps({'task_ids':[r['task']['task_id'] for r in rows],
                                      'source_ids':sorted(sources)},sort_keys=True).encode()).hexdigest()[:16]
    page=page.replace("const storeKey='avrb-review-20261010-v0'","const storeKey='avrb-review-"+review_id+"'")
    page=page.replace("const storeKey='avrb-train-review-20261010-v0'","const storeKey='avrb-train-review-"+review_id+"'")
    page=page.replace('AV-ResearchBench-review-notes.json','AV-train-cases-review-notes.json' if is_train else 'AV-ResearchBench-review-notes.json')
    (out/'review.html').write_text(page,encoding='utf-8')
    write_json(out/'thumbnail-provenance.json',frame_receipts)
    write_json(out/'batch-summary.json',summary)
    write_json(out/'case-candidates.json',rows)
    return {'case_count':len(rows),'contact_sheet_count':sum('contact_sheet' in v for v in views.values()),'review_html':str(out/'review.html'),
            'status':'candidate_visualization_not_gold','created_at':datetime.now(timezone.utc).isoformat()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();print(json.dumps(render_review(a.batch_dir,a.out_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

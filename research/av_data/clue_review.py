"""Playable authoring review with blind clues, publisher references and export.

Reviewer records are downloaded drafts; browser choices do not become gold.
"""
import argparse,json
from pathlib import Path

def render_page(payload):
    data=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>OneSearch-AV 匿名片段题库审核</title><style>body{font:16px/1.7 system-ui;background:#f3f5f8;color:#192235;margin:0}main{max-width:1080px;margin:auto;padding:28px}article{padding:24px;border:1px solid #dde3ed;background:white;border-radius:12px;margin:20px 0}h1{line-height:1.35}h2{font-size:21px}video{display:block;width:100%;max-width:854px;max-height:520px;background:#101828;border-radius:8px;margin:12px 0}label{display:inline-block;margin-right:18px}select,input,textarea,button{font:inherit;padding:8px;border:1px solid #aebbd0;border-radius:6px}textarea{display:block;width:96%;min-height:60px;margin-top:12px}button{background:#234fc0;color:white;cursor:pointer}a{color:#225cc4}blockquote{border-left:3px solid #92a7cc;padding-left:14px;margin:14px 0}details{padding:12px;background:#f4f7fc;margin:14px 0}.tag{color:#58657d}.toolbar{position:sticky;top:0;background:#f3f5f8;padding:12px 0;z-index:2}</style><main><h1>OneSearch-AV：匿名音视频片段 → 搜索问答</h1><p>16 个 development 候选，8 个录制来源。新版提供约 9–15 秒的审核上下文和完整视频；旧版 2 秒片段仅用于最小试跑。先播放、判断题目是否自然，再展开发布者说明核对答案。</p><p>初始片段不提供原标题、URL 或目标来源 ID。候选答案来自发布者文本，仍待人工确认。审核时请特别判断：声音能否帮助辨认目标，以及题目是否真正需要搜索。</p><div class="toolbar"><label>审核人 <input id="reviewer" placeholder="填写审核标识"></label><button id="download">下载审核记录</button> <span id="count"></span></div><div id="cases"></div><p>记录保存在本浏览器并可下载；不会自动变成 benchmark gold。</p></main><script id="data" type="application/json">'''+data+'''</script><script>
const payload=JSON.parse(document.querySelector('#data').textContent),store='onesearch-av-anonymous-dev-review-'+payload.summary.tasks_sha256;let draft={};try{draft=JSON.parse(localStorage.getItem(store)||'{}')}catch{}
const h=(tag,text,parent)=>{const e=document.createElement(tag);if(text!==null)e.textContent=text;if(parent)parent.append(e);return e};
const save=()=>{localStorage.setItem(store,JSON.stringify(draft));document.querySelector('#count').textContent=`已填写 ${Object.values(draft).filter(v=>v.decision&&v.decision!=='pending').length}/16 题`};
for(const [i,c] of payload.cases.entries()){const id=c.task.task_id,record=draft[id]||{decision:'pending',audio_role:'pending',needs_search:'pending',note:''};draft[id]=record;
const box=h('article',null,document.querySelector('#cases'));h('h2',`${i+1}. ${c.task.question}`,box);h('p',`${c.reference_candidate.scenario} · 片段约 ${c.duration_s.toFixed(2)} 秒`,box).className='tag';
const v=h('video',null,box);v.controls=true;v.preload='auto';v.playsInline=true;v.poster=c.thumbnail;
const status=h('p','点击“播放 / 重播”查看；如果播放器黑屏，可切换格式或查看下方静帧。',box);status.setAttribute('aria-live','polite');
let full=false,format=c.webm?'webm':'mp4';
const choose=()=>{const src=full?(format==='webm'?c.full_webm:c.full_media):(format==='webm'?c.webm:c.media);if(!src)return;v.src=src;v.load();status.textContent=(full?'完整视频':'审核上下文')+' · '+(full?c.full_duration_s:c.duration_s).toFixed(2)+' 秒 · '+format.toUpperCase()};
const play=h('button','播放 / 重播',box);play.onclick=()=>{document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause()});if(v.ended)v.currentTime=0;v.play().catch(()=>{status.textContent='播放未成功，请切换格式或打开视频文件。'})};
for(const fmt of ['mp4','webm']){if(fmt==='webm'&&!c.webm)continue;const button=h('button','切换 '+fmt.toUpperCase(),box);button.style.margin='4px';button.onclick=()=>{format=fmt;choose()}};
if(c.full_media){const fullButton=h('button','播放完整视频（'+c.full_duration_s.toFixed(1)+' 秒）',box);fullButton.style.margin='4px';fullButton.onclick=()=>{full=true;choose();v.play().catch(()=>{})};const shortButton=h('button','返回审核片段',box);shortButton.onclick=()=>{full=false;choose()}}
v.onerror=()=>{status.textContent='播放器加载失败。请切换 MP4/WebM 或打开视频文件。'};
const open=h('a','单独打开 MP4',box);open.href=c.media;open.target='_blank';open.rel='noopener';open.style.margin='12px';
const still=h('details',null,box);h('summary','查看视频静帧（播放器黑屏时可先检查画面）',still);const image=h('img',null,still);image.src=c.thumbnail;image.alt='视频中的实际静帧';image.style.cssText='width:100%;max-width:854px;max-height:520px;object-fit:contain';
choose();
for(const [name,label,options] of [['decision','题目结论',[['pending','待审核'],['keep','保留'],['revise','修改'],['reject','删除']]],['audio_role','声音作用',[['pending','待判断'],['necessary','必要线索'],['helpful','辅助线索'],['redundant','没有额外帮助'],['unusable','音轨不适合']]],['needs_search','搜索必要性',[['pending','待判断'],['required','需要外部信息'],['perception_only','仅感知即可'],['ambiguous','问题不明确']]]]){const labelEl=h('label',label+' ',box),select=h('select',null,labelEl);for(const [value,text]of options){const option=h('option',text,select);option.value=value}select.value=record[name];select.onchange=()=>{record[name]=select.value;save()}}
const d=h('details',null,box);h('summary','展开候选答案与发布者说明',d);const link=h('a',c.publisher.public_metadata.title,d);link.href=c.publisher.origin_url;link.target='_blank';link.rel='noopener';h('blockquote',c.publisher.public_metadata.description||'',d);
for(const f of c.reference_candidate.answer_fields)h('p',f.field_id+'：'+f.accepted_values.join(' / '),d);
for(const q of c.reference_candidate.support_quotes)h('blockquote',q.field_id+' — '+q.quote,d);
const note=h('textarea',null,box);note.placeholder='修改建议、答案问题或声画线索判断';note.value=record.note;note.oninput=()=>{record.note=note.value;save()};}
save();document.querySelector('#download').onclick=()=>{const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer){alert('请填写审核人标识');return}const body={schema_version:'av-searchqa-authoring-review-draft.v1',reviewer_id:reviewer,created_at:new Date().toISOString(),benchmark_gold:false,task_batch_sha256:payload.summary.tasks_sha256,reviews:Object.entries(draft).map(([task_id,v])=>({task_id,...v}))};const blob=new Blob([JSON.stringify(body,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='onesearch-av-anonymous-dev-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    with out.open('x',encoding='utf-8') as f:f.write(render_page(json.loads(Path(a.cases).read_text())))
if __name__=='__main__':main()

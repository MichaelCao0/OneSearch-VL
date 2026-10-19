"""Build playable, source-bound train and benchmark candidate review packages."""
from pathlib import Path
import argparse,json,hashlib,subprocess,shutil
from concurrent.futures import ThreadPoolExecutor
from research.av_data.clue_review import render_page

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def probe(p):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],text=True))
 assert any(s['codec_type']=='video' for s in d['streams']),p
 assert any(s['codec_type']=='audio' for s in d['streams']),p
 return d
def render_review(payload):
 html=render_page(payload).replace('完整视频','来源上下文').replace('development 候选','benchmark development 候选')
 html=html.replace("${c.reference_candidate.scenario} · 审核上下文", "采集查询类别：${c.reference_candidate.scenario}（真实内容待核实） · 真实模型输入")
 html=html.replace('<p>初始片段不提供','<p>播放器默认展示真实模型输入；“来源上下文”提供附近 60–90 秒，部分短录制不足此时长。来源上下文只供审核。采集查询类别不是已核实的内容场景；若内容与标签不同，请在修改建议中记录真实类别。题目、答案及声音作用均待人工判断。</p><p>初始片段不提供')
 return html

def build(batch_paths,out_dir,*,split,limit=10,allowed_task_ids=None):
 out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=False);(out/'media').mkdir()
 entries={};seen_tasks=set()
 for bp in batch_paths:
  b=Path(bp);ts={x['task_id']:x for x in rows(b/'tasks-public.jsonl')}
  refs={x['task_id']:x for x in rows(b/'answer-candidates.private.jsonl')}
  li={x['task_id']:x for x in json.loads((b/'clue-lineage.private.json').read_text())}
  bindings={x['clue_id']:x['source'] for x in json.loads((b/'clue-bindings.private.json').read_text())}
  if (b/'sources.private.jsonl').exists():
   sources={x['source_id']:x for x in rows(b/'sources.private.jsonl')}
  else:
   from research.av_search.corpus import load_frozen
   corpus,_=load_frozen(b/'corpus');sources={x['source_id']:x for x in rows(corpus/'documents.jsonl')}
  for tid,t in ts.items():
   if tid in seen_tasks or allowed_task_ids is not None and tid not in allowed_task_ids:continue
   seen_tasks.add(tid);r=refs[tid];l=li[tid];s=sources[r['target_source_id']];v=bindings[t['initial_sources'][0]]
   entries[tid]={'task':t,'reference_candidate':r,'lineage':l,'publisher':s,'input':v,'batch':str(b)}
 grouped={}
 for tid,e in entries.items():grouped.setdefault(e['reference_candidate']['scenario'],[]).append((tid,e))
 for g in grouped.values():g.sort(key=lambda x:hashlib.sha256(('20261015'+x[0]).encode()).hexdigest())
 selected=[];seen=set()
 for i in range(max((len(x) for x in grouped.values()),default=0)):
  for _,g in sorted(grouped.items()):
   if i>=len(g):continue
   tid,e=g[i];origin=e['lineage']['acquisition_source_id']
   if origin in seen:continue
   seen.add(origin);selected.append((tid,e))
   if len(selected)>=limit:break
  if len(selected)>=limit:break
 if not selected:raise ValueError('No real candidate tasks for review')
 def one(pair):
  tid,e=pair;folder=out/'media'/tid;folder.mkdir()
  inp=Path(e['input']['asset_path']);assert sha(inp)==e['input']['asset_sha256'];ip=probe(inp)
  shutil.copyfile(inp,folder/'input.mp4')
  parent=Path(e['publisher']['asset_path']);assert sha(parent)==e['publisher']['asset_sha256']
  pp=probe(parent);duration=float(pp['format']['duration']);query=e['lineage']['interval']
  start=max(0.,float(query['start_s'])-15.);end=min(duration,max(start+60.,float(query['end_s'])+15.));end=min(end,start+90.)
  subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(start),'-i',str(parent),'-t',str(end-start),
   '-map','0:v:0','-map','0:a:0','-vf','scale=min(854\\,iw):-2,fps=15','-c:v','libx264','-preset','veryfast','-crf','26','-pix_fmt','yuv420p',
   '-c:a','aac','-b:a','96k','-movflags','+faststart','-threads','2',str(folder/'context.mp4')],check=True,capture_output=True,timeout=300)
  cp=probe(folder/'context.mp4')
  subprocess.run(['ffmpeg','-v','error','-nostdin','-ss','0.5','-i',str(inp),'-frames:v','1','-vf','scale=640:-2','-threads','1',str(folder/'poster.jpg')],check=True,capture_output=True,timeout=60)
  prefix='media/'+tid+'/'
  case={'task':e['task'],'reference_candidate':e['reference_candidate'],'publisher':e['publisher'],
   'media':prefix+'input.mp4','full_media':prefix+'context.mp4','thumbnail':prefix+'poster.jpg',
   'duration_s':float(ip['format']['duration']),'actual_input_duration_s':float(ip['format']['duration']),
   'full_duration_s':float(cp['format']['duration']),
   'source_context_interval':{'start_s':start,'end_s':end},'original_duration_s':duration}
  receipt={'task_id':tid,'scenario':e['reference_candidate']['scenario'],'source_id':e['lineage']['acquisition_source_id'],
   'batch_dir':e['batch'],'input_source_path':str(inp),'input_sha256':sha(inp),'parent_source_path':str(parent),'parent_sha256':sha(parent),
   'query_interval_on_parent':query,'context_interval_on_parent':case['source_context_interval'],
   'actual_input_duration_s':case['duration_s'],'context_duration_s':case['full_duration_s'],
   'files':[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in folder.iterdir()],
   'video_codec':next(s['codec_name'] for s in ip['streams'] if s['codec_type']=='video'),
   'audio_codec':next(s['codec_name'] for s in ip['streams'] if s['codec_type']=='audio'),'human_gold':False}
  print(json.dumps({'rendered':tid,'scene':receipt['scenario'],'input_seconds':case['duration_s'],'context_seconds':case['full_duration_s']}),flush=True)
  return case,receipt
 with ThreadPoolExecutor(max_workers=3) as pool:built=list(pool.map(one,selected))
 cases=[x[0] for x in built];tasks=[x['task'] for x in cases];refs=[x['reference_candidate'] for x in cases]
 task_file=out/'tasks-public.jsonl';ref_file=out/'answer-candidates.private.jsonl'
 task_file.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks));ref_file.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in refs))
 payload={'summary':{'split':split,'parent_recordings':len(cases),'tasks_sha256':sha(task_file),'candidate_references_sha256':sha(ref_file),'benchmark_gold':0},'cases':cases}
 (out/'payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 html=render_review(payload)
 (out/'index.html').write_text(html)
 manifest={'schema':'onesearch-av-review-package.v1','split':split,'cases':len(cases),'human_gold':0,'selection':'deterministic acquisition-query-category round-robin, unique recordings, no student score selection','scenario_labels':'unverified acquisition query categories, not validated content domains',
  'receipts':[x[1] for x in built],'task_batch_sha256':sha(task_file),'candidate_references_sha256':sha(ref_file),
  'review_drafts_are_gold':False,'sources_read_only':True,'paid_api_calls_by_renderer':0}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'review_dir':str(out),'cases':len(cases),'bytes':sum(p.stat().st_size for p in out.glob('**/*') if p.is_file())},ensure_ascii=False))
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--batch',action='append',required=True);p.add_argument('--out',required=True);p.add_argument('--split',choices=['train','dev'],required=True);p.add_argument('--limit',type=int,default=10)
 a=p.parse_args();build(a.batch,a.out,split=a.split,limit=a.limit)

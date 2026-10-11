"""Replay executed AV observations on byte-verified, decode-equivalent query clips.

Exact parent windows only; source intervals are rebased to query time. No fresh
observer inference occurs and changed/partial crops cannot reuse descriptions.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from research.av_bench.contracts import json_copy,number
from research.av_data.media_clock import sha256_file
from research.av_data.build_research_batch import write_json
from .searchqa import SmokeObserverReplay

def rebase_observation(observation,offset_s,duration_s):
    offset=number(offset_s,'offset');duration=number(duration_s,'duration')
    if offset<0 or duration<=0:raise ValueError('Valid crop interval required')
    result=json_copy(observation)
    for event in result.get('events',[]):
        w=event['source_interval'];start=number(w['start_s'],'event start');end=number(w['end_s'],'event end')
        if not offset<=start<end<=offset+duration:raise ValueError('Observation interval outside exact query crop')
        w['start_s']=start-offset;w['end_s']=end-offset
    return result

def same_decoded_media(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b)=={'video','audio'} and all(isinstance(a[k],str) and a[k] and a[k]==b[k] for k in a)

def decoded_signature(path):
    result={}
    for kind,codec in [('video','rawvideo'),('audio','pcm_f32le')]:
        stream='v:0' if kind=='video' else 'a:0'
        data=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:'+stream,'-c',codec,
          '-f','framemd5','-'],timeout=90)
        # Ignore provenance comments but retain timebase, PTS, sample/frame
        # lengths and every decoded payload checksum.
        lines=[line for line in data.splitlines() if not line.startswith((b'#software:',b'#format:',b'#version:',b'#hash:'))]
        result[kind]=hashlib.sha256(b'\n'.join(lines)).hexdigest()
    return result

def prepare_replay_admission(batch_dir,smoke_dir,out_dir):
    batch=Path(batch_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    replay=SmokeObserverReplay(smoke_dir)
    sources={s['source_id']:s for s in map(json.loads,(batch/'sources.private.jsonl').read_text().splitlines())}
    bindings={b['clue_id']:b for b in json.loads((batch/'clue-bindings.private.json').read_text())}
    admitted=[];rejected=[]
    for lineage in json.loads((batch/'clue-lineage.private.json').read_text()):
        w=lineage['original_interval'];key=(lineage['parent_source_id'],w['start_s'],w['end_s']);b=bindings[lineage['clue_id']]
        if key not in replay.records:
            rejected.append({'task_id':lineage['task_id'],'reason':'no_exact_executed_parent_observation'});continue
        record_path=replay.records[key];record=json.loads(record_path.read_text());source=b['source'];parent=sources[key[0]]
        if sha256_file(source['asset_path'])!=lineage['clip_sha256'] or source['asset_sha256']!=lineage['clip_sha256']:raise ValueError('Query clip hash changed')
        if parent['asset_sha256']!=lineage['parent_sha256'] or parent['asset_sha256']!=record['input']['asset_sha256']:raise ValueError('Parent identity mismatch')
        derived=json.loads((batch/'query-derived'/lineage['task_id']/'derivation.private.json').read_text())
        if derived['parent_source_id']!=key[0] or derived['mapping']!=lineage['mapping'] or derived['original_interval']!=w:raise ValueError('Query derivation mismatch')
        signatures={'query':decoded_signature(source['asset_path']),'executed_observer_input':decoded_signature(record['input']['clip_path'])}
        if not same_decoded_media(signatures['query'],signatures['executed_observer_input']):
            rejected.append({'task_id':lineage['task_id'],'reason':'decoded_media_not_identical','signatures':signatures});continue
        rebase_observation(record['structured_observation'],w['start_s'],w['end_s']-w['start_s'])
        admitted.append({'task_id':lineage['task_id'],'clue_id':lineage['clue_id'],'source':source,'parent_source':parent,
          'parent_window':w,'record_path':str(record_path),'record_sha256':sha256_file(record_path),'signatures':signatures,
          'query_duration_s':w['end_s']-w['start_s']})
    write_json(out/'admission.private.json',{'schema_version':'av-exact-query-observer-replay-admission.v1','admitted':admitted,'rejected':rejected,
      'smoke_dir':str(smoke_dir),'batch_dir':str(batch),'fresh_observer_executed':False,'semantic_verified':False})
    print(json.dumps({'admitted':len(admitted),'rejected':len(rejected),'fresh_observer_executed':False}),flush=True)
    return admitted

class MappedClueReplay:
    max_clip_seconds=2.
    def __init__(self,admission_file):
        config=json.loads(Path(admission_file).read_text());self.parent_replay=SmokeObserverReplay(config['smoke_dir']);self.admissions={a['source']['source_id']:a for a in config['admitted']}
        if not self.admissions:raise ValueError('No decode-equivalent query replay admissions')
        self.available_windows=json_copy(self.parent_replay.available_windows)
    def __call__(self,source,start_s,end_s,out_dir):
        key=source['source_id']
        if key not in self.admissions:return self.parent_replay(source,start_s,end_s,out_dir)
        a=self.admissions[key]
        if start_s!=0. or end_s!=a['query_duration_s']:raise ValueError('Query replay requires the exact whole admitted crop')
        if source['asset_sha256']!=a['source']['asset_sha256'] or sha256_file(source['asset_path'])!=a['source']['asset_sha256']:raise ValueError('Query asset changed')
        if sha256_file(a['record_path'])!=a['record_sha256']:raise ValueError('Observer record changed')
        w=a['parent_window'];receipt=self.parent_replay(a['parent_source'],w['start_s'],w['end_s'],out_dir)
        receipt['source_id']=key;receipt['asset_sha256']=source['asset_sha256'];receipt['original_interval']={'start_s':0.,'end_s':a['query_duration_s']}
        receipt['observation']=rebase_observation(receipt['observation'],w['start_s'],a['query_duration_s'])
        receipt['replay_basis']='exact_parent_crop_and_verified_identical_decoded_av; query_clock_rebased'
        return receipt

"""Versioned full-duration AV compression before inline teacher production.

Keeps real recorded audio and the common timeline. No content truncation,
caption synthesis, frame-only input or replacement of the parent dataset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'onesearch_vl'))
from onesearch_infer.av_io import make_native_av_handle,crop_native_av,probe_av
from .native_sft import file_sha

RECIPE={'schema':'native-teacher-media.v1','max_width':640,'video_codec':'h264','video_crf':28,
        'video_preset':'veryfast','video_maxrate':'600k','video_bufsize':'1200k','audio_codec':'aac','audio_bitrate':'64k','fps_mode':'vfr',
        'max_output_bytes':3000000,'duration_tolerance_s':.15,'truncated':False}


def normalize_media(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists() or source==destination:raise FileExistsError('Normalized media must use a new destination')
    handle=make_native_av_handle(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='teacher-full-av-',dir=destination.parent) as root:
        canonical=Path(root)/'canonical.mp4';encoded=Path(root)/'encoded.mp4'
        # Existing AV helper normalizes nonzero origins and delayed audio on a
        # common timeline, including legitimate timestamp-alignment padding.
        crop_native_av(handle,0,handle['duration'],canonical,timeout=300)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(canonical),'-map','0:v:0','-map','0:a:0?',
                        '-vf',"scale=w='min(640,iw)':h=-2",'-c:v','libx264','-preset','veryfast','-crf','28',
                        '-maxrate','600k','-bufsize','1200k','-threads','2','-c:a','aac','-b:a','64k','-pix_fmt','yuv420p','-fps_mode','vfr',
                        '-map_metadata','-1','-movflags','+faststart',str(encoded)],
                        check=True,capture_output=True,timeout=300)
        info=probe_av(encoded)
        if info['has_audio']!=handle['has_audio'] or abs(info['duration']-handle['duration'])>RECIPE['duration_tolerance_s']:
            raise ValueError('Full duration or recorded audio was not preserved')
        if encoded.stat().st_size>RECIPE['max_output_bytes']:
            raise ValueError('Normalized teacher media exceeds byte target; preserve failure without truncation')
        encoded.replace(destination)
    return {'source_path':str(source),'source_sha256':file_sha(source),'source_bytes':source.stat().st_size,
            'normalized_path':str(destination),'normalized_sha256':file_sha(destination),'normalized_bytes':destination.stat().st_size,
            'source_duration_s':handle['duration'],'normalized_duration_s':info['duration'],'has_audio':info['has_audio'],
            'source_origin_s':handle['source_origin_s'],'truncated':False,'normalization_recipe':RECIPE,
            'audio_operation':'recorded_track_only; common-origin padding preserves delayed onset'}


def prepare_teacher_batch(batch_dir,out_dir):
    parent=Path(batch_dir).resolve();out=Path(out_dir).resolve()
    scope=json.loads((parent/'scope.json').read_text())
    tasks=[json.loads(l) for l in (parent/'tasks-public.jsonl').read_text().splitlines() if l]
    if scope.get('split')!='train' or not tasks or any(t.get('split')!='train' for t in tasks):
        raise ValueError('Teacher media batch requires train-only tasks')
    bindings=json.loads((parent/'clue-bindings.private.json').read_text())
    for row in bindings:
        if file_sha(row['source']['asset_path'])!=row['source']['asset_sha256']:
            raise ValueError('Parent clue media changed')
    out.mkdir(parents=True,exist_ok=False);(out/'media').mkdir()
    receipts=[];prepared=[];cache={}
    for index,row in enumerate(bindings):
        source=row['source'];sha=source['asset_sha256']
        if sha not in cache:
            cache[sha]=normalize_media(source['asset_path'],out/'media'/f'input-av-{index:04d}.mp4')
        receipt=cache[sha];receipts.append({'clue_id':row['clue_id'],**receipt})
        prepared.append({**row,'source':{**source,'asset_path':receipt['normalized_path'],
                         'asset_sha256':receipt['normalized_sha256'],'parent_asset_sha256':sha}})
    for name in ('scope.json','tasks-public.jsonl','answer-candidates.private.jsonl'):
        if (parent/name).exists():shutil.copyfile(parent/name,out/name)
    (out/'clue-bindings.private.json').write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
    report={'schema':'native-teacher-prepared-batch.v1','parent_batch':str(parent),'prepared_clues':len(prepared),
            'distinct_parent_media':len(cache),'task_count':len(tasks),'source_family_identity_changed':False,
            'parent_task_file_sha256':file_sha(parent/'tasks-public.jsonl'),'normalization_recipe':RECIPE,'receipts':receipts}
    (out/'normalization.private.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch-dir',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();report=prepare_teacher_batch(args.batch_dir,args.out_dir)
    print(json.dumps({k:v for k,v in report.items() if k!='receipts'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

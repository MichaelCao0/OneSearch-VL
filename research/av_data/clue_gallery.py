"""Generate reviewer-only playable previews from the actual anonymous query assets."""
import argparse,json,subprocess,tarfile
from pathlib import Path
from research.av_data.media_clock import sha256_file
from research.av_data.build_research_batch import write_json
from research.av_data.clue_review import render_page

def transcode(path,target,duration,webm=False):
    common=['ffmpeg','-nostdin','-v','error','-i',str(path),'-t',str(duration),'-map','0:v:0','-map','0:a:0','-map_metadata','-1','-vf',"scale=w='min(854,iw)':h=-2,setsar=1",'-threads','2']
    codec=(['-c:v','libvpx','-deadline','realtime','-cpu-used','8','-b:v','650k','-c:a','libopus','-b:a','64k'] if webm else ['-c:v','libx264','-profile:v','baseline','-pix_fmt','yuv420p','-preset','veryfast','-crf','24','-c:a','aac','-b:a','96k','-movflags','+faststart'])
    subprocess.run(common+codec+[str(target)],check=True,timeout=300)

def build(batch_dir,out_dir):
    batch=Path(batch_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    for name in ('media','thumbnails'):(out/name).mkdir()
    summary=json.loads((batch/'summary.json').read_text())
    tasks=list(map(json.loads,(batch/'tasks-public.jsonl').read_text().splitlines()))
    refs={r['task_id']:r for r in map(json.loads,(batch/'answer-candidates.private.jsonl').read_text().splitlines())}
    sources={r['source_id']:r for r in map(json.loads,(batch/'sources.private.jsonl').read_text().splitlines())}
    bindings={r['clue_id']:r['source'] for r in json.loads((batch/'clue-bindings.private.json').read_text())}
    assets={};cases=[]
    for task in tasks:
        ref=refs[task['task_id']];query=bindings[task['initial_sources'][0]];parent=sources[ref['target_source_id']]
        key=query['asset_sha256']
        if key not in assets:
            assert query['public_metadata']=={} and sha256_file(query['asset_path'])==key
            assert sha256_file(parent['asset_path'])==parent['asset_sha256']
            name=f'recording-{len(assets)+1:02d}';duration=query['duration_s'];full_duration=parent['duration_s']
            for label,source,length in [('input',query,duration),('full',parent,full_duration)]:
                for suffix in ('mp4','webm'):transcode(source['asset_path'],out/'media'/f'{name}-{label}.{suffix}',length,suffix=='webm')
            thumb=out/'thumbnails'/f'{name}.jpg'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(min(1.,duration/2)),'-i',str(out/'media'/f'{name}-input.mp4'),'-frames:v','1',str(thumb)],check=True,timeout=60)
            assets[key]={'media':f'media/{name}-input.mp4','webm':f'media/{name}-input.webm','full_media':f'media/{name}-full.mp4','full_webm':f'media/{name}-full.webm','thumbnail':f'thumbnails/{name}.jpg','duration_s':duration,'actual_input_duration_s':duration,'full_duration_s':full_duration}
            print(json.dumps({'prepared_recording':len(assets),'input_seconds':duration,'full_seconds':full_duration}),flush=True)
        cases.append({'task':task,'reference_candidate':ref,'publisher':{'origin_url':parent['origin_url'],'public_metadata':parent['public_metadata']},**assets[key]})
    payload={'schema_version':'av-anonymous-query-review.v3','summary':summary,'cases':cases,'review_only':True}
    write_json(out/'cases.json',payload);(out/'index.html').write_text(render_page(payload))
    files=[]
    for file in sorted((out/'media').glob('*')):
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(file)],text=True))
        assert {s['codec_type'] for s in probe['streams']}=={'audio','video'}
        files.append({'file':str(file.relative_to(out)),'sha256':sha256_file(file),'duration_s':float(probe['format']['duration']),'codecs':[s['codec_name'] for s in probe['streams']]})
    audit={'cases':len(cases),'unique_recordings':len(assets),'preview_files':files,'human_verified':False,'browser_playback_verified':False,'actual_query_previews':True,'tasks_sha256':sha256_file(batch/'tasks-public.jsonl')}
    write_json(out/'preview-audit.json',audit)
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='review')
    print(json.dumps({'cases':len(cases),'recordings':len(assets),'archive':str(archive),'archive_sha256':sha256_file(archive)}),flush=True)
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();build(a.batch_dir,a.out_dir)
if __name__=='__main__':main()

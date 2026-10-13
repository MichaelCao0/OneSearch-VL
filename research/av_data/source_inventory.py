"""Versioned CPU inventory and candidate recording families, including held-out media.

Families are duplicate-risk groups, not semantic proof of independent recordings.
Publisher text plus four real frame fingerprints are used without model calls.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,difflib,hashlib,html,json,re,subprocess,time
from research.av_data.clue_batch import origin_key
from research.av_train.native_sft import file_sha
from research.av_train.teacher_rollouts import atomic_json


def norm(value):
    return re.sub(r'[^\w]+',' ',html.unescape(re.sub('<[^>]+>',' ',value)).casefold()).strip()


def group_recordings(rows):
    rows=sorted(rows,key=lambda r:r['source_id']);pairs=[];parents={r['source_id']:r['source_id'] for r in rows}
    def root(x):
        while parents[x]!=x:x=parents[x]
        return x
    keys={r['source_id']:(norm(r['title']),norm(r['creator']),origin_key(r['origin_url']),[int(x['dhash64'],16) for x in r['video_fingerprints']]) for r in rows}
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            reasons=[];similarity=None
            at,ac,au,af=keys[a['source_id']];bt,bc,bu,bf=keys[b['source_id']]
            if a['asset_sha256']==b['asset_sha256']:reasons.append('same_asset_sha256')
            if au==bu:reasons.append('same_public_page')
            if ac and ac==bc:
                similarity=difflib.SequenceMatcher(None,at,bt).ratio()
                if similarity>=.62:reasons.append('same_creator_similar_title')
            distance=None
            if len(af)==len(bf)==4:
                distance=sum((x^y).bit_count() for x,y in zip(af,bf))/4
                if distance<=8:reasons.append('aligned_thumbnail_similarity')
            if reasons:
                if similarity is None:similarity=difflib.SequenceMatcher(None,at,bt).ratio()
                pairs.append({'source_a':a['source_id'],'source_b':b['source_id'],'reasons':reasons,'title_similarity':similarity,
                    'mean_aligned_dhash_hamming':distance,'status':'candidate_family_requires_review'})
                ra,rb=root(a['source_id']),root(b['source_id']);parents[max(ra,rb)]=min(ra,rb)
    clusters={}
    for row in rows:clusters.setdefault(root(row['source_id']),[]).append(row['source_id'])
    groups=[{'candidate_family_id':'family_'+hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:24],
        'source_ids':ids,'independence_verified':False} for _,ids in sorted(clusters.items())]
    return pairs,groups


def build_inventory(manifests,reserved_registries,out_dir,workers=4,*,cached_inventories=()):
    if not 1<=workers<=8:raise ValueError('Bounded CPU worker count required')
    paths=[Path(p).resolve() for p in manifests]+[Path(p).resolve() for p in reserved_registries]
    seeds={}
    for path in manifests:
        for s in json.loads(Path(path).read_text())['sources']:
            if s.get('status')=='candidate_unreviewed_av' and s.get('description_page_html_path'):
                seeds[s['source_id']]={**s,'reserved':False}
    for path in reserved_registries:
        for s in map(json.loads,Path(path).read_text().splitlines()):
            if s['media_type']=='audio_video':
                seeds[s['source_id']]={'source_id':s['source_id'],'page_id':None,'asset_path':s['asset_path'],'sha256':s['asset_sha256'],
                    'duration_s':s['duration_s'],'origin_url':s['origin_url'],'public_metadata':s['public_metadata'],
                    'candidate_domain':'reserved_dev_test','reserved':True}
    if not seeds or len(seeds)>10000:raise ValueError('One to 10000 usable inventory inputs required')
    cached={}
    for inventory in cached_inventories:
        for row in json.loads(Path(inventory).read_text())['rows']:
            if row.get('sha256_verified') is True and len(row.get('video_fingerprints',[]))==4 and not row.get('fingerprint_errors'):
                cached[(row['asset_sha256'],row['duration_s'])]=row
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def one(s):
        path=Path(s['asset_path']);sha=file_sha(path)
        if sha!=s['sha256']:raise ValueError('Inventory input hash differs from acquisition/reservation')
        prior=cached.get((sha,s['duration_s']))
        fps=list(prior['video_fingerprints']) if prior else [];errors=[]
        for ratio in (() if prior else (.1,.35,.65,.9)):
            ss=max(0.,s['duration_s']*ratio)
            try:
                raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-ss',str(ss),'-i',str(path),'-map','0:v:0',
                    '-frames:v','1','-vf','scale=9:8,format=gray','-f','rawvideo','-'],timeout=30)
                if len(raw)!=72:raise ValueError('thumbnail_size_'+str(len(raw)))
                value=0
                for y in range(8):
                    for x in range(8):value=(value<<1)|int(raw[y*9+x]>raw[y*9+x+1])
                fps.append({'sample_s':ss,'dhash64':f'{value:016x}'})
            except (subprocess.SubprocessError,ValueError) as exc:errors.append(type(exc).__name__)
        md=s['public_metadata']
        return {'source_id':s['source_id'],'page_id':s['page_id'],'asset_sha256':sha,'sha256_verified':True,
            'title':md.get('title',''),'description':md.get('description',''),'creator':md.get('creator',''),
            'license':md.get('license',''),'origin_url':s['origin_url'],'domain':s['candidate_domain'].split('_')[0],
            'duration_s':s['duration_s'],'video_fingerprints':fps,'fingerprint_errors':errors,'reserved':s['reserved'],
            'semantic_review':'pending','family_review':'pending','fingerprints_from_verified_cache':bool(prior)}
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(one,s) for s in seeds.values()]):
            rows.append(f.result());atomic_json(out/'progress.json',{'completed':len(rows),'target':len(seeds),'wall_seconds':time.monotonic()-start})
    rows.sort(key=lambda r:r['source_id']);pairs,groups=group_recordings(rows)
    atomic_json(out/'inventory.private.json',{'rows':rows,'candidate_pairs':pairs,'candidate_families':groups})
    report={'schema':'av-source-inventory.v2','input_files_sha256':{str(p):file_sha(p) for p in paths},
        'total_asset_records':len(rows),'reserved_asset_records':sum(r['reserved'] for r in rows),'all_asset_hashes_verified':True,
        'exact_unique_asset_hashes':len({r['asset_sha256'] for r in rows}),'candidate_family_groups':len(groups),
        'multi_asset_candidate_families':sum(len(g['source_ids'])>1 for g in groups),'candidate_family_pairs':len(pairs),
        'fingerprint_error_sources':sum(bool(r['fingerprint_errors']) for r in rows),
        'cached_fingerprint_records':sum(r['fingerprints_from_verified_cache'] for r in rows),
        'cached_inventory_files_sha256':{str(Path(p).resolve()):file_sha(p) for p in cached_inventories},
        'sampled_video_fingerprints':sum(len(r['video_fingerprints']) for r in rows),'wall_seconds':time.monotonic()-start,
        'by_domain':{d:sum(r['domain']==d for r in rows) for d in sorted({r['domain'] for r in rows})},
        'independent_recordings_verified':False,'thumbnail_similarity_does_not_prove_same_recording':True,'human_gold':0}
    atomic_json(out/'summary.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',action='append',required=True)
    p.add_argument('--reserved-registry',action='append',default=[]);p.add_argument('--out-dir',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();print(json.dumps(build_inventory(a.manifest,a.reserved_registry,a.out_dir,a.workers),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Bounded, create-only Wikimedia Commons AV seed acquisition (not task gold).

Run with Pixi from the repository root. Original bytes, public publisher metadata,
page API snapshots and actual ffprobe/audio-energy evidence are kept separately.
No metadata assertion is promoted to an observed fact or human-reviewed relation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import requests

API = 'https://commons.wikimedia.org/w/api.php'
QUERIES = [
    ('technical_mechanisms', 'steam engine'),
    ('field_performance', 'street musician'),
    ('environment_animals', 'waterfall'),
    ('technical_mechanisms', 'printing press'),
    ('field_performance', 'drumming'),
    ('environment_animals', 'frog pond'),
    ('technical_mechanisms', 'tractor engine'),
    ('environment_animals', 'bird feeding tree'),
    ('field_performance', 'bell ringing'),
    ('technical_mechanisms', 'water wheel'),
    ('environment_animals', 'rain forest'),
    ('field_performance', 'street performance'),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.ignore = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}: self.ignore += 1
        elif tag in {'p', 'br', 'div', 'li', 'tr'}: self.parts.append(' ')
    def handle_endtag(self, tag):
        if tag in {'script', 'style'} and self.ignore: self.ignore -= 1
        elif tag in {'p', 'div', 'li', 'tr'}: self.parts.append(' ')
    def handle_data(self, data):
        if not self.ignore: self.parts.append(data)


def public_text(value):
    p = _TextOnly(); p.feed(str(value or ''))
    return ' '.join(html.unescape(''.join(p.parts)).split())


def original_url(value):
    parts = urlsplit(value)
    # Media URL parameters returned here are only Commons imageinfo tracking.
    if parts.hostname != 'upload.wikimedia.org': raise ValueError('Unexpected media host')
    return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))


def write_json_once(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def select_candidates(pages, seen, max_asset_bytes, limit=3):
    candidates, refused = [], []
    for page in sorted(pages.values(), key=lambda p: (p.get('index', 10**9), p.get('pageid', 0))):
        if page['pageid'] in seen: continue
        size = page.get('imageinfo', [{}])[0].get('size', 0)
        if not isinstance(size, int) or size <= 0 or size > max_asset_bytes:
            refused.append(page)
        elif len(candidates) < limit:
            candidates.append(page)
    return candidates, refused


class AcquisitionBudget:
    def __init__(self, max_downloads=25, max_bytes=2_000_000_000):
        self.max_downloads, self.max_bytes = max_downloads, max_bytes
        self.download_attempts, self.bytes_received = 0, 0
    def begin(self, declared_bytes):
        if self.download_attempts >= self.max_downloads: raise ValueError('download_attempt_cap')
        if declared_bytes <= 0 or declared_bytes > self.max_bytes - self.bytes_received:
            raise ValueError('byte_cap')
        self.download_attempts += 1
    def account(self, actual_bytes):
        if actual_bytes < 0 or actual_bytes > self.max_bytes - self.bytes_received:
            raise ValueError('byte_cap')
        self.bytes_received += actual_bytes


def ffprobe(path):
    cmd = ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode: raise RuntimeError('ffprobe: '+result.stderr[-2000:])
    return json.loads(result.stdout)


def audio_energy(path):
    cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-threads', '1', '-i', str(path),
           '-map', '0:a:0', '-vn', '-af', 'volumedetect', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode: raise RuntimeError('ffmpeg audio analysis: '+result.stderr[-1500:])
    means = re.findall(r'mean_volume:\s*(-?[\d.]+|-inf) dB', result.stderr)
    maxima = re.findall(r'max_volume:\s*(-?[\d.]+|-inf) dB', result.stderr)
    if not means: raise RuntimeError('volumedetect mean missing')
    mean, maximum = float(means[-1]), float(maxima[-1]) if maxima else float('-inf')
    return {'mean_volume_dbfs': mean if math.isfinite(mean) else None,
            'max_volume_dbfs': maximum if math.isfinite(maximum) else None,
            'rms_normalized': 10**(mean/20) if math.isfinite(mean) else 0.0,
            'silent_or_near_silent': not math.isfinite(mean) or mean <= -80,
            'analysis': 'ffmpeg volumedetect over full first audio stream',
            'semantic_audio_review': 'not_performed'}


def usable_av(probe, mean_db):
    kinds = {s.get('codec_type') for s in probe.get('streams', [])}
    return {'audio', 'video'} <= kinds and mean_db is not None and mean_db > -80


def declared_duration(info):
    for field in info.get('metadata', []):
        if field.get('name', '').lower() in {'length', 'duration', 'durationseconds'}:
            try: return float(field['value'])
            except (TypeError, ValueError): pass
    return None


def safe_ext(url):
    ext = Path(urlsplit(url).path).suffix.lower()
    return ext if ext in {'.webm', '.ogv', '.ogg', '.mp4', '.mov'} else '.media'


def _fetch_json(session, params):
    resp = session.get(API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if 'error' in data: raise RuntimeError('Commons API '+json.dumps(data['error']))
    return data, resp.url


def download(session, url, expected, path, budget):
    budget.begin(expected)
    digest, count = hashlib.sha256(), 0
    # No retry: failed attempts and their received bytes remain in the denominator.
    with session.get(url, stream=True, timeout=(15, 45)) as response:
        response.raise_for_status()
        header = response.headers.get('Content-Length')
        if header and int(header) != expected:
            raise ValueError('Content-Length differs from declared bytes')
        with Path(path).open('xb') as f:
            while count < expected:
                chunk = response.raw.read(min(1024*1024, expected-count))
                if not chunk: break
                budget.account(len(chunk)); digest.update(chunk); f.write(chunk); count += len(chunk)
    if count != expected: raise ValueError(f'truncated download {count}/{expected}')
    return digest.hexdigest(), count


def draft_questions(row):
    return {
        'draft_id': 'draft_'+row['source_id'],
        'status': 'draft_unreviewed',
        'initial_source_ids': [row['source_id']],
        'candidate_domain': row['candidate_domain'],
        'research_question': '这段自然录制呈现了怎样的声画过程，现有材料能够确认哪些对应与来源边界？',
        'target_questions': [
            '选择一个可听的非语言声学变化：它在原始播放轴的哪个区间出现，前后可见过程是什么？',
            '声音与可见对象或动作的对应能确认到什么程度，哪些画外声、旁白或未知解释仍需核验？',
            '原始文件和来源描述页各能支持什么；要判断其他材料是否同一事件或独立录制，还需要什么证据？',
        ],
        'gold_status': 'not_created',
        'human_review': 'not_performed',
        'note': '候选问题表述，不含答案；自然性、声源与跨源关系尚未审核。',
    }


def run(out_dir, target=10, max_downloads=25, max_bytes=2_000_000_000, max_asset_bytes=100_000_000, *, queries=None, seen_page_ids=None):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in ['raw', 'metadata', 'search']: (out_dir/name).mkdir()
    session = requests.Session()
    session.headers['User-Agent'] = 'AVDeepResearchPilot/0.1 (bounded academic public AV seed acquisition)'
    budget = AcquisitionBudget(max_downloads, max_bytes)
    seen, rows, search_failures = set(seen_page_ids or []), [], []
    queries = list(queries or QUERIES)
    accepted_by_domain = {domain: 0 for domain,_ in queries}
    started = utc_now()
    for query_index, (domain, query) in enumerate(queries):
        if sum(accepted_by_domain.values()) >= target: break
        if budget.download_attempts >= max_downloads: break
        # Reserve all three domains; cap a domain at four accepted seeds.
        if accepted_by_domain[domain] >= 4: continue
        params = {'action':'query','format':'json','generator':'search',
                  'gsrsearch':'filetype:video '+query,'gsrnamespace':6,'gsrlimit':3,
                  'prop':'imageinfo','iiprop':'url|size|extmetadata|metadata|timestamp|sha1'}
        try:
            data, request_url = _fetch_json(session, params)
            search_path = out_dir/'search'/f'{query_index:02d}-api.json'
            write_json_once(search_path, {'retrieved_at':utc_now(),'request_url':request_url,'params':params,'response':data})
        except Exception as e:
            search_failures.append({'query':query,'domain':domain,'error':str(e),'retrieved_at':utc_now()})
            continue
        candidates, refused = select_candidates(data.get('query',{}).get('pages',{}), seen, max_asset_bytes)
        for page in refused:
            seen.add(page['pageid'])
            rows.append({'page_id':page['pageid'],'public_metadata':{'title':page['title']},'candidate_domain':domain,
                         'status':'refused_oversize_or_unknown_size','declared_bytes':page.get('imageinfo',[{}])[0].get('size'),
                         'search_metadata_path':str(search_path),'query':query})
        for page in candidates:
            if sum(accepted_by_domain.values()) >= target or accepted_by_domain[domain] >= 4: break
            if budget.download_attempts >= max_downloads: break
            seen.add(page['pageid'])
            info = page['imageinfo'][0]; ext = info.get('extmetadata', {})
            val = lambda key: public_text(ext.get(key, {}).get('value',''))
            source_id = 'wc_'+hashlib.sha256(('commons:'+str(page['pageid'])).encode()).hexdigest()[:20]
            url = original_url(info['url']); suffix = safe_ext(url)
            metadata_path = out_dir/'metadata'/(source_id+'-api.json')
            write_json_once(metadata_path, {'retrieved_at':utc_now(),'request_url':request_url,'page':page})
            row = {'source_id':source_id,'page_id':page['pageid'],'origin_url':info.get('descriptionurl'),
                   'descriptionurl':info.get('descriptionurl'),'original_url':url,'fetched_at':utc_now(),
                   'declared_bytes':info['size'],'declared_duration_s':declared_duration(info),'candidate_domain':domain,
                   'query':query,'metadata_path':str(metadata_path),'search_metadata_path':str(search_path),
                   'public_metadata':{'title':page['title'],'description':val('ImageDescription'),
                                      'creator':val('Artist'),'license':val('LicenseShortName'),
                                      'license_url':val('LicenseUrl'),'attribution':val('Attribution'),
                                      'credit':val('Credit'),'usage_terms':val('UsageTerms'),'categories':val('Categories')},
                   'relation_status':'unknown','page_relation':'publisher_description_of_asset_not_independent_recording',
                   'human_review':'not_performed','status':'download_pending'}
            partial = out_dir/'raw'/(source_id+'.partial')
            try:
                sha, n = download(session, url, info['size'], partial, budget)
                asset = out_dir/'raw'/(sha+suffix)
                if asset.exists(): partial.unlink()
                else: partial.rename(asset)
                row.update({'asset_path':str(asset),'sha256':sha,'bytes':n,'download_status':'complete'})
                probe = ffprobe(asset)
                probe_path = out_dir/'metadata'/(source_id+'-ffprobe.json')
                write_json_once(probe_path, probe)
                row['ffprobe_path'] = str(probe_path)
                row['stream_specs'] = probe.get('streams',[])
                row['duration_s'] = float(probe.get('format',{}).get('duration') or 0)
                kinds = {s.get('codec_type') for s in probe.get('streams',[])}
                if {'audio','video'} <= kinds:
                    energy = audio_energy(asset); row['audio_energy'] = energy
                    row['status'] = 'candidate_unreviewed_av' if usable_av(probe,energy['mean_volume_dbfs']) else 'rejected_silent_or_near_silent'
                    if row['status'] == 'candidate_unreviewed_av': accepted_by_domain[domain] += 1
                else: row['status'] = 'rejected_missing_audio_or_video'
            except Exception as e:
                row['status'] = 'failed_download_or_probe'; row['error'] = str(e)
                if partial.exists(): row['partial_path'] = str(partial); row['partial_bytes'] = partial.stat().st_size
            # Publisher page is a separate versioned public source, never independent recording evidence.
            try:
                page_data, page_request = _fetch_json(session, {'action':'parse','format':'json','pageid':page['pageid'],'prop':'text|revid|displaytitle'})
                page_path = out_dir/'metadata'/(source_id+'-page-api.json')
                write_json_once(page_path, {'retrieved_at':utc_now(),'request_url':page_request,'response':page_data})
                body = page_data.get('parse',{}).get('text',{}).get('*','')
                html_path = out_dir/'metadata'/(source_id+'-page.html')
                with html_path.open('x',encoding='utf-8') as f: f.write(body)
                text_path = out_dir/'metadata'/(source_id+'-page.txt')
                with text_path.open('x',encoding='utf-8') as f: f.write(public_text(body)+'\n')
                row.update({'description_page_api_path':str(page_path),'description_page_html_path':str(html_path),
                            'description_page_text_path':str(text_path),'description_page_sha256':hashlib.sha256(body.encode()).hexdigest(),
                            'description_page_revision':page_data.get('parse',{}).get('revid')})
            except Exception as e: row['description_page_error'] = str(e)
            rows.append(row)
            print(json.dumps({'source_id':source_id,'status':row['status'],'bytes':row.get('bytes'),
                              'duration_s':row.get('duration_s'),'accepted_by_domain':accepted_by_domain},ensure_ascii=False),flush=True)
            time.sleep(0.5)
    accepted = [r for r in rows if r['status']=='candidate_unreviewed_av']
    manifest = {'schema':'commons-av-seed-acquisition/v1','started_at':started,'finished_at':utc_now(),
                'status':'acquired_seeds_not_gold','constraints':{'target':target,'max_downloads':max_downloads,
                    'max_bytes':max_bytes,'max_asset_bytes':max_asset_bytes,'queries_max':len(queries),'per_query_max':3},
                'actual':{'download_attempts':budget.download_attempts,'bytes_received_including_failed':budget.bytes_received,
                    'unique_pages_considered':len(rows),'accepted_av_seeds':len(accepted),'accepted_by_domain':accepted_by_domain,
                    'complete_assets':sum(r.get('download_status')=='complete' for r in rows),
                    'accepted_bytes':sum(r.get('bytes',0) for r in accepted),
                    'accepted_duration_s':sum(r.get('duration_s',0) for r in accepted)},
                'semantic_validation':'not_performed','independence_validation':'not_performed',
                'search_failures':search_failures,'sources':rows}
    write_json_once(out_dir/'acquisition-manifest.json',manifest)
    with (out_dir/'acquisition-manifest.jsonl').open('x',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write_json_once(out_dir/'natural-question-drafts.json', [draft_questions(r) for r in accepted[:10]])
    write_json_once(out_dir/'tool-versions.json',{'ffprobe':subprocess.check_output(['ffprobe','-version'],text=True).splitlines()[0],
        'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(manifest['actual'],ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--target',type=int,default=10)
    p.add_argument('--max-downloads',type=int,default=25)
    p.add_argument('--max-bytes',type=int,default=2_000_000_000)
    args=p.parse_args()
    if not 1 <= args.target <= 10 or not 1 <= args.max_downloads <= 25 or not 1 <= args.max_bytes <= 2_000_000_000:
        p.error('Caps must stay within target10/downloads25/2GB')
    run(args.out_dir,args.target,args.max_downloads,args.max_bytes)


if __name__ == '__main__': main()

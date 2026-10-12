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
import os
from email.utils import parsedate_to_datetime
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
        self.on_change = None
    def begin(self, declared_bytes):
        if self.download_attempts >= self.max_downloads: raise ValueError('download_attempt_cap')
        if declared_bytes <= 0 or declared_bytes > self.max_bytes - self.bytes_received:
            raise ValueError('byte_cap')
        self.download_attempts += 1
        if self.on_change: self.on_change()
    def account(self, actual_bytes):
        if actual_bytes < 0 or actual_bytes > self.max_bytes - self.bytes_received:
            raise ValueError('byte_cap')
        self.bytes_received += actual_bytes
        if self.on_change: self.on_change()


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


class CommonsCooldown(RuntimeError):
    """Stop the batch until retry_at; do not advance to the next API query."""
    def __init__(self, reason, retry_at):
        super().__init__(reason)
        self.reason, self.retry_at = reason, retry_at


def _retry_after(value, now):
    try:
        seconds = float(value)
        if math.isfinite(seconds): return max(0.0, seconds)
    except (TypeError, ValueError): pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - now)
    except (TypeError, ValueError, OverflowError): return 0.0


class CommonsAPIPolicy:
    """Serial API access with pacing, bounded retry, and durable long-cooldown exit."""
    def __init__(self, *, pace_s=2.0, max_attempts=4, backoff_s=5.0, max_wait_s=60.0,
                 sleep=None, clock=None, now=None):
        if pace_s < 0 or backoff_s <= 0 or max_wait_s <= 0 or not 1 <= max_attempts <= 8:
            raise ValueError('Invalid Commons retry policy')
        self.pace_s, self.max_attempts = pace_s, max_attempts
        self.backoff_s, self.max_wait_s = backoff_s, max_wait_s
        self.sleep, self.clock, self.now = sleep or time.sleep, clock or time.monotonic, now or time.time
        self.last_request = None

    def fetch_json(self, session, params):
        params = {**params, 'maxlag': 5}
        for attempt in range(self.max_attempts):
            if self.last_request is not None:
                delay = self.pace_s - (self.clock() - self.last_request)
                if delay > 0: self.sleep(delay)
            self.last_request = self.clock()
            response, retry_after, reason, lag = None, 0.0, None, 0.0
            try:
                response = session.get(API, params=params, timeout=30)
                retry_after = _retry_after(response.headers.get('Retry-After'), self.now())
                if response.status_code in {429, 503}:
                    reason = 'HTTP ' + str(response.status_code)
                else:
                    response.raise_for_status()
                    data = response.json()
                    error = data.get('error', {})
                    if error.get('code') in {'maxlag', 'ratelimited'}:
                        reason = 'Commons API ' + error['code']
                        try: lag = max(0.0, float(error.get('lag', 0)))
                        except (TypeError, ValueError): pass
                    elif error:
                        raise RuntimeError('Commons API ' + json.dumps(error))
                    else:
                        return data, response.url
            except (requests.ConnectionError, requests.Timeout) as exc:
                reason = type(exc).__name__
            finally:
                if response is not None: response.close()
            # Never shorten a server-requested wait. Large Retry-After values
            # stop the process so run() can persist the precise retry deadline.
            delay = max(retry_after, min(self.max_wait_s, self.backoff_s * 2**attempt), lag)
            if delay > self.max_wait_s or attempt + 1 == self.max_attempts:
                raise CommonsCooldown(reason or 'transient API failure', self.now() + delay)
            print(json.dumps({'event':'commons_api_retry','reason':reason,'attempt':attempt+1,
                              'wait_s':delay}), flush=True)
            self.sleep(delay)
        raise AssertionError('unreachable retry loop')


def _fetch_json(session, params, *, policy=None):
    return (policy or CommonsAPIPolicy()).fetch_json(session, params)


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


def _atomic_json(path, data):
    """Replace only mutable progress pointers; original evidence is create-only."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def run(out_dir, target=10, max_downloads=25, max_bytes=2_000_000_000, max_asset_bytes=100_000_000,
        *, queries=None, seen_page_ids=None, resume=False, per_query_limit=3,
        per_domain_limit=4, api_pace_s=2.0, max_api_attempts=4,
        api_backoff_s=5.0, max_inline_wait_s=60.0):
    """Collect serially, checkpoint every received chunk, and resume immutably.

    Explicit caps retain the original defaults. A run directory is create-only
    unless resume=True; resume requires identical acquisition configuration.
    Complete media, API responses, and partial bytes are never overwritten or
    downloaded again. In-flight interrupted transfers are retained as failures.
    """
    if (not 1 <= target <= 10000 or not 1 <= max_downloads <= 10000 or max_bytes <= 0
            or max_asset_bytes <= 0 or not 1 <= per_query_limit <= 50 or per_domain_limit < 1):
        raise ValueError('Invalid acquisition caps')
    out_dir = Path(out_dir).resolve()
    queries = [list(pair) for pair in (queries or QUERIES)]
    constraints = {'target':target, 'max_downloads':max_downloads, 'max_bytes':max_bytes,
                   'max_asset_bytes':max_asset_bytes, 'queries_max':len(queries),
                   'per_query_max':per_query_limit, 'per_domain_max':per_domain_limit,
                   'api_pace_s':api_pace_s, 'max_api_attempts':max_api_attempts,
                   'api_backoff_s':api_backoff_s, 'max_inline_wait_s':max_inline_wait_s}
    checkpoint_path = out_dir/'acquisition-checkpoint.json'
    if resume:
        state = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if state['constraints'] != constraints or state['queries'] != queries:
            raise ValueError('Resume must preserve the original query plan and budgets')
        if (state.get('retry_at') or 0) > time.time():
            return json.loads((out_dir/'acquisition-manifest.json').read_text(encoding='utf-8'))
        state['retry_at'] = None
        state.pop('stop_reason', None)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        for name in ['raw','metadata','search','snapshots']: (out_dir/name).mkdir()
        state = {'schema':'commons-acquisition-checkpoint/v2', 'started_at':utc_now(),
                 'constraints':constraints, 'queries':queries, 'query_index':0,
                 'active_query':None, 'seen_page_ids':sorted(set(seen_page_ids or [])),
                 'sources':[], 'search_failures':[], 'download_attempts':0,
                 'bytes_received':0, 'snapshot_sequence':0, 'status':'acquiring', 'retry_at':None}
    rows = state['sources']
    budget = AcquisitionBudget(max_downloads, max_bytes)
    budget.download_attempts, budget.bytes_received = state['download_attempts'], state['bytes_received']
    manifest = None

    def counts():
        accepted = {domain:0 for domain,_ in queries}
        for row in rows:
            if row['status'] == 'candidate_unreviewed_av':
                domain = row['candidate_domain']; accepted[domain] = accepted.get(domain,0)+1
        return accepted

    def save(snapshot=True):
        nonlocal manifest
        state['download_attempts'], state['bytes_received'] = budget.download_attempts, budget.bytes_received
        state['updated_at'] = utc_now()
        if snapshot: state['snapshot_sequence'] += 1
        _atomic_json(checkpoint_path, state)
        if not snapshot: return
        accepted = [row for row in rows if row['status']=='candidate_unreviewed_av']
        manifest = {'schema':'commons-av-seed-acquisition/v2', 'started_at':state['started_at'],
                    'updated_at':state['updated_at'], 'status':state['status'], 'constraints':constraints,
                    'progress':{'query_index':state['query_index'], 'queries_total':len(queries),
                                'retry_at':state.get('retry_at'), 'stop_reason':state.get('stop_reason')},
                    'actual':{'download_attempts':budget.download_attempts,
                              'bytes_received_including_failed':budget.bytes_received,
                              'unique_pages_considered':len(rows), 'accepted_av_seeds':len(accepted),
                              'accepted_by_domain':counts(),
                              'complete_assets':sum(row.get('download_status')=='complete' for row in rows),
                              'accepted_bytes':sum(row.get('bytes',0) for row in accepted),
                              'accepted_duration_s':sum(row.get('duration_s',0) for row in accepted)},
                    'semantic_validation':'not_performed', 'independence_validation':'not_performed',
                    'search_failures':state['search_failures'], 'sources':rows}
        if state.get('finished_at'): manifest['finished_at'] = state['finished_at']
        write_json_once(out_dir/'snapshots'/f"manifest-{state['snapshot_sequence']:06d}.json", manifest)
        _atomic_json(out_dir/'acquisition-manifest.json', manifest)
        _atomic_json(out_dir/'natural-question-drafts.json', [draft_questions(row) for row in accepted[:10]])
        temporary = out_dir/'acquisition-manifest.jsonl.tmp'
        with temporary.open('w',encoding='utf-8') as stream:
            for row in rows: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        temporary.replace(out_dir/'acquisition-manifest.jsonl')

    budget.on_change = lambda: save(snapshot=False)
    state['status'] = 'acquiring'
    save()
    version_path = out_dir/'tool-versions.json'
    if not version_path.exists():
        write_json_once(version_path, {'ffprobe':subprocess.check_output(['ffprobe','-version'],text=True).splitlines()[0],
                         'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
                         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    session = requests.Session()
    session.headers['User-Agent'] = ('AVDeepResearchSeedCollector/0.2 '
        '(https://github.com/MichaelCao0/OneSearch-VL; bounded academic AV acquisition)')
    policy = CommonsAPIPolicy(pace_s=api_pace_s, max_attempts=max_api_attempts,
                             backoff_s=api_backoff_s, max_wait_s=max_inline_wait_s)
    try:
        while state['query_index'] < len(queries):
            index = state['query_index']; domain, query = queries[index]
            if state['active_query'] is None:
                if sum(counts().values()) >= target:
                    state['stop_reason'] = 'target_reached'; break
                if budget.download_attempts >= max_downloads or budget.bytes_received >= max_bytes:
                    state['stop_reason'] = 'acquisition_budget_reached'; break
                if counts()[domain] >= per_domain_limit:
                    state['query_index'] += 1; save(); continue
                search_path = out_dir/'search'/f'{index:03d}-api.json'
                params = {'action':'query','format':'json','generator':'search',
                          'gsrsearch':'filetype:video '+query,'gsrnamespace':6,
                          'gsrlimit':min(50,max(3,per_query_limit*2)),
                          'prop':'imageinfo','iiprop':'url|size|extmetadata|metadata|timestamp|sha1'}
                try:
                    if search_path.exists():
                        saved = json.loads(search_path.read_text()); data, request_url = saved['response'], saved['request_url']
                    else:
                        data, request_url = _fetch_json(session, params, policy=policy)
                        write_json_once(search_path, {'retrieved_at':utc_now(),'request_url':request_url,'params':params,'response':data})
                except CommonsCooldown: raise
                except Exception as exc:
                    state['search_failures'].append({'query':query,'domain':domain,'error':str(exc),'retrieved_at':utc_now()})
                    state['query_index'] += 1; save(); continue
                candidates, refused = select_candidates(data.get('query',{}).get('pages',{}),
                                                         set(state['seen_page_ids']), max_asset_bytes, per_query_limit)
                for page in refused:
                    state['seen_page_ids'].append(page['pageid'])
                    rows.append({'page_id':page['pageid'],'public_metadata':{'title':page['title']},
                                 'candidate_domain':domain,'status':'refused_oversize_or_unknown_size',
                                 'declared_bytes':page.get('imageinfo',[{}])[0].get('size'),
                                 'search_metadata_path':str(search_path),'query':query})
                state['active_query'] = {'pages':candidates,'next_page':0,'search_path':str(search_path),'request_url':request_url}
                save()
            active = state['active_query']
            while active['next_page'] < len(active['pages']):
                page = active['pages'][active['next_page']]
                row = next((row for row in rows if row.get('page_id')==page['pageid']), None)
                if row is None:
                    if (sum(counts().values()) >= target or counts()[domain] >= per_domain_limit or
                            budget.download_attempts >= max_downloads or budget.bytes_received >= max_bytes): break
                    info = page['imageinfo'][0]; ext = info.get('extmetadata',{})
                    val = lambda key: public_text(ext.get(key,{}).get('value',''))
                    source_id = 'wc_'+hashlib.sha256(('commons:'+str(page['pageid'])).encode()).hexdigest()[:20]
                    url = original_url(info['url']); metadata_path = out_dir/'metadata'/(source_id+'-api.json')
                    row = {'source_id':source_id,'page_id':page['pageid'],'origin_url':info.get('descriptionurl'),
                           'descriptionurl':info.get('descriptionurl'),'original_url':url,'fetched_at':utc_now(),
                           'declared_bytes':info['size'],'declared_duration_s':declared_duration(info),
                           'candidate_domain':domain,'query':query,'metadata_path':str(metadata_path),
                           'search_metadata_path':active['search_path'],
                           'public_metadata':{'title':page['title'],'description':val('ImageDescription'),
                               'creator':val('Artist'),'license':val('LicenseShortName'),'license_url':val('LicenseUrl'),
                               'attribution':val('Attribution'),'credit':val('Credit'),'usage_terms':val('UsageTerms'),
                               'categories':val('Categories')}, 'relation_status':'unknown',
                           'page_relation':'publisher_description_of_asset_not_independent_recording',
                           'human_review':'not_performed','status':'download_pending','pending_description_page':True}
                    rows.append(row); state['seen_page_ids'].append(page['pageid']); save()
                source_id = row['source_id']; metadata_path = Path(row['metadata_path'])
                if not metadata_path.exists():
                    write_json_once(metadata_path, {'retrieved_at':utc_now(),'request_url':active['request_url'],'page':page})
                partial = out_dir/'raw'/(source_id+'.partial')
                if row['status'] == 'download_in_progress':
                    # A crash may have left a partial transfer. Preserve every
                    # byte and charge the already checkpointed attempt, never retry.
                    row['status'] = 'failed_download_or_probe'; row['download_status'] = 'interrupted_preserved'
                    if partial.exists(): row.update({'partial_path':str(partial),'partial_bytes':partial.stat().st_size})
                    save()
                if row['status'] == 'download_pending':
                    row['status'] = 'download_in_progress'; save()
                    try:
                        sha, size = download(session, row['original_url'], row['declared_bytes'], partial, budget)
                        asset = out_dir/'raw'/(sha+safe_ext(row['original_url']))
                        if asset.exists(): partial.unlink()
                        else: partial.rename(asset)
                        row.update({'asset_path':str(asset),'sha256':sha,'bytes':size,
                                    'download_status':'complete','status':'download_complete'})
                    except Exception as exc:
                        row['status'] = 'failed_download_or_probe'; row['error'] = str(exc)
                        if partial.exists(): row.update({'partial_path':str(partial),'partial_bytes':partial.stat().st_size})
                        save()
                        if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code in {429,503}:
                            delay = max(api_backoff_s, _retry_after(exc.response.headers.get('Retry-After'), time.time()))
                            raise CommonsCooldown('media HTTP '+str(exc.response.status_code), time.time()+delay)
                    save()
                if row['status'] == 'download_complete':
                    try:
                        asset = Path(row['asset_path']); probe_path = out_dir/'metadata'/(source_id+'-ffprobe.json')
                        if probe_path.exists(): probe = json.loads(probe_path.read_text())
                        else:
                            probe = ffprobe(asset); write_json_once(probe_path,probe)
                        row['ffprobe_path'] = str(probe_path); row['stream_specs'] = probe.get('streams',[])
                        row['duration_s'] = float(probe.get('format',{}).get('duration') or 0)
                        kinds = {stream.get('codec_type') for stream in probe.get('streams',[])}
                        if {'audio','video'} <= kinds:
                            energy = audio_energy(asset); row['audio_energy'] = energy
                            row['status'] = ('candidate_unreviewed_av' if usable_av(probe,energy['mean_volume_dbfs'])
                                             else 'rejected_silent_or_near_silent')
                        else: row['status'] = 'rejected_missing_audio_or_video'
                    except Exception as exc:
                        row['status'] = 'failed_download_or_probe'; row['error'] = str(exc)
                    save()
                if row.get('pending_description_page'):
                    try:
                        page_path = out_dir/'metadata'/(source_id+'-page-api.json')
                        if page_path.exists():
                            saved = json.loads(page_path.read_text()); page_data = saved['response']
                        else:
                            page_data,page_request = _fetch_json(session, {'action':'parse','format':'json',
                                'pageid':page['pageid'],'prop':'text|revid|displaytitle'}, policy=policy)
                            write_json_once(page_path, {'retrieved_at':utc_now(),'request_url':page_request,'response':page_data})
                        body = page_data.get('parse',{}).get('text',{}).get('*','')
                        html_path = out_dir/'metadata'/(source_id+'-page.html')
                        if not html_path.exists():
                            with html_path.open('x',encoding='utf-8') as stream: stream.write(body)
                        text_path = out_dir/'metadata'/(source_id+'-page.txt')
                        if not text_path.exists():
                            with text_path.open('x',encoding='utf-8') as stream: stream.write(public_text(body)+'\n')
                        row.update({'description_page_api_path':str(page_path),'description_page_html_path':str(html_path),
                                    'description_page_text_path':str(text_path),
                                    'description_page_sha256':hashlib.sha256(body.encode()).hexdigest(),
                                    'description_page_revision':page_data.get('parse',{}).get('revid')})
                    except CommonsCooldown: raise
                    except Exception as exc: row['description_page_error'] = str(exc)
                    row['pending_description_page'] = False
                active['next_page'] += 1; save()
                print(json.dumps({'source_id':source_id,'status':row['status'],'bytes':row.get('bytes'),
                                  'duration_s':row.get('duration_s'),'accepted_by_domain':counts()},ensure_ascii=False),flush=True)
            state['query_index'] += 1; state['active_query'] = None; save()
        state['status'] = 'acquired_seeds_not_gold'; state['finished_at'] = utc_now()
        state.setdefault('stop_reason','queries_exhausted'); save()
    except CommonsCooldown as exc:
        state.update({'status':'paused_cooldown','retry_at':exc.retry_at,'stop_reason':exc.reason})
        state['search_failures'].append({'query':queries[state['query_index']][1],
                                        'domain':queries[state['query_index']][0],
                                        'error':exc.reason,'retry_at':exc.retry_at,'retrieved_at':utc_now()})
        save()
        print(json.dumps({'event':'commons_cooldown_stop','reason':exc.reason,'retry_at':exc.retry_at}),flush=True)
    except BaseException:
        state['status'] = 'interrupted'; save(); raise
    finally:
        session.close()
    print(json.dumps({'status':manifest['status'],**manifest['actual']},ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--target',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-downloads',type=int,default=25)
    p.add_argument('--max-bytes',type=int,default=2_000_000_000)
    args=p.parse_args()
    if not 1 <= args.target <= 10 or not 1 <= args.max_downloads <= 25 or not 1 <= args.max_bytes <= 2_000_000_000:
        p.error('Caps must stay within target10/downloads25/2GB')
    run(args.out_dir,args.target,args.max_downloads,args.max_bytes,resume=args.resume)


if __name__ == '__main__': main()

"""Immutable web snapshots with deterministic extraction and exact quote replay.

A snapshot records captured publisher bytes, never a generated summary. HTML
fragments from a versioned API response are supported when their transport and
publisher revision are recorded explicitly. Publisher text is not an additional
independent recording. Raw paths and acquisition identities stay private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlsplit

from research.av_search.corpus import canonical, META_FIELDS
from .tasks import reject_private_fields

SCHEMA = 'av-deepresearch.web-source.v1'
REVISION = 'web-snapshot-v1'
FIELDS = frozenset({'schema_version','source_id','origin_url','fetched_at','ingested_at',
    'media_type','duration_s','asset_bytes','asset_sha256','asset_path','asset_store_root',
    'rights_basis','redistribution_scope','public_metadata','access_status','created_by',
    'body_text','body_text_sha256','extraction','snapshot_identity'})
SNAPSHOT_FIELDS = frozenset({'transport','request_url','publisher_revision'})


class _BodyParser(HTMLParser):
    """Skip non-body script/style/head content; separate tags before whitespace collapse."""
    excluded = frozenset({'head','script','style','noscript','template'})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[]
        self.skipped=[]

    def handle_starttag(self, tag, attrs):
        if tag in self.excluded:
            self.skipped.append(tag)
        if not self.skipped:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in self.skipped:
            # Malformed nested excluded tags cannot leak their text back out.
            position=len(self.skipped)-1-self.skipped[::-1].index(tag)
            self.skipped=self.skipped[:position]
        if not self.skipped:
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.skipped:
            self.parts.append(data)


def _extract(raw, content_type, encoding):
    if encoding != 'utf-8' or content_type not in {'text/html','text/plain'}:
        raise ValueError('Only explicit UTF-8 HTML/plain-text snapshots are supported')
    text=raw.decode('utf-8',errors='strict')
    if content_type == 'text/html':
        parser=_BodyParser();parser.feed(text);parser.close()
        text=''.join(parser.parts)
    body=' '.join(text.split())
    if not body:
        raise ValueError('Captured snapshot has no extractable body text')
    return body


def _extraction(content_type, encoding):
    identity={'tool':'stdlib-htmlparser-body-whitespace','revision':REVISION,
        'content_type':content_type,'encoding':encoding,
        'rule':'exclude head/script/style/noscript/template; tags separate text; collapse Unicode whitespace'}
    identity['identity_sha256']=hashlib.sha256(canonical(identity).encode()).hexdigest()
    return identity


def _url(value):
    if not isinstance(value,str):raise ValueError('origin/request URL must be public HTTP(S)')
    parsed=urlsplit(value)
    if parsed.scheme not in {'http','https'} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('origin/request URL must be public HTTP(S) without credentials')
    return value


def _time(value):
    if not isinstance(value,str):raise ValueError('fetched_at must be an ISO timestamp with timezone')
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('fetched_at must contain a timezone')
    return value


def _metadata(value):
    value={} if value is None else value
    if not isinstance(value,dict) or set(value)-META_FIELDS:raise ValueError('Only public corpus metadata is allowed')
    reject_private_fields(value)
    for key,item in value.items():
        if not isinstance(item,str) and not (key=='categories' and isinstance(item,list) and all(isinstance(x,str) for x in item)):
            raise ValueError('public_metadata must contain public text/categories')
    return json.loads(canonical(value))


def _snapshot(value, origin_url):
    value={'transport':'local_snapshot','request_url':origin_url,'publisher_revision':None} if value is None else value
    if not isinstance(value,dict) or set(value)!=SNAPSHOT_FIELDS:raise ValueError('Unexpected snapshot identity fields')
    if value['transport'] not in {'http','commons_api_parse_fragment','local_snapshot'}:raise ValueError('Unknown capture transport')
    _url(value['request_url'])
    if value['publisher_revision'] is not None and (not isinstance(value['publisher_revision'],str) or not value['publisher_revision'].strip()):
        raise ValueError('publisher_revision must be explicit string or null')
    return json.loads(canonical(value))


def ingest_web_snapshot(local_path, origin_url, asset_root, public_metadata=None,
        rights_basis='unknown', redistribution_scope='metadata_only', *, fetched_at=None,
        content_type='text/html', encoding='utf-8', snapshot_identity=None):
    """Copy captured bytes and derive replayable body text; performs no fetching.

    Pass the capture timestamp when importing older snapshots. When omitted, the
    timestamp records local intake and transport defaults to local_snapshot.
    """
    source=Path(local_path).resolve()
    if not source.is_file() or source.stat().st_size<=0:raise ValueError('Snapshot is absent or empty')
    origin_url=_url(origin_url);metadata=_metadata(public_metadata)
    fetched_at=_time(fetched_at or datetime.now(timezone.utc).isoformat())
    snapshot=_snapshot(snapshot_identity,origin_url)
    if not isinstance(rights_basis,str) or not rights_basis.strip():raise ValueError('rights_basis must be explicit')
    if redistribution_scope not in {'metadata_only','annotation_only','full_asset','unknown'}:raise ValueError('Unsupported redistribution_scope')
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest();body=_extract(raw,content_type,encoding)
    root_input=Path(asset_root).absolute()
    if root_input!=root_input.resolve():raise ValueError('asset_root must not traverse symlinks')
    root=root_input.resolve();directory=root/digest[:2];directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(digest+('.html' if content_type=='text/html' else '.txt'))
    if directory.is_symlink() or destination.is_symlink():raise ValueError('Content-addressed paths must not be symlinks')
    if destination.exists():
        if destination.read_bytes()!=raw:raise ValueError('Existing content-addressed snapshot is corrupt')
    else:
        descriptor,name=tempfile.mkstemp(prefix='.web-',dir=directory);os.close(descriptor);tmp=Path(name)
        try:
            shutil.copyfile(source,tmp)
            if tmp.read_bytes()!=raw:raise ValueError('Snapshot changed during intake')
            try:os.link(tmp,destination)
            except FileExistsError:
                if destination.read_bytes()!=raw:raise ValueError('Concurrent snapshot is corrupt')
        finally:tmp.unlink(missing_ok=True)
    record={'schema_version':SCHEMA,'source_id':'src_'+digest[:24],'origin_url':origin_url,
        'fetched_at':fetched_at,'ingested_at':datetime.now(timezone.utc).isoformat(),
        'media_type':'webpage','duration_s':None,'asset_bytes':len(raw),'asset_sha256':digest,
        'asset_path':str(destination),'asset_store_root':str(root),'rights_basis':rights_basis,
        'redistribution_scope':redistribution_scope,'public_metadata':metadata,'access_status':'available_local',
        'created_by':{'tool':'ingest_web_snapshot','revision':REVISION},'body_text':body,
        'body_text_sha256':hashlib.sha256(body.encode()).hexdigest(),'extraction':_extraction(content_type,encoding),
        'snapshot_identity':snapshot}
    errors=validate_web_source(record)
    if errors:raise ValueError('Invalid web snapshot: '+'; '.join(errors))
    return record


def validate_web_source(record, verify_hash=True):
    """Return integrity/schema errors; re-extract from raw bytes when verifying."""
    errors=[]
    try:
        if not isinstance(record,dict) or set(record)!=FIELDS:raise ValueError('Unexpected/private or missing web source fields')
        canonical(record)
        if record['schema_version']!=SCHEMA or record['created_by']!={'tool':'ingest_web_snapshot','revision':REVISION}:raise ValueError('Unknown web source revision')
        digest=record['asset_sha256']
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Malformed raw hash')
        if record['source_id']!='src_'+digest[:24]:raise ValueError('Opaque source identity does not match raw bytes')
        if record['media_type']!='webpage' or record['duration_s'] is not None:raise ValueError('Web snapshots require webpage and null duration')
        if isinstance(record['asset_bytes'],bool) or not isinstance(record['asset_bytes'],int) or record['asset_bytes']<=0:raise ValueError('Positive byte count required')
        _url(record['origin_url']);_time(record['fetched_at']);_time(record['ingested_at']);_metadata(record['public_metadata']);_snapshot(record['snapshot_identity'],record['origin_url'])
        if not isinstance(record['rights_basis'],str) or not record['rights_basis'].strip():raise ValueError('rights_basis missing')
        if record['redistribution_scope'] not in {'metadata_only','annotation_only','full_asset','unknown'} or record['access_status']!='available_local':raise ValueError('Unknown rights/access state')
        extraction=record['extraction']
        if not isinstance(extraction,dict) or extraction!=_extraction(extraction.get('content_type'),extraction.get('encoding')):raise ValueError('Extraction identity changed')
        if extraction['content_type'] not in {'text/html','text/plain'} or extraction['encoding']!='utf-8':raise ValueError('Unsupported extraction')
        body=record['body_text']
        if not isinstance(body,str) or not body.strip() or hashlib.sha256(body.encode()).hexdigest()!=record['body_text_sha256']:raise ValueError('Body text hash mismatch')
        if not isinstance(record['asset_path'],str) or not isinstance(record['asset_store_root'],str):raise ValueError('Private asset path missing')
        asset=Path(record['asset_path']);root=Path(record['asset_store_root'])
        if not root.is_absolute() or root!=root.resolve() or not asset.is_absolute() or asset!=asset.resolve():raise ValueError('Paths must be absolute and not traverse symlinks')
        suffix='.html' if extraction['content_type']=='text/html' else '.txt'
        if asset!=root/digest[:2]/(digest+suffix):raise ValueError('Asset path is outside its content-addressed store')
        if verify_hash:
            raw=asset.read_bytes()
            if len(raw)!=record['asset_bytes'] or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Raw snapshot bytes/hash mismatch')
            if _extract(raw,extraction['content_type'],extraction['encoding'])!=body:raise ValueError('Body text does not replay from raw snapshot')
    except (ValueError,TypeError,KeyError,OSError,UnicodeError) as exc:
        errors.append(str(exc))
    return errors


def web_corpus_document(record):
    errors=validate_web_source(record)
    if errors:raise ValueError('; '.join(errors))
    result={k:record[k] for k in ('source_id','origin_url','asset_sha256','asset_path','public_metadata','media_type','duration_s')}
    result.update(text=record['body_text'],text_kind='web_body')
    return json.loads(canonical(result))


def quote_web_source(record, char_start, char_end):
    errors=validate_web_source(record)
    if errors:raise ValueError('; '.join(errors))
    if any(isinstance(x,bool) or not isinstance(x,int) for x in (char_start,char_end)) or not 0<=char_start<char_end<=len(record['body_text']):
        raise ValueError('Quote requires integer 0 <= char_start < char_end <= body length')
    return {'source_id':record['source_id'],'asset_sha256':record['asset_sha256'],'text_sha256':record['body_text_sha256'],
        'char_start':char_start,'char_end':char_end,'quote':record['body_text'][char_start:char_end]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',required=True);parser.add_argument('--source-id',required=True)
    parser.add_argument('--char-start',required=True,type=int);parser.add_argument('--char-end',required=True,type=int)
    args=parser.parse_args()
    records=[json.loads(line) for line in Path(args.sources).read_text().splitlines() if line.strip()]
    matches=[r for r in records if r.get('source_id')==args.source_id]
    if len(matches)!=1:raise ValueError('Expected one source identity')
    print(json.dumps(quote_web_source(matches[0],args.char_start,args.char_end),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

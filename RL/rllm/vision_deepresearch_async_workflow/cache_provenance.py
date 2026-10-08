"""Separate cache-quality sidecar; never changes the existing cache value/schema."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import threading
from collections import OrderedDict

_LOCK = threading.Lock()
_RECENT = OrderedDict()


def identity(key):
    return json.dumps(key, separators=(',', ':'), ensure_ascii=True)


def database():
    from vision_deepresearch_async_workflow.tools.shared import CACHE_CONFIG
    return Path(os.getenv('TOOL_QUALITY_CACHE_DB', str(CACHE_CONFIG['db_path']) + '.quality.sqlite'))


def snapshot(collector):
    with collector.lock:
        return dict(version=2, source_call_id=collector.call_id, source_rollout_id=collector.rollout_id,
            pages=copy.deepcopy(list(collector.pages.values())))


def publish(key, result, data):
    entry = (key, str(result))
    with _LOCK:
        _RECENT[entry] = copy.deepcopy(data)
        _RECENT.move_to_end(entry)
        while len(_RECENT)>512: _RECENT.popitem(last=False)


def read(key, result):
    result = str(result)
    with _LOCK:
        data=copy.deepcopy(_RECENT.get((key, result)))
    if data is not None:return data
    path=database()
    if not path.exists():return None
    try:
        with sqlite3.connect(str(path),timeout=.1) as conn:
            row=conn.execute('SELECT payload FROM provenance_by_result WHERE identity=? AND result=?',(identity(key),result)).fetchone()
        return json.loads(row[0]) if row else None
    except (sqlite3.Error,OSError,ValueError):return None


def write(key, result, data):
    publish(key,result,data)
    path=database()
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(str(path),timeout=.1) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS provenance_by_result(identity TEXT, result TEXT, payload TEXT, PRIMARY KEY(identity,result))')
            conn.execute('INSERT OR REPLACE INTO provenance_by_result VALUES(?,?,?)',(identity(key),str(result),json.dumps(data)))
        return True
    except (sqlite3.Error,OSError):return False


def install(visual, current):
    if getattr(visual,'_quality_cache_observed',False):return
    import asyncio
    old_get,old_set=visual.get_cache_async,visual.set_cache_async
    async def get(table,key,*args,**kwargs):
        result=await old_get(table,key,*args,**kwargs)
        collector=current.get()
        if collector is not None and table=='text_search':
            collector.cache_identity=(table,key)
            if result is not None:
                collector.cache_provenance=await asyncio.to_thread(read,(table,key),result)
        return result
    async def put(table,key,original_input,result,*args,**kwargs):
        value=await old_set(table,key,original_input,result,*args,**kwargs)
        collector=current.get()
        if collector is not None and table=='text_search':
            data=snapshot(collector)
            ok=await asyncio.to_thread(write,(table,key),result,data)
            collector.add(cache_provenance_write_failures=int(not ok))
        return value
    visual.get_cache_async,visual.set_cache_async=get,put
    visual._quality_cache_observed=True


def write_review_evidence(collector, rank, body, summary, query, title):
    """Retain bounded source/summary pairs for actual semantic review, no URLs/headers."""
    path=database()
    payload={'call_id':collector.call_id,'rollout_id':collector.rollout_id,'page_id':f'{collector.call_id}:page{rank}',
             'body':str(body or '')[:2000],'summary':str(summary or '')[:4000],
             'query':str(query or '')[:1000],'title':str(title or '')[:500]}
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(str(path),timeout=.1) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS review_evidence(call_id TEXT,page_id TEXT,payload TEXT,PRIMARY KEY(call_id,page_id))')
            conn.execute('INSERT OR REPLACE INTO review_evidence VALUES(?,?,?)',(collector.call_id,payload['page_id'],json.dumps(payload)))
        return True
    except (OSError,sqlite3.Error):return False

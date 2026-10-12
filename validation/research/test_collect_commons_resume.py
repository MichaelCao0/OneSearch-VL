"""Commons transient-response and crash-resume boundaries; no live requests."""
from datetime import datetime, timezone
from email.utils import format_datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import requests
from research.av_data import collect_commons as c


class Response:
    def __init__(self, status=200, data=None, headers=None, body=b''):
        self.status_code, self.data = status, data or {}
        self.headers, self.url, self.raw = headers or {}, 'https://commons.wikimedia.org/w/api.php', io.BytesIO(body)
    def json(self): return self.data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Session:
    def __init__(self, responses): self.responses=list(responses); self.calls=[]; self.headers={}
    def get(self, url, **kwargs):
        self.calls.append((url,kwargs))
        value=self.responses.pop(0)
        if isinstance(value,Exception): raise value
        return value
    def close(self): pass


class CommonsRetryTests(unittest.TestCase):
    def policy(self, **kw):
        self.assertTrue(hasattr(c, 'CommonsAPIPolicy'), 'retry policy missing')
        self.clock=100.; self.sleeps=[]
        def sleep(delay): self.sleeps.append(delay); self.clock+=delay
        return c.CommonsAPIPolicy(sleep=sleep,clock=lambda:self.clock,now=lambda:1000+self.clock,**kw)

    def test_429_honors_retry_after_and_paces_following_queries(self):
        policy=self.policy(pace_s=3,backoff_s=1)
        session=Session([Response(429,headers={'Retry-After':'7'}),Response(data={'ok':True}),Response(data={'ok':2})])
        data,_=c._fetch_json(session,{'action':'query'},policy=policy)
        self.assertEqual(data,{'ok':True})
        c._fetch_json(session,{'action':'query'},policy=policy)
        self.assertEqual(self.sleeps,[7,3])
        self.assertEqual(len(session.calls),3)
        self.assertEqual(session.calls[0][1]['params']['maxlag'],5)

    def test_network_503_and_maxlag_use_bounded_backoff(self):
        policy=self.policy(pace_s=0,backoff_s=2,max_attempts=5)
        session=Session([requests.ConnectionError('offline'),Response(503),Response(data={'error':{'code':'maxlag','lag':6}}),Response(data={'ok':1})])
        c._fetch_json(session,{},policy=policy)
        self.assertEqual(self.sleeps,[2,4,8])

    def test_long_http_date_cooldown_stops_without_sleep_or_second_request(self):
        policy=self.policy(pace_s=0,max_wait_s=30)
        date=format_datetime(datetime.fromtimestamp(1400,tz=timezone.utc),usegmt=True)
        session=Session([Response(429,headers={'Retry-After':date})])
        with self.assertRaises(c.CommonsCooldown) as caught:
            c._fetch_json(session,{},policy=policy)
        self.assertEqual(caught.exception.retry_at,1400)
        self.assertEqual(len(session.calls),1)
        self.assertEqual(self.sleeps,[])

    def test_exhausted_rate_limit_does_not_continue_to_other_queries(self):
        policy=self.policy(pace_s=0,backoff_s=2,max_attempts=2)
        session=Session([Response(429),Response(429)])
        with self.assertRaises(c.CommonsCooldown): c._fetch_json(session,{},policy=policy)
        self.assertEqual(len(session.calls),2)
        self.assertEqual(self.sleeps,[2])


class CommonsResumeTests(unittest.TestCase):
    def run_config(self,path,**kw):
        return c.run(path,target=1,max_downloads=2,max_bytes=20,max_asset_bytes=10,
                     queries=[('bird','pulse')],api_pace_s=0,**kw)

    def test_checkpointed_cooldown_resumes_cached_search_and_raw_without_redownload(self):
        self.assertTrue(hasattr(c,'CommonsCooldown'),'checkpoint/resume implementation missing')
        page={'pageid':101,'title':'File:pulse.webm','imageinfo':[{'size':4,'url':'https://upload.wikimedia.org/pulse.webm','descriptionurl':'https://commons.wikimedia.org/wiki/File:pulse.webm'}]}
        first=Session([Response(data={'query':{'pages':{'101':page}}}),Response(body=b'AV01'),Response(429,headers={'Retry-After':'600'})])
        with tempfile.TemporaryDirectory() as tmp, patch.object(c.time,'time',return_value=1000), patch.object(c.time,'sleep'), patch.object(c,'ffprobe',return_value={'streams':[{'codec_type':'audio'},{'codec_type':'video'}],'format':{'duration':'1'}}), patch.object(c,'audio_energy',return_value={'mean_volume_dbfs':-20}):
            path=Path(tmp)/'batch'
            with patch.object(c.requests,'Session',return_value=first): manifest=self.run_config(path)
            self.assertEqual(manifest['status'],'paused_cooldown')
            self.assertEqual(manifest['actual']['accepted_av_seeds'],1)
            self.assertEqual(manifest['actual']['download_attempts'],1)
            self.assertTrue((path/'acquisition-checkpoint.json').exists())
            raw=list((path/'raw').glob('*'));self.assertEqual(len(raw),1)
            before=(raw[0].read_bytes(),raw[0].stat().st_mtime_ns)
            with patch.object(c.requests,'Session',side_effect=AssertionError('request before cooldown')):
                early=self.run_config(path,resume=True)
            self.assertEqual(early['status'],'paused_cooldown')
            second=Session([Response(data={'parse':{'text':{'*':'<p>public body</p>'},'revid':9}})])
            with patch.object(c.time,'time',return_value=2000),patch.object(c.requests,'Session',return_value=second):
                final=self.run_config(path,resume=True)
            self.assertEqual(final['actual']['download_attempts'],1)
            self.assertEqual(final['actual']['bytes_received_including_failed'],4)
            self.assertEqual(final['actual']['accepted_av_seeds'],1)
            self.assertEqual(len(second.calls),1)
            self.assertEqual(second.calls[0][1]['params']['action'],'parse')
            self.assertEqual((raw[0].read_bytes(),raw[0].stat().st_mtime_ns),before)
            self.assertGreater(len(list((path/'snapshots').glob('*.json'))),1)

    def test_interrupted_partial_is_preserved_and_charged_without_redownload(self):
        page={'pageid':102,'title':'File:partial.webm','imageinfo':[{'size':4,'url':'https://upload.wikimedia.org/partial.webm','descriptionurl':'https://commons.wikimedia.org/wiki/File:partial.webm'}]}
        class InterruptedRaw:
            def __init__(self): self.first=True
            def read(self,size):
                if self.first: self.first=False; return b'AV'
                raise KeyboardInterrupt('simulated transfer interruption')
        response=Response();response.raw=InterruptedRaw()
        first=Session([Response(data={'query':{'pages':{'102':page}}}),response])
        with tempfile.TemporaryDirectory() as tmp,patch.object(c.requests,'Session',return_value=first):
            path=Path(tmp)/'batch'
            with self.assertRaises(KeyboardInterrupt): self.run_config(path)
            raw=list((path/'raw').glob('*.partial'));self.assertEqual(len(raw),1)
            before=(raw[0].read_bytes(),raw[0].stat().st_mtime_ns)
            second=Session([Response(data={'parse':{'text':{'*':'public body'},'revid':10}})])
            with patch.object(c.requests,'Session',return_value=second): final=self.run_config(path,resume=True)
            self.assertEqual(final['actual']['download_attempts'],1)
            self.assertEqual(final['actual']['bytes_received_including_failed'],2)
            self.assertEqual(final['sources'][0]['download_status'],'interrupted_preserved')
            self.assertEqual((raw[0].read_bytes(),raw[0].stat().st_mtime_ns),before)
            self.assertEqual(len(second.calls),1)
            self.assertEqual(second.calls[0][1]['params']['action'],'parse')

    def test_search_cooldown_is_incrementally_saved_with_query_cursor(self):
        self.assertTrue(hasattr(c,'CommonsCooldown'),'checkpoint implementation missing')
        first=Session([Response(429,headers={'Retry-After':'900'})])
        with tempfile.TemporaryDirectory() as tmp,patch.object(c.time,'time',return_value=1000),patch.object(c.requests,'Session',return_value=first):
            path=Path(tmp)/'batch';manifest=self.run_config(path)
            self.assertEqual(manifest['status'],'paused_cooldown')
            state=json.loads((path/'acquisition-checkpoint.json').read_text())
            self.assertEqual(state['query_index'],0)
            self.assertEqual(state['retry_at'],1900)
            self.assertEqual(manifest['actual']['download_attempts'],0)


if __name__=='__main__': unittest.main()

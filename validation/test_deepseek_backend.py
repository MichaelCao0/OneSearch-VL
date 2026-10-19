"""Request integrity and original native AV loop integration; no paid calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from research.av_search.deepseek_backend import DeepSeekSearchBackend, SearchError, public_url, read_private_key

KEY = "fixture-key-do-not-use"


def hit():
    return {"title": "Example", "url": "https://example.com/evidence", "snippet": "short",
            "summary": "Source-supported detail.", "content": "", "rerank_score": 2.1,
            "authority": 3, "date": None, "site_icon": None, "site_name": "Example"}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.calls = []

    def tearDown(self): self.tmp.cleanup()

    def make(self, result=None, **kw):
        def transport(endpoint, payload):
            self.calls.append((endpoint, payload))
            return result or (200, json.dumps({"results": [hit()]}).encode())
        return DeepSeekSearchBackend(self.root/'run', api_key=KEY, transport=transport,
                                     rate_dir=self.root/'rate', **kw)

    def test_search_normalization_and_success_replay(self):
        b = self.make(original_question="Original question?")
        self.assertIn('URL: https://example.com/evidence', b.text_search('clue'))
        self.assertIn('Source-supported detail.', b.text_search('clue'))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1]['ori_question'], 'Original question?')
        self.assertEqual(b.summary()['current_process_cache_hits'], 1)
        self.assertAlmostEqual(b.summary()['known_estimated_yuan'], .005)

    def test_invalid_query_and_count_send_nothing(self):
        b = self.make()
        for q in ['', '  ', 'x'*129, 3]:
            with self.assertRaises(ValueError): b.search(q)
        for n in [True, 0, 51, 1.2]:
            with self.assertRaises(ValueError): b.search('valid', n)
        self.assertFalse(self.calls)

    def test_unicode_length_and_no_silent_truncation(self):
        b = self.make(original_question='q'*129)
        b.search('中'*128)
        self.assertEqual(self.calls[0][1]['query'], '中'*128)
        self.assertNotIn('ori_question', self.calls[0][1])

    def test_no_results_is_valid_no_charge(self):
        b = self.make((200, b'{"results":[]}'))
        self.assertIn('No relevant sources', b.text_search('empty'))
        self.assertEqual(b.summary()['nonempty_requests'], 0)
        self.assertEqual(b.summary()['known_estimated_yuan'], 0)

    def test_fetch_is_markdown_and_bounded(self):
        b = self.make((200, json.dumps({'title':'Page', 'content':'a'*21000}).encode()))
        out = b.visit_webpage('https://example.com/page')
        self.assertIn('fetched_web_markdown', out)
        self.assertIn('truncated', out)
        self.assertLess(len(out), 20500)

    def test_private_urls_and_credentials_rejected(self):
        b = self.make()
        for u in ['file:///tmp/x', 'http://127.0.0.1/x', 'http://[::1]/', 'https://u:p@example.com',
                  'http://localhost/x', 'http://service.internal/', 'https://example.com/a\nb']:
            with self.assertRaises(ValueError): b.fetch(u)
        self.assertFalse(self.calls)

    def test_auth_error_sanitized_and_never_replayed(self):
        b = self.make((401, ('{"error":{"message":"'+KEY+'"}}').encode()))
        for _ in range(2):
            with self.assertRaises(SearchError) as ctx: b.search('bad-auth')
            self.assertNotIn(KEY, str(ctx.exception))
        self.assertEqual(len(self.calls), 1)
        for p in (self.root/'run').rglob('*'):
            if p.is_file(): self.assertNotIn(KEY.encode(), p.read_bytes())

    def test_uncertain_request_not_replayed_after_recovery(self):
        b = self.make(); b.search('clue')
        p = next((self.root/'run/requests').glob('*/receipt.json'))
        v = json.loads(p.read_text()); v['status'] = 'started'; p.write_text(json.dumps(v))
        with self.assertRaises(SearchError): b.search('clue')
        self.assertEqual(len(self.calls), 1)

    def test_failed_response_schema_not_counted_success(self):
        b = self.make((200, b'{"something_else":[]}'))
        with self.assertRaises(SearchError): b.search('clue')
        self.assertEqual(b.summary()['completed_requests'], 0)

    def test_budget_survives_client_restart(self):
        b = self.make(max_requests=1); b.search('one')
        b = self.make(max_requests=1)
        with self.assertRaisesRegex(SearchError, 'budget'): b.search('two')
        b.search('one')
        self.assertEqual(len(self.calls), 1)

    def test_cache_hash_change_detected(self):
        b = self.make(); b.search('one')
        p = next((self.root/'run/requests').glob('*/response.json')); p.write_text('{}')
        with self.assertRaisesRegex(SearchError, 'changed'): b.search('one')

    def test_config_change_rejected(self):
        self.make()
        with self.assertRaisesRegex(ValueError, 'identity'): self.make(need_content=True)

    def test_private_file_mode_enforced(self):
        p = self.root/'key.env'; p.write_text('DEEPSEEK_API_KEY='+KEY+'\n'); p.chmod(0o600)
        self.assertEqual(read_private_key(p), KEY)
        p.chmod(0o644)
        with self.assertRaises(ValueError): read_private_key(p)

    def test_real_native_loop_uses_backend_and_preserves_av(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'onesearch_vl'))
        from onesearch_infer.pipeline import process_native_av_case
        from onesearch_infer.runners import _wrap_gemini, InferenceConfig
        media = self.root/'input.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=56x56:rate=12',
                        '-f','lavfi','-i','sine=frequency=440:sample_rate=16000','-t','1',
                        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(media)],check=True)
        def transport(endpoint, payload):
            self.calls.append((endpoint,payload))
            body = {'results':[hit()]} if endpoint == 'search' else {'title':'Page','content':'Retrieved fact.'}
            return 200,json.dumps(body).encode()
        b = self.make(); b.transport = transport
        class Scripted:
            supports_audio_video = True
            spec = SimpleNamespace(supports_video=True)
            histories = []
            def infer(self, contents, **kw):
                self.histories.append(json.loads(json.dumps(contents)))
                actions = ['<tool_call>{"name":"text_search","arguments":{"q":"clue"}}</tool_call>',
                           '<tool_call>{"name":"visit_webpage","arguments":{"url":"https://example.com/evidence"}}</tool_call>',
                           '<response>fixture</response>']
                return _wrap_gemini(actions[len(self.histories)-1], 'STOP', {}, 'fixture')
        runner = Scripted()
        with patch.dict(os.environ, {'ONESEARCH_DISABLE_RETRIEVAL':'0','AV_AGENT_MAX_TURNS':'4'}):
            trace = process_native_av_case(question='Fixture?', source_path=str(media), runner=runner,
                     output_dir=str(self.root/'episode'),case_id='fixture',search_backend=b,
                     inference_cfg=InferenceConfig(max_tokens=128))
        self.assertEqual([c[0] for c in self.calls], ['search','fetch'])
        self.assertIn('Retrieved fact.', json.dumps(runner.histories[-1]))
        self.assertTrue(runner.histories[-1][0]['parts'][0]['video_url']['has_audio'])
        self.assertEqual(len(trace['turns']), 3)


    def test_full_search_body_reaches_tool_observation(self):
        row = hit(); row['content'] = 'Unique full page evidence.'
        b = self.make((200, json.dumps({'results':[row]}).encode()), need_content=True)
        out = b.text_search('clue')
        self.assertTrue(self.calls[0][1]['need_content'])
        self.assertIn('Unique full page evidence.', out)
        self.assertIn('web_search_summary_and_body', out)
        self.assertIn('Source-supported detail.', out)

    def test_search_body_budget_preserves_all_result_urls(self):
        rows = []
        for i in range(8):
            row = hit(); row['url'] = 'https://example.com/page'+str(i); row['content'] = 'X'*10000
            rows.append(row)
        b = self.make((200, json.dumps({'results':rows}).encode()), need_content=True)
        out = b.text_search('clue', top_k=8)
        self.assertLessEqual(out.count('X'), 12000)
        self.assertIn('https://example.com/page7', out)
        self.assertIn('truncated', out)
        self.assertLess(len(out), 16000)

    def test_summary_mode_preserves_legacy_observation(self):
        row = hit(); row['content'] = 'Body not requested.'
        b = self.make((200, json.dumps({'results':[row]}).encode()))
        out = b.text_search('clue')
        self.assertFalse(self.calls[0][1]['need_content'])
        self.assertIn('web_search_summary', out)
        self.assertNotIn('Body not requested.', out)


if __name__ == '__main__': unittest.main()

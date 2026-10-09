"""Mixed text/media visibility, exact quotes and operation-level cost contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from research.av_agent import tools as public_tools
import unittest

from research.av_agent.adapter import EpisodeTools
from research.av_data.web_sources import ingest_web_snapshot
from research.av_search.corpus import build_corpus


class MixedEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.html = self.root / 'page.html'
        self.html.write_text('<html><body>Publisher says rhythmic motion; not an independent observation.</body></html>')
        self.record = ingest_web_snapshot(self.html, 'https://example.org/page', self.root/'assets',
                                          public_metadata={'title':'Publisher rhythmic motion'},
                                          fetched_at='2026-10-10T00:00:00Z')
        self.sid = self.record['source_id']
        body = self.record['body_text']
        self.doc = {k:self.record[k] for k in ('source_id','origin_url','asset_path','asset_sha256','media_type','duration_s','public_metadata')}
        self.doc.update(text=body, text_kind='web_body')
        self.index = self.root / 'corpus'; build_corpus([self.doc], self.index, 'mixed-unit', 'exploration')
        self.scope = {'corpus_id':'mixed-unit','split':'exploration','source_ids':[self.sid]}

    def episode(self, record=None, suffix='episode'):
        return EpisodeTools(self.index, {self.sid:record or self.record}, self.scope, self.root/suffix)

    def call(self, tools, name, **args):
        return tools.dispatch(json.dumps({'name':name,'arguments':args}))

    def test_web_snapshot_search_and_exact_quote_remain_scoped(self):
        tools = self.episode()
        search = self.call(tools, 'search_sources', query='rhythmic')
        hit = search['results'][0]
        self.assertEqual(hit['excerpt'], self.record['body_text'][hit['quote_start']:hit['quote_end']])
        receipt = self.call(tools, 'read_text', source_id=self.sid, start_char=0, end_char=30)
        self.assertEqual(receipt['quote']['text'], self.record['body_text'][:30])
        self.assertEqual(receipt['text_sha256'], self.record['body_text_sha256'])
        self.assertEqual(receipt['manifest_sha256'], search['manifest_sha256'])
        self.assertNotIn(str(self.root), json.dumps(receipt))
        self.assertEqual(receipt['interpretation'], 'frozen_source_text_not_semantic_verification')

    def test_corpus_aba_replacement_during_read_cannot_change_pinned_quote(self):
        tools = self.episode()
        replacement = self.root/'replacement'
        build_corpus([{**self.doc,'text':'REPLACEMENT ' * 20}],replacement,'mixed-unit','exploration')
        original = public_tools.read_text
        stash = self.root/'stash'
        def swap_then_read(index_dir, *args):
            self.index.rename(stash); replacement.rename(self.index)
            try:
                return original(index_dir, *args)
            finally:
                self.index.rename(replacement); stash.rename(self.index)
        with patch('research.av_agent.tools.read_text', side_effect=swap_then_read):
            receipt = self.call(tools, 'read_text', source_id=self.sid, start_char=0, end_char=30)
        self.assertEqual(receipt['status'],'ok')
        self.assertEqual(receipt['quote']['text'],self.record['body_text'][:30])
        self.assertNotIn('REPLACEMENT', receipt['quote']['text'])

    def test_web_source_cannot_enter_media_decoder(self):
        tools = self.episode()
        with self.assertRaises(ValueError):
            self.call(tools, 'inspect_av', source_id=self.sid, start_s=0, end_s=1)
        self.assertEqual(tools.costs['tool_calls'], 0)

    def test_quote_bounds_extra_arguments_and_foreign_source_rejected(self):
        tools = self.episode()
        for args in ({'source_id':self.sid,'start_char':True,'end_char':10},
                     {'source_id':self.sid,'start_char':0,'end_char':999},
                     {'source_id':self.sid,'start_char':0,'end_char':10,'path':'private'},
                     {'source_id':'src_'+'f'*24,'start_char':0,'end_char':10}):
            with self.subTest(args=args), self.assertRaises(ValueError):self.call(tools,'read_text',**args)
        self.assertEqual(tools.costs['tool_calls'],0)

    def test_body_or_public_identity_mismatch_rejected(self):
        for n, change in enumerate(({'body_text':'forged body'}, {'origin_url':'https://example.org/other'},
                                    {'public_metadata':{'title':'Changed title'}})):
            record = {**copy.deepcopy(self.record), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):self.episode(record,str(n))


if __name__ == '__main__':unittest.main()

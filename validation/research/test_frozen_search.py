"""Frozen search must ground receipts in actual bytes, and respect episode scope."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from research.av_search.corpus import build_corpus, visible_corpus
    from research.av_search.retrieval import search_sources, get_source
except ImportError:
    build_corpus = visible_corpus = search_sources = get_source = None

class FrozenSearchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(build_corpus, 'Frozen corpus production interface missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docs=[]
        for body in ['The original recording shows a hammer striking metal. The sound repeats.', 'A second recording shows rain falling into water.', 'This publisher page describes a hammer workshop.']:
            raw=body.encode(); digest=hashlib.sha256(raw).hexdigest()
            p=self.root/digest;p.write_bytes(raw)
            self.docs.append({'source_id':'src_'+digest[:24], 'origin_url':'https://example.org/'+digest[:8], 'asset_sha256':digest,'asset_path':str(p),'text':body,'text_kind':'web_body','public_metadata':{'title':'Public source','description':body},'media_type':'text/html'})
        self.index=self.root/'frozen'
        self.manifest=build_corpus(self.docs,self.index,'pilot-exploration-v1','exploration')
        self.scope={'corpus_id':'pilot-exploration-v1','split':'exploration','source_ids':[d['source_id'] for d in self.docs]}

    def test_actual_search_returns_versioned_quote_and_no_private_paths(self):
        r=search_sources(self.index,'hammer',self.scope,top_k=5)
        self.assertEqual(len(r['results']),2)
        for hit in r['results']:
            d=next(x for x in self.docs if x['source_id']==hit['source_id'])
            self.assertEqual(d['text'][hit['quote_start']:hit['quote_end']],hit['excerpt'])
            self.assertEqual(hashlib.sha256(d['text'].encode()).hexdigest(),hit['text_sha256'])
        self.assertNotIn('asset_path',json.dumps(r));self.assertNotIn(self.tmp.name,json.dumps(r))
        self.assertEqual(r['logical_cost']['search_calls'],1)
        self.assertEqual(r['actual_cost']['provider_requests'],0)

    def test_fixed_scope_rejects_split_corpus_and_source_escalation(self):
        for field,value in [('split','test'),('corpus_id','private-gold'),('source_ids',['src_'+'f'*24])]:
            scope={**self.scope,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):
                search_sources(self.index,'hammer',scope)

    def test_restricted_episode_never_returns_unseen_sources(self):
        scope={**self.scope,'source_ids':[self.docs[1]['source_id']]}
        self.assertEqual(search_sources(self.index,'hammer',scope)['results'],[])
        with self.assertRaises(ValueError):get_source(self.index,self.docs[0]['source_id'],scope)

    def test_repeated_query_is_deterministic_and_global(self):
        a=search_sources(self.index,'hammer',self.scope)
        b=search_sources(self.index,'hammer',self.scope)
        self.assertEqual(a['results'],b['results'])
        self.assertEqual(a['corpus_hash'],b['corpus_hash'])
        self.assertEqual(a['index_sha256'],b['index_sha256'])
        public=visible_corpus(self.index,self.scope)
        self.assertEqual(set(public['sources']),set(self.scope['source_ids']))
        self.assertNotIn('asset_path',json.dumps(public))

    def test_missing_tampered_assets_and_private_fields_fail_ingestion(self):
        bad=[dict(self.docs[0]) for _ in range(4)]
        bad[0]['asset_path']=str(self.root/'missing')
        bad[1]['asset_sha256']='0'*64
        bad[2]['public_metadata']={'title':'valid','gold_answer':'hidden'}
        bad[3]['text_kind']='non_speech_audio_summary'
        for i,d in enumerate(bad):
            with self.subTest(i=i),self.assertRaises((ValueError,FileNotFoundError)):
                build_corpus([d],self.root/('bad'+str(i)),'pilot','exploration')

    def test_quote_text_and_index_tampering_are_detected(self):
        con=self.index/'index.sqlite'
        con.write_bytes(con.read_bytes()+b'tamper')
        with self.assertRaises(ValueError):search_sources(self.index,'hammer',self.scope)

    def test_empty_results_and_fts_metacharacters_do_not_inject_sql(self):
        r=search_sources(self.index,'unmatchedzz',self.scope)
        self.assertEqual(r['status'],'no_results')
        r=search_sources(self.index,'hammer" OR 1=1 --',self.scope)
        self.assertIsInstance(r['results'],list)
        with self.assertRaises(ValueError):search_sources(self.index,'',self.scope)
        with self.assertRaises(ValueError):search_sources(self.index,'hammer',self.scope,top_k=0)

    def test_frozen_output_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):build_corpus(self.docs,self.index,'pilot','exploration')

if __name__=='__main__':unittest.main()

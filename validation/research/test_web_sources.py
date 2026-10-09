"""Web snapshots must replay actual bytes/text and reject forged identities."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

class WebSourceTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('research.av_data.web_sources'), 'web snapshot API missing')
        return importlib.import_module('research.av_data.web_sources')

    def snapshot(self, root):
        raw=root/'page.html'
        raw.write_bytes(b'<html><head><title>Hidden title</title></head><body><p>Motor &amp; pump</p><script>fake answer</script><p>Own work.</p></body></html>')
        return self.api().ingest_web_snapshot(raw,'https://example.org/page',root/'assets',
            public_metadata={'title':'Publisher page'}, fetched_at='2026-10-09T12:00:00+00:00',
            snapshot_identity={'transport':'http','request_url':'https://example.org/page','publisher_revision':'123'})

    def test_snapshot_preserves_raw_bytes_and_deterministic_body_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.snapshot(root)
            self.assertEqual(Path(r['asset_path']).read_bytes(),(root/'page.html').read_bytes())
            self.assertEqual(r['body_text'],'Motor & pump Own work.')
            self.assertEqual(r['body_text_sha256'],hashlib.sha256(r['body_text'].encode()).hexdigest())
            self.assertEqual(r['source_id'],'src_'+r['asset_sha256'][:24])
            self.assertEqual(r['media_type'],'webpage');self.assertIsNone(r['duration_s'])
            self.assertEqual(r['fetched_at'],'2026-10-09T12:00:00+00:00')
            self.assertEqual(self.api().validate_web_source(r),[])

    def test_raw_and_body_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.snapshot(Path(d));api=self.api()
            changed=copy.deepcopy(r);changed['body_text']='Unobserved summary'
            self.assertTrue(api.validate_web_source(changed))
            changed=copy.deepcopy(r);changed['body_text']='Unobserved summary';changed['body_text_sha256']=hashlib.sha256(changed['body_text'].encode()).hexdigest()
            self.assertTrue(api.validate_web_source(changed))
            Path(r['asset_path']).write_bytes(b'changed')
            self.assertTrue(api.validate_web_source(r))

    def test_foreign_path_even_with_identical_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.snapshot(root);other=root/'foreign.html'
            other.write_bytes(Path(r['asset_path']).read_bytes());r['asset_path']=str(other)
            self.assertTrue(self.api().validate_web_source(r))

    def test_symlink_store_and_gold_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'page.html';raw.write_text('<p>Actual page</p>')
            with self.assertRaises(ValueError):self.api().ingest_web_snapshot(raw,'https://example.org',root/'assets',public_metadata={'gold':'answer'})
            real=root/'real';real.mkdir();link=root/'link';link.symlink_to(real,target_is_directory=True)
            with self.assertRaises(ValueError):self.api().ingest_web_snapshot(raw,'https://example.org',link)

    def test_extraction_identity_and_opaque_id_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.snapshot(Path(d));api=self.api()
            for key,value in [('source_id','src_'+'0'*24),('duration_s',1.0),('body_text_sha256','0'*64)]:
                changed=copy.deepcopy(r);changed[key]=value;self.assertTrue(api.validate_web_source(changed))
            changed=copy.deepcopy(r);changed['extraction']['revision']='unobserved';self.assertTrue(api.validate_web_source(changed))

    def test_quote_is_exact_and_tampered_or_invalid_bounds_fail(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.snapshot(Path(d));api=self.api();quote=api.quote_web_source(r,0,12)
            self.assertEqual(quote['quote'],'Motor & pump');self.assertEqual(quote['text_sha256'],r['body_text_sha256'])
            for bounds in [(True,12),(-1,5),(0,999),(12,5)]:
                with self.assertRaises(ValueError):api.quote_web_source(r,*bounds)
            r['body_text']='summary'
            with self.assertRaises(ValueError):api.quote_web_source(r,0,2)

    def test_corpus_web_body_hash_matches_snapshot_and_public_has_no_path(self):
        from research.av_search.corpus import build_corpus,visible_corpus
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.snapshot(root);doc=self.api().web_corpus_document(r)
            manifest=build_corpus([doc],root/'corpus','web-fixture','exploration')
            scope={'corpus_id':'web-fixture','split':'exploration','source_ids':manifest['source_ids']}
            public=visible_corpus(root/'corpus',scope)['sources'][r['source_id']]
            self.assertEqual(public['text'],r['body_text']);self.assertEqual(public['text_sha256'],r['body_text_sha256'])
            self.assertNotIn('asset_path',public);self.assertNotIn('snapshot_identity',public)

if __name__=='__main__':unittest.main()

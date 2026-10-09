"""Actual media receipts, fixed visibility, strict dispatch and bounded work."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

from research.av_data.ingest import ingest_media
from research.av_search.corpus import build_corpus
try:
    from research.av_agent.adapter import EpisodeTools, strict_json
except ImportError:
    EpisodeTools = strict_json = None

class AVToolsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(EpisodeTools, 'Fixed-scope AV tools missing')
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        media = self.root / 'input.mkv'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=red:s=64x64:r=10:d=2','-f','lavfi','-i','sine=frequency=440:sample_rate=8000:duration=2','-c:v','ffv1','-c:a','pcm_s16le',str(media)], check=True)
        self.source = ingest_media(media,'https://example.org/recording',self.root/'assets',{'title':'Public red sine recording'})
        self.sid = self.source['source_id']
        self.doc = {k:self.source[k] for k in ('source_id','origin_url','asset_sha256','asset_path','media_type','duration_s','public_metadata')}
        self.doc.update(text='Public red sine recording, an artificial timing fixture.',text_kind='public_metadata')
        self.index = self.root/'corpus';build_corpus([self.doc],self.index,'unit-exploration','exploration')
        self.scope={'corpus_id':'unit-exploration','split':'exploration','source_ids':[self.sid]}
        self.tools=EpisodeTools(self.index,{self.sid:self.source},self.scope,self.root/'episode',max_calls=8,max_av_seconds=3)

    def action(self,name,**kwargs):
        return self.tools.dispatch(json.dumps({'name':name,'arguments':kwargs}))

    def test_strict_duplicate_keys_nonfinite_and_schema_fail_before_dispatch(self):
        bad=['{"name":"search_sources","name":"inspect_av","arguments":{}}',
             '{"name":"search_sources","arguments":{"query":"x","query":"y"}}',
             '{"name":"inspect_av","arguments":{"start_s":NaN}}',
             '{"name":"search_sources","arguments":{"query":"red"},"scope":{}}']
        for raw in bad:
            with self.subTest(raw=raw),self.assertRaises(ValueError):self.tools.dispatch(raw)
        self.assertEqual(self.tools.costs['tool_calls'],0)

    def test_actual_clip_has_bytes_timing_counts_without_semantic_output(self):
        receipt=self.action('inspect_av',source_id=self.sid,start_s=.2,end_s=1.2)
        self.assertEqual(receipt['status'],'media_ready_observer_unavailable')
        self.assertEqual(receipt['asset_sha256'],self.source['asset_sha256'])
        self.assertEqual(receipt['original_interval'],{'start_s':.2,'end_s':1.2})
        self.assertFalse(receipt['capabilities']['raw_av_observer_ready'])
        self.assertFalse(receipt['capabilities']['full_decode_work_accounting_ready'])
        self.assertEqual(receipt['actual_cost']['count_scope'],'prepared_clip_only')
        self.assertGreater(receipt['actual_cost']['decoded_audio_samples'],0)
        self.assertGreater(receipt['actual_cost']['returned_video_frames'],0)
        self.assertNotIn(str(self.root),json.dumps(receipt))
        self.assertNotIn('raw_output',receipt)
        trusted=json.loads((self.root/'episode'/'events.jsonl').read_text().splitlines()[0])
        derived=trusted['private_derived_record']
        self.assertEqual(hashlib.sha256(Path(derived['output_path']).read_bytes()).hexdigest(),receipt['output_sha256'])
        self.assertAlmostEqual(derived['mapping']['output_to_source']['offset_s'],.2)

    def test_actor_cannot_change_scope_or_access_foreign_source(self):
        with self.assertRaises(ValueError):self.action('search_sources',query='red',scope=self.scope)
        with self.assertRaises(ValueError):self.action('provenance',source_id='src_'+'f'*24)
        self.assertEqual(self.tools.costs['tool_calls'],0)

    def test_scope_and_registry_are_copied_at_construction(self):
        self.scope['source_ids'].clear();self.source['asset_path']='/does/not/exist'
        hit=self.action('search_sources',query='red')['results'][0]
        self.assertEqual(hit['source_id'],self.sid)
        self.assertEqual(self.action('inspect_av',source_id=self.sid,start_s=0,end_s=.5)['status'],'media_ready_observer_unavailable')

    def test_search_provenance_compare_remain_attributable_without_independence_labels(self):
        hit=self.action('search_sources',query='red')['results'][0]
        self.assertEqual(hit['excerpt'],self.doc['text'][hit['quote_start']:hit['quote_end']])
        prov=self.action('provenance',source_id=self.sid)
        self.assertEqual(prov['origin_url'],self.doc['origin_url'])
        result=self.action('compare_sources',source_ids=[self.sid])
        self.assertEqual(result['relationship_status'],'unknown')
        self.assertNotIn('asset_path',json.dumps(result))

    def test_caps_reject_work_before_extra_media_reads(self):
        self.action('inspect_av',source_id=self.sid,start_s=0,end_s=2)
        with self.assertRaises(ValueError):self.action('inspect_av',source_id=self.sid,start_s=0,end_s=2)
        self.assertEqual(self.tools.costs['requested_av_seconds'],2)
        bounded=EpisodeTools(self.index,{self.sid:self.source},{'corpus_id':'unit-exploration','split':'exploration','source_ids':[self.sid]},self.root/'bounded',max_calls=1)
        bounded.dispatch('{"name":"search_sources","arguments":{"query":"red"}}')
        with self.assertRaises(ValueError):bounded.dispatch('{"name":"search_sources","arguments":{"query":"red"}}')

    def test_bad_interval_and_registry_identity_rejected(self):
        for start,end in ((-1,1),(1,1),(0,3),(True,1)):
            with self.subTest(start=start,end=end),self.assertRaises(ValueError):self.action('inspect_av',source_id=self.sid,start_s=start,end_s=end)
        other={**self.source,'asset_sha256':'0'*64}
        with self.assertRaises(ValueError):EpisodeTools(self.index,{self.sid:other},self.scope,self.root/'invalid')

    def test_source_tamper_is_recorded_as_failed_without_claiming_observation(self):
        Path(self.source['asset_path']).write_bytes(b'tamper')
        receipt=self.action('inspect_av',source_id=self.sid,start_s=0,end_s=1)
        self.assertEqual(receipt['status'],'tool_failed')
        self.assertEqual(receipt['error_code'],'ValueError')
        self.assertNotIn('output_sha256',receipt)
        self.assertNotIn(str(self.root),json.dumps(receipt))

    def test_consistent_corpus_replacement_after_episode_init_is_rejected(self):
        replacement=self.root/'replacement'
        changed={**self.doc,'public_metadata':{'title':'Changed public snapshot'}}
        build_corpus([changed],replacement,'unit-exploration','exploration')
        shutil.rmtree(self.index)
        replacement.rename(self.index)
        with self.assertRaises(ValueError):self.action('search_sources',query='red')
        self.assertEqual(self.tools.costs['tool_calls'],0)

    def test_extreme_integer_intervals_and_caps_raise_contract_value_error(self):
        with self.assertRaises(ValueError):self.action('inspect_av',source_id=self.sid,start_s=0,end_s=10**999)
        with self.assertRaises(ValueError):EpisodeTools(self.index,{self.sid:self.source},self.scope,self.root/'giant',max_av_seconds=10**999)

    def test_registry_public_duration_and_media_type_must_match_corpus(self):
        for n,changed in enumerate(({'duration_s':200.0},{'media_type':'audio'})):
            index=self.root/('public-mismatch-'+str(n))
            build_corpus([{**self.doc,**changed}],index,'unit-exploration','exploration')
            with self.assertRaises(ValueError):EpisodeTools(index,{self.sid:self.source},self.scope,self.root/('bad-public-'+str(n)))

    def test_media_receipt_pins_same_corpus_identity_as_search(self):
        search=self.action('search_sources',query='red')
        media=self.action('inspect_av',source_id=self.sid,start_s=0,end_s=.5)
        self.assertEqual(media['corpus_id'],search['corpus_id'])
        self.assertEqual(media['corpus_hash'],search['corpus_hash'])
        self.assertEqual(media['manifest_sha256'],search['manifest_sha256'])
        header=json.loads((self.root/'episode'/'episode.json').read_text())
        self.assertEqual(header['manifest_sha256'],media['manifest_sha256'])

    def test_run_output_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):EpisodeTools(self.index,{self.sid:self.source},self.scope,self.root/'episode')

if __name__=='__main__':unittest.main()

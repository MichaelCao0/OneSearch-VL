"""Benchmark reservations must stay independent of teacher attempts/train views."""
import unittest
from research.av_train.reserved_benchmark import select_reserved_families, read_training_views
from research.av_train.question_authoring import select_sources

class ReservedBenchmarkSelection(unittest.TestCase):
 def setUp(self):
  self.seeds=[dict(source_id=s,status='candidate_unreviewed_av',sha256=s,origin_url='https://example.org/'+s,candidate_domain=d+'_0',description_page_html_path='/fixture') for s,d in [('a','bird'),('alias','bird'),('b','machine'),('c','music'),('train','event')]]
  self.families=[dict(candidate_family_id=f,source_ids=ids) for f,ids in [('fa',['a','alias']),('fb',['b']),('fc',['c']),('ft',['train'])]]
  self.reserved=[self.seeds[i] for i in [0,2,3]]
 def choose(self,**kw):
  return select_reserved_families(self.reserved,self.seeds,self.families,limit=3,**kw)
 def test_only_reserved_families_can_reach_real_author_selector(self):
  result=self.choose();chosen,audit=select_sources(self.seeds,self.families,[],result['used_source_ids'],100)
  self.assertEqual({x['candidate_family_id'] for x in chosen},{'fa','fb','fc'})
  self.assertNotIn('train',{x['source_id'] for x in chosen})
 def test_paid_alias_blocks_entire_recording_family(self):
  result=self.choose(attempted_source_ids={'alias'})
  self.assertEqual({x['candidate_family_id'] for x in result['selected']},{'fb','fc'})
 def test_observed_training_url_blocks_percent_encoded_equivalent(self):
  self.seeds[0]['origin_url']='https://example.org/one%20bird/'
  result=self.choose(blocked_urls={'https://example.org/one_bird'})
  self.assertNotIn('fa',{x['candidate_family_id'] for x in result['selected']})
 def test_training_media_hash_blocks_alias(self):
  result=self.choose(blocked_hashes={'alias'})
  self.assertNotIn('fa',{x['candidate_family_id'] for x in result['selected']})
 def test_round_robin_is_bounded_and_keeps_recording_families_unique(self):
  result=select_reserved_families(self.reserved,self.seeds,self.families,limit=2)
  self.assertEqual(len(result['selected']),2)
  self.assertEqual(len({x['candidate_family_id'] for x in result['selected']}),2)
  self.assertFalse(result['human_gold'])
 def test_unknown_family_cannot_be_claimed_independent(self):
  with self.assertRaises(ValueError):select_reserved_families(self.reserved,self.seeds,self.families[:-2],limit=3)
 def test_non_reserved_train_source_is_never_promoted(self):
  result=self.choose()
  self.assertNotIn('train',{x['source_id'] for x in result['selected']})
  self.assertIn('train',result['used_source_ids'])

class TrainingViewIsolation(unittest.TestCase):
 def test_uses_hash_bound_lineage_and_actual_url_and_audio_receipts(self):
  from tempfile import TemporaryDirectory
  import json,hashlib
  from pathlib import Path
  with TemporaryDirectory() as tmp:
   d=Path(tmp);trace=d/'actual.json';trace.write_text(json.dumps({'tool_output':'https://example.org/one%20bird'}))
   (d/'lineage.private.json').write_text(json.dumps([{'trajectory_path':str(trace),'trajectory_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'audio_receipts':[{'source_sha256':'abc'}]}]))
   got=read_training_views([d]);self.assertIn('https://example.org/one_bird',got['observed_urls']);self.assertIn('abc',got['media_hashes'])
   trace.write_text('{}')
   with self.assertRaises(ValueError):read_training_views([d])

if __name__=='__main__':unittest.main()


class ReadyCohortQueue(unittest.TestCase):
 def test_waits_for_complete_reservation_and_skips_consumed_cohorts(self):
  import tempfile
  from pathlib import Path
  from research.av_train.reserved_benchmark import ready_reserved_cohorts
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name in ['cohort-00','cohort-01','cohort-02']:
    c=root/name;c.mkdir();(c/'inventory').mkdir()
    for f in ['combined-manifest.private.json','reserved-registry.private.jsonl','inventory/inventory.private.json']:(c/f).write_text('{}')
    if name!='cohort-02':(c/'benchmark-reserved-sources.private.json').write_text('[]')
   self.assertEqual([c.name for c in ready_reserved_cohorts(root)],['cohort-00','cohort-01'])
   self.assertEqual([c.name for c in ready_reserved_cohorts(root,completed={'cohort-00'})],['cohort-01'])
   self.assertEqual([c.name for c in ready_reserved_cohorts(root,cohort_name='cohort-01')],['cohort-01'])

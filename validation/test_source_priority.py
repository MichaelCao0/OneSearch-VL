import unittest
from research.av_train.source_priority import candidate_stratum, schedule_sources, reserve_candidates

class SourcePriorityTests(unittest.TestCase):
 def row(self,sid,domain='machine_1',query='',title=''):
  return dict(source_id=sid,candidate_domain=domain,query=query,public_metadata={'title':title},status='candidate_unreviewed_av',description_page_html_path='/page',origin_url='https://example.com/'+sid,sha256='hash'+sid)
 def test_query_labels_remain_sampling_suggestions(self):
  self.assertEqual(candidate_stratum(self.row('a','event_1','ice hockey overtime')),'sports')
  self.assertEqual(candidate_stratum(self.row('b','event_2','carnival parade')),'live')
  self.assertEqual(candidate_stratum(self.row('c','event_3','waterfall sound')),'other')
  self.assertEqual(candidate_stratum(self.row('d','music_2','piano')),'music')
  self.assertEqual(candidate_stratum(self.row('e','machine_2','motorcycle engine','Motorcycle stunt show')),'live')
 def test_weighted_queue_is_deterministic_complete_and_prioritized(self):
  rows=[self.row('o'+str(i)) for i in range(20)]+[self.row('m'+str(i),'music_1') for i in range(10)]+[self.row('s'+str(i),'sports_1') for i in range(10)]+[self.row('l'+str(i),'live_1') for i in range(10)]
  got=schedule_sources(rows)
  self.assertEqual(len(got),50)
  self.assertEqual(len({r['source_id'] for r in got}),50)
  self.assertEqual([candidate_stratum(x) for x in got[:10]],['music','sports','live','music','sports','other','music','sports','live','other'])
  self.assertEqual(got,schedule_sources(list(reversed(rows))))
 def test_reservation_blocks_aliases_paid_and_actual_training_views(self):
  rows=[self.row(x,'music_1') for x in 'abcdef']
  families=[dict(candidate_family_id='f'+x,source_ids=[x]) for x in 'abcd']+[dict(candidate_family_id='fe',source_ids=['e','f'])]
  selected=reserve_candidates(rows,families,limit=10,blocked_source_ids=['a','e'],blocked_urls=[rows[1]['origin_url']],blocked_hashes=[rows[2]['sha256']])
  self.assertEqual([r['source_id'] for r in selected],['d'])
 def test_one_reservation_per_family(self):
  rows=[self.row(x,'sports_1') for x in 'abc']
  families=[dict(candidate_family_id='ab',source_ids=['a','b']),dict(candidate_family_id='c',source_ids=['c'])]
  self.assertEqual(len(reserve_candidates(rows,families,limit=8)),2)
 def test_unknown_family_rejected(self):
  with self.assertRaises(ValueError):reserve_candidates([self.row('x')],[],limit=2)

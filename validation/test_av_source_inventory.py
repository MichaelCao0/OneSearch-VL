import unittest
from research.av_data import source_inventory as s
class InventoryTests(unittest.TestCase):
 def row(self,sid,creator='recorder',title='bird singing take 1',sha=None):
  return {'source_id':sid,'asset_sha256':sha or sid,'origin_url':'https://example.com/'+sid,'creator':creator,'title':title,'video_fingerprints':[]}
 def test_candidate_family_ids_and_members_are_order_invariant(self):
  rows=[self.row('a'),self.row('b',title='bird singing take 2'),self.row('c',creator='another',title='steam engine')]
  pairs,groups=s.group_recordings(rows);pairs2,groups2=s.group_recordings(list(reversed(rows)))
  self.assertEqual(groups,groups2);self.assertEqual(len(groups),2)
  self.assertFalse(any(g['independence_verified'] for g in groups))
 def test_common_species_title_alone_does_not_merge_recordings(self):
  pairs,groups=s.group_recordings([self.row('a',creator='first'),self.row('b',creator='second')])
  self.assertEqual(len(groups),2)
 def test_exact_duplicate_bytes_merge_even_if_publisher_changes(self):
  pairs,groups=s.group_recordings([self.row('a',creator='first',sha='same'),self.row('b',creator='second',title='different',sha='same')])
  self.assertEqual(len(groups),1);self.assertIn('same_asset_sha256',pairs[0]['reasons'])
if __name__=='__main__':unittest.main()

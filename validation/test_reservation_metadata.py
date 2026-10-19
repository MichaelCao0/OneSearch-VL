import unittest
from research.av_train.reservation_metadata import enrich_reservations
class ReservationMetadata(unittest.TestCase):
 def test_sparse_reservation_keeps_identity_and_joins_verified_metadata(self):
  r={'source_id':'a','asset_sha256':'sha','origin_url':'https://x'}
  s={'source_id':'a','sha256':'sha','public_metadata':{'title':'Actual publisher title'}}
  out=enrich_reservations([r],[s]);self.assertEqual(out[0]['public_metadata'],s['public_metadata']);self.assertEqual(out[0]['asset_sha256'],'sha');self.assertNotIn('public_metadata',r)
 def test_existing_metadata_is_preserved(self):
  r={'source_id':'a','asset_sha256':'sha','public_metadata':{'title':'Frozen'}}
  self.assertEqual(enrich_reservations([r],[]),[r])
 def test_cannot_join_other_bytes_or_invent_missing_metadata(self):
  r={'source_id':'a','asset_sha256':'sha'}
  with self.assertRaises(ValueError):enrich_reservations([r],[{'source_id':'a','sha256':'other','public_metadata':{}}])
  with self.assertRaises(ValueError):enrich_reservations([r],[])

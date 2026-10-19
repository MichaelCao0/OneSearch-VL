import unittest
class SourceIsolationTests(unittest.TestCase):
 def test_public_metadata_and_web_body_removed_by_canonical_url(self):
  from research.av_train.source_isolation import partition_documents
  docs=[{'source_id':'a','origin_url':'https://example.org/A%20B','asset_sha256':'x'},{'source_id':'b','origin_url':'https://example.org/A B'},{'source_id':'c','origin_url':'https://example.org/A%20BC'}]
  kept,removed=partition_documents(docs,blocked_urls=['https://example.org/A B'])
  self.assertEqual([x['source_id'] for x in kept],['c']);self.assertEqual(len(removed),2)
 def test_recording_hash_and_family_source_ids_removed(self):
  from research.av_train.source_isolation import partition_documents
  docs=[{'source_id':'a','origin_url':'https://example.org/other','asset_sha256':'same'},{'source_id':'b','origin_url':'https://example.org/dup'},{'source_id':'c','origin_url':'https://example.org/clean'}]
  kept,_=partition_documents(docs,blocked_hashes=['same'],blocked_source_ids=['b'])
  self.assertEqual([x['source_id'] for x in kept],['c'])
if __name__=='__main__':unittest.main()

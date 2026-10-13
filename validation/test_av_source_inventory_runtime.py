import json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research.av_data import source_inventory as s
class InventoryRuntimeTests(unittest.TestCase):
 def test_matching_cached_fingerprints_skip_decode_but_verify_asset_hash(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);media=root/'video.mp4'
   subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=96x64:r=20:d=2','-c:v','libx264',str(media)],check=True,capture_output=True)
   source={'source_id':'s1','status':'candidate_unreviewed_av','description_page_html_path':'fixture','asset_path':str(media),'sha256':s.file_sha(media),'duration_s':2.,'page_id':1,'origin_url':'https://example.com/1','public_metadata':{'title':'test','creator':'fixture'},'candidate_domain':'test_1'}
   manifest=root/'manifest.json';manifest.write_text(json.dumps({'sources':[source]}))
   s.build_inventory([manifest],[],root/'first',workers=1)
   with patch('research.av_data.source_inventory.subprocess.check_output',side_effect=AssertionError('cached asset should not decode again')):
    report=s.build_inventory([manifest],[],root/'second',workers=1,cached_inventories=[root/'first/inventory.private.json'])
   self.assertEqual(report['cached_fingerprint_records'],1)
   self.assertEqual(json.loads((root/'second/inventory.private.json').read_text())['rows'][0]['video_fingerprints'],json.loads((root/'first/inventory.private.json').read_text())['rows'][0]['video_fingerprints'])
   media.write_bytes(b'TAMPERED')
   with self.assertRaises(ValueError):s.build_inventory([manifest],[],root/'third',workers=1,cached_inventories=[root/'first/inventory.private.json'])
if __name__=='__main__':unittest.main()

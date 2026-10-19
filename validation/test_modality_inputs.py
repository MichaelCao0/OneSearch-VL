from pathlib import Path
import tempfile,subprocess,json,unittest
from research.av_train.modality_inputs import MutedMedia,streams,video_packet_sha
class ModalityInputTests(unittest.TestCase):
 def exercise(self,suffix,video_codec,audio_codec):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/('input'+suffix)
   subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=blue:s=32x32:r=5','-f','lavfi','-i','anullsrc=r=16000:cl=mono','-t','0.4','-c:v',video_codec,'-c:a',audio_codec,'-shortest',str(source)],check=True,capture_output=True)
   muted=MutedMedia(root/'muted').video(source);self.assertEqual(streams(muted),['video']);self.assertEqual(video_packet_sha(source),video_packet_sha(muted))
   self.assertEqual(MutedMedia(root/'muted').video(source),muted)
   self.assertTrue(json.loads(next((root/'muted').glob('*.json')).read_text())['encoded_video_packets_identical'])
 def test_ffv1_matroska_and_receipt_reuse(self):self.exercise('.mkv','ffv1','pcm_s16le')
 def test_h264_mp4_and_receipt_reuse(self):self.exercise('.mp4','libx264','aac')

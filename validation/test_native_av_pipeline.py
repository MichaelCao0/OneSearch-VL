"""Exercise raw-AV tool media through the original multi-turn loop."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'onesearch_vl'))
from onesearch_infer.pipeline import process_native_av_case
from onesearch_infer.runners import InferenceConfig, _wrap_gemini


class ScriptedRunner:
    supports_audio_video = True
    spec = SimpleNamespace(supports_video=True)

    def __init__(self):
        self.histories = []

    def infer(self, contents, system_instruction=None, cfg=None):
        # Copies ensure later appends cannot make a missing earlier media pass.
        self.histories.append(json.loads(json.dumps(contents)))
        responses = [
            '<tool_call>{"name":"text_search","arguments":{"query":"mechanical sound","top_k":2}}</tool_call>',
            '<tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":0.2,"end_s":0.8}}</tool_call>',
            '<response>fixture answer</response>',
        ]
        return _wrap_gemini(responses[len(self.histories)-1], 'STOP', {}, 'fixture')


class FakeSearch:
    def text_search(self, **kwargs):
        self.query = kwargs['query']
        return 'Source https://example.invalid/source: mechanical device.'


class NativeAVPipelineTest(unittest.TestCase):
    def test_search_and_real_crop_media_survive_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp/'input.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=112x112:rate=12',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=16000','-t','1',
                '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(src)], check=True)
            runner = ScriptedRunner(); backend = FakeSearch()
            result = process_native_av_case(question='What device is this?', source_path=str(src), runner=runner,
                output_dir=str(tmp/'out'),case_id='fixture-av',search_backend=backend,
                inference_cfg=InferenceConfig(max_tokens=64))
            self.assertEqual(backend.query,'mechanical sound')
            self.assertEqual(result['final_response_text'],'<response>fixture answer</response>')
            self.assertEqual(len(runner.histories),3)
            for history in runner.histories:
                self.assertEqual(history[0]['parts'][0]['video_url']['url'],str(src.resolve()))
                self.assertTrue(history[0]['parts'][0]['video_url']['has_audio'])
            videos=[part['video_url'] for m in runner.histories[2] for part in m['parts'] if 'video_url' in part]
            self.assertEqual(len(videos),2)
            self.assertTrue(Path(videos[1]['url']).is_file())
            self.assertTrue(videos[1]['has_audio'])
            self.assertAlmostEqual(videos[1]['source_start_s'],.2)
            self.assertAlmostEqual(videos[1]['source_end_s'],.8)
            self.assertTrue(result['turns'][1]['tool_output'].startswith('<observation>'))
            snapshot=result['turns'][2]['chat_completions']
            snapshot_videos=[part for m in snapshot if isinstance(m['content'],list) for part in m['content'] if part.get('type')=='video_url']
            self.assertEqual(len(snapshot_videos),2)
            self.assertTrue(snapshot_videos[1]['has_audio'])

    def test_incomplete_think_turn_is_repaired_before_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); src = tmp/'input.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=56x56:rate=12',
                '-t','1','-c:v','libx264','-pix_fmt','yuv420p',str(src)],check=True)
            class IncompleteRunner(ScriptedRunner):
                def infer(self, contents, **kwargs):
                    self.histories.append(json.loads(json.dumps(contents)))
                    return _wrap_gemini('<think>Need to decide.</think>' if len(self.histories)==1 else
                        '<response>Complete fixture.</response>','STOP',{},'fixture')
            runner=IncompleteRunner()
            result=process_native_av_case(question='Fixture?',source_path=str(src),runner=runner,
                output_dir=str(tmp/'out'),case_id='repair-fixture')
            self.assertEqual(len(result['turns']),2)
            self.assertTrue(result['turns'][0]['protocol_repair'])
            self.assertIn('Protocol error',runner.histories[1][-1]['parts'][0]['text'])
            self.assertEqual(result['final_response_text'],'<response>Complete fixture.</response>')
            self.assertEqual(runner.histories[1][0]['parts'][0]['video_url']['url'],str(src.resolve()))

    def test_visual_runner_cannot_silently_drop_audio(self):
        with self.assertRaisesRegex(ValueError,'audio-video runner'):
            process_native_av_case(question='test',source_path='unused.mp4',runner=SimpleNamespace(),
                output_dir='unused',case_id='case')


if __name__=='__main__':unittest.main()

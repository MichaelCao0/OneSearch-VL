"""Full-duration native teacher media preparation with decoded AV witnesses."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'onesearch_vl'));sys.path.insert(0,str(ROOT/'validation/research'))
from media_fixtures import make_offset_vfr,decoded_markers
from onesearch_infer.av_io import probe_av


class TeacherMediaTests(unittest.TestCase):
    def test_complete_vfr_clip_and_recorded_audio_survive_size_normalization(self):
        from research.av_train.teacher_media import normalize_media
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'source.mkv';make_offset_vfr(source)
            before=probe_av(source)
            out=Path(root)/'anonymous.mp4';receipt=normalize_media(source,out)
            visual,audio,_=decoded_markers(out)
            self.assertLess(abs(visual-1.6),.06);self.assertLess(abs(audio-1.6),.04)
            self.assertLess(abs(probe_av(out)['duration']-before['duration']),.15)
            self.assertTrue(receipt['has_audio']);self.assertFalse(receipt['truncated'])
            self.assertEqual(receipt['source_sha256'],__import__('research.av_train.native_sft',fromlist=['file_sha']).file_sha(source))
            self.assertEqual(receipt['normalization_recipe']['max_width'],640)

    def test_new_batch_preserves_task_identity_and_binds_actual_normalized_bytes(self):
        from research.av_train.teacher_media import prepare_teacher_batch
        from research.av_train.native_sft import file_sha
        from research.av_train.native_searchqa import prepare_cases
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'source.mkv';make_offset_vfr(source)
            batch=Path(root)/'parent';batch.mkdir()
            (batch/'scope.json').write_text(json.dumps({'split':'train'}))
            (batch/'tasks-public.jsonl').write_text(json.dumps({'task_id':'train-001','split':'train','question':'TEST','initial_sources':['clue']})+'\n')
            (batch/'clue-bindings.private.json').write_text(json.dumps([{'clue_id':'clue','source':{'asset_path':str(source),'asset_sha256':file_sha(source)}}]))
            (batch/'answer-candidates.private.jsonl').write_text('"PRIVATE_SENTINEL"\n')
            out=Path(root)/'normalized';report=prepare_teacher_batch(batch,out)
            self.assertEqual(report['prepared_clues'],1)
            self.assertEqual((out/'tasks-public.jsonl').read_bytes(),(batch/'tasks-public.jsonl').read_bytes())
            cases=prepare_cases(out);self.assertEqual(cases[0]['case_id'],'train-001')
            self.assertNotEqual(cases[0]['source_sha256'],file_sha(source))
            self.assertNotIn('PRIVATE_SENTINEL',json.dumps(cases))
            self.assertEqual((batch/'clue-bindings.private.json').read_text(),json.dumps([{'clue_id':'clue','source':{'asset_path':str(source),'asset_sha256':file_sha(source)}}]))
            with self.assertRaises(FileExistsError):prepare_teacher_batch(batch,out)


if __name__=='__main__':unittest.main()

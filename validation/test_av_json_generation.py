"""Generation-only syntax controls: never repair the decoded candidate."""
import importlib.metadata,json,unittest
from pathlib import Path
import torch
from research.av_agent.json_generation_control import generation_controls
REV='ae9e1690543ffd5c0221dc27f79834d0294cba00'
SNAPSHOT=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B'/'snapshots'/REV
@unittest.skipUnless(importlib.metadata.version('transformers')=='4.57.6' and (SNAPSHOT/'tokenizer_config.json').is_file(),
                     'requires pinned Omni tokenizer and generation runtime')
class JsonGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained(str(SNAPSHOT),local_files_only=True)
    def ids(self,text):
        return torch.tensor([self.tokenizer.encode(text,add_special_tokens=False)])
    def test_compound_period_token_is_excluded_at_generation_not_decode(self):
        controls,receipt=generation_controls(self.tokenizer,0)
        token=self.tokenizer.encode('}.',add_special_tokens=False)
        self.assertEqual(len(token),1)
        self.assertIn(token,controls['thinker_bad_words_ids'])
        self.assertEqual(receipt['policy'],'json_object_v1')
        self.assertFalse(receipt['raw_output_repair'])
    def test_stops_only_when_generated_text_is_a_complete_object(self):
        controls,_=generation_controls(self.tokenizer,0)
        stop=controls['thinker_stopping_criteria']
        for raw in ['','{','{"a":"brace } in string"','{"a":1}.','[]','"{}"','{"a":1,"a":2}','{"a":NaN}']:
            with self.subTest(raw=raw):self.assertFalse(stop(self.ids(raw),None).item())
        for raw in ['{"a":"brace } in string"}','{"a":{"b":[]}}',' {"a":1} ']:
            with self.subTest(raw=raw):self.assertTrue(stop(self.ids(raw),None).item())
    def test_pinned_omni_wrapper_forwards_thinker_controls(self):
        from types import SimpleNamespace
        from transformers import Qwen2_5OmniForConditionalGeneration
        controls, _ = generation_controls(self.tokenizer, 1)
        seen = {}
        def receive(**kwargs):
            seen.update(kwargs)
            return kwargs['input_ids']
        wrapper = SimpleNamespace(speaker_map={'Chelsie': {}}, has_talker=False,
                                  thinker=SimpleNamespace(generate=receive))
        ids = torch.tensor([[10]])
        result = Qwen2_5OmniForConditionalGeneration.generate(
            wrapper, input_ids=ids, return_audio=False, thinker_max_new_tokens=384,
            use_audio_in_video=True, do_sample=False, **controls)
        self.assertIs(result, ids)
        self.assertIs(seen['stopping_criteria'], controls['thinker_stopping_criteria'])
        self.assertEqual(seen['bad_words_ids'], controls['thinker_bad_words_ids'])
        self.assertEqual(seen['max_new_tokens'], 384)
        self.assertTrue(seen['use_audio_in_video'])
        self.assertNotIn('thinker_stopping_criteria', seen)

    def test_prompt_json_never_triggers_stop_and_batch_is_bounded(self):
        prefix=self.tokenizer.encode('{"prompt":true}',add_special_tokens=False)
        controls,_=generation_controls(self.tokenizer,len(prefix))
        stop=controls['thinker_stopping_criteria']
        self.assertFalse(stop(torch.tensor([prefix]),None).item())
        tail=self.tokenizer.encode('{"reply":true}',add_special_tokens=False)
        self.assertTrue(stop(torch.tensor([prefix+tail]),None).item())
        with self.assertRaises(ValueError):stop(torch.tensor([prefix+tail,prefix+tail]),None)
if __name__=='__main__':unittest.main()

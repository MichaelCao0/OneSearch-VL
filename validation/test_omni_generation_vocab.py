"""Generation must never select output rows absent from the tokenizer."""
from types import SimpleNamespace
import unittest
import torch
from validation import test_native_omni_runner as native_tests
from onesearch_vl.onesearch_infer.runners import InferenceConfig

class GenerationVocabTests(unittest.TestCase):
    def test_undefined_highest_logit_cannot_crash_native_action_decode(self):
        runner, processor, _ = native_tests.NativeOmniTests().runner()
        processor.tokenizer.get_vocab = lambda: {chr(i): i for i in range(127)}
        original_decode = processor.tokenizer.decode
        def decode(ids, **kwargs):
            if any(int(i) == 127 for i in ids):
                raise TypeError('sequence item 0: expected str instance, NoneType found')
            return original_decode(ids, **kwargs)
        processor.tokenizer.decode = decode
        runner._model.thinker.get_output_embeddings = lambda: SimpleNamespace(weight=torch.zeros(128, 1))
        runner._model.thinker.generation_config = SimpleNamespace(suppress_tokens=None)
        observed = []
        def generate(**kwargs):
            observed.append(kwargs)
            scores = torch.zeros((1, 128)); scores[0, 127] = 99; scores[0, ord('<')] = 5
            suppressed = kwargs.get('thinker_suppress_tokens', [])
            if suppressed:
                from transformers import SuppressTokensLogitsProcessor
                scores = SuppressTokensLogitsProcessor(suppressed)(kwargs['input_ids'], scores)
            first = int(scores.argmax(-1)[0])
            raw = '<response>ok</response>'
            tail = [first] + processor.tokenizer.encode(raw[1:])
            ids = torch.cat([kwargs['input_ids'], torch.tensor([tail])], dim=1)
            kwargs['thinker_stopping_criteria'](ids, None)
            return ids
        runner._model.generate = generate
        try:
            response = runner.infer([{'parts': [{'text': 'Answer with a response.'}]}], cfg=InferenceConfig(max_tokens=32, temperature=0))
        except TypeError as exc:
            self.fail('Generation selected an undefined output row: ' + str(exc))
        self.assertEqual(response['candidates'][0]['content']['parts'][0]['text'], '<response>ok</response>')
        self.assertEqual(observed[0]['thinker_suppress_tokens'], [127])

if __name__ == '__main__': unittest.main()

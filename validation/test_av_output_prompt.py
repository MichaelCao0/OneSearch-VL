"""Output prompt contract and actual pinned chat-template regression.

No model weights, generation, GPU computation, or semantic gold are used.
The official processor test reads already-verified local tokenizer assets only.
"""
import importlib.metadata
import json
from pathlib import Path
import unittest

from research.av_agent.qwen_omni_backend import QwenOmniBackend, parse_clip_observation

PINNED_REVISION = 'ae9e1690543ffd5c0221dc27f79834d0294cba00'
SNAPSHOT = (Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B'
            / 'snapshots' / PINNED_REVISION)
EXAMPLE_START = '\nFORMAT EXAMPLE ONLY (do not copy into observations):\n'
EXAMPLE_END = '\nEND FORMAT EXAMPLE\n'


class OutputPromptTests(unittest.TestCase):
    def setUp(self):
        # _messages is pure preparation; constructing via __new__ avoids weight
        # integrity checks, processor/model instantiation, and any CUDA access.
        self.backend = object.__new__(QwenOmniBackend)
        self.backend._video_fps = 2.0
        self.backend._max_pixels = 256 * 28 * 28
        self.bound = {
            'original_interval': {'start_s': 15.0, 'end_s': 17.0},
            'mapping': {'output_to_source': {'scale': 1.0, 'offset_s': 15.0}},
            'modalities': ['audio', 'video'],
            'clip_path': '/opaque/private/clip.mkv',
            'question': 'Describe directly heard and seen events.',
        }

    def _example(self, text):
        self.assertEqual(text.count(EXAMPLE_START), 1)
        self.assertEqual(text.count(EXAMPLE_END), 1)
        return text.split(EXAMPLE_START, 1)[1].split(EXAMPLE_END, 1)[0]

    def test_literal_example_is_complete_and_accepted_by_strict_parser(self):
        text = self.backend._messages(self.bound)[1]['content'][1]['text']
        raw = self._example(text)
        self.assertTrue(raw.startswith('{') and raw.endswith('}'))
        self.assertNotIn('\\"', raw, 'structural quotes must be literal, not escaped')
        example = json.loads(raw)
        self.assertIsInstance(example, dict)
        self.assertEqual(set(example), {'events', 'uncertainty', 'cannot_determine'})
        self.assertEqual(example['events'], [], 'format example must not inject an observed event')
        self.assertTrue(example['uncertainty'].strip())
        parsed, issue = parse_clip_observation(raw, self.bound)
        self.assertIsNone(issue)
        self.assertEqual(parsed['events'], [])

    def test_prompt_limits_output_and_preserves_causal_and_evidence_boundaries(self):
        text = self.backend._messages(self.bound)[1]['content'][1]['text']
        for rule in ('at most 2', 'nonempty', '0 <= start_s < end_s <= 2.000000',
                     'Do not escape the structural double quotes',
                     'Do not wrap the object in a JSON string',
                     'do not copy into observations',
                     'co-occurrence alone', 'Do not infer identity or unseen events',
                     'This question is a request, not evidence'):
            with self.subTest(rule=rule):
                self.assertIn(rule, text)
        self.assertIn('Use an empty events list when support is absent', text)
        for forbidden in ('15.0', '17.0', 'asset_sha256', 'source_id', 'gold'):
            self.assertNotIn(forbidden, text)

    def test_final_instruction_explicitly_forbids_trailing_sentence_period(self):
        text = self.backend._messages(self.bound)[1]['content'][1]['text']
        final_rule = text.split(self.bound['question'], 1)[1]
        self.assertIn('Your final output character must be }', final_rule)
        self.assertIn('Do not append a period', final_rule)
        self.assertIn('End immediately after the closing brace', final_rule)

    def test_trailing_period_is_rejected_without_repair(self):
        example = self._example(self.backend._messages(self.bound)[1]['content'][1]['text'])
        parsed, issue = parse_clip_observation(example + '.', self.bound)
        self.assertEqual(issue, 'invalid_json')
        self.assertEqual(parsed['events'], [])

    def test_boundary_uses_current_clip_duration_instead_of_fixed_two_seconds(self):
        self.bound['original_interval']['end_s'] = 15.375
        text = self.backend._messages(self.bound)[1]['content'][1]['text']
        example = json.loads(self._example(text))
        self.assertEqual(example['events'], [])
        self.assertIn('0 <= start_s < end_s <= 0.375000', text)
        _, issue = parse_clip_observation(json.dumps(example), self.bound)
        self.assertIsNone(issue)

    @unittest.skipUnless(importlib.metadata.version('transformers') == '4.57.6'
                         and (SNAPSHOT / 'tokenizer_config.json').is_file(),
                         'requires pinned isolated Omni runtime and cached processor assets')
    def test_actual_pinned_chat_template_preserves_literal_json_example(self):
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(
            str(SNAPSHOT), local_files_only=True, trust_remote_code=False)
        messages = self.backend._messages(self.bound)
        rendered = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        raw = self._example(rendered)
        self.assertEqual(raw, self._example(messages[1]['content'][1]['text']))
        self.assertNotIn('\\"', raw)
        self.assertIsInstance(json.loads(raw), dict)
        _, issue = parse_clip_observation(raw, self.bound)
        self.assertIsNone(issue)
        self.assertIn('co-occurrence alone', rendered)
        self.assertIn('<|im_start|>assistant', rendered)


if __name__ == '__main__':
    unittest.main()

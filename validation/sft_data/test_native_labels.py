"""Run only in runtime/sft-data; imports real native trainer and local tokenizer."""
import copy
import unittest
from research.av_train.label_audit import audit_rows,fixture_row

TOKENIZER='repro/assets/model-8b'

class NativeLabelTests(unittest.TestCase):
    def test_every_observation_masked_and_all_three_assistant_targets_kept(self):
        report=audit_rows([fixture_row()],TOKENIZER,fixture=True)
        self.assertTrue(report['fixture_only']);self.assertFalse(report['training_ready'])
        row=report['rows'][0]
        self.assertEqual(row['assistant_turns'],3)
        self.assertGreater(row['supervised_tokens'],0)
        self.assertLess(row['supervised_tokens'],row['total_tokens'])
        for pair in row['pairs']:
            self.assertEqual(pair['source_ignore_count'],pair['source_token_count'])
            self.assertGreater(pair['assistant_token_count'],0)
    def test_cutoff_cannot_silently_drop_final(self):
        with self.assertRaisesRegex(ValueError,'exceeds cutoff'):audit_rows([fixture_row()],TOKENIZER,cutoff_len=32,fixture=True)
    def test_native_converter_cannot_accept_wrong_role_pairing(self):
        row=fixture_row();row['conversations'][2]['from']='gpt'
        with self.assertRaisesRegex(ValueError,'role alignment'):audit_rows([row],TOKENIZER,fixture=True)
    def test_training_target_envelope_matches_deployment(self):
        row=fixture_row();row['conversations'][1]['value']='{"tool":"search_sources"}'
        with self.assertRaisesRegex(ValueError,'JSON target'):audit_rows([row],TOKENIZER,fixture=True)

if __name__=='__main__':unittest.main()

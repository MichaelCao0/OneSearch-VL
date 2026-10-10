import importlib.util
import unittest


def api(test):
    name='research.av_train.label_audit'
    test.assertIsNotNone(importlib.util.find_spec(name),'Missing actual trainer label audit')
    return __import__(name,fromlist=['*'])

class LabelAuditContractTests(unittest.TestCase):
    def test_prompt_training_cannot_pass_av_trajectory_audit(self):
        m=api(self)
        with self.assertRaisesRegex(ValueError,'train_on_prompt'):m.validate_settings(32768,True,False)
    def test_masking_history_cannot_drop_early_actions(self):
        m=api(self)
        with self.assertRaisesRegex(ValueError,'mask_history'):m.validate_settings(32768,False,True)
    def test_empty_rows_are_not_label_verified_training_data(self):
        m=api(self)
        with self.assertRaisesRegex(ValueError,'empty'):m.audit_rows([], '/not-used')
    def test_fixture_export_alternates_actions_and_realistic_observation_context(self):
        m=api(self);row=m.fixture_row()
        self.assertEqual([r['from'] for r in row['conversations']],['human','gpt','observation','gpt','observation','gpt'])
        self.assertIn('fixture',row['system'].lower())
    def test_fixture_has_no_training_readiness_claim(self):
        m=api(self);row=m.fixture_row()
        self.assertNotIn('semantic_verified',row)

if __name__=='__main__':unittest.main()

import unittest
from unittest.mock import patch
from research.av_train import searchqa
class ClueEntryTests(unittest.TestCase):
 def flags(self):return ['--batch-dir','none','--checkpoint','none','--observer-smoke-dir','none','--out-dir','none','--clue-bindings','private.json','--protocol','canonical']
 def test_existing_controller_cli_accepts_anonymous_query_binding_profile(self):
  try:a=searchqa.make_parser().parse_args(self.flags())
  except SystemExit:self.fail('Missing anonymous clue CLI profile')
  self.assertEqual(a.clue_bindings,'private.json');self.assertEqual(a.protocol,'canonical')
 def test_title_assisted_guidance_rejected_for_anonymous_clue_profile(self):
  with patch('sys.argv',['searchqa']+self.flags()+['--guided-curriculum']):
   with self.assertRaisesRegex(ValueError,'guided'):searchqa.main()
if __name__=='__main__':unittest.main()

class MediaEntryTests(unittest.TestCase):
 def test_media_preparation_profile_does_not_require_observer_replay_path(self):
  try:a=searchqa.make_parser().parse_args(['--batch-dir','b','--checkpoint','c','--out-dir','o','--clue-bindings','q','--observer-mode','media'])
  except SystemExit:self.fail('Missing honest media-only anonymous clip profile')
  self.assertEqual(a.observer_mode,'media');self.assertIsNone(a.observer_smoke_dir)

class NativeEntryTests(unittest.TestCase):
 def test_dynamic_observer_profile_accepts_pinned_native_configuration(self):
  flags=['--batch-dir','b','--checkpoint','c','--out-dir','o','--clue-bindings','q','--observer-mode','native','--observer-model-manifest','m','--observer-code-revision','r','--observer-runtime-manifest','p','--observer-runtime-lock','l']
  try:a=searchqa.make_parser().parse_args(flags)
  except SystemExit:self.fail('Missing dynamic anonymous AV observer profile')
  self.assertEqual(a.observer_mode,'native');self.assertEqual(a.observer_model_manifest,'m');self.assertEqual(a.observer_device,'cuda:0')

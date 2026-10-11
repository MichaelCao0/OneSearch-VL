import unittest
from research.av_train import searchqa
class ControllerRouteTests(unittest.TestCase):
    def test_published_controller_remains_explicit_default(self):
        self.assertTrue(hasattr(searchqa,'make_parser'),'Missing explicit local teacher selection')
        a=searchqa.make_parser().parse_args(['--batch-dir','batch','--checkpoint','model','--observer-smoke-dir','obs','--out-dir','out'])
        self.assertEqual(a.controller_kind,'onesearch')
    def test_omni_teacher_requires_pinned_artifact_manifest(self):
        self.assertTrue(hasattr(searchqa,'make_parser'))
        a=searchqa.make_parser().parse_args(['--batch-dir','batch','--checkpoint','model','--observer-smoke-dir','obs','--out-dir','out',
            '--controller-kind','omni','--controller-model-manifest','pinned-manifest'])
        self.assertEqual(a.controller_kind,'omni');self.assertEqual(a.controller_model_manifest,'pinned-manifest')
        self.assertTrue(hasattr(searchqa,'LocalOmniController'))
if __name__=='__main__':unittest.main()

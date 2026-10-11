import unittest
from research.av_train.trajectory import source_ids
class SourceCollectionTests(unittest.TestCase):
    def test_tool_parameter_schema_is_not_executed_source_reference(self):
        trace={'tools':[{'name':'inspect_av','arguments':{'source_id':'string'}}],
               'steps':[{'action':{'tool':'inspect_av','arguments':{'source_id':'src_actual'}}}]}
        self.assertEqual(source_ids(trace),{'src_actual'})
    def test_actual_foreign_reference_is_still_collected(self):
        self.assertEqual(source_ids({'final':{'evidence':[{'source_id':'src_foreign'}]}}),{'src_foreign'})
if __name__=='__main__':unittest.main()

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from test_research_pipeline import task,reference,sources


def api(test):
    name='research.av_bench.annotations'
    test.assertIsNotNone(importlib.util.find_spec(name),'Missing semantic annotation import')
    return __import__(name,fromlist=['*'])


def review(reviewer):
    r=reference()
    return {'schema_version':'av-researchbench.review.v1','reviewer_id':reviewer,'submitted_at':'2026-10-10T10:00:00Z',
            'status':'submitted','task_id':r['task_id'],'family':r['family'],'facts':r['facts'],'evidence_packages':r['evidence_packages'],
            'source_family_checked':True,'modality_path':'joint_av','notes':'Reviewed original media and source spans.'}

class AnnotationImportTests(unittest.TestCase):
    def test_agreement_creates_reference_with_real_reviewer_names(self):
        m=api(self);result=m.merge_reviews(task(),sources(),review('A-person'),review('B-person'))
        self.assertEqual(result['status'],'agreed')
        self.assertEqual(result['reference']['reviewer_ids'],['A-person','B-person'])
        self.assertEqual(result['reference']['status'],'adjudicated')
    def test_same_reviewer_does_not_count_as_two_independent_reviews(self):
        m=api(self)
        with self.assertRaisesRegex(ValueError,'independent'):m.merge_reviews(task(),sources(),review('A'),review('A'))
    def test_disagreeing_answer_needs_adjudication(self):
        m=api(self);a,b=review('A'),review('B');b['facts'][0]['accepted_values']=['different']
        result=m.merge_reviews(task(),sources(),a,b)
        self.assertEqual(result['status'],'needs_adjudication');self.assertIsNone(result['reference'])
    def test_disagreeing_interval_cannot_be_silently_averaged(self):
        m=api(self);a,b=review('A'),review('B');b['evidence_packages'][1][0]['end_s']=4.2
        self.assertEqual(m.merge_reviews(task(),sources(),a,b)['status'],'needs_adjudication')
    def test_blank_packet_not_a_submitted_review(self):
        m=api(self);packet=m.blank_review(task(),'source_discovery')
        self.assertIsNone(packet['reviewer_id']);self.assertEqual(packet['facts'],[])
        with self.assertRaises(ValueError):m.merge_reviews(task(),sources(),packet,review('B'))
    def test_reference_with_out_of_media_bounds_rejected(self):
        m=api(self);a,b=review('A'),review('B')
        for r in [a,b]:r['evidence_packages'][1][0]['end_s']=100
        with self.assertRaises(ValueError):m.merge_reviews(task(),sources(),a,b)
    def test_export_packets_never_fills_human_signatures(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'case-authoring.private.json').write_text(json.dumps([{'task':task(),'family':'source_discovery'}]))
            result=m.prepare_review_packets(p,p/'packets')
            self.assertEqual(result['case_count'],1)
            for reviewer in ['A','B']:
                r=json.loads((p/'packets'/reviewer/'t1.json').read_text())
                self.assertIsNone(r['reviewer_id']);self.assertIsNone(r['submitted_at']);self.assertEqual(r['status'],'pending')

if __name__=='__main__':unittest.main()

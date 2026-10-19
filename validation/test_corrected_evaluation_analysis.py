"""Regression checks for common-question selection and paired inference."""
import unittest
from research.av_train.corrected_evaluation_analysis import common_reviews, paired, summarize

def case(t, family='f'):
    return {'task_id':t,'family_id':family,'split':'heldout_silver','task':{'question':t}}

def review(t, native, muted, status='completed'):
    return {'task_id':t,'status':status,'ratings':[{'variant_modality':'m__'+mode,'answer_success':score,'supported_success':score}
        for mode, score in [('native_av',native),('muted_video',muted)]]}

class CorrectedAnalysisTests(unittest.TestCase):
    def test_unknown_review_does_not_become_a_failure(self):
        valid,unknown=common_reviews([case('a'),case('b')],[review('a',True,False),review('b',False,False,'failed')],['m'])
        result=summarize(valid,[case('a'),case('b')],['m'],draws=100)
        row=result['all']['table']['m__native_av']
        self.assertEqual(unknown,['b']);self.assertEqual(row['n'],1);self.assertEqual(row['answer_success']['rate'],1)
        self.assertEqual(row['answer_success']['all_planned_bounds'],[.5,1])

    def test_duplicate_question_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate planned'):common_reviews([case('a'),case('a')],[],['m'])

    def test_duplicate_variant_rejected(self):
        r=review('a',True,False);r['ratings'][1]['variant_modality']='m__native_av'
        with self.assertRaisesRegex(ValueError,'variant'):common_reviews([case('a')],[r],['m'])

    def test_impossible_supported_score_rejected(self):
        r=review('a',False,False);r['ratings'][0]['supported_success']=True
        with self.assertRaisesRegex(ValueError,'Supported'):common_reviews([case('a')],[r],['m'])

    def test_family_bootstrap_preserves_question_weighting(self):
        r=paired([1,1,0],['a','a','b'],draws=1000)
        self.assertAlmostEqual(r['delta'],2/3);self.assertEqual(r['family_groups'],2)
        self.assertEqual(r['bootstrap95'],[0,1])

    def test_paired_binomial_tail_and_direction(self):
        r=paired([1]*5,['a','b','c','d','e'],draws=100)
        self.assertEqual(r['wins'],5);self.assertEqual(r['losses'],0);self.assertEqual(r['paired_binomial_p_two_sided'],.0625)
        self.assertEqual(r['bootstrap95'],[1,1])

    def test_question_identity_controls_flip_direction(self):
        members=[case('a'),case('b')];valid,_=common_reviews(members,[review('b',False,True),review('a',True,False)],['m'])
        c=summarize(valid,members,['m'],draws=100)['all']['contrasts']['m__native_av minus m__muted_video']['answer_success']
        self.assertEqual(c['left_win_task_ids'],['a']);self.assertEqual(c['right_win_task_ids'],['b']);self.assertEqual(c['delta'],0)

    def test_no_completed_questions_has_no_accuracy(self):
        r=summarize({},[case('a')],['m'],draws=10)
        self.assertIsNone(r['all']['table']['m__native_av']['answer_success']['rate'])
        self.assertIsNone(r['all']['contrasts']['m__native_av minus m__muted_video']['answer_success']['delta'])

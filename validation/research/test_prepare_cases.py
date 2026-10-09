"""Three-case packages preserve exploration and review/gold boundaries."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

class PrepareCaseContractTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('research.av_data.prepare_cases'),'case package API missing')
        return importlib.import_module('research.av_data.prepare_cases')

    def plans(self):
        domains=['technical_mechanisms','field_performance','environment_animals']
        ids=['src_'+format(n,'024x') for n in range(1,13)]
        result=[]
        for n,domain in enumerate(domains):
            sid=ids[n*4:n*4+4]
            result.append({'task_id':f'case-{n}','candidate_domain':domain,'question':'可裁决研究问题',
                'target_questions':[{'claim_id':f'c{i}','question':'搜索额外来源并保留边界'} for i in range(3)],
                'initial_sources':[sid[0]],'source_roles':[{'source_id':s,'role':'primary_recording' if i==0 else 'comparison_recording'} for i,s in enumerate(sid)]})
        return result

    def test_three_cases_and_four_to_six_distinct_assets_are_required(self):
        plans=self.plans();self.assertEqual(len(self.api().validate_case_plan(plans, {r['source_id'] for p in plans for r in p['source_roles']})),3)
        with self.assertRaises(ValueError):self.api().validate_case_plan(plans[:2], {r['source_id'] for p in plans for r in p['source_roles']})
        changed=copy.deepcopy(plans);changed[0]['source_roles']=changed[0]['source_roles'][:3]
        with self.assertRaises(ValueError):self.api().validate_case_plan(changed,{r['source_id'] for p in plans for r in p['source_roles']})

    def test_duplicate_foreign_and_answer_fields_are_rejected(self):
        plans=self.plans();allowed={r['source_id'] for p in plans for r in p['source_roles']}
        for change in ('duplicate','foreign','gold'):
            modified=copy.deepcopy(plans)
            if change=='duplicate':modified[0]['source_roles'][1]['source_id']=modified[0]['source_roles'][0]['source_id']
            elif change=='foreign':modified[0]['source_roles'][0]['source_id']='src_'+'f'*24
            else:modified[0]['answer']='fabricated human gold'
            with self.assertRaises(ValueError):self.api().validate_case_plan(modified,allowed)

    def test_publisher_pages_require_related_media_and_never_become_independent(self):
        plans=self.plans();allowed={r['source_id'] for p in plans for r in p['source_roles']}
        modified=copy.deepcopy(plans);modified[0]['source_roles'][1]={'source_id':modified[0]['source_roles'][1]['source_id'],'role':'publisher_assertion'}
        with self.assertRaises(ValueError):self.api().validate_case_plan(modified,allowed)
        modified[0]['source_roles'][1]['related_source_id']=modified[0]['source_roles'][0]['source_id']
        self.api().validate_case_plan(modified,allowed)
        modified[0]['source_roles'][1]['independent_observation']=True
        with self.assertRaises(ValueError):self.api().validate_case_plan(modified,allowed)

class PrepareCaseArtifactTests(unittest.TestCase):
    api = PrepareCaseContractTests.api
    plans = PrepareCaseContractTests.plans
    def fixture(self, root):
        from research.av_data.ingest import ingest_media
        from research.av_data.web_sources import ingest_web_snapshot
        from validation.research.media_fixtures import make_offset_vfr
        raw=root/'fixture.mkv';make_offset_vfr(raw)
        media=ingest_media(raw,'https://example.org/media',root/'assets',{'title':'Synthetic clock fixture'})
        sources=[media]
        for n in range(3):
            html=root/f'page{n}.html';html.write_text(f'<p>Publisher assertion {n}.</p>')
            sources.append(ingest_web_snapshot(html,f'https://example.org/page{n}',root/'assets',{'title':f'Page {n}'}))
        plans=self.plans()
        for plan in plans:
            plan['initial_sources']=[media['source_id']]
            plan['source_roles']=[{'source_id':media['source_id'],'role':'primary_recording'}]+[
                {'source_id':r['source_id'],'role':'publisher_assertion','related_source_id':media['source_id'],'quote':r['body_text']} for r in sources[1:]]
        sources_path=root/'sources.jsonl';sources_path.write_text(''.join(json.dumps(r)+'\n' for r in sources))
        plan_path=root/'plans.json';plan_path.write_text(json.dumps(plans))
        acq={'sources':[{'source_id':'acq_fixture','status':'candidate_unreviewed_av','sha256':media['asset_sha256']},
                        {'source_id':'excluded','status':'rejected_silent_or_near_silent','sha256':'0'*64}]}
        acq_path=root/'acquisition.json';acq_path.write_text(json.dumps(acq))
        return sources,plans,sources_path,plan_path,acq_path

    def test_package_global_corpus_quote_replay_and_blank_independent_review(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,plans,sp,pp,ap=self.fixture(root);out=root/'package'
            summary=api.prepare_cases(sp,pp,ap,out,corpus_id='case-fixture')
            self.assertEqual(summary['case_count'],3);self.assertEqual(summary['source_count'],4)
            self.assertEqual(summary['human_gold_count'],0);self.assertEqual(summary['audio_necessity'],'pending')
            scope=json.loads((out/'scope.json').read_text());self.assertEqual(len(scope['source_ids']),4)
            self.assertEqual(json.loads((out/'acquisition-exclusions.json').read_text())[0]['source_id'],'excluded')
            for form_path in (out/'review-packets'/'A').glob('*.json'):
                form=json.loads(form_path.read_text());self.assertIsNone(form['reviewer_identity'])
                self.assertEqual(len(form['source_catalog']),4)
                self.assertTrue(all(c['expected_decision'] is None and c['valid_evidence_sets']==[] for c in form['target_annotations']))
            quotes=json.loads((out/'publisher-quotes.json').read_text())
            from research.av_data.web_sources import quote_web_source
            registry={r['source_id']:r for r in sources}
            for q in quotes:self.assertEqual(quote_web_source(registry[q['source_id']],q['char_start'],q['char_end'])['quote'],q['quote'])
            self.assertFalse(any('gold' in p.name for p in out.rglob('*')))
            with self.assertRaises(FileExistsError):api.prepare_cases(sp,pp,ap,out,corpus_id='case-fixture')

    def test_excluded_media_and_fabricated_quote_refuse_package(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,plans,sp,pp,ap=self.fixture(root)
            acq=json.loads(ap.read_text());acq['sources'][0]['status']='candidate_av_metadata_concern';ap.write_text(json.dumps(acq))
            with self.assertRaises(ValueError):api.prepare_cases(sp,pp,ap,root/'bad',corpus_id='case-fixture')
            self.assertFalse((root/'bad').exists())
            acq['sources'][0]['status']='candidate_unreviewed_av';ap.write_text(json.dumps(acq))
            plans[0]['source_roles'][1]['quote']='Fabricated assertion';pp.write_text(json.dumps(plans))
            with self.assertRaises(ValueError):api.prepare_cases(sp,pp,ap,root/'badquote',corpus_id='case-fixture')
            self.assertFalse((root/'badquote').exists())

if __name__=='__main__':unittest.main()

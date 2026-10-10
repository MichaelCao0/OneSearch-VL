"""General source-group authoring; synthetic fixtures are never semantic labels."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


def api(test):
    name='research.av_train.prepare_groups'
    test.assertIsNotNone(importlib.util.find_spec(name),'Missing scalable train source-group builder')
    return __import__(name,fromlist=['*'])


def specs():
    return {'schema_version':'av-train-group-specs.v1','corpus_id':'train-groups-fixture',
      'budget_recipe':{'max_tool_calls':12,'max_requested_av_seconds':30},'tasks':[
        {'task_id':'train-compare-001','family':'cross_source_comparison','domain':'mechanical',
         'source_group':'group-mechanical','candidate_page_ids':[101,102],'initial_page_ids':[101],
         'question':'Compare the observed movement in these recordings.',
         'target_questions':[{'fact_id':'f1','question':'What movement is observed in the first recording?'},
                             {'fact_id':'f2','question':'What movement is observed in the other recording?'}],
         'suitability_risk':'Synthetic fixture; no semantic model or human review.'}]}

class GroupPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.seeds=[]
        for i,color in enumerate(('blue','red'),101):
            p=cls.root/f'{i}.mkv'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',f'color=c={color}:s=32x32:r=5:d=2',
                '-f','lavfi','-i',f'sine=frequency={i*3}:sample_rate=8000:duration=2',
                '-c:v','ffv1','-c:a','pcm_s16le',str(p)],check=True)
            cls.seeds.append({'status':'candidate_unreviewed_av','page_id':i,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'asset_path':str(p),'origin_url':f'https://example.org/source-{i}',
                'candidate_domain':'synthetic','public_metadata':{'title':f'Synthetic source {i}','license':'synthetic_fixture'}})
        cls.manifest=cls.root/'manifest.json';cls.manifest.write_text(json.dumps({'sources':cls.seeds}))
        cls.reserved=cls.root/'reserved.jsonl';cls.reserved.write_text('')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_two_sources_can_build_explicit_non_four_case_pool(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'specs.json';p.write_text(json.dumps(specs()))
            summary=m.prepare_group_pool([self.manifest],self.reserved,p,root/'pool')
            self.assertEqual(summary['case_count'],1);self.assertEqual(summary['av_source_count'],2)
            self.assertEqual(summary['source_group_count'],1)
            self.assertFalse(summary['teacher_rollout_allowed']);self.assertFalse(summary['train_split_frozen'])
            self.assertFalse((root/'pool/tasks-public.jsonl').exists())
            self.assertFalse((root/'pool/scope.json').exists())
            row=json.loads((root/'pool/case-authoring.private.json').read_text())[0]
            self.assertEqual(row['task']['target_questions'],specs()['tasks'][0]['target_questions'])
            self.assertEqual(len(row['candidate_source_ids']),2)
            self.assertEqual(row['reference']['facts'],[]);self.assertFalse(row['semantic_gold'])
    def test_reserved_source_removal_prevents_incomplete_task_creation(self):
        m=api(self)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'specs.json';p.write_text(json.dumps(specs()))
            reserved=root/'reserved.jsonl';reserved.write_text(json.dumps({'asset_sha256':self.seeds[0]['sha256']})+'\n')
            with self.assertRaisesRegex(ValueError,'unavailable page'):
                m.prepare_group_pool([self.manifest],reserved,p,root/'pool')
            self.assertFalse((root/'pool').exists())
    def test_comparison_cannot_silently_degrade_to_single_source(self):
        m=api(self);s=specs();s['tasks'][0]['candidate_page_ids']=[101]
        with self.assertRaisesRegex(ValueError,'two'):
            m.validate_group_specs(s,self.seeds)
    def test_duplicate_task_id_rejected_before_writing(self):
        m=api(self);s=specs();s['tasks'].append(copy.deepcopy(s['tasks'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate task'):
            m.validate_group_specs(s,self.seeds)
    def test_initial_media_must_belong_to_declared_source_group(self):
        m=api(self);s=specs();s['tasks'][0]['initial_page_ids']=[103]
        with self.assertRaisesRegex(ValueError,'initial'):
            m.validate_group_specs(s,self.seeds)
    def test_unknown_or_duplicate_page_is_not_invented_as_source(self):
        m=api(self)
        for pages in ([101,103],[101,101]):
            s=specs();s['tasks'][0]['candidate_page_ids']=pages
            with self.assertRaises(ValueError):m.validate_group_specs(s,self.seeds)
    def test_task_specs_cannot_smuggle_gold_into_public_question(self):
        m=api(self);s=specs();s['tasks'][0]['gold_answer']='invented'
        with self.assertRaisesRegex(ValueError,'schema'):
            m.validate_group_specs(s,self.seeds)

if __name__=='__main__':unittest.main()

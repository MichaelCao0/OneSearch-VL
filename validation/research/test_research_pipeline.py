import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

A='src_'+ 'a'*24
B='src_'+ 'b'*24
C='src_'+ 'c'*24

def api(test, module):
    test.assertIsNotNone(importlib.util.find_spec(module), 'Missing pipeline feature: '+module)
    return __import__(module,fromlist=['*'])

def task():
    return {'schema_version':'av-researchbench.task.v1','task_id':'t1','question':'Which event?',
            'initial_sources':[],'target_questions':[{'fact_id':'f1','question':'What happens?'}],
            'corpus_id':'c1','split':'dev','budget_recipe':{'max_tool_calls':8,'max_requested_av_seconds':40}}

def sources():
    return {s:{'source_id':s,'media_type':'audio_video','duration_s':20} for s in [A,B,C]}

def reference():
    item=lambda s:{'fact_id':'f1','source_id':s,'start_s':2.,'end_s':4.,'event_id':'e1'}
    return {'schema_version':'av-researchbench.reference.v1','task_id':'t1','status':'adjudicated',
            'reviewer_ids':['reviewer-A','reviewer-B'],'facts':[{'fact_id':'f1','accepted_values':['impact']}],
            'evidence_packages':[[item(A),item(B)],[item(C)]], 'family':'source_discovery'}

def prediction():
    return {'status':'ok','facts':[{'fact_id':'f1','value':'impact'}], 'ranked_sources':[C],
            'ranked_windows':[{'source_id':C,'start_s':2,'end_s':4,'event_id':'e1'}],
            'evidence':[{'fact_id':'f1','source_id':C,'start_s':2,'end_s':4,'event_id':'e1'}],
            'cost':{'tool_calls':2,'requested_av_seconds':2}}

class BenchmarkTests(unittest.TestCase):
    def score(self,r=None,p=None):
        return api(self,'research.av_bench.evaluate').score_prediction(task(),r or reference(),p or prediction(),sources(),k=2)
    def test_alternative_complete_package_succeeds_without_union(self):
        result=self.score()
        self.assertEqual(result['complete_package_source_recall_at_k'],1)
        self.assertEqual(result['ebs'],1)
    def test_partial_package_cannot_be_ebs(self):
        p=prediction();p['ranked_sources']=[A];p['evidence'][0]['source_id']=A
        p['ranked_windows'][0]['source_id']=A
        self.assertEqual(self.score(p=p)['ebs'],0)
        self.assertEqual(self.score(p=p)['best_valid_package_coverage_at_k'],.5)
    def test_wrong_event_id_with_perfect_iou_fails(self):
        p=prediction();p['evidence'][0]['event_id']='other';p['ranked_windows'][0]['event_id']='other'
        self.assertEqual(self.score(p=p)['ebs'],0)
        self.assertEqual(self.score(p=p)['moment_package_coverage_at_k'],0)
    def test_ranking_deduplicates_before_cutoff(self):
        p=prediction();p['ranked_sources']=[A,A,B];p['evidence']=reference()['evidence_packages'][0]
        self.assertEqual(self.score(p=p)['complete_package_source_recall_at_k'],1)
    def test_correct_source_and_false_fact_fails(self):
        p=prediction();p['facts'][0]['value']='invented'
        self.assertEqual(self.score(p=p)['ebs'],0)
    def test_unreviewed_reference_is_not_scored(self):
        r=reference();r['status']='pending'
        with self.assertRaisesRegex(ValueError,'adjudicated'):self.score(r=r)
    def test_public_task_rejects_hidden_answer(self):
        m=api(self,'research.av_bench.contracts');t=task();t['answer']='impact'
        with self.assertRaises(ValueError):m.validate_task(t,{'corpus_id':'c1','split':'dev','source_ids':[A,B,C]})
    def test_timeout_kept_in_aggregate_denominator(self):
        m=api(self,'research.av_bench.evaluate')
        good=self.score();bad=self.score(p={'status':'timeout'});bad['task_id']='t2'
        self.assertEqual(m.aggregate_scores([good,bad])['ebs'],.5)
    def test_out_of_range_reference_is_rejected(self):
        r=reference();r['evidence_packages'][1][0]['end_s']=21
        with self.assertRaises(ValueError):self.score(r=r)
    def test_budget_excess_is_failed_attempt(self):
        p=prediction();p['cost']['requested_av_seconds']=41
        self.assertEqual(self.score(p=p)['ebs'],0)
    def test_malformed_prediction_is_failed_not_deleted(self):
        p=prediction();p['evidence'][0]['end_s']=float('nan')
        self.assertEqual(self.score(p=p)['ebs'],0)

class TrainingTests(unittest.TestCase):
    def make_trace(self):
        m=api(self,'research.av_train.trajectory')
        observation={'status':'ok','source_id':A,'original_interval':{'start_s':2,'end_s':4},
                     'semantic_status':'observed','facts':['impact'],'actual_cost':{'wall_seconds':1}}
        action={'tool':'inspect_av','arguments':{'source_id':A,'start_s':2,'end_s':4}}
        t=task();t['split']='train'
        trace={'schema_version':'av-research-trajectory.v1','task':t,'teacher':{'kind':'model','model':'test-teacher','revision':'r1'},
               'system':'Search using executed tools.','tools':[],'steps':[m.make_step(action,observation)],
               'final':dict(prediction(),ranked_sources=[A],ranked_windows=[dict(prediction()['ranked_windows'][0],source_id=A)],evidence=[dict(prediction()['evidence'][0],source_id=A)]),'execution_kind':'live_tools','referenced_source_ids':[A]}
        verdict={'semantic_verified':True,'evaluation':{'ebs':1},'reviewer_id':'semantic-reviewer'}
        return trace,verdict,observation
    def test_exporter_preserves_observation_as_context(self):
        m=api(self,'research.av_train.export');trace,verdict,obs=self.make_trace()
        row=m.export_trajectory(trace,{A:'train'},lambda a:obs,verdict)
        self.assertEqual([x['from'] for x in row['conversations']],['human','gpt','observation','gpt'])
        self.assertEqual(json.loads(row['conversations'][2]['value'])['facts'],['impact'])
    def test_dev_source_rejected_from_train(self):
        m=api(self,'research.av_train.export');trace,v,obs=self.make_trace()
        with self.assertRaisesRegex(ValueError,'split'):m.export_trajectory(trace,{A:'dev'},lambda a:obs,v)
    def test_fixture_never_retained(self):
        m=api(self,'research.av_train.export');trace,v,obs=self.make_trace();trace['teacher']['kind']='fixture'
        with self.assertRaisesRegex(ValueError,'teacher'):m.export_trajectory(trace,{A:'train'},lambda a:obs,v)
    def test_fake_tool_observation_replay_rejected(self):
        m=api(self,'research.av_train.export');trace,v,obs=self.make_trace()
        replay=dict(obs,facts=['different'])
        with self.assertRaisesRegex(ValueError,'replay'):m.export_trajectory(trace,{A:'train'},lambda a:replay,v)
    def test_tampered_observation_hash_rejected(self):
        m=api(self,'research.av_train.export');trace,v,obs=self.make_trace();trace['steps'][0]['observation']['facts']=['false']
        with self.assertRaisesRegex(ValueError,'hash'):m.export_trajectory(trace,{A:'train'},lambda a:obs,v)
    def test_wall_time_can_differ_on_replay(self):
        m=api(self,'research.av_train.export');trace,v,obs=self.make_trace();obs=copy.deepcopy(obs);obs['actual_cost']['wall_seconds']=2
        self.assertTrue(m.export_trajectory(trace,{A:'train'},lambda a:obs,v))
    def test_semantic_review_required(self):
        m=api(self,'research.av_train.export');trace,v,obs=self.make_trace();v['semantic_verified']=False
        with self.assertRaisesRegex(ValueError,'semantic'):m.export_trajectory(trace,{A:'train'},lambda a:obs,v)
    def test_observation_paths_not_exported(self):
        m=api(self,'research.av_train.export');trace,v,obs=self.make_trace();obs['asset_path']='/private/path'
        trace['steps'][0]=api(self,'research.av_train.trajectory').make_step(trace['steps'][0]['action'],obs)
        row=m.export_trajectory(trace,{A:'train'},lambda a:obs,v)
        self.assertNotIn('/private/path',json.dumps(row))


class BatchTests(unittest.TestCase):
    def registry(self):
        titles=['Steam engine 1903','Steam Engine','Steam engine operating','Printing presses',
                'Tchêlistas','Playing a Stroh violin','Gong and Drum','Cardio Drumming','Shila Waterfall','A little corella feeding']
        return {('src_'+format(i+1,'024x')):{'source_id':'src_'+format(i+1,'024x'),'media_type':'audio_video',
                   'duration_s':20,'public_metadata':{'title':title},'origin_url':'https://example.org/'+str(i)} for i,title in enumerate(titles)}
    def test_sixteen_tasks_four_per_family(self):
        m=api(self,'research.av_data.build_research_batch');rows=m.build_tasks(self.registry(),'dev-corpus')
        self.assertEqual(len(rows),16)
        for family in api(self,'research.av_bench.contracts').FAMILIES:self.assertEqual(sum(r['family']==family for r in rows),4)
    def test_actor_tasks_have_no_candidate_shortlist(self):
        m=api(self,'research.av_data.build_research_batch')
        for row in m.build_tasks(self.registry(),'dev-corpus'):
            self.assertNotIn('candidate_source_ids',row['task'])
            self.assertEqual(row['reference']['status'],'pending')
            self.assertEqual(row['reference']['evidence_packages'],[])
            self.assertEqual(row['task']['split'],'dev')
    def test_missing_natural_source_group_is_explicit(self):
        m=api(self,'research.av_data.build_research_batch');reg=self.registry();del reg['src_'+format(10,'024x')]
        with self.assertRaisesRegex(ValueError,'Missing source'):m.build_tasks(reg,'dev-corpus')
    def test_review_html_escapes_publisher_text(self):
        m=api(self,'research.av_data.visualize_batch')
        row=api(self,'research.av_data.build_research_batch').build_tasks(self.registry(),'dev-corpus')[0]
        row['task']['question']='<script>bad()</script>'
        html=m.case_html(row,[])
        self.assertIn('&lt;script&gt;',html);self.assertNotIn('<script>bad()',html)

class PreviewClockTests(unittest.TestCase):
    def test_pts_mapping_honors_original_container_origin(self):
        m=api(self,'research.av_data.visualize_batch')
        self.assertTrue(hasattr(m,'source_time'),'Missing exact original PTS conversion')
        origin={'pts_to_source':{'scale':1.0,'offset_s':-12.5}}
        self.assertEqual(m.source_time(14.5,origin),2.0)
    def test_missing_clock_mapping_is_rejected(self):
        m=api(self,'research.av_data.visualize_batch')
        self.assertTrue(hasattr(m,'source_time'),'Missing exact original PTS conversion')
        with self.assertRaises(ValueError):m.source_time(2,{})

class RuntimeTests(unittest.TestCase):
    def test_actor_controller_never_receives_private_reference(self):
        m=api(self,'research.av_train.rollout');seen=[]
        class Runtime:
            source_ids=[A,B,C]
            cost={'tool_calls':0,'requested_av_seconds':0}
            def execute(self,action):self.cost['tool_calls']+=1;return {'status':'ok','results':[{'source_id':A}]}
        def controller(history):
            seen.append(copy.deepcopy(history))
            if len(seen)==1:return {'action':{'tool':'search_sources','arguments':{'query':'machine','top_k':2}}}
            return {'final':prediction()}
        teacher={'kind':'fixture','model':'scripted-fixture','revision':'v1'}
        trace=m.run_episode(task(),Runtime(),controller,teacher)
        self.assertNotIn('evidence_packages',json.dumps(seen))
        self.assertEqual(trace['final']['cost']['tool_calls'],1)
        self.assertEqual(trace['teacher']['kind'],'fixture')
    def test_controller_cannot_override_measured_cost(self):
        m=api(self,'research.av_train.rollout')
        class Runtime:
            source_ids=[A,B,C]
            cost={'tool_calls':3,'requested_av_seconds':12}
        trace=m.run_episode(task(),Runtime(),lambda history:{'final':prediction()}, {'kind':'fixture','model':'script','revision':'v1'})
        self.assertEqual(trace['final']['cost'],Runtime.cost)
    def test_only_metadata_trajectory_cannot_be_av_sft(self):
        m=api(self,'research.av_train.export');t,v,obs=TrainingTests.make_trace(self)
        action={'tool':'search_sources','arguments':{'query':'machine'}}
        observation={'status':'ok','results':[{'source_id':A}]}
        t['steps']=[api(self,'research.av_train.trajectory').make_step(action,observation)]
        with self.assertRaisesRegex(ValueError,'AV observation'):m.export_trajectory(t,{A:'train'},lambda a:observation,v)
    def test_citation_outside_observed_window_rejected(self):
        m=api(self,'research.av_train.export');t,v,obs=TrainingTests.make_trace(self)
        t['final']['evidence'][0].update(start_s=8,end_s=10)
        with self.assertRaisesRegex(ValueError,'observed'):m.export_trajectory(t,{A:'train'},lambda a:obs,v)

class FrameExtractionTests(unittest.TestCase):
    def test_ffmpeg_frame_uses_logged_original_pts(self):
        import subprocess
        m=api(self,'research.av_data.visualize_batch')
        self.assertTrue(hasattr(m,'frame_at_or_after'),'Missing ffmpeg review extraction')
        with tempfile.TemporaryDirectory() as directory:
            media=Path(directory)/'fixture.mkv'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=64x64:r=5:d=1','-c:v','ffv1','-output_ts_offset','12.5',str(media)],check=True)
            image,seconds=m.frame_at_or_after(str(media),.25,{'pts_to_source':{'scale':1.,'offset_s':-12.5}})
            self.assertEqual(image.size,(64,64));self.assertAlmostEqual(seconds,.4,places=3)

class BatchExportTests(unittest.TestCase):
    def test_rejected_fixture_writes_explicit_zero_retained_summary(self):
        m=api(self,'research.av_train.export');self.assertTrue(hasattr(m,'export_batch'),'Missing batch retention pipeline')
        trace,v,obs=TrainingTests.make_trace(self);trace['teacher']['kind']='fixture'
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'export';result=m.export_batch([trace],{A:'train'},lambda t:lambda a:obs,{'t1':v},out)
            self.assertEqual(result['retained_count'],0);self.assertEqual(result['rejected_count'],1)
            self.assertEqual(json.loads((out/'retained.json').read_text()),[])
            self.assertFalse(result['training_ready'])
    def test_retained_export_records_lineage_and_masking(self):
        m=api(self,'research.av_train.export');self.assertTrue(hasattr(m,'export_batch'),'Missing batch retention pipeline')
        trace,v,obs=TrainingTests.make_trace(self)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'export';result=m.export_batch([trace],{A:'train'},lambda t:lambda a:obs,{'t1':v},out)
            self.assertEqual(result['retained_count'],1)
            self.assertFalse(result['training_ready'])
            lineage=json.loads((out/'lineage.private.json').read_text())
            self.assertEqual(lineage[0]['task_id'],'t1')
            config=(out/'data-settings.yaml').read_text()
            self.assertIn('train_on_prompt: false',config);self.assertIn('mask_history: false',config)

class SerializationTests(unittest.TestCase):
    def test_exported_action_matches_controller_json_protocol(self):
        m=api(self,'research.av_train.export');t,v,obs=TrainingTests.make_trace(self)
        row=m.export_trajectory(t,{A:'train'},lambda a:obs,v)
        self.assertEqual(json.loads(row['conversations'][1]['value']),{'action':t['steps'][0]['action']})
        self.assertEqual(set(json.loads(row['conversations'][-1]['value'])),{'final'})
        self.assertNotIn('cost',json.loads(row['conversations'][-1]['value'])['final'])
        self.assertEqual(row['tools'],'')
        self.assertIn('Tools: ',row['system'])

if __name__=='__main__':unittest.main()

class TrainingPoolTests(unittest.TestCase):
    def test_reserved_dev_hash_cannot_enter_new_training_pool(self):
        m=api(self,'research.av_train.prepare_pool')
        row={'sha256':'a'*64,'status':'candidate_unreviewed_av','asset_path':'x'}
        accepted,rejected=m.screen_candidates([{'sources':[row]}],{'a'*64})
        self.assertEqual(accepted,[]);self.assertEqual(rejected[0]['reason'],'reserved_or_duplicate_asset')
    def test_silent_seed_is_not_promoted_to_training_candidate(self):
        m=api(self,'research.av_train.prepare_pool')
        row={'sha256':'b'*64,'status':'rejected_silent_or_near_silent','asset_path':'x'}
        accepted,rejected=m.screen_candidates([{'sources':[row]}],set())
        self.assertEqual(accepted,[]);self.assertEqual(rejected[0]['reason'],'acquisition_not_accepted_av')

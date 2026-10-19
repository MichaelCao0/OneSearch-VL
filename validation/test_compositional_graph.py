import copy
import unittest
from research.av_train.compositional_graph import validate_graph,validate_task,actor_case,validate_anchors

class GraphTests(unittest.TestCase):
    def setUp(self):
        self.anchors={'anchors':[{'id':'a1','modality':'audio','start_s':0,'end_s':4,'observation':'audible melody','entity_hint':'piece','search_queries':['melody title']},
            {'id':'a2','modality':'visual','start_s':4,'end_s':9,'observation':'visible poster','entity_hint':'composer','search_queries':['poster text']}]}
        self.pages={'p1':{'url':'https://example.org/work','content':'The work was composed by Composer One. Composer One was born in 1800. Composer Two was born in 1825.','kind':'fetched_web_body'}}
        self.graph={'entities':[{'id':'e1','name':'Hidden Work','anchor_ids':['a1'],'binding_verified':True,'binding_reason':'melody matches'},
            {'id':'e2','name':'Composer One','anchor_ids':[]},{'id':'e3','name':'Composer Two','anchor_ids':['a2'],'binding_verified':True,'binding_reason':'poster text'}],
            'facts':[{'id':'f1','head':'e1','relation':'composer','tail':'e2','tail_kind':'entity','source_id':'p1','quote':'The work was composed by Composer One.'},
                {'id':'f2','head':'e2','relation':'birth_year','tail':'1800','tail_kind':'number','unit':'year','source_id':'p1','quote':'Composer One was born in 1800.'},
                {'id':'f3','head':'e3','relation':'birth_year','tail':'1825','tail_kind':'number','unit':'year','source_id':'p1','quote':'Composer Two was born in 1825.'}]}
        self.task={'task_id':'t1','question':'音频中作品的作曲家与海报所示作曲家的出生年份相差多少年？','accepted_answers':['25'],
            'operation':'arithmetic','anchor_ids':['a1','a2'],'fact_paths':[['f1','f2'],['f3']],
            'final_operation':{'kind':'absolute_difference','input_fact_ids':['f2','f3']}}
    def test_valid_derived_answer_need_not_appear_in_source(self):
        validate_anchors(self.anchors,10);validate_graph(self.graph,self.anchors,self.pages)
        self.assertEqual(validate_task(self.task,self.graph,self.anchors)['computed_answer'],25)
    def test_fabricated_quote_rejected(self):
        self.graph['facts'][0]['quote']='A made up unsupported fact.'
        with self.assertRaisesRegex(ValueError,'Quote'):validate_graph(self.graph,self.anchors,self.pages)
    def test_search_summary_not_body_evidence(self):
        self.pages['p1']['kind']='search_summary'
        with self.assertRaises(ValueError):validate_graph(self.graph,self.anchors,self.pages)
    def test_broken_hop_rejected(self):
        self.graph['facts'][1]['head']='e3'
        with self.assertRaisesRegex(ValueError,'Broken'):validate_task(self.task,self.graph,self.anchors)
    def test_wrong_arithmetic_rejected(self):
        self.task['accepted_answers']=['26']
        with self.assertRaisesRegex(ValueError,'wrong'):validate_task(self.task,self.graph,self.anchors)
    def test_mismatched_units_rejected(self):
        self.graph['facts'][2]['unit']='month'
        with self.assertRaisesRegex(ValueError,'units'):validate_task(self.task,self.graph,self.anchors)
    def test_entity_leak_rejected(self):
        self.task['question']='音频中的 Composer One 和海报所示作曲家出生年份相差多少？'
        with self.assertRaisesRegex(ValueError,'leaked'):validate_task(self.task,self.graph,self.anchors)
    def test_not_multi_hop_from_two_disconnected_facts(self):
        t={**self.task,'operation':'multi_hop','anchor_ids':['a1'],'fact_paths':[['f1','f3']],
            'final_operation':{'kind':'lookup','input_fact_ids':['f3']}}
        with self.assertRaisesRegex(ValueError,'Broken'):validate_task(t,self.graph,self.anchors)
    def test_actor_has_no_graph_answer_or_publisher(self):
        x=actor_case(self.task,{'source_path':'/tmp/media.mp4','source_sha256':'abc','publisher':'secret'})
        self.assertEqual(set(x),{'case_id','question','source_path','source_sha256'})
    def test_anchor_duration_invalid(self):
        self.anchors['anchors'][0]['end_s']=11
        with self.assertRaises(ValueError):validate_anchors(self.anchors,10)

    def test_failed_model_attempt_is_not_replayed(self):
        import tempfile,json
        from pathlib import Path
        from research.av_train.online_data_engine import model_call
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'model-attempts').mkdir()
            (out/'model-attempts/qa.json').write_text(json.dumps({'status':'running'}))
            with self.assertRaisesRegex(RuntimeError,'cannot be replayed'):
                model_call(out,'qa',None,'system',{})
    def test_comparison_requires_distinct_anchors(self):
        self.graph['entities'][2]['anchor_ids']=['a1']
        self.task['anchor_ids']=['a1']
        with self.assertRaisesRegex(ValueError,'distinct media anchors'):
            validate_task(self.task,self.graph,self.anchors)

    def test_native_question_only_uses_generate_content_without_media(self):
        from types import SimpleNamespace
        from unittest.mock import patch,Mock
        from research.av_train.online_data_engine import text_response
        teacher=SimpleNamespace(identity={'provider':'gemini_native'},base_url='https://example.org/api/v1beta',model='gemini-test',_key='fixture-not-real',settings=SimpleNamespace(timeout_s=1))
        response=Mock(status_code=200)
        response.json.return_value={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{\"can_answer_reliably\":false}'}]}}],'usageMetadata':{'candidatesTokenCount':10}}
        with patch('research.av_train.online_data_engine.requests.post',return_value=response) as post:
            text_response(teacher,'question','system',100)
            self.assertEqual(post.call_args.args[0],'https://example.org/api/v1beta/models/gemini-test:generateContent')
            self.assertEqual(post.call_args.kwargs['json']['contents'][0]['parts'],[{'text':'question'}])
    def test_openai_config_does_not_inherit_native_base_url(self):
        import os
        from unittest.mock import patch
        from onesearch_infer.openai_av_runner import OpenAIAVSettings
        with patch.dict(os.environ,{'ONESEARCH_TEACHER_BASE_URL':'https://example.org/api/v1beta','ONESEARCH_TEACHER_MODEL':'gemini-test','ONESEARCH_TEACHER_API_KEY':'fixture-not-real'},clear=True):
            cfg=OpenAIAVSettings(base_url_env='ONESEARCH_COMPOSITIONAL_TEACHER_BASE_URL',default_base_url='https://example.org/api/v1')
            self.assertEqual(cfg.resolve()[1],'https://example.org/api/v1')

    def test_known_bad_video_transport_rejected_before_model_request(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from research.av_train.online_data_engine import model_call
        teacher=SimpleNamespace(identity={'provider':'openai_compatible_av','model_id':'gemini-3.8-flash','endpoint_sha256':'5b1eae318300cda1510e4ef520b3e382d0c5300acfd9d8cc81eb8ccbc36b2f4b'})
        with patch('research.av_train.online_data_engine.requests.post') as post:
            with self.assertRaisesRegex(ValueError,'video-content mismatch'):
                model_call(Path('/nonexistent'),'anchor',teacher,'system',{},media='/not-read.mp4')
            post.assert_not_called()
    def test_native_transport_is_not_confused_with_bad_gateway(self):
        from types import SimpleNamespace
        from research.av_train.online_data_engine import ensure_media_transport
        ensure_media_transport(SimpleNamespace(identity={'provider':'gemini_native','model_id':'gemini-3.8-flash','endpoint_sha256':'other-native-endpoint'}))

if __name__=='__main__':unittest.main()

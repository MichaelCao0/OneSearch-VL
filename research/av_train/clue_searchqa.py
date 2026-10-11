"""Audio-video query clues have independent identities and no publisher lookup.

Corpus targets must first be returned by a real search. Raw clue provenance is
saved privately; the controller receives only local intervals and observations.
"""
from pathlib import Path
from research.av_bench.contracts import json_copy
from research.av_data.tasks import _opaque
from research.av_data.media_clock import sha256_file
from research.av_data.build_research_batch import write_json
from .searchqa import SearchQARuntime

CLUE_SYSTEM='''The initial source IDs identify anonymous query clips, not indexed publisher documents. They have no title, URL or text to read. Use inspect_av on the query clip to obtain audio-video clues, then search_sources to discover publisher sources. read_text and target inspect_av accept only IDs actually returned by search. Query clip time starts at zero; it is separate from the original publisher recording time. Do not infer that two IDs refer to the same recording from their strings or durations. Return one action or final in the stated JSON protocol.'''
PUBLIC_CLUE_FIELDS={'schema_version','status','original_interval','modalities','model_executed','model_executed_this_episode','semantic_status','semantic_verified','observation','actual_cost','capabilities'}

def clue_receipt(receipt,clue_id,internal_source_id):
    def project(value):
        if isinstance(value,dict):
            return {k:project(v) for k,v in value.items() if k not in {'asset_path','output_path','asset_sha256','output_sha256','record_sha256','origin_url','public_metadata','parent_source_id','parent_sha256','private_path','asset_root'}}
        if isinstance(value,list):return [project(v) for v in value]
        if isinstance(value,str):return value.replace(internal_source_id,clue_id)
        return value
    public=project({k:v for k,v in receipt.items() if k in PUBLIC_CLUE_FIELDS})
    public['source_id']=clue_id;public['time_axis']='query_clip_local_presentation'
    return public

class ClueSearchQARuntime(SearchQARuntime):
    def __init__(self,index_dir,scope,sources,bindings,output_dir,budget,observer=None):
        corpus_ids=set(scope['source_ids']);clues={}
        for binding in bindings:
            if not isinstance(binding,dict) or set(binding)!={'clue_id','source'}:raise ValueError('Closed clue binding required')
            clue_id=_opaque(binding['clue_id']);source=binding['source']
            if clue_id in corpus_ids or clue_id in clues:raise ValueError('Clue/corpus identity collision')
            if source.get('public_metadata')!={} or source.get('media_type')!='audio_video':raise ValueError('Clue source must be anonymous audio-video')
            if sha256_file(source['asset_path'])!=source['asset_sha256']:raise ValueError('Clue asset hash mismatch')
            clues[clue_id]=json_copy(source)
        if not clues:raise ValueError('At least one query clue required')
        all_sources=dict(sources);all_sources.update(clues)
        super().__init__(index_dir,scope,all_sources,output_dir,budget,observer)
        self.clues=clues;self.source_ids+=list(clues);self.discovered=set()
    def tool_definitions(self):
        # Do not publish the observer's entire corpus window registry.
        from .rollout import TOOL_DEFINITIONS
        definitions=json_copy(TOOL_DEFINITIONS)
        next(t for t in definitions if t['name']=='inspect_av')['constraints']={
          'query_clips':[{'source_id':sid,'duration_s':s['duration_s'],'time_axis':'query_clip_local_presentation',
             'suggested_first_window':{'start_s':0.,'end_s':min(2.,s['duration_s'])}} for sid,s in self.clues.items()],
          'publisher_sources':'only after actual search discovery; use original-source time',
          'max_window_seconds':getattr(self.observer,'max_clip_seconds',2.)}
        next(t for t in definitions if t['name']=='read_text')['constraints']={'query_clips_have_text':False,'requires_search_discovery':True}
        return definitions
    def rejection(self,message):
        if self.cost['tool_calls']>=self.budget['max_tool_calls']:raise ValueError('Tool budget exceeded')
        self.cost['tool_calls']+=1
        return {'status':'tool_error','error_type':'UndiscoveredSource','message':message}
    def execute(self,action):
        tool=action.get('tool');a=action.get('arguments',{})
        if tool in {'read_text','inspect_av'}:
            sid=a.get('source_id')
            if tool=='read_text' and sid in self.clues:return self.rejection('Query clips have no publisher text. Inspect the query clip and search for a source.')
            if sid not in self.clues and sid not in self.discovered:return self.rejection('Source not discovered. Call search_sources before reading or inspecting a publisher source.')
        # Native media preparation and semantic observers both see the actual
        # clipped SourceRecord, not an alias carrying a target title or URL.
        if tool=='inspect_av' and a.get('source_id') in self.clues:
            from .rollout import ToolRuntime
            receipt=ToolRuntime.execute(self,action)
            write_json(self.output_dir/f'clue-receipt-{self.cost["tool_calls"]:03d}.private.json',receipt)
            return clue_receipt(receipt,a['source_id'],self.clues[a['source_id']]['source_id'])
        receipt=super().execute(action)
        if tool=='search_sources' and receipt['status']=='ok':
            self.discovered.update(r['source_id'] for r in receipt['results'])
            if self.observer is not None and hasattr(self.observer,'available_windows'):
                for r in receipt['results']:
                    r['available_inspection_windows']=[w for w in self.observer.available_windows if w['source_id']==r['source_id']]
        return receipt

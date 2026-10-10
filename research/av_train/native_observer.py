"""Native observer adapter and independently verified fixed-observer replay.

Private model records and media receipts remain separate from actor observations.
Cache replay verifies original/derived bytes, exact input binding and fixed backend
identity; it reuses an actual executed model record rather than inventing outputs
or regenerating potentially nondeterministic descriptions. Original observer cost
is included in the actor receipt. Cache access wall time is a separate audit cost.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from research.av_agent.observer import prepare_av_input
from research.av_agent.semantic_observer import (make_observer_input,validate_observation,
                                                validate_observer_input,DEFAULT_QUESTION)
from research.av_data.media_clock import sha256_file
from research.av_data.build_research_batch import write_json
from research.av_train.trajectory import digest


def observation_receipt(record,prepared_cost,backend_metrics):
    if validate_observation(record):raise ValueError('Invalid native model record')
    bound=record['input'];success=record['status']=='model_executed'
    fixture=record['status']=='contract_test_executed'
    cost=record['actual_cost']
    result={
      'schema_version':'av-research-native-observation.v1',
      'status':'ok' if success else record['status'],
      'source_id':bound['source_id'],'asset_sha256':bound['asset_sha256'],
      'output_sha256':bound['output_sha256'],'original_interval':deepcopy(bound['original_interval']),
      'modalities':deepcopy(bound['modalities']),'question':bound['question'],
      'backend_identity':deepcopy(record['backend_identity']),
      'semantic_status':'observed_unreviewed' if success else 'contract_fixture' if fixture else 'unavailable',
      'model_executed':success,'semantic_verified':False,
      'observation':deepcopy(record['structured_observation']) if success or fixture else None,
      'error':deepcopy(record['error']),
      'actual_cost':{'observer_backend_attempts':cost['backend_attempts'],
                    'observer_provider_requests':cost['provider_requests'],
                    'observer_wall_seconds':cost['wall_seconds'],
                    'prepared_video_frames':prepared_cost.get('returned_video_frames',0),
                    'prepared_audio_samples':prepared_cost.get('decoded_audio_samples',0),
                    'prepared_output_bytes':prepared_cost.get('output_bytes',bound['output_bytes'])},
    }
    # Known numeric model costs are charged, full backend diagnostic paths stay private.
    for key in ('input_tokens','generated_tokens','input_video_frames','input_audio_samples'):
        value=backend_metrics.get(key)
        if isinstance(value,int) and not isinstance(value,bool):result['actual_cost'][key]=value
    return result


class NativeObserverAdapter:
    def __init__(self,observer,*,max_clip_seconds=2.,question=DEFAULT_QUESTION,backend=None):
        if isinstance(max_clip_seconds,bool) or not isinstance(max_clip_seconds,(int,float)) or max_clip_seconds<=0:
            raise ValueError('Bounded native observer duration required')
        self.observer=observer;self.max_clip_seconds=max_clip_seconds;self.question=question;self.backend=backend
    def __call__(self,source,start_s,end_s,output_dir):
        if end_s-start_s>self.max_clip_seconds:raise ValueError('Native observer window exceeds configured duration')
        out=Path(output_dir)
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=False)
        derived,cost=prepare_av_input(source,start_s,end_s,out/'media')
        bound=make_observer_input(source,derived,question=self.question)
        record=self.observer.observe(bound)
        metrics=self.backend.last_metrics if self.backend is not None else {}
        write_json(out/'derived.private.json',derived);write_json(out/'input.private.json',bound)
        write_json(out/'observation.private.json',record);write_json(out/'preparation-cost.private.json',cost)
        write_json(out/'backend-metrics.private.json',metrics)
        receipt=observation_receipt(record,cost,metrics);write_json(out/'receipt.json',receipt)
        write_json(out/'execution-manifest.private.json',{
            'schema_version':'av-observer-execution-sidecar.v1','backend_identity':record['backend_identity'],
            'input_sha256':bound['input_sha256'],'observation_sha256':digest(record),
            'receipt_sha256':digest(receipt),'source_id':source['source_id'],
            'note':'Integrity and declared execution; semantic review remains separate.'})
        return receipt


class RecordedObserverReplay:
    """Replay independently stored observer records under the same fixed identity.

    This object reads native execution directories, never the trajectory being
    checked. Caller supplies a trusted expected backend manifest. A matching
    content hash proves integrity, not authentication of a malicious provider.
    """
    def __init__(self,execution_dirs,*,expected_backend,question=DEFAULT_QUESTION):
        self.expected_backend=deepcopy(expected_backend);self.question=question;self.records={}
        for root in map(Path,execution_dirs):
            record=json.loads((root/'observation.private.json').read_text())
            if validate_observation(record):raise ValueError('Invalid independent observer cache record')
            if record['backend_identity']!=expected_backend:raise ValueError('Observer cache backend identity mismatch')
            bound=record['input'];window=bound['original_interval']
            key=(bound['source_id'],bound['asset_sha256'],window['start_s'],window['end_s'],bound['question'])
            if key in self.records:raise ValueError('Ambiguous observer cache key; use one fixed executed record')
            self.records[key]=root
    def __call__(self,source,start_s,end_s,output_dir):
        key=(source['source_id'],source['asset_sha256'],start_s,end_s,self.question)
        if key not in self.records:raise ValueError('No exact observer cache binding')
        root=self.records[key]
        record=json.loads((root/'observation.private.json').read_text())
        cost=json.loads((root/'preparation-cost.private.json').read_text())
        metrics=json.loads((root/'backend-metrics.private.json').read_text())
        if validate_observation(record) or record['backend_identity']!=self.expected_backend:raise ValueError('Changed observer cache record or identity')
        if sha256_file(source['asset_path'])!=source['asset_sha256']:raise ValueError('Changed original observer source')
        if validate_observer_input(record['input'],verify_files=True):raise ValueError('Changed observer cache AV input')
        receipt=observation_receipt(record,cost,metrics)
        manifest=json.loads((root/'execution-manifest.private.json').read_text())
        if manifest['observation_sha256']!=digest(record) or manifest['receipt_sha256']!=digest(receipt) or manifest['input_sha256']!=record['input']['input_sha256']:
            raise ValueError('Observer cache execution manifest mismatch')
        return receipt

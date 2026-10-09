"""Private semantic observation contracts; no model/service is silently selected.

The CPU preflight does not make network requests or download weights. A trusted
future backend supplies a pinned manifest and infer(bound_input); declared model
execution and independently reviewed semantic capability remain separate layers.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from research.av_data.media_clock import sha256_file
from research.av_data.validate import validate_derived

INPUT_SCHEMA = 'av-observer-input-0.1'
OBSERVATION_SCHEMA = 'av-semantic-observation-0.1'
_HASH = re.compile(r'^[0-9a-f]{64}$')
_REVISION = re.compile(r'^[0-9a-f]{40,64}$')
DEFAULT_QUESTION = 'Describe only directly observed audible and visible events in the requested interval, their source-local times, and uncertainty. Distinguish correspondence from causal attribution; abstain when support is absent.'
_INPUT_KEYS = {'schema_version','source_id','asset_sha256','derived_id','output_sha256','output_bytes',
               'clip_path','original_interval','mapping','modalities','input_mode','question','probe_id','input_sha256'}
_MANIFEST_KEYS = {'backend_id','model_id','model_revision','config_sha256','mode','transport',
                  'execution_kind','runtime_device','backend_code_revision'}
_RECORD_KEYS = {'schema_version','observation_id','created_at','status','input','backend_identity',
                'model_executed','raw_output','raw_output_sha256','structured_observation','error',
                'human_validation','raw_av_observer_ready','actual_cost','record_sha256'}


def _json(value):
    text=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    text.encode('utf-8')  # Reject lone surrogates before storing/hashing a reply.
    return text


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _without(value,key):
    return {k:v for k,v in value.items() if k!=key}


def _finite(value):
    try:return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)
    except OverflowError:return False


def _text(value,limit=65536):
    if not isinstance(value,str) or not value.strip() or len(value)>limit:return False
    try:value.encode('utf-8')
    except UnicodeError:return False
    return True


def _hash(value):
    return isinstance(value,str) and bool(_HASH.fullmatch(value))


def make_observer_input(source,derived,question=DEFAULT_QUESTION,probe_id=None):
    """Verify bytes/parent clocks, then minimize private backend input.

    Existing DerivedRecord validation checks original/clip hashes and shared AV
    clocks. Its media probes are real work and are included by media accounting.
    No publisher metadata, gold, source URL or future observation is included.
    """
    errors=validate_derived(derived,source)
    if errors:raise ValueError('Invalid observer media binding')
    record={'schema_version':INPUT_SCHEMA,'source_id':source['source_id'],
            'asset_sha256':source['asset_sha256'],'derived_id':derived['derived_id'],
            'output_sha256':derived['output_sha256'],'output_bytes':derived['output_bytes'],
            'clip_path':str(Path(derived['output_path']).resolve()),
            'original_interval':deepcopy(derived['original_interval']),
            'mapping':deepcopy(derived['mapping']),
            'modalities':sorted({s['codec_type'] for s in derived['mapping']['streams']}),
            'input_mode':'raw_av','question':question,'probe_id':probe_id}
    record['input_sha256']=_digest(record)
    if validate_observer_input(record,verify_files=True):raise ValueError('Invalid observer input contract')
    return record


def validate_observer_input(record,verify_files=False):
    """Validate binding digest/clocks; optionally rehash the actual clip at call time."""
    if not isinstance(record,dict) or set(record)!=_INPUT_KEYS:return ['invalid input fields']
    try:_json(record)
    except (TypeError,ValueError,RecursionError):return ['input is not finite JSON']
    errors=[]
    if record['schema_version']!=INPUT_SCHEMA:errors.append('unsupported input schema')
    if not _hash(record['asset_sha256']) or record['source_id']!='src_'+record['asset_sha256'][:24]:errors.append('invalid source identity')
    if not _hash(record['output_sha256']) or record['derived_id']!='derived-'+record['output_sha256'][:24]:errors.append('invalid derived identity')
    if not _hash(record['input_sha256']) or record['input_sha256']!=_digest(_without(record,'input_sha256')):errors.append('input digest mismatch')
    if not _text(record['question'],4000):errors.append('invalid bounded question')
    if record['probe_id'] is not None and (not _text(record['probe_id'],100)):errors.append('invalid probe id')
    if record['input_mode']!='raw_av':errors.append('unsupported media input mode')
    if record['modalities'] not in (['audio'],['video'],['audio','video']):errors.append('invalid modalities')
    if not isinstance(record['output_bytes'],int) or isinstance(record['output_bytes'],bool) or record['output_bytes']<1:errors.append('invalid byte count')
    if not isinstance(record['clip_path'],str) or not Path(record['clip_path']).is_absolute():errors.append('invalid private clip path')
    interval=record['original_interval']
    if not isinstance(interval,dict) or set(interval)!={'start_s','end_s'} or any(not _finite(interval.get(k)) for k in ('start_s','end_s')):
        return errors+['invalid source interval']
    start,end=interval['start_s'],interval['end_s']
    if not 0<=start<end:errors.append('invalid source interval')
    mapping=record['mapping']
    if not isinstance(mapping,dict) or (mapping.get('axis')!='source_local_presentation' or mapping.get('interval')!='half_open' or
      mapping.get('output_axis')!='presentation_pts_seconds' or mapping.get('output_to_source')!={'scale':1.0,'offset_s':start} or
      mapping.get('source_to_output')!={'scale':1.0,'offset_s':-start}):errors.append('invalid reversible clock mapping')
    elif (not isinstance(mapping.get('streams'),list) or any(not isinstance(item,dict) or item.get('codec_type') not in ('audio','video') for item in mapping['streams']) or sorted({item['codec_type'] for item in mapping['streams']})!=record['modalities']):
        errors.append('mapping/modalities mismatch')
    if verify_files and not errors:
        clip=Path(record['clip_path'])
        if not clip.is_file() or clip.stat().st_size!=record['output_bytes'] or sha256_file(clip)!=record['output_sha256']:errors.append('clip byte/hash mismatch')
    return errors


def validate_backend_identity(identity):
    if not isinstance(identity,dict) or set(identity)!=_MANIFEST_KEYS:return ['invalid backend identity fields']
    errors=[]
    for key in ('backend_id','model_id','runtime_device','backend_code_revision'):
        if not _text(identity[key],300):errors.append('invalid backend '+key)
    if not isinstance(identity['model_revision'],str) or not _REVISION.fullmatch(identity['model_revision']):errors.append('model revision is not pinned')
    if not _hash(identity['config_sha256']):errors.append('config hash is not pinned')
    if identity['mode'] not in ('native_joint_av','asr_plus_frames','asr_only','visual_only'):errors.append('unsupported explicit backend mode')
    if identity['transport'] not in ('local','service'):errors.append('unsupported backend transport')
    if identity['execution_kind'] not in ('model_inference','contract_test'):errors.append('unsupported execution kind')
    return errors


def _structured_errors(value,bound,mode=None):
    if not isinstance(value,dict) or set(value)!={'events','uncertainty','cannot_determine'}:return ['invalid structured observation fields']
    try:_json(value)
    except (TypeError,ValueError,RecursionError):return ['structured observation is not finite JSON']
    if not isinstance(value['events'],list) or len(value['events'])>100:return ['invalid bounded events']
    errors=[]
    if not _text(value['uncertainty'],4000):errors.append('explicit uncertainty required')
    if not isinstance(value['cannot_determine'],list) or any(not _text(item,1000) for item in value['cannot_determine']):errors.append('invalid cannot_determine list')
    for event in value['events']:
        if not isinstance(event,dict) or set(event)!={'modality','description','source_interval','uncertainty'}:
            errors.append('invalid event fields');continue
        if event['modality'] not in ('audio','visual','joint_av') or not _text(event['description'],4000) or not _text(event['uncertainty'],1000):errors.append('invalid event content')
        interval=event['source_interval']
        if not isinstance(interval,dict) or set(interval)!={'start_s','end_s'} or any(not _finite(interval.get(k)) for k in ('start_s','end_s')):
            errors.append('invalid event interval');continue
        if not bound['original_interval']['start_s']<=interval['start_s']<interval['end_s']<=bound['original_interval']['end_s']:
            errors.append('event support outside requested interval')
        if (mode=='visual_only' and event['modality']!='visual') or (mode=='asr_only' and event['modality']!='audio'):errors.append('event exceeds backend modality')
        required={'audio'} if event['modality']=='audio' else {'video'} if event['modality']=='visual' else {'audio','video'}
        if not required<=set(bound['modalities']):errors.append('event claims absent modality')
    return errors


def validate_observation(record):
    """Check integrity and execution/readiness consistency, not semantic truth.

    This v0.1 record writer cannot approve a human review. Review evidence belongs
    to an independent, future adjudicated probe report. Hashes are integrity
    checks, not signatures or authentication of a backend's execution claim.
    """
    if not isinstance(record,dict) or set(record)!=_RECORD_KEYS:return ['invalid observation fields']
    try:_json(record)
    except (TypeError,ValueError,RecursionError):return ['observation is not finite JSON']
    errors=validate_observer_input(record['input'])
    if record['schema_version']!=OBSERVATION_SCHEMA:errors.append('unsupported observation schema')
    if not _hash(record['record_sha256']) or record['record_sha256']!=_digest(_without(record,'record_sha256')):errors.append('observation digest mismatch')
    if not _text(record['observation_id'],100) or not _text(record['created_at'],100):errors.append('invalid observation identity')
    if record['human_validation']!={'status':'pending','reviewer_id':None,'reviewed_at':None,'reference_evidence_sha256':None}:errors.append('independent human review is pending in this contract')
    if record['raw_av_observer_ready'] is not False:errors.append('capability cannot be claimed without adjudicated probes')
    identity=record['backend_identity']
    if identity is not None:errors+=validate_backend_identity(identity)
    expected_executed=record['status']=='model_executed'
    if record['model_executed'] is not expected_executed:errors.append('model execution/status mismatch')
    if record['status'] not in ('unavailable','failed','budget_exceeded','contract_test_executed','model_executed'):errors.append('invalid observation status')
    raw=record['raw_output']
    if raw is None:
        if record['raw_output_sha256'] is not None:errors.append('raw hash without output')
    elif not _text(raw) or record['raw_output_sha256']!=hashlib.sha256(raw.encode()).hexdigest():errors.append('raw output digest mismatch')
    success=record['status'] in ('model_executed','contract_test_executed')
    if success:
        if not identity or validate_backend_identity(identity):errors.append('successful execution lacks valid pinned identity')
        elif (identity['execution_kind']=='model_inference')!=expected_executed:errors.append('execution kind/status mismatch')
        if raw is None or record['error'] is not None:errors.append('success missing raw output or has error')
        if not validate_observer_input(record['input']):errors+=_structured_errors(record['structured_observation'],record['input'],identity.get('mode') if isinstance(identity,dict) else None)
    elif record['structured_observation'] is not None or not isinstance(record['error'],dict):errors.append('failure cannot supply structured semantic support')
    cost=record['actual_cost']
    if not isinstance(cost,dict) or set(cost)!={'backend_attempts','provider_requests','wall_seconds'}:errors.append('invalid actual cost fields')
    else:
        if cost['backend_attempts'] not in (0,1) or isinstance(cost['backend_attempts'],bool):errors.append('invalid backend attempts')
        if cost['provider_requests'] is not None and (not isinstance(cost['provider_requests'],int) or isinstance(cost['provider_requests'],bool) or cost['provider_requests']<0):errors.append('invalid provider request count')
        if not _finite(cost['wall_seconds']) or cost['wall_seconds']<0:errors.append('invalid observer wall seconds')
    return errors


class SemanticObserver:
    """Fail-closed trusted backend boundary with fixed call budget.

    Backend.manifest is the exact identity schema; infer(input) returns exactly
    backend_identity, consumed_input_sha256, raw_output, structured_observation,
    provider_requests. A backend must account for internal retries in that last
    field. Exceptions preserve only safe class names; provider work then stays
    unknown for services, rather than being invented as zero.
    """
    def __init__(self,backend=None,max_calls=20):
        if isinstance(max_calls,bool) or not isinstance(max_calls,int) or max_calls<1:raise ValueError('max_calls must be positive integer')
        self._backend=backend;self._max_calls=max_calls
        self._identity=deepcopy(backend.manifest) if backend is not None else None
        if self._identity is not None and validate_backend_identity(self._identity):raise ValueError('Backend identity must be pinned and explicit')
        self._counts={'calls':0,'backend_attempts':0,'failures':0,'provider_requests_known':0,'provider_requests_unknown_attempts':0}

    @property
    def counts(self):return deepcopy(self._counts)

    @property
    def capabilities(self):
        return {'semantic_observer_contract_ready':True,'backend_configured':self._backend is not None,
                'raw_av_observer_ready':False,'model':None if self._identity is None else self._identity['model_id'],
                'model_revision':None if self._identity is None else self._identity['model_revision'],
                'mode':'unavailable' if self._identity is None else self._identity['mode'],
                'semantic_probe_count':0,'human_semantic_validation_status':'pending'}

    def observe(self,bound_input):
        if validate_observer_input(bound_input,verify_files=True):raise ValueError('Input contract changed or is invalid')
        started=time.monotonic();self._counts['calls']+=1
        record={'schema_version':OBSERVATION_SCHEMA,'observation_id':'observation-'+uuid.uuid4().hex,
                'created_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                'status':'unavailable','input':deepcopy(bound_input),'backend_identity':deepcopy(self._identity),
                'model_executed':False,'raw_output':None,'raw_output_sha256':None,'structured_observation':None,
                'error':{'code':'backend_unavailable'},'raw_av_observer_ready':False,
                'human_validation':{'status':'pending','reviewer_id':None,'reviewed_at':None,'reference_evidence_sha256':None},
                'actual_cost':{'backend_attempts':0,'provider_requests':0,'wall_seconds':0.0}}
        if self._counts['calls']>self._max_calls:
            record.update(status='budget_exceeded',error={'code':'observer_call_budget_exceeded'})
        elif self._backend is not None:
            self._counts['backend_attempts']+=1;record['actual_cost']['backend_attempts']=1
            record['actual_cost']['provider_requests']=0 if self._identity['transport']=='local' else None
            try:
                reply=self._backend.infer(deepcopy(bound_input))
            except Exception as exc:
                record.update(status='failed',error={'code':'backend_failed','type':type(exc).__name__})
            else:
                if isinstance(reply,dict):
                    if _text(reply.get('raw_output')):
                        record['raw_output']=reply['raw_output'];record['raw_output_sha256']=hashlib.sha256(reply['raw_output'].encode()).hexdigest()
                    requests=reply.get('provider_requests')
                    if isinstance(requests,int) and not isinstance(requests,bool) and requests>=0:record['actual_cost']['provider_requests']=requests
                valid=(isinstance(reply,dict) and set(reply)=={'backend_identity','consumed_input_sha256','raw_output','structured_observation','provider_requests'}
                       and reply['backend_identity']==self._identity and reply['consumed_input_sha256']==bound_input['input_sha256']
                       and record['raw_output'] is not None and isinstance(reply['provider_requests'],int)
                       and not isinstance(reply['provider_requests'],bool) and reply['provider_requests']>=0
                       and not _structured_errors(reply['structured_observation'],bound_input,self._identity['mode']))
                if not valid:record.update(status='failed',error={'code':'backend_contract_invalid'})
                else:
                    executed=self._identity['execution_kind']=='model_inference'
                    record.update(status='model_executed' if executed else 'contract_test_executed',model_executed=executed,
                                  structured_observation=deepcopy(reply['structured_observation']),error=None)
        if record['status'] in ('unavailable','failed','budget_exceeded'):self._counts['failures']+=1
        requests=record['actual_cost']['provider_requests']
        if requests is None:self._counts['provider_requests_unknown_attempts']+=1
        else:self._counts['provider_requests_known']+=requests
        record['actual_cost']['wall_seconds']=time.monotonic()-started
        record['record_sha256']=_digest(record)
        if validate_observation(record):raise ValueError('Observer produced invalid internal record')
        return record


def semantic_probe_templates():
    """Eight approved categories; unbound templates are not evaluated probes."""
    definitions=[
      ('visible_speaker_and_speech','Which visible person, if any, corresponds to the actual audible speech?',
       'Describe speech and visible speaking motion separately; only bind a speaker when replayable evidence supports it. Preserve uncertain or offscreen alternatives.'),
      ('nonverbal_impact_or_sound_change','Locate a nonverbal impact or audible change and any accompanying visible state.',
       'Localize the audible event in original source time and visible event independently; do not infer a unique cause from sound alone.'),
      ('offscreen_sound_and_candidates','Which sound is audible, and what source candidates remain plausible beyond the frame?',
       'Retain offscreen/ambiguous attribution when the clip cannot identify a source; public title is not sound-source truth.'),
      ('narration_or_edited_track','Does the audible track support a visible action, or could it be narration or editing?',
       'Distinguish co-occurrence, apparent synchronization, narration and edited-track uncertainty; avoid declaring live sound solely from matching rhythm.'),
      ('known_av_time_offset','Estimate correspondence between audible and visible events without silently realigning them.',
       'Compare against independently reviewed original-clock event intervals and known synthetic offset; report offset and uncertainty with a human-fixed tolerance.'),
      ('silence_or_audio_masking','Describe support when audio is silent or removed, including absent evidence.',
       'Do not hallucinate speech/sound. For a masking control remove audio-derived captions, transcripts and metadata from the model input as well.'),
      ('wrong_requested_interval','Does this specifically requested interval support the target event?',
       'Refuse support when the event is outside the interval; do not use observations from earlier/future clips as if the requested interval contained them.'),
      ('compatible_sources_or_cross_source_identity','Are multiple sound sources or cross-recording identities compatible with this evidence?',
       'Keep alternatives explicit; do not claim unique sound source, same identity, independence or shared recording without sufficient additional evidence.'),
    ]
    return [{'schema_version':'av-semantic-probe-template-0.1','probe_id':f'probe-{i:02d}','category':category,
             'question':question,'acceptance_criterion':criterion,'status':'input_and_human_reference_pending',
             'input_binding':None,'control_bindings':[],
             'human_reference':{'status':'pending','independent_reviewers':[],
                                'reference_evidence_sha256':None,'source_intervals':[],
                                'adjudicated_criterion':None,'time_tolerance_s':None},
             'report_separately_by':['natural','synthetic_timing'],
             'model_execution_status':'unavailable','semantic_validation_status':'pending',
             'denominator_policy':'Missing backend, failed calls and budget blocks remain in the fixed probe denominator; unbound templates are not completed model attempts.'}
            for i,(category,question,criterion) in enumerate(definitions,1)]


def candidate_model_manifest():
    """Approved plan candidate; no weight/revision/capability is invented."""
    return {'schema_version':'av-observer-candidate-0.1','candidate_model_id':'Qwen/Qwen2.5-Omni-7B',
            'candidate_mode':'native_joint_av','candidate_basis':'AV-DeepResearch-Next-Steps-2026-10-10.md',
            'model_revision':None,'config_sha256':None,'weight_hash_verified':False,
            'availability':'not_deployed','revision_status':'unpinned_candidate_only',
            'native_joint_av_capability':'not_verified_here','backend_implementation_status':'contract_only',
            'required_next_step':'Select and pin a supported local GPU runtime or authorized service, verify weight/config identities, bind probes with independent human reference evidence, then execute and adjudicate.',
            'prohibited_substitution':'OneSearch-VL-8B visual smoke or ASR+frames cannot establish native joint-AV capability.'}


def preflight_report():
    """Fresh local device/config presence only; never contacts a model service."""
    try:
        import torch
        device={'torch_version':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),
                'cuda_device_count':torch.cuda.device_count(),'mps_available':bool(torch.backends.mps.is_available()),
                'torch_import_status':'available'}
    except ImportError:
        device={'torch_version':None,'cuda_available':False,'cuda_device_count':0,'mps_available':False,'torch_import_status':'unavailable'}
    variables=('AV_OBSERVER_ENDPOINT','AV_OBSERVER_MODEL_ID','AV_OBSERVER_MODEL_REVISION','AV_OBSERVER_API_KEY')
    root=Path(__file__).resolve().parents[2]
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    return {'schema_version':'av-observer-preflight-0.1','created_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            'status':'backend_unavailable','device':device,'config_presence':{key:bool(os.getenv(key)) for key in variables},
            'config_presence_scope':'Only dedicated observer environment variables; no unrelated credential/config files read.',
            'source_git_head':revision,'observer_source_sha256':sha256_file(__file__),
            'environment_lock_sha256':sha256_file(root/'pixi.lock'),
            'candidate_model_manifest':candidate_model_manifest(),
            'readiness':{'contract_ready':True,'device_inventory_checked':True,'backend_implemented':False,
                         'backend_identity_pinned':False,'actual_model_execution_complete':False,
                         'human_semantic_validation_complete':False,'raw_av_observer_ready':False},
            'semantic_probe_denominator':8,'bound_probe_count':0,'model_executed_probe_count':0,
            'human_validated_probe_count':0,'blocked_probe_count':8,'actual_provider_requests':0,
            'network_requests':0,'weight_download_bytes':0,
            'semantic_claim':'No model inference or independent human semantic validation was executed by this preflight.'}


def write_preflight(output_dir):
    directory=Path(output_dir)
    directory.mkdir(parents=True,exist_ok=False)
    report=preflight_report()
    for name,value in (('preflight.json',report),('probe-templates.json',semantic_probe_templates()),
                       ('candidate-model-manifest.json',candidate_model_manifest())):
        (directory/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (directory/'README.md').write_text(
      '# Observer preflight\n\nThis is a device/config inventory and eight fixed semantic probe templates. '
      'The candidate is unpinned and unavailable. No model/service was called and no semantic probe passed. '
      'All eight planned categories remain in the denominator; actual model attempts and human reviews are zero.\n\n'
      'Input bindings and reviewer evidence must be supplied in private artifacts. Natural and synthetic timing '
      'results must be reported separately. Contract unit-test fixtures do not establish model capability.\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    report=write_preflight(args.output_dir)
    print(json.dumps({'status':report['status'],'raw_av_observer_ready':False,
                      'model_executed_probe_count':0,'semantic_probe_denominator':8},sort_keys=True))

if __name__=='__main__':main()

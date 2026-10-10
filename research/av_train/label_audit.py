"""Audit actual repository LLaMA-Factory conversion, Qwen3-VL template and labels.

CPU only; local tokenizer only. No model load, backward or training-readiness claim.
An explicit fixture probes data preprocessing without pretending to be teacher data.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from .export import dataset_registration

ROOT=Path(__file__).resolve().parents[2]


def validate_settings(cutoff_len,train_on_prompt,mask_history):
    if train_on_prompt is not False:raise ValueError('train_on_prompt must be false: observations cannot be training targets')
    if mask_history is not False:raise ValueError('mask_history must be false: all assistant actions are training targets')
    if isinstance(cutoff_len,bool) or not isinstance(cutoff_len,int) or cutoff_len<32:raise ValueError('Invalid token cutoff')


def fixture_row():
    sid='src_'+'a'*24
    def message(role,value):return {'from':role,'value':json.dumps(value,ensure_ascii=False)}
    return {'system':'PREPROCESSING FIXTURE ONLY. Search with real tools when deployed. No teacher model ran in this fixture.',
            'tools':'','conversations':[
              message('human',{'question':'找到机械录制并说明观察到的动作。','target_questions':[{'fact_id':'f1','question':'运动是什么？'}]}),
              message('gpt',{'action':{'tool':'search_sources','arguments':{'query':'steam engine','top_k':3}}}),
              message('observation',{'status':'ok','results':[{'source_id':sid,'excerpt':'FIXTURE_METADATA_CONTEXT'}]}),
              message('gpt',{'action':{'tool':'inspect_av','arguments':{'source_id':sid,'start_s':2.,'end_s':4.}}}),
              message('observation',{'status':'contract_fixture','source_id':sid,'original_interval':{'start_s':2.,'end_s':4.},
                                     'observation':{'events':[{'modality':'joint_av','description':'FIXTURE_AV_CONTEXT',
                                     'source_interval':{'start_s':2.,'end_s':4.},'uncertainty':'Fixture only.'}]}}),
              message('gpt',{'final':{'status':'fixture_completed','facts':[{'fact_id':'f1','value':'FIXTURE_FINAL_TARGET'}],
                                     'ranked_sources':[sid],'ranked_windows':[],'evidence':[]}})]}


def source_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_components(tokenizer_dir,cutoff_len):
    sys.path.insert(0,str(ROOT/'SFT/src')) if str(ROOT/'SFT/src') not in sys.path else None
    from transformers import AutoTokenizer,AutoProcessor
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    media_processor=AutoProcessor.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    data=DataArguments(template='qwen3_vl_nothink',cutoff_len=cutoff_len,train_on_prompt=False,mask_history=False,packing=False)
    attr=DatasetAttr('file','audit-row');attr.join(dataset_registration('audit-only.json'))
    converter=SharegptDatasetConverter(attr,data)
    template=get_template_and_fix_tokenizer(tokenizer,data)
    processor=SupervisedDatasetProcessor(template,tokenizer,media_processor,data)
    return converter,template,processor,tokenizer


def audit_rows(rows,tokenizer_dir,*,cutoff_len=32768,train_on_prompt=False,mask_history=False,fixture=False):
    validate_settings(cutoff_len,train_on_prompt,mask_history)
    if not isinstance(rows,list) or not rows:raise ValueError('Cannot audit empty training rows')
    converter,template,processor,tokenizer=native_components(tokenizer_dir,cutoff_len)
    from llamafactory.extras.constants import IGNORE_INDEX
    reports=[]
    for i,row in enumerate(rows):
        if not isinstance(row,dict) or set(row)!={'system','tools','conversations'}:raise ValueError('Modular controller rows require closed text ShareGPT fields')
        turns=row['conversations']
        if not isinstance(turns,list) or len(turns)<4 or len(turns)%2:raise ValueError('Require multi-turn complete assistant final')
        for j,turn in enumerate(turns):
            if not isinstance(turn,dict) or set(turn)!={'from','value'}:raise ValueError('Invalid conversation turn')
            expected='human' if j==0 else 'gpt' if j%2 else 'observation'
            if turn['from']!=expected:raise ValueError('Action/observation role alignment mismatch')
        aligned=converter(row)
        messages=aligned['_prompt']+aligned['_response']
        if len(messages)!=len(turns):raise ValueError('Native converter discarded or changed conversation')
        pairs=template.encode_multiturn(tokenizer,messages,aligned['_system'],aligned['_tools'])
        if len(pairs)!=len(turns)//2:raise ValueError('Native template dropped assistant turns')
        # Detect truncation before the real processor can silently discard an early observation or final.
        expected_len=sum(len(s)+len(t) for s,t in pairs)+(1 if template.efficient_eos else 0)
        if expected_len>cutoff_len:raise ValueError(f'Untruncated row {i} length {expected_len} exceeds cutoff {cutoff_len}')
        ids,labels=processor._encode_data_example(aligned['_prompt'],aligned['_response'],aligned['_system'],
              aligned['_tools'],aligned['_images'] or [],aligned['_videos'] or [],aligned['_audios'] or [])
        if len(ids)!=expected_len or len(labels)!=expected_len:raise ValueError('Native processor truncated or altered encoded length')
        position=0;pair_reports=[]
        for pair_index,(source_ids,target_ids) in enumerate(pairs):
            if not target_ids:raise ValueError('Assistant action/final target lost')
            source_labels=labels[position:position+len(source_ids)]
            expected_mask=[IGNORE_INDEX]*len(source_ids)
            if template.efficient_eos and pair_index:expected_mask[0]=tokenizer.eos_token_id
            if source_labels!=expected_mask:raise ValueError('Source/observation tokens incorrectly trained')
            if ids[position:position+len(source_ids)]!=source_ids:raise ValueError('Native source IDs disagree with template')
            target_start=position+len(source_ids);target_end=target_start+len(target_ids)
            if labels[target_start:target_end]!=target_ids or ids[target_start:target_end]!=target_ids:
                raise ValueError('Assistant target missing or masked')
            assistant_text=tokenizer.decode(target_ids,skip_special_tokens=False)
            payload=json.loads(turns[pair_index*2+1]['value'])
            envelope='final' if pair_index==len(pairs)-1 else 'action'
            if set(payload)!={envelope} or envelope not in assistant_text:raise ValueError('Action/final JSON target mismatched')
            pair_reports.append({'pair_index':pair_index,'source_role':'human' if pair_index==0 else 'observation',
               'source_token_start':position,'source_token_count':len(source_ids),'source_ignore_count':source_labels.count(IGNORE_INDEX),
               'assistant_token_start':target_start,'assistant_token_count':len(target_ids),'assistant_envelope':envelope,
               'target_sha256':hashlib.sha256(turns[pair_index*2+1]['value'].encode()).hexdigest()})
            position=target_end
        if template.efficient_eos and ids[-1]!=tokenizer.eos_token_id:raise ValueError('Final EOS lost')
        reports.append({'row_index':i,'total_tokens':len(ids),'supervised_tokens':sum(x!=IGNORE_INDEX for x in labels),
                        'assistant_turns':len(pairs),'pairs':pair_reports,'row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()})
    files=['data/converter.py','data/template.py','data/processor/supervised.py','data/mm_plugin.py','data/formatter.py']
    token_files={p.name:source_hash(p) for p in Path(tokenizer_dir).iterdir() if p.is_file() and p.name in {'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','preprocessor_config.json','processor_config.json','video_preprocessor_config.json'}}
    return {'schema_version':'av-native-trainer-label-audit.v1','status':'passed','fixture_only':fixture,
            'row_count':len(reports),'template':'qwen3_vl_nothink','train_on_prompt':False,'mask_history':False,'cutoff_len':cutoff_len,
            'observation_targets_masked':True,'all_assistant_turns_supervised':True,'truncation_detected':False,
            'model_loaded':False,'backward_executed':False,'training_ready':False,
            'rows':reports,'tokenizer_files':token_files,'native_source_sha256':{f:source_hash(ROOT/'SFT/src/llamafactory'/f) for f in files},
            'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','datasets','accelerate','peft','trl')}}


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--rows');group.add_argument('--fixture',action='store_true')
    p.add_argument('--tokenizer-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--cutoff-len',type=int,default=32768)
    a=p.parse_args();out=Path(a.out_dir)
    if out.exists():raise FileExistsError(out)
    rows=[fixture_row()] if a.fixture else json.loads(Path(a.rows).read_text())
    report=audit_rows(rows,a.tokenizer_dir,cutoff_len=a.cutoff_len,fixture=a.fixture)
    from research.av_data.build_research_batch import write_json
    out.mkdir(parents=True,exist_ok=False);write_json(out/'report.private.json',report)
    if a.fixture:write_json(out/'preprocessing-fixture.private.json',rows)
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','tokenizer_files','native_source_sha256'}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

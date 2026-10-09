"""Read-only byte/visibility delivery audit; semantic and cost gaps stay explicit."""
from datetime import datetime, timezone
import hashlib,json,re,subprocess
from pathlib import Path
from research.av_search.corpus import load_frozen,visible_corpus,file_sha256
from research.av_data.audit import export_actor_task,audit_split_manifest

root=Path('/home/jovyan/OneSearch-VL');run=root/'research/runs/phase1-20261009';pilot=run/'natural-pilot-v1'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x]
_,manifest=load_frozen(pilot/'corpus');scope=read(pilot/'scope.json')
sources=rows(pilot/'sources.jsonl');tasks=rows(pilot/'tasks-public.jsonl')
public=visible_corpus(pilot/'corpus',scope)
assert len(sources)==len(tasks)==len(scope['source_ids'])==10
for source in sources:
 assert file_sha256(Path(source['asset_path']))==source['asset_sha256']
 assert Path(source['asset_path']).stat().st_size==source['asset_bytes']
for task in tasks:export_actor_task(task,public)
audit_split_manifest(read(pilot/'candidate-components.json'),read(pilot/'split-manifest.json'))
summary=read(run/'natural-replay-v2/summary.json');checked=[];receipt_identity_count=0
for task in tasks:
 episode=run/'natural-replay-v2'/task['task_id']
 events=rows(episode/'events.jsonl');actor=rows(episode/'actor_receipts.jsonl')
 assert [e['receipt'] for e in events]==actor
 raw=json.dumps(actor)
 assert not any(x in raw for x in ('asset_path','output_path','/home/jovyan','private_gold','expected_decision'))
 for event in events:
  receipt=event['receipt'];receipt_identity_count+=receipt.get('manifest_sha256')==manifest['manifest_sha256']
  if receipt['tool']=='search_sources':
   for hit in receipt['results']:
    doc=public['sources'][hit['source_id']]
    assert doc['text'][hit['quote_start']:hit['quote_end']]==hit['excerpt']
    assert hit['text_sha256']==doc['text_sha256'] and hit['asset_sha256']==doc['asset_sha256']
  derived=event['private_derived_record']
  if derived:
   assert file_sha256(Path(derived['output_path']))==derived['output_sha256']
   assert receipt['status']=='media_ready_observer_unavailable'
   assert receipt['capabilities']['raw_av_observer_ready'] is False
   checked.append(derived['output_sha256'])
assert len(checked)==10 and summary['totals']['failures']==0
latest=run/'final-cli-replay-v2';new_receipts=rows(latest/'actor_receipts.jsonl');header=read(latest/'episode.json')
assert len(new_receipts)==2 and header['manifest_sha256']==manifest['manifest_sha256']
assert all(r['manifest_sha256']==manifest['manifest_sha256'] for r in new_receipts)
assert new_receipts[1]['actual_cost']['count_scope']=='prepared_clip_only'
assert new_receipts[1]['capabilities']['full_decode_work_accounting_ready'] is False
forms=root/'research/private/annotation/natural-pilot-v1';generation=read(forms/'generation-receipt.json')
for entry in generation['files']:assert file_sha256(forms/entry['path'])==entry['sha256']
blank=[read(p) for p in forms.glob('*/*.json')]
assert len(blank)==30 and all(x['annotation_status']=='blank_unreviewed' for x in blank)
assert generation['human_signatures']==generation['gold_records']==0
research_log=(run/'integrated-tests-delivery.log').read_text();base_log=(run/'final-baseline-retrieval.log').read_text()
assert re.search(r'Ran 84 tests',research_log) and '\nOK\n' in research_log
assert re.search(r'Ran 7 tests',base_log) and '\nOK\n' in base_log
files=[p for folder in ('research','validation/research','docs/research/annotation') for p in (root/folder).rglob('*')
       if p.is_file() and p.suffix in ('.py','.md') and not any(x in p.parts for x in ('runs','assets','cache','private','__pycache__'))]
files.extend(root/x for x in ('pixi.toml','pixi.lock','.gitignore','AGENTS.md'))
storage={x:subprocess.check_output(['du','-sb',str(root/x)],text=True).split()[0] for x in ('research/assets','research/runs','research/private')}
assert int(storage['research/assets'])<50*1024**3
result={'schema_version':'phase1-delivery-verification-0.1','verified_at':datetime.now(timezone.utc).isoformat(),
 'status':'infrastructure_verified_semantics_and_full_decode_cost_pending',
 'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
 'baseline_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'tests':{'research':84,'baseline_retrieval':7,'failures':0},
 'sources':10,'draft_tasks':10,'source_hash_pass':10,'derived_hash_pass':10,
 'historical_replay':{'version':'natural-replay-v2','tool_calls':40,'clip_count':10,'failures':0,
     'manifest_hash_receipts':receipt_identity_count,'episode_header_manifest_hash_count':0},
 'latest_abi_probe':{'version':'final-cli-replay-v2','calls':2,'manifest_hash_receipts':2,'episode_manifest_hash':True,
     'real_clip_seconds':.5,'count_scope':'prepared_clip_only'},
 'corpus_identity':{k:manifest[k] for k in ('corpus_id','corpus_hash','index_sha256','documents_sha256','manifest_sha256')},
 'source_bytes':sum(x['asset_bytes'] for x in sources),'source_seconds':sum(x['duration_s'] for x in sources),
 'blank_review_forms':30,'human_signatures':0,'gold_records':0,'raw_av_observer_ready':False,
 'full_decode_work_accounting_ready':False,'G0':'pending','teacher_episodes':0,'method_experiments':0,
 'storage_bytes':storage,
 'public_actions':{'commit':False,'push':False,'PR':False,'publication':False},
 'code_snapshots':[{'path':str(p.relative_to(root)),'sha256':file_sha256(p)} for p in sorted(set(files))]}
(run/'delivery-verification-v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='code_snapshots'},ensure_ascii=False,indent=2))

"""Collect completed retained episodes, including after a bounded teacher timeout."""
import argparse,json
from pathlib import Path
from research.av_data.build_research_batch import write_json
from research.av_train.trajectory import digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes-dir',required=True);p.add_argument('--batch-dir',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args()
    root=Path(a.episodes_dir);refs={x['task_id']:x for x in map(json.loads,(Path(a.batch_dir)/'answer-candidates.private.jsonl').read_text().splitlines())}
    rows=[];notes=[];seen=set();completed=0;questions=set();recordings=set();invalid=[]
    for folder in sorted(root.glob('attempt-*')):
        path=folder/'retention.json'
        if not path.exists():continue
        try:
            result=json.loads(path.read_text());completed+=1
            if not result.get('retained'):continue
            row=json.loads((folder/'row.private.json').read_text());note=json.loads((folder/'note.private.json').read_text());key=digest(row)
            if result['row_sha256']!=key or note['row_sha256']!=key or note['quality_label']!='silver_anonymous_publisher_fields' or note['authored_assistant_targets']!=0:raise ValueError('Retained row binding changed')
            if key in seen:continue
            seen.add(key);rows.append(row);notes.append(note);questions.add(note['task_id']);recordings.add(refs[note['task_id']]['target_source_id'])
        except (ValueError,KeyError,OSError) as exc:invalid.append({'attempt':folder.name,'error_type':type(exc).__name__})
    if invalid:raise ValueError('Completed episode artifacts are incomplete or changed: '+str(invalid))
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);write_json(out/'rows.private.json',rows);write_json(out/'notes.private.json',notes)
    ready=len(rows)>=6 and len(questions)>=6 and len(recordings)>=3
    summary={'completed_attempts':completed,'retained_rows':len(rows),'retained_unique_questions':len(questions),'retained_recordings':len(recordings),'bounded_training_ready':ready,'semantic_verified':False,'benchmark_gold':False,'optimizer_updates':0}
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    if not ready:raise SystemExit('Need at least six retained questions across three recordings before paired training')
if __name__=='__main__':main()

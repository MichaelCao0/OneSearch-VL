"""Common-question analysis of corrected native AV and muted evaluation."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import numpy as np

METRICS = ('answer_success', 'supported_success')
MODES = ('native_av', 'muted_video')

def common_reviews(members, records, variants):
    index = {c['task_id']: c for c in members}
    if len(index) != len(members):
        raise ValueError('Duplicate planned question')
    keys = {v + '__' + m for v in variants for m in MODES}
    seen, valid, unknown = set(), {}, []
    for record in records:
        task = record['task_id']
        if task in seen or task not in index:
            raise ValueError('Duplicate or unplanned review')
        seen.add(task)
        if record.get('status') != 'completed':
            unknown.append(task)
            continue
        ratings = record['ratings']
        by_key = {x['variant_modality']: x for x in ratings}
        if len(by_key) != len(ratings) or set(by_key) != keys:
            raise ValueError('Incomplete or duplicate variant ratings')
        for rating in ratings:
            if any(type(rating.get(m)) is not bool for m in METRICS):
                raise ValueError('Boolean scores required')
            if rating['supported_success'] and not rating['answer_success']:
                raise ValueError('Supported success requires answer success')
        if not index[task].get('family_id'):
            raise ValueError('Recording family is required')
        valid[task] = {'case': index[task], 'ratings': by_key}
    return valid, sorted(unknown + list(set(index) - seen))

def paired(values, families, *, draws=10000, seed=42):
    if not values:
        return {'n': 0, 'delta': None, 'bootstrap95': None, 'wins': 0, 'losses': 0, 'ties': 0}
    if len(values) != len(families):
        raise ValueError('Each question needs one family')
    groups = {}
    for value, family in zip(values, families):
        groups.setdefault(family, []).append(value)
    grouped = list(groups.values())
    sums = np.array([sum(v) for v in grouped], dtype=float)
    sizes = np.array([len(v) for v in grouped], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    bootstrap = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    wins, losses = sum(v > 0 for v in values), sum(v < 0 for v in values)
    discordant = wins + losses
    p_value = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
    return {'n': len(values), 'family_groups': len(grouped), 'delta': sum(values) / len(values),
            'bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'wins': wins, 'losses': losses, 'ties': len(values) - discordant,
            'paired_binomial_p_two_sided': p_value,
            'bootstrap_unit': 'automatic recording family', 'draws': draws, 'seed': seed}

def summarize(valid, planned, variants, *, draws=10000):
    keys = [v + '__' + m for v in variants for m in MODES]
    comparisons = [(v + '__native_av', v + '__muted_video') for v in variants]
    for kind in ('trajectory', 'answer'):
        if kind + '2000' in variants and kind + '500' in variants:
            comparisons.extend((kind + '2000__' + m, kind + '500__' + m) for m in MODES)
    for size in ('500', '2000'):
        if 'trajectory' + size in variants and 'answer' + size in variants:
            comparisons.extend(('trajectory' + size + '__' + m, 'answer' + size + '__' + m) for m in MODES)
    report = {}
    for split in ('heldout_silver', 'dev16_publisher_candidates', 'all'):
        ids = sorted(t for t, v in valid.items() if split == 'all' or v['case']['split'] == split)
        count = sum(split == 'all' or c['split'] == split for c in planned)
        unknown = count - len(ids)
        table, contrasts = {}, {}
        for key in keys:
            table[key] = {'n': len(ids), 'planned': count, 'unknown': unknown}
            for metric in METRICS:
                good = sum(valid[t]['ratings'][key][metric] for t in ids)
                table[key][metric] = {'count': good, 'rate': good / len(ids) if ids else None,
                    'all_planned_bounds': [good/count, (good+unknown)/count] if count else None}
        for left, right in comparisons:
            contrast = {}
            for metric in METRICS:
                values = [int(valid[t]['ratings'][left][metric]) - int(valid[t]['ratings'][right][metric]) for t in ids]
                result = paired(values, [valid[t]['case']['family_id'] for t in ids], draws=draws)
                result['left_win_task_ids'] = [t for t, d in zip(ids, values) if d > 0]
                result['right_win_task_ids'] = [t for t, d in zip(ids, values) if d < 0]
                contrast[metric] = result
            contrasts[left + ' minus ' + right] = contrast
        report[split] = {'table': table, 'contrasts': contrasts}
    return report

def audit_execution(suite, variants):
    suite = Path(suite)
    result = {'arms': {}, 'initial_visual_mismatches': [], 'new_model_calls': 0}
    for variant in variants:
        arm = suite / variant
        prefix = 'base' if variant == 'base' else 'trained'
        receipt = json.loads((arm / 'gpu-v1/eval' / (prefix + '__paired-execution-receipt.json')).read_text())
        if receipt['status'] != 'passed' or receipt['questions_per_modality'] != 56:
            raise ValueError('Incomplete GPU execution receipt')
        item = {'modes': {}, 'max_allocated_gib': receipt['max_allocated_gib']}
        initial = {}
        for mode in MODES:
            out = arm / 'gpu-v1/eval' / (prefix + '__' + mode)
            summary = json.loads((out / 'summary.json').read_text())
            if summary['completed'] != summary['planned'] or summary['planned'] != 56:
                raise ValueError('Incomplete GPU summary')
            count, audio_min, audio_max, generated = 0, None, 0, 0
            seen = set()
            for row in summary['results']:
                if row['status'] != 'ok' or row['task_id'] in seen:
                    raise ValueError('Execution error or duplicate task')
                seen.add(row['task_id'])
                trace = json.loads(Path(row['trace_path']).read_text())
                for i, turn in enumerate(trace['turns']):
                    m = turn['response']['raw_response']['metrics']
                    tokens = m['audio_token_count']
                    if not m['audio_token_binding_verified'] or tokens != m['expected_audio_token_count']:
                        raise ValueError('Audio feature/token binding failed')
                    if mode == 'native_av' and tokens <= 0 or mode == 'muted_video' and (tokens or m['audio_count']):
                        raise ValueError('Wrong native or muted audio inputs')
                    gpu = m['input_tensors_on_model']
                    if gpu['input_ids']['device'] != 'cuda:0':
                        raise ValueError('GPU tensor delivery missing')
                    if mode == 'native_av' and 'input_features' not in gpu:
                        raise ValueError('Native GPU audio features missing')
                    count += 1; generated += m['generated_tokens']
                    audio_min = tokens if audio_min is None else min(audio_min, tokens)
                    audio_max = max(audio_max, tokens)
                    if i == 0:
                        initial.setdefault(row['task_id'], {})[mode] = {k:m.get(k) for k in ['video_count','prepared_video_frames','prepared_video_shapes','actual_video_fps']}
            item['modes'][mode] = {'questions': len(seen), 'turns': count, 'audio_token_min': audio_min,
                'audio_token_max': audio_max, 'generated_tokens': generated, 'all_audio_bindings_verified': True}
        for task, modes in initial.items():
            if set(modes) != set(MODES) or modes['native_av'] != modes['muted_video']:
                result['initial_visual_mismatches'].append({'variant':variant,'task_id':task,'inputs':modes})
        receipts = list((arm / 'muted-eval-media').glob('*.json'))
        for path in receipts:
            data = json.loads(path.read_text())
            if not data['encoded_video_packets_identical'] or not data['no_audio_stream']:
                raise ValueError('Encoded video or muted stream contract failed')
        item['verified_muted_media_receipts'] = len(receipts)
        result['arms'][variant] = item
    return result

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();suite=Path(args.suite).resolve();output=Path(args.output).resolve()
    if output.exists():raise ValueError('Preserve previous analysis output')
    judge=suite/'judge-all';plan=json.loads((judge/'plan.json').read_text());members=json.loads((judge/'eval-members.private.json').read_text())
    paths=sorted((judge/'semantic-review-v1/records').glob('*.json'));records=[json.loads(p.read_text()) for p in paths]
    valid, unknown=common_reviews(members, records, plan['variants'])
    if not (judge/'learning-results.private.json').is_file():raise ValueError('Joint blinded review not finished')
    audit=audit_execution(suite, plan['variants'])
    report={'scope':'corrected 56-question fixed-corpus automatic silver evaluation; joint blinded reviews',
        'unknown_task_ids':unknown,'common_completed_questions':len(valid),'human_gold':False,
        'bootstrap_note':'paired question differences; 10000 recording-family resamples; NumPy seed 42; training seed variation not measured',
        'metrics':summarize(valid,members,plan['variants']),'execution_audit':audit,
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[judge/'plan.json',judge/'eval-members.private.json',Path(__file__)]}}
    output.mkdir();(output/'paired-results.private.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10))
    case_ids=set()
    for entry in report['metrics']['all']['contrasts'].values():
        for metric in METRICS:
            case_ids.update(entry[metric]['left_win_task_ids']);case_ids.update(entry[metric]['right_win_task_ids'])
    (output/'flip-reviews.private.json').write_text(json.dumps([dict(task_id=t,**valid[t]) for t in sorted(case_ids)],ensure_ascii=False,indent=2)+chr(10))
    lines=['# 修复后的原声／静音与 500／2K 对照','',f'联合盲评有效题 {len(valid)}/56；unknown {len(unknown)}；自动 silver 评分。',
        '','| split | model/input | 答案成功 | 支持充分 |','|---|---|---:|---:|']
    for split, entry in report['metrics'].items():
        for key, v in entry['table'].items():lines.append(f"| {split} | {key} | {v['answer_success']['count']}/{v['n']} | {v['supported_success']['count']}/{v['n']} |")
    lines+=['','配对差值及录制家族 bootstrap 95% 区间：','','| split | comparison | metric | delta pp | 95% CI pp | win/loss |','|---|---|---|---:|---:|---:|']
    for split, entry in report['metrics'].items():
        for name, contrast in entry['contrasts'].items():
            for metric, v in contrast.items():
                if v['delta'] is not None:
                    lo,hi=v['bootstrap95'];lines.append(f"| {split} | {name} | {metric} | {100*v['delta']:.1f} | [{100*lo:.1f}, {100*hi:.1f}] | {v['wins']}/{v['losses']} |")
    (output/'重评结果与配对分析-中文.md').write_text(chr(10).join(lines)+chr(10))
    print(json.dumps({'output':str(output),'completed_questions':len(valid),'unknown':unknown,'initial_visual_mismatches':audit['initial_visual_mismatches']},ensure_ascii=False))

if __name__=='__main__':main()

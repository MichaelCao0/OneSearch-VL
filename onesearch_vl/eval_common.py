#!/usr/bin/env python3
"""Shared gold matching, judge transport, scoring, and reports for evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import time

import requests

from eval_prompts import JUDGE_PROMPT, OFFICIAL_SYSTEM_MESSAGE, extract_final_answer, official_judge_prompt


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    max_retries: int = 3
    retry_sleep: float = 1.0

    @classmethod
    def from_env(cls):
        config = cls(
            base_url=os.getenv('JUDGE_BASE_URL', 'https://api.openai.com/v1').rstrip('/'),
            api_key=os.getenv('JUDGE_API_KEY', ''),
            model=os.getenv('JUDGE_MODEL', 'gpt-4o'),
            timeout=float(os.getenv('JUDGE_TIMEOUT', '120')),
            max_retries=int(os.getenv('JUDGE_MAX_RETRY', '3')),
            retry_sleep=float(os.getenv('JUDGE_RETRY_SLEEP', '1')),
        )
        if not config.api_key.strip():
            raise ValueError('JUDGE_API_KEY is not set.')
        if not config.base_url or not config.model.strip():
            raise ValueError('JUDGE_BASE_URL and JUDGE_MODEL must be nonempty.')
        if not math.isfinite(config.timeout) or config.timeout <= 0 or config.max_retries < 1:
            raise ValueError('Judge timeout and retry count must be positive.')
        if not math.isfinite(config.retry_sleep) or config.retry_sleep < 0:
            raise ValueError('Judge retry sleep must be finite and nonnegative.')
        return config


def build_messages(protocol: str, question: str, answer: str, response: str) -> list[dict]:
    if protocol == 'onesearch':
        return [{'role': 'user', 'content': JUDGE_PROMPT.format(
            question=question, correct_answer=answer, response=response[:8000])}]
    if protocol == 'videodr':
        return [{'role': 'system', 'content': OFFICIAL_SYSTEM_MESSAGE},
                {'role': 'user', 'content': official_judge_prompt(question, answer, response[:8000])}]
    raise ValueError(f'Unknown evaluation protocol: {protocol}')


def call_judge(messages: list[dict], config: JudgeConfig, request_fn=None) -> str:
    if not config.api_key.strip():
        raise ValueError('JUDGE_API_KEY is not set.')
    request_fn = request_fn or requests.post
    url = config.base_url.rstrip('/')
    if not url.endswith('/chat/completions'):
        url += '/chat/completions'
    for attempt in range(config.max_retries):
        try:
            response = request_fn(url, headers={'Content-Type': 'application/json',
                                  'Authorization': f'Bearer {config.api_key}'},
                                  json={'model': config.model, 'messages': messages,
                                        'temperature': 0.0, 'max_tokens': 1024}, timeout=config.timeout)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Empty judge response')
            return content
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            if attempt + 1 < config.max_retries:
                time.sleep(config.retry_sleep)
    # Do not put response bodies, headers or endpoint credentials in reports.
    raise RuntimeError(f'Judge request failed after {config.max_retries} attempts')


def parse_verdict(raw: str, protocol: str) -> dict:
    cleaned = re.sub(r'\*\*|__', '', raw or '')
    if protocol == 'onesearch':
        verdicts = re.findall(r'(?<!\w)correct\s*[:：]\s*\[?\s*(yes|no)\s*\]?(?!\w)', cleaned, re.I)
        if not verdicts or len({v.lower() for v in verdicts}) != 1:
            raise ValueError('Missing or conflicting correct: yes/no verdict')
        reason = re.search(r'reasoning\s*[:：]\s*(.+)', cleaned, re.I | re.S)
        return {'acc': int(verdicts[0].lower() == 'yes'),
                'reasoning': reason.group(1).strip() if reason else '', 'raw_judge': raw}
    if protocol != 'videodr':
        raise ValueError(f'Unknown evaluation protocol: {protocol}')
    block = re.search(r'\{.*\}', raw or '', re.S)
    if block:
        try:
            obj = json.loads(block.group())
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            value = obj.get('is_correct')
            if isinstance(value, str) and value.lower() in {'true', 'false', 'yes', 'no'}:
                value = value.lower() in {'true', 'yes'}
            if isinstance(value, bool):
                return {'acc': int(value), 'reasoning': str(obj.get('reasoning', '')), 'raw_judge': raw}
            raise ValueError('VideoDR is_correct must be an explicit boolean')
        # Recover an explicit field from malformed JSON (e.g. unescaped LaTeX).
        verdicts = re.findall(r'''["']?is_correct["']?\s*:\s*["']?(true|false)["']?(?=\s*[,}])''', block.group(), re.I)
    else:
        verdicts = re.findall(r'(?<!\w)is[_ ]correct\s*[:：]?\s*(true|false|yes|no)(?!\w)', cleaned, re.I)
    if not verdicts or len({v.lower() for v in verdicts}) != 1:
        raise ValueError('Missing or conflicting VideoDR verdict')
    return {'acc': int(verdicts[0].lower() in {'true', 'yes'}),
            'reasoning': 'recovered explicit is_correct verdict', 'raw_judge': raw}


def _text(value) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return ''
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return ', '.join(_text(v) for v in value)
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return _text(decoded)
        except ValueError:
            pass
    return str(value).strip()


def load_gold(path: Path) -> dict:
    import pandas as pd
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
    else:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(path).schema_arrow.names
        wanted = {'id', 'case_id', 'data_id', 'index', 'question', 'query', 'prompt',
                  'answer', 'answers', 'gold', 'ground_truth'}
        # Never materialize the benchmark's embedded image/video columns.
        df = pq.read_table(path, columns=[n for n in names if n.lower().strip() in wanted]).to_pandas()
    columns = {str(c).lower().strip(): c for c in df.columns}
    def choose(names):
        return next((columns[n] for n in names if n in columns), None)
    id_col = choose(('index', 'id', 'case_id', 'data_id')) if path.suffix.lower() == '.csv' else choose(('id', 'case_id', 'data_id', 'index'))
    q_col = choose(('question', 'query', 'prompt'))
    a_col = choose(('answer', 'answers', 'gold', 'ground_truth'))
    if q_col is None or a_col is None:
        raise ValueError('Gold data must contain question and answer columns')
    by_id, by_idx = {}, {}
    for row_idx, (_, row) in enumerate(df.iterrows()):
        case_id = _text(row[id_col]) if id_col is not None else ''
        record = {'question': _text(row[q_col]), 'answer': _text(row[a_col]),
                  'gold_row': row_idx, 'gold_id': case_id}
        if case_id:
            if case_id in by_id:
                raise ValueError(f'Duplicate gold ID: {case_id}')
            by_id[case_id] = record
        lookup_idx = row_idx
        # CSV inference uses the explicit numeric index, including sparse indices.
        if path.suffix.lower() == '.csv' and case_id.isdigit():
            lookup_idx = int(case_id)
        if lookup_idx in by_idx:
            raise ValueError('Ambiguous gold row indices')
        by_idx[lookup_idx] = record
    return {'by_id': by_id, 'by_idx': by_idx, 'size': len(df)}


def resolve_gold(traj: dict, index: dict):
    case_id = str(traj.get('case_id', ''))
    if case_id in index['by_id']:
        return index['by_id'][case_id], 'id'
    case_idx = traj.get('case_idx')
    match = re.fullmatch(r'case_(\d+)', case_id)
    if match:
        case_idx = int(match.group(1))
    if isinstance(case_idx, int) and not isinstance(case_idx, bool) and case_idx >= 0:
        rec = index['by_idx'].get(case_idx)
        if rec is not None:
            return rec, 'index'
    return None, None


def prepare_case(path: Path, index: dict) -> dict:
    record = {'trajectory_file': path.name, 'case_id': None, 'case_idx': None,
              'status': 'input_error', 'acc': None, 'raw_judge': ''}
    try:
        traj = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(traj, dict):
            raise ValueError('Trajectory must be a JSON object')
        record.update(case_id=traj.get('case_id'), case_idx=traj.get('case_idx'),
                      trajectory_status=traj.get('status'))
        text = traj.get('final_response_text') or ''
        if not isinstance(text, str):
            raise ValueError('final_response_text must be a string')
        record['model_response'] = extract_final_answer(text)[:8000]
        gold, matched_by = resolve_gold(traj, index)
        if gold is None or not gold['question'] or not gold['answer']:
            record.update(status='missing_gold', reasoning='No matching nonempty question/answer')
        else:
            record.update(question=gold['question'], correct_answer=gold['answer'],
                          gold_row=gold['gold_row'], gold_id=gold['gold_id'], matched_by=matched_by,
                          status='pending')
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        record['reasoning'] = f'Invalid trajectory: {type(exc).__name__}'
    return record


def evaluate_case(record: dict, protocol: str, config: JudgeConfig, judge_fn=None) -> dict:
    if record['status'] != 'pending':
        return record
    if not record['model_response']:
        return dict(record, status='empty_response', acc=0, reasoning='Empty model response')
    record = dict(record)
    messages = build_messages(protocol, record['question'], record['correct_answer'], record['model_response'])
    try:
        raw = (judge_fn or call_judge)(messages, config)
        record['raw_judge'] = raw
        parsed = parse_verdict(raw, protocol)
        record.update(parsed, status='judged')
    except (RuntimeError, ValueError, requests.RequestException, TypeError, KeyError, IndexError) as exc:
        record.update(status='judge_failed', acc=None, reasoning=f'Judge failure: {type(exc).__name__}')
    return record


def run_evaluation(argv=None, judge_fn=None, *, modality: str) -> int:
    if modality not in {'image', 'video'}:
        raise ValueError(f'Unknown evaluation modality: {modality}')
    protocols = ('onesearch',) if modality == 'image' else ('onesearch', 'videodr')
    description = ('Evaluate single-image and multi-image trajectories.' if modality == 'image'
                   else 'Evaluate OneSearch-Video or VideoDR trajectories.')
    parser = argparse.ArgumentParser(description=description)
    default_protocol = 'onesearch' if modality == 'image' else 'videodr'
    parser.add_argument('--protocol', choices=protocols,
                        default=os.getenv('EVAL_PROTOCOL') or default_protocol,
                        help=f'Judge protocol (default: {default_protocol}; EVAL_PROTOCOL overrides it)')
    parser.add_argument('--traj_dir', required=True, type=Path)
    gold = parser.add_mutually_exclusive_group(required=True)
    gold.add_argument('--parquet', type=Path)
    gold.add_argument('--csv', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    if args.protocol not in protocols:
        parser.error(f'{modality} evaluation supports protocols: {", ".join(protocols)}')
    if args.max_workers < 1 or args.limit < 0:
        parser.error('--max_workers must be positive and --limit nonnegative')
    files = sorted(args.traj_dir.glob('*_trajectory.json'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error('No *_trajectory.json files found')
    try:
        config = JudgeConfig.from_env()
        gold_path = args.parquet or args.csv
        index = load_gold(gold_path)
        records = [prepare_case(path, index) for path in files]
        matched = [r['gold_row'] for r in records if 'gold_row' in r]
        if len(matched) != len(set(matched)):
            raise ValueError('Multiple trajectories match one gold row; supply one trajectory per case')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        results = list(pool.map(lambda r: evaluate_case(r, args.protocol, config, judge_fn), records))
    completed = sum(r['status'] in {'judged', 'empty_response'} for r in results)
    correct = sum(r['acc'] or 0 for r in results)
    total = len(results)
    complete = completed == total
    report = {
        'judge': config.model, 'modality': modality, 'protocol': args.protocol,
        'answer_extraction': 'boxed_first',
        'traj_dir': str(args.traj_dir), 'parquet': str(args.parquet) if args.parquet else None,
        'csv': str(args.csv) if args.csv else None, 'status': 'complete' if complete else 'incomplete',
        'total': total, 'completed': completed, 'correct': correct,
        'failed': total - completed, 'missing_gold': sum(r['status'] == 'missing_gold' for r in results),
        'empty_response': sum(r['status'] == 'empty_response' for r in results),
        'trajectory_non_ok': sum(r.get('trajectory_status') not in (None, 'ok') for r in results),
        'gold_rows': index['size'], 'matched_gold_rows': len(matched),
        'gold_rows_without_trajectory': index['size'] - len(matched), 'limit': args.limit,
        'accuracy': f'{100.0 * correct / total:.2f}%' if complete else None,
    }
    output = args.output or args.traj_dir / 'eval_report.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    details = output.with_suffix('.details.jsonl')
    details.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Report: {output}; details: {details}')
    return 0 if complete else 1

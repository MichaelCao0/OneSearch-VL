"""Offline source receipt audit; original generation/evaluation artifacts stay frozen."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from urllib.parse import unquote
from .native_sft import file_sha
from .teacher_rollouts import atomic_json, now


def source_metrics(trace, reference_urls):
    """Match exact URL lines from executed tool output to frozen reference IDs."""
    canonical = {source_id: unquote(url) for source_id, url in reference_urls.items()}
    retrieved, read, matched = False, False, set()
    for turn in trace.get('turns', []):
        if turn.get('error') or not isinstance(turn.get('tool_output'), str):
            continue
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', turn.get('response_text', ''), re.S)
        if len(calls) != 1:
            continue
        try:
            name = json.loads(calls[0]).get('name')
        except (ValueError, AttributeError):
            continue
        urls = {unquote(url) for url in re.findall(r'^URL: (https?://[^\s]+)\s*$', turn['tool_output'], re.M)}
        hits = {source_id for source_id, url in canonical.items() if url in urls}
        if name in {'text_search', 'image_search'}:
            retrieved |= bool(hits); matched.update(hits)
        elif name == 'visit_webpage':
            read |= bool(hits); matched.update(hits)
    return {'target_retrieved': retrieved, 'target_read': read, 'matched_reference_source_ids': sorted(matched)}


def analyze(run_dir):
    r = Path(run_dir).resolve()
    cases = {case['task_id']: case for case in json.loads((r/'eval-members.private.json').read_text())}
    docs, doc_hashes, results, summaries = {}, {}, {}, {}
    for variant in ['base', 'trajectory', 'answer_only']:
        p = r/'gpu-v1/eval'/variant/'summary.json'
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        rows = []
        for row in raw['results']:
            case = cases[row['task_id']]; index = Path(case['index_dir'])/'documents.jsonl'
            if str(index) not in docs:
                docs[str(index)] = {d['source_id']: d['origin_url'] for d in (json.loads(line) for line in index.read_text().splitlines() if line)}
                doc_hashes[str(index)] = file_sha(index)
            targets = {case['reference']['target_source_id']} | {quote['source_id'] for quote in case['reference'].get('support_quotes', [])}
            missing = targets - docs[str(index)].keys()
            if missing:
                raise ValueError('Reference source absent from frozen documents: '+str(missing))
            trace = json.loads(Path(row['trace_path']).read_text())
            source = source_metrics(trace, {sid: docs[str(index)][sid] for sid in targets})
            rows.append({'task_id': row['task_id'], 'split': row['split'], 'protocol_final': row.get('protocol_final',False), **source})
        results[variant] = rows
        summaries[variant] = {'completed':raw['completed'], 'planned':raw['planned'], 'execution_errors':sum(row['status']=='error' for row in raw['results']),
            'protocol_finals':sum(row['protocol_final'] for row in rows), 'tool_using_cases':sum(bool(row.get('tool_calls')) for row in raw['results']),
            'all_fields_string_matches_raw':sum(row.get('all_fields_string_match',False) for row in raw['results']),
            'all_fields_string_matches_final_only':sum(row.get('all_fields_string_match',False) and row.get('protocol_final',False) for row in raw['results']),
            'target_retrieved':sum(row['target_retrieved'] for row in rows), 'target_read':sum(row['target_read'] for row in rows)}
    report = {'at_utc':now(), 'schema':'onesearch-av-offline-source-audit.v1', 'generation_unchanged':True, 'metric':'any target or support source URL returned by an executed search or visit',
        'raw_source_id_substring_metrics_valid':False, 'partial':not all(v in summaries and summaries[v]['completed']==summaries[v]['planned'] for v in ['base','trajectory','answer_only']),
        'reference_documents_sha256':doc_hashes, 'variants':summaries, 'cases':results}
    atomic_json(r/'source-metrics-corrected.private.json', report)
    lines=['# OneSearch-AV 实时评测与来源命中核验', '', '更新时间：'+report['at_utc'], '',
        '所有数据来自已保存的执行轨迹；不改生成、推理设置或冻结评测文件。当前尚未完成匿名语义评分，字段字符串匹配不等于准确率。', '',
        '| 模型 | 完成 | 执行错误 | 正式答案 | 使用工具的题 | 返回目标或支持来源 | 读取目标或支持来源 | 最终答案字段匹配 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v, x in summaries.items():
        lines.append(f"| {v} | {x['completed']}/{x['planned']} | {x['execution_errors']} | {x['protocol_finals']} | {x['tool_using_cases']} | {x['target_retrieved']} | {x['target_read']} | {x['all_fields_string_matches_final_only']} |")
    lines.extend(['', '来源命中按工具返回的独立 URL 行与冻结 corpus 的 reference ID→URL 映射核验；actor 自己提到 URL、excerpt 内提到 URL、相似 URL 前缀和失败工具调用均不计。旧按内部 source_id 字符串匹配得到的命中率无效。', '',
        '训练 172 题，两组 SFT 各 86 optimizer updates；评测 40 source-family-heldout silver + 16 dev candidates。数据池目前 230 条自动保留且通过 CPU 检查的独立题，数据队列暂停。'])
    if (r/'pilot-results.private.json').exists():
        semantic=json.loads((r/'pilot-results.private.json').read_text())
        lines.extend(['', '## 匿名语义评分', '', '完成 '+str(semantic['completed_reviews'])+'/'+str(semantic['planned_questions'])+' 个问题；参考和 judge 均为 silver。'])
        for split, table in semantic['metrics'].items():
            lines.extend(['', split, '', '| 模型 | 正确题数 | 语义正确率 | 返回目标来源 | 读取目标来源 |','|---|---:|---:|---:|---:|'])
            valid_ids={json.loads(p.read_text())['task_id'] for p in (r/'semantic-review-v1/records').glob('*.json') if json.loads(p.read_text()).get('status')=='completed'}
            for v, metric in table.items():
                subset=[row for row in results[v] if row['task_id'] in valid_ids and (split=='all' or row['split']==split)]
                n=len(subset);found=sum(row['target_retrieved'] for row in subset);read=sum(row['target_read'] for row in subset)
                pct=lambda x: '未评分' if x is None else f'{x:.1%}'
                lines.append(f"| {v} | {metric['semantic_correct']}/{metric['n']} | {pct(metric['semantic_accuracy'])} | {pct(found/n if n else None)} | {pct(read/n if n else None)} |")
            for comparison, value in semantic['paired_deltas'][split].items():
                interval=value['paired_bootstrap95']
                lines.append(f"- {comparison}：{value['accuracy_delta']*100:.1f} 个百分点，录制家族 bootstrap 95% 区间 [{interval[0]*100:.1f}, {interval[1]*100:.1f}]。")
    (r/'Evaluation-Progress-and-Source-Audit-中文.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
    value=analyze(a.run_dir);print(json.dumps({'at_utc':value['at_utc'],'partial':value['partial'],'variants':value['variants']},ensure_ascii=False))

"""Exploration diagnostics against pending publisher QA candidates, not gold QA.

Free-form accuracy needs a separately reviewed semantic answer verdict. Exact
field substrings do not validate negation, contradictions or all assertions.
"""
import re
import unicodedata

def field_present(value,answer):
    value=unicodedata.normalize("NFKC",value).casefold()
    answer=unicodedata.normalize("NFKC",answer).casefold()
    parts=re.findall(r"[^\W_]+",value,flags=re.UNICODE)
    return bool(parts and re.search(r"(?<!\w)"+r"[\W_]*".join(re.escape(p) for p in parts)+r"(?!\w)",answer,re.UNICODE))

def score_candidate_diagnostic(trace,reference):
    if trace['task']['task_id']!=reference['task_id']:raise ValueError('Trace/reference task mismatch')
    if reference.get('status')!='pending_human_review' or reference.get('benchmark_gold') is not False:raise ValueError('Pending candidate diagnostic only')
    fields=reference['answer_fields']
    if not fields:raise ValueError('Require requested fields')
    final=trace['final'];answer=final.get('answer','');valid=final.get('status')=='ok' and isinstance(answer,str)
    matched={f['field_id']:bool(valid and any(field_present(v,answer) for v in f['accepted_values'])) for f in fields}
    steps=trace['steps'];target=reference['target_source_id'];budget=trace['task']['budget_recipe'];cost=final.get('cost',{})
    return {'task_id':reference['task_id'],'status':final.get('status'),'answer_accuracy':None,'semantic_verified':False,
      'metric_scope':'pending publisher candidate diagnostics','requested_fields':matched,
      'requested_field_match_fraction':sum(matched.values())/len(matched),'all_requested_fields_match':all(matched.values()),
      'target_retrieved':any(s['action']['tool']=='search_sources' and s['observation'].get('status')=='ok' and any(r['source_id']==target for r in s['observation'].get('results',[])) for s in steps),
      'target_read':any(s['action']['tool']=='read_text' and s['observation'].get('status')=='ok' and s['observation'].get('source_id')==target for s in steps),
      'tool_attempts_in_trace':len(steps),'successful_tool_receipts':sum(s['observation'].get('status')=='ok' for s in steps),
      'tool_error_receipts':sum(s['observation'].get('status')=='tool_error' for s in steps),
      'media_preparation_receipts':sum(s['observation'].get('status')=='media_ready_observer_unavailable' for s in steps),
      'semantic_observer_receipts':sum(s['action']['tool']=='inspect_av' and s['observation'].get('status')=='ok' and s['observation'].get('model_executed') is True for s in steps),
      'budget_compliant':cost.get('tool_calls',float('inf'))<=budget['max_tool_calls'] and cost.get('requested_av_seconds',float('inf'))<=budget['max_requested_av_seconds'],
      'cost':cost,'citation_correctness':'not_evaluated'}

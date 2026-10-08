"""Side-channel tool statistics. Tool results/retries/prompts are unchanged.

Only this process's onesearch module is decorated. SearchControl carries the
collector across existing executor/page threads; concurrent rollouts never use
process-wide counter deltas. Cached results have unknown provenance, not newly
successful fetches/summaries. No URLs, credentials, or page text are recorded.
"""
from __future__ import annotations

import contextvars
import functools
import copy
import hashlib
import json
import os
import re
import inspect
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from vision_deepresearch_async_workflow.tool_quality import quality_signals, response_semantics, execution_category

_CURRENT = contextvars.ContextVar('rllm_tool_statistics', default=None)
_STAGE = contextvars.ContextVar('rllm_tool_stage', default='other')
_SUMMARY_RESULT = contextvars.ContextVar('rllm_summary_generated', default=None)
_PAGE = contextvars.ContextVar('rllm_tool_page', default=None)
_INSTALL_LOCK = threading.Lock()


class CallStatistics:
    def __init__(self, rollout_id=None, step_index=None):
        self.rollout_id = str(rollout_id) if rollout_id is not None else uuid.uuid4().hex
        self.step_index = step_index
        self.cache_identity = None
        self.cache_provenance = None
        self.pages = {}
        self.operations = {}
        self.call_id = uuid.uuid4().hex
        self.counts = Counter()
        self.requests = []
        self.lock = threading.Lock()
        self.closed = False

    def add(self, **values):
        with self.lock:
            if not self.closed:
                self.counts.update(values)

    def begin_request(self, route, operation_key=None):
        with self.lock:
            if self.closed:
                return None
            index = len(self.requests)
            page = _PAGE.get()
            key = (route, page, operation_key or uuid.uuid4().hex)
            operation = self.operations.setdefault(key, dict(
                operation_id=f'{self.call_id}:op{len(self.operations)+1}', attempts=0))
            operation['attempts'] += 1
            attempt = operation['attempts']
            self.counts[f'http_{route}_attempts'] += 1
            self.counts[f'http_{route}_retries'] += int(attempt > 1)
            self.requests.append(dict(request_id=f'{self.call_id}:{index+1}', route=route,
                status=None, success=None, pending=True, started_epoch=time.time(),
                rollout_id=self.rollout_id, step_index=self.step_index, call_id=self.call_id,
                page_id=f'{self.call_id}:page{page}' if page is not None else None,
                operation_id=operation['operation_id'], attempt=attempt, is_retry=attempt > 1))
            return index

    def request(self, index, route, response, elapsed, exception=None):
        status = getattr(response, 'status_code', None)
        ok = isinstance(status, int) and 200 <= status < 300
        headers = getattr(response, 'headers', {}) or {}
        proxy_failure = headers.get('X-Tool-Proxy-Failure') == '1'
        semantic = response_semantics(route, response) if ok else False
        with self.lock:
            if self.closed or index is None:
                return
            self.counts[f'http_{route}_completed'] += 1
            self.counts[f'http_{route}_success'] += int(ok)
            self.counts[f'http_{route}_failure'] += int(not ok)
            self.counts[f'http_{route}_proxy_failure'] += int(proxy_failure)
            self.requests[index].update(dict(
                status=status, success=ok, semantic_success=semantic, proxy_failure=proxy_failure,
                pending=False, completed_epoch=time.time(),
                proxy_request_id=headers.get('X-Search-Proxy-Request-Id') or headers.get('X-Job-API-Request-Id'),
                reader_source=headers.get('X-Tool-Reader-Source'),
                reader_owner_request_id=headers.get('X-Tool-Reader-Owner'),
                elapsed_s=round(elapsed,6), exception=type(exception).__name__ if exception else None,
            ))

    def update_page(self, rank, **values):
        if rank is None:
            return
        with self.lock:
            if not self.closed:
                page = self.pages.setdefault(rank, dict(rank=rank, page_id=f'{self.call_id}:page{rank}'))
                page.update(values)

    def record_page(self, page):
        self.update_page(page.get('rank'), fetch_status=page.get('fetch_status', 'unknown'),
            returned_nonempty=bool(page.get('qwen_summary') or page.get('snippet')),
            page_completed=True, body_fetch_success=(page.get('fetch_status') == 'fetched')
                if page.get('fetch_status') in {'fetched', 'fetched_empty', 'error', 'snippet_only'} else None)

    def trace_headers(self, index):
        if index is None:
            return {}
        request = self.requests[index]
        fields = {'Rollout-Id': self.rollout_id, 'Step-Index': self.step_index,
            'Call-Id': self.call_id, 'Request-Id': request['request_id'],
            'Operation-Id': request['operation_id'], 'Attempt': request['attempt'],
            'Page-Id': request['page_id']}
        # Side-channel identifiers only. Never include URLs, text or credentials.
        return {'X-Tool-' + key: str(value) for key, value in fields.items()
            if value is not None and re.fullmatch(r'[A-Za-z0-9_.:@/+\-]{1,200}', str(value))}

    def finish(self, tool, observation, step_error, timing):
        if timing.get("singleflight_wait") and self.cache_identity:
            from vision_deepresearch_async_workflow.cache_provenance import read
            self.cache_provenance = read(self.cache_identity, observation)
        with self.lock:
            self.closed = True
            counts = dict(self.counts)
            requests = copy.deepcopy(self.requests)
            pages = copy.deepcopy(list(self.pages.values()))
        if pages:
            for key in ('pages_recorded', 'pages_fetched', 'pages_fetch_empty', 'pages_fetch_error',
                        'pages_snippet_only', 'pages_status_unknown', 'pages_degraded',
                        'pages_full_quality', 'pages_quality_known', 'pages_incomplete',
                        'body_fetch_observed_pages', 'body_fetch_success_pages'):
                counts[key] = 0
            for page in pages:
                status = page.get('fetch_status', 'unknown')
                generated = page.get('summary_generated')
                reasons = []
                if status in {'fetched_empty', 'error', 'snippet_only'}:
                    reasons.append('body_unavailable')
                if page.get('summary_fallback'):
                    reasons.append('summary_fallback')
                page['degradation_reasons'] = reasons
                complete = page.get('page_completed', False)
                page['quality_known'] = bool(complete and (reasons or (status == 'fetched' and generated is not None)))
                page['full_quality'] = status == 'fetched' and generated is True and complete
                page['degraded'] = bool(reasons)
                counts['body_fetch_observed_pages'] += int(page.get('body_fetch_success') is not None)
                counts['body_fetch_success_pages'] += int(page.get('body_fetch_success') is True)
                counts['pages_recorded'] += int(complete)
                counts['pages_incomplete'] += int(not complete)
                counts['pages_fetched'] += int(status == 'fetched')
                counts['pages_fetch_empty'] += int(status == 'fetched_empty')
                counts['pages_fetch_error'] += int(status == 'error')
                counts['pages_snippet_only'] += int(status in {'snippet_only', 'snippet_only_mode', 'snippet_summary_mode'})
                counts['pages_status_unknown'] += int(status == 'unknown')
                counts['pages_degraded'] += int(page['degraded'] and complete)
                counts['pages_quality_known'] += int(page['quality_known'])
                counts['pages_full_quality'] += int(page['full_quality'])
        if counts.get('summary_started'):
            counts['summary_pending_at_return'] = counts['summary_started'] - counts.get('summary_calls', 0) - counts.get('summary_aborted', 0)
        counts['http_pending_at_return'] = sum(r['pending'] for r in requests)
        explicit = bool(step_error) or str(observation).startswith((
            'Error:', 'Tool execution error:', 'Search error', 'image_search error',
            'Error calling tool ', '[Json Parse Error]', '[Python Interpreter Error]',
        )) or ('is not callable' in str(observation)) or ('arguments must be a JSON object' in str(observation))
        if not explicit:
            outcome = 'success'
        elif 'Unknown tool:' in str(observation) or 'not available in this modality' in str(observation):
            outcome = 'unavailable_tool'
        elif 'empty results' in str(observation):
            outcome = 'empty_search_results'
        elif 'timed out' in str(observation) or 'deadline' in str(observation):
            outcome = 'timeout'
        else:
            outcome = 'execution_error'
        quality_known = bool(pages) and all(p['quality_known'] for p in pages)
        degraded = bool(counts.get('pages_degraded') or counts.get('summary_fallbacks'))
        degradation_known = degraded or bool(pages and all(p.get('page_completed') for p in pages))
        cached_quality = None
        if self.cache_provenance and not pages:
            source_pages = self.cache_provenance.get('pages', [])
            source_bad = any(p.get('summary_fallback') or p.get('fetch_status') in {'fetched_empty','error','snippet_only'} for p in source_pages)
            source_good = bool(source_pages) and all(p.get('body_fetch_success') is True and p.get('summary_generated') is True and p.get('page_completed') for p in source_pages)
            if source_bad or source_good:
                degradation_known = True
                degraded = source_bad
                cached_quality = not explicit and source_good
        return dict(version=3, failure_category=execution_category(observation, explicit),
            judge_decision_valid=getattr(self,"judge_decision_valid",None),
            cache_provenance=self.cache_provenance, call_id=self.call_id, rollout_id=self.rollout_id,
            step_index=self.step_index, tool=tool, success=not explicit,
            degraded_return=(degraded if degradation_known else None) if not explicit else False,
            full_quality_success=(not explicit and all(p['full_quality'] for p in pages)) if quality_known else cached_quality,
            pages=pages,
            workflow_error=bool(step_error),
            outcome=outcome, counts=counts, requests=requests,
            cache_hit=bool(timing.get('cache_hit')), singleflight_wait=bool(timing.get('singleflight_wait')),
            page_quality_observed=bool(counts.get('pages_recorded')),
            quality_provenance='observed_pages' if counts.get('pages_recorded') else 'cache_recorded' if self.cache_provenance else 'unobserved',
            # This is the existing image/backend counter, not a claimed total
            # across Reader, summary and cached/subsequent calls.
            backend_http_attempts=timing.get('http_attempts'))


class _RequestsObserver:
    """Facade on the search module only; does not patch requests globally."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        target = getattr(self.original, name)
        if name not in {'get', 'post', 'request'}:
            return target

        @functools.wraps(target)
        def request(*args, **kwargs):
            collector = _CURRENT.get()
            if collector is None:
                return target(*args, **kwargs)
            started = time.perf_counter()
            # Only a digest is used in memory to identify repeated sends; the
            # payload/URL/digest are never exported. Different pages are separate.
            identity = json.dumps([name, args, {k: kwargs.get(k) for k in ('url', 'json', 'data', 'params')}],
                                  sort_keys=True, default=str)
            index = collector.begin_request(_STAGE.get(), hashlib.sha256(identity.encode()).hexdigest())
            kwargs['headers'] = dict(kwargs.get('headers') or {})
            kwargs['headers'].update(collector.trace_headers(index))
            try:
                result = target(*args, **kwargs)
            except BaseException as exc:
                collector.request(index, _STAGE.get(), None, time.perf_counter()-started, exc)
                raise
            collector.request(index, _STAGE.get(), result, time.perf_counter()-started)
            return result
        return request


def _wrap_helper(function, route=None, kind=None):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def observed(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        control = arguments.get('control')
        collector = getattr(control, '_rllm_statistics', None) or _CURRENT.get()
        if collector is None:
            return function(*args, **kwargs)
        token = _CURRENT.set(collector)
        stage_token = _STAGE.set(route or _STAGE.get())
        page_token = None
        if kind == 'summary_generation':
            collector.add(summary_started=1)
        summary_token = _SUMMARY_RESULT.set(None) if kind == 'summary_fallback' else None
        try:
            result = function(*args, **kwargs)
            if route == 'reader':
                nonempty = bool(result and str(result).strip())
                collector.add(reader_calls=1, reader_nonempty=int(nonempty), reader_empty=int(not nonempty))
                collector.update_page(_PAGE.get(), body_fetch_success=nonempty)
            if kind == 'summary_generation':
                generated = bool(result and str(result).strip())
                collector.add(summary_calls=1, summary_generated=int(generated), summary_generation_failed=int(not generated))
                _SUMMARY_RESULT.set(generated)
                collector.update_page(_PAGE.get(), summary_generated=generated)
            elif kind == 'summary_fallback':
                # The helper returns original content only after its LLM call
                # returns empty. A nonempty fallback is not an LLM success.
                generated = _SUMMARY_RESULT.get()
                if generated is not None:
                    collector.add(summary_fallbacks=int(not generated))
                    collector.update_page(_PAGE.get(), summary_fallback=not generated, summary_returned_nonempty=bool(result),
                        content_quality=quality_signals(str(arguments.get('content') or '')[:2000], result,
                            arguments.get('query',''),arguments.get('title',''),body_complete=True))
                    if _PAGE.get() is not None:
                        from vision_deepresearch_async_workflow.cache_provenance import write_review_evidence
                        saved=write_review_evidence(collector,_PAGE.get(),arguments.get('content'),result,arguments.get('query'),arguments.get('title'))
                        collector.add(quality_evidence_saved=int(saved),quality_evidence_write_failed=int(not saved))
            if function.__name__ == 'text_search' and collector.cache_identity:
                from vision_deepresearch_async_workflow.cache_provenance import publish, snapshot
                publish(collector.cache_identity, result, snapshot(collector))
            return result
        except BaseException:
            if kind == 'summary_generation':
                collector.add(summary_aborted=1)
                _SUMMARY_RESULT.set(False)
            raise
        finally:
            if page_token is not None:
                _PAGE.reset(page_token)
            if summary_token is not None:
                _SUMMARY_RESULT.reset(summary_token)
            _STAGE.reset(stage_token)
            _CURRENT.reset(token)
    return observed


def install_search_observers():
    from vision_deepresearch_async_workflow.tools.onesearch_engine import search
    with _INSTALL_LOCK:
        if getattr(search, '_rllm_statistics_installed', False):
            return
        original_control = search.SearchControl

        class ObservedSearchControl(original_control):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._rllm_statistics = _CURRENT.get()

        search.SearchControl = ObservedSearchControl
        search.requests = _RequestsObserver(search.requests)
        helpers = {
            'text_search': (None, None),
            'search_web': ('text', None),
            'visual_lookup': ('image', None),
            'read_page': ('reader', None),
            'layout_parsing': ('ocr', None),
            '_chat': ('summary', 'summary_generation'),
            'summarize': ('summary', 'summary_fallback'),
        }
        for name, (route, kind) in helpers.items():
            setattr(search, name, _wrap_helper(getattr(search, name), route, kind))
        search._rllm_statistics_installed = True


@contextmanager
def observe_call(collector):
    install_search_observers()
    token = _CURRENT.set(collector)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def aggregate_steps(steps):
    counts = Counter()
    known_tools = {'crop','select_frame','select_timespan','image_search','text_search',
        'layout_parsing','super_resolution','sharpen','perspective_correct','web_search'}
    for step in steps:
        data = step.info.get('tool_statistics')
        if not data:
            continue
        counts['calls'] += 1
        counts['success'] += int(data['success'])
        counts['failed'] += int(not data['success'])
        tool = data['tool'] if isinstance(data['tool'],str) and data['tool'] in known_tools else 'unknown_tool'
        counts[f"tool/{tool}/calls"] += 1
        counts[f"tool/{tool}/success"] += int(data['success'])
        counts[f"outcome/{data['outcome']}"] += 1
        counts['page_quality_observed_calls'] += int(data['page_quality_observed'])
        counts['cache_hit_calls'] += int(data['cache_hit'])
        counts['degraded_return_calls'] += int(data.get('degraded_return') is True)
        counts['degradation_observed_success_calls'] += int(data['success'] and data.get('degraded_return') is not None)
        counts['quality_eligible_calls'] += int(tool == 'text_search')
        counts['full_quality_observed_calls'] += int(data.get('full_quality_success') is not None)
        counts['full_quality_success_calls'] += int(data.get('full_quality_success') is True)
        counts[f'tool/{tool}/degraded_return_calls'] += int(data.get('degraded_return') is True)
        counts['singleflight_wait_calls'] += int(data['singleflight_wait'])
        counts.update(data['counts'])
        counts.update({f'tool/{tool}/{key}': value for key, value in data['counts'].items()})
        counts[f'tool/{tool}/degradation_observed_success_calls'] += int(data['success'] and data.get('degraded_return') is not None)
        counts[f'tool/{tool}/full_quality_observed_calls'] += int(data.get('full_quality_success') is not None)
        counts[f'tool/{tool}/full_quality_success_calls'] += int(data.get('full_quality_success') is True)
    return dict(counts)


@contextmanager
def observe_judge(collector, route='judge'):
    # Module-local references only; no global requests patch.
    from rllm.rewards import deepresearch_reward
    from vision_deepresearch_async_workflow.utils import api_gateway_client
    for module in (deepresearch_reward, api_gateway_client):
        if hasattr(module, "requests") and not isinstance(module.requests, _RequestsObserver):
            module.requests = _RequestsObserver(module.requests)
    import sys
    workflow = sys.modules.get('vision_deepresearch_async_workflow.deepresearch_workflow')
    for module, names in ((deepresearch_reward, ('_parse_video_judge_decision','_parse_paper_judge_decision')),
                          (workflow, ('parse_query',))):
        if module is None:
            continue
        for name in names:
            parser = getattr(module, name, None)
            if parser is None or getattr(parser, '_tool_statistics_observed', False):
                continue
            def wrap(function):
                @functools.wraps(function)
                def parse(*args, **kwargs):
                    result = function(*args, **kwargs)
                    current = _CURRENT.get()
                    if current is not None:
                        current.add(judge_parse_calls=1, judge_parse_valid=int(result is not None), judge_parse_invalid=int(result is None))
                        current.judge_decision_valid = result is not None
                    return result
                parse._tool_statistics_observed = True
                return parse
            setattr(module, name, wrap(parser))
    token, stage = _CURRENT.set(collector), _STAGE.set(route)
    try:
        yield
    finally:
        _STAGE.reset(stage)
        _CURRENT.reset(token)

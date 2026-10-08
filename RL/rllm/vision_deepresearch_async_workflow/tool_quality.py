"""Read-only quality signals. These are review flags, never rewards or filters."""
import re

_STOP = set('the a an and or of to in is it for on with this that be as by from at are was were has have not can'.split())
_BLOCK = re.compile(r'^(?:\s|[#*>_-])*(?:you are unable to access\b|access denied\b|403 forbidden\b|502 bad gateway\b|checking your browser\b|just a moment\b|please enable javascript and cookies\b|verify you are human\b|attention required.{0,20}cloudflare)', re.I)
_UNAVAILABLE = re.compile(r'(?:unable to (?:access|retrieve)|could not (?:access|retrieve)|blocked by (?:a )?captcha|page (?:is |was )?(?:unavailable|inaccessible))', re.I)


def words(text):
    return {w for w in re.findall(r'[a-zA-Z]{3,}', str(text).lower()) if w not in _STOP}


def quality_signals(body, summary='', query='', title='', *, body_complete=False):
    body, summary = str(body or ''), str(summary or '')
    source, target, q = words(body), words(summary), words(query)
    overlap = len(target & source) / len(target) if target and (body or body_complete) else None
    q_overlap = len(q & (source | words(title))) / len(q) if q and (body or title or body_complete) else None
    numbers = set(re.findall(r'\b\d+(?:[.,]\d+)*\b', summary))
    supported = set(re.findall(r'\b\d+(?:[.,]\d+)*\b', body))
    return dict(version=1, method='lexical_review_flags', body_chars=len(body), summary_chars=len(summary),
        body_complete=body_complete, block_page_suspected=bool(_BLOCK.search(body[:400])) if body or body_complete else None,
        summary_reports_unavailable=bool(_UNAVAILABLE.search(summary)),
        summary_source_token_overlap=overlap, query_source_token_overlap=q_overlap,
        summary_low_overlap_review=overlap < .15 if overlap is not None and len(target) >= 8 else None,
        query_low_overlap_review=q_overlap == 0 if q_overlap is not None and len(q) >= 3 else None,
        unsupported_number_count=len(numbers-supported) if body_complete else None,
        semantic_relevance=None, summary_faithfulness=None,
        # The tests above cannot establish semantic correctness.
        semantic_review_required=True)


def response_semantics(route, response):
    try:
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        if route in {'summary', 'judge', 'query_judge', 'ocr', 'evgr'} and 'choices' in payload:
            return any(bool(str((c.get('message') or {}).get('content') or '').strip()) for c in payload['choices'])
        if route == 'reader' and 'status' in payload:
            return payload.get('status') == 'fetched' and bool(str(payload.get('content') or '').strip())
        if route in {'image', 'text'}:
            if payload.get('code') not in (None, 0, 200):
                return False
            for container in (payload, payload.get('data')):
                if isinstance(container, dict):
                    for key in ('results', 'googleItems', 'organic'):
                        if isinstance(container.get(key), list):
                            return bool(container[key])
    except (ValueError, TypeError, AttributeError, KeyError):
        pass
    return None


def execution_category(observation, failed):
    if not failed:
        return 'success'
    s = str(observation).lower()
    groups = (
        ('tool_contract', ('unknown tool:', 'not callable', 'not available in this modality')),
        ('invalid_arguments', ('image reference', 'image handle', 'image id', 'unexpected keyword', 'missing required', 'arguments must', 'invalid argument')),
        ('parse_error', ('json parse', 'jsondecode', 'expecting value')),
        ('empty_results', ('empty results', 'no search results')),
        ('rate_limit', ('429', 'rate limit')),
        ('timeout_unattributed', ('timeout', 'timed out', 'deadline', '504')),
        ('upstream_unavailable', ('503', '502', 'service unavailable', 'bad gateway')),
        ('connection_error', ('connection', 'connecterror', 'sslerror')),
    )
    return next((name for name, markers in groups if any(m in s for m in markers)), 'execution_error_unattributed')

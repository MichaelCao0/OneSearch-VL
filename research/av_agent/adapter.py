"""Trusted fixed-scope dispatcher. Actor only supplies closed tool arguments."""
from __future__ import annotations
from copy import deepcopy
import json
import hashlib
import math
from pathlib import Path
import time
import uuid

from research.av_data.validate import validate_source
from research.av_search.corpus import load_frozen, validate_scope, visible_corpus, canonical
from research.av_search.retrieval import search_sources
from . import tools
from .observer import CAPABILITIES


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def strict_json(raw, max_chars=65536):
    if not isinstance(raw, str) or len(raw) > max_chars:
        raise ValueError('Expected bounded JSON text')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON value')
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        # JSON numeric literals such as 1e999 must also be rejected.
        json.dumps(value, allow_nan=False)
        return value
    except (json.JSONDecodeError, RecursionError, TypeError) as exc:
        raise ValueError('Invalid JSON') from exc


class EpisodeTools:
    def __init__(self, index_dir, source_registry, scope, output_dir,
                 max_calls=20, max_av_seconds=60):
        _, manifest = load_frozen(index_dir)
        allowed = validate_scope(manifest, scope)
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError('max_calls must be a positive integer')
        if not finite_number(max_av_seconds) or max_av_seconds <= 0:
            raise ValueError('max_av_seconds must be positive and finite')
        if not isinstance(source_registry, dict):
            raise ValueError('Trusted source registry must be a mapping')
        public = visible_corpus(index_dir, scope)['sources']
        for sid in allowed:
            if sid not in source_registry:
                raise ValueError('Missing trusted source record')
            record = source_registry[sid]
            if (validate_source(record) or record['source_id'] != sid or
                    record['asset_sha256'] != public[sid]['asset_sha256'] or
                    record['media_type'] != public[sid]['media_type'] or
                    abs(float(record['duration_s']) - float(public[sid]['duration_s'])) > 1e-9):
                raise ValueError('Registry/source corpus identity mismatch')
        self._manifest_identity = hashlib.sha256(canonical(manifest).encode()).hexdigest()
        self._scope = deepcopy(scope)
        self._registry = {sid: deepcopy(source_registry[sid]) for sid in allowed}
        self._allowed = frozenset(allowed)
        self._index_dir = Path(index_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=False)
        self._max_calls = max_calls
        self._max_av_seconds = float(max_av_seconds)
        self._costs = {'tool_calls': 0, 'requested_av_seconds': 0.0,
                       'provider_requests': 0, 'wall_seconds': 0.0, 'failures': 0}
        self._episode_id = uuid.uuid4().hex
        config = {'episode_id': self._episode_id, 'scope': self._scope,
                  'corpus_hash': manifest['corpus_hash'], 'index_sha256': manifest['index_sha256'],
                  'manifest_sha256': manifest['manifest_sha256'],
                  'capabilities': CAPABILITIES, 'max_calls': max_calls,
                  'max_av_seconds': max_av_seconds}
        (self._output_dir / 'episode.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    @property
    def costs(self):
        return deepcopy(self._costs)

    def _source(self, sid):
        if not isinstance(sid, str) or sid not in self._allowed:
            raise ValueError('Source outside fixed episode visibility')

    def _validate(self, action):
        if not isinstance(action, dict) or set(action) != {'name', 'arguments'}:
            raise ValueError('Tool action requires only name and arguments')
        name, args = action['name'], action['arguments']
        if not isinstance(name, str) or not isinstance(args, dict):
            raise ValueError('Tool name/arguments type invalid')
        schemas = {
            'search_sources': ({'query'}, {'top_k'}),
            'inspect_av': ({'source_id', 'start_s', 'end_s'}, set()),
            'provenance': ({'source_id'}, set()),
            'compare_sources': ({'source_ids'}, set()),
        }
        if name not in schemas:
            raise ValueError('Unknown tool')
        required, optional = schemas[name]
        if not required <= set(args) or set(args) - required - optional:
            raise ValueError('Unexpected or missing tool argument')
        seconds = 0.0
        if name == 'search_sources':
            q, k = args['query'], args.get('top_k', 5)
            if not isinstance(q, str) or not q.strip() or len(q) > 1000:
                raise ValueError('Nonempty bounded query required')
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
                raise ValueError('top_k outside 1..20')
        elif name == 'compare_sources':
            ids = args['source_ids']
            if not isinstance(ids, list) or not 1 <= len(ids) <= 5 or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
                raise ValueError('Expected 1..5 distinct source IDs')
            for sid in ids:
                self._source(sid)
        else:
            self._source(args['source_id'])
        if name == 'inspect_av':
            start, end = args['start_s'], args['end_s']
            if any(not finite_number(x) for x in (start, end)) or start < 0 or end <= start or end > self._registry[args['source_id']]['duration_s']:
                raise ValueError('Invalid source-local half-open interval')
            seconds = float(end-start)
        if self._costs['tool_calls'] >= self._max_calls or self._costs['requested_av_seconds'] + seconds > self._max_av_seconds + 1e-9:
            raise ValueError('Episode work budget exceeded')
        return name, args, seconds

    def dispatch(self, action_json):
        action = strict_json(action_json)
        _, current_manifest = load_frozen(self._index_dir)
        if hashlib.sha256(canonical(current_manifest).encode()).hexdigest() != self._manifest_identity:
            raise ValueError('Frozen episode corpus identity changed')
        name, args, seconds = self._validate(action)
        started = time.monotonic()
        self._costs['tool_calls'] += 1
        self._costs['requested_av_seconds'] += seconds
        private_derived = None
        try:
            if name == 'search_sources':
                receipt = search_sources(self._index_dir, args['query'], self._scope, args.get('top_k', 5))
            elif name == 'inspect_av':
                receipt, private_derived = tools.inspect_av(self._registry[args['source_id']], args['start_s'], args['end_s'], self._output_dir/'clips')
            elif name == 'provenance':
                receipt = tools.provenance(self._index_dir, args['source_id'], self._scope)
            else:
                receipt = tools.compare_sources(self._index_dir, args['source_ids'], self._scope)
        except Exception as exc:
            self._costs['failures'] += 1
            receipt = {'status': 'tool_failed', 'error_code': type(exc).__name__}
        wall = time.monotonic()-started
        self._costs['wall_seconds'] += wall
        receipt.update(schema_version='av-tool-receipt-0.1', tool=name,
                       episode_id=self._episode_id, call_index=self._costs['tool_calls'],
                       corpus_id=current_manifest['corpus_id'], split=current_manifest['split'],
                       corpus_hash=current_manifest['corpus_hash'],
                       index_sha256=current_manifest['index_sha256'],
                       manifest_sha256=current_manifest['manifest_sha256'])
        receipt['logical_cost'] = {**receipt.get('logical_cost', {}), 'tool_calls': 1, 'requested_av_seconds': seconds}
        receipt['actual_cost'] = {**receipt.get('actual_cost', {}), 'provider_requests': 0, 'wall_seconds': wall}
        trusted = {'action': action, 'receipt': receipt, 'private_derived_record': private_derived}
        with (self._output_dir/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(trusted, ensure_ascii=False, allow_nan=False)+'\n')
        with (self._output_dir/'actor_receipts.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False)+'\n')
        return deepcopy(receipt)

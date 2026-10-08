"""Admission before rollout execution; congestion never changes episode rewards."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)


class ToolCapacityUnavailable(RuntimeError):
    pass


def normalize_pressure(data):
    """Normalize API, search and reader-cache stats without summing two hops."""
    if isinstance(data.get("routes"), dict) and data["routes"]:
        routes = dict(data["routes"])
    elif isinstance(data.get("gate"), dict):
        routes = {"search": {**data["gate"], "traffic": data}}
    elif "tail_budget_fallbacks" in data:
        routes = {"reader": {"traffic": {
            "pressure_instance": data.get("pressure_instance", "reader"),
            "local_timeout_total": data["tail_budget_fallbacks"],
            "local_timeout_last_60s": data.get("tail_budget_fallbacks_last_60s", 0),
        }}}
    else:
        raise ValueError("pressure endpoint has no recognized routes")
    if data.get("server_pressure"):
        routes["__server__"] = {"traffic": data["server_pressure"]}
    return routes


class ToolAdmission:
    def __init__(self, capacity, url=None, fetch=None):
        self.url = url if url is not None else os.getenv("RLLM_TOOL_PRESSURE_URL", "")
        raw = os.getenv("RLLM_TOOL_PRESSURE_URLS", "") if url is None else ""
        self.urls = json.loads(raw) if raw else ({"shared": self.url} if self.url else {})
        if not isinstance(self.urls, dict) or any(not isinstance(v, str) or not v for v in self.urls.values()):
            raise ValueError("RLLM_TOOL_PRESSURE_URLS must be a named URL object")
        self.enabled = bool(self.urls) or bool(fetch and self.url)
        self.maximum = max(1, min(capacity, int(os.getenv("RLLM_TOOL_MAX_ROLLOUTS", str(capacity)))))
        self.limit = min(self.maximum, max(1, int(os.getenv("RLLM_TOOL_INITIAL_ROLLOUTS", "8"))))
        self.active = 0
        self.lock = asyncio.Lock()
        self.last_poll = None
        self.paused = False
        self.congested = False
        self.state = "healthy"
        self.fetch = fetch or self._fetch
        self.poll_seconds = max(.01, float(os.getenv("RLLM_TOOL_PRESSURE_POLL_S", "5")))
        self.max_pause = float(os.getenv("RLLM_TOOL_PRESSURE_MAX_PAUSE_S", "600"))
        self.sustain_s = float(os.getenv("RLLM_TOOL_QUEUE_SUSTAIN_S", "5"))
        self.queue_high_ms = float(os.getenv("RLLM_TOOL_QUEUE_HIGH_MS", "2000"))
        self.cooldown_s = float(os.getenv("RLLM_TOOL_DECREASE_COOLDOWN_S", "20"))
        self.probe_interval_s = float(os.getenv("RLLM_TOOL_CONGESTED_START_INTERVAL_S", "5"))
        self.start_interval_s = float(os.getenv("RLLM_TOOL_START_INTERVAL_S", "0"))
        if min(self.start_interval_s, self.probe_interval_s, self.sustain_s, self.cooldown_s) < 0:
            raise ValueError("admission intervals must be nonnegative")
        self._counters = {}
        self._queue_since = None
        self._outage_since = None
        self._last_decrease = float("-inf")
        self._backoff_until = 0
        self._last_start = float("-inf")
        self._healthy_polls = 0
        self._budget_blocked = False
        self.decreases = 0
        self.polls = 0
        self._last_log = None

    def _fetch(self):
        def fetch_one(item):
            name, url = item
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=2) as response:
                return name, normalize_pressure(json.loads(response.read()))
        routes = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.urls))) as executor:
            for name, values in executor.map(fetch_one, self.urls.items()):
                routes.update({f"{name}/{route}": value for route, value in values.items()})
        return {"routes": routes}

    @staticmethod
    def pressured(snapshot):
        """Stateless diagnostic; admission uses deltas and persistence."""
        routes = snapshot.get("routes", {})
        if not routes:
            return True
        for route in routes.values():
            traffic = route.get("traffic", {})
            age = traffic.get("oldest_wait_ms", traffic.get("queue_p95_last_60s_ms", 0))
            if age > 2000 or any(traffic.get(k, 0) > 0 for k in ("limited_last_60s", "local_timeout_last_60s", "rejected_last_60s")):
                return True
        return False

    def _signals(self, data):
        routes = data.get("routes")
        if not isinstance(routes, dict) or not routes:
            raise ValueError("pressure metrics unavailable")
        queued = False
        fresh_failure = False
        for name, route in routes.items():
            traffic = route.get("traffic", {})
            age = traffic.get("oldest_wait_ms", traffic.get("queue_p95_last_60s_ms", 0))
            queued |= age > self.queue_high_ms
            for total, window in (("limited_total", "limited_last_60s"), ("local_timeout_total", "local_timeout_last_60s"), ("rejected_total", "rejected_last_60s")):
                key = (name, traffic.get("pressure_instance", "legacy"), total)
                value = traffic.get(total, traffic.get(window, 0))
                previous = self._counters.get(key)
                # Baseline old lifetime counters; only recent failures matter
                # on first contact. Instance IDs disambiguate server restarts.
                fresh_failure |= (traffic.get(window, 0) > 0 if previous is None else value > previous)
                self._counters[key] = value
        return queued, fresh_failure

    def update(self, data):
        """One observed poll, also callable by deterministic workload tests."""
        now = time.monotonic()
        had_poll = self.last_poll is not None
        self.last_poll = now
        self.polls += 1
        try:
            queued, fresh = self._signals(data)
        except (ValueError, TypeError, AttributeError):
            if self._outage_since is None:
                self._outage_since = now
            self.paused = True
            self.state = "unavailable"
            self._healthy_polls = 0
            self._report()
            return
        self._outage_since = None
        self._budget_blocked = any(route.get("quota_exhausted") or route.get("cooldown_remaining_s", 0) > 0
                                   for route in data["routes"].values())
        if queued:
            if self._queue_since is None:
                self._queue_since = now
        else:
            self._queue_since = None
        sustained = self._queue_since is not None and now - self._queue_since >= self.sustain_s
        pressure = sustained or fresh
        entered = pressure and not self.congested
        if pressure:
            self._healthy_polls = 0
            self.congested = True
            # Persistent queue/old failure: one reduction. Further reductions
            # require NEW failures and a cooldown, not repeated window counts.
            if (entered or fresh) and now - self._last_decrease >= self.cooldown_s:
                self.limit = max(1, self.limit // 2)
                self._last_decrease = now
                self._backoff_until = now + self.poll_seconds
                self.decreases += 1
        elif not queued:
            self._healthy_polls += 1
            if self._healthy_polls >= 2 and now >= self._backoff_until:
                self.congested = False
                if had_poll:
                    self.limit = min(self.maximum, self.limit + 1)
        self.paused = self._budget_blocked or now < self._backoff_until
        self.state = "congested" if self.congested else "healthy"
        self._report()

    def snapshot(self):
        return {"state": self.state, "limit": self.limit, "active": self.active,
                "paused": self.paused, "polls": self.polls, "decreases": self.decreases,
                "pressure_sources": len(self.urls)}

    def _report(self):
        status = (self.state, self.limit, self.paused)
        if status != self._last_log:
            logger.info("Tool admission: state=%s limit=%s paused=%s active=%s decreases=%s", *status, self.active, self.decreases)
            self._last_log = status

    def _ready(self):
        now = time.monotonic()
        if self._outage_since is not None and now - self._outage_since >= self.max_pause:
            raise ToolCapacityUnavailable("tool pressure endpoints unavailable before rollout admission")
        interval = max(self.start_interval_s, self.probe_interval_s if self.congested else 0)
        return (not self.paused and self.active < self.limit and now - self._last_start >= interval)

    async def acquire(self):
        if not self.enabled:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                if self.last_poll is None or now - self.last_poll >= self.poll_seconds:
                    try:
                        data = await asyncio.to_thread(self.fetch)
                    except Exception:
                        data = {}
                    self.update(data)
                if self._ready():
                    self.active += 1
                    self._last_start = time.monotonic()
                    return
            await asyncio.sleep(.1)

    def release(self):
        if self.enabled:
            self.active -= 1


_evaluation_gate = None


def wait_for_tool_capacity():
    """Keep pressure history across serial evaluation cases in this process."""
    global _evaluation_gate
    if _evaluation_gate is None:
        _evaluation_gate = ToolAdmission(1)
    gate = _evaluation_gate
    if not gate.enabled:
        return
    while True:
        try:
            data = gate.fetch()
        except Exception:
            data = {}
        gate.update(data)
        if gate._ready():
            gate._last_start = time.monotonic()
            return
        time.sleep(gate.poll_seconds)

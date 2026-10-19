"""DeepSeek web Search/Fetch backend for the original OneSearch dispatcher.

Receipts are scoped to a versioned run. Successful identical requests replay
from immutable bodies; failed/uncertain attempts are never sent again in that
run. This module has no dependency on corpus annotations or reference answers.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

API_BASE = "https://api.deepseek.com/v1/tools"
SCHEMA = "onesearch-deepseek-web.v1"


class SearchError(RuntimeError):
    """Sanitized provider/receipt error; never carries response body or key."""


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def locked(path):
    with Path(path).open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield stream


def public_url(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 8192:
        raise ValueError("Expected a public HTTP(S) URL of at most 8192 characters")
    if any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError("Whitespace/control characters in URL")
    try:
        p = urlsplit(value)
        host = (p.hostname or "").lower().rstrip(".")
        port = p.port
    except ValueError:
        raise ValueError("Malformed public URL") from None
    if p.scheme not in {"http", "https"} or not host or p.username is not None or p.password is not None:
        raise ValueError("Public HTTP(S) URL without credentials required")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Private host is not a web source")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or "%" in host:
            raise ValueError("Public host required") from None
    else:
        if not address.is_global:
            raise ValueError("Private IP is not a web source")
    return value


def read_private_key(path):
    path = Path(path).expanduser()
    st = path.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError("Private config must be owned by current user with mode 0600")
    values = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    key = values.get("DEEPSEEK_API_KEY", "")
    if not key or any(c.isspace() for c in key):
        raise ValueError("Missing DeepSeek API credential")
    return key


class DeepSeekSearchBackend:
    """Duck-typed text_search/visit_webpage provider; no offline fallback.

    lang remains a compatibility hint, as in FrozenSearchBackend. Callers can
    explicitly use documented lang: operators in query to restrict language.
    rate_dir must be shared by all project processes using the same account.
    """

    def __init__(self, run_dir, *, api_key=None, key_file=None, max_requests=100,
                 original_question=None, need_content=False, transport=None,
                 rate_dir=None, min_interval=.30, timeout=60):
        if type(max_requests) is not int or not 1 <= max_requests <= 100000:
            raise ValueError("Explicit positive request budget required")
        if not math.isfinite(min_interval) or min_interval < .21:
            raise ValueError("Account rate interval must be at least .21 seconds")
        if not math.isfinite(timeout) or not 1 <= timeout <= 300:
            raise ValueError("Invalid timeout")
        self.root = Path(run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipts = self.root / "requests"
        self.receipts.mkdir(exist_ok=True)
        self.rate_dir = Path(rate_dir or "~/.cache/onesearch-av/deepseek-rate").expanduser()
        self.rate_dir.mkdir(parents=True, exist_ok=True)
        self._key = api_key or (read_private_key(key_file) if key_file else None)
        if not self._key:
            raise ValueError("Explicit private credential required")
        self.transport = transport or self._http
        self.max_requests = max_requests
        self.original_question = original_question
        self.need_content = bool(need_content)
        self.min_interval = min_interval
        self.timeout = timeout
        self.events = []
        identity = {"schema": SCHEMA, "provider": "deepseek", "base_url": API_BASE,
                    "max_requests": max_requests, "need_content": self.need_content,
                    "original_question_sha256": digest(original_question.encode()) if original_question else None,
                    "automatic_retries": 0, "offline_fallback": False,
                    "language_policy": "query unchanged; lang: operator available",
                    "per_success_request_yuan": .005}
        with locked(self.root / ".lock"):
            config = self.root / "backend.json"
            if config.exists() and json.loads(config.read_text()) != identity:
                raise ValueError("Backend identity changed; use a new run directory")
            if not config.exists():
                atomic_json(config, identity)

    def _http(self, endpoint, payload):
        import requests
        response = requests.post(API_BASE + "/" + endpoint,
                                 headers={"Authorization": "Bearer " + self._key,
                                          "Content-Type": "application/json"},
                                 json=payload, timeout=(10, self.timeout),
                                 allow_redirects=False, stream=True)
        try:
            chunks = []; size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 8 * 1024 * 1024:
                    raise SearchError("Provider response exceeds 8 MiB limit")
                chunks.append(chunk)
            return response.status_code, b"".join(chunks)
        finally:
            response.close()

    def _pace(self, endpoint):
        # Wall-clock timestamp permits serialization across independent workers.
        with locked(self.rate_dir / (endpoint + ".lock")):
            path = self.rate_dir / (endpoint + ".json")
            old = json.loads(path.read_text()).get("last_epoch", 0) if path.exists() else 0
            time.sleep(max(0, min(self.min_interval, self.min_interval - (time.time() - old))))
            atomic_json(path, {"last_epoch": time.time()})

    @staticmethod
    def _validate(endpoint, body):
        if not isinstance(body, dict):
            raise ValueError("Object response required")
        if endpoint == "fetch":
            if not all(isinstance(body.get(k), str) for k in ["title", "content"]):
                raise ValueError("Malformed Fetch response")
            return
        if not isinstance(body.get("results"), list):
            raise ValueError("Malformed Search response")
        for hit in body["results"]:
            if not isinstance(hit, dict) or not all(isinstance(hit.get(k), str) for k in ["title", "url", "snippet", "summary", "content"]):
                raise ValueError("Malformed Search result")
            public_url(hit["url"])
            score = hit.get("rerank_score")
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 3:
                raise ValueError("Malformed rank score")
            if type(hit.get("authority")) is not int or not -1 <= hit["authority"] <= 4:
                raise ValueError("Malformed authority score")

    def _request(self, endpoint, payload):
        request = {"endpoint": endpoint, "payload": payload, "schema": SCHEMA}
        rid = digest(json.dumps(request, ensure_ascii=False, sort_keys=True).encode())
        folder = self.receipts / rid
        with locked(self.root / ".lock"):
            if folder.exists():
                state = json.loads((folder / "receipt.json").read_text()) if (folder / "receipt.json").exists() else {}
                if state.get("status") != "completed":
                    raise SearchError("Previous failed or uncertain request preserved; no automatic replay")
                raw = (folder / "response.json").read_bytes()
                if digest(raw) != state["stored_response_sha256"]:
                    raise SearchError("Cached provider response changed")
                body = json.loads(raw)
                self._validate(endpoint, body)
                self.events.append({"request_id": rid, "cache_hit": True, "endpoint": endpoint})
                return body, state
            if len(list(self.receipts.iterdir())) >= self.max_requests:
                raise SearchError("DeepSeek request budget exhausted")
            folder.mkdir()
            atomic_json(folder / "request.json", request)
            state = {"schema": SCHEMA, "request_id": rid, "endpoint": endpoint,
                     "status": "started", "started_at": utcnow(), "http_status": None}
            atomic_json(folder / "receipt.json", state)
            before = time.monotonic()
            try:
                self._pace(endpoint)
                status, raw = self.transport(endpoint, payload)
                state["http_status"] = status
                if not isinstance(raw, bytes):
                    raise ValueError("Transport must return bytes")
                state["wire_response_sha256"] = digest(raw)
                # Provider errors may echo request headers; never persist the credential.
                raw = raw.replace(self._key.encode(), b"[REDACTED]")
                state["stored_response_sha256"] = digest(raw)
                (folder / "response.json").write_bytes(raw)
                if status != 200:
                    raise SearchError("DeepSeek HTTP " + str(status))
                body = json.loads(raw)
                self._validate(endpoint, body)
                nonempty = bool(body["results"]) if endpoint == "search" else bool(body["content"].strip())
                state.update(status="completed", nonempty=nonempty,
                             estimated_yuan=.005 if nonempty else 0.)
            except Exception as exc:
                state.update(status="failed", error_type=type(exc).__name__, estimated_yuan=None)
                raise SearchError("DeepSeek request failed; receipt preserved (HTTP " + str(state["http_status"]) + ")") from None
            finally:
                state.update(finished_at=utcnow(), wall_seconds=time.monotonic() - before)
                atomic_json(folder / "receipt.json", state)
                self.events.append({"request_id": rid, "cache_hit": False, "endpoint": endpoint})
            return body, state

    def search(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip() or not 1 <= len(query) <= 128:
            raise ValueError("Search query must contain 1-128 Unicode characters")
        if type(top_k) is not int or not 1 <= top_k <= 50:
            raise ValueError("Search count must be between 1 and 50")
        payload = {"query": query, "count": top_k, "need_content": self.need_content}
        q = self.original_question
        if isinstance(q, str) and q.strip() and len(q) <= 128:
            payload["ori_question"] = q
        return self._request("search", payload)

    def fetch(self, url):
        return self._request("fetch", {"url": public_url(url)})

    def text_search(self, query, lang="en", top_k=5):
        body, state = self.search(query, top_k)
        passages = []
        remaining = 12000
        for rank, hit in enumerate(body["results"], 1):
            text = hit["summary"] or hit["snippet"]
            kind = "web_search_summary"
            if self.need_content:
                content = hit.get("content", "").strip()
                if content:
                    kind = "web_search_summary_and_body"
                    text = "Summary:\n" + text[:600] + "\nPage body excerpt:\n" + content
                allowance = min(2400, remaining)
                excerpt = text[:allowance]
                remaining -= len(excerpt)
                if len(text) > allowance:
                    excerpt += "\n[Excerpt truncated; visit_webpage can read the page.]"
            else:
                excerpt = text[:2400]
            passages.append(f'[Passage {rank}]\nTitle: {hit["title"]}\nURL: {hit["url"]}\n'
                            f'Text kind: {kind}\nExcerpt:\n{excerpt}')
        header = "Tool execution result:\nProvider: DeepSeek web search; retrieved at " + state["finished_at"] + ".\n"
        return header + ("\n\n".join(passages) if passages else "No relevant sources found for the query.")

    def visit_webpage(self, url):
        body, state = self.fetch(url)
        content = body["content"]
        suffix = "\n[Body truncated after 20000 characters.]" if len(content) > 20000 else ""
        return (f'Tool execution result:\nTitle: {body["title"]}\nURL: {url}\n'
                f'Text kind: fetched_web_markdown\nRetrieved at: {state["finished_at"]}\nBody:\n'
                + (content[:20000] + suffix if content.strip() else "No extractable webpage content."))

    def summary(self):
        rows = [json.loads(p.read_text()) for p in self.receipts.glob("*/receipt.json")]
        return {"provider": "deepseek", "attempted_requests": len(rows),
                "completed_requests": sum(r["status"] == "completed" for r in rows),
                "failed_or_uncertain": sum(r["status"] != "completed" for r in rows),
                "nonempty_requests": sum(r.get("nonempty", False) for r in rows),
                "known_estimated_yuan": sum(r.get("estimated_yuan") or 0 for r in rows),
                "current_process_cache_hits": sum(e["cache_hit"] for e in self.events),
                "automatic_retries": 0, "offline_fallback": False}

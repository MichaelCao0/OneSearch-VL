"""Shared OneSearch-VL providers: Serper search/Lens, Jina, and chat summaries.

RL imports this module too. Credentials are read at call time, never embedded.
Gateway mode follows the upstream provider bearer protocol; direct mode uses
public APIs. Tool observations retain the training/inference text envelope.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests


class SearchCancelled(TimeoutError):
    pass


class ProviderError(RuntimeError):
    pass


@dataclass
class SearchControl:
    deadline: float | None = None
    timing: dict = field(default_factory=dict)
    kind: str = "search"
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def check(self):
        if self.cancel_event.is_set() or (self.deadline is not None and time.monotonic() >= self.deadline):
            raise SearchCancelled(f"{self.kind} deadline exceeded")

    def timeout(self):
        self.check()
        limit = max(0.01, float(os.getenv("SEARCH_HTTP_TIMEOUT", "30")))
        return min(limit, max(0.001, self.deadline - time.monotonic())) if self.deadline else limit


def _env(name, default="", alias=""):
    return os.getenv(name, "").strip() or (os.getenv(alias, "").strip() if alias else "") or default


def _gateway():
    values = tuple(_env(k, alias=a) for k, a in (
        ("API_HOST", "API_GATEWAY_HOST"), ("API_USER", "API_GATEWAY_USER"), ("API_KEY", "API_GATEWAY_KEY")))
    if any(values) and not all(values):
        raise ProviderError("Set API_HOST, API_USER and API_KEY together, or leave all three empty")
    return values if all(values) else None


def _headers(provider, gateway):
    _, user, key = gateway
    return {"Authorization": f"Bearer {user}:{key}?provider={provider}&timeout=60", "Content-Type": "application/json"}


def _request(method, url, *, control=None, **kwargs):
    """Bound retries; never include request URLs, headers or response bodies in errors."""
    control = control or SearchControl()
    attempts = max(1, min(5, int(_env("SEARCH_MAX_RETRIES", "3"))))
    for attempt in range(attempts):
        control.check()
        try:
            response = requests.request(method, url, timeout=control.timeout(), **kwargs)
            control.check()
            status = response.status_code
            if status == 429 or status >= 500:
                if attempt + 1 < attempts:
                    delay = min(2 ** attempt, 4)
                    if control.deadline is not None:
                        delay = min(delay, max(0, control.deadline - time.monotonic()))
                    control.cancel_event.wait(delay)
                    continue
            if status >= 400:
                raise ProviderError(f"Provider returned HTTP {status}")
            return response
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 == attempts:
                raise ProviderError("Provider connection failed or timed out") from None
        except requests.RequestException:
            raise ProviderError("Provider request failed") from None
    raise ProviderError("Provider retries exhausted")


def _json(response):
    try:
        data = response.json()
    except (ValueError, TypeError):
        raise ProviderError("Provider returned invalid JSON") from None
    if not isinstance(data, dict):
        raise ProviderError("Provider returned an invalid response object")
    if data.get("code") not in (None, 0, 200):
        raise ProviderError("Provider reported an unsuccessful response")
    return data


def normalize_top_k(value):
    try:
        return max(1, min(20, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 5


def search_web(query, lang="en", top_k=5, control=None):
    gateway = _gateway()
    payload = {"q": query, "hl": lang or "en", "num": normalize_top_k(top_k)}
    if gateway:
        url, headers = gateway[0].rstrip("/") + "/search", _headers("serper", gateway)
        payload["location"] = "United States"
    else:
        key = _env("SERPER_API_KEY", alias="SERP_API_KEY")
        if not key:
            raise ProviderError("SERPER_API_KEY is required for direct search")
        url = _env("SERPER_SEARCH_URL", "https://google.serper.dev/search")
        headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    data = _json(_request("POST", url, headers=headers, json=payload, control=control))
    hits = data.get("organic", [])
    if not isinstance(hits, list):
        raise ProviderError("Serper organic results must be a list")
    return [hit for hit in hits if isinstance(hit, dict)][:payload["num"]]


def read_page(url, control=None):
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ProviderError("A HTTP(S) webpage URL is required")
    gateway = _gateway()
    if gateway:
        response = _request("POST", gateway[0].rstrip("/") + "/images", headers=_headers("jina_ai", gateway), json={"url": url}, control=control)
        data = _json(response)
    else:
        headers = {"Accept": "application/json"}
        if _env("JINA_API_KEY"):
            headers["Authorization"] = "Bearer " + _env("JINA_API_KEY")
        response = _request("GET", _env("JINA_READER_URL", "https://r.jina.ai/").rstrip("/") + "/" + url, headers=headers, control=control)
        try:
            response.json()
        except ValueError:
            return response.text
        data = _json(response)
    body = data.get("data", data)
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        return str(body.get("content") or body.get("text") or body.get("markdown") or "")
    raise ProviderError("Reader returned an invalid content object")


def _chat(prompt, control=None):
    base = _env("QWEN_API_BASE", "http://localhost:8000/v1")
    headers = {"Content-Type": "application/json"}
    if _env("QWEN_API_KEY"):
        headers["Authorization"] = "Bearer " + _env("QWEN_API_KEY")
    payload = {"model": _env("QWEN_MODEL_NAME", "Qwen/Qwen3-32B", "QWEN_MODEL"),
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 1024, "temperature": 0.3, "top_p": 0.95}
    # Plain HTTP JSON: OpenAI SDK extra_body fields belong at the top level.
    if _env("QWEN_ENABLE_THINKING", "").lower() in {"0", "false"}:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = _json(_request("POST", base.rstrip("/") + "/chat/completions", headers=headers, json=payload, control=control))
    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError()
        return content.strip()
    except (KeyError, IndexError, TypeError, ValueError):
        raise ProviderError("Summary model returned no text") from None


def summarize(content, query="", title="", control=None):
    if not content:
        return ""
    try:
        return _chat(f"Summarize the webpage in 2-4 sentences relevant to the query. Use only the supplied evidence.\nQuery: {query}\nTitle: {title}\nContent:\n{content[:2000]}", control)
    except ProviderError:
        return content[:500] + ("..." if len(content) > 500 else "")


def _passages(hits, query, control, with_reader):
    passages = []
    for index, hit in enumerate(hits, 1):
        title, url = str(hit.get("title", "")), str(hit.get("link") or hit.get("url") or "")
        content = str(hit.get("snippet") or "")
        if with_reader and url:
            try:
                content = read_page(url, control) or content
            except ProviderError:
                pass  # A real Serper snippet remains valid evidence if the reader fails.
        summary = summarize(content, query, title, control) if with_reader else content
        passages.append(f"[Passage {index}]\nTitle: {title}\nURL: {url}\nSummary:\n{summary}")
    return "Tool execution result:\n" + (("\n\n" + "=" * 60 + "\n\n").join(passages) if passages else "No relevant web pages found for the query.")


def text_search(query, lang="en", top_k=5, control=None):
    if not isinstance(query, str) or not query.strip():
        return "Tool execution error:\nText search requires a non-empty query."
    try:
        control = control or SearchControl(deadline=time.monotonic() + float(_env("TEXT_SEARCH_BACKEND_TIMEOUT", "120")))
        return _passages(search_web(query, lang, top_k, control), query, control, True)
    except (ProviderError, SearchCancelled) as exc:
        return f"Tool execution error:\nText search failed: {exc}"


def web_search(query, lang="en", top_k=5, control=None):
    try:
        return _passages(search_web(query, lang, top_k, control), query, control, False)
    except (ProviderError, SearchCancelled) as exc:
        return f"Tool execution error:\nWeb search failed: {exc}"


def visual_lookup(image_url, control=None):
    if not isinstance(image_url, str) or not image_url.startswith(("http://", "https://")):
        raise ProviderError("Image search requires a public HTTP(S) image URL")
    gateway = _gateway()
    if gateway:
        # Upstream RL accepts an explicit gateway Lens endpoint.
        url = _env("IMAGE_SEARCH_URL")
        if not url:
            raise ProviderError("IMAGE_SEARCH_URL is required in gateway mode")
        headers = _headers("serper", gateway)
    else:
        key = _env("SERPER_API_KEY", alias="SERP_API_KEY")
        if not key:
            raise ProviderError("SERPER_API_KEY is required for image search")
        url = _env("IMAGE_SEARCH_URL", "https://google.serper.dev/lens")
        headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    hits = _json(_request("POST", url, headers=headers, json={"url": image_url}, control=control)).get("organic", [])
    if not isinstance(hits, list):
        raise ProviderError("Lens organic results must be a list")
    return [{"title": str(h.get("title") or ""), "source": str(h.get("link") or h.get("source") or ""),
             "imageUrl": str(h.get("imageUrl") or ""), "thumbnailUrl": str(h.get("thumbnailUrl") or "")}
            for h in hits if isinstance(h, dict)]


def image_search(image_url, visual_lookup=None, max_retries=3, base_delay=2, control=None):
    try:
        control = control or SearchControl(deadline=time.monotonic() + float(_env("IMAGE_SEARCH_BACKEND_TIMEOUT", "120")))
        lookup = visual_lookup
        if lookup is None:
            hits = globals()["visual_lookup"](image_url, control)
        else:
            for attempt in range(max(1, max_retries)):
                control.check()
                try:
                    hits = lookup(image_url=image_url)
                    break
                except Exception:
                    if attempt + 1 == max(1, max_retries):
                        raise ProviderError("Visual lookup failed") from None
                    control.cancel_event.wait(min(base_delay * 2 ** attempt, max(0, control.deadline - time.monotonic()) if control.deadline else 8))
        control.check()
        # Preserve exact retrieved titles/URLs instead of asking a model to recreate links.
        return "Tool execution result:\n" + json.dumps(hits, ensure_ascii=False)
    except (ProviderError, SearchCancelled) as exc:
        return f"Tool execution error:\nImage search failed: {exc}"


def layout_parsing(file_path, use_chart_recognition=False, use_doc_orientation_classify=False, use_doc_unwarping=False, control=None):
    empty = {"rec_texts": [], "formatted_text": "", "blocks": [], "error": None}
    try:
        endpoint = _env("LAYOUT_PARSING_API_URL")
        if not endpoint:
            raise ProviderError("LAYOUT_PARSING_API_URL is not configured")
        headers = {"Content-Type": "application/json"}
        if _env("LAYOUT_PARSING_TOKEN"):
            headers["Authorization"] = "token " + _env("LAYOUT_PARSING_TOKEN")
        payload = {"file": base64.b64encode(Path(file_path).read_bytes()).decode(), "fileType": 1,
                   "useChartRecognition": use_chart_recognition, "useDocOrientationClassify": use_doc_orientation_classify,
                   "useDocUnwarping": use_doc_unwarping}
        data = _json(_request("POST", endpoint, headers=headers, json=payload, control=control))
        blocks = data["result"]["layoutParsingResults"][0]["prunedResult"]["parsing_res_list"]
        texts = [str(b["block_content"]) for b in blocks if b.get("block_label") in {"paragraph_title", "text", "vision_footnote"}]
        return {"rec_texts": texts, "formatted_text": "\n".join(texts), "blocks": blocks, "error": None}
    except (ProviderError, SearchCancelled) as exc:
        return dict(empty, error=str(exc))
    except (OSError, KeyError, IndexError, TypeError):
        return dict(empty, error="Unable to read image or parse layout response")

"""Optional frozen search provider for the original inference tool dispatcher."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from research.av_search.corpus import load_frozen, validate_scope, visible_corpus
from research.av_search.retrieval import get_source, search_sources


class FrozenSearchBackend:
    """Search/visit only documents in a fixed, validated episode scope.

    No HTTP requests, answer annotations, or internal asset paths are returned.
    The manifest identity is pinned when this backend is created. The actor
    supplies queries and public URLs; it cannot replace index or scope config.
    """

    def __init__(self, index_dir, scope: dict):
        self.index_dir = Path(index_dir).resolve()
        self.scope = deepcopy(scope)
        _, manifest = load_frozen(self.index_dir)
        validate_scope(manifest, self.scope)
        self._manifest_sha256 = manifest["manifest_sha256"]

    def _check(self):
        _, manifest = load_frozen(self.index_dir)
        if manifest["manifest_sha256"] != self._manifest_sha256:
            raise ValueError("Frozen corpus changed since backend initialization")
        validate_scope(manifest, self.scope)

    def text_search(self, query: str, lang: str = "en", top_k: int = 5) -> str:
        self._check()
        receipt = search_sources(self.index_dir, query, self.scope, top_k=top_k)
        passages = []
        for hit in receipt["results"]:
            title = hit["public_metadata"].get("title", "")
            passages.append(f'[Passage {hit["rank"]}]\nTitle: {title}\nURL: {hit["origin_url"]}\n'
                            f'Text kind: {hit["text_kind"]}\nExcerpt:\n{hit["excerpt"]}')
        return "Tool execution result:\n" + ("\n\n".join(passages) if passages else "No relevant sources found for the query.")

    def visit_webpage(self, url: str) -> str:
        self._check()
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("A public HTTP(S) source URL is required")
        # Only the validated public index is consulted. Never resolve a URL to
        # an internal asset, and never fetch an unknown URL over the network.
        public = visible_corpus(self.index_dir, self.scope)
        matches = [doc for doc in public["sources"].values() if doc["origin_url"] == url]
        if not matches:
            raise ValueError("URL is outside the visible frozen corpus")
        passages = []
        for doc in matches:
            source = get_source(self.index_dir, doc["source_id"], self.scope)
            body = source["text"]
            # Bound context without generating a summary or changing the quote.
            suffix = "\n[Body truncated after 20000 characters.]" if len(body) > 20000 else ""
            passages.append(f'Title: {source["public_metadata"].get("title", "")}\nURL: {source["origin_url"]}\n'
                            f'Text kind: {source["text_kind"]}\nBody:\n{body[:20000]}{suffix}')
        return "Tool execution result:\n" + "\n\n".join(passages)

"""Read-only scoped retrieval with quotes into a validated pinned content version."""
from __future__ import annotations

import math
import re
import time

from .corpus import _source_id, load_frozen, open_index, strict_json_loads, validate_scope


def get_source(index_dir, source_id: str, scope: dict) -> dict:
    _source_id(source_id)
    root, manifest = load_frozen(index_dir)
    allowed = validate_scope(manifest, scope)
    if source_id not in allowed:
        raise ValueError("Source outside episode visibility")
    with open_index(root) as connection:
        row = connection.execute("SELECT payload FROM docs WHERE source_id=?", (source_id,)).fetchone()
    if row is None:
        raise ValueError("Missing visible source")
    return strict_json_loads(row[0])


def search_sources(index_dir, query: str, scope: dict, top_k: int = 5) -> dict:
    started = time.monotonic()
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ValueError("Nonempty bounded query required")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k outside 1..20")
    tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)[:32]
    if not tokens:
        raise ValueError("Query contains no searchable tokens")
    root, manifest = load_frozen(index_dir)
    allowed = validate_scope(manifest, scope)
    expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
    with open_index(root) as connection:
        rows = connection.execute(
            "SELECT source_id,bm25(search) FROM search WHERE search MATCH ? ORDER BY bm25(search),source_id",
            (expression,),
        ).fetchall()
        matches = [(source, score) for source, score in rows if source in allowed][:top_k]
        results = []
        for rank, (source, score) in enumerate(matches, 1):
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Invalid search score")
            row = connection.execute("SELECT payload FROM docs WHERE source_id=?", (source,)).fetchone()
            if row is None:
                raise ValueError("Search source has no registered public document")
            document = strict_json_loads(row[0])
            text = document["text"]
            lower = text.lower()
            positions = [lower.find(token.lower()) for token in tokens if lower.find(token.lower()) >= 0]
            start = max(0, min(positions, default=0) - 160)
            end = min(len(text), start + 800)
            results.append({
                "rank": rank, "source_id": source, "score": -score,
                "origin_url": document["origin_url"], "asset_sha256": document["asset_sha256"],
                "text_sha256": document["text_sha256"], "text_kind": document["text_kind"],
                "excerpt": text[start:end], "quote_start": start, "quote_end": end,
                "public_metadata": document["public_metadata"],
            })
    return {
        "schema_version": "av-search-receipt-0.2",
        "status": "ok" if results else "no_results",
        "corpus_id": manifest["corpus_id"], "split": manifest["split"],
        "corpus_hash": manifest["corpus_hash"], "index_sha256": manifest["index_sha256"],
        "manifest_sha256": manifest["manifest_sha256"], "retriever": manifest["retriever"],
        "query": query, "top_k": top_k, "results": results,
        "logical_cost": {"search_calls": 1, "returned_chars": sum(len(result["excerpt"]) for result in results)},
        "actual_cost": {"provider_requests": 0, "wall_seconds": time.monotonic() - started},
    }

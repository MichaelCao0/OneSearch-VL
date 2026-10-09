"""Frozen public corpus with validated JSON, document/FTS identity and config."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Iterable
from urllib.parse import urlsplit

SCHEMA_VERSION = "av-corpus-0.2"
SOURCE_ID = re.compile(r"src_[0-9a-f]{16,64}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CORPUS_ID = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
SPLITS = frozenset({"exploration", "train", "dev", "test", "ood", "iid_test", "ood_test"})
DOC_FIELDS = frozenset({
    "source_id", "origin_url", "asset_sha256", "asset_path", "text", "text_kind",
    "public_metadata", "media_type", "duration_s",
})
PUBLIC_DOC_FIELDS = (DOC_FIELDS - {"asset_path"}) | {"text_sha256"}
META_FIELDS = frozenset({"title", "description", "creator", "published_at", "license", "license_url", "categories"})
TEXT_KINDS = frozenset({"public_metadata", "web_body", "asr", "ocr"})
MANIFEST_FIELDS = frozenset({
    "schema_version", "corpus_id", "split", "source_ids", "corpus_hash",
    "index_sha256", "documents_sha256", "retriever", "manifest_sha256",
})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_json(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for child in value.values():
            _check_json(child)
    elif isinstance(value, list):
        for child in value:
            _check_json(child)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise ValueError("Expected a JSON value")


def canonical(data) -> str:
    _check_json(data)
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def strict_json_loads(text):
    try:
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed JSON record") from exc
    _check_json(result)
    return result


def _source_id(value):
    if not isinstance(value, str) or not SOURCE_ID.fullmatch(value):
        raise ValueError("source_id must be an opaque string")
    return value


def _sha256(value, field):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("Invalid " + field)
    return value


def _corpus_scope(corpus_id, split):
    if not isinstance(corpus_id, str) or not CORPUS_ID.fullmatch(corpus_id):
        raise ValueError("Invalid corpus ID")
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError("Unknown split")


def retriever_config():
    """Exact executable recipe; config is part of both corpus and manifest IDs."""
    return {
        "name": "sqlite-fts5-bm25", "sqlite_version": sqlite3.sqlite_version,
        "tokenizer": "unicode61", "query_recipe": "quoted tokens OR, max32",
    }


def _validate_public_document(doc):
    if not isinstance(doc, dict) or set(doc) != PUBLIC_DOC_FIELDS:
        raise ValueError("Unexpected/private or missing public document field")
    _check_json(doc)
    _source_id(doc["source_id"])
    _sha256(doc["asset_sha256"], "asset_sha256")
    _sha256(doc["text_sha256"], "text_sha256")
    url = doc["origin_url"]
    if not isinstance(url, str):
        raise ValueError("Public origin URL required")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError("Invalid public origin URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Public origin URL required")
    kind = doc["text_kind"]
    if not isinstance(kind, str) or kind not in TEXT_KINDS:
        raise ValueError("Disallowed derived text kind")
    media = doc["media_type"]
    symbolic = {"audio_video", "av", "audio", "video", "image", "text", "webpage"}
    if (not isinstance(media, str) or not media.strip() or
            (media not in symbolic and not re.fullmatch(r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+", media))):
        raise ValueError("media_type must be a supported symbolic type or MIME string")
    duration = doc["duration_s"]
    timed = media in {"audio_video", "av", "audio", "video"} or media.lower().startswith(("audio/", "video/"))
    if duration is None:
        if timed:
            raise ValueError("Audio/video requires a finite positive duration_s")
    elif (isinstance(duration, bool) or not isinstance(duration, (int, float)) or
          not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration_s must be finite and positive, not bool")
    meta = doc["public_metadata"]
    if not isinstance(meta, dict) or set(meta) - META_FIELDS:
        raise ValueError("Unexpected/private metadata key")
    for key, value in meta.items():
        if not isinstance(value, str) and not (
                key == "categories" and isinstance(value, list) and all(isinstance(item, str) for item in value)):
            raise ValueError("Metadata must contain public text/categories")
    text = doc["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty public content required")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != doc["text_sha256"]:
        raise ValueError("Document text hash mismatch")
    return doc


def _public_document(doc):
    if not isinstance(doc, dict) or set(doc) - DOC_FIELDS or not (DOC_FIELDS - {"duration_s"}) <= set(doc):
        raise ValueError("Unexpected/private or missing document field")
    _check_json(doc)
    public = {key: value for key, value in doc.items() if key != "asset_path"}
    public.setdefault("duration_s", None)
    if not isinstance(public.get("text"), str):
        raise ValueError("Nonempty public content required")
    public["text_sha256"] = hashlib.sha256(public["text"].encode("utf-8")).hexdigest()
    _validate_public_document(public)
    asset_path = doc["asset_path"]
    if not isinstance(asset_path, str) or not asset_path.strip():
        raise ValueError("asset_path must be a nonempty internal path")
    asset = Path(asset_path)
    if not asset.is_file():
        raise FileNotFoundError(asset)
    if file_sha256(asset) != doc["asset_sha256"]:
        raise ValueError("Asset hash mismatch")
    # Roundtrip guarantees that caller mutations cannot change frozen content.
    return strict_json_loads(canonical(public))


def manifest_digest(manifest):
    """SHA256 of canonical full manifest excluding only manifest_sha256.

    This is a consistency digest, not an author signature. The episode adapter
    must pin the initially accepted digest and compare it before every action.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _corpus_digest(corpus_id, split, docs, config):
    payload = {
        "schema_version": SCHEMA_VERSION, "corpus_id": corpus_id, "split": split,
        "documents": docs, "retriever": config,
    }
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValueError("Unexpected/private or missing manifest field")
    _check_json(manifest)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported frozen corpus schema")
    _corpus_scope(manifest["corpus_id"], manifest["split"])
    ids = manifest["source_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("Manifest requires source IDs")
    for source_id in ids:
        _source_id(source_id)
    if ids != sorted(set(ids)):
        raise ValueError("Manifest IDs must be unique and sorted")
    for field in ("corpus_hash", "index_sha256", "documents_sha256", "manifest_sha256"):
        _sha256(manifest[field], field)
    config = manifest["retriever"]
    if not isinstance(config, dict) or config != retriever_config():
        raise ValueError("Unexpected/private or unsupported retriever recipe/version")
    if manifest_digest(manifest) != manifest["manifest_sha256"]:
        raise ValueError("Frozen manifest identity changed")
    return manifest


def open_index(root):
    """Open an already-validated SQLite asset read-only, with URI path encoding."""
    return sqlite3.connect((Path(root) / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True)


def _validate_database(root, docs):
    expected = {doc["source_id"]: doc for doc in docs}
    try:
        with open_index(root) as connection:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("Frozen SQLite integrity check failed")
            rows = connection.execute("SELECT source_id, payload FROM docs ORDER BY source_id").fetchall()
            if len(rows) != len(docs):
                raise ValueError("SQLite/document source coverage differs")
            seen = set()
            for source_id, payload in rows:
                _source_id(source_id)
                document = _validate_public_document(strict_json_loads(payload))
                if (source_id in seen or source_id != document["source_id"] or source_id not in expected or
                        canonical(document) != canonical(expected[source_id])):
                    raise ValueError("SQLite payload/primary ID differs from frozen JSONL")
                seen.add(source_id)
            fts_rows = connection.execute("SELECT source_id, content FROM search ORDER BY source_id, rowid").fetchall()
            if fts_rows != [(doc["source_id"], doc["text"]) for doc in docs]:
                raise ValueError("FTS source IDs/text/multiplicity differ from frozen documents")
    except sqlite3.Error as exc:
        raise ValueError("Malformed frozen SQLite index") from exc


def build_corpus(documents: Iterable[dict], out_dir, corpus_id: str, split: str) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(out)
    _corpus_scope(corpus_id, split)
    docs = sorted((_public_document(document) for document in documents), key=lambda document: document["source_id"])
    if not docs or len({document["source_id"] for document in docs}) != len(docs):
        raise ValueError("Empty corpus or duplicate source IDs")
    config = retriever_config()
    corpus_hash = _corpus_digest(corpus_id, split, docs, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="corpus-build-", dir=out.parent) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        database = staging / "index.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE docs(source_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute('CREATE VIRTUAL TABLE search USING fts5(source_id UNINDEXED, content, tokenize="unicode61")')
            for document in docs:
                connection.execute("INSERT INTO docs VALUES(?,?)", (document["source_id"], canonical(document)))
                connection.execute("INSERT INTO search VALUES(?,?)", (document["source_id"], document["text"]))
        (staging / "documents.jsonl").write_text("".join(canonical(document) + "\n" for document in docs), encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION, "corpus_id": corpus_id, "split": split,
            "source_ids": [document["source_id"] for document in docs],
            "corpus_hash": corpus_hash, "index_sha256": file_sha256(database),
            "documents_sha256": file_sha256(staging / "documents.jsonl"), "retriever": config,
        }
        manifest["manifest_sha256"] = manifest_digest(manifest)
        (staging / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
        # Prove cross-representation consistency before publishing the directory.
        load_frozen(staging)
        staging.rename(out)
    return manifest


def load_frozen(index_dir) -> tuple[Path, dict]:
    """Validate full manifest, byte hashes, public documents and SQLite/FTS state."""
    root = Path(index_dir)
    manifest = _validate_manifest(strict_json_loads((root / "manifest.json").read_text(encoding="utf-8")))
    if (file_sha256(root / "index.sqlite") != manifest["index_sha256"] or
            file_sha256(root / "documents.jsonl") != manifest["documents_sha256"]):
        raise ValueError("Frozen index/content changed")
    lines = (root / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Frozen documents require nonempty JSONL records")
    docs = [_validate_public_document(strict_json_loads(line)) for line in lines]
    if [document["source_id"] for document in docs] != manifest["source_ids"]:
        raise ValueError("Frozen document source IDs differ")
    if _corpus_digest(manifest["corpus_id"], manifest["split"], docs, manifest["retriever"]) != manifest["corpus_hash"]:
        raise ValueError("Corpus content/config identity mismatch")
    _validate_database(root, docs)
    return root, manifest


def validate_scope(manifest: dict, scope: dict) -> set[str]:
    _validate_manifest(manifest)
    if not isinstance(scope, dict) or set(scope) != {"corpus_id", "split", "source_ids"}:
        raise ValueError("Scope must come from fixed episode config")
    _check_json(scope)
    _corpus_scope(scope["corpus_id"], scope["split"])
    if scope["corpus_id"] != manifest["corpus_id"] or scope["split"] != manifest["split"]:
        raise ValueError("Wrong episode corpus/split")
    ids = scope["source_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("Scope requires source IDs")
    for source_id in ids:
        _source_id(source_id)
    if len(set(ids)) != len(ids) or not set(ids) <= set(manifest["source_ids"]):
        raise ValueError("Unknown or duplicate source IDs in scope")
    return set(ids)


def visible_corpus(index_dir, scope: dict) -> dict:
    root, manifest = load_frozen(index_dir)
    allowed = validate_scope(manifest, scope)
    with open_index(root) as connection:
        docs = [strict_json_loads(row[0]) for row in connection.execute("SELECT payload FROM docs ORDER BY source_id")]
    return {
        "corpus_id": manifest["corpus_id"], "split": manifest["split"],
        "sources": {document["source_id"]: document for document in docs if document["source_id"] in allowed},
    }

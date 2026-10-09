"""Independent review regressions for frozen identity and public episode scope.

All assets/corpora and mutations live in TemporaryDirectory. These tests express
the required rejection behavior and intentionally fail on the original Task3
implementation. No production source is mutated.
"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research.av_search.corpus import build_corpus, load_frozen, validate_scope, visible_corpus
from research.av_search.retrieval import get_source, search_sources


class FrozenSearchReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="frozen-search-independent-review-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.documents = []
        for number, body in enumerate(("Visible source describes a hammer.",
                                       "An out-of-scope source describes rain."), 1):
            path = self.root / ("source" + str(number) + ".txt")
            path.write_text(body)
            self.documents.append({
                "source_id": "src_" + str(number) * 32,
                "origin_url": "https://example.test/source" + str(number),
                "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "asset_path": str(path), "text": body, "text_kind": "web_body",
                "public_metadata": {"title": "Actual synthetic public title"},
                "media_type": "text/html", "duration_s": None})
        self.index = self.root / "frozen"
        self.manifest = build_corpus(self.documents, self.index, "review-v1", "exploration")
        self.scope = {"corpus_id": "review-v1", "split": "exploration",
                      "source_ids": [self.documents[0]["source_id"]]}

    def rewrite_manifest(self, mutate, reseal=False):
        path = self.index / "manifest.json"
        manifest = json.loads(path.read_text())
        mutate(manifest)
        if reseal:
            payload = dict(manifest)
            payload.pop("manifest_sha256", None)
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode()
            manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        path.write_text(json.dumps(manifest))

    def refresh_index_hash(self):
        self.rewrite_manifest(lambda manifest: manifest.update(
            index_sha256=hashlib.sha256((self.index / "index.sqlite").read_bytes()).hexdigest()),
            reseal=True)

    def test_manifest_private_retriever_fields_must_not_reach_receipt(self):
        self.rewrite_manifest(lambda manifest: manifest["retriever"].update(
            private_gold={"expected_decision": "same_event"}))
        with self.assertRaises(ValueError):
            search_sources(self.index, "hammer", self.scope)

    def test_manifest_changed_retriever_identity_must_fail_freeze(self):
        self.rewrite_manifest(lambda manifest: manifest["retriever"].update(
            query_recipe="Changed ranking recipe"))
        with self.assertRaises(ValueError):
            load_frozen(self.index)

    def test_database_payload_must_match_frozen_jsonl_even_after_checksum_refresh(self):
        with sqlite3.connect(self.index / "index.sqlite") as connection:
            row = connection.execute("SELECT payload FROM docs WHERE source_id=?",
                                     (self.documents[0]["source_id"],)).fetchone()
            payload = json.loads(row[0])
            payload["private_gold"] = {"expected_decision": "same_event"}
            payload["asset_path"] = "/private/gold/asset.mp4"
            connection.execute("UPDATE docs SET payload=? WHERE source_id=?",
                               (json.dumps(payload), self.documents[0]["source_id"]))
        self.refresh_index_hash()
        with self.assertRaises(ValueError):
            get_source(self.index, self.documents[0]["source_id"], self.scope)

    def test_fts_content_must_match_document_text_even_after_checksum_refresh(self):
        with sqlite3.connect(self.index / "index.sqlite") as connection:
            connection.execute("UPDATE search SET content=? WHERE source_id=?",
                               ("phantomterm", self.documents[0]["source_id"]))
        self.refresh_index_hash()
        with self.assertRaises(ValueError):
            search_sources(self.index, "phantomterm", self.scope)

    def test_nonfinite_negative_and_boolean_media_durations_must_fail(self):
        for number, duration in enumerate((float("nan"), float("inf"), -1, True)):
            with self.subTest(duration=duration):
                document = copy.deepcopy(self.documents[0])
                document.update(media_type="video/mp4", duration_s=duration)
                with self.assertRaises(ValueError):
                    build_corpus([document], self.root / ("invalid-duration-" + str(number)),
                                 "review-v1", "exploration")

    def test_invalid_media_type_must_fail(self):
        document = copy.deepcopy(self.documents[0])
        document["media_type"] = {"gold_answer": "same_event"}
        with self.assertRaises(ValueError):
            build_corpus([document], self.root / "invalid-media-type", "review-v1", "exploration")

    def test_malformed_source_id_must_raise_contract_value_error(self):
        document = copy.deepcopy(self.documents[0])
        document["source_id"] = None
        with self.assertRaises(ValueError):
            build_corpus([document], self.root / "invalid-source-id", "review-v1", "exploration")

    def test_unhashable_scope_member_must_raise_contract_value_error(self):
        scope = copy.deepcopy(self.scope)
        scope["source_ids"] = [{"source_id": self.documents[0]["source_id"]}]
        with self.assertRaises(ValueError):
            validate_scope(self.manifest, scope)

    def test_scope_subset_never_returns_or_reads_disallowed_source(self):
        source_id = self.documents[1]["source_id"]
        self.assertEqual(search_sources(self.index, "rain", self.scope)["results"], [])
        with self.assertRaises(ValueError):
            get_source(self.index, source_id, self.scope)
        self.assertEqual(set(visible_corpus(self.index, self.scope)["sources"]),
                         set(self.scope["source_ids"]))

    def test_manifest_digest_covers_config_index_and_document_identity(self):
        self.assertIn("manifest_sha256", self.manifest)
        self.assertEqual(self.manifest["schema_version"], "av-corpus-0.2")
        payload = dict(self.manifest)
        expected = payload.pop("manifest_sha256")
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False).encode()
        self.assertEqual(expected, hashlib.sha256(encoded).hexdigest())
        receipt = search_sources(self.index, "hammer", self.scope)
        self.assertEqual(receipt["manifest_sha256"], expected)

    def test_resealed_manifest_still_rejects_private_retriever_config(self):
        self.rewrite_manifest(lambda manifest: manifest["retriever"].update(
            private_gold={"expected_decision": "same_event"}), reseal=True)
        with self.assertRaises(ValueError):
            load_frozen(self.index)

    def test_database_primary_source_id_must_match_payload_source_id(self):
        with sqlite3.connect(self.index / "index.sqlite") as connection:
            row = connection.execute("SELECT payload FROM docs WHERE source_id=?",
                                     (self.documents[0]["source_id"],)).fetchone()
            payload = json.loads(row[0])
            payload["source_id"] = self.documents[1]["source_id"]
            connection.execute("UPDATE docs SET payload=? WHERE source_id=?",
                               (json.dumps(payload), self.documents[0]["source_id"]))
        self.refresh_index_hash()
        with self.assertRaises(ValueError):
            get_source(self.index, self.documents[0]["source_id"], self.scope)

    def test_fts_duplicate_rows_must_fail_with_refreshed_checksum_and_identity(self):
        with sqlite3.connect(self.index / "index.sqlite") as connection:
            connection.execute("INSERT INTO search(source_id, content) VALUES (?, ?)",
                               (self.documents[0]["source_id"], self.documents[0]["text"]))
        self.refresh_index_hash()
        with self.assertRaises(ValueError):
            search_sources(self.index, "hammer", self.scope)

    def test_missing_manifest_schema_must_fail(self):
        self.rewrite_manifest(lambda manifest: manifest.pop("schema_version"), reseal=True)
        with self.assertRaises(ValueError):
            load_frozen(self.index)

    def test_duplicate_manifest_json_keys_must_fail(self):
        path = self.index / "manifest.json"
        text = path.read_text()
        path.write_text(text[:-1] + ', "corpus_id": "review-v1"}')
        with self.assertRaises(ValueError):
            load_frozen(self.index)



if __name__ == "__main__":
    unittest.main()

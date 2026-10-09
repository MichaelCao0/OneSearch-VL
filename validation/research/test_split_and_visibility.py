"""Executable public/private and event-lineage contracts; all fixtures are synthetic."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest

FIXTURES = Path(__file__).with_name("fixtures_lineage")
SOURCES = ["src_" + str(i) * 32 for i in range(1, 10)]


class ContractTests(unittest.TestCase):
    def api(self, module, name):
        try:
            imported = importlib.import_module("research.av_data." + module)
        except ModuleNotFoundError as exc:
            self.fail("Missing implementation module: " + str(exc))
        function = getattr(imported, name, None)
        self.assertTrue(callable(function), "Missing contract API: " + module + "." + name)
        return function

    def task(self):
        return json.loads(json.dumps({
            "schema_version": "1.0", "created_by": "synthetic-fixture", "revision": "v1",
            "task_id": "task_demo", "question": "What does each recording establish?",
            "target_questions": [
                {"claim_id": "c1", "question": "At what stage is the sound audible?"},
                {"claim_id": "c2", "question": "Do the clips show the same event?"},
                {"claim_id": "c3", "question": "What remains uncertain?"}],
            "initial_sources": [SOURCES[0]], "corpus_id": "corpus_demo_v1",
            "split": "exploration", "budget_recipe": {"max_actions": 6, "credits": 12}}))

    def registry(self):
        payload = (FIXTURES / "synthetic_page.txt").read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        return {SOURCES[0]: {
            "source_id": SOURCES[0], "asset_sha256": sha, "duration_s": 10,
            "media_type": "av", "corpus_id": "corpus_demo_v1", "split": "exploration",
            "public_metadata": {"title": "Public title mentions a recording and applause"}}}

    def gold(self):
        sha = self.registry()[SOURCES[0]]["asset_sha256"]
        return json.loads(json.dumps({
            "schema_version": "1.0", "created_by": "reviewer_fixture", "revision": "v1",
            "task_id": "task_demo", "claim_id": "c1",
            "proposition": "The sound occurs during the visible contact.",
            "expected_decision": "unknown", "status": "undetermined",
            "qualifiers": {"causality": "not established"},
            "valid_evidence_sets": [[{"source_id": SOURCES[0], "asset_sha256": sha,
                                     "interval": [1.0, 2.0]}]],
            "required_bindings": [], "missing_evidence": ["No identified sound source."],
            "scope": "within_snapshot"}))

    def records(self):
        return json.loads((FIXTURES / "lineage_records.json").read_text())["records"]

    def test_public_task_roundtrip_and_actor_title(self):
        task = self.task()
        self.api("tasks", "validate_task_public")(task, self.registry(), task["corpus_id"], "exploration")
        exported = self.api("audit", "export_actor_task")(task, self.registry())
        self.assertEqual(json.loads(json.dumps(exported)), task)
        exported["question"] = "Changed copy"
        self.assertNotEqual(task["question"], exported["question"])

    def test_public_task_rejects_private_fields_in_nested_budget(self):
        validate = self.api("tasks", "validate_task_public")
        for key in ("gold", "valid_evidence_sets", "required_bindings", "adjudication",
                    "review_summary", "future_observations", "answer", "private_path"):
            with self.subTest(key=key):
                task = self.task()
                task["budget_recipe"]["nested"] = {key: "leak"}
                with self.assertRaises(ValueError):
                    validate(task, self.registry(), task["corpus_id"], "exploration")

    def test_actor_export_rejects_private_instead_of_silently_stripping(self):
        export = self.api("audit", "export_actor_task")
        task = self.task()
        task["gold"] = {"expected_decision": "same_event"}
        with self.assertRaises(ValueError):
            export(task, self.registry())

    def test_actor_export_rejects_wrong_episode_corpus_and_split(self):
        export = self.api("audit", "export_actor_task")
        for field, value in (("corpus_id", "other_corpus"), ("split", "test")):
            with self.subTest(field=field):
                registry = self.registry()
                registry[SOURCES[0]][field] = value
                with self.assertRaises(ValueError):
                    export(self.task(), registry)

    def test_actor_export_rejects_private_source_metadata_and_paths(self):
        export = self.api("audit", "export_actor_task")
        registry = self.registry()
        registry[SOURCES[0]]["public_metadata"]["lineage_gold"] = {"event_id": "evt_A"}
        with self.assertRaises(ValueError):
            export(self.task(), registry)
        registry = self.registry()
        registry[SOURCES[0]]["asset_path"] = "/private/gold/replay.mp4"
        with self.assertRaises(ValueError):
            export(self.task(), registry)

    def test_task_rejects_foreign_sources_and_encoded_opaque_ids(self):
        validate = self.api("tasks", "validate_task_public")
        task = self.task()
        task["initial_sources"] = [SOURCES[1]]
        with self.assertRaises(ValueError):
            validate(task, self.registry(), task["corpus_id"], "exploration")
        task["initial_sources"] = ["src_same_event_answer"]
        with self.assertRaises(ValueError):
            validate(task, ["src_same_event_answer"], task["corpus_id"], "exploration")
        task = self.task()
        task["budget_recipe"]["note"] = SOURCES[1]
        with self.assertRaises(ValueError):
            validate(task, self.registry(), task["corpus_id"], "exploration")

    def test_task_rejects_wrong_split_and_duplicate_claims(self):
        validate = self.api("tasks", "validate_task_public")
        task = self.task()
        with self.assertRaises(ValueError):
            validate(task, self.registry(), task["corpus_id"], "test")
        task["target_questions"][1]["claim_id"] = "c1"
        with self.assertRaises(ValueError):
            validate(task, self.registry(), task["corpus_id"], "exploration")

    def test_claim_gold_validates_real_sha_and_and_or_packages(self):
        gold = self.gold()
        gold["valid_evidence_sets"].append(copy.deepcopy(gold["valid_evidence_sets"][0]))
        validated = self.api("gold", "validate_claim_gold")(gold, "task_demo", self.registry())
        self.assertEqual(validated, gold)
        declared = json.loads((FIXTURES / "fixture_hashes.json").read_text())
        actual = hashlib.sha256((FIXTURES / "synthetic_page.txt").read_bytes()).hexdigest()
        self.assertEqual(declared["synthetic_page.txt"], actual)

    def test_gold_rejects_wrong_hash_source_interval_and_task(self):
        validate = self.api("gold", "validate_claim_gold")
        mutations = [
            lambda g: g.update(task_id="other"),
            lambda g: g["valid_evidence_sets"][0][0].update(source_id=SOURCES[1]),
            lambda g: g["valid_evidence_sets"][0][0].update(asset_sha256="0" * 64),
            lambda g: g["valid_evidence_sets"][0][0].update(interval=[2, 1]),
            lambda g: g["valid_evidence_sets"][0][0].update(interval=[0, 11]),
            lambda g: g["valid_evidence_sets"][0][0].update(interval=[0, float("nan")])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                gold = self.gold()
                mutation(gold)
                with self.assertRaises(ValueError):
                    validate(gold, "task_demo", self.registry())

    def test_unknown_requires_snapshot_scope_and_missing_evidence(self):
        validate = self.api("gold", "validate_claim_gold")
        for field, value in (("scope", "universal"), ("missing_evidence", []),
                             ("valid_evidence_sets", [[]])):
            with self.subTest(field=field):
                gold = self.gold()
                gold[field] = value
                with self.assertRaises(ValueError):
                    validate(gold, "task_demo", self.registry())

    def test_lineage_relation_unknown_is_explicit_and_auditable(self):
        validate = self.api("gold", "validate_lineage")
        registry = self.registry()
        registry[SOURCES[1]] = dict(registry[SOURCES[0]], source_id=SOURCES[1])
        relation = {
            "schema_version": "1.0", "created_by": "reviewer_fixture", "revision": "v1",
            "source_ids": [SOURCES[0], SOURCES[1]], "relation_type": "independent_observation",
            "relation_status": "unknown", "basis_refs": [], "adjudication": "not established"}
        self.assertEqual(validate(relation, registry), relation)
        for bad in ("", "independent_by_default"):
            relation["relation_status"] = bad
            with self.assertRaises(ValueError):
                validate(relation, registry)
        relation["relation_status"] = "same"
        relation["relation_type"] = "recording"
        with self.assertRaises(ValueError):
            validate(relation, registry)

    def test_components_merge_crops_shared_audio_event_and_near_duplicates(self):
        components = self.api("split", "build_lineage_components")(self.records())
        groups = [set(c["source_ids"]) for c in components]
        self.assertIn(set(SOURCES[:3]), groups)
        self.assertIn(set(SOURCES[3:5]), groups)

    def test_generic_knowledge_url_and_uploader_do_not_merge_events(self):
        components = self.api("split", "build_lineage_components")(self.records())
        groups = [set(c["source_ids"]) for c in components]
        for source_id in SOURCES[5:]:
            self.assertIn({source_id}, groups)

    def test_unknown_recording_link_is_conservative_but_independence_is_not_assumed(self):
        records = self.records()[5:8]
        records[0]["lineage_links"] = [{
            "source_id": SOURCES[6], "relation_type": "event", "relation_status": "unknown"}]
        components = self.api("split", "build_lineage_components")(records)
        self.assertIn(set(SOURCES[5:7]), [set(c["source_ids"]) for c in components])
        self.assertIn({SOURCES[7]}, [set(c["source_ids"]) for c in components])

    def test_split_is_frozen_deterministic_and_exploration_never_test(self):
        build = self.api("split", "build_lineage_components")
        assign = self.api("split", "assign_event_splits")
        components = build(self.records())
        ratios = {"train": 0.5, "dev": 0.25, "test": 0.25}
        manifest = assign(components, 19, ratios)
        self.assertEqual(manifest, assign(list(reversed(components)), 19, ratios))
        self.assertTrue(manifest["frozen"])
        self.assertEqual(manifest["source_splits"][SOURCES[0]], "exploration")
        self.assertEqual(manifest["source_splits"][SOURCES[0]], manifest["source_splits"][SOURCES[2]])
        self.api("audit", "audit_split_manifest")(components, json.loads(json.dumps(manifest)))

    def test_audit_rejects_same_cluster_cross_split_and_exploration_test(self):
        components = self.api("split", "build_lineage_components")(self.records())
        manifest = self.api("split", "assign_event_splits")(components, 19, {"train": 1})
        audit = self.api("audit", "audit_split_manifest")
        manifest["source_splits"][SOURCES[1]] = "test"
        with self.assertRaises(ValueError):
            audit(components, manifest)
        manifest = self.api("split", "assign_event_splits")(components, 19, {"train": 1})
        for source in SOURCES[:3]:
            manifest["source_splits"][source] = "test"
        with self.assertRaises(ValueError):
            audit(components, manifest)

    def test_split_rejects_dangling_duplicate_and_invalid_ratio_records(self):
        build = self.api("split", "build_lineage_components")
        records = self.records()
        records[0]["near_duplicate_ids"] = ["src_" + "a" * 32]
        with self.assertRaises(ValueError):
            build(records)
        with self.assertRaises(ValueError):
            build(self.records() + [self.records()[0]])
        components = build(self.records())
        assign = self.api("split", "assign_event_splits")
        for ratios in ({"train": -1, "test": 2}, {"test": float("nan")}, {"train": 0.3},
                       {"invalid": 1}, {}):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                assign(components, 19, ratios)

    def test_actor_rejects_evidence_and_camelcase_private_keys(self):
        export = self.api("audit", "export_actor_task")
        for key in ("evidence", "EvidenceSets", "hiddenBindings", "PrivateNotes", "answer_key"):
            with self.subTest(key=key):
                registry = self.registry()
                registry[SOURCES[0]]["public_metadata"][key] = "not public"
                with self.assertRaises(ValueError):
                    export(self.task(), registry)

    def test_gold_web_quote_must_match_source_bytes_and_exact_characters(self):
        validate = self.api("gold", "validate_claim_gold")
        body = (FIXTURES / "synthetic_page.txt").read_text()
        registry = self.registry()
        registry[SOURCES[0]].update(media_type="webpage", duration_s=None, body_text=body)
        gold = self.gold()
        gold["valid_evidence_sets"] = [[{
            "source_id": SOURCES[0], "asset_sha256": registry[SOURCES[0]]["asset_sha256"],
            "quote": {"start_char": 0, "end_char": 9, "text": body[:9]}}]]
        self.assertEqual(validate(gold, "task_demo", registry), gold)
        gold["valid_evidence_sets"][0][0]["quote"]["text"] = "fabricated"
        with self.assertRaises(ValueError):
            validate(gold, "task_demo", registry)

    def test_real_audio_crop_hashes_differ_but_stay_in_recording_component(self):
        hashes = json.loads((FIXTURES / "fixture_hashes.json").read_text())
        for filename in ("synthetic_original.wav", "synthetic_crop.wav"):
            self.assertEqual(hashes[filename], hashlib.sha256((FIXTURES / filename).read_bytes()).hexdigest())
        self.assertNotEqual(hashes["synthetic_original.wav"], hashes["synthetic_crop.wav"])
        components = self.api("split", "build_lineage_components")(self.records())
        self.assertTrue(any(set(SOURCES[:2]) <= set(c["source_ids"]) for c in components))

    def test_malformed_task_initial_sources_are_rejected_with_value_error(self):
        validate = self.api("tasks", "validate_task_public")
        task = self.task()
        task["initial_sources"] = [{"source_id": SOURCES[0]}]
        with self.assertRaises(ValueError):
            validate(task, self.registry(), task["corpus_id"], "exploration")


    def test_actor_export_accepts_frozen_corpus_scope_envelope(self):
        export = self.api("audit", "export_actor_task")
        source = self.registry()[SOURCES[0]]
        source.pop("corpus_id")
        source.pop("split")
        source.update(text="Public transcript", text_kind="asr", text_sha256="a" * 64)
        visible = {"corpus_id": "corpus_demo_v1", "split": "exploration",
                   "sources": {SOURCES[0]: source}}
        try:
            exported = export(self.task(), visible)
        except ValueError as exc:
            self.fail("Frozen corpus envelope is not yet supported: " + str(exc))
        self.assertEqual(exported, self.task())
        visible["corpus_id"] = "other_corpus"
        with self.assertRaises(ValueError):
            export(self.task(), visible)

    def test_actor_export_rejects_unclassified_source_fields(self):
        export = self.api("audit", "export_actor_task")
        registry = self.registry()
        registry[SOURCES[0]]["evaluation_notes"] = "Private reviewer inference"
        with self.assertRaises(ValueError):
            export(self.task(), registry)

    def test_split_supports_corpus_ood_name(self):
        components = self.api("split", "build_lineage_components")(self.records())
        try:
            manifest = self.api("split", "assign_event_splits")(components, 19, {"ood": 1})
        except ValueError as exc:
            self.fail("Corpus OOD split name is not yet supported: " + str(exc))
        self.assertEqual(manifest["source_splits"][SOURCES[5]], "ood")
        self.api("audit", "audit_split_manifest")(components, manifest)



if __name__ == "__main__":
    unittest.main()

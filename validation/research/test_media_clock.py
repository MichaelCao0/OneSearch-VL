"""Clock contract tests use real ffmpeg, pixels and samples; no clock mocks."""
import copy
import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path

from validation.research.media_fixtures import make_audio_only, make_offset_vfr, decoded_markers, probe


class MediaClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="av-clock-")
        cls.root = Path(cls.tmp.name)
        cls.media = cls.root / "source ; literal.mkv"
        make_offset_vfr(cls.media)
        cls.audio = cls.root / "speech.wav"
        make_audio_only(cls.audio)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        try:
            self.ingest = importlib.import_module("research.av_data.ingest").ingest_media
            self.clip = importlib.import_module("research.av_data.media_clock").clip_media
            validate = importlib.import_module("research.av_data.validate")
            self.validate_source = validate.validate_source
            self.validate_derived = validate.validate_derived
        except (ImportError, AttributeError) as exc:
            self.fail(f"Required source-clock implementation is missing: {exc}")

    def source(self):
        return self.ingest(self.media, "https://example.org/public-recording",
                           self.root / "assets", {"title": "Actual public title"},
                           rights_basis="synthetic_fixture", redistribution_scope="full_asset")

    def test_ingest_preserves_original_bytes_offsets_and_vfr(self):
        # Catches recoding, discarding audio, timestamp zeroing, and treating grid fps as source PTS.
        record = self.source()
        self.assertEqual(Path(record["asset_path"]).read_bytes(), self.media.read_bytes())
        self.assertEqual(record["asset_sha256"], hashlib.sha256(self.media.read_bytes()).hexdigest())
        self.assertEqual(record["asset_bytes"], self.media.stat().st_size)
        self.assertTrue(record["source_id"].startswith("src_"))
        self.assertEqual(record["public_metadata"]["title"], "Actual public title")
        self.assertEqual(record["rights_basis"], "synthetic_fixture")
        self.assertEqual(record["timestamp_origin"]["container_start_time_s"], 5.0)
        self.assertAlmostEqual(record["duration_s"], 4.0, delta=.07)
        streams = {s["codec_type"]: s for s in record["stream_specs"]}
        self.assertAlmostEqual(streams["audio"]["source_offset_s"], .4, delta=.005)
        self.assertEqual(streams["audio"]["start_time"], "5.400000")
        self.assertTrue(streams["video"]["vfr"])
        self.assertGreater(len(streams["video"]["frame_pts"]), 10)
        self.assertGreater(len(streams["audio"]["sample_mapping"]), 10)
        self.assertEqual(self.validate_source(record), [])
        json.dumps(record, allow_nan=False)
        print("SOURCE_SAMPLE", json.dumps({k: v for k, v in record.items() if k != "stream_specs"}, sort_keys=True))

    def test_clip_preserves_delayed_audio_and_decoded_sync(self):
        # Catches per-stream STARTPTS reset, dropping AV, or metadata-only clock verification.
        source = self.source()
        derived = self.clip(source, 0.0, 2.2, self.root / "clips")
        visual, audio, data = decoded_markers(derived["output_path"])
        self.assertAlmostEqual(visual, 1.6, delta=.06)
        self.assertAlmostEqual(audio, 1.6, delta=.02)
        self.assertLessEqual(abs(visual - audio), .1)
        streams = {s["codec_type"]: s for s in data["streams"]}
        self.assertAlmostEqual(float(streams["audio"]["start_time"]), .4, delta=.005)
        self.assertEqual(derived["original_interval"], {"start_s": 0.0, "end_s": 2.2})
        self.assertEqual(derived["mapping"]["output_to_source"], {"scale": 1.0, "offset_s": 0.0})
        self.assertEqual(derived["parent_sha256"], source["asset_sha256"])
        self.assertEqual(self.validate_derived(derived, source), [])
        print("SYNC_MEASUREMENT delayed_clip", json.dumps({"visual_s": visual, "audio_s": audio, "error_s": abs(visual-audio)}))

    def test_clip_nonzero_interval_maps_to_original_presentation_clock(self):
        # Catches trimming container-absolute seconds as if they were source-local seconds.
        source = self.source()
        derived = self.clip(source, .63, 2.2, self.root / "clips")
        visual, audio, _ = decoded_markers(derived["output_path"])
        self.assertAlmostEqual(visual + .63, 1.6, delta=.001)
        self.assertAlmostEqual(audio + .63, 1.6, delta=.02)
        self.assertLessEqual(abs(visual - audio), .1)
        self.assertEqual(derived["mapping"]["output_to_source"], {"scale": 1.0, "offset_s": .63})
        self.assertEqual(self.validate_derived(derived, source), [])
        print("SYNC_MEASUREMENT nonzero_clip", json.dumps({"visual_s": visual, "audio_s": audio, "error_s": abs(visual-audio)}))

    def test_clip_rejects_invalid_intervals_and_missing_or_changed_assets(self):
        # Catches accepting NaN/empty/out-of-range intervals and unverified original bytes.
        source = self.source()
        for start, end in [(-.01, 1), (1, 1), (2, 1), (0, 8), (float("nan"), 1), (10**400, 10**401)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.clip(source, start, end, self.root / "bad-clips")
        bad = copy.deepcopy(source)
        bad["asset_path"] = str(self.root / "absent.mkv")
        with self.assertRaises(ValueError):
            self.clip(bad, 0, 1, self.root / "bad-clips")
        changed = self.root / "changed.mkv"
        changed.write_bytes(self.media.read_bytes() + b"changed")
        bad["asset_path"] = str(changed)
        with self.assertRaises(ValueError):
            self.clip(bad, 0, 1, self.root / "bad-clips")

    def test_validators_detect_forged_mapping_parent_and_output(self):
        # Catches an empty-success validator and mapping or output-byte tampering.
        source = self.source()
        derived = self.clip(source, 0, 2.2, self.root / "clips")
        bad = copy.deepcopy(derived)
        bad["mapping"]["output_to_source"]["offset_s"] = 7
        self.assertTrue(self.validate_derived(bad, source))
        bad = copy.deepcopy(derived)
        bad["parent_sha256"] = "0" * 64
        self.assertTrue(self.validate_derived(bad, source))
        output = Path(derived["output_path"])
        output.write_bytes(output.read_bytes() + b"tampered")
        self.assertTrue(self.validate_derived(derived, source))
        missing = copy.deepcopy(source)
        del missing["timestamp_origin"]
        self.assertTrue(self.validate_source(missing))

    def test_source_validator_rejects_self_consistent_forged_origin(self):
        # Catches a clock validator that trusts mutable manifest fields over original bytes.
        source = self.source()
        forged = copy.deepcopy(source)
        forged["timestamp_origin"]["container_start_time_s"] = 6.0
        forged["timestamp_origin"]["pts_to_source"]["offset_s"] = -6.0
        forged["duration_s"] = 3.0
        for stream in forged["stream_specs"]:
            stream["source_offset_s"] -= 1.0
            for frame in stream["frame_pts"]:
                frame["source_time_s"] -= 1.0
            for sample in stream["sample_mapping"]:
                sample["source_start_s"] -= 1.0
        self.assertTrue(self.validate_source(forged, verify_hash=True))

    def test_source_validator_rejects_bad_audio_sampling_map(self):
        # Catches accepting a sample clock that can place decoded samples at arbitrary times.
        source = self.source()
        forged = copy.deepcopy(source)
        audio = next(s for s in forged["stream_specs"] if s["codec_type"] == "audio")
        audio["sample_mapping"][0]["seconds_per_sample"] = 1.0
        self.assertTrue(self.validate_source(forged, verify_hash=False))

    def test_validators_fail_closed_on_malformed_json_structures(self):
        # Catches crashes or accidental acceptance when a manifest has corrupted nested fields.
        source = self.source()
        for value in ([None], [{"index": [], "codec_type": "video", "time_base": "1/1000",
                                "frame_pts": [None], "source_offset_s": 0}], "streams"):
            bad = copy.deepcopy(source)
            bad["stream_specs"] = value
            with self.subTest(value=value):
                self.assertTrue(self.validate_source(bad, verify_hash=False))
        bad = copy.deepcopy(source)
        bad["public_metadata"] = {"invalid": float("nan")}
        self.assertTrue(self.validate_source(bad, verify_hash=False))

    def test_clip_rejects_interval_without_original_audio(self):
        # Catches returning an apparently joint AV clip before the delayed audio exists.
        with self.assertRaises(ValueError):
            self.clip(self.source(), 0, .2, self.root / "clips")

    def test_external_source_rejects_nonfinite_duration_and_forged_asset_identity(self):
        # Catches accepting a source from untrusted callers without byte/clock integrity.
        source = self.source()
        for duration in (float("nan"), float("inf"), -float("inf"), 0, -1, 10**400):
            bad = copy.deepcopy(source)
            bad["duration_s"] = duration
            with self.subTest(duration=duration):
                self.assertTrue(self.validate_source(bad))
                with self.assertRaises(ValueError):
                    self.clip(bad, 0, 1, self.root / "bad-clips")
        bad = copy.deepcopy(source)
        bad["asset_path"] = str(self.audio)
        self.assertTrue(self.validate_source(bad))
        with self.assertRaises(ValueError):
            self.clip(bad, 0, 1, self.root / "bad-clips")
        bad = copy.deepcopy(source)
        bad["asset_sha256"] = "0" * 64
        bad["source_id"] = "src_" + "0" * 24
        self.assertTrue(self.validate_source(bad))
        with self.assertRaises(ValueError):
            self.clip(bad, 0, 1, self.root / "bad-clips")

    def test_derived_validator_requires_complete_original_stream_mapping(self):
        # Catches accepting a clip receipt that omits or substitutes an original AV stream.
        source = self.source()
        derived = self.clip(source, 0, 2.2, self.root / "clips")
        bad = copy.deepcopy(derived)
        bad["mapping"]["streams"].pop()
        self.assertTrue(self.validate_derived(bad, source))
        bad = copy.deepcopy(derived)
        bad["mapping"]["streams"][1]["source_stream_index"] = 0
        self.assertTrue(self.validate_derived(bad, source))

    def test_audio_only_ingest_has_sample_mapping_and_clip(self):
        # Catches assumptions that every media asset has a video clock.
        source = self.ingest(self.audio, "https://example.org/speech", self.root / "assets")
        self.assertEqual(source["media_type"], "audio")
        self.assertEqual(source["timestamp_origin"]["kind"], "earliest_decoded_pts_fallback")
        self.assertEqual(source["timestamp_origin"]["container_start_time_s"], 0.0)
        self.assertEqual(source["rights_basis"], "unknown")
        self.assertEqual(source["redistribution_scope"], "metadata_only")
        self.assertTrue(source["stream_specs"][0]["sample_mapping"])
        derived = self.clip(source, .2, .7, self.root / "clips")
        self.assertEqual(self.validate_derived(derived, source), [])
        self.assertEqual([s["codec_type"] for s in probe(derived["output_path"])["streams"]], ["audio"])


if __name__ == "__main__":
    unittest.main()

"""Real-media cost contracts distinguish decoder work from prepared outputs."""
import asyncio
import importlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.av_data.ingest import ingest_media
from research.av_data.media_clock import clip_media, probe_media, run_media, tool_version
from research.av_data.validate import validate_source
from validation.research.media_fixtures import make_offset_vfr, probe, run


class MediaCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="private-av-cost-")
        cls.root = Path(cls.tmp.name)
        cls.media = cls.root / "private source ; literal.mkv"
        make_offset_vfr(cls.media)
        cls.source = ingest_media(cls.media, "https://example.org/recording", cls.root / "assets")
        cls.frames = probe(cls.media, frames=True)["frames"]
        cls.video_frames = sum(f["media_type"] == "video" for f in cls.frames)
        cls.audio_samples = sum(int(f["nb_samples"]) for f in cls.frames if f["media_type"] == "audio")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        try:
            self.collect = importlib.import_module("research.av_data.media_cost").collect_media_work
        except (ImportError, AttributeError) as exc:
            self.fail(f"Required media-work collector is missing: {exc}")

    def test_repeated_validation_and_probe_count_every_real_scan(self):
        # Removing validation scans or deduplicating repeated paths must fail this.
        with self.collect() as ledger:
            self.assertEqual(validate_source(self.source), [])
            self.assertEqual(validate_source(self.source), [])
            probe_media(self.media)
            tool_version("ffprobe")
        summary = ledger.public_summary()
        self.assertEqual(summary["operation_count"], 4)
        self.assertEqual(summary["successful_operations"], 4)
        self.assertEqual(summary["ffprobe_scan_operations"], 3)
        self.assertEqual(summary["observed_ffprobe_video_frames"], 3 * self.video_frames)
        self.assertEqual(summary["observed_ffprobe_audio_samples"], 3 * self.audio_samples)
        self.assertTrue(summary["ffprobe_observation_complete"])
        self.assertEqual(summary["subprocess_cache_hits"], 0)
        self.assertEqual(summary["input_asset_bytes"], 3 * self.media.stat().st_size)
        self.assertGreater(summary["wall_seconds"], 0)
        self.assertEqual([e["operation"] for e in summary["events"]][-1], "tool_version")
        self.assertFalse(summary["full_decode_work_accounting_ready"])

    def test_clip_counts_measured_input_decoding_separately_and_preserves_bytes(self):
        # Output frame/sample counts cannot stand for input decoding, scans or source clocks.
        plain = clip_media(self.source, .4, 1.4, self.root / "plain")
        with self.collect() as ledger:
            derived = clip_media(self.source, .4, 1.4, self.root / "accounted")
        summary = ledger.public_summary()
        output_frames = probe(derived["output_path"], frames=True)["frames"]
        prepared_video = sum(f["media_type"] == "video" for f in output_frames)
        prepared_samples = sum(int(f["nb_samples"]) for f in output_frames if f["media_type"] == "audio")
        self.assertEqual(summary["operation_count"], 6)
        self.assertEqual(summary["ffprobe_scan_operations"], 4)
        self.assertEqual(summary["ffmpeg_decode_operations"], 1)
        self.assertEqual(summary["observed_ffprobe_video_frames"], 2 * (self.video_frames + prepared_video))
        self.assertEqual(summary["observed_ffprobe_audio_samples"], 2 * (self.audio_samples + prepared_samples))
        self.assertGreater(summary["observed_ffmpeg_input_video_frames"], prepared_video)
        self.assertGreater(summary["observed_ffmpeg_input_audio_samples"], prepared_samples)
        self.assertTrue(summary["ffmpeg_input_decode_observation_complete"])
        ffmpeg = next(e for e in summary["events"] if e["operation"] == "ffmpeg_decode_transform")
        self.assertEqual(ffmpeg["decode_measurement_basis"], "ffmpeg_verbose_input_stream_summary")
        self.assertGreater(ffmpeg["observed_io_read_bytes"], 0)
        self.assertEqual(ffmpeg["output_asset_bytes"], derived["output_bytes"])
        # Matroska UIDs make even two uninstrumented clips byte-nondeterministic.
        # Prove unchanged decoded content/PTS and exact receipt-to-byte hash instead.
        for record in (plain, derived):
            self.assertEqual(record["output_sha256"], hashlib.sha256(Path(record["output_path"]).read_bytes()).hexdigest())
        def decoded_digest(record):
            return run(["ffmpeg", "-v", "error", "-i", record["output_path"], "-map", "0", "-f", "framemd5", "-"]).stdout
        self.assertEqual(decoded_digest(derived), decoded_digest(plain))
        self.assertEqual(self.media.read_bytes(), Path(self.source["asset_path"]).read_bytes())
        self.assertEqual(derived["mapping"], plain["mapping"])
        self.assertEqual(derived["transform"], plain["transform"])
        self.assertFalse(summary["full_decode_work_accounting_ready"])

    def test_failed_process_is_counted_without_claiming_zero_unknown_work_or_leaking_paths(self):
        # Dropping failed calls, treating absent decoder stats as zero, or exposing stderr must fail.
        secret = self.root / "private gold SECRET_SOURCE_PATH.mkv"
        secret.write_bytes(b"not a media file")
        with self.collect() as ledger:
            with self.assertRaises(ValueError):
                probe_media(secret)
            with self.assertRaises(ValueError):
                run_media(["ffmpeg", "-nostdin", "-v", "error", "-i", str(secret), "-f", "null", "-"])
        summary = ledger.public_summary()
        self.assertEqual(summary["operation_count"], 2)
        self.assertEqual(summary["failed_operations"], 2)
        self.assertEqual(summary["unknown_partial_work_operations"], 2)
        self.assertFalse(summary["ffprobe_observation_complete"])
        self.assertFalse(summary["ffmpeg_input_decode_observation_complete"])
        self.assertFalse(summary["full_decode_work_accounting_ready"])
        for event in summary["events"]:
            self.assertEqual(event["status"], "failed")
            self.assertEqual(event["failure_kind"], "nonzero_exit")
            self.assertIsNone(event["observed_input_audio_samples"])
            self.assertTrue(event["unknown_partial_work"])
            self.assertGreater(event["stderr_bytes"], 0)
            self.assertGreaterEqual(event["wall_seconds"], 0)
        public = json.dumps(summary, allow_nan=False)
        for value in (str(self.root), "SECRET_SOURCE_PATH", "private gold", "Invalid data", "stderr\"", "argv"):
            self.assertNotIn(value, public)

    def test_success_without_input_decode_statistics_remains_unknown(self):
        # Stream copying produces output frames but cannot prove input decoder work.
        with self.collect() as ledger:
            run_media(["ffmpeg", "-nostdin", "-v", "error", "-i", str(self.media),
                       "-map", "0", "-c", "copy", "-f", "null", "-"])
        summary = ledger.public_summary()
        self.assertEqual(summary["successful_operations"], 1)
        self.assertEqual(summary["unknown_partial_work_operations"], 1)
        self.assertFalse(summary["ffmpeg_input_decode_observation_complete"])
        self.assertIsNone(summary["events"][0]["observed_input_audio_samples"])
        self.assertIsNone(summary["events"][0]["observed_input_video_frames"])
        self.assertTrue(summary["observed_totals_are_lower_bounds"])

    def test_cannot_launch_tool_is_counted_and_does_not_poison_next_scope(self):
        # OS errors need a cost denominator even without stderr or a process exit code.
        with self.collect() as ledger:
            with self.assertRaises(ValueError):
                run_media([str(self.root / "private_missing_ffmpeg")])
        summary = ledger.public_summary()
        self.assertEqual(summary["failed_operations"], 1)
        self.assertEqual(summary["events"][0]["failure_kind"], "os_error")
        self.assertIsNone(summary["events"][0]["returncode"])
        self.assertNotIn(str(self.root), json.dumps(summary))
        with self.collect() as next_ledger:
            tool_version("ffprobe")
        self.assertEqual(next_ledger.public_summary()["successful_operations"], 1)

    def test_context_local_capture_does_not_leak_across_tasks_or_outside_scope(self):
        # A process-global collector would mix the one-scan and two-scan request costs.
        async def capture(repeats):
            with self.collect() as ledger:
                for _ in range(repeats):
                    await asyncio.to_thread(probe_media, self.media)
            return ledger.public_summary()
        async def both():
            return await asyncio.gather(capture(1), capture(2))
        first, second = asyncio.run(both())
        self.assertEqual(first["operation_count"], 1)
        self.assertEqual(second["operation_count"], 2)
        probe_media(self.media)
        self.assertEqual(first["operation_count"], 1)
        with self.collect() as empty:
            pass
        self.assertEqual(empty.public_summary()["operation_count"], 0)

    def test_nested_scopes_include_child_work_and_public_summary_is_detached(self):
        # Nested helpers cannot make operations disappear from the enclosing request.
        with self.collect() as outer:
            probe_media(self.media)
            with self.collect() as inner:
                tool_version("ffprobe")
        self.assertEqual(outer.public_summary()["operation_count"], 2)
        self.assertEqual(inner.public_summary()["operation_count"], 1)
        changed = outer.public_summary()
        changed["events"].clear()
        self.assertEqual(len(outer.public_summary()["events"]), 2)


if __name__ == "__main__":
    unittest.main()

"""Real AV content and scoped frozen corpus tests for native inference tools."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "onesearch_vl"))
sys.path.insert(0, str(ROOT / "validation/research"))
from onesearch_infer import prompts, tools
from media_fixtures import make_offset_vfr, decoded_markers, probe, run
from research.av_search.corpus import build_corpus


class NativeAVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.av_io = importlib.import_module("onesearch_infer.av_io")
        except ImportError:
            cls.av_io = None
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / "private_title_source.mkv"
        make_offset_vfr(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.assertIsNotNone(self.av_io, "native AV implementation missing")
        self.env = patch.dict(os.environ, {"ONESEARCH_DISABLE_RETRIEVAL": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_crop_vfr_retains_delayed_audio_on_common_timeline(self):
        handle = self.av_io.make_native_av_handle(self.source)
        self.assertEqual(handle["source_start_s"], 0)
        self.assertAlmostEqual(handle["duration"], 4, delta=.05)
        clip = self.av_io.crop_native_av(handle, 0, 2, self.root / "delayed.mp4")
        visual, audio, data = decoded_markers(clip["video_url"])
        self.assertAlmostEqual(visual, 1.6, delta=.08)
        self.assertAlmostEqual(audio, 1.6, delta=.08)
        self.assertAlmostEqual(audio, visual, delta=.06)
        codecs = {s["codec_type"]: s["codec_name"] for s in data["streams"]}
        self.assertEqual(codecs, {"video": "h264", "audio": "aac"})
        self.assertTrue(clip["has_audio"])

    def test_nested_crop_uses_original_source_seconds(self):
        handle = self.av_io.make_native_av_handle(self.source)
        first = self.av_io.crop_native_av(handle, .5, 3.5, self.root / "first.mp4")
        nested = self.av_io.crop_native_av(first, .5, 2, self.root / "nested.mp4")
        self.assertEqual(nested["source_path"], str(self.source))
        self.assertAlmostEqual(nested["source_start_s"], 1)
        self.assertAlmostEqual(nested["source_end_s"], 2.5)
        self.assertAlmostEqual(nested["duration"], 1.5)
        visual, audio, _ = decoded_markers(nested["video_url"])
        self.assertAlmostEqual(visual, .6, delta=.08)
        self.assertAlmostEqual(audio, .6, delta=.08)
        self.assertAlmostEqual(audio, visual, delta=.06)

    def test_inspect_dispatch_returns_media_without_private_path_in_observation(self):
        handle = self.av_io.make_native_av_handle(self.source)
        call = tools.extract_tool_call('<tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":1,"end_s":3}}</tool_call>')
        text, media = tools.execute_tool(call, {"video_1": handle}, "case", 0, 1,
                                         str(self.root), modality="audio_video")
        self.assertNotIn("Tool execution error", text)
        self.assertNotIn(self.source.name, text)
        self.assertNotIn(str(self.root), text)
        self.assertEqual(set(media), {"clip_1"})
        self.assertEqual(media["clip_1"]["kind"], "native_av")
        self.assertEqual(media["clip_1"]["media_type"], "native_av")
        self.assertTrue(Path(media["clip_1"]["video_url"]).is_file())

    def test_invalid_handle_types_return_observation_instead_of_unwinding_episode(self):
        handle = self.av_io.make_native_av_handle(self.source)
        for ref in (["video_1"], {"id": "video_1"}, 123, True):
            with self.subTest(ref=ref):
                text, media = tools.execute_tool(
                    {"name": "inspect_av", "arguments": {"video": ref, "start_s": 0, "end_s": 1}},
                    {"video_1": handle}, "case", 0, 1, str(self.root), modality="audio_video")
                self.assertIn("Tool execution error", text)
                self.assertEqual(media, {})

    def test_invalid_intervals_are_rejected_before_media_creation(self):
        handle = self.av_io.make_native_av_handle(self.source)
        out = self.root / "invalid.mp4"
        for start, end in [(False, 1), (0, True), (float("nan"), 1),
                           (0, float("inf")), (-1, 1), (1, 1), (0, 4.5)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.av_io.crop_native_av(handle, start, end, out)
        self.assertFalse(out.exists())

    def test_interval_before_delayed_audio_is_a_valid_silent_clip(self):
        handle = self.av_io.make_native_av_handle(self.source)
        clip = self.av_io.crop_native_av(handle, 0, .2, self.root / "before_audio.mp4")
        self.assertFalse(clip["has_audio"])
        self.assertEqual([s["codec_type"] for s in probe(clip["video_url"])["streams"]], ["video"])

    def test_video_without_audio_is_valid_and_reported(self):
        source = self.root / "silent.mp4"
        run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
             "color=c=red:s=64x64:r=10:d=2", "-c:v", "libx264", "-y", str(source)])
        handle = self.av_io.make_native_av_handle(source)
        self.assertFalse(handle["has_audio"])
        clip = self.av_io.crop_native_av(handle, .25, 1.5, self.root / "silent_clip.mp4")
        self.assertFalse(clip["has_audio"])
        self.assertEqual([s["codec_type"] for s in probe(clip["video_url"])["streams"]], ["video"])


class FrozenBackendTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("onesearch_infer.search_backends")
        except ImportError:
            self.module = None
        self.assertIsNotNone(self.module, "frozen backend implementation missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docs = []
        for i, body in enumerate(["Public hammer demonstration with ringing metal.",
                                  "Private-scope hammer answer in a hidden source."]):
            asset = self.root / f"asset-{i}"; asset.write_text(body)
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            self.docs.append({"source_id": "src_" + digest[:24], "origin_url": f"https://example.org/source-{i}",
                              "asset_sha256": digest, "asset_path": str(asset), "text": body,
                              "text_kind": "web_body", "public_metadata": {"title": f"Source {i}"},
                              "media_type": "text/html"})
        self.index = self.root / "index"
        build_corpus(self.docs, self.index, "native-tools", "exploration")
        self.scope = {"corpus_id": "native-tools", "split": "exploration",
                      "source_ids": [self.docs[0]["source_id"]]}
        self.backend = self.module.FrozenSearchBackend(self.index, self.scope)
        self.env = patch.dict(os.environ, {"ONESEARCH_DISABLE_RETRIEVAL": "0"})
        self.env.start(); self.addCleanup(self.env.stop)

    def dispatch(self, name, **parameters):
        return tools.execute_tool({"name": name, "arguments": parameters}, {}, "case", 0, 1,
                                  str(self.root), modality="audio_video", search_backend=self.backend)

    def test_search_returns_real_scoped_urls_and_visit_returns_frozen_body(self):
        text, media = self.dispatch("text_search", q="hammer", top_k=5)
        self.assertIn(self.docs[0]["origin_url"], text)
        self.assertIn(self.docs[0]["text"], text)
        self.assertNotIn(self.docs[1]["origin_url"], text)
        self.assertNotIn(self.docs[1]["text"], text)
        self.assertNotIn(str(self.root), text)
        self.assertEqual(media, {})
        text, media = self.dispatch("visit_webpage", url=self.docs[0]["origin_url"])
        self.assertIn(self.docs[0]["text"], text)
        self.assertEqual(media, {})

    def test_visit_never_opens_a_url_outside_scope(self):
        with patch("requests.sessions.Session.request", side_effect=AssertionError("HTTP crossed frozen boundary")):
            for url in [self.docs[1]["origin_url"], "https://unseen.org/page", "file:///private/gold.json"]:
                text, media = self.dispatch("visit_webpage", url=url)
                self.assertIn("Tool execution error", text)
                self.assertEqual(media, {})

    def test_disable_gate_blocks_injected_search_and_visit(self):
        os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = "1"
        for name, params in [("text_search", {"q": "hammer"}),
                             ("visit_webpage", {"url": self.docs[0]["origin_url"]})]:
            text, media = self.dispatch(name, **params)
            self.assertIn("Retrieval disabled", text)
            self.assertEqual(media, {})
        self.assertEqual(prompts.tool_names("audio_video"), {"inspect_av"})

    def test_initial_scope_copy_prevents_caller_mutation_escalation(self):
        self.scope["source_ids"].append(self.docs[1]["source_id"])
        text = self.backend.text_search("hammer")
        self.assertNotIn(self.docs[1]["origin_url"], text)

    def test_tampered_index_is_rejected(self):
        index = self.index / "index.sqlite"
        index.write_bytes(index.read_bytes() + b"tampered")
        text, media = self.dispatch("text_search", q="hammer")
        self.assertIn("Tool execution error", text)
        self.assertEqual(media, {})

    def test_default_visit_remains_unavailable(self):
        text, media = tools.execute_tool({"name": "visit_webpage", "arguments": {"url": "https://example.org"}},
                                         {}, "case", 0, 1, str(self.root), modality="video")
        self.assertIn("visit_webpage is not available", text)
        self.assertEqual(media, {})


if __name__ == "__main__":
    unittest.main()

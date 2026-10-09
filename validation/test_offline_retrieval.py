"""Contracts for the optional inference-only retrieval gate; no live providers."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "onesearch_vl"))
from PIL import Image
from onesearch_infer import prompts, tools


class OfflineRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"ONESEARCH_DISABLE_RETRIEVAL": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def dispatch(self, name, **parameters):
        return tools.execute_tool(
            {"name": name, "parameters": parameters}, {},
            "offline-test", 0, 0, "/tmp", modality="video",
        )

    def test_default_prompt_is_byte_identical_to_published_files(self):
        for value in (None, "0"):
            if value is None:
                os.environ.pop("ONESEARCH_DISABLE_RETRIEVAL", None)
            else:
                os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = value
            for modality in ("single_image", "multi_image", "video"):
                with self.subTest(flag=value, modality=modality):
                    expected = Path(prompts.__file__).with_name(modality + ".txt").read_bytes()
                    self.assertEqual(prompts.get_system_prompt(modality).encode("utf-8"), expected)

    def test_disabled_actor_and_dispatch_schema_hide_all_retrieval_tools(self):
        os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = "1"
        for modality in ("single_image", "multi_image", "video"):
            with self.subTest(modality=modality):
                specs = prompts.get_tool_specs(modality)
                names = {spec["function"]["name"] for spec in specs}
                self.assertIn("crop", names)
                self.assertFalse({"text_search", "image_search", "web_search"} & names)
                self.assertFalse(any("search" in name or name.startswith("visit") for name in names))
                self.assertEqual(prompts.tool_names(modality), names)
                self.assertEqual(json.loads(tools.get_tools_definition(modality)), specs)
                if modality == "video":
                    self.assertTrue({"select_frame", "select_timespan"} <= names)

    def test_flag_switching_does_not_reuse_a_prompt_or_name_cache_from_other_mode(self):
        expected = Path(prompts.__file__).with_name("video.txt").read_bytes().decode("utf-8")
        for value in ("0", "1", "0", "1", "0"):
            os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = value
            prompt = prompts.get_system_prompt("video")
            names = prompts.tool_names("video")
            if value == "0":
                self.assertEqual(prompt, expected)
                self.assertIn("text_search", names)
            else:
                self.assertFalse(prompt == expected, "offline prompt reused the online cache")
                self.assertNotIn("text_search", names)
                self.assertNotIn("image_search", names)

    def test_disabled_dispatch_blocks_search_and_visit_aliases_before_provider_or_upload(self):
        os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = "1"
        aliases = (
            "text_search", "image_search", "web_search", "local_search",
            "local_image_search", "search", "search_text", "search_image",
            "search_web", "lens_scan", "visit", "visit_page", "visit_webpage",
        )
        with ExitStack() as stack:
            for target in (
                "onesearch_infer.tools.search.text_search",
                "onesearch_infer.tools.search.image_search",
                "onesearch_infer.tools._resolve_image_for_search",
                "onesearch_infer.tools.cos_upload.upload_pil_image",
                "requests.sessions.Session.request",
            ):
                stack.enter_context(patch(target, side_effect=AssertionError("retrieval crossed a disabled boundary")))
            for name in aliases:
                with self.subTest(name=name):
                    message, media = self.dispatch(name, q="query", image="img_1", url="https://example.org")
                    self.assertIn("Retrieval disabled", message)
                    self.assertEqual(media, {})

    def test_legacy_tag_aliases_are_blocked_after_actual_parsing(self):
        os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = "1"
        for tag in ("local_search", "local_image_search", "lens_scan"):
            with self.subTest(tag=tag):
                call = tools.extract_tool_call(f'<{tag}>{{"q":"query","image":"img_1"}}</{tag}>')
                self.assertIsNotNone(call)
                with patch.object(tools.search, "text_search", side_effect=AssertionError("provider reached")), \
                     patch.object(tools.search, "image_search", side_effect=AssertionError("provider reached")):
                    message, media = tools.execute_tool(call, {}, "offline-test", 0, 0, "/tmp")
                self.assertIn("Retrieval disabled", message)
                self.assertEqual(media, {})

    def test_default_search_dispatch_still_returns_real_provider_result(self):
        expected = "Tool execution result:\n[Passage 1]\nURL: https://example.org\nSummary: Fixture evidence."
        with patch.object(tools.search, "text_search", return_value=expected):
            message, media = self.dispatch("text_search", q="query")
        self.assertEqual(message, expected)
        self.assertEqual(media, {})

    def test_local_crop_preserves_pixels_and_handles_with_retrieval_disabled(self):
        os.environ["ONESEARCH_DISABLE_RETRIEVAL"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            original = Image.new("RGB", (40, 20), "red")
            original.paste("blue", (20, 0, 40, 20))
            source = Path(tmp) / "input.png"
            original.save(source)
            handles = {"img_1": str(source)}
            with patch.object(tools.cos_upload, "upload_pil_image", return_value=None), \
                 patch("requests.sessions.Session.request", side_effect=AssertionError("local crop used HTTP")):
                message, produced = tools.execute_tool(
                    {"name": "crop", "parameters": {"image": "img_1", "bbox_1000": [0, 0, 500, 1000]}},
                    handles, "offline-test", 0, 0, tmp, modality="single_image",
                )
            self.assertNotIn("Tool execution error", message)
            self.assertEqual(len(produced), 1)
            self.assertIn("img_2", produced)
            with Image.open(produced["img_2"]) as crop:
                self.assertEqual(crop.size, (20, 20))
                self.assertEqual(crop.getpixel((19, 10)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

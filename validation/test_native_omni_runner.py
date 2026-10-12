"""CPU contracts for the native AV runner (no weight load or GPU required)."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from onesearch_vl.onesearch_infer import config
from onesearch_vl.onesearch_infer.runners import build_runner, InferenceConfig
from onesearch_vl.onesearch_infer.omni_runner import (
    OmniRunner, normalize_explicit_action, action_stopping_criteria, complete_action_end, move_inputs, to_omni_messages,
)


def video(path, audio=True, fps=2.0):
    return {"video_url": {"url": path, "native_av": True, "has_audio": audio,
                          "fps": fps, "min_pixels": 3136, "max_pixels": 100352}}


class CharacterTokenizer:
    init_kwargs = {}

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(int(token)) for token in ids)

    def __call__(self, texts, **kwargs):
        ids = torch.tensor([self.encode(texts[0])], dtype=torch.int64)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class RecordingProcessor:
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.video_processor = SimpleNamespace(temporal_patch_size=2)
        self.received = []
        self.expanded_seconds = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = deepcopy(messages)
        return "\n".join(str(message["role"]) + ":" + " ".join(
            block["text"] if block["type"] == "text" else "<" + block["type"] + ">"
            for block in message["content"]) for message in messages)

    def __call__(self, **kwargs):
        self.received.append(kwargs)
        result = self.tokenizer([kwargs["text"]])
        if kwargs["audio"]:
            result["input_features"] = torch.full((len(kwargs["audio"]), 128, 100), 0.25)
            result["feature_attention_mask"] = torch.ones((len(kwargs["audio"]), 100), dtype=torch.int64)
        if kwargs["videos"]:
            result["pixel_values_videos"] = torch.full((sum(len(v) for v in kwargs["videos"]), 8), 0.5)
            result["video_grid_thw"] = torch.tensor([[len(v)//2, 2, 2] for v in kwargs["videos"]], dtype=torch.int64)
        if kwargs["images"]:
            result["pixel_values"] = torch.full((len(kwargs["images"]), 8), 0.75)
            result["image_grid_thw"] = torch.tensor([[1, 2, 2] for _ in kwargs["images"]], dtype=torch.int64)
        return result

    def replace_multimodal_special_tokens(self, text, audio_lengths, image_grid_thw,
                                         video_grid_thw, video_second_per_grid,
                                         use_audio_in_video, position_id_per_seconds, seconds_per_chunk):
        self.expanded_seconds = list(video_second_per_grid)
        return text

    def batch_decode(self, ids, **kwargs):
        return [self.tokenizer.decode(ids[0])]


class RecordingModel:
    def __init__(self, processor):
        self.processor = processor
        self.weight = torch.zeros((1, 1), dtype=torch.bfloat16)
        self.calls = []
        self.thinker = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=self.weight))

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        # Simulate autoregressive calls, including a second action that must not run.
        output = '<think>inspect clip</think><tool_call>{"name":"av_search","arguments":{"query":"sound"}}</tool_call><response>extra</response>'
        for token in self.processor.tokenizer.encode(output):
            ids = torch.cat([ids, torch.tensor([[token]], dtype=ids.dtype)], dim=1)
            if kwargs["thinker_stopping_criteria"](ids, None).item():
                break
        return ids


class NativeOmniTests(unittest.TestCase):
    def runner(self):
        runner = build_runner("omni-7b", checkpoint="/mock/snapshot", gpus="cpu")
        processor = RecordingProcessor()
        calls = []
        def process(messages, use_audio_in_video, return_video_kwargs):
            block = messages[0]["content"][0]
            calls.append((deepcopy(block), use_audio_in_video))
            kind = block["type"]
            audio = [torch.ones(1600).numpy()] if kind == "audio" or use_audio_in_video else None
            images = [object()] if kind == "image" else None
            videos = [torch.ones(20, 3, 56, 56)] if kind == "video" else None
            return audio, images, videos, {"fps": [block["fps"]] if kind == "video" else [], "do_sample_frames": False}
        runner._processor, runner._process_mm_info = processor, process
        runner._has_audio = lambda block: block.get("_has_audio") is True
        runner._model = RecordingModel(processor)
        return runner, processor, calls

    def test_registration_factory_is_lazy_and_original_runners_remain(self):
        spec = config.MODEL_REGISTRY["omni-7b"]
        self.assertEqual(spec.family, "qwen2_5_omni")
        self.assertTrue(spec.supports_video)
        self.assertEqual(spec.default_checkpoint, "Qwen/Qwen2.5-Omni-7B")
        self.assertIsInstance(build_runner("omni-7b"), OmniRunner)
        self.assertTrue(build_runner("omni-7b").supports_audio_video)
        self.assertEqual(build_runner("8b", checkpoint="original").checkpoint, "original")
        script = ('import sys; from onesearch_vl.onesearch_infer.runners import build_runner; '
                  'r=build_runner("omni-7b"); '
                  'assert "qwen_omni_utils" not in sys.modules; assert "transformers" not in sys.modules')
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_explicit_action_normalization_preserves_model_arguments(self):
        for output in ['{"name":"text_search","arguments":{"q":"tram"}}',
                       'text_search({"q":"tram"})',
                       '<think>Need location.</think>\n{"name":"text_search","arguments":{"q":"tram"}}']:
            with self.subTest(output=output):
                normalized, form = normalize_explicit_action(output)
                self.assertNotEqual(form, 'unchanged')
                from onesearch_vl.onesearch_infer.tools import extract_tool_call
                call = json.loads(extract_tool_call(normalized))
                self.assertEqual(call['name'], 'text_search')
                self.assertEqual(call['parameters'], {'q': 'tram'})
        for output in ['Search for tram', 'text_search("tram")',
                       '{"name":"text_search"}', '{"name":"delete","arguments":{}}',
                       '{"name":"text_search","arguments":{"q":"tram"}} extra',
                       '<response>text_search({"q":"tram"})</response>']:
            self.assertEqual(normalize_explicit_action(output), (output, 'unchanged'))

    def test_pure_conversion_preserves_all_history_without_path_text(self):
        contents = [{"role": "user", "parts": [video("/private/source_title.mp4"), {"text": "What happened?"}]},
                    {"role": "model", "parts": [{"text": '<tool_call>{"name":"av_search"}</tool_call>'}]},
                    {"role": "user", "parts": [{"image_url": {"url": "/private/frame.png"}},
                                               video("/private/new_clip.mp4", fps=1.7), {"text": "Tool result"}]}]
        before = deepcopy(contents)
        messages = to_omni_messages(contents, "Use the tools.")
        self.assertEqual(contents, before)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"][0]["video"], "/private/source_title.mp4")
        self.assertEqual(messages[3]["content"][1]["video"], "/private/new_clip.mp4")
        self.assertEqual(messages[3]["content"][1]["fps"], 1.7)
        text = " ".join(b["text"] for m in messages for b in m["content"] if b["type"] == "text")
        self.assertNotIn("/private", text)
        self.assertNotIn("source_title", text)

    def test_unknown_media_or_invalid_sampling_never_silently_dropped(self):
        for part in [{"caption": "discard me"}, video("/clip", fps=0), {"video_url": {"url": "x", "has_audio": "yes"}}]:
            with self.subTest(part=part), self.assertRaises(ValueError):
                to_omni_messages([{"parts": [part]}])

    def test_inline_images_and_explicit_audio_are_media(self):
        messages = to_omni_messages([{"parts": [{"inline_data": {"mime_type": "image/png", "data": "AA=="}},
                                                {"audio_url": {"url": "/recording.wav", "audio_start": 2.0}}, {"text": "Listen."}]}])
        self.assertEqual(messages[0]["content"][0]["image"], "data:image/png;base64,AA==")
        self.assertEqual(messages[0]["content"][1]["audio_start"], 2.0)

    def test_actual_av_features_and_all_historical_videos_reach_processor(self):
        runner, processor, calls = self.runner()
        contents = [{"role": "user", "parts": [video("/first")]}, {"role": "model", "parts": [{"text": "inspect"}]},
                    {"role": "user", "parts": [video("/second", fps=1.7), {"image": "/frame"}]}]
        inputs = runner.prepare_inputs(contents, "system")
        self.assertEqual([b[0].get("video") for b in calls if b[0]["type"] == "video"], ["/first", "/second"])
        self.assertTrue(all(flag for block, flag in calls if block["type"] == "video"))
        self.assertEqual(inputs["input_features"].shape[0], 2)
        self.assertEqual(inputs["video_grid_thw"].shape[0], 2)
        self.assertEqual(inputs["image_grid_thw"].shape[0], 1)
        self.assertEqual(processor.received[0]["fps"], 2.0)
        self.assertTrue(processor.received[0]["use_audio_in_video"])
        self.assertFalse(processor.received[0]["videos_kwargs"]["do_resize"])
        self.assertFalse(processor.received[0]["images_kwargs"]["do_resize"])
        self.assertEqual(processor.expanded_seconds, [1.0, 2/1.7])
        self.assertEqual(runner.last_metrics["actual_video_fps"], [2.0, 1.7])
        self.assertEqual(runner.last_metrics["audio_count"], 2)
        self.assertEqual(calls[0][0]["max_pixels"], 100352)
        self.assertEqual(calls[0][0]["min_pixels"], 3136)

    def test_visual_only_has_no_fabricated_audio(self):
        runner, processor, calls = self.runner()
        inputs = runner.prepare_inputs([{"parts": [video("/silent", audio=False)]}])
        self.assertNotIn("input_features", inputs)
        self.assertIsNone(processor.received[0]["audio"])
        self.assertFalse(processor.received[0]["use_audio_in_video"])
        self.assertEqual(runner.last_metrics["audio_count"], 0)
        self.assertEqual(runner.last_metrics["video_has_audio"], [False])

    def test_mixed_history_keeps_real_audio_and_silent_video(self):
        runner, processor, calls = self.runner()
        inputs = runner.prepare_inputs([{"parts": [video("/audio"), video("/silent", audio=False)]}])
        self.assertEqual(inputs["input_features"].shape[0], 1)
        self.assertEqual(inputs["video_grid_thw"].shape[0], 2)
        self.assertEqual([b["type"] for b in processor.messages[0]["content"]], ["audio", "video", "video"])
        self.assertEqual([flag for block, flag in calls], [True, False])
        self.assertFalse(processor.received[0]["use_audio_in_video"])
        self.assertEqual(runner.last_metrics["audio_video_mode"], "separate_streams_mixed_history")

    def test_generation_receives_mm_tensors_and_stops_at_one_complete_action(self):
        runner, processor, _ = self.runner()
        response = runner.infer([{"parts": [video("/clip"), {"image": "/frame"}, {"text": "Find the sound"}]}],
                                cfg=InferenceConfig(max_tokens=180, temperature=0))
        seen = runner._model.calls[0]
        for key in ("input_features", "pixel_values_videos", "pixel_values", "video_grid_thw", "input_ids", "feature_attention_mask"):
            self.assertIn(key, seen)
        self.assertEqual(seen["input_features"].dtype, torch.bfloat16)
        self.assertEqual(seen["input_ids"].dtype, torch.int64)
        self.assertEqual(seen["video_grid_thw"].dtype, torch.int64)
        self.assertTrue(seen["use_audio_in_video"])
        self.assertFalse(seen["return_audio"])
        self.assertEqual(seen["thinker_max_new_tokens"], 180)
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        self.assertTrue(text.endswith("</tool_call>"))
        self.assertNotIn("<response>", text)
        self.assertTrue(response["raw_response"]["metrics"]["stopped_on_complete_action"])
        self.assertGreater(response["usageMetadata"]["candidatesTokenCount"], 0)
        self.assertEqual(response["usageMetadata"]["totalTokenCount"],
                         response["usageMetadata"]["promptTokenCount"]+response["usageMetadata"]["candidatesTokenCount"])

    def test_prompt_tags_and_think_examples_do_not_trigger_terminal_stop(self):
        tokenizer = CharacterTokenizer()
        prefix = tokenizer.encode('<response>prompt example</response>')
        stop = action_stopping_criteria(tokenizer, len(prefix))
        for tail in ["", "<response>unfinished", "<think><response>example</response>", "<think>still thinking"]:
            with self.subTest(tail=tail):
                self.assertFalse(stop(torch.tensor([prefix+tokenizer.encode(tail)]), None).item())
        for tail in ['<response>done</response>', '<think>ok</think><tool_call>{}</tool_call>']:
            with self.subTest(tail=tail):
                self.assertTrue(stop(torch.tensor([prefix+tokenizer.encode(tail)]), None).item())
        raw = '<tool_call>{}</tool_call><response>extra</response>'
        self.assertEqual(raw[:complete_action_end(raw)], '<tool_call>{}</tool_call>')

    def test_integer_and_boolean_tensors_never_cast_to_model_float(self):
        inputs = {"input_ids": torch.tensor([[1,2]]), "mask": torch.tensor([True]),
                  "features": torch.ones(2), "nested": [torch.tensor([3]), torch.ones(1)]}
        moved = move_inputs(inputs, "cpu", torch.bfloat16)
        self.assertEqual(moved["input_ids"].dtype, torch.int64)
        self.assertEqual(moved["mask"].dtype, torch.bool)
        self.assertEqual(moved["features"].dtype, torch.bfloat16)
        self.assertEqual(moved["nested"][0].dtype, torch.int64)
        self.assertEqual(moved["nested"][1].dtype, torch.bfloat16)

    def test_missing_audio_is_detected_and_probe_cache_reused(self):
        runner = build_runner("omni-7b", gpus="cpu")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"video.mp4"; path.write_bytes(b"media mock")
            block = {"video": str(path), "_has_audio": True}
            with patch("onesearch_vl.onesearch_infer.omni_runner.subprocess.run",
                       return_value=SimpleNamespace(stdout=json.dumps({"streams":[]}))) as probe:
                self.assertFalse(runner._has_audio(block))
                self.assertFalse(runner._has_audio(block))
                self.assertEqual(probe.call_count, 1)
        self.assertFalse(runner._has_audio({"video":"/does/not/exist", "_has_audio":False}))

    def test_pinned_tokenizer_terminal_stop_on_real_tokenization(self):
        import importlib.metadata
        manifest_path = Path(__file__).resolve().parents[1] / "research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json"
        if importlib.metadata.version("transformers") != "4.57.6" or not manifest_path.is_file():
            self.skipTest("requires pinned Omni processor environment and acquired local tokenizer")
        manifest = json.loads(manifest_path.read_text())
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["snapshot_path"], local_files_only=True)
        prefix = tokenizer.encode('<response>prompt example</response>', add_special_tokens=False)
        stop = action_stopping_criteria(tokenizer, len(prefix))
        for raw, expected in [("", False), ("<tool_call>{", False),
                              ('<tool_call>{"name":"av_search","arguments":{}}</tool_call>', True),
                              ('<response>Final answer.</response>', True),
                              ('<think><response>quoted example</response>', False)]:
            ids = torch.tensor([prefix + tokenizer.encode(raw, add_special_tokens=False)])
            with self.subTest(raw=raw):
                self.assertEqual(stop(ids, None).item(), expected)

    def test_model_load_requires_verified_manifest(self):
        runner = build_runner("omni-7b", checkpoint="/unverified/local", gpus="cpu")
        runner._manifest_path = ""
        with self.assertRaisesRegex(RuntimeError, "ONESEARCH_OMNI_MODEL_MANIFEST"):
            runner.load()


if __name__ == "__main__":
    unittest.main()

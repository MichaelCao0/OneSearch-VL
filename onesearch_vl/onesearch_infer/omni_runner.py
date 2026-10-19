"""Native Qwen2.5-Omni Thinker in the original OneSearch agent loop.

Imports of Omni, torch and multimedia decoding are lazy. Conversion never
summarizes media or adds media filenames to the text prompt. Processor-only
preparation is available without loading model weights or allocating a GPU.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, Iterable, Optional
from urllib.parse import unquote, urlparse

from .config import ModelSpec
from .runners import BaseRunner, InferenceConfig, _wrap_gemini

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
_VIDEO_KEYS = ("fps", "min_pixels", "max_pixels", "max_frames", "total_pixels", "video_start", "video_end")


def to_omni_messages(contents: Iterable[Dict[str, Any]], system_instruction: Optional[str] = None):
    """Pure conversion; preserve every historical media block and its order.

    ``video_url`` takes a URL/local path plus optional ``has_audio``, ``fps``
    and pixel bounds. Unknown stream availability is resolved during preparation.
    Internal ``_has_audio`` metadata is ignored by the official chat template.
    """
    result = []
    if system_instruction:
        result.append({"role": "system", "content": [{"type": "text", "text": system_instruction}]})
    for message in contents:
        role = message.get("role", "user")
        role = "assistant" if role == "model" else role
        if role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported Omni conversation role")
        blocks = []
        for part in message.get("parts", []) or []:
            if "text" in part:
                blocks.append({"type": "text", "text": part["text"]})
            elif "video_url" in part or "video" in part:
                value = part.get("video_url", part.get("video"))
                metadata = value if isinstance(value, dict) else {}
                source = metadata.get("url", metadata.get("video")) if metadata else value
                if not isinstance(source, str) or not source:
                    raise ValueError("native video requires a nonempty URL or local path")
                block = {"type": "video", "video": source, "fps": 2.0,
                         "min_pixels": 3136, "max_pixels": 100352, "max_frames": 128}
                for key in _VIDEO_KEYS:
                    if key in metadata or key in part:
                        block[key] = metadata.get(key, part.get(key))
                for key in ("fps", "min_pixels", "max_pixels", "max_frames"):
                    value = block[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError("video sampling bounds must be finite and positive")
                if block["min_pixels"] > block["max_pixels"]:
                    raise ValueError("video min_pixels exceeds max_pixels")
                has_audio = metadata.get("has_audio", part.get("has_audio"))
                if has_audio is not None and not isinstance(has_audio, bool):
                    raise ValueError("has_audio must be boolean or unspecified")
                block["_has_audio"] = has_audio
                blocks.append(block)
            elif "image_url" in part or "image" in part:
                value = part.get("image_url", part.get("image"))
                metadata = value if isinstance(value, dict) else {}
                source = metadata.get("url", metadata.get("image")) if metadata else value
                if source is None or (isinstance(source, str) and not source):
                    raise ValueError("image requires media")
                blocks.append({"type": "image", "image": source,
                               "min_pixels": metadata.get("min_pixels", part.get("min_pixels", 3136)),
                               "max_pixels": metadata.get("max_pixels", part.get("max_pixels", 262144))})
            elif "audio_url" in part or "audio" in part:
                value = part.get("audio_url", part.get("audio"))
                metadata = value if isinstance(value, dict) else {}
                source = metadata.get("url", metadata.get("audio")) if metadata else value
                if source is None:
                    raise ValueError("audio requires media")
                block = {"type": "audio", "audio": source}
                for key in ("audio_start", "audio_end"):
                    if key in metadata or key in part:
                        block[key] = metadata.get(key, part.get(key))
                blocks.append(block)
            elif "inline_data" in part:
                value = part["inline_data"]
                mime = value.get("mime_type", "")
                media_type = "image" if mime.startswith("image/") else "audio" if mime.startswith("audio/") else None
                if media_type is None:
                    raise ValueError("inline media must be image/audio; native video uses a file or URL")
                blocks.append({"type": media_type, media_type: "data:" + mime + ";base64," + value["data"]})
            else:
                raise ValueError("unsupported Omni content part; media must not be silently dropped")
        if blocks:
            result.append({"role": role, "content": blocks})
    return result


def complete_action_end(text: str):
    """Return the first closed action boundary outside generated think blocks."""
    thinking = 0
    action = None
    for match in re.finditer(r"</?(?:think|tool_call|response)>", text):
        tag = match.group()
        if tag == "<think>" and action is None:
            thinking += 1
        elif tag == "</think>" and action is None:
            thinking = max(0, thinking - 1)
        elif not thinking and action is None and tag in {"<tool_call>", "<response>"}:
            action = tag[1:-1]
        elif action is not None and tag == "</" + action + ">":
            return match.end()
    return None



def normalize_explicit_action(text: str):
    """Wrap one explicit model-produced AV call in the original tool protocol.

    Accept a complete JSON envelope or named function with a JSON object.
    Arguments and query text are never invented. Keep the raw model output.
    """
    if '<tool_call>' in text or '<response>' in text:
        return text, 'unchanged'
    prefix, tail = '', text.strip()
    if tail.startswith('<think>') and '</think>' in tail:
        end = tail.index('</think>') + len('</think>')
        prefix, tail = tail[:end] + '\n', tail[end:].strip()
    if tail.startswith('```json\n') and tail.endswith('```'):
        tail = tail[len('```json\n'):-3].strip()
    supported = {'text_search', 'visit_webpage', 'inspect_av'}
    form = 'json_envelope'
    match = re.fullmatch(r'(text_search|visit_webpage|inspect_av)\(\s*(\{.*\})\s*\)', tail, re.S)
    try:
        if match:
            args = json.loads(match.group(2))
            call = {'name': match.group(1), 'arguments': args}
            form = 'named_json_object'
        else:
            call = json.loads(tail)
    except (ValueError, TypeError):
        return text, 'unchanged'
    if (not isinstance(call, dict) or set(call) != {'name', 'arguments'}
            or not isinstance(call['name'], str) or call['name'] not in supported or not isinstance(call['arguments'], dict)):
        return text, 'unchanged'
    return prefix + '<tool_call>' + json.dumps(call, ensure_ascii=False) + '</tool_call>', form


def action_stopping_criteria(tokenizer, prompt_tokens: int):
    """Inspect generated tokens only; prompt examples cannot stop generation."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class CompleteActionStop(StoppingCriteria):
        stopped_on_action = False

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.shape[0] != 1:
                raise ValueError("Omni agent generation requires one conversation")
            tail = input_ids[0, prompt_tokens:]
            text = tokenizer.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            self.stopped_on_action = complete_action_end(text) is not None
            return torch.tensor([self.stopped_on_action], dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([CompleteActionStop()])


def move_inputs(inputs, device, dtype):
    """Cast floating features only; token IDs, masks and grids stay integers."""
    import torch

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        if isinstance(value, (list, tuple)):
            return type(value)(move(item) for item in value)
        return value

    return {key: move(value) for key, value in inputs.items()}


def tensor_metadata(inputs):
    result = {}
    for key, value in inputs.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "device": str(getattr(value, "device", "cpu"))}
    return result



def audio_token_binding(processor, inputs):
    """Require a token slot for every encoded audio feature before generation."""
    ids = inputs["input_ids"]
    audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    video_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    actual = int((ids == audio_id).sum().item())
    mask = inputs.get("feature_attention_mask")
    if (mask is None) != (inputs.get("input_features") is None):
        raise ValueError("Audio token/feature inputs require both features and their mask")
    expected = 0
    if mask is not None:
        input_lengths = (mask.sum(-1) - 1) // 2 + 1
        expected = int(((input_lengths - 2) // 2 + 1).sum().item())
    if actual != expected:
        raise ValueError(f"Audio token/feature length mismatch: tokens={actual}, features={expected}")
    return {"audio_token_count": actual, "expected_audio_token_count": expected,
            "video_token_count": int((ids == video_id).sum().item()),
            "audio_token_binding_verified": True}


class OmniRunner(BaseRunner):
    supports_audio_video = True

    def __init__(self, spec: ModelSpec, checkpoint: str = MODEL_ID, gpus: str = "0", dtype: str = "bfloat16"):
        super().__init__(spec)
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported Omni floating dtype")
        self.checkpoint, self.gpus, self.dtype_name = checkpoint, gpus, dtype
        self._model = self._processor = self._process_mm_info = None
        self._manifest_path = os.environ.get("ONESEARCH_OMNI_MODEL_MANIFEST", "").strip()
        self._snapshot = None
        self._stream_cache = {}
        self._last_metrics = {}
        self._artifact_identity = {}
        self._use_audio_in_video = False

    @property
    def last_metrics(self):
        return deepcopy(self._last_metrics)

    def _processor_checkpoint(self):
        if self._snapshot is not None:
            return self._snapshot
        if self._manifest_path:
            manifest = json.loads(Path(self._manifest_path).read_text())
            if (manifest.get("schema_version") != "av-omni-model-manifest-v1" or manifest.get("status") != "verified"
                    or manifest.get("repository_id") != MODEL_ID or manifest.get("all_lfs_verified") is not True):
                raise ValueError("verified Qwen2.5-Omni model manifest required")
            snapshot = Path(manifest["snapshot_path"])
            if not snapshot.is_absolute() or not snapshot.is_dir():
                raise ValueError("manifest snapshot directory is unavailable")
            if self.checkpoint != MODEL_ID and Path(self.checkpoint).resolve() != snapshot.resolve():
                raise ValueError("checkpoint does not match configured verified manifest")
            self._snapshot = str(snapshot)
        elif Path(self.checkpoint).is_dir():
            self._snapshot = self.checkpoint
        else:
            raise RuntimeError("Set ONESEARCH_OMNI_MODEL_MANIFEST to the pinned local model manifest (no automatic download)")
        return self._snapshot

    def _ensure_processor(self):
        if self._processor is None:
            from transformers import Qwen2_5OmniProcessor
            self._processor = Qwen2_5OmniProcessor.from_pretrained(
                self._processor_checkpoint(), local_files_only=True, trust_remote_code=False, use_fast=False)
        if self._process_mm_info is None:
            from qwen_omni_utils import process_mm_info
            self._process_mm_info = process_mm_info

    def load(self):
        if self._model is not None:
            return
        if not self._manifest_path:
            raise RuntimeError("Model loading requires ONESEARCH_OMNI_MODEL_MANIFEST with pinned verified artifact hashes")
        # One full rehash before weight load, never once per agent turn.
        from research.av_agent.qwen_omni_backend import _verified_snapshot
        started = time.monotonic()
        snapshot, manifest, config_hash, manifest_hash = _verified_snapshot(self._manifest_path)
        if self.checkpoint != MODEL_ID and Path(self.checkpoint).resolve() != snapshot.resolve():
            raise ValueError("checkpoint does not match configured verified manifest")
        self._snapshot = str(snapshot)
        self._artifact_identity = {"repository_id": MODEL_ID, "revision": manifest["revision"],
                                   "config_sha256": config_hash, "manifest_sha256": manifest_hash,
                                   "verified_artifact_bytes": manifest["total_size_bytes"],
                                   "verification_seconds": time.monotonic() - started}
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration
        if self.gpus.strip().lower() == "cpu":
            device_map = "cpu"
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA for Omni is unavailable; no CPU fallback")
            requested = [int(value.strip()) for value in (self.gpus or "0").split(",")]
            if any(value < 0 or value >= torch.cuda.device_count() for value in requested):
                raise ValueError("requested Omni CUDA device is unavailable")
            device_map = "auto" if len(requested) > 1 else {"": "cuda:" + str(requested[0])}
        self._ensure_processor()
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self._snapshot, torch_dtype=getattr(torch, self.dtype_name), device_map=device_map,
            local_files_only=True, trust_remote_code=False, enable_audio_output=False)
        self._model.disable_talker()
        self._model.eval()

    def _has_audio(self, block):
        declared = block.get("_has_audio")
        if declared is False:
            return False
        source = block["video"]
        parsed = urlparse(source)
        if parsed.scheme not in {"", "file"}:
            if declared is None:
                raise ValueError("remote video must declare has_audio explicitly")
            return declared
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else source)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self._stream_cache:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", str(path)],
                check=True, capture_output=True, text=True, timeout=20)
            self._stream_cache[key] = bool(json.loads(probe.stdout).get("streams"))
        return self._stream_cache[key]

    def prepare_inputs(self, contents, system_instruction=None):
        """Decode raw historical AV and return real processor tensors on CPU.

        This loads the processor only. ``last_metrics`` describes sampled FPS,
        actual input tensors and stream availability; no semantic claim follows.
        """
        self._ensure_processor()
        messages = to_omni_messages(contents, system_instruction)
        video_blocks = [block for message in messages for block in message["content"] if block["type"] == "video"]
        availability = [self._has_audio(block) for block in video_blocks]
        mixed = bool(availability) and any(availability) and not all(availability)
        use_joint = bool(availability) and all(availability)
        audios, images, videos, sampled_fps = [], [], [], []
        # Decode each original media with the official helper. Its global AV flag
        # cannot represent mixed silent/sounding history, so select per video.
        video_index = 0
        for message in messages:
            processed_blocks = []
            for block in message["content"]:
                kind = block["type"]
                if kind == "text":
                    processed_blocks.append(block)
                    continue
                has_audio = availability[video_index] if kind == "video" else False
                clean_block = {key: value for key, value in block.items() if not key.startswith("_")}
                audio, image, video, kwargs = self._process_mm_info(
                    [{"role": "user", "content": [clean_block]}],
                    use_audio_in_video=has_audio, return_video_kwargs=True)
                audios.extend(audio if audio is not None else [])
                images.extend(image if image is not None else [])
                videos.extend(video if video is not None else [])
                if kind == "video":
                    rates = kwargs.get("fps", [])
                    if len(rates) != 1 or not math.isfinite(float(rates[0])) or rates[0] <= 0:
                        raise ValueError("official helper must report one positive actual sampled video FPS")
                    sampled_fps.append(float(rates[0]))
                    if mixed and has_audio:
                        # The pinned Thinker's global RoPE switch also cannot mix
                        # joint and visual-only blocks. Explicit streams preserve
                        # real audio and video without inserting fake zero audio.
                        audio_block = {"type": "audio", "audio": clean_block["video"]}
                        for src, dst in (("video_start", "audio_start"), ("video_end", "audio_end")):
                            if src in clean_block:
                                audio_block[dst] = clean_block[src]
                        processed_blocks.append(audio_block)
                    video_index += 1
                processed_blocks.append(clean_block)
            message["content"] = processed_blocks
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(
            text=text, audio=audios or None, images=images or None, videos=videos or None,
            return_tensors="pt", padding=True,
            # The helper already applies each block's min/max pixels. The
            # processor's default resize would otherwise upscale bounded clips.
            images_kwargs={"do_resize": False},
            # ProcessorMixin ignores flat video kwargs when this dict exists.
            # Keep every video option together so AV token slots are expanded.
            videos_kwargs={"do_resize": False, "use_audio_in_video": use_joint,
                           "fps": sampled_fps[0] if sampled_fps else 2.0,
                           "do_sample_frames": False})
        if videos:
            # 4.57.6 computes every grid from one scalar FPS. Retokenize its
            # original prompt using its official expander and per-video values;
            # the actual audio/vision tensors remain exactly those it produced.
            seconds = [self._processor.video_processor.temporal_patch_size / fps for fps in sampled_fps]
            import torch
            inputs["video_second_per_grid"] = torch.tensor(seconds, dtype=torch.float32)
            if len(set(sampled_fps)) > 1:
                mask = inputs.get("feature_attention_mask")
                if mask is None:
                    audio_lengths = iter([])
                else:
                    input_lengths = (mask.sum(-1) - 1) // 2 + 1
                    audio_lengths = iter((input_lengths - 2) // 2 + 1)
                expanded = self._processor.replace_multimodal_special_tokens(
                    [text], audio_lengths, iter(inputs.get("image_grid_thw", [])),
                    iter(inputs["video_grid_thw"]), iter(seconds), use_joint,
                    25, 2.0)  # pinned Qwen2.5-Omni temporal defaults
                tokenized = self._processor.tokenizer(expanded, padding=True, return_tensors="pt")
                inputs.update(tokenized)
        self._use_audio_in_video = use_joint
        self._last_metrics = {
            "model_id": MODEL_ID, "artifact_identity": deepcopy(self._artifact_identity),
            "weights_verified_for_load": bool(self._artifact_identity),
            "audio_count": len(audios), "image_count": len(images), "video_count": len(videos),
            "video_has_audio": availability, "actual_video_fps": sampled_fps,
            "prepared_video_frames": [int(video.shape[0]) for video in videos],
            "prepared_video_shapes": [list(video.shape) for video in videos],
            "audio_sample_counts": [int(audio.shape[0]) for audio in audios],
            "video_second_per_grid": [self._processor.video_processor.temporal_patch_size / fps for fps in sampled_fps],
            "use_audio_in_video": use_joint,
            "audio_video_mode": "separate_streams_mixed_history" if mixed else "joint_av" if use_joint else "visual_or_explicit_audio",
            "input_tensors": tensor_metadata(inputs), "prompt_tokens": int(inputs["input_ids"].shape[1]),
            **audio_token_binding(self._processor, inputs),
        }
        return inputs

    def infer(self, contents, system_instruction=None, cfg=None):
        cfg = cfg or InferenceConfig()
        self.load()
        inputs = self.prepare_inputs(contents, system_instruction)
        import torch
        model = self._model
        embedding = model.thinker.get_input_embeddings() if hasattr(model, "thinker") else model.get_input_embeddings()
        device, dtype = embedding.weight.device, embedding.weight.dtype
        inputs = move_inputs(inputs, device, dtype)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        stop = action_stopping_criteria(self._processor.tokenizer, prompt_tokens)
        # Omni pads its output head beyond registered tokenizer IDs. These
        # rows have no text representation and must never be generated.
        thinker = model.thinker if hasattr(model, "thinker") else model
        if not hasattr(self, "_generation_suppressed_ids"):
            output_vocab_size = int(thinker.get_output_embeddings().weight.shape[0])
            known_ids = set(self._processor.tokenizer.get_vocab().values())
            existing = thinker.generation_config.suppress_tokens or []
            self._generation_suppressed_ids = sorted(set(existing) | (set(range(output_vocab_size)) - known_ids))
            self._generation_vocab_size = output_vocab_size
        kwargs = {"use_audio_in_video": self._use_audio_in_video, "return_audio": False,
                  "thinker_max_new_tokens": cfg.max_tokens, "thinker_do_sample": cfg.temperature > 0,
                  "thinker_stopping_criteria": stop,
                  "thinker_suppress_tokens": self._generation_suppressed_ids}
        self._last_metrics.update({"output_vocab_size": self._generation_vocab_size,
                                   "generation_suppressed_token_ids": self._generation_suppressed_ids})
        if cfg.temperature > 0:
            kwargs["thinker_temperature"] = cfg.temperature
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        ids = generated.sequences if hasattr(generated, "sequences") else generated
        output_ids = ids[:, prompt_tokens:]
        raw_text = self._processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        boundary = complete_action_end(raw_text)
        text = raw_text[:boundary] if boundary is not None else raw_text
        text, normalization = normalize_explicit_action(text)
        generated_tokens = int(output_ids.shape[1])
        self._last_metrics.update({"input_tensors_on_model": tensor_metadata(inputs),
                                   "generated_tokens": generated_tokens,
                                   "generation_seconds": time.monotonic() - started,
                                   "stopped_on_complete_action": stop[0].stopped_on_action,
                                   "complete_action": complete_action_end(text) is not None,
                                   "protocol_normalization": normalization})
        usage = {"promptTokenCount": prompt_tokens, "candidatesTokenCount": generated_tokens,
                 "totalTokenCount": prompt_tokens + generated_tokens}
        finish = "STOP" if boundary is not None or generated_tokens < cfg.max_tokens else "MAX_TOKENS"
        return _wrap_gemini(text, finish, usage, self.display_name,
                            raw_response={"text": raw_text, "metrics": self.last_metrics})

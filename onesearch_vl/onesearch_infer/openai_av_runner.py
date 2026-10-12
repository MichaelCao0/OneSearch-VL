"""Full AV containers to Gemini through a video-capable chat/completions gateway.

video_url is a gateway extension, not a universal OpenAI video input contract.
Capability must be probed on the concrete endpoint. No frame/text fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
import requests

from .gemini_av_runner import GeminiAVRunner, GeminiAVSettings, RECEIPT_SCHEMA, json_bytes
from .runners import InferenceConfig, _empty_response, _wrap_gemini

SCHEMA = "onesearch-openai-av-teacher.v1"


@dataclass(frozen=True)
class OpenAIAVSettings(GeminiAVSettings):
    schema: str = SCHEMA
    default_base_url: str = "https://llm-api.cc/api/v1"

    def status(self):
        return {**super().status(), "provider": "openai_compatible_av", "native_endpoint_required": False,
                "media_transport": "video_url_data_uri", "endpoint_av_capability_requires_probe": True}

    def resolve(self):
        model, base, key = super().resolve()
        if not model.startswith("gemini-"):
            raise ValueError("This video_url gateway adapter is scoped to Gemini AV models; other providers require a verified media contract")
        return model, base, key


class OpenAIAVRunner(GeminiAVRunner):
    @property
    def identity(self):
        return {**super().identity, "provider": "openai_compatible_av", "media_transport": "video_url_data_uri"}

    def prepare_request(self, contents, system_instruction, cfg):
        native, media = super().prepare_request(contents, system_instruction, cfg)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        for item in json.loads(native)["contents"]:
            parts = []
            for part in item["parts"]:
                if "text" in part:
                    parts.append({"type": "text", "text": part["text"]})
                else:
                    inline = part["inlineData"]
                    kind = "video_url" if inline["mimeType"].startswith("video/") else "image_url"
                    parts.append({"type": kind, kind: {"url": f'data:{inline["mimeType"]};base64,{inline["data"]}'}})
            messages.append({"role": "assistant" if item["role"] == "model" else "user", "content": parts})
        encoded = json_bytes({"model": self.model, "messages": messages, "temperature": cfg.temperature,
                              "max_tokens": cfg.max_tokens, "stream": False})
        if len(encoded) > self.settings.max_request_bytes:
            raise ValueError("Teacher request byte budget exceeded after gateway serialization")
        return encoded, media

    def infer(self, contents, system_instruction=None, cfg=None):
        cfg = cfg or InferenceConfig()
        encoded, media = self.prepare_request(contents, system_instruction, cfg)
        try:
            response = requests.post(self.base_url + "/chat/completions", data=encoded,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key},
                    timeout=self.settings.timeout_s, allow_redirects=False)
        except requests.RequestException as exc:
            return _empty_response(self.model, f"AV gateway transport failed ({type(exc).__name__}); response unavailable")
        if response.status_code != 200:
            return _empty_response(self.model, f"AV gateway HTTP {response.status_code}; response body withheld")
        try:
            value = response.json()
            choice = value["choices"][0]
            text = choice["message"]["content"]
            usage = value.get("usage", {})
            tokens = usage.get("completion_tokens")
            version = value.get("model") or self.model
            if (choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip()
                    or type(tokens) is not int or tokens <= 0 or not isinstance(version, str) or not version.strip()):
                return _empty_response(self.model, "AV gateway output incomplete, blocked, or generated-token receipt missing")
        except (ValueError, KeyError, TypeError, IndexError):
            return _empty_response(self.model, "AV gateway response schema invalid")
        receipt = {"schema": RECEIPT_SCHEMA, **self.identity, "model_version": version,
                   "input_mode": "native_audio_video", "status": "complete",
                   "request_sha256": hashlib.sha256(encoded).hexdigest(),
                   "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "generated_tokens": tokens, "media": media}
        safe_usage = {new: usage[old] for old, new in (("prompt_tokens", "promptTokenCount"),
                      ("completion_tokens", "candidatesTokenCount"), ("total_tokens", "totalTokenCount"))
                      if type(usage.get(old)) is int and usage[old] >= 0}
        return _wrap_gemini(text, "STOP", safe_usage, version, raw_response={"generation_receipt": receipt})

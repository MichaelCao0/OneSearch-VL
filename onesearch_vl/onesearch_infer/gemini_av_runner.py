"""Native Gemini AV teacher through the original OneSearch runner contract.

Uses the native generateContent REST API, not an OpenAI-compatible gateway.
Only local media is inlined. Recorded video audio is retained in the container;
no frame-only, transcription or text fallback is performed.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

import requests

from .config import ModelSpec
from .runners import BaseRunner, InferenceConfig, _empty_response, _wrap_gemini
from .teacher_actions import PROTOCOL, RESPONSE_SCHEMA, TRANSPORT_INSTRUCTION, render_teacher_action


SCHEMA = "onesearch-gemini-av-teacher.v1"
RECEIPT_SCHEMA = "onesearch-native-generation.v1"
MEDIA_MIMES = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
               ".mpeg": "video/mpeg", ".mpg": "video/mpeg", ".avi": "video/x-msvideo",
               ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class GeminiAVSettings:
    schema: str = SCHEMA
    model_env: str = "ONESEARCH_TEACHER_MODEL"
    api_key_env: str = "ONESEARCH_TEACHER_API_KEY"
    base_url_env: str = "ONESEARCH_TEACHER_BASE_URL"
    default_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: int = 120
    max_request_bytes: int = 20_000_000
    action_protocol: str = "tags"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("Teacher config permits environment variable names, not literal credentials or unknown fields")
        result = cls(**value)
        if result.schema != cls().schema:
            raise ValueError("Unsupported teacher config schema")
        if result.action_protocol not in {"tags", "json_v1"}:
            raise ValueError("Unsupported teacher action protocol")
        for name in (result.model_env, result.api_key_env, result.base_url_env):
            if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
                raise ValueError("Teacher environment variable name is invalid")
        if (type(result.timeout_s) is not int or not 1 <= result.timeout_s <= 600
                or type(result.max_request_bytes) is not int or not 128 <= result.max_request_bytes <= 20_000_000):
            raise ValueError("Invalid teacher request timeout or byte budget")
        result.endpoint_base(result.default_base_url)
        return result

    @staticmethod
    def endpoint_base(value):
        if not isinstance(value, str):
            raise ValueError("Native Gemini base URL is invalid")
        parsed = urlparse(value)
        local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if ((parsed.scheme != "https" and not local_http) or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Native Gemini requires HTTPS (loopback HTTP allowed for tests) and a credential-free API prefix")
        return value.rstrip("/")

    def status(self):
        missing = [name for name in (self.model_env, self.api_key_env) if not os.environ.get(name, "").strip()]
        return {"provider": "gemini_native", "configured": not missing,
                "missing_environment_variables": missing, "native_endpoint_required": True}

    def resolve(self):
        status = self.status()
        if not status["configured"]:
            raise RuntimeError("Teacher configuration missing: " + ", ".join(status["missing_environment_variables"]))
        model = os.environ[self.model_env].strip().removeprefix("models/")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", model):
            raise ValueError("Teacher model must be an explicit Gemini model identifier")
        base = self.endpoint_base(os.environ.get(self.base_url_env) or self.default_base_url)
        return model, base, os.environ[self.api_key_env].strip()


class GeminiAVRunner(BaseRunner):
    supports_audio_video = True

    def __init__(self, settings: GeminiAVSettings):
        self.settings = settings
        self.model, self.base_url, self._key = settings.resolve()
        super().__init__(ModelSpec(name="gemini-av", family="gemini_native", display_name=self.model,
                                 supports_video=True, supports_multi_gpu=False))

    @property
    def identity(self):
        return {"provider": "gemini_native", "model_id": self.model,
                "endpoint_sha256": hashlib.sha256(self.base_url.encode()).hexdigest(),
                "config_schema": self.settings.schema}

    def prepare_request(self, contents, system_instruction, cfg):
        if (type(cfg.max_tokens) is not int or cfg.max_tokens < 1
                or isinstance(cfg.temperature, bool) or not math.isfinite(cfg.temperature)
                or not 0 <= cfg.temperature <= 2):
            raise ValueError("Invalid teacher generation budget")
        converted, media = [], []
        raw_bytes = 0
        for message in contents:
            if message.get("role") not in {"user", "model"}:
                raise ValueError("Teacher history requires original user/model roles")
            parts = []
            for part in message.get("parts", []):
                if set(part) == {"text"} and isinstance(part["text"], str):
                    parts.append({"text": part["text"]})
                    continue
                if "video_url" in part or "image_url" in part:
                    kind = "video" if "video_url" in part else "image"
                    ref = part[kind + "_url"]
                    if not isinstance(ref, dict) or not isinstance(ref.get("url"), str):
                        raise ValueError("Native teacher media requires a local URL/path binding")
                    if kind == "video" and ref.get("native_av") is not True:
                        raise ValueError("Native AV video receipt is missing")
                    parsed = urlparse(ref["url"])
                    if parsed.scheme not in {"", "file"} or parsed.netloc:
                        raise ValueError("Native teacher requires local recorded AV, not remote or inline data URLs")
                    path = Path(unquote(parsed.path) if parsed.scheme else ref["url"])
                    mime = MEDIA_MIMES.get(path.suffix.lower())
                    if not mime or not mime.startswith(kind + "/") or not path.is_file():
                        raise ValueError("Unsupported or missing native media; normalize video to MP4/WebM first")
                    raw_bytes += path.stat().st_size
                    if raw_bytes * 4 // 3 > self.settings.max_request_bytes:
                        raise ValueError("Teacher request byte budget exceeded before media encoding")
                    data = path.read_bytes()
                elif "inline_data" in part:
                    value = part["inline_data"]
                    mime = value.get("mime_type", "")
                    if not mime.startswith("image/"):
                        raise ValueError("Only original inline image blocks are supported alongside local AV")
                    kind = "image"
                    try:
                        data = base64.b64decode(value["data"], validate=True)
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError("Invalid inline image") from exc
                    raw_bytes += len(data)
                else:
                    raise ValueError("Unsupported teacher content block; media cannot be replaced by text")
                media.append({"kind": kind, "mime_type": mime, "sha256": hashlib.sha256(data).hexdigest(),
                              "bytes": len(data)})
                parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}})
            if not parts:
                raise ValueError("Empty teacher history message")
            converted.append({"role": message["role"], "parts": parts})
        if not any(x["kind"] == "video" for x in media):
            raise ValueError("Native AV teacher request requires a recorded video")
        payload = {"contents": converted, "generationConfig": {"temperature": cfg.temperature,
                                                                "maxOutputTokens": cfg.max_tokens}}
        if self.settings.action_protocol == "json_v1":
            payload["generationConfig"].update(responseMimeType="application/json", responseSchema=RESPONSE_SCHEMA)
            system_instruction = (system_instruction or "") + "\n\n" + TRANSPORT_INSTRUCTION
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        encoded = json_bytes(payload)
        if len(encoded) > self.settings.max_request_bytes:
            raise ValueError("Teacher request byte budget exceeded after serialization")
        return encoded, media

    def infer(self, contents, system_instruction=None, cfg=None):
        cfg = cfg or InferenceConfig()
        encoded, media = self.prepare_request(contents, system_instruction, cfg)
        endpoint = f"{self.base_url}/models/{self.model}:generateContent"
        # No transport retries: the queue preserves failed attempts and their
        # charged request count, including uncertain responses after a timeout.
        try:
            response = requests.post(endpoint, data=encoded,
                    headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
                    timeout=self.settings.timeout_s, allow_redirects=False)
        except requests.RequestException as exc:
            return _empty_response(self.model, f"Native teacher transport failed ({type(exc).__name__}); response unavailable")
        if response.status_code != 200:
            from .teacher_errors import summarize_http_error
            failed = _empty_response(self.model, f"Native teacher HTTP {response.status_code}; response body withheld")
            failed.setdefault("raw_response", {})["provider_error"] = summarize_http_error(response.status_code, response.content)
            return failed
        try:
            value = response.json()
            candidate = value["candidates"][0]
            parts = candidate["content"]["parts"]
            if candidate.get("finishReason") != "STOP" or not all("text" in p for p in parts):
                return _empty_response(self.model, "Native teacher did not return a complete text action/answer")
            text = "".join(p["text"] for p in parts if not p.get("thought", False))
            usage = value.get("usageMetadata", {})
            tokens = usage.get("candidatesTokenCount")
            if not text.strip() or type(tokens) is not int or tokens <= 0:
                return _empty_response(self.model, "Native teacher output or generated-token receipt is missing")
            version = value.get("modelVersion") or self.model
            if not isinstance(version, str):
                raise ValueError("invalid modelVersion")
        except (ValueError, KeyError, TypeError, IndexError):
            return _empty_response(self.model, "Native teacher response schema invalid or blocked")
        extra = {}
        if self.settings.action_protocol == "json_v1":
            provider_text = text
            try:
                text = render_teacher_action(provider_text)
            except (ValueError, TypeError, OverflowError):
                return _empty_response(self.model, "Native teacher returned an invalid structured action")
            extra["structured_action"] = {"protocol": PROTOCOL, "provider_text": provider_text}
        receipt = {"schema": RECEIPT_SCHEMA, **self.identity, "model_version": version,
                   "input_mode": "native_audio_video", "status": "complete",
                   "request_sha256": hashlib.sha256(encoded).hexdigest(),
                   "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "generated_tokens": tokens, "media": media}
        if extra:
            receipt.update(output_transport=PROTOCOL,
                           provider_output_sha256=hashlib.sha256(provider_text.encode()).hexdigest())
        safe_usage = {k: v for k, v in usage.items() if k in {"promptTokenCount", "candidatesTokenCount", "totalTokenCount"}
                      and type(v) is int and v >= 0}
        return _wrap_gemini(text, "STOP", safe_usage, version,
                            raw_response={"generation_receipt": receipt, **extra})

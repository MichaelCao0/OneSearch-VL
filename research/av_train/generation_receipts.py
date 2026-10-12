"""Generation provenance for local Omni and native remote teacher trajectories.

These records check integrity/identity, not answer correctness or model quality.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from urllib.parse import unquote, urlparse


def generation_identity(response, snapshot):
    raw = response.get("raw_response", {})
    receipt = raw.get("generation_receipt")
    if receipt is None:
        metrics = raw.get("metrics", {})
        if (metrics.get("model_id") != "Qwen/Qwen2.5-Omni-7B"
                or type(metrics.get("generated_tokens")) is not int or metrics["generated_tokens"] <= 0):
            raise ValueError("native generation receipt required")
        return {"provider": "local_omni", "model_id": metrics["model_id"]}
    if (not isinstance(receipt, dict) or receipt.get("schema") != "onesearch-native-generation.v1"
            or receipt.get("provider") not in {"gemini_native", "openai_compatible_av"} or receipt.get("status") != "complete"
            or receipt.get("input_mode") != "native_audio_video"
            or type(receipt.get("generated_tokens")) is not int or receipt["generated_tokens"] <= 0):
        raise ValueError("invalid native teacher generation receipt")
    if receipt["provider"] == "openai_compatible_av" and receipt.get("media_transport") != "video_url_data_uri":
        raise ValueError("unverified gateway media serialization")
    if not all(isinstance(receipt.get(k), str) and receipt[k].strip() for k in ("model_id", "model_version")):
        raise ValueError("teacher model identity is missing")
    for key in ("request_sha256", "output_sha256", "endpoint_sha256"):
        if not isinstance(receipt.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt[key]):
            raise ValueError("invalid teacher generation hash")
    candidates = response.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        raise ValueError("incomplete native teacher generation")
    text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
    if hashlib.sha256(text.encode()).hexdigest() != receipt["output_sha256"]:
        raise ValueError("teacher generated text changed after receipt")
    structured = raw.get("structured_action")
    transport = receipt.get("output_transport")
    if structured is not None or transport is not None:
        from onesearch_vl.onesearch_infer.teacher_actions import PROTOCOL, render_teacher_action
        if (receipt["provider"] != "gemini_native" or transport != PROTOCOL
                or not isinstance(structured, dict) or structured.get("protocol") != PROTOCOL
                or not isinstance(structured.get("provider_text"), str)):
            raise ValueError("invalid structured teacher output provenance")
        source = structured["provider_text"]
        if (hashlib.sha256(source.encode()).hexdigest() != receipt.get("provider_output_sha256")
                or render_teacher_action(source) != text):
            raise ValueError("teacher selected action changed during protocol serialization")
    expected = []
    for message in snapshot[:-1]:
        blocks = message.get("content")
        for block in blocks if isinstance(blocks, list) else []:
            kind = block.get("type")
            if kind not in {"video_url", "image_url"}:
                continue
            source = block.get(kind, {}).get("url", "")
            parsed = urlparse(source)
            if parsed.scheme not in {"", "file"} or parsed.netloc:
                raise ValueError("teacher media binding is unavailable")
            path = Path(unquote(parsed.path) if parsed.scheme else source)
            if not path.is_file():
                raise ValueError("teacher input media missing")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(chunk)
            expected.append({"kind": "video" if kind == "video_url" else "image", "sha256": h.hexdigest(),
                             "bytes": path.stat().st_size})
    media = receipt.get("media")
    if not isinstance(media, list) or not all(isinstance(row, dict) for row in media):
        raise ValueError("teacher native media receipts missing")
    actual = [{k: row.get(k) for k in ("kind", "sha256", "bytes")} for row in media]
    if not expected or not any(r["kind"] == "video" for r in expected) or actual != expected:
        raise ValueError("teacher media changed or lost after generation")
    return {k: receipt[k] for k in ("provider", "model_id", "model_version", "endpoint_sha256")}

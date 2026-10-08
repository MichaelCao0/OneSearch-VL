"""Judge transport and strict response validation for the current reward."""
from __future__ import annotations
import json
import os
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_TRANSIENT_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class RewardError(RuntimeError):
    """An EVGR request or rubric-response validation failed."""


class RequestError(RewardError):
    """A full HTTP request sequence failed after a known number of calls."""

    def __init__(self, message: str, *, attempts: int):
        super().__init__(message)
        self.attempts = int(attempts)


def parse_json(text: str) -> Mapping[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RewardError(f"EVGR returned non-JSON content: {cleaned[:240]!r}") from exc
    if not isinstance(value, Mapping):
        raise RewardError("EVGR response must be one JSON object")
    return value


@dataclass(frozen=True)
class JudgeConfig:
    api_base: str
    api_key: str
    model: str = "Doubao-Seed-2.0-pro"
    backend: str = "seed"
    timeout_s: float = 90.0
    max_attempts: int = 3
    max_tokens: int = 500

    def validate(self) -> JudgeConfig:
        if not self.api_base.strip():
            raise RewardError(f"{self.backend} EVGR api_base is required")
        if not self.api_key.strip():
            raise RewardError(f"{self.backend} EVGR api_key is required")
        if not self.model.strip():
            raise RewardError(f"{self.backend} EVGR model is required")
        if self.timeout_s <= 0 or self.max_attempts < 1 or self.max_tokens < 1:
            raise RewardError(f"invalid {self.backend} EVGR configuration")
        return self

    @classmethod
    def from_env(cls) -> JudgeConfig:
        api_key = os.getenv("SEED_API_KEY", "").strip()
        if not api_key:
            raise RewardError("SEED_API_KEY is required when EVGR is enabled")
        return cls(
            api_base=os.getenv(
                "SEED_API_BASE", ""
            ).rstrip("/"),
            api_key=api_key,
            model=os.getenv("EVGR_JUDGE_MODEL", os.getenv("SEED_MODEL", "Doubao-Seed-2.0-pro")),
            timeout_s=float(os.getenv("EVGR_JUDGE_TIMEOUT", "90")),
            max_attempts=int(os.getenv("EVGR_JUDGE_MAX_ATTEMPTS", "3")),
            max_tokens=int(os.getenv("EVGR_JUDGE_MAX_TOKENS", "500")),
        ).validate()


class JudgeClient:
    """Synchronous client with injectable transport for offline replay."""

    def __init__(
        self,
        config: JudgeConfig,
        request_fn: Callable[..., Any] | None = None,
    ):
        self.config = config
        self.request_fn = request_fn

    def _post(self, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
        if self.request_fn is None:
            import requests

            request_fn = requests.post
        else:
            request_fn = self.request_fn
        url = f"{self.config.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = request_fn(
                    url=url,
                    headers=headers,
                    json=dict(payload),
                    timeout=(10.0, self.config.timeout_s),
                )
                if isinstance(response, Mapping):
                    return response, attempt
                status = int(getattr(response, "status_code", 0))
                if status == 200:
                    data = response.json()
                    if not isinstance(data, Mapping):
                        raise RequestError(
                            f"{self.config.backend} EVGR HTTP body is not a JSON object",
                            attempts=attempt,
                        )
                    return data, attempt
                body = str(getattr(response, "text", ""))[:240]
                error = RewardError(
                    f"{self.config.backend} EVGR HTTP {status}: {body}"
                )
                if status not in _TRANSIENT_HTTP_STATUS:
                    raise RequestError(str(error), attempts=attempt)
                last_error = error
                retry_after = getattr(response, "headers", {}).get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after is not None else 0.0
                except (TypeError, ValueError):
                    wait_s = 0.0
            except RequestError:
                raise
            except RewardError as exc:
                raise RequestError(str(exc), attempts=attempt) from exc
            except Exception as exc:  # network-bound transport
                last_error = exc
                wait_s = 0.0
            if attempt < self.config.max_attempts:
                wait_s = max(wait_s, min(8.0, float(2 ** (attempt - 1))))
                time.sleep(wait_s + random.uniform(0.0, 0.25))
        raise RequestError(
            f"{self.config.backend} EVGR failed after "
            f"{self.config.max_attempts} attempts: {last_error}",
            attempts=self.config.max_attempts,
        )

def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def decode_judgment(text, targets):
    value = parse_json(text)
    items = value.get("items")
    expected = {t["target_id"] for t in targets}
    if not isinstance(items, list) or len(items) != len(expected) or any(not isinstance(i, dict) for i in items):
        raise RewardError("Missing or malformed per-target judgments")
    if {i.get("target_id") for i in items} != expected:
        raise RewardError("Unknown, duplicate or missing target ID")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise RewardError("Missing overall reason")
    for item in items:
        if not isinstance(item.get("reason"), str) or not item["reason"].strip() or not isinstance(item.get("citations"), list):
            raise RewardError("Missing target reason/citation list")
    return value


def holistic_score(value, credits):
    """Validate a continuous score without rounding or aggregating diagnostics."""
    score = value.get("score")
    if type(score) not in (int, float) or not 0.0 <= score <= 1.0:
        raise RewardError("Holistic score must be a finite JSON number in [0, 1]")
    if not credits:
        raise RewardError("Holistic score requires target diagnostics")
    if score > 0 and not any(credit > 0 for credit in credits):
        raise RewardError("Positive holistic score requires a supported target with validated citations")
    if score == 1 and any(credit != 1 for credit in credits):
        raise RewardError("Holistic score 1 requires every target to be fully supported or grounded")
    return float(score)

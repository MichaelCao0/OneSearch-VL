"""Stop native-teacher queues on service failures without retrying requests.

Each queue has a persistent local stop receipt. An optional run-scoped shared
path stops sibling queues too. Already in-flight calls remain charged; pending
work never creates an attempt. Only sanitized error classes/status are recorded.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


def provider_failure(response):
    candidates = response.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "ERROR":
        return None
    text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
    status = re.search(r"teacher HTTP (\d{3})", text)
    if status:
        return {"kind": "provider_http", "http_status": int(status[1])}
    for fragment, kind in (("transport failed", "provider_transport"),
                           ("generated-token receipt is missing", "provider_missing_usage"),
                           ("response schema invalid", "provider_invalid_response")):
        if fragment in text:
            return {"kind": kind}
    return None


class ProviderGuard:
    def __init__(self, queue_dir):
        self.paths = [Path(queue_dir) / "provider-stop.json"]
        shared = os.environ.get("ONESEARCH_TEACHER_STOP_FILE")
        if shared and Path(shared) not in self.paths:
            self.paths.append(Path(shared))

    @property
    def stopped(self):
        return any(p.exists() for p in self.paths)

    def observe(self, response):
        failure = provider_failure(response)
        if failure:
            record = {"schema": "onesearch-provider-stop.v1", **failure,
                      "at_utc": datetime.now(timezone.utc).isoformat(),
                      "automatic_retries": 0, "in_flight_calls_may_finish": True}
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with path.open("x") as stream:
                        json.dump(record, stream, indent=2)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                except FileExistsError:
                    pass
        return failure

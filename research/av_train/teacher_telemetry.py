"""Opt-in timing around unchanged teacher calls; no prompts, keys or media logged.

API wait includes transport and response parsing. Preparation measures local media
read/hash/base64/serialization. ContextVar isolates concurrent per-call receipts.
The rollout CLI invokes the original teacher loop with the same arguments.
"""
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

_CURRENT=ContextVar("onesearch_teacher_profile", default=None)

def now():
    return datetime.now(timezone.utc).isoformat()

def instrument_runner_class(cls):
    if cls.__dict__.get("_onesearch_profile_installed", False): return
    original_infer=cls.infer
    original_prepare=cls.prepare_request
    @wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        record=_CURRENT.get()
        started=time.perf_counter()
        try:
            encoded, media=original_prepare(self, *args, **kwargs)
            if record is not None:
                record["request_bytes"]=len(encoded)
                record["media_count"]=len(media)
                record["raw_media_bytes"]=sum(m.get("bytes", 0) for m in media)
            return encoded, media
        finally:
            if record is not None: record["prepare_s"]+=time.perf_counter()-started
    @wraps(original_infer)
    def infer(self, *args, **kwargs):
        folder=os.environ.get("ONESEARCH_TEACHER_PROFILE_DIR")
        if not folder: return original_infer(self, *args, **kwargs)
        record={"schema":"onesearch-teacher-timing.v1", "stage":os.environ.get("ONESEARCH_TEACHER_PROFILE_STAGE", "unspecified"),
                "started_at":now(), "prepare_s":0., "error_type":None, "pid":os.getpid()}
        token=_CURRENT.set(record); started=time.perf_counter()
        try:
            response=original_infer(self, *args, **kwargs)
            candidates=response.get("candidates", [])
            record["finish_reason"]=candidates[0].get("finishReason") if candidates else None
            usage=response.get("usageMetadata", {})
            for source, dest in [("promptTokenCount", "prompt_tokens"), ("candidatesTokenCount", "generated_tokens")]:
                if type(usage.get(source)) is int: record[dest]=usage[source]
            from research.av_train.provider_guard import provider_failure
            failure=provider_failure(response)
            if failure: record["provider_failure"]=failure
            diagnostics=response.get("raw_response", {}).get("provider_error")
            if isinstance(diagnostics, dict):
                record["provider_error"]={key:diagnostics[key] for key in
                    ("http_status", "category", "body_bytes", "body_sha256", "provider_status") if key in diagnostics}
            return response
        except Exception as exc:
            record["error_type"]=type(exc).__name__
            raise
        finally:
            record["elapsed_s"]=time.perf_counter()-started
            record["api_wait_and_parse_s"]=max(0., record["elapsed_s"]-record["prepare_s"])
            record["finished_at"]=now(); _CURRENT.reset(token)
            out=Path(folder); out.mkdir(parents=True, exist_ok=True)
            path=out/(uuid.uuid4().hex+".json")
            descriptor=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(record, stream, sort_keys=True); stream.write("\n")
    cls.prepare_request=prepare; cls.infer=infer; cls._onesearch_profile_installed=True

def install():
    # Both import names are used by original runner and data-authoring modules.
    importlib.import_module("research.av_train.teacher_rollouts")
    for name in ["onesearch_infer.gemini_av_runner", "onesearch_vl.onesearch_infer.gemini_av_runner"]:
        instrument_runner_class(importlib.import_module(name).GeminiAVRunner)

def summarize(folder):
    records=[json.loads(p.read_text()) for p in Path(folder).glob("*.json")]
    stages={}
    for stage in sorted({r["stage"] for r in records}):
        rs=[r for r in records if r["stage"]==stage]
        latencies=sorted(r["elapsed_s"] for r in rs)
        events=[]
        for r in rs:
            events.extend([(datetime.fromisoformat(r["started_at"]).timestamp(), 1),
                           (datetime.fromisoformat(r["finished_at"]).timestamp(), -1)])
        active=peak=0
        for _,delta in sorted(events): active+=delta; peak=max(peak, active)
        window=max(x[0] for x in events)-min(x[0] for x in events)
        total=sum(latencies); prep=sum(r["prepare_s"] for r in rs)
        stages[stage]={"calls":len(rs), "max_inflight":peak, "window_s":window,
            "calls_per_minute":60*len(rs)/window if window else None,
            "median_s":statistics.median(latencies), "p90_s":latencies[int((len(latencies)-1)*.9)],
            "preparation_fraction":prep/total if total else None, "prepare_sum_s":prep,
            "api_wait_and_parse_sum_s":sum(r["api_wait_and_parse_s"] for r in rs),
            "request_bytes_sum":sum(r.get("request_bytes",0) for r in rs),
            "failures":sum(r.get("error_type") is not None or r.get("finish_reason")=="ERROR" for r in rs),
            "provider_failures":sum(bool(r.get("provider_failure")) for r in rs)}
    return {"schema":"onesearch-teacher-performance.v1", "at_utc":now(), "stages":stages}

def main():
    if len(sys.argv)<2: raise SystemExit("Expected rollout or summarize")
    mode=sys.argv[1]
    if mode=="summarize":
        print(json.dumps(summarize(sys.argv[2]), indent=2)); return
    if mode!="rollout": raise SystemExit("Unsupported profiled teacher command")
    os.environ["ONESEARCH_TEACHER_PROFILE_STAGE"]="rollout"; install()
    if int(os.environ.get("ONESEARCH_TEACHER_TRANSIENT_LIMIT", "1"))>1:
        from research.av_train.provider_resilience import install as install_resilience
        install_resilience()
    sys.argv=["research.av_train.teacher_rollouts"]+sys.argv[2:]
    importlib.import_module("research.av_train.teacher_rollouts").main()

if __name__=="__main__": main()

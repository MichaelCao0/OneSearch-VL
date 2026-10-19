"""Opt-in burst circuit for new data runs; no request is retried.

Record isolated sample errors (400/413/422) and HTTP 500/502/503/504 failures and proceed to untouched samples.
Three such failures within 60 seconds pause the run. Default retains the original
first-error stop policy. Auth/rate-limit failures stop immediately. Transport/schema failures use the same burst circuit.
"""
import fcntl
import json
import os
from pathlib import Path
import time
from .provider_guard import ProviderGuard, provider_failure

class BurstProviderGuard(ProviderGuard):
    def observe(self, response):
        failure=provider_failure(response)
        limit=int(os.environ.get("ONESEARCH_TEACHER_TRANSIENT_LIMIT", "1"))
        if limit<1 or limit>8: raise ValueError("Transient failure limit must be within 1..8")
        transient=failure and (failure.get("http_status") in {400,413,422,500,502,503,504}
                              or failure.get("kind") in {"provider_transport","provider_missing_usage","provider_invalid_response"})
        if not transient or limit==1: return super().observe(response)
        shared=os.environ.get("ONESEARCH_TEACHER_STOP_FILE")
        base=Path(shared) if shared else self.paths[0]
        path=base.with_name(base.name+".transient.json");path.parent.mkdir(parents=True,exist_ok=True)
        lock_path=path.with_suffix(path.suffix+".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            old=json.loads(path.read_text()) if path.exists() else {"failures":[], "total_failures":0}
            at=time.time(); recent=[v for v in old["failures"] if at-v["at_epoch"]<60]
            recent.append({"at_epoch":at, **failure})
            record={"schema":"onesearch-provider-burst.v1", "failures":recent,
                    "total_failures":old["total_failures"]+1,"limit":limit,"window_s":60,"automatic_retries":0}
            temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(record)+"\n");temporary.replace(path)
            if len(recent)>=limit: super().observe(response)
        return failure

def install():
    from . import teacher_rollouts,teacher_silver_review,question_authoring,scale_searchqa
    for module in [teacher_rollouts,teacher_silver_review,question_authoring,scale_searchqa]:
        module.ProviderGuard=BurstProviderGuard

# Semantic observer contract and CPU preflight

Development remains in workspace 3, `/home/jovyan/OneSearch-VL`, branch `research/av-deepresearch`. This work implements an input/observation boundary and eight fixed semantic probe templates. It does not deploy a model, contact an API or complete semantic validation.

## Integration API

```python
from research.av_agent.semantic_observer import (
    SemanticObserver, make_observer_input, validate_observation,
)

runtime = SemanticObserver(backend=None, max_calls=20)
bound = make_observer_input(source_record, derived_record)
private_observation = runtime.observe(bound)
assert validate_observation(private_observation) == []
```

`make_observer_input(source, derived, question=DEFAULT_QUESTION, probe_id=None)` verifies the existing DerivedRecord and its original source. Existing validators rehash original/clip bytes and re-probe presentation clocks; these real ffprobe operations must remain within the media cost collection context. The minimized input contains source/derived opaque IDs, original/clip hashes, clip bytes, original half-open interval, reversible shared-clock mapping, modalities, bounded question, optional probe ID and a canonical `input_sha256`. Only the trusted private record contains the clip path. Publisher metadata, original source URL, gold and future observations are excluded.

`validate_observer_input(record, verify_files=False)` audits the JSON binding, clock mapping and digest. With `verify_files=True`, it also rehashes actual clip bytes. `observe` performs that byte check before invoking the backend. Invalid bindings raise `ValueError` before inference; actor-side validation remains the dispatcher's responsibility.

`SemanticObserver(backend=None, max_calls=20)` does not select or call a service from ambient config. Every valid observation request increments `counts.calls`. Missing backend, backend errors, malformed replies and budget blocks increment `counts.failures`; budget blocks preserve the denominator and do not invoke the backend. `counts` also includes `backend_attempts`, `provider_requests_known` and `provider_requests_unknown_attempts`. The call budget includes unavailable calls and is owned by this observer instance.

`capabilities` reports `semantic_observer_contract_ready`, `backend_configured`, explicit mode, model/revision and pending human validation. It always reports `raw_av_observer_ready=False`. The v0.1 writer cannot approve human semantic review; a future independently adjudicated probe report must supply that capability decision.

## Future backend boundary

A trusted backend exposes `.manifest` and `.infer(bound_input)`. The manifest has exactly these fields:

| Field | Contract |
|---|---|
| `backend_id`, `model_id` | Nonempty bounded identifiers |
| `model_revision` | Pinned 40–64 hex revision; `main` is rejected |
| `config_sha256` | Configuration digest, never raw credentials |
| `mode` | `native_joint_av`, `asr_plus_frames`, `asr_only`, or `visual_only` |
| `transport` | `local` or `service` |
| `execution_kind` | `model_inference` or explicitly non-model `contract_test` |
| `runtime_device`, `backend_code_revision` | Bounded execution identity |

The reply has exactly `backend_identity`, `consumed_input_sha256`, `raw_output`, `structured_observation`, and `provider_requests`. The backend must report the hash of the input it actually consumed, preserve exact bounded raw text and count internal provider retries. Expected and returned identities must match. HTTP success alone is insufficient.

A structured observation contains `events`, explicit `uncertainty`, and `cannot_determine`. Each event contains only `modality` (`audio`, `visual`, or `joint_av`), `description`, `source_interval`, and `uncertainty`. Event intervals must lie wholly in the requested source-local interval; an output cannot claim an input modality absent from the clip. `visual_only` cannot claim audible events and `asr_only` cannot claim visual events. These checks establish contract validity, not semantic accuracy or causal attribution. Model candidates cannot automatically become citations.

The observation record preserves the private input, pinned expected identity, exact raw output and its SHA-256, structured candidate output, safe errors, actual observer cost, pending human validation and a canonical `record_sha256`. Available statuses are `unavailable`, `failed`, `budget_exceeded`, `contract_test_executed`, and `model_executed`. A trusted backend's execution declaration is an attestation, not independent proof of model capability. Contract fixtures used in tests are labelled; they do not count as actual inference evidence. Integrity hashes are not authentication signatures.

Provider exceptions retain only the exception class. An exception from a service leaves `provider_requests=None` and increments unknown-work attempts; it does not invent zero requests. Backend return errors preserve bounded raw output if available, discard invalid structured support, and retain failure in the denominator. Observer wall time measures its call boundary; prior media preparation/validation remains separately metered media work.

## CPU evidence and eight probe categories

Run a fresh non-network inventory into a new directory:

```bash
pixi run python -m research.av_agent.semantic_observer \
  --preflight --output-dir research/runs/<new-run>/observer-preflight
```

The CLI refuses to overwrite existing directories. It saves `preflight.json`, `candidate-model-manifest.json`, `probe-templates.json` and a README. It reads only presence booleans for the four dedicated `AV_OBSERVER_*` variables, without reading or printing endpoint/key values or unrelated config files. Environment-variable presence is not endpoint reachability or authenticated-service verification.

The observed runtime is torch `2.8.0+cpu`, no CUDA devices and no MPS. None of the four dedicated observer variables was present. The plan's `Qwen/Qwen2.5-Omni-7B` remains an unpinned, undeployed candidate; no weights or model configs were downloaded. OneSearch-VL-8B visual smoke and ASR+frames cannot establish native joint-AV capability.

The eight fixed templates cover visible speaker/speech, nonverbal impact or sound change, offscreen sound candidates, narration/editing, known AV timing offset, silence/audio masking, wrong requested interval and compatible-source/cross-source identity ambiguity. Every template keeps input/control bindings and independent human reference evidence pending. Actual model executions and human semantic passes are both zero; the planned denominator is eight and blocked categories remain visible. Natural and synthetic timing results must be reported separately, including missing backend, failures and budget blocks.

Artifacts are private under `research/runs/next-stage-20261010-v1/observer-v1/`. A bounded real synthetic-media smoke successfully bound/rehashed a `[0.5,1.5)` AV clip and recorded one `unavailable` observation, zero backend attempts and zero provider requests. It is an input/clock proof, not a semantic probe result. Existing runs were preserved.

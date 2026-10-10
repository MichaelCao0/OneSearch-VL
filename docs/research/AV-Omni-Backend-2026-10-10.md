# Bounded local native AV observer

The adapter is `research/av_agent/qwen_omni_backend.py`; the independent worker
CLI is `research/scripts/run_av_observer_smoke.py`. It is not integrated into
`EpisodeTools`. The model is `Qwen/Qwen2.5-Omni-7B`, with the runtime acquisition
pinned to `ae9e1690543ffd5c0221dc27f79834d0294cba00`. Its isolated dependencies and
full verification recipe are in `runtime/av-observer/`; the default CPU Pixi
manifest and lock remain independent.

The backend accepts only an existing absolute local snapshot and the verified
`av-omni-model-manifest-v1` acquisition manifest. It freezes manifest bytes,
records their SHA-256 and rehashes every listed file before loading. Each file
must match its pinned expected SHA-256 and size; config and weight artifacts
are required. `local_files_only=True` and `trust_remote_code=False` prevent
implicit downloads and remote-code execution. Integrity depends on the
trusted downloader's pinned HF metadata; hashes are not semantic truth or
signatures.

The supplied derived clip contains both native streams on one shared clock.
`use_audio_in_video=True` is passed to `process_mm_info`, the official
processor and `generate`; raw audio and raw frames both enter the model.
`enable_audio_output=False` avoids constructing Talker/token2wav at load, and
`return_audio=False` returns text. `return_video_kwargs=True` preserves the
helper's actual sampled FPS, which can differ from the request due to frame
rounding or the four-frame minimum on short clips. The sole FPS-list element
is validated as positive/finite and forwarded as a scalar to the pinned Omni
processor. `do_sample_frames=False` avoids a second sampling pass. Detailed
costs retain requested and actual FPS plus measured prepared-frame counts. The official wrapper loads `spk_dict.pt`
even with Talker disabled, so runtime acquisition includes and verifies it.
The text budget is passed as `thinker_max_new_tokens`, because a generic
`max_new_tokens` is overridden by the wrapper's default Thinker budget.
Completion decoding slices off the input tokens; exact decoded raw text is
retained separately from the structured candidate observation.

Only media and the bounded question are supplied to the model. Source hashes,
opaque IDs, source clock offsets, metadata, source URLs, answers and review
material are not rendered into the prompt. The model reports `clip_interval`
within `[0, clip_duration)` and the backend validates that complete interval
before applying the verified `output_to_source` mapping. The source offset is
never guessed from a sampled video grid. Duplicate JSON keys, nonfinite
numbers, invalid fields, modality claims and outside-clip intervals produce
an empty event list with `observer_parse_issue:<code>` in `cannot_determine`.
Raw text stays intact and the private cost sidecar records the parse issue.
A completed generation with abstention remains model execution, while JSON
contract validity and independent semantic accuracy are reported separately.

## Input and worker command

Prepare a private JSON file using existing `make_observer_input` bindings:

```json
{
  "schema_version": "av-observer-smoke-bindings-0.1",
  "inputs": ["replace this string with complete exact observer input objects"]
}
```

The worker validates all binding digests and clip bytes before model load.
Callers select fixed natural clips; this worker does not synthesize probe gold.
The hard smoke limits are 6 clips, 2 seconds per clip, 1024 completion tokens
(default 256), video sampling at most 4 fps (default 2), and at most
`256 * 28 * 28` pixels per prepared frame. Limits may be reduced. These limits
bound prepared inputs; decoder/frame work is not inferred from sample counts.

```bash
# First use the existing CPU default to check argument/import safety.
pixi run python research/scripts/run_av_observer_smoke.py --help

# Actual worker: run only on the separately allocated GPU, with the locked
# runtime and verified manifest visible at the same shared mount paths.
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python \
  research/scripts/run_av_observer_smoke.py \
  --bindings research/runs/<run>/bindings.private.json \
  --model-manifest research/runs/<run>/runtime-setup/model-manifest.json \
  --output-dir research/runs/<run>/observer-worker-v1
```

Every output directory must be new; it is created with mode `0700`.
`observations.private/*.json` retain input paths, exact raw output, candidate
structure and existing observer costs. `costs.private.jsonl` is persisted per
attempt and contains preprocessing/processor/generation/decode/parse/total
wall seconds, input/generated tokens, safe error class, parse issue and CUDA
allocator peaks. `model-identity.private.json` preserves the artifact/backend
identities, weight verification time, load time and load-memory peak.
`run-manifest.private.json` records bindings, source/code/lock hashes and
bounded settings. Before model load, `live-runtime.private.json` records the
actual interpreter/Python series, imported six core package versions and
module hashes, Torch CUDA build, and the imported model/processor/AV helper
source hashes. These are compared with `runtime/av-observer/runtime-manifest.json`
and its lock hash; mismatch or unavailable identity blocks every planned clip
before inference. The run identity and public summary retain the live receipt
SHA-256 and status, while paths stay private. A safe load failure records zero attempted clips and keeps
all planned inputs visible as blocked, without printing private exception text.

`summary.public.json` contains only hashes, model identity, aggregate counts,
budgets, elapsed costs and pending review status; it omits paths, raw outputs,
questions, source IDs and model descriptions. It is a publication candidate
for a separate audit. Peak memory is the PyTorch CUDA allocator's measured
peak, including model load; driver/decoder/non-PyTorch allocations are outside
that measurement. Failed local attempts still count, with zero external
provider requests. Failure and parsing counts stay in the fixed denominator.

No output grants `raw_av_observer_ready=True`. Independent human semantic
validation remains pending. Mocked tests are contract checks, not model
capability evidence. Natural observations and synthetic timing controls must
be evaluated and reported separately under the existing eight-category probe
protocol.

Validation:

```bash
pixi run python -m unittest discover -s validation -p test_av_omni_backend.py -v
pixi run python -m unittest discover -s validation/research -p test_semantic_observer.py -v
# Adds real CPU AV decoding + the pinned official processor clock boundary.
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -m unittest discover -s validation -p test_av_omni_backend.py -v
```

## Native array clock limitation

The pinned multimedia helper drops leading container PTS when producing audio/video arrays. `check_native_clip_clocks` rejects stream origins beyond 2 ms of zero before helper execution; this allowance covers container rounding only and does not validate fine AV timing. It never pads or reselects clips. The first fixed natural input has audio start 1.012 s and remains in the failed denominator; its generation is blocked. Per-attempt metrics are still preserved, and later fixed inputs continue. Nonzero stream origins require a future explicit timestamp-aware input representation.

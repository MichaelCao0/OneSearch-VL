# OneSearch-VL · Inference Reference

[Inference & Evaluation](README.md) · [Project Overview](../README.md)

This reference covers input schemas, tool behavior, video preprocessing, and
how evaluation matches and scores saved trajectories. Start with the
[quick-start guide](README.md) for installation and commands.

## Service Configuration

The shell wrappers load `onesearch_vl/.env`, or the file selected by `ENV_FILE`.
Values in the selected file replace inherited environment variables with the
same names. Direct Python entry points require these settings to be exported.

Direct retrieval uses Serper text search and Lens, Jina page reading, and an
OpenAI-compatible summary model. The optional retrieval gateway uses `API_HOST`,
`API_USER`, and `API_KEY` together, plus `IMAGE_SEARCH_URL` for its Lens endpoint.
Partial gateway configuration returns an error. `API_GATEWAY_*`, `SERP_API_KEY`,
and `QWEN_MODEL` are accepted aliases when the corresponding canonical variables
are empty.

Image search needs a URL reachable by the retrieval service. The bundled
`upload.py` copies generated media into `ASSET_DIR` and returns URLs under
`ASSET_BASE_URL`; an existing static server must serve that directory. Set
`COS_UPLOAD_PATHS` to select a custom `upload_cos` implementation.

## Input Formats and Model Backends

### Single image

Provide a parquet dataset with a question and one resolvable image per row:

| Field | Accepted form |
|---|---|
| `question` | Question string; alternatively `prompt` may contain a message list |
| `images` | A one-element list containing image bytes, a local path, an HTTP(S) URL, or a dictionary with `bytes` / `url` |
| `image` | Alternative singular image column |
| `data_id` | Optional identifier saved with the trajectory; otherwise the row index is used to construct one |
| `answer` | Reference answer for subsequent evaluation |

For example, `images=[{"bytes": encoded_image_bytes}]` embeds the image in
parquet. Keep reference-answer rows aligned with the inference data.

```bash
python onesearch_vl/run_infer.py --modality single_image --model vllm \
  --data-path /data/single_image.parquet --output-dir ./outputs/single_image
```

### Multiple images

Provide a QA CSV with `index`, `question`, and `answer`, plus a JSONL manifest
whose `case_id` matches the integer CSV index. Each record supplies HTTP(S)
image URLs with a stable index mapping:

```json
{"case_id":"0","modality":"multi_image","ok":true,"n_images":2,"image_urls":["https://assets.example.org/a.png","https://assets.example.org/b.png"]}
```

```bash
python onesearch_vl/run_infer.py --modality multi_image --model vllm \
  --csv /data/multi_image.csv --manifest /data/mi_manifest.jsonl \
  --output-dir ./outputs/multi_image
```

Images become `img_1`, `img_2`, and so on. This modality has no video timeline
and does not expose `select_timespan` or `select_frame`.

### Video

Provide a QA CSV and a video/frame-grid JSONL manifest with matching case IDs:

```bash
python onesearch_vl/run_infer.py --modality video --model vllm \
  --csv /data/video.csv --manifest /data/grids.jsonl --output-dir ./outputs/video
```

To prepare raw silent MP4s, name them after their integer CSV indices, configure
the asset publisher, and run:

```bash
python onesearch_vl/prepare_video.py --videos-dir /data/videos \
  --manifest /data/grids.jsonl
```

The preprocessor samples at a target 2 FPS, caps the sequence at 128 frames,
and repacks the selected frames into a **1-FPS frame-grid video**. This changes
the playback timeline. The manifest retains source frame indices and duration
for mapping back to the original. Report this representation when evaluating;
model support for native video does not make the repacked clip an original-time
input. Inference defaults to 100,352 maximum pixels per video frame, matching SFT.

### Model backends

| `--model` | Backend | Configuration |
|---|---|---|
| `vllm` | HTTP Chat Completions | `VLLM_*` variables |
| `8b`, `32b`, `30b-a3b` | Local Qwen3-VL checkpoint | `--checkpoint`, `--gpus` |
| `vllm-local` | In-process vLLM | Local checkpoint and a compatible vLLM installation |
| `claude`, `claude-api` | Public Anthropic Messages API | `ANTHROPIC_*` variables |

For example, load the SFT checkpoint locally:

```bash
python onesearch_vl/run_infer.py --modality single_image --model 8b \
  --checkpoint /models/onesearch-sft --gpus 0 \
  --data-path /data/single_image.parquet --output-dir ./outputs/local_single_image
```

The `vllm-frames` variant explicitly sends video frames as images. Keep its
results separate from video-input runs. List the complete arguments with:

```bash
python onesearch_vl/run_infer_single.py --help
python onesearch_vl/run_infer_mi.py --help
python onesearch_vl/run_infer_video.py --help
```

The router's `--dry-run` prints the selected command.

## Tool Environment

| Tool | Operation |
|---|---|
| `select_timespan`, `select_frame` | Select temporal evidence and a frame from video |
| `crop` | Crop a region and return a new `img_*` reference |
| `image_search` | Retrieve matching titles and source URLs through Serper Lens |
| `text_search` | Search, read webpages, and summarize supporting evidence |
| `layout_parsing` | Read visible text through a configured PP-Structure endpoint |
| `perspective_correct`, `super_resolution`, `sharpen` | Enhance visual regions |

RL and inference import the same `search.py` provider implementation. Existing
`q`/`query`, `hl`/`lang`, image references, and observation envelopes are retained.
Prompts use provider-neutral names while keeping their tool parameter schemas.
The `visual_lookup` callback remains available for custom image retrieval.

If page reading or summarization fails, text search can return the retrieved
snippet or a page excerpt. Missing provider configuration returns a tool error.
The available tools follow the modality's prompt; select the tool policy
appropriate to your benchmark.

## Evaluation Protocols and Reports

`run_eval.sh image` loads `onesearch_vl/.env` and calls **`eval_image.py`** for
single-image and multi-image benchmarks. `run_eval.sh video` calls
**`eval_video.py`** for video benchmarks. Both share `eval_common.py` for gold
matching, judge transport, failure handling, and reports. Configure
`JUDGE_BASE_URL`, `JUDGE_API_KEY`, and `JUDGE_MODEL`; the default judge is
`gpt-4o`. Both protocols use Chat Completions with temperature 0
and a maximum of 1,024 output tokens. `JUDGE_TIMEOUT`, `JUDGE_MAX_RETRY`, and
`JUDGE_RETRY_SLEEP` control transport retries.

Choose the entry point by input type, then the protocol by benchmark:

| Benchmark | Entry point | `--protocol` / `EVAL_PROTOCOL` |
|---|---|---|
| Single-image and multi-image benchmarks | `eval_image.py` | `onesearch` |
| OneSearch-Video | `eval_video.py` | `videodr` (default) |
| VideoDR-Benchmark | `eval_video.py` | `videodr` (default) |

The `onesearch` protocol sends the question, reference answer and extracted
answer, expecting `correct: yes/no`. The `videodr` protocol uses the official
system and user prompts, expecting JSON `is_correct` and `reasoning`. Image
evaluation accepts only `onesearch`; video evaluation defaults to `videodr`
and accepts `--protocol onesearch` to select the alternative protocol.
Leave `EVAL_PROTOCOL` empty or unset to use these modality defaults. An explicit
`--protocol` takes precedence over `EVAL_PROTOCOL`. If an existing `.env` sets
`EVAL_PROTOCOL=onesearch`, clear it to adopt the new video default.

Both image and video evaluation prefer `\boxed{...}` or `boxed{...}` content,
including boxed answers inside `<response>`. If no complete, nonempty boxed
answer is found, extraction falls back to `<answer>`, then `<response>`, then
the final text after thinking. Both judge protocols use this same extraction
order; there is no extraction-mode switch.
The submitted answer is capped at 8,000 characters; the plain-text fallback is
capped at 2,000. Reports record `answer_extraction: boxed_first`. The fixed
prompts and extraction functions are in
[`eval_prompts.py`](eval_prompts.py). Judge inputs are text only; tool traces
and media are not attached.

```bash
# Single-image benchmark: use the same parquet supplied to inference.
bash onesearch_vl/run_eval.sh image \
  --traj_dir ./outputs/single_image --parquet /data/single_image.parquet

# OneSearch multi-image and video: use the corresponding QA CSV.
bash onesearch_vl/run_eval.sh image \
  --traj_dir ./outputs/multi_image --csv /data/multi_image.csv
bash onesearch_vl/run_eval.sh video \
  --traj_dir ./outputs/video --csv /data/video.csv

# VideoDR and OneSearch-Video both default to the official VideoDR protocol.
bash onesearch_vl/run_eval.sh video \
  --traj_dir ./outputs/videodr --csv /data/videodr.csv
```

Use exactly one of `--csv` and `--parquet`. Gold files must contain a question
column (`question`, `query`, or `prompt`) and an answer column (`answer`,
`answers`, `gold`, or `ground_truth`). Parquet IDs use `id`, `case_id`,
`data_id`, or `index`; CSV accepts these fields and prefers `index`.
Only text columns are loaded from parquet.

Evaluation reads `*_trajectory.json` directly under `--traj_dir`. It matches
`case_id` against the gold ID first, then falls back to a nonnegative
`case_idx` (or the numeric suffix of `case_N`). CSV numeric indices retain
their explicit values; parquet indices refer to row order. Supply the same
ordered benchmark used for inference when an index fallback is needed.
Duplicate gold IDs and duplicate trajectories for one gold row are rejected.

The default outputs are `eval_report.json` and `eval_report.details.jsonl`.
The report records `modality` (`image` includes multi-image) and `protocol`.
`--output`, `--max_workers`, and `--limit` control output placement, parallelism,
and subset evaluation. Details record the matching method, submitted answer,
raw judge response, binary score, and status for every selected trajectory.

- A valid negative verdict and an empty model answer score zero.
- Missing gold, invalid trajectory JSON, failed judge requests, and unparseable
  verdicts retain `acc: null` and a failure status. They remain in `total`.
- If any selected trajectory fails evaluation, the report is `incomplete`,
  `accuracy` is null, and the command exits with status 1. Fix the failures
  before reporting accuracy.
- A complete report computes accuracy over all selected trajectories. It also
  records `gold_rows`, `matched_gold_rows`, `gold_rows_without_trajectory`,
  `trajectory_non_ok`, and `limit`. A complete subset evaluation is not full
  benchmark coverage; inspect these counts before reporting a full score.

## Runtime Dependencies

Every retained runtime module supports an inference entry point, a shared RL
import, or video preparation:

| Files | Used by |
|---|---|
| `run_infer.py`, `run_infer_single.py`, `run_infer_mi.py`, `run_infer_video.py` | Modality routing and dataset loading |
| `onesearch_infer/pipeline*.py`, `prompts/`, `sft_chat_format.py` | Agent loop, shared system prompts, and tool observations |
| `runners.py`, `config.py`, `messages.py` | Model backends and multimodal message conversion |
| `tools.py`, `search.py`, `image_engines.py` | Tool dispatch, retrieval, and image transformations |
| `image_io.py`, `video_io.py`, `cos_upload.py`, `upload.py` | Media loading, frame selection, and asset publication |
| `rollout_admission.py` | Shared tool-capacity check before starting inference |
| `prepare_video.py`, `preprocessing/preprocess_grids.py` | Silent sampled-frame grid preparation |
| `preprocessing/video_prep.py` | Local-video preparation for the Seed backend |
| `eval_image.py`, `eval_video.py` | Image and video evaluation entry points |
| `eval_common.py`, `eval_prompts.py` | Shared evaluation logic and fixed judge inputs |
| `load_env.sh` | Environment loading for both shell entry points |

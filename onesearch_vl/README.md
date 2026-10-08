# OneSearch-VL · Inference & Evaluation

[Project Overview](../README.md) · [SFT](../SFT/README.md) · [RL](../RL/README.md) · [Reference](REFERENCE.md)

Run **OneSearch-VL** on single images, multi-image collections, and videos,
then evaluate its saved trajectories. A shared tool environment connects visual
inspection and temporal localization with image and text retrieval. The runtime
uses the same actor prompts and retrieval implementation as RL.

All commands below run from the **repository root**.

## Repository Layout

```text
onesearch_vl/
├── run_infer.py          # Unified --modality / --model entry point
├── run_infer.sh          # Environment-driven inference wrapper
├── run_infer_single.py   # Single-image parquet input
├── run_infer_mi.py       # Multi-image CSV + manifest input
├── run_infer_video.py    # Video CSV + frame manifest input
├── prepare_video.py     # Prepare custom silent videos
├── run_eval.sh          # Image/video evaluation wrapper
├── eval_image.py        # Single-image and multi-image judge
├── eval_video.py        # Video judge
├── onesearch_infer/      # Prompts, model backends, tools, and agent loops
├── .env.example         # Model, retrieval, media, and judge settings
└── REFERENCE.md         # Input schemas and evaluation details
```

## 1. Install

Use a dedicated Python environment and install the inference dependencies:

```bash
python -m pip install -r onesearch_vl/requirements.txt
```

Install `ffmpeg` for video decoding and clip extraction. Local checkpoint
inference requires sufficient GPU memory; HTTP inference uses a separately
running vLLM, SGLang, or other compatible model server.

## 2. Configure Services

Copy the [environment template](.env.example) and fill in the services you use:

```bash
cp onesearch_vl/.env.example onesearch_vl/.env
# Edit model, retrieval, asset-publishing, and judge settings.
set -a
source onesearch_vl/.env
set +a
```

- **Retrieval:** `SERPER_API_KEY` enables text and image search; `JINA_API_KEY`
  configures the optional webpage reader. Set `QWEN_API_BASE`, `QWEN_API_KEY`,
  and `QWEN_MODEL_NAME` for evidence summarization.
- **Media publication:** set `ASSET_DIR` and `ASSET_BASE_URL` for an existing
  static server. Image search needs URLs reachable by the retrieval service;
  the bundled uploader copies media but does not start a server.
- **Evaluation:** set `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and `JUDGE_MODEL`.
  The default judge is `gpt-4o`.

The shell wrappers load `onesearch_vl/.env`, or the file selected by `ENV_FILE`.
Direct Python commands need the exported environment shown above. See the
[template](.env.example) for optional gateway, OCR, and provider settings, and
[service configuration](REFERENCE.md#service-configuration) for aliases and
custom uploaders.

## 3. Run Inference

The examples below use the published
[OneSearch-VL-8B](https://huggingface.co/OneSearch-VL/OneSearch-VL-8B) checkpoint.
Use `--checkpoint /models/your-checkpoint` to evaluate a local SFT or RL model.

### Single Image

Supply a parquet dataset with a question and one image per row. See the
[input reference](REFERENCE.md#single-image) for accepted image fields and IDs.

```bash
python onesearch_vl/run_infer.py --modality single_image \
  --model 8b --checkpoint OneSearch-VL/OneSearch-VL-8B --gpus 0 \
  --data-path /data/single_image.parquet --output-dir ./outputs/single_image
```

### Multiple Images

Download [OneSearch-MI-Bench](https://huggingface.co/datasets/OneSearch-VL/OneSearch-MI-Bench)
and follow its dataset card to run `prepare.py`. The generated CSV and manifest
preserve the image-index mapping used by the questions.

```bash
python onesearch_vl/run_infer.py --modality multi_image \
  --model 8b --checkpoint OneSearch-VL/OneSearch-VL-8B --gpus 0 \
  --csv /data/OneSearch-MI-Bench/runtime/mi.csv \
  --manifest /data/OneSearch-MI-Bench/runtime/mi_manifest.jsonl \
  --output-dir ./outputs/onesearch_mi
```

For custom inputs, use a QA CSV and a matching image manifest as described in
[Multiple images](REFERENCE.md#multiple-images). Image indices identify visual
anchors; they do not imply a video timeline.

### Video

Download [OneSearch-Video-Bench](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench),
extract both video and original-resolution frame archives, and run the dataset's
`prepare.py` before inference.

```bash
python onesearch_vl/run_infer.py --modality video \
  --model 8b --checkpoint OneSearch-VL/OneSearch-VL-8B --gpus 0 \
  --csv /data/OneSearch-Video-Bench/runtime/video.csv \
  --manifest /data/OneSearch-Video-Bench/runtime/video_manifest.jsonl \
  --output-dir ./outputs/onesearch_video
```

For custom videos, [prepare_video.py](prepare_video.py) builds a sampled frame-grid
representation. Its playback timeline differs from the original video; see
[Video preparation](REFERENCE.md#video) for the command and frame-mapping details.

### Model Backends

- **Local Transformers:** `--model 8b`, `32b`, or `30b-a3b`, with `--checkpoint`
  and `--gpus` selecting the model and devices.
- **HTTP server:** `--model vllm`, with `VLLM_API_BASE`, `VLLM_API_KEY`, and
  `VLLM_MODEL`; omit the local checkpoint argument.
- **Public Claude API:** `--model claude`, with `ANTHROPIC_BASE_URL`,
  `ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL`.

See [all backend options](REFERENCE.md#model-backends) for additional variants.
Add `--dry-run` to the unified entry point to inspect the selected command
without loading a model.

## 4. Tool Environment

The agent uses `crop` and optional visual enhancement/OCR tools to inspect
anchors, `image_search` and `text_search` to retrieve evidence, and
`select_timespan` / `select_frame` to navigate videos. Available tools follow
the [modality-specific prompts](onesearch_infer/prompts/), shared with RL.

See the [tool reference](REFERENCE.md#tool-environment) for individual tools,
provider behavior, and custom retrieval support.

## 5. Evaluate

Evaluate saved trajectories with the same benchmark inputs used for inference:

```bash
# Single image
bash onesearch_vl/run_eval.sh image \
  --traj_dir ./outputs/single_image --parquet /data/single_image.parquet

# OneSearch-MI-Bench
bash onesearch_vl/run_eval.sh image \
  --traj_dir ./outputs/onesearch_mi \
  --csv /data/OneSearch-MI-Bench/runtime/mi.csv

# OneSearch-Video-Bench
bash onesearch_vl/run_eval.sh video \
  --traj_dir ./outputs/onesearch_video \
  --csv /data/OneSearch-Video-Bench/runtime/video.csv
```

Image and multi-image evaluation use the `onesearch` protocol; video evaluation
uses `videodr` by default. Leave `EVAL_PROTOCOL` unset for these defaults, or
select a supported protocol with `--protocol`.

Results are written to `eval_report.json` and `eval_report.details.jsonl`.
Failed judge requests or unmatched inputs produce an incomplete report with
null accuracy. Inspect coverage counts before reporting a full benchmark score.
The [evaluation reference](REFERENCE.md#evaluation-protocols-and-reports) explains
answer extraction, gold matching, protocol overrides, and report fields.

## Acknowledgements and License

The runtime builds on [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL).
See the root [LICENSE](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md).

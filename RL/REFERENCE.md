# RL Data and Launcher Reference

For installation and an end-to-end training run, start with the
[RL guide](README.md). This page describes the prepared data contract and
launcher defaults needed when adapting the workflow.

## Dataset Contract

The published [RL dataset](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-RL-10K)
contains **10,059 unique training tasks**, one row per task: 3,691 single-image,
2,780 multi-image, and 3,588 video tasks. It has no test split. The registration
name and default adjacent directory are both `OneSearch-VL-RL-10K`.

The prepared bundle retains fields needed by training, reward construction,
and runtime alignment. Its task projection preserves questions, answers,
split membership, ordered rows, media, and reward targets. Annotation
provenance stays in the source bundle.

### Published Task Fields

| Modality | JSONL keys |
| --- | --- |
| Single image | `id`, `question`, `answer`, `modality`, `images` |
| Multi-image | `id`, `question`, `answer`, `modality`, `images`, `image_frame_indices` |
| Video | `id`, `question`, `answer`, `modality`, `video_grid` |

`modality` is `single_image`, `multi_image`, or `video`. Single-image rows have
one image; multi-image rows preserve the initial image order.

`video_grid` contains `video_url` and ordered `frame_urls`, both relative media
references. Root frame indices are zero-based positions, frame count is the
list length, and playback is fixed at 1 fps. Runtime duration and subclip
offsets are derived by the loader. They are not repeated in published JSONL.

`image_frame_indices[i]` maps multi-image `images[i]` to its source frame
number. A source frame number is not necessarily the image-list position.
Registration creates `evgr_media_identity` before resolving local paths;
this runtime field is not part of the published task schema.

### Judge-Side Rubrics

`data/<modality>.jsonl` and `evgr_gt/<modality>.jsonl` pair by task ID. The
runtime index additionally checks the question, answer, and media identity.
The same task schema is used by `train.jsonl` and `data/`; no `tasks.jsonl`
copy is required. Held-out rows are excluded from both paired directories.

GT common fields are `id` and `claim_facts`. Modality follows from the paired
task and file; the reference answer stays in the task.

- Single-image GT adds `gt_bboxes`, or the explicit
  `grounding_contract="source_only"` fallback.
- Multi-image/video GT adds `grounding_annotations`: ordered entries with a
  required `frame_index` and optional `object_description`. Missing appearance
  text is omitted. Target frames are derived from these entries; GT boxes,
  duplicate `gt_frames`, and annotation metadata are absent.

Claims and target annotations remain judge-side supervision. They are not
added to the actor prompt and cannot substitute for actual execution evidence.

### Preparation and Registration

To reproduce the projection from an annotation bundle, run from the
repository root:

```bash
python RL/prepare_training_data.py \
  --source /data/annotation-bundle \
  --output /data/OneSearch-VL-RL-10K
```

Preparation checks task identity and normalized evidence equality for every
task. It removes identical repeated rows in first-occurrence order and rejects
conflicting rows with the same ID. Its local `media/` link reuses the original
media tree; dereference or copy the media when distributing the bundle
independently.

The underlying registration command accepts prepared JSONL or parquet:

```bash
python RL/register_dataset.py \
  --name OneSearch-VL-RL-10K \
  --train /data/OneSearch-VL-RL-10K/train.jsonl \
  --media-root /data/OneSearch-VL-RL-10K \
  --evgr-root /data/OneSearch-VL-RL-10K
```

`--evgr-root` defaults to the train-file directory. `--source-map` accepts a
JSONL mapping with `path` and `source` fields instead of `--media-root`.
Registration rejects duplicate task IDs, validates task/rubric alignment, and
writes local registry and Verl files without expanding rows. Re-register
after changing tasks, rubrics, media, or frame mappings.

An optional `--test` input must contain held-out rows with globally distinct
IDs and matching rubrics. The standard training entry deliberately loads only
the train split and clears validation inputs. Supplying a registered test
split alone does not enable validation; configure explicit validation data
for a validation invocation.

## Launcher Defaults

Both provided 8B launchers use the same training recipe except for Megatron
parallelism and cluster size:

| Setting | Default |
| --- | --- |
| Advantage estimator | RLOO |
| Rollout backend / mode | SGLang / asynchronous |
| Prompt batch / responses per prompt / minibatch | 256 / 8 / 64 |
| Maximum prompt / response tokens | 40,960 / 16,384 |
| Learning rate | 1e-6 |
| KL coefficient / upper clip ratio | 0.001 / 0.28 |
| Single-node Megatron TP / PP / CP | 2 / 1 / 1 |
| Multi-node Megatron TP / PP / CP | 4 / 2 / 4 |
| Rollout tensor parallelism | 4 |
| Concurrent rollout tasks / tools | 512 / 1,024 |
| Training steps / checkpoint interval | 201 / 10 |

The 1 × 8 GPU and 4 × 8 GPU launchers are separate presets. Changing `NNODES`
alone does not adjust parallelism, batch sizes, or memory limits. Start the
multi-node Ray cluster externally and invoke the driver once on its head.

With batch size 256 and `drop_last=True`, the 10,059-row dataset has 39 full
batches per epoch. `data.shuffle=True` reshuffles each epoch, avoiding a fixed
excluded tail.

Environment variables such as `TOTAL_TRAINING_STEPS`, `SAVE_FREQ`,
`TRAIN_PROMPT_BSZ`, `N_RESP_PER_PROMPT`, `TRAIN_PROMPT_MINI_BSZ`,
`N_PARALLEL_TASKS`, and `N_PARALLEL_TOOLS` override their launcher defaults.
Set them in the selected environment file, whose assignments take precedence
over inherited variables. Trailing Hydra arguments override settings on the
final training command. Use `--dry-run` to inspect the effective command.

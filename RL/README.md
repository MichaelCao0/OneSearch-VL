# OneSearch-VL · RL Training

Joint agentic reinforcement learning trains the SFT checkpoint across
**single-image, multi-image, and video** tasks. The
**Evidence-aware Visual-Grounded Rubric reward (EVGR)** supervises evidence
traceability and visual grounding alongside answer accuracy and query utility.

Training uses [rLLM](https://github.com/rllm-org/rllm),
[verl](https://github.com/volcengine/verl), and SGLang for asynchronous rollouts,
with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and
[mbridge](https://github.com/ISEEKYAN/mbridge) for model-parallel updates.
The included launchers train the **8B model** with RLOO.

**Run all commands below from the repository root.** For the project overview,
see the [main README](../README.md).

## Repository Layout

```text
RL/
├── rllm/
│   ├── rllm/                         # Workflow engine and trainer
│   ├── verl/                         # Training and rollout backend
│   └── vision_deepresearch_async_workflow/
│       ├── deepresearch_agent.py     # Agent loop and tool execution
│       ├── deepresearch_workflow.py  # Rollout and reward orchestration
│       ├── train_deepresearch_workflow_megatron.py
│       ├── tools/                    # Visual and web tools
│       ├── data_prepare/             # Dataset registration entry points
│       └── run/                      # Single- and multi-node launchers
├── Megatron-LM/
├── mbridge/
├── register_dataset.py
└── .env.example
```

## 1. Install

Use a dedicated Python/CUDA environment and install the bundled framework
copies, which contain the changes required by this workflow:

```bash
python -m pip install pybind11
python -m pip install -e ./RL/Megatron-LM --no-build-isolation
python -m pip install -e ./RL/mbridge
python -m pip install -e './RL/rllm/verl[sglang]'
python -m pip install -e ./RL/rllm
python -m pip install qwen-vl-utils opencv-python-headless tokenizers
```

The SGLang extra pins **SGLang 0.5.5** and **PyTorch 2.8.0**. Install compatible
NVIDIA Transformer Engine and attention kernels for your CUDA environment,
then check the runtime:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformer_engine.pytorch; print('Transformer Engine: OK')"
```

## 2. Configure the Environment

Create the component environment file and fill in your local paths and service
credentials:

```bash
cp RL/.env.example RL/.env
```

| Setting | Purpose |
| --- | --- |
| `MODEL_PATH`, `OUTPUT_DIR` | SFT checkpoint and training output directory; absolute paths are recommended |
| `SERPER_API_KEY`, `JINA_API_KEY` | Search and page-reading services; the Jina key is optional |
| `QWEN_API_BASE`, `QWEN_API_KEY`, `QWEN_MODEL_NAME` | Evidence summarizer |
| `ASSET_DIR`, `ASSET_BASE_URL` | Publish frames and crops for image search |
| `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` | Accuracy and query-utility judges |
| `SEED_API_BASE`, `SEED_API_KEY`, `EVGR_JUDGE_MODEL` | Multimodal EVGR judges; the default model is `Doubao-Seed-2.0-pro` |
| `FORMAL_EVGR_DATA_ROOT` | RL dataset directory containing paired `data/` and `evgr_gt/` files |
| `WANDB_API_KEY` | Authentication when using Weights & Biases logging |

Launchers load `RL/.env` automatically, falling back to `RL/rllm/.env` and then
the root `.env` if the component file is absent. Use
`ENV_FILE=/absolute/path/to/.env` to select another file. Values in that file
replace inherited environment variables with the same names.

Set `FORMAL_EVGR_DATA_ROOT` to the **same dataset root used for registration**.
Its default is `../OneSearch-VL-RL-10K/` next to this repository; relative
values resolve against the repository root. See the
[tool-service guide](../onesearch_vl/README.md#2-configure-services) for optional
gateway, layout-parsing, and uploader settings.

## 3. Register RL-10k

Download and extract the
[RL dataset](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-RL-10K)
to a local directory. The prepared bundle contains `train.jsonl`, media,
per-modality task files in `data/`, and judge-only rubrics in `evgr_gt/`.
Register it with the provided helper:

```bash
DATA_ROOT=/data/OneSearch-VL-RL-10K \
  bash RL/rllm/vision_deepresearch_async_workflow/data_prepare/register_rl_dataset.sh \
  --media-root /data/OneSearch-VL-RL-10K
```

For this example, set `FORMAL_EVGR_DATA_ROOT=/data/OneSearch-VL-RL-10K` in
`RL/.env`. The registration name defaults to `OneSearch-VL-RL-10K`; if you
change `RL_DATASET_NAME`, use the same name for registration and training.
The registration helper reads exported variables directly; it does not load
`RL/.env` itself.

Registration preserves task order, rejects duplicate IDs, checks each task
against its rubric, and creates the local rLLM registry and Verl inputs. It
does not split, filter, or resample the data. The release is **train-only**:
the training entry clears validation inputs and does not reuse train rows for
validation. A validation run requires an explicit held-out dataset.

If your media lives elsewhere, use `--source-map` instead of `--media-root`.
For custom data, parquet inputs, and the published task schema, see the
[data reference](REFERENCE.md#dataset-contract).

An offline preflight checks dataset alignment and judge configuration without
sending judge requests:

```bash
PYTHONPATH=RL/rllm python -m vision_deepresearch_async_workflow.reward_preflight \
  --data-root /data/OneSearch-VL-RL-10K --check-env
```

This standalone check reads the current environment; export your judge
settings first. Training launchers run preflight automatically after loading
the environment file and also check the registered rows.

## 4. Launch Training

Two launchers are provided for the 8B model:

| Launcher | Default allocation |
| --- | --- |
| [Single node](rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-single-node.sh) | 1 node × 8 GPUs |
| [Multi-node](rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-multi-node.sh) | 4 nodes × 8 GPUs |

```bash
# Single node.
bash RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-single-node.sh

# Multiple nodes: run the driver once on the Ray head.
bash RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-multi-node.sh
```

For multi-node training, start Ray through your scheduler, set `RAY_ADDRESS`,
and match `NNODES` and `GPUS_PER_NODE` to the allocation. Every node needs the
same code, Python environment, model, media, dataset registry, and shared
output mounts. The multi-node parallelism preset requires at least 32 GPUs;
use the single-node preset for an 8-GPU machine.

The released scripts default to **201 training steps**, with shuffled training
data and checkpoints every 10 steps.

Set `TOTAL_TRAINING_STEPS`, `TRAIN_PROMPT_BSZ`, `N_RESP_PER_PROMPT`, and
`SAVE_FREQ` in `RL/.env` to adjust the run. Trailing Hydra overrides can change
training settings directly; for example, inspect a one-update command:

```bash
bash RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-single-node.sh \
  --dry-run trainer.total_training_steps=1
```

`--dry-run` validates reward weights and prints the command without loading
GPU libraries, requiring dataset/judge credentials, or creating output
directories. Remove `--dry-run` to execute the run. Parallelism, sequence
lengths, and memory/concurrency settings are grouped in each launcher; see the
[launcher reference](REFERENCE.md#launcher-defaults) before changing topology.

## 5. Rewards and Outputs

EVGR is enabled by default with `FORMAL_EVGR_ENABLED=1` and
`FORMAL_EVGR_MODE=both`. Its Trace judge checks retrieved evidence against
required claims, and its Ground judge checks visual localization and actual
retrieval use. Accuracy and query utility use the separate `JUDGE_*` service.

The configured `OUTPUT_DIR` contains:

```text
OUTPUT_DIR/
├── checkpoints/
├── episodes/
├── trajectory_dumps/
├── wandb/
└── tensorboard/
```

Without `OUTPUT_DIR`, the output base is
`RL/rllm/checkpoints/onesearch-vl/<experiment-name>/`. Relative output paths
resolve from `RL/rllm/`. Use episode logs and trajectory dumps to inspect the
agent's calls, retrieved evidence, and reward diagnostics.

## Acknowledgements and License

Built on rLLM, verl, Megatron-LM, and mbridge. Bundled frameworks retain their
upstream licenses; project changes use the [root license](../LICENSE).
See the [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution.

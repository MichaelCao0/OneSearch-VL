# OneSearch-VL · Agentic SFT

[Project Overview](../README.md) · [Agentic RL](../RL/README.md) ·
[Inference & Evaluation](../onesearch_vl/README.md)

This directory contains the supervised fine-tuning (SFT) pipeline for
**OneSearch-VL**. Built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
it trains Qwen3-VL-8B on joint single-image, multi-image, and video trajectories
to learn a unified visual research policy before RL.

All commands below run from the **repository root**. Replace the example
`/data`, `/models`, and `/checkpoints` paths with your local directories.

## Repository Layout

```text
SFT/
├── src/llamafactory/        # Trainer, multimodal data loader, and CLI
├── data/dataset_info.json   # Single-image, multi-image, and video registrations
├── examples/
│   ├── onesearch/qwen3_vl_8b_joint_110k.yaml
│   └── deepspeed/ds_z2_config.json
├── scripts/relocate_media.py
├── pyproject.toml
├── run_train.sh
└── LICENSE
```

## Supported Training Recipe

The released [Qwen3-VL-8B recipe](examples/onesearch/qwen3_vl_8b_joint_110k.yaml)
uses the `qwen3_vl` template and DeepSpeed ZeRO-2. It fine-tunes the language
model while keeping the vision tower and multimodal projector frozen.
The YAML contains the learning rate, sequence length, media limits, and batch
settings; use it as the reference when adapting the run to your hardware.

## 1. Install

Use Python 3.11+ in a dedicated CUDA environment:

```bash
python -m pip install -e ./SFT
python -m pip install deepspeed qwen-vl-utils huggingface_hub
llamafactory-cli version
```

The dependency bounds are recorded in [pyproject.toml](pyproject.toml).
Choose PyTorch, CUDA, and DeepSpeed builds that work together on your hardware.

## 2. Prepare Data

Download
[OneSearch-VL-SFT-110K](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-SFT-110K)
and extract every media ZIP shard into its corresponding modality directory,
following the dataset card:

```bash
hf download OneSearch-VL/OneSearch-VL-SFT-110K \
  --repo-type dataset \
  --local-dir /data/OneSearch-VL-SFT-110K
```

Then resolve the downloaded release's relative image/video paths and write
training JSON files into this checkout:

```bash
python SFT/scripts/relocate_media.py \
  --base-dir /data/OneSearch-VL-SFT-110K \
  --output-dir SFT/data
```

The prepared layout matches [data/dataset_info.json](data/dataset_info.json):

```text
SFT/data/
├── dataset_info.json
├── image/image_llama_factory.json
├── multi_image/multi_image_llama_factory.json
└── video/video_llama_factory.json
```

The recipe combines `image_agent_sft`, `multi_image_agent_sft`, and
`video_agent_sft`. Each JSON uses ShareGPT conversations, media references,
a system prompt, and tool definitions.

The relocation script checks that media exists and refuses to overwrite
existing training JSON files. It leaves the media in the download directory
and writes absolute paths, so that directory must remain available on every
training worker. For a different training-data directory, copy the registry
there and pass `dataset_dir=/absolute/path/to/data` at launch.

## 3. Launch Training

```bash
bash SFT/run_train.sh \
  model_name_or_path=/models/Qwen3-VL-8B-Instruct \
  output_dir=/checkpoints/onesearch-sft
```

The launcher changes into `SFT/` and runs the joint recipe with torchrun.
Additional `key=value` arguments override the YAML; use absolute paths for
local models, data, and outputs. Preview the command without starting training:

```bash
bash SFT/run_train.sh --dry-run
```

Checkpoints default to `SFT/saves/onesearch-sft-110k`. To restrict a single-node
run, set `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` to the GPUs you allocated.

## 4. Cluster Notes

For multi-node training, run the following on each allocated node, using a
distinct `NODE_RANK` and the same reachable master address and port. This
example shows rank 0 of an 8-node cluster:

```bash
NNODES=8 NODE_RANK=0 MASTER_ADDR=master.example.org MASTER_PORT=29500 \
  NPROC_PER_NODE=8 bash SFT/run_train.sh \
  model_name_or_path=/models/Qwen3-VL-8B-Instruct \
  output_dir=/checkpoints/onesearch-sft
```

- Keep the checkout, model, media, and output paths consistent across workers.
- The reference 8-node × 8-GPU run uses global batch size 128. Adjust
  `gradient_accumulation_steps` when changing the number of GPUs.
- Allocate machines through your cluster scheduler before launching; configure
  NCCL networking for your own cluster.

### Tokenized Cache

The tokenized cache is `SFT/cache/onesearch-sft-110k`. Use a fresh `tokenized_path`
when changing datasets, templates, system prompts, media paths, or preprocessing
settings so training does not reuse an earlier tokenized dataset.

The resulting checkpoint can initialize [agentic RL](../RL/README.md) or be
used with the [inference pipeline](../onesearch_vl/README.md).

## Acknowledgements

This subtree derives from LLaMA-Factory (Apache-2.0). See [LICENSE](LICENSE)
and the root [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution.

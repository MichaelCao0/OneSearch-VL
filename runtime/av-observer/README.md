# Native AV observer runtime

This Linux CUDA environment is isolated from the repository's default CPU
Pixi manifest and lock. All dependencies are resolved in the adjacent
`pixi.lock`; do not install them into the default environment.

The pinned observer is `Qwen/Qwen2.5-Omni-7B` at immutable revision
`ae9e1690543ffd5c0221dc27f79834d0294cba00`. The 20-file full snapshot is
22,379,297,323 bytes, below the 30,000,000,000-byte acquisition cap. Weight
shards, `spk_dict.pt`, and `tokenizer.json` must pass the pinned HF LFS
SHA-256 checks before model loading. Non-LFS files also pass their pinned Git
blob SHA-1 checks and receive local SHA-256 entries in the manifest.

## Installation and preparation

Run from the repository root on the user-selected workspace:

```bash
pixi install --locked --manifest-path runtime/av-observer/pixi.toml
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_omni_setup.py preflight
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_omni_setup.py acquire
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_omni_setup.py verify
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_omni_setup.py probe
```

Acquisition can also run using the already installed, locked CPU runtime
while the isolated CUDA environment installs. This reads its existing Hub
package and does not install or modify CPU dependencies; the manifest records
the acquisition Python and Hub version.

Acquisition uses the standard public Hugging Face cache under
`/home/jovyan/.cache/huggingface/hub`, HTTP downloads with at most two
workers, no remote model code, and no credentials. The acquisition helper
aborts on an unknown file size, oversized snapshot, missing file, size
mismatch, or digest mismatch. It does not load weights.

Evidence and the verified `model-manifest.json` are written under
`research/runs/av-native-smoke-20261010-v1/runtime-setup/` (ignored). The
manifest carries `repository_id`, `revision`, `snapshot_path`,
`all_lfs_verified`, and per-file expected/actual SHA-256 and verified
status. A successful acquisition has `status: "verified"`.
Keep run-specific output paths distinct for future runs.

## Genuine joint audio-video API

Use the official `Qwen2_5OmniForConditionalGeneration`,
`Qwen2_5OmniProcessor`, and `qwen_omni_utils.process_mm_info`.
A content item is `{"type": "video", "video": "/absolute/path/clip.mp4"}`;
the MP4 must contain its audio track. Set `use_audio_in_video=True`
consistently in `process_mm_info`, the processor call and `generate`.
The utility decodes the video audio to a 16 kHz waveform as well as video
frames; both enter the official processor and model.

Load the verified local snapshot with `local_files_only=True`,
`enable_audio_output=False`, CUDA placement and BF16 on a compatible GPU.
Use `return_audio=False` for text output. In Transformers 4.57.6 the
wrapper's `generate` explicitly accepts `thinker_max_new_tokens`; use
that keyword to control the text budget. Slice input prompt tokens before
decoding completion text.

`enable_audio_output=False` skips constructing talker/token2wav modules.
The official wrapper still reads `spk_dict.pt` during `from_pretrained`,
so acquisition and verification include this asset even for text-only
output. Torch 2.8.0 satisfies the official safety gate for loading that
speaker dictionary.

## Optional CPU input-path probe

```bash
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_omni_setup_processor_probe.py
```

This verifies the pinned processor/tokenizer assets, creates a two-second
synthetic MP4 containing a blue scene and sine-wave audio, and runs the
official joint AV utility and processor. It records nonzero decoded audio,
video shapes and the actual modality tensors. It does not load model
weights or generate a response; the probe cannot establish semantic AV
understanding.

## Scope of evidence

The workspace3 setup stage is CPU-only installation, imports, acquisition
and integrity verification. A CUDA-linked Torch wheel and class imports do
not prove access to a GPU or successful native AV inference. Actual GPU
load, processor tensors, generation, semantic behavior and memory use must
be recorded by the separate GPU smoke worker. The runtime uses SDPA by
default; FlashAttention is not installed or claimed.

The environment pins Torch 2.8.0+cu126 / torchvision 0.23.0+cu126,
Transformers 4.57.6, qwen-omni-utils 0.0.9 and Python 3.11. A CUDA 12.6
compatible NVIDIA driver is required by the GPU worker. The pinned official
model card and Transformers/utility source inspection are preserved in the
ignored setup evidence directory.

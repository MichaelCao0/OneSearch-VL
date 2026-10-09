# OneSearch HF assets audit

Observed UTC: 2026-10-09T08:16:07.163415+00:00

Scope: public unauthenticated Hugging Face API, pinned metadata, and small README/config/prepare/manifest files only. No model weights, ZIP media, training JSON/JSONL, model inference, extraction, or training downloaded/executed. Only current work/ was written.

## Access, revisions and compressed/download size

All five pinned API reads returned the requested SHA. All report private=false and gated=false. None has cardData.license or a LICENSE/COPYING file in the pinned file tree; license is therefore undeclared, not inferred from the base model or linked GitHub repository.

| Asset | Pinned HF revision | Files | All listed download bytes | Decimal GB | License / access |
|---|---|---:|---:|---:|---|
| OneSearch-VL/OneSearch-VL-8B | `2623603767b9fef7bab05ad0c3cd4d464301e8df` | 14 | 17,550,226,046 | 17.550226 | undeclared / public, ungated |
| OneSearch-VL/OneSearch-VL-SFT-110K | `64c9da7f2af3362f26f497ad28f13ca834797b37` | 42 | 139,285,156,960 | 139.285157 | undeclared / public, ungated |
| OneSearch-VL/OneSearch-VL-RL-10K | `1e0c8334a1604d8cc8e0edabf34c15c94d2899e2` | 32 | 214,472,195,071 | 214.472195 | undeclared / public, ungated |
| OneSearch-VL/OneSearch-MI-Bench | `b466c14271948c987568a79700d9077b3a165a71` | 7 | 432,779,147 | 0.432779 | undeclared / public, ungated |
| OneSearch-VL/OneSearch-Video-Bench | `8694a570d777066e053fc258623a429cc13a2294` | 12 | 18,514,944,798 | 18.514945 | undeclared / public, ungated |

Byte totals sum sibling.size from the pinned HF API and include all listed files. They are download sizes, not extracted disk requirements. All-five total: 390,255,302,022 bytes (390.255 GB). Full LFS hashes/file sizes remain in *.pinned.metadata.json.

## Per-asset findings

### Model: OneSearch-VL-8B

- Single weight file model.safetensors: 17,534,340,584 bytes (17.534 GB / 16.330 GiB). LFS SHA-256: f73186b544f7ff267d1150cc5ef3340edbf196d21b46794e4a623734304c8c5b.
- config.json lines 2–10, 54: Qwen3VLForConditionalGeneration, model_type=qwen3_vl, dtype=bfloat16, transformers_version=5.2.0. The saved library version does not establish a minimum supported version; 4.57.x compatibility and CPU behavior require actual probes.
- No README.md, model card, license, prepare/inference entry, optimizer/scheduler/RNG state in the pinned file tree. This is a released inference weight asset, not evidenced as a resumable training checkpoint.

### SFT: OneSearch-VL-SFT-110K

- README.md is 245 bytes, only YAML configs image/multi_image/video; no prose, sample count, provenance, license, or prepare command.
- Three JSON arrays total 3,354,736,959 bytes: image/image_llama_factory.json=1,484,415,754; multi_image/multi_image_llama_factory.json=837,682,274; video/video_llama_factory.json=1,032,638,931.
- ZIPs: image=3 files / 9,879,679,389 bytes; multi_image=8 / 28,310,124,901 bytes; video=3 image ZIPs +22 video ZIPs / 97,740,608,487 bytes. All-media ZIP total=135,930,412,777 bytes.
- relocate_media.py lines 44–51 and 54–96 resolves extracted relative images/videos paths under base_dir/modality/, streams JSON arrays, supports --dry-run and --modalities, writes new files via --output-dir and refuses overwrite. All relevant ZIP shards must be extracted before path verification.
- "110K" is repository naming; exact row count was not verified because large JSON arrays were not fetched. No native prepare.py; relocation is the supplied preparation entry.

### RL: OneSearch-VL-RL-10K

- README lines 20–44 declares 10,059 unique train-only tasks: 3,691 single-image / 2,780 multi-image / 3,588 video. Media: 3,691 original images; 8,336 multi-image PNGs; 3,337 prepared MP4s +426,040 video PNGs. Counts are card declarations, not extracted-media verification.
- Media ZIPs total 214,423,591,229 bytes: single-image 1,103,968,384; multi-image 3,852,512,437; video 21 ZIPs totaling 209,467,110,408. Extract independently at dataset root to preserve media/ paths.
- train.jsonl=22,856,892 bytes; grouped data/*.jsonl also total 22,856,892; evgr_gt/*.jsonl=2,878,934 bytes. Only train released; no test or validation.
- README lines 126–142 points to linked GitHub repository appletea233/onesearch-vl-submit: RL/rllm/vision_deepresearch_async_workflow/data_prepare/register_rl_dataset.sh and python -m vision_deepresearch_async_workflow.reward_preflight with media/data roots. HF has no prepare.py or training code; linked code revision was not audited.
- README lines 120–124 specifies reward boundaries and Format * (0.6 * Accuracy + 0.2 * Query + 0.2 * (Trace + Ground) / 2). Metadata/scripts existence does not verify reward execution or semantic annotation quality.

### MI-Bench

- Card: 301 questions, 910 unique original PNG images. media.jsonl independently parsed: exactly 910 distinct paths, total declared image bytes=432,462,201. The count is verified at manifest level, not against actual image files.
- mi-00001.zip=432,556,863 bytes (0.432557 GB); whole repository=432,779,147 bytes. All images included; Video-Bench not needed.
- prepare.py --verify-only checks ID/reference consistency, file size and PNG dimensions. --media-base-url HTTP(S) creates runtime/mi.csv and ordered runtime/mi_manifest.jsonl; Python>=3.9 and standard library. Must expose dataset directory through media server reachable by inference backend.
- README lines 91–107 points to onesearch_vl/run_infer_mi.py and run_eval.sh image; requires inference/search/evaluation credentials in code repo. No released evaluation result. Reference answers occur in evaluator CSV, not actor manifest.

### Video-Bench

- Card: 307 QA pairs over 293 silent videos. media.jsonl independently parsed: 293 distinct MP4 paths, declared MP4 bytes=841,919,035 and total frame slots=37,354. 1 FPS is the playback rate of prepared ordered frame sequences, not source-time sampling.
- video-00001.zip=841,949,529 bytes. Official original-resolution frame archives: video-frames-00001.zip=9,972,747,515 and video-frames-00002.zip=7,697,145,322; frame archives total=17,669,892,837 bytes. Whole repo=18,514,944,798 bytes.
- Official preparation needs both the MP4 archive and both original PNG archives extracted preserving media/ and runtime/frames/. prepare.py --verify-only verifies MP4 size/stream/fps/frame count plus official PNG file order/bytes/dimensions. It imports extract_frames.py; needs Python>=3.9, ffprobe, and README lists ffmpeg too.
- prepare.py --media-base-url HTTP(S) writes runtime/video.csv plus video_manifest.jsonl with video/frame URLs. Loader does not generate frames and errors on missing/mismatched original PNGs. extract_frames.py decodes diagnostic MP4 frames to runtime/decoded_frames/, explicitly refuses replacing official runtime/frames/; decoded frames can be lower resolution.
- alignment.json is publisher-supplied evidence: 253/293 MP4s allegedly reproduced byte for byte; remaining 40 allegedly checked framewise; 565 annotation-target references /544 unique target frames. Scope explicitly excludes QA semantic correctness and model/judge calls. This audit read the artifact, did not reproduce checks.
- README lines 156–166 expressly excludes audio, source timestamps, source URLs and source metadata as actor evidence. This benchmark cannot by itself validate audio-video cross-source event/entity binding.
- Small official input subset: HF tree contains ZIPs, no individual MP4/PNG resolve files and prepare.py has no --limit/--ids. Smallest documented video download is the 841,949,529-byte MP4 ZIP, from which one can select an MP4; the official full-tool PNG subset still requires access to relevant frame ZIP(s). No published remote ZIP-range loader is evidenced. A custom filtered adapter would be a separate subset smoke, not a successful full official prepare run.
- The first manifest MP4 is media/0000.mp4, declared 1,904,512 bytes, 128 frame slots, 512x288. It can be a pixel-input smoke after obtaining the ZIP; exact QA ID mapping was not read because test.jsonl was not downloaded.

## Missing/untested boundaries

- No license declaration for any of the five pinned assets. Public accessibility does not resolve permission/provenance terms.
- Model runtime/minimum transformers version, CPU BF16 generation, search APIs, judge APIs, training registration and media integrity are untested.
- SFT exact row counts and provenance are undocumented in its tiny card; RL/bench count and integrity claims remain publisher claims except independently parsed manifests.
- No audio asset or original source timestamps/URLs evidenced in Video-Bench; no independent QA semantic evaluation or released benchmark inference scores in package.
- Extracted disk requirements cannot be inferred exactly from archive sizes; retain archive plus extraction headroom. MI/Video media manifests provide original bytes only for their directly listed MP4/PNG inventory; original Video PNG byte total was not fetched because frames.jsonl exceeds this small-file cap.

## Pinned references

- OneSearch-VL/OneSearch-VL-8B [tree](https://huggingface.co/OneSearch-VL/OneSearch-VL-8B/tree/2623603767b9fef7bab05ad0c3cd4d464301e8df); [pinned public API](https://huggingface.co/api/models/OneSearch-VL/OneSearch-VL-8B/revision/2623603767b9fef7bab05ad0c3cd4d464301e8df?blobs=true).
  - [config.json](https://huggingface.co/OneSearch-VL/OneSearch-VL-8B/resolve/2623603767b9fef7bab05ad0c3cd4d464301e8df/config.json), 1,855 bytes, SHA-256 `f425bb696d8ac7919642f9479e03cea995452fd284cc538582bed768693550c4`.
- OneSearch-VL/OneSearch-VL-SFT-110K [tree](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-SFT-110K/tree/64c9da7f2af3362f26f497ad28f13ca834797b37); [pinned public API](https://huggingface.co/api/datasets/OneSearch-VL/OneSearch-VL-SFT-110K/revision/64c9da7f2af3362f26f497ad28f13ca834797b37?blobs=true).
  - [README.md](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-SFT-110K/resolve/64c9da7f2af3362f26f497ad28f13ca834797b37/README.md), 245 bytes, SHA-256 `06db7b594cd1022d17645603112c1140283062da4247aa7ab0c315e6875782d3`.
  - [relocate_media.py](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-SFT-110K/resolve/64c9da7f2af3362f26f497ad28f13ca834797b37/relocate_media.py), 4,262 bytes, SHA-256 `38f653a144fa977949aa0987238194b0a4414fbd58cd5fd146ab45e8b9b19569`.
- OneSearch-VL/OneSearch-VL-RL-10K [tree](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-RL-10K/tree/1e0c8334a1604d8cc8e0edabf34c15c94d2899e2); [pinned public API](https://huggingface.co/api/datasets/OneSearch-VL/OneSearch-VL-RL-10K/revision/1e0c8334a1604d8cc8e0edabf34c15c94d2899e2?blobs=true).
  - [README.md](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-RL-10K/resolve/1e0c8334a1604d8cc8e0edabf34c15c94d2899e2/README.md), 8,519 bytes, SHA-256 `d6f6f5dce426d5f586c98deacaa3bead432f972845d77a6c1557d361bba90bad`.
- OneSearch-VL/OneSearch-MI-Bench [tree](https://huggingface.co/datasets/OneSearch-VL/OneSearch-MI-Bench/tree/b466c14271948c987568a79700d9077b3a165a71); [pinned public API](https://huggingface.co/api/datasets/OneSearch-VL/OneSearch-MI-Bench/revision/b466c14271948c987568a79700d9077b3a165a71?blobs=true).
  - [README.md](https://huggingface.co/datasets/OneSearch-VL/OneSearch-MI-Bench/resolve/b466c14271948c987568a79700d9077b3a165a71/README.md), 3,314 bytes, SHA-256 `65f41c503138d341756eb63fa3efc5220755fe39edc4fe3753969c56090dbebd`.
  - [prepare.py](https://huggingface.co/datasets/OneSearch-VL/OneSearch-MI-Bench/resolve/b466c14271948c987568a79700d9077b3a165a71/prepare.py), 5,044 bytes, SHA-256 `d2f21d978460cd4441c72c4463fdf1eb194536fecf810bd6c52643bac8742c83`.
  - [media.jsonl](https://huggingface.co/datasets/OneSearch-VL/OneSearch-MI-Bench/resolve/b466c14271948c987568a79700d9077b3a165a71/media.jsonl), 66,399 bytes, SHA-256 `ee17c3c8badfc4e0793cc5c98bdd3e3080b21b3a020094a38f8d48c65e7c66bd`.
- OneSearch-VL/OneSearch-Video-Bench [tree](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/tree/8694a570d777066e053fc258623a429cc13a2294); [pinned public API](https://huggingface.co/api/datasets/OneSearch-VL/OneSearch-Video-Bench/revision/8694a570d777066e053fc258623a429cc13a2294?blobs=true).
  - [README.md](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/8694a570d777066e053fc258623a429cc13a2294/README.md), 6,240 bytes, SHA-256 `2756241eddfa097b3b6878f5f0a65493dc3c1085740ab96e7848da4a45262cb9`.
  - [prepare.py](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/8694a570d777066e053fc258623a429cc13a2294/prepare.py), 5,100 bytes, SHA-256 `af8cbba033fbbab0154c381a4664d503dcfb40fd11c6b2d8a0fbe286fe1913f1`.
  - [extract_frames.py](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/8694a570d777066e053fc258623a429cc13a2294/extract_frames.py), 7,480 bytes, SHA-256 `452a36b09e29a0dafdf9650efb668d7b6c25b866909f1594883654e737b80f7d`.
  - [alignment.json](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/8694a570d777066e053fc258623a429cc13a2294/alignment.json), 1,231 bytes, SHA-256 `4925972ec4e69743ec5b0237778ec7e19f1660b4130e57ba7d76ac8abaa7c3e1`.
  - [media.jsonl](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/8694a570d777066e053fc258623a429cc13a2294/media.jsonl), 29,573 bytes, SHA-256 `54bb7b84176ba4f9d34f0037259215359ac6b2e06eed2e5a6e6ea799be73e7cf`.

Local evidence: work/onesearch-hf-assets/{initial-index.json,pinned-api-checks.json,audit-computed.json,fetched-small-files.json,*.pinned.metadata.json} and per-asset small files. Downloaded small asset content: 139,262 bytes, excluding public API JSON.

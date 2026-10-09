# OneSearch-VL fork / code / asset audit

核对时间：2026-10-09。只读 GitHub/HF API、小型源码与 dataset cards；没有修改本地/远端 repository，没有下载权重/媒体/训练 JSON，没有执行推理或训练。HF 大小来自 pinned API 的 file metadata，而非实际下载。此报告不将源码可读或公开下载当作运行验证。

## 1. 固定源码与 fork 关系

- Fork：`MichaelCao0/OneSearch-VL`；parent：`appletea233/OneSearch-VL`。
- 两者当前 `main` 均为 **`0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd`**，commit message `Initial commit`，committer time `2026-10-08T13:24:23Z`。相同 commit 意味着本轮代码差异为零。
- GitHub API `size=8389`（GitHub 的 KB 量级字段，不等于安装后大小）；recursive tree `truncated=false`，1599 tree entries。源码包含 vendored LLaMA-Factory、rLLM/verl、Megatron-LM、mbridge；未见 `.gitmodules`、Pixi manifest/lock 或独立项目运行锁文件。
- API证据：[fork commit](https://api.github.com/repos/MichaelCao0/OneSearch-VL/commits/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd)、[official commit](https://api.github.com/repos/appletea233/OneSearch-VL/commits/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd)、[tree](https://api.github.com/repos/MichaelCao0/OneSearch-VL/git/trees/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd?recursive=1)。本地 API cache 为 `work/onesearch-{fork,official}-{repo,head}.json` 与 `work/onesearch-fork-tree.json`。

## 2. 真实入口与执行契约

所有下列链接固定到上述 SHA。

| 路径 | 实际作用与边界 |
|---|---|
| [onesearch_vl/run_infer.py](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/run_infer.py#L10) | argparse modality router；subprocess 运行 `run_infer_single.py` / `run_infer_mi.py` / `run_infer_video.py`。`--dry-run`只打印命令，不load模型 |
| [run_infer_video.py](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/run_infer_video.py#L43) | 有 `--model`、`--checkpoint`、`--gpus`、`--dtype`、CSV/manifest、`--indices`/`--limit`；默认dtype bf16、max_tokens32768。CPU smoke须显式缩小token/media |
| [runners.py](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/runners.py#L1057) | 原生 local Qwen3VL runner；无CUDA时`device_map='cpu'`，不必先写CPU adapter。L1177支持 `HF_ATTN_IMPLEMENTATION=eager/sdpa/flash_attention_2`；L1192真正from_pretrained；L1543按实际device搬输入并generate |
| [run_eval.sh](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/run_eval.sh#L9) / [eval_common.py](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/eval_common.py#L234) | shell loads env，然后eval_image/video；image默认onesearch、video默认videodr。judge必需API key；失败/unmatched使report incomplete及accuracy null |
| [SFT/run_train.sh](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/SFT/run_train.sh#L4) | `FORCE_TORCHRUN=1`，运行 `llamafactory-cli train examples/onesearch/qwen3_vl_8b_joint_110k.yaml`；`--dry-run`可打印 |
| [SFT joint YAML](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/SFT/examples/onesearch/qwen3_vl_8b_joint_110k.yaml#L3) | base `Qwen/Qwen3-VL-8B-Instruct`；freeze vision/projector、full LM、ZeRO2、LR1e-5、8epochs、cutoff32768、batch1/gradacc2/bf16。论文LR2e-5与此发布recipe不同 |
| [SFT/scripts/relocate_media.py](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/SFT/scripts/relocate_media.py) | 已解压媒体路径检查及另写absolute paths JSON，不是数据合成engine |
| [RL registration helper](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/data_prepare/register_rl_dataset.sh) / [preflight](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/reward_preflight.py) | 注册固定train rows及对应EVGR GT；offline judge/config alignment check不能证明在线judge正确 |
| [RL single node launcher](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-single-node.sh#L56) | **RLOO**，8responses、prompt256/minibatch64、201updates、genTP4/trainTP2、默认8GPUs；multi-node默认4×8。与论文GRPO不是字面相同训练配方 |
| [train_deepresearch_workflow_megatron.py](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/train_deepresearch_workflow_megatron.py#L31) | Hydra `agent_ppo_trainer_megatron` → DatasetRegistry.load(train) → AgentTrainer/DeepResearchWorkflow；显式val_files=[]、val_dataset=None，train-only，无发布held-outvalidation |

官方 eval 的固定judge输入是question、reference answer与抽取最终答案，**没有媒体与完整tool trace**：[REFERENCE.md Evaluation](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/REFERENCE.md#L141)。它给answer accuracy，不证明报告证据充分性或AV绑定正确。Judge默认gpt-4o；RL EVGR默认Doubao-Seed-2.0-pro，训练rubric不可进入actor input。

## 3. HF assets：固定revision、传输规模与准备

五个HF仓库均由API确认public、non-gated；没有model/dataset LICENSE文件或card license声明。每个repo总bytes含metadata/JSON；ZIP解压会另需空间。基模型`Qwen/Qwen3-VL-8B-Instruct`是SFT依赖，与发布的OneSearch权重不能混同。

| HF ID | 固定 revision | 所有文件bytes |
|---|---|---:|
| [OneSearch-VL/OneSearch-VL-8B](https://huggingface.co/OneSearch-VL/OneSearch-VL-8B/tree/2623603767b9fef7bab05ad0c3cd4d464301e8df) | `2623603767b9fef7bab05ad0c3cd4d464301e8df` | 17,550,226,046 |
| [OneSearch-VL/OneSearch-VL-SFT-110K](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-SFT-110K/tree/64c9da7f2af3362f26f497ad28f13ca834797b37) | `64c9da7f2af3362f26f497ad28f13ca834797b37` | 139,285,156,960 |
| [OneSearch-VL/OneSearch-VL-RL-10K](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-RL-10K/tree/1e0c8334a1604d8cc8e0edabf34c15c94d2899e2) | `1e0c8334a1604d8cc8e0edabf34c15c94d2899e2` | 214,472,195,071 |
| [OneSearch-VL/OneSearch-MI-Bench](https://huggingface.co/datasets/OneSearch-VL/OneSearch-MI-Bench/tree/b466c14271948c987568a79700d9077b3a165a71) | `b466c14271948c987568a79700d9077b3a165a71` | 432,779,147 |
| [OneSearch-VL/OneSearch-Video-Bench](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/tree/8694a570d777066e053fc258623a429cc13a2294) | `8694a570d777066e053fc258623a429cc13a2294` | 18,514,944,798 |

合计390,255,302,022 bytes（390.255 GB decimal），不含解压或训练cache；初始复现不必拉训练assets。

- Model只有一个`model.safetensors`，17,534,340,584 bytes；[config](https://huggingface.co/OneSearch-VL/OneSearch-VL-8B/resolve/2623603767b9fef7bab05ad0c3cd4d464301e8df/config.json)为Qwen3VL/bf16，保存时transformers5.2.0，顶层及text_config均use_cache=false。该版本字段不证明minimum runtime，4.57.x必须实测；仓库仅weights，不含optimizer/scheduler/RNG，不能称可resume训练checkpoint。
- SFT ZIP合计135,930,412,777 bytes。HF README只有configs YAML；prepare以repo/HF relocate_media脚本为准。“110K”是发布名，未下载JSON逐条核数。
- RL card声明10,059train-only tasks，single3691/multi2780/video3588。ZIP合计214,423,591,229 bytes。GT需paired `data/`、`evgr_gt/` +媒体；不能只下train.jsonl跑EVGR。
- MI card301QA，实际media manifest910unique PNGs；ZIP432,556,863 bytes。prepare核对byte/尺寸/顺序，需reachablemediaURL。
- Video card307QA/293silentvideos；实际manifest293unique MP4、37354frame slots。MP4ZIP841,949,529 bytes，两个original-resolution PNG ZIP共17,669,892,837 bytes。HF只有ZIP，不提供单个MP4/PNG；官方prepare没有limit/ids，并检查全部媒体。可下MP4ZIP选择解压1.9MB首case做pixel-only smoke，但官方frame工具依赖PNG；不能用MP4解码图替换并称exactbenchmark。参考[prepare](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/8694a570d777066e053fc258623a429cc13a2294/prepare.py)、[card](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/8694a570d777066e053fc258623a429cc13a2294/README.md)。
- 发布Video是1FPS播放的采样frame-grid，无音频、source原时钟与原始sourceURL；公开AV数据不能从该release直接恢复。MP4/PNG alignment信息是发布者完整性声明，不是语义QA审核。

详细资产证据在`work/onesearch-hf-assets-audit.md`、`work/onesearch-hf-assets/audit-computed.json`及pinned API caches。

## 4. 不由Python安装解决的依赖与功能降级

1. Serper text/Lens或完整gateway：missingkey会明确ProviderError。[search.py L109](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/search.py#L109)、[L215](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/search.py#L215)。Jina key可选；Qwen summary默认localhost8000/Qwen3-32B，需要实际服务。summary/read失败可降级snippet，不能假设完整网页阅读等价。
2. 对外静态媒体服务器：ASSET_DIR/ASSET_BASE_URL；[upload.py](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/upload.py#L14)只复制hash命名文件，不启动server；Lens须publicURL，localhost内网URL不够。
3. OCR/PP-Structure：需LAYOUT_PARSING_API_URL及可选token，没有自带OCRserver：[search.py L262](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/search.py#L262)。
4. Super-resolution：需要EDSR_x4.pb或SR_MODEL_PATH，tree没有相应weights；代码用cv2.dnn_superres，但requirements列opencv-python-headless而非contrib包。模型不存在时返回原图，cv2失败时也保留原图，应记录为no-op而非实际增强。[image_engines.py L191](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/image_engines.py#L191)。
5. Eval必须JUDGE_API_KEY；RL additionally requiresaccuracy/queryjudge+SeedmultimodalEVGRjudge，WANDB若选择W&Blogging也需认证；无服务的dry-run不是完整复现。
6. 推理requirements松bound：[requirements](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/requirements.txt)含torch>=2.4、transformers>=4.55、qwen-vl-utils/decord，ffmpeg外部安装。需Pixi固定实际解版本。SFT pyproject明确排除transformers4.57.0（[L43](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/SFT/pyproject.toml#L43)）。
7. RL不适合CPU复现：verl sglangextra固定sglang0.5.5/torch2.8.0；Megatron/mbridge、TransformerEngine与attentionkernels需CUDA及allocatedGPU：[setup.py L53](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/verl/setup.py#L53)。单节点配置本身genTP4默认8GPUs，不能把GPUS_PER_NODE改成1就认定compatible。
8. 已发布源码是SFT/RL/inference及releaseexport；本次tree没有识别到论文完整VGEG数据合成engine入口。不能仅凭数据发布声称其上游250万视频→QA流水线可复跑。

## 5. License与复现边界

- Root code[LICENSE](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/LICENSE)为Apache2.0；[THIRD_PARTY_NOTICES](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/THIRD_PARTY_NOTICES.md)明确vendoredcopies有localmodifications并保留各自license。
- `RL/LICENSE`另为MIT（copyright2026WenxuanHuang）；Megatron-LM与mbridge license首节为NVIDIA3-clause-style条款并有后续third-party条款，不能统一写成全部Apache。LLaMA-Factory/rLLM/verl notices链接各自license。
- HF模型/数据未发现license声明；公开/non-gated不自动赋予codeApachelicense，外部素材/APIterms另计。报告只是读源状态，不建议阻止用户已授权的本地科研复现，但发布衍生assets前要确立条款。
- 能独立推进的是PixiCPU安装、native checkpoint短生成、静态crop/frame解析、metadata和pathchecks；有实际服务才进入fulltoolretrieval与officialjudgeeval；GPU+models+data+services完整才进入训练。
- 现有代码与服务快照、发布权重可以组成可运行基础；不能据此确认论文所有表格或训练预算可严格复现。API/网页动态、发布RLOOvs论文GRPO、LR差异、缺少训练state及VGEGengine都需单列。

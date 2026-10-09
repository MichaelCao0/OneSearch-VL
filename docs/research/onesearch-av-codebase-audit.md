# OneSearch-VL → AV Deep Research：只读 codebase audit

日期：2026-10-09。用户指定未来 codebase 为 `https://github.com/MichaelCao0/OneSearch-VL`，后续执行环境为 `coder ssh 2` + Pixi，并先完成官方复现。本轮仅读取公开源码和已有本地研究计划；没有连接/修改远端仓库，没有安装环境、下载权重/benchmark、执行推理或训练，也没有实现研究方法。

## 审查版本与结论

- Fork HEAD：`0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd`。
- 同时读得 upstream `appletea233/OneSearch-VL` HEAD 为同一 SHA；这只证明检查时的两份 HEAD 一致。
- GitHub recursive tree 的 `truncated=false`；本地只读源码在 `work/onesearch-av-audit-src/`，使用 sparse checkout，没有改源文件。
- 依据还包括 `outputs/AV-DeepResearch-Research-Plan-v1-2026-10-09.md` 与已有 OneSearch paper/README 阅读材料。

**判断：可以作为视觉研究 agent 和训练基础设施的 codebase，但不是现成的原生 AV researcher。** 最小可行路线是在独立研究组件中增加多源媒体 registry、固定原生 AV observer、可审核 evidence records 和决策模块，复用 controller、检索与 rollout/训练协议。保持发布 checkpoint 的原始视觉推理路径用于官方复现。直接将 Qwen3-VL 改为原生 audio policy 将涉及模型、processor、multimodal rollout 与训练 backend，不能算小型工具嫁接。

## 源码架构与最小可分离接点

以下路径均相对仓库根目录。`W` 代表 `RL/rllm/vision_deepresearch_async_workflow/`。

| 层 | 已有模块与实际职责 | 可复用接点 / 当前缺口 |
|---|---|---|
| 推理入口 | `onesearch_vl/run_infer.py` 转发 modality/backend；`run_infer_single.py`、`run_infer_mi.py`、`run_infer_video.py` 读取对应 inputs/manifest | 官方复现入口；未来另加研究任务 adapter，避免把开放报告任务冒充官方 QA CSV。 |
| 推理状态与 loop | `onesearch_vl/onesearch_infer/pipeline.py::_bootstrap_video`、`process_single_case`、`run_agent_loop`；状态是 messages、`image_paths_dict` 和 trajectory | 最小决策接点是 `runner.infer()` 前及 tool observation 返回后。没有运行时联合 hypothesis、claim dependency、来源独立性状态。宜另建 sidecar 状态及 selector，而非重写视觉 loop。 |
| 模型 backend | `onesearch_vl/onesearch_infer/runners.py::BaseRunner` 及 Transformers/vLLM/HTTP runners；`config.py::ModelSpec` | observer 与 controller 可分离。已有 capability 主要是 `supports_video`，没有 audio capability；更换 HTTP 地址不自动得到 audio 支持。 |
| 媒体消息 | `onesearch_vl/onesearch_infer/messages.py::to_qwen3vl_messages`、`to_oai_messages`、`to_claude_messages` | 处理 text/image/video。没有 audio payload、采样率、声事件 span 或声画同步契约。外部 AV observer 的文本/证据返回可先保持 controller ABI；原生 audio tokens 则需要新模型路径。 |
| 推理工具 | `onesearch_vl/onesearch_infer/tools.py::execute_tool`，返回 `(message, new_images)`；tool schemas 来自 `prompts/` | dispatcher 是将来 AV tool adapter 的接点，但目前返回契约针对视觉产物，不能只登记新工具名。同步 audio/video span 与 lineage 应作为独立结构化产物。 |
| RL 工具 | `W/deepresearch_tools_async_executor.py::DEEPRESEARCH_TOOLS/get_all_tools`，`W/tools/shared.py::DeepResearchTool`，`W/tools/video_tools.py` 与 `visual_tools.py` | 复用有界执行、超时、cache、统计；与 inference dispatcher 是两套执行实现。新工具需要两边一致契约，不能只改一个注册表。 |
| Prompt 契约 | `onesearch_vl/onesearch_infer/prompts/__init__.py` 限定 `single_image/multi_image/video`；`W/prompts.py` 共享 actor prompts；`W/tool_contract.py::validate_sft_tools` 核验 SFT inventory | 复现路径严格保持。研究路径需版本化新 modality/schema 并更新契约；仅传 custom prompt 不足以启用工具，agent 仍按现有 modality 过滤 active tools。 |
| RL agent | `W/deepresearch_agent.py::MultiTurnReactAgent._run/custom_call_tool/run` | `_run` 建立 message/history 和媒体 handles；`custom_call_tool` dispatch；工具产物通过 images/videos + ordered_content 回到模型。没有公开 `run_from_state`/fork-state 接口；`run` reset 后 bootstrap 新 episode。 |
| RL episode | `W/deepresearch_workflow.py::DeepResearchWorkflow.run/_convert_result_to_episode` | 转为 rLLM `Episode/Trajectory/Step`，已有 action、observation、chat history、step errors、token/timing/tool statistics。未来研究 state/evidence/cost labels 可放独立序列化字段，避免进入 gold-free actor input。 |
| Evidence/reward | `W/evidence.py`、`reward_data.py`、`trace_reward.py`、`ground_reward.py`、`reward.py`、`reward_config.py` | 可借鉴“仅执行后的 observations 构成 evidence”和裁决输出验证；不是 AV-GCR scorer。音频、跨源绑定、不可判定、独立来源和报告额外断言均需另设协议。 |
| Reward hook | `RL/rllm/rllm/engine/agent_workflow_engine.py::_apply_rewards` 在转换训练数据前调用 formal EVGR；`W/deepresearch_workflow.py` 先算 provisional accuracy/query/format reward | future reward 插件应从该边界分离，而非改全局 trainer。官方 base reward 与 formal hook 的真实覆盖率必须分别核验。 |
| SFT | `SFT/examples/onesearch/qwen3_vl_8b_joint_110k.yaml`、`SFT/run_train.sh`、`SFT/scripts/relocate_media.py`、`SFT/data/dataset_info.json` | 可复用 policy SFT pipeline；当前 template 为 `qwen3_vl`，视觉/projector 冻结。外部 observer 的文本 observations 比 native audio 模型迁移更小，但数据格式与 observation loss mask 仍要真实验证。 |
| RL train | `W/train_deepresearch_workflow_megatron.py`、`W/run/qwen3-vl-8b-single-node.sh`；bundled rLLM/verl、Megatron-LM、mbridge | 是大型异步训练 stack；不是第一轮 10/40 案取证实验的必要依赖。已有发布入口 train-only，正式 held-out evaluation 需独立入口。 |

## 原生 audio 与时间引用的直接缺口

1. `onesearch_vl/onesearch_infer/prompts/video.txt:2` 明确输入为 silent video，no audio/captions/metadata。现有工具集合没有 ASR、read_audio、audio search、声源或声事件工具。
2. `onesearch_vl/preprocessing/video_prep.py:101` 与 inference `video_io.py:266`、RL `W/tools/video_io.py:333` 的 ffmpeg 命令使用 `-an`。现有加工输出主动去掉音轨。
3. `select_timespan/select_frame` 操作的是 sampled-frame grid index，而不是原始 AV 中任意秒数；grid playback 为 1 fps，默认输入采样和模型读取设置可为 2 fps，这些是不同时间概念。
4. 两份 `video_io.py::validate_grid_contract` 可以核验 `source_frame_indices/source_timestamps_s` 长度，但两份 `build_grid_from_manifest` 仅返回 video URL、frame URLs、n_frames、playback duration、grid_fps、url_offset；原始时间映射没有进入该 runtime grid。`grid_slice` 同样没有保留原始 timestamp 数组。因此不能把当前 frame index 或 subclip 秒数直接作为报告中的原始媒体 timestamp。
5. 初始注册的是一个 `video_1`；虽可在 handle dict 中存多个 grid，发布 bootstrap/task schema 并未提供完整多来源 AV registry。事件/录制/转载/独立观察关系也没有现成支持。
6. `text_search` 是检索、页面读取和模型摘要的组合：`search.py::summarize` 最多读取前 2,000 字符，失败时回退到 content 前 500 字符；actor 返回 URL/title/summary，没有完整页面快照和可靠原文 quote span。报告级核验需要保存原始结果与 reader body，而不能把生成 summary 当成原文。

最小 AV 路线应保留原始媒体（含音轨）及每来源独立 local clock，以原始 `source_id + version/hash + [start_s,end_s)` 调用固定 observer；采样 grid 只是视觉辅助表示。不要把原始音轨直接拼到重定时的 1-fps grid 上，否则声画同步会失真。

## Research plan 的 graft 边界：未来建议，未实现

建议将以下五类研究组件独立于官方 loop/backend：

1. **Media/evidence registry**：原始 source/version/hash、本地时间范围、派生素材映射、谱系、证据类型。已有 `image_paths_dict` 继续处理官方视觉 handles，新 registry 保存完整 AV 引用。
2. **AV observer tools**：真正处理原始音频与画面，返回 evidence records、模型观察及原始片段引用。controller 仍可为 Qwen3-VL；这应称“固定 AV observer + policy controller”，不称 checkpoint 自身原生听懂声音。
3. **Decision adapter**：在 LLM/action 边界输入 gold-free `O_t/H_t/D_t/B_t`，输出候选行动或结构化后果；执行后的 observer receipt 才更新 verified evidence。普通 ReAct/flat/Q 等对照复用相同 observer 与信息。
4. **Branch runner**：显式序列化 prefix messages、registry/state、剩余预算、随机种子、工具缓存身份；执行一个候选后 continuation。现有完整轨迹 JSON 不是可直接恢复的完整环境 snapshot，现有 resume 仅跳过已完成案例。此接口需独立开发与测试，不能宣称已经支持同前缀 counterfactual 分支。
5. **Research scorer**：GCR、重大额外无依据断言、binding、duplicate-independent、预算。保持官方 accuracy/EVGR 协议独立，研究 reward 不替换官方复现 reward。

特别注意：`reward_data.validate_gt_fields` 严格限定视觉 modality 与 GT 字段，多源音频字段不能直接塞入现有 GT；当前 Ground rubric 对 video 要求 successful select_frame receipt 与 subsequent retrieval use，Trace 只看 retrieval evidence。一个凭原始 AV 直接正确解决的研究结论可能在该 reward 下得 0，这说明它与研究计划任务目标不同。当前 published judge 使用 continuous holistic score，semantic anchors 是校准参考；不要把论文描述的离散档位当成代码真实实现。

## 官方复现优先顺序与验收

### P0：coder ssh 2 / Pixi preflight

之后由主任务执行 fresh inventory：主机身份、GPU 数/显存、CUDA/driver、磁盘、ffmpeg、已有 Pixi 与模型缓存。历史硬件数量不能作为本次可用性证据。不要用其它 Coder workspace 的成功环境代替 workspace 2。

建议独立 Pixi environments 或 manifests：inference、SFT、RL，分别锁版本；先 inference。Inference `requirements.txt` 有宽松下界，并非可复现 lockfile。RL 的 bundled frameworks 必须使用仓库版本，不能替换成 unmodified pip upstream 后称官方运行。

外部服务最低核验：Serper text/Lens、页面 reader、Qwen evidence summarizer、可达媒体发布 URL、evaluation judge。图像搜索依赖检索服务可访问的 HTTP(S) URL；上传器只复制到 static asset directory，不启动服务器。HTTP 200 也不证明返回的是有效媒体/页面/摘要，需要检查真实 observation 和 artifacts。仅检查变量存在不算服务通过，日志不得输出 credentials。

### P1：发布 checkpoint 的小规模官方推理

- 用 `OneSearch-VL/OneSearch-VL-8B` 的发布 checkpoint 与官方输入 preparation，先 single-image 1 案，再 MI/video 各 2–4 案。
- 逐项核验初始媒体可见、一次真实检索、frame/crop/clip 输出、原始-resolution frame 对应、final response、trajectory JSON 和 judge。使用官方 `--limit`/case selection 做 bounded smoke，不改变 actor prompt/reward。
- 建立 run manifest：code SHA、model revision、Pixi lock hash、dataset/media hashes、preparation 参数、model/observer/judge identities、tool errors/retries、cost。
- 使用全新版本化 output directory。`run_infer_video.py::_is_complete_trajectory` 主要检查 JSON 可读及 status 非 error，没有校验模型/config identity；旧同 ID 轨迹可能被 resume 复用。只在完整 run identity 一致时恢复，不能复用不同 baseline 的 output directory。
- 8–16 案运行通过是 integration smoke，不是完整 benchmark 复现或方法收益。

### P2：完整官方 benchmark inference + eval

- OneSearch-MI 301 案与 OneSearch-Video 307 案分别执行和评分；先确认 benchmark preparation 的 image-index/frame mapping。
- `run_eval.sh image/video` 对应协议分别是 `onesearch` / 默认 `videodr`；judge 默认 gpt-4o。保存评测服务与模型 revision，不能把 judge 更换后的结果视为同协议严格匹配。
- `eval_common.py` 主要比较 question/gold answer/final response，不审核工具 evidence/citations，也不是 GCR。报告 coverage、工具/actor/judge failures，不能只挑成功轨迹报 accuracy；官方 incomplete report 的 null accuracy 需保留。
- 网络检索内容与 provider 可变化；记录快照和降级路径。数值接近作者报告是必要比较之一，不能用单次成功样例替代全量结果。

### P3：SFT / RL 训练连通性与发布 recipe

- SFT 先小样本 1–2 optimizer updates，验证真实可训练参数、工具 observation loss mask、checkpoint 保存和重载；这只是连通性。发布 YAML 为 full finetuning、vision/projector frozen、cutoff 32768、lr `1e-5`、8 epochs、ZeRO-2；原论文表给 `2e-5`，先记录差异并固定选择，不能称两者完全相同。
- RL 先 `reward_preflight`（offline，不发 judge 请求）、launcher `--dry-run`（不证明 GPU 可运行），再少量真实 rollout/两个 EVGR judges，最后 one-update smoke。
- 发布 single-node recipe 默认 8 GPUs，rollout TP=4、train TP=2；少 GPU 时不能仅改 `GPUS_PER_NODE` 而保留该拓扑。缩小拓扑/批量/响应数只能标为 bounded compatibility smoke。
- 发布 single-node launcher 的 `adv_estimator="rloo"`、responses=8、prompt batch=256、minibatch=64、201 steps；论文讨论 GRPO。先复现发布 recipe，再单列 paper-aligned variation，避免悄悄更换 estimator。
- `train_deepresearch_workflow_megatron.py` 清空 val_files、`val_dataset=None`；launcher禁用 before/final validation。训练 rollout reward 不是 held-out eval；训练后须显式运行冻结 checkpoint 的官方独立 benchmark。
- reward smoke 验收包含 formal EVGR 是否实际 applied、Trace/Ground request/validation failures、真实 evidence inputs、finite final rewards。离线 GT alignment 不证明 judge 语义可靠。

## 进入研究实现的停止线

在 published checkpoint 的官方工具推理和 benchmark evaluation 尚未通过之前，不启动 hypothesis/graph/VoI/branch/GCR 方法实现。若外部检索、媒体出版、frame mapping、judge coverage 或 checkpoint加载失败，优先修复官方路径。若只有 bounded 小样本通过，应按 smoke 状态报告，不能写成完整官方复现。

## 主要源码入口（固定 SHA）

- [Inference loop](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/pipeline.py#L440)
- [Inference tool dispatcher](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/tools.py#L435)
- [Message adapters](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/messages.py#L204)
- [Runtime grid normalization](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/onesearch_infer/video_io.py#L95)
- [RL agent](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/deepresearch_agent.py#L398)
- [RL workflow / Episode conversion](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/deepresearch_workflow.py#L887)
- [GT contract](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/reward_data.py)
- [Reward hook](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/rllm/engine/agent_workflow_engine.py#L273)
- [SFT recipe](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/SFT/examples/onesearch/qwen3_vl_8b_joint_110k.yaml)
- [RL launcher](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-single-node.sh#L56)
- [Train-only entry](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/RL/rllm/vision_deepresearch_async_workflow/train_deepresearch_workflow_megatron.py)
- [Official evaluation protocol](https://github.com/MichaelCao0/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/onesearch_vl/eval_common.py#L50)

## 本轮证据边界

已核验：公开源码 SHA、上游/ fork HEAD 一致、文件级接口、音频删除与消息缺口、时间映射丢失、recipe defaults 和评测协议。未核验：coder ssh 2 当前硬件与依赖、真实服务可达性、权重/数据可用性、模型加载、任何推理/评测/训练。上述 graft 是设计建议，不是现有功能。

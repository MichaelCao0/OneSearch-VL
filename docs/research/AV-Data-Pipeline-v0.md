# AV-ResearchBench 与 OneSearch-AV-SFT 数据管线 v0

日期：2026-10-10。实施依据：总体 Research Proposal v2。主工作目录 `/home/jovyan/OneSearch-VL`，Coder workspace 3，branch `research/av-deepresearch`，Pixi 管理环境。

## 本轮交付

已实现 actor task/reference 分离、完整有效证据包评分、来源隔离的执行轨迹验证、ShareGPT 导出、真实工具 episode runner 和 16 个 development 候选的可视化审核页。候选问题从实际已登记自然素材生成；reference 的事实、区间和证据包尚待回放标注。这批数据的价值是审查任务设计、运行构建流程，为后续扩量提供明确样例。

| 阶段 | 本轮实际状态 |
|---|---|
| Development 素材 | 10 AV + 6 冻结说明页，299.36 秒 AV，原始文件 hash 检查通过 |
| 候选任务 | 16，每个任务族 4；具体中文问题，每题 3 个初始目标项 |
| 公共搜索环境 | 所有题共享一个 SQLite FTS corpus；目前仅 metadata/正文索引 |
| 工具连通性 | batch builder 执行 4 次 search + 4 次 read；另一个脚本 episode 执行 search/read |
| 语义 reference | 0；用户 suitability 审核与正式双人语义标注分开 |
| 新训练候选来源 | 新采集 4 AV + 4 说明页，90.70 秒，53.42 MB；4 个训练题型候选，字节级无 dev 重复；录制家族和语义尚未裁决 |
| 真正模型 teacher 轨迹 | 0；当前进程无 teacher API 配置；原生 observer callable 尚未接入此 runner |
| SFT 保留数据 | 0；脚本/dev episode 已实际尝试导出并被拒绝 |
| GPU / SFT | 本轮 0 GPU job seconds；未启动训练；此前剩余 GPU 预算约 1373 job seconds 保持不变 |

## 文件和接口

- `research/av_bench/contracts.py`：`validate_task(task, scope)` 和 `validate_reference(reference, task, sources)`。公共 task 支持 0–2 个 initial source，1–6 个 necessary target；新 schema 独立于较早的 binding/gold contract，避免改变旧 run 含义。
- `research/av_bench/evaluate.py`：`score_prediction(task, reference, prediction, sources, k=5, iou_threshold=.5)`；`aggregate_scores(rows)` 包含失败任务。只接受已裁决 reference；未知或格式错误预测记失败，不删除。
- `research/av_data/build_research_batch.py`：生成公共问题、私有候选来源、空 reference、shared corpus、输入 hashes、dev reservation 和实际工具 receipts。
- `research/av_data/visualize_batch.py`：原始 PTS 抽帧联系图，原片播放链接，问题/风险/必要目标，用户选择和 JSON 下载。用户意见不自动变成 gold。
- `research/av_train/trajectory.py`：执行 step hashes、source split 校验和轨迹 contract。hash 是完整性证据，实际执行由独立 replay 验证。
- `research/av_train/rollout.py`：`ToolRuntime` 和 `run_episode(task, runtime, controller, teacher)`。controller 只看 public task 与实际历史。`inspect_av` 无语义 backend 时明确返回 unavailable；模型 teacher 必须显式指定。
- `research/av_train/export.py`：`export_trajectory` / `export_batch`。要求 train-only 来源、模型 teacher、语义 AV observation、引用落在实际观察区间、成功语义验收以及独立 replay；生成 retained JSON、lineage、rejections、dataset_info 和 loss masking 配置。

## 场景与任务分布

| 场景 | 当前素材 | 优先构建方式 |
|---|---|---|
| 技术演示/机械 | 三份蒸汽机、一份印刷机 | 按运动周期、声学节律和部件差异组织自然相似来源组 |
| 现场演奏 | Stroh violin、cello ensemble、gong/drum、cardio drumming | 弓弦/敲击动作、音符/节奏、不同录制的实际音轨比较 |
| 动物 | 一份 corella feeding | 喙/头部活动实例、实际背景声与动物声；需扩同类录制 |
| 环境 | 一份 waterfall | 连续/离散声音和画面运动；需扩水流、降雨等相似来源 |

四类任务各 4 题：Source discovery、Event-instance localization、Cross-source comparison、Research composition。第一批例子保留素材的真实长度 9–121 秒；没有把 observer 的 2 秒 smoke 限制变成 benchmark 定义。

候选作者面板列出备选素材，actor 不看该 shortlist。全部 dev 来源已在旧 run 中探索过，不能随机切成新的 held-out test 或训练集。正式 test 和训练数据从新的录制/事件/近重复连通分量采集；新页面不同不等于独立录制，仍需家族审核。

## 评测维度与模型

| 维度 | v0 可执行指标 | 后续分析 |
|---|---|---|
| 来源搜索 | complete-package source recall@k；best-valid-package coverage@k；package-completion reciprocal rank | source recall、MRR/nDCG、episode first-discovery curve |
| 定位 | 对每个有效包，逐条 span 的 IoU + 事件实例一致性，取最佳包覆盖 | IoU 0.3/0.5/0.7、多次重复、不同媒体长度 |
| 必要事实 | accepted-values 的规范化字符串/数字准确率、all-correct | 自由文本事实需独立语义审阅或校准 judge，不能直接套 exact match |
| 证据研究成功 | EBS = 必要事实正确 AND 一个完整引用包被支持 AND 未超预算 | 端到端成功与各阶段 oracle 差距 |
| 成本 | measured tool calls、logical requested AV seconds；可附加 controller/observer tokens | frames/audio 准备成本、真实 GPU 秒数、wall time |

多个可接受包以 OR 组合，每个包内部是 AND；只取一个完整包即可。不要把所有等价来源取并集后要求系统全找齐。`event_id` 若进入答案协议，必须是任务要求中可观察的序号/实例描述，不是 actor 不可能知道的私有随机 ID。

v0 尚未实现 nDCG、自由文本 judge、real perceptual cost 完整聚合与批量主榜。这些不应被报告为当前运行结果。

| Baseline | 控制/感知路线 | 本轮运行状态 |
|---|---|---|
| B0 question-only / initial-AV-only | 固定 controller；无外部搜索 | 已定义，未评测 |
| B1 metadata/text search | 固定 controller；无 AV 派生信息 | 仅工具 smoke；不是语义 baseline 分数 |
| B2 ASR+frames workflow | 同 controller，固定 ASR/frames reader | 已定义，未评测 |
| B3 joint-AV workflow | 同 controller，固定 Qwen2.5-Omni-7B observer | observer 修复另有证据；本批未接入和评测 |
| B4 generic iterative AV agent | query/search/inspect/rewrite，固定 observer | runner 接口已建；未运行语义 benchmark |
| B5a OneSearch-AV-Prompted | 已发布 OneSearch-VL-8B controller + 固定 observer | 主训练前对照；本批未评测 |
| B5b OneSearch-AV-SFT | 同一 starting checkpoint，训练 controller | 数据产出和验收后启动 |
| B5c SFT + instance reranking | 同一 SFT policy，额外实例评分 | 由 dev 瓶颈确定，未实现 |
| B6 strong omni agent | 实际部署的 omni model/OmniAtlas adaptation | 服务未配置，不计已运行模型 |
| D source/span/fact oracle | 分阶段诊断 | 待 adjudicated reference |

优先锁定 B5a/B5b 同 checkpoint、observer、tool/index 与预算，做训练前后对照。Observer 本轮沿用 Qwen/Qwen2.5-Omni-7B 的既定修订；不把 modular controller + observer 叫 native Omni-SFT。训练 baseline 是论文正式组成部分。

## 可重跑命令

在 repo root 执行，所有输出目录须不存在：

```bash
pixi run --locked python -m research.av_data.build_research_batch \
  --sources research/runs/phase1-20261009/natural-pilot-v1/sources.jsonl \
  --sources research/runs/next-stage-20261010-v1/cases-v3/sources.jsonl \
  --out-dir research/runs/av-data-pipeline-20261010-v0

pixi run --locked python -m research.av_data.visualize_batch \
  --batch-dir research/runs/av-data-pipeline-20261010-v0 \
  --out-dir research/runs/av-data-pipeline-20261010-v0/review-v1

pixi run --locked python -m research.av_train.rollout \
  --batch-dir research/runs/av-data-pipeline-20261010-v0 \
  --out-dir research/runs/av-data-pipeline-20261010-v0/connectivity-episode-v1 \
  --fixture-smoke

pixi run --locked python -m unittest discover \
  -s validation/research -p test_research_pipeline.py -v
```

真实 teacher 通过进程配置 `AV_TEACHER_BASE_URL` / `AV_TEACHER_API_KEY`，显式传 `--teacher-model` 和 `--teacher-revision`；不在文档或仓库写密钥。CLI 当前没有默认语义 observer，接入时调用 `ToolRuntime(..., observer=...)` 的真实 native backend adapter，并保存模型、媒体与成本 receipts。应先做 4–8 题真实多轮 rollout 检查，再扩至 64–256 条连通集；没有配置时不拿脚本生成假轨迹。

## SFT 数据保留与训练前条件

导出的 conversation 遵循 `human → gpt(action) → observation → gpt(action/final)`；controller 的 JSON envelope 与 rollout 一致。工具定义写入 system，不触发与 JSON action 不兼容的 native function formatter。Observation 不作为模型目标，teacher 私有解释和 reference 不入历史；final 的 measured cost 仅用于审计，不作为要生成的标签。

训练配置必须 `train_on_prompt: false`、`mask_history: false`。下一步在真实 LLaMA-Factory 环境用本地 tokenizer 检查每轮标签，确认 observation/user/system 的 `IGNORE_INDEX`、每条 action/final 的监督范围和整条轨迹长度。当前 CPU Pixi 环境缺少 datasets/peft/trl；不将角色格式测试叫 token-level 训练验证。

SFT 启动顺序：独立 train 来源与必要事实 → 无 reference teacher rollout → 实际工具执行 → 基于 train reference 验收/多路径保留 → replay/export → token/length audit → 小规模真实 backward/save/reload → 同预算 Prompted vs SFT development 对比 → 约 2K pilot，之后按收益决定 10–20K main。

当前 16 个候选的用户审核重点是：问题自然不自然、是否值得研究、声画是否真正帮助回答、具体事实能否稳定标注、是否需要多份原始 AV。适合的题型进入下一轮人工回放标注；弱题和捷径题直接改写或淘汰，再采更丰富的自然相似候选。

## 新训练素材采集与审核

使用已有 collect_commons.run，逐域最多 2 次下载、60 MB，共 4 个查询（总 8 次尝试/240 MB 上限）。实际 4 次下载，共 53,422,671 bytes，4 份通过双流/音量筛查的自然 AV。查询为 tractor engine、saxophone street、duck pond、rain thunder；duck pond 实际返回池塘降雨，不是已核实的动物素材。

`research/av_train/prepare_pool.py` 对比已预留 dev 的 SHA-256，去除同字节重复及 silent/rejected acquisitions；ingest 原 AV/冻结说明页，产生私有训练候选、家族审查表和 authoring corpus。共有 4 个中文训练题型候选，四类各 1；不输出 actor tasks 或冻结 split，teacher_rollout_allowed=false。录制/事件/近重复家族裁决和事实 reference 完成后，才能启用真实 teacher。

训练候选目录：`research/runs/av-data-pipeline-20261010-v0/train-authoring-pool`；可视化目录 `review-v0`。Development 最终审核页目录更新为 `research/runs/av-data-pipeline-20261010-v0/review-v2`，增加对应冻结说明页正文与原页链接。

## 本轮验证结果

最终 Pixi research regression 165/165 通过，其中新增 35 个数据管线 contract/integration tests；Python compileall 通过。审核页面在本机浏览器加载确认，缩略图来自 14 份实际原片并保留原始 PTS。在线播放仍取决于 Commons 访问和浏览器对原编码的支持，提供原片页面入口。没有执行自由文本语义评分、teacher AV rollout、token-label audit 或 SFT。

# OneSearch-AV：三条实验主线与执行计划

工作标题：**OneSearch-AV: Learning to Search from Audio-Visual Clues**。

执行时间：2026-10-16。现场读数见文末证据快照；实验矩阵、实际执行与正式结果分别记录。

## 1. 研究主张

用户提供一段演出、赛事或现场活动片段时，往往不知道其中实体的准确名称。声音与画面提供辨识入口，搜索提供片段之外的知识。我们训练一个原生 Omni 模型，使其从声画线索出发形成查询、读取来源、必要时再检查真实片段，并完成搜索问答。

三条实验主线分别回答：**搜索能力能否随训练与数据规模增长；声音在什么任务中改变搜索结果；什么数据构造方式更有利于学习。** 数据引擎、模型与 benchmark 围绕同一个任务组织。完整目标保持 **20K train + 500 train-validation + 200 dev + 1K test**，先完成 2K 训练节点。

| 主线 | 核心问题 | 对应证据 |
|---|---|---|
| A：学习与规模 | 完整轨迹与更多数据能否提高任务成功率？ | 同基座 500/2K × trajectory/answer-only，以及有效 prompted search |
| B：声音与互补 | 声音是否改善辨认、查询和来源命中？ | 独立 benchmark 上 AV、muted、audio-only、video+ASR 同题比较 |
| C：数据方法 | AV 条件构造和模态分层是否提高学习效率？ | 同来源构造诊断、共同 prompt 构造消融、匹配预算训练 |

## 2. A：学习与规模

学生固定为 `Qwen/Qwen2.5-Omni-7B`，revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`；沿用原仓库 `SFT/src/train.py`。初始媒体与 `inspect_av` 返回的真实裁剪均进入同一个 Omni Thinker。

主实验采用 full Thinker：训练语言 decoder 与 lm_head，冻结 AV 塔及 projector，排除 Talker。CPU meta 已验证 7,615,616,512 个可训练参数和 1,316,197,376 个冻结参数。

| 组别 | 问题数 | 监督 | Epoch | 计划更新数 |
|---|---:|---|---:|---:|
| full_500_trajectory | 500 | 搜索轨迹 | 2 | 250 |
| full_500_answer_only | 500 | 同题答案 | 2 | 250 |
| full_2000_trajectory | 2,000 | 搜索轨迹 | 2 | 1,000 |
| full_2000_answer_only | 2,000 | 同题答案 | 2 | 1,000 |

学习率 `1e-5`，effective batch 4，各组从相同基座独立初始化。新版 clean500 必须严格嵌套于 clean2000，同一规模的两种监督使用完全相同的问题。规模比较固定 epoch 和配方，同时报告 context/supervised tokens、更新数和 GPU hours；2K 的计划更新数为 500 的四倍。

基线包括 direct base 和 dev-selected prompted search。Prompt 只在 dev 选择，先检查真实工具执行，再检查 AV 线索到查询的能力。统一可见输入、工具、最大轮数、返回长度与生成预算；报告实际工具执行和最终完成率。

主要指标为 `answer_success`、`supported_success`，以及目标来源命中、有效工具调用和最终完成率。使用预先固定的问题清单，报告未完成和未知评分。

候选资源为 4 张每张至少 80 GiB 的 GPU 与 ZeRO-2，此时 micro batch 1、accumulation 1。先执行真实 AV forward/backward 和至少 3 次 optimizer updates，验证有限梯度、decoder 变化、AV 冻结和 checkpoint 回读，再开始完整训练。四个模板已完成；当前标记 `template_not_launchable`，2K dataset 必须由 hash 核验后的 freeze 解析。真实 full Thinker GPU smoke 尚未执行。

历史 500 题 LoRA 结果单独保留。新版 clean500 保留旧集合中 429 题并替换后续被排除的 71 题，后续规模曲线使用新版一致口径。每个正式训练完成后立即评测。

## 3. B：声音与互补

已冻结 **192 个 dev 候选，126 段录制、126 个自动家族**。17:31 对包含新增子批次在内的 **1,405 个合格训练题**复核：录制、家族、媒体交集均为 0，目标 URL 在实际训练轨迹中曝光为 0。人工 gold 当前为 0。

每题核实问题、答案事实、支持来源、输入区间与实体关联，再标注模态角色和研究操作。模态标签为 `audio_primary`、`joint_av`、`visual_primary`、`speech_text_control`，依据实际片段判断；领域类别不替代模态标签。操作覆盖 single-anchor lookup、multi-hop、knowledge-conditioned count、multi-anchor join、arithmetic、comparison，实际覆盖由标注统计。

| 输入 | 可见内容 | 全过程控制 |
|---|---|---|
| native AV | 原始声音与画面 | 保留时间对应 |
| muted video | 去音轨视频 | 保留编码视频包；所有 inspect_av 返回同样去音轨 |
| audio-only | 同一区间原声 | 无帧或由画面派生的 caption |
| video+ASR | 静音视频与固定转写 | 同一区间、固定 ASR、各策略共用 |

比较同题 answer/supported success、来源命中、查询/实体变化和工具预算，按录制家族 bootstrap 置信区间。选题和标签不依赖学生是否答对。

已从清单中按固定 hash 选择 **32 题、32 录制**，排除此前已付费审核的 task 和输入媒体，启动原生 AV 自动预审。预算 32 次新调用、1 worker。自动 verdict 是 silver 建议，人审结论独立记录。

## 4. C：数据方法

### 4.1 同来源构造诊断，已启动

按固定 hash 选择 **32 个训练来源、32 个不同家族**，不参考学生结果。核验并复用历史成功的 AV 出题回执、实际媒体和 publisher 文本，再对同来源新增 publisher-text-only 出题。

逐来源沿用原 teacher model ID、`maxOutputTokens=4096`、temperature 0.3。对照请求只发送文本，记录 `has_media=false`、请求 hash 和原回执 hash；最多 32 次新调用、1 worker。检查封闭 schema、答案字面支持、题干答案泄漏和重复事实。新候选经过后续轨迹与审核后才可进入训练。

本轮是构造诊断：AV 侧为历史成功来源，且两侧 prompt/schema 有差异。它用于发现 publisher-only 题型与实际 AV 可解性之间的问题；不能单独估计 AV 输入的因果收益。失败或未知付费回执保留，不自动重放。

### 4.2 正式构造消融

在新来源上固定共同 prompt、schema、teacher、输出预算和事实支持规则，比较 `AV+publisher` 与 `publisher-only`，只改变是否提供 AV。两臂问题混合盲审：实际输入匹配、线索到实体的搜索路径、外部知识需求、答案支持、事实独立性和操作覆盖。

报告每来源有效问题数、主搜索比例、语义验收产率及成本。再使用同题数、匹配 supervised tokens 的训练比较，检验下游任务成功率。

### 4.3 自然与模态分层采样

准备各 500 题的自然采样与模态分层 trajectory 集合，统一来源独立和质量规则，匹配监督 token 预算。分析整体及各模态层的成功率、来源命中与工具行为。采集 query 权重只作为调度提示。

## 5. 执行顺序与交付

1. 完成本批 114 候选的搜索、保留与 CPU 检查，下一批接续生产；可选问题达到 2K 后冻结，并验证 clean500 嵌套。
2. 汇总两个 32 来源小试的原始回执、失败类别和质量问题，确定共同 prompt 的正式构造消融。
3. 完成 dev 的事实、区间和模态标注，继续独立 test；形成固定评测清单。
4. full Thinker GPU smoke 通过后推进四组训练和即时评测，同时完成有效 prompted baseline。
5. 论文主表/图按学习与规模、模态作用、数据方法组织，分析连接线索、查询、来源命中与最终答案。

本轮四模块合计 **26 项回归通过**，另核验四个训练模板的 full/freeze/no-LoRA/effective-batch 参数。没有新增 GPU job，未 push。数据生产与小试直接运行在 Coder CPU 工作区。

### 17:32 现场结果

| 项目 | 已核验结果 |
|---|---|
| CPU 合格训练问题 | **1,405**；比外层上一汇总新增 46 |
| 排除已知 146 题后的可选问题 | **1,259**，距最终 2K 差 **741** |
| 32 题模态自动预审 | **31 完成 / 1 失败**；28 keep、3 reject 建议 |
| 完成预审的模态角色 | joint 15、visual-primary 11、audio-primary 4、speech-text-primary 1 |
| 32 来源 publisher-only 对照 | **28 完成 / 4 失败**，56 个通过格式和字面支持检查的候选 |
| 对照失败类别 | Gemini 3.1 Pro 的 4 次 HTTP 503；保留回执，无重试 |
| Benchmark 最新隔离复核 | 192 题对 1,405 训练题，来源/家族/媒体交集及目标 URL 曝光均为 0 |
| 证据归档 | 35 个远端文件逐文件 SHA256 核验并回读到本机 |

模态角色统计包含全部 31 个完成的自动 verdict，尚未经过人审；候选数不等于合格训练数。两个小试合计实际调用 64 次，均在预定上限内。预审已给出可用于人审组织的模态分层，publisher-only 候选还需要检查实际 AV 中是否有足够线索；正式共同 prompt 构造消融按第 4 节执行。

## 6. 代码与证据

主仓库：`coder ssh 3:/home/jovyan/OneSearch-VL`，branch `research/av-deepresearch`。

- 新逻辑：`/home/jovyan/OneSearch-VL/research/av_train/experiment_suite.py`
- 测试：`/home/jovyan/OneSearch-VL/validation/test_experiment_suite.py`
- 配置：`/home/jovyan/OneSearch-VL/research/configs/onesearch_av_experiments.v1.json`
- 运行：`/home/jovyan/OneSearch-VL/research/runs/av-three-axis-experiments-20261016-v1/`
- 运行内目录：`benchmark-dev-candidates-v2/`、`training-arm-templates/`、`modality-silver-preaudit/`、`publisher-construction-control/`
- 本机证据：`reports/three-axis-experiments-2026-10-16/`

Benchmark membership SHA256：`d836ab106444903e89c44632696f9175f758ff2f1a17eb28dd085f46e5fd3263`。

矩阵 SHA256：`8be03c87efab9e1878762d3c4c198ccded6077f120685913da3e663048b16bf0`。矩阵为启动前设计快照，实际执行以 worker identity 和结果回执为准。

合格数通过 `verified_pool()` 核验实际成对导出、去重和原 trainer CPU 媒体/标签检查。实时核验同时纳入生产子批次的已完成导出，避免外层等待整批结束造成低报。Answer-only 行不重复计题。

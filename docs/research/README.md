> **2026-10-19 完整在线服务准备**：[Search/Fetch配置与调用预算](OneSearch-AV-Full-Online-Service-and-Budget-2026-10-19.md)。Search正文进入工具返回，Fetch启用；17测试、96条件CPU准备通过。六组共768次上限/¥3.84；GPU尚未提交，尚无新增在线成绩。

> **2026-10-19 评测边界复核**：[倒挂解释与泄露审计](OneSearch-AV-Audio-Inversion-and-Evaluation-Leakage-Audit-2026-10-19.md)。当前500/2K直接样本、登记媒体/家族、目标/支持URL与页标题重叠均0；40+16题是反复使用的诊断/开发集合，dev仅8录制家族/10URL。相同2K答案的3个评分标签变化可改变音频差值正负；不能归因于语音。

> **2026-10-19 修复重评完成（Coder 4）**：[原声/静音与500/2K完整结果](OneSearch-AV-Corrected-Audio-and-2K-Evaluation-2026-10-19.md)。560 rollout无执行错误；联合52有效/4 unknown。500轨迹原声33/52、静音28/52；2K为36/52、34/52。规模和音频差值CI跨0；轨迹SFT学习收益明确，3个评分上下文差异及22翻转案例已归档。五个评测kjob全部结束。

> **2026-10-16 21:59：Full Thinker 与数据方法诊断**：[执行报告](OneSearch-AV-Full-Thinker-Validation-and-Data-Diagnostics-2026-10-16.md)。1652 CPU合格/1506可选；240 benchmark候选，208自动审核。4卡full Thinker完成3次真实更新与checkpoint回读，两个问题×AV/muted四条件推理通过；18项回归通过。

> **2026-10-16 17:32 三条实验主线：学习、模态、数据方法**：[实验设计与执行](OneSearch-AV-Three-Axis-Experiments-v1-2026-10-16.md)。1405 CPU合格/1259可选；192 dev候选对1405训练题隔离复核通过。32题模态预审31完成，32来源publisher对照28完成；四组full Thinker模板和26项回归通过。

> **16:35 Motivation与证据**：[研究主线与进度](OneSearch-AV-Motivation-and-Progress-2026-10-16.md)。1359 CPU合格/1213可选，本批114候选构建共享语料中；192 dev候选。

> **2026-10-16 16:22 批次接续**：1359 CPU合格/1213可选；本批82接受出题候选，192 dev候选；v12继续出题，v13预取新素材并等待正常接续。[执行记录](OneSearch-AV-Continuation-v13-2026-10-16.md)。

> **2026-10-16 16:00**：1359 CPU合格/1213可选；benchmark192候选。恢复v12：复用已准备媒体、2 workers、暂停新增下载。[执行记录](OneSearch-AV-2K-Recovery-v12-2026-10-16.md)。

> **2026-10-16 14:17:26：扩产恢复v11**：1317 CPU合格问题、1171可选，173 dev候选；503停止后已启动成功回执回收和未调用素材续产。详见 [实时进度](OneSearch-AV-Production-Recovery-v11-2026-10-16.md)。

> **2026-10-16 12:07：2K扩产与训练准备**：1175 CPU合格问题，排除已知问题后1029可选；新版500题子集已冻结，自动2K冻结worker运行。153新录制正在出题，benchmark继续。详见 [最新执行记录 v2](OneSearch-AV-2K-Continuation-v2-2026-10-16.md)。

> **11:00 现场读数**：1,151 合格训练题、139 dev 候选；256录制审核106完成、8未知、4执行中。详见 [当前进度](OneSearch-AV-Latest-Status-2026-10-16.md)。

> **2026-10-16 研究方案 v5 与执行**：[完整论文构想及投稿计划](OneSearch-AV-Paper-Dossier-v5-2026-10-16.md) · [当前执行与交付](OneSearch-AV-Execution-Update-v1-2026-10-16.md) · [当日 ledger](OneSearch-AV-Research-Ledger-2026-10-16.md)。训练/benchmark/256录制审核继续；新增24案例通过本机媒体和播放验证。完整20K/500/200/1K目标保持。

> **最新中文论文整理（2026-10-15 v4）**：[论文构想与投稿计划](OneSearch-AV-Paper-Dossier-v4-2026-10-15.md)。1,069 合格训练题、85 dev 候选；同题500训练/评测已完成，heldout trajectory 24/38 对 answer-only 3/38；128录制内容审核、新优先批次24例四模式审核。20K/500/200/1K完整目标继续。

> **最新中文论文整理（2026-10-15 v3）**：[论文构想与投稿计划](OneSearch-AV-Paper-Dossier-v3-2026-10-15.md)。20K train/500 validation、200 dev/1K test；988 合格训练题、同题 answer-only 训练进展、三版 dev prompt 结果、124 题精确媒体质量视图及吞吐分析。

> **最新中文论文整理（2026-10-15 v2）**：[OneSearch-AV 论文构想与投稿计划](OneSearch-AV-Paper-Dossier-v2-2026-10-15.md)。完整 motivation、related work、方法、数据、贡献、实验和建议投稿日程；17:40 核验 979 条合格训练题，500 题训练与评测结果、同题 answer-only 准备状态分别报告。

> **当前主线（2026-10-15）**：OneSearch-VL 风格的原生 Omni 音视频搜索问答；Qwen2.5-Omni-7B 主基座，真实工具轨迹 SFT。以下早期多源报告/observer 段落作为历史记录保留，当前方案请从投稿主计划、基座训练方案与最新执行记录进入。

# AV-DeepResearch 研究分支

本分支在 OneSearch-VL 官方发布代码上增加可复跑的数据与工具基础设施。第一篇研究多源音视频检索、证据核验和研究报告；Search for Generation 留作后续项目。官方介绍、署名、许可证与原发布入口保留。

当前已验证原始媒体登记、时间映射、冻结 metadata/网页正文检索、精确 quote、公私隔离和同步声画片段收据。已有 10 个自然素材候选与 3 个多源草稿，均为 `draft_unreviewed`。`inspect_av` 当前仅准备真实声画输入，返回 `media_ready_observer_unavailable`，不提供语义观察结果。人工 gold、原生 AV observer、G0/G1 和方法实验仍待完成。

## 环境与验证

主开发位置为用户指定的 Coder workspace，当前为 `coder ssh 4`，项目 `/home/jovyan/OneSearch-VL`。此前 workspace 2 的运行证据保留在原版本中。Pixi default 为 Linux CPU 环境。

```bash
cd /home/jovyan/OneSearch-VL
pixi install --locked
pixi run validate-retrieval
pixi run validate-research
pixi run av-prepare-pilot --help
pixi run av-replay --help
pixi run av-pilot-smoke --help
pixi run av-review-forms --help
```

测试会在临时目录生成合成声画素材；提交的两条 WAV 仅用于谱系测试。测试不需要已下载的自然媒体、8B 权重、搜索 API 或 Gemini。媒体处理依赖 Pixi 中的 ffmpeg/ffprobe。

`ONESEARCH_DISABLE_RETRIEVAL=1` 仅控制发布 inference 中的外部检索工具；unset/0 保留默认内容。研究适配器使用单独的固定 corpus，不把本地搜索冒充 Serper 服务。

## 入口与文档

- [完整研究计划 v2](AV-DeepResearch-Research-Plan-v2-2026-10-09.md)：假设、近邻、对照、预算及 G0–G4。
- [Phase 1 验收报告](AV-DeepResearch-Phase1-Progress-2026-10-10.md)：实际完成的素材、工具、测试及限制。
- [Stage 2 验收报告](AV-DeepResearch-Stage2-Progress-2026-10-10.md)：三案来源包、成本插桩、observer 契约和新回放。
- [AV observer 契约](AV-Observer-Contract-2026-10-10.md)：真实输入身份、模型输出和八类待人工验证 probes。
- [下一阶段工作](AV-DeepResearch-Next-Steps-2026-10-10.md)：先建立可裁决多源案例与真实语义 observer。
- [独立审核说明](annotation/Independent-Review-Instructions-zh.md)：A/B 与裁决流程；空模板不等于 gold。
- [OneSearch codebase audit](onesearch-av-codebase-audit.md)：原视觉路径的复用边界。

主要模块为 `research/av_data`、`research/av_search`、`research/av_agent`；有界复现脚本为 `repro`。提交包括代码、Pixi lock、验证代码、合成 fixtures 和研究文档。自然原件、模型权重、运行目录、cache、密钥与具体私有审核表在 Git ignore 目录中，需按 receipt/manifest 独立管理。

## 重放范围与成本

新一次采集、建库或回放必须使用新的输出目录，不能覆盖既有版本。公共任务和收据不暴露内部资产路径、gold、未来观察或已标定有效来源集合；所有任务共享所属 split 的全局来源池。

`max_av_seconds` 限制请求区间长度。`decoded_audio_samples` 等仍只计 prepared clip，`count_scope=prepared_clip_only`。另有 `actual_cost.media_work` 记录每次实际媒体进程、FFprobe emitted frames/samples、FFmpeg input decoder summaries、失败/partial unknown、wall time 与 AVIO 观测值；episode totals 包括初始化扫描，preflight 也单列。观测量是 lower bounds，文件大小不是实际 I/O，hash/copy/physical I/O 和 decoder discarded work 等未完整计量；`full_decode_work_accounting_ready=false`。不得将逻辑区间预算当实际总解码预算。

Episode 在构造时复制并重新核验完整 corpus identity，随后所有检索和文本读取执行于这一私有快照；每次调用仍检查原 corpus 是否持续被替换。收据和 header 保存同一 manifest identity，避免新文本带旧版本 hash。`read_text` 只返回冻结正文的 1–4000 字符精确片段。

`validation/verify_phase1_delivery.py` 是特定远端 run 的归档验收脚本，需要本地保存的自然素材/运行产物；它不属于零数据 checkout 的通用测试。旧 `natural-replay-v2` 的部分收据尚无 manifest hash，最新 adapter/CLI 已补齐；历史产物保留该归档缺口。


新增入口（同一 Pixi CPU 环境，无模型/API依赖）：

```bash
pixi run python -m research.av_data.prepare_cases --help
pixi run python -m research.av_data.web_sources --help
pixi run python -m research.av_agent.semantic_observer --help
```

`semantic_observer` 目前提供私有输入/输出契约、不可用状态和离线 preflight；没有部署具体 backend，也尚未接入 `inspect_av` 作为语义提供者。八类 probe 模板不等于八次模型推理。

## 2026-10-15 投稿主计划与证据总账

- [Paper Master Plan v5（中文）](OneSearch-AV-Paper-Master-Plan-v5-2026-10-15.md)：20K train、200 dev/1K test、OneSearch-VL 风格 Omni 轨迹 SFT、论文贡献和工作量。
- [Dataset Pipeline v2](OneSearch-AV-Dataset-Pipeline-v2-2026-10-15.md)
- [Benchmark Blueprint v2](OneSearch-AV-Benchmark-Blueprint-v2-2026-10-15.md)
- [研究过程与实验总账](OneSearch-AV-Research-Ledger-2026-10-15.md)
- [500 条训练最终 silver 结果](OneSearch-AV-500-Final-Silver-Results-2026-10-15.md)

以上规模为目标，当前量和状态见 records/2026-10-15-paper-plan-v5/evidence-snapshot.json；历史文档保留。

- [基座训练与强基线方案](OneSearch-AV-Base-Model-Training-v1-2026-10-15.md)：官方full recipe、当前LoRA pilot、2K full Thinker候选与公平对照。
- [最新执行记录](OneSearch-AV-Execution-Update-v1-2026-10-15.md)：恢复数据生产、benchmark候选、可播放案例与质量发现。

## 2026-10-15 单基座主线和原生质量审核

- [单基座主线与投稿执行方案](OneSearch-AV-One-Base-Model-Mainline-v1-2026-10-15.md)：固定Qwen2.5-Omni-7B，原agent loop/原trainer，full Thinker主性能路线与matched answer-only。
- [24例原生AV质量检查](OneSearch-AV-Native-Content-Quality-Audit-v1-2026-10-15.md)：21完成/3unknown，16主搜索保留建议/5感知控制；人工gold仍0。
- 2026-10-15T08:13:25.738228+00:00 合格训练题904；持续CPU生产、预留benchmark producer及新100录制训练质量检查的身份和进展见records/2026-10-15-onesearch-mainline/execution-snapshot.json。候选YAML通过原trainer schema检查，尚未运行新GPU训练。

## 2026-10-15 100录制内容抽查与模态审核

- [100录制抽查与模态审核包 v2](OneSearch-AV-100-Recording-Quality-and-Modality-Review-v2-2026-10-15.md)：91完成/9unknown，66keep/15control/4revise/6reject建议；训练14例与benchmark14例提供真实AV、去音轨视频、仅音频和上下文。
- 未自动筛改904训练池；人工gold仍0。媒体与逐包时间验证通过；浏览器播放单独记录。

- [用户审核后的音乐、赛事与现场活动优先扩产](OneSearch-AV-Priority-Expansion-v1-2026-10-15.md)


## 2026-10-15 论文构想与投稿计划 v1

- [论文构想、研究方案与投稿计划（中文）](OneSearch-AV-Paper-Dossier-v1-2026-10-15.md)：motivation、近邻 related work、三项预期贡献、三个方法模块、20K train/500 validation 与 200 dev/1K test、强基线和分析矩阵、现有结果及建议投稿日程。
- 数据以 17:15 现场核验为准：938 retained/938 CPU-qualified，新增 benchmark dev 候选 43，human gold 0。500 pilot 的 AV/静音及 unknown 完整报告；full Thinker 与正式 matched controls 按计划表述。
- 配套证据与 9 项来源、24 组主张的独立引用核验：[records/2026-10-15-paper-dossier-v1/](records/2026-10-15-paper-dossier-v1/)。

## 2026-10-15 新 benchmark 审核与 matched 500 对照

- [OneSearch-AV-Benchmark-Review-and-Matched-Control-v1-2026-10-15.md](OneSearch-AV-Benchmark-Review-and-Matched-Control-v1-2026-10-15.md)：新增24题可播放声画审核，23完成/1unknown的自动意见，120文件哈希/96 Range/四模式实际播放；500同题answer-only数据与原trainer配置已冻结，GPU训练未执行。
- 17:31现场池为979 retained/938 CPU-qualified/41 pending，43新增dev候选，human gold仍0。见records/2026-10-15-benchmark-review-v3/execution-snapshot.json；生产保持运行。


## 2026-10-15 新批次数据与审核闭环

- [数据构建与审核执行记录 v1](OneSearch-AV-Data-Construction-and-Review-v1-2026-10-15.md)：988 道合格训练题，新增24训练/5 benchmark四模式案例；审核记录导入模块与11项测试、机器可读benchmark设计、29次原生AV审核及浏览器播放收据。音乐/赛事/活动新采集继续运行；参考现场快照，完整目标保持20K train、500 validation、200 dev/1K test。


## 2026-10-15 500条同题answer-only训练与评测已提交

- [OneSearch-AV-Matched500-Answer-Control-v2-2026-10-15.md](OneSearch-AV-Matched500-Answer-Control-v2-2026-10-15.md)：job `slurm-profile-slurm-g4hbg` 已实际Running，1×H20Z/6h上限/无自动重试；同500问题、250更新，训练后接112个新rollout和匿名配对评分。该时点训练及成绩尚未完成。数据扩产与benchmark producer保持运行。

- [2K 训练数据扩产方案（2026-10-16）](OneSearch-AV-2K-Expansion-v1-2026-10-16.md)

- [2026-10-17 实际执行进度：1974 CPU / 1828 eligible / 304 benchmark candidates](OneSearch-AV-Progress-2026-10-17.md)

- [2026-10-18 实际进度：2092 CPU / 1946 eligible / 304 benchmark；恢复最后54条](OneSearch-AV-Progress-2026-10-18.md)

- [2026-10-18 2K已冻结；Full Thinker正式训练与自动评测启动](OneSearch-AV-2K-Freeze-and-Full-Training-2026-10-18.md)

- [2026-10-18 DeepSeek 在线检索接入、实测与迁移方案](OneSearch-AV-DeepSeek-Online-Search-2026-10-18.md)

- [Full Thinker 500题统一38题结果与2K/在线检索状态](OneSearch-AV-FullThinker500-Results-and-Online-Status-2026-10-18.md)

- [实际训练格式、原生Gemini监督与senior Omni升级审查](OneSearch-AV-Training-Format-and-Senior-Omni-Supervision-2026-10-19.md)

- [与OneSearch-VL的数据引擎差距、2K深度审查及500组合题修正方案](OneSearch-AV-Data-Engine-Alignment-and-Revision-2026-10-19.md)

- [2026-10-19 Coder 4、audio token 修复重评与 checkpoint 清理](OneSearch-AV-Research-Ledger-2026-10-19.md)：22 项回归和 8 条 GPU rollout 通过；同 56 题五模型 kjob 重评已启动；两份旧 smoke checkpoint 的 275.0 GiB 文件载荷已清理。

- [在线组合构题实现与媒体传输诊断（2026-10-19）](OneSearch-AV-Online-Compositional-Engine-and-Transport-Diagnostic-2026-10-19.md)

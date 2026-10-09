# AV-DeepResearch 研究分支

本分支在 OneSearch-VL 官方发布代码上增加可复跑的数据与工具基础设施。第一篇研究多源音视频检索、证据核验和研究报告；Search for Generation 留作后续项目。官方介绍、署名、许可证与原发布入口保留。

当前已验证原始媒体登记、时间映射、冻结 metadata/网页正文检索、精确 quote、公私隔离和同步声画片段收据。已有 10 个自然素材候选与 3 个多源草稿，均为 `draft_unreviewed`。`inspect_av` 当前仅准备真实声画输入，返回 `media_ready_observer_unavailable`，不提供语义观察结果。人工 gold、原生 AV observer、G0/G1 和方法实验仍待完成。

## 环境与验证

主开发位置为用户指定的 Coder workspace，当前为 `coder ssh 3`，项目 `/home/jovyan/OneSearch-VL`。此前 workspace 2 的运行证据保留在原版本中。Pixi default 为 Linux CPU 环境。

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
